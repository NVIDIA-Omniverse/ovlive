// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! End-to-end recovery integration test for `SubscribeReadLayer` (Subtask
//! `t3.fgj.ya.wg.cm`, Task `t2.fgj.ya.wg`, Epic `t1.fgj.ya`, Bee `b.fgj`).
//!
//! Epic AC #7b requires that the SR-2.4 recovery hook fires correctly on
//! `SubscribeReadLayer` bootstrap after a Valkey layer-state wipe. After
//! recovery, the subscriber observes the restored state via the Regime-3
//! coalesced-diff catch-up path (SDD §12.3) because the layer stream
//! (`le:{oid}:obj:stream`) is wiped alongside the rest of `le:{oid}:obj:*`
//! and is NOT re-XADDed by the recovery apply pipeline — see the
//! "deliberately does NOT XADD a `1-0` entry" comment in
//! `src/checkpoint/apply.rs` step 5.
//!
//! ## What this test pins
//!
//! Six observable invariants tied to AC #7b:
//!
//! 1. The handler-level `SubscribeReadLayer` recovery branch (engineered in
//!    Task `t2.fgj.ya.qm`) is reached on the natural `read_object_header`
//!    NotFound symptom that follows a `wipe_layer_state` of every
//!    `le:{oid}:obj:*` key.
//! 2. The recovery hook restores the latest checkpoint's state — the
//!    `recovery_metric{result="restored"}` cell increments by EXACTLY 1
//!    (the at-most-once retry contract, also pinned by sibling Task
//!    `t2.fgj.ya.qm`'s handler-level recovery test).
//! 3. The post-recovery subscribe handshake completes (gRPC stream emits
//!    a header, not an error).
//! 4. The bootstrap message body carries every section / field /
//!    time-sample from the pre-wipe state. We decode the body's OODF reset
//!    delta into a local `ObjectState` and compare it to a fresh
//!    `read_object_state_for_test` of the just-restored Valkey state.
//! 5. The catch-up path is recorded as `out_of_window` (PM-review patch):
//!    metric is `live_subscribe_regime_total{regime="out_of_window",
//!    service=<label>}`, NOT the original Subtask's mis-spelled
//!    `live_subscribe_regime_total{regime="regime3"}`.
//! 6. A subsequent live `UpdateLayer` write surfaces to the subscriber via
//!    the steady-state in-window per-commit path. This confirms the
//!    open_consumer position is intact across the recovery boundary and
//!    that the subscriber is NOT silently stuck on a stale stream cursor.
//!
//! ## How the test wires up
//!
//! - `setup_checkpoint_harness(1)` builds a real `LocalDiskCheckpointStore`
//!   plus a real (enabled) `RecoveryHook`, a live `LiveEditService`, and the
//!   periodic checkpoint loop with `interval=0` / `quiesce=0` so a
//!   just-written layer is immediately due on the next 1 s tick.
//! - We CreateSession + CreateLayer + 3 UpdateLayers (each adding one
//!   unique `DiffChangeSection`) so the layer reaches `seq=4` with a
//!   non-trivial materialised state.
//! - `wait_for_checkpoint` blocks until the leader-elected loop lands at
//!   least one checkpoint generation for our `object_id`.
//! - `wipe_layer_state` `UNLINK`s every `le:{oid}:obj:*` key — same
//!   data-loss surface that a Valkey failover-to-fresh-replica or
//!   persistence-wipe presents to the service. The `{oid}` hash tag keeps
//!   the multi-key UNLINK slot-safe on Cluster.
//! - `handle_subscribe_read_layer` is called with
//!   `have_object_id=object_id, have_version=0`. The non-brand-new
//!   subscriber path runs through `read_object_header(object_id)`, sees
//!   NotFound, and enters the recovery branch — the wired site for AC #7b.
//! - The retry header read after recovery succeeds (apply re-wrote
//!   `obj:hdr`), `have_version=0 < current_seq`, XRANGE on the freshly
//!   wiped (now-empty) stream from `1-0` returns nothing → the catch-up
//!   falls through to `catch_up_out_of_window` → ONE coalesced diff body
//!   containing every restored section (FullContent plan triggered by
//!   `have_version == 0`).
//!
//! ## Sandbox env
//!
//! `TestValkey::setup` (inside `setup_checkpoint_harness`) reads
//! `TEST_VALKEY_URL`. Under the project sandbox the URL must include the
//! password and `host.docker.internal` host; see CLAUDE.md.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::time::Duration;

use bytes::Bytes;
use fred::interfaces::StreamsInterface;
use futures::{Stream, StreamExt};
use tonic::{Request, Status};

use live_edit::domain::delta::apply_delta;
use live_edit::domain::ompf::OmpfContainer;
use live_edit::domain::types::{FieldValue, ObjectState};
use live_edit::grpc::layer::{handle_create_layer, handle_update_layer};
use live_edit::grpc::proto::{
    subscribe_read_layer_response, SubscribeReadLayerRequest, SubscribeReadLayerResponse,
};
use live_edit::grpc::subscribe::handle_subscribe_read_layer;
use live_edit::grpc::BearerToken;
use live_edit::storage::diff_read::snapshot_to_object_state;
use live_edit::storage::session::SessionStore;
use live_edit::storage::test_support::read_object_state_for_test;
use oodf_ffi::{CommandDiffChangeSection, OwnedDelta, SetField, StructCommand};

use super::helpers::*;
use crate::common::owned_delta::{delta_to_ompf_bytes, empty_delta};
use crate::grpc::helpers::{make_create_data, make_create_header, make_request, streaming_create};

const LAYER_NAME: &str = "subscribe-recovery-layer";
const SESSION_NAME: &str = "subscribe-recovery-session";
const SUBSCRIBER_CLIENT_ID: &str = "subscribe-recovery-subscriber";
const WRITER_CLIENT_ID: &str = "subscribe-recovery-writer";
const PRE_CHECKPOINT_UPDATE_COUNT: u64 = 3;

// ── OwnedDelta / streaming helpers ──────────────────────────────────────────
//
// Mirrors the sibling subscribe-regimes / subscribe-regime-metrics tests
// shape — each UpdateLayer adds one new `DiffChangeSection` via a non-reset
// delta so the per-commit diff is non-empty and the materialised state
// genuinely diverges from the bootstrap `empty_delta`. The varying
// `section_id` lets the bootstrap-body assertion below identify each
// section unambiguously.

fn incremental_diff_change_section(section_id: u64) -> OwnedDelta {
    OwnedDelta {
        is_diff: false,
        base_version: 0,
        struct_commands: vec![StructCommand::DiffChangeSection(CommandDiffChangeSection {
            parent_id: 0,
            section_id,
            section_name: format!("sec-{section_id}"),
            section_type: 0,
            section_order: 0,
        })],
        ..Default::default()
    }
}

/// Issue a single UpdateLayer carrying a non-reset `DiffChangeSection` with
/// the given `section_id`. The varying section_id guarantees each commit
/// produces a non-empty, distinguishable per-commit diff so the bootstrap
/// assertion can verify each pre-checkpoint section is preserved through
/// recovery.
async fn apply_section_update(
    harness: &CheckpointHarness,
    session_id: &str,
    object_id: u64,
    section_id: u64,
) {
    let delta = incremental_diff_change_section(section_id);
    let ompf_bytes = delta_to_ompf_bytes(&delta, &[]);
    let stream = streaming_update(vec![
        make_update_header(
            session_id,
            LAYER_NAME,
            object_id,
            section_id * 10,
            WRITER_CLIENT_ID,
        ),
        make_update_data(ompf_bytes),
    ]);
    let request = make_request(stream);
    handle_update_layer(&harness.svc, request)
        .await
        .unwrap_or_else(|e| panic!("UpdateLayer (section_id={section_id}) failed: {e}"));
}

// ── Subscribe-stream helpers ────────────────────────────────────────────────

fn subscribe_request(
    session_id: &str,
    object_id: u64,
    sequence_num: u64,
) -> Request<SubscribeReadLayerRequest> {
    let mut request = Request::new(SubscribeReadLayerRequest {
        session_id: session_id.to_owned(),
        layer_name: LAYER_NAME.to_owned(),
        object_id,
        sequence_num,
        client_id: SUBSCRIBER_CLIENT_ID.to_owned(),
    });
    request
        .extensions_mut()
        .insert(BearerToken::new("test-token"));
    request
}

/// Collect a `Header` plus any `Chunk`s until `total_size` bytes are
/// gathered, then return both. Mirrors `recv_message` in
/// `subscribe_regimes_test.rs`.
async fn recv_message<S>(
    stream: &mut S,
) -> (live_edit::grpc::proto::SubscribeReadLayerHeader, Vec<u8>)
where
    S: Stream<Item = Result<SubscribeReadLayerResponse, Status>> + Unpin,
{
    let first = tokio::time::timeout(Duration::from_secs(10), stream.next())
        .await
        .expect("stream timed out waiting for next header")
        .expect("stream closed unexpectedly")
        .expect("stream yielded an error");

    let header = match first.response_type {
        Some(subscribe_read_layer_response::ResponseType::Header(h)) => h,
        other => panic!("expected Header, got: {other:?}"),
    };

    let mut buf = Vec::new();
    while (buf.len() as u64) < header.total_size {
        let next = tokio::time::timeout(Duration::from_secs(10), stream.next())
            .await
            .expect("stream timed out waiting for chunk")
            .expect("stream closed before total_size reached")
            .expect("stream yielded an error");
        match next.response_type {
            Some(subscribe_read_layer_response::ResponseType::Chunk(c)) => {
                buf.extend_from_slice(&c.data);
            }
            other => panic!("expected Chunk while filling body, got: {other:?}"),
        }
    }

    (header, buf)
}

/// Apply an OMPF-bytes diff to a local `ObjectState` via the exact same
/// decode pipeline the server uses (`OmpfContainer::parse` →
/// `oodf_ffi::parse_delta` → `apply_delta`). Mirrors the helper in
/// `subscribe_regimes_test.rs`.
fn apply_ompf_to_state(state: &mut ObjectState, ompf_bytes: &[u8]) {
    let container = OmpfContainer::parse(ompf_bytes).expect("parse OMPF");
    let owned = if container.delta_bytes.is_empty() {
        None
    } else {
        Some(oodf_ffi::parse_delta(container.delta_bytes).expect("parse OODF delta"))
    };
    apply_delta(state, owned).expect("apply_delta");
}

// ── Metric-reader helpers ───────────────────────────────────────────────────

fn recovery_attempts(harness: &CheckpointHarness, result: &str) -> u64 {
    live_edit::observability::metrics::test_util::sample_value(
        &live_edit::observability::metrics::test_util::render(&harness.metrics),
        "live_edit_checkpoint_recovery_attempts_total",
        &[
            ("result", result),
            ("scope", "layer"),
            ("service", SERVICE_LABEL),
        ],
    )
    .unwrap_or(0.0) as u64
}

fn subscribe_regime_count(harness: &CheckpointHarness, regime: &str) -> u64 {
    live_edit::observability::metrics::test_util::sample_value(
        &live_edit::observability::metrics::test_util::render(&harness.metrics),
        "live_subscribe_regime_total",
        &[("regime", regime), ("service", SERVICE_LABEL)],
    )
    .unwrap_or(0.0) as u64
}

// ═══════════════════════════════════════════════════════════════════════════
// Epic AC #7b — SubscribeReadLayer recovers layer state after Valkey wipe.
// ═══════════════════════════════════════════════════════════════════════════

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn subscribe_read_layer_recovers_layer_state_after_valkey_wipe() {
    // Setup: real LocalDisk checkpoint store + enabled recovery hook +
    // periodic checkpoint loop with interval=0 / quiesce=0 so writes are
    // immediately due on the next 1 s tick.
    let harness = setup_checkpoint_harness(1).await;

    // Create the session via SessionStore directly (mirrors the
    // create_session_and_layer helper but the harness wiring exposes
    // session_store as Arc<SessionStore>).
    let session_store =
        SessionStore::new(harness.tv.client().clone(), harness.tv.key_schema().clone());
    let session_id = session_store
        .create(SESSION_NAME, "/stage", "test-user")
        .await
        .expect("session creation must succeed");

    // CreateLayer at seq=1 with an empty delta.
    let create_ompf = delta_to_ompf_bytes(&empty_delta(), &[]);
    let create_stream = streaming_create(vec![
        make_create_header(&session_id, LAYER_NAME),
        make_create_data(create_ompf),
    ]);
    let create_resp = handle_create_layer(&harness.svc, make_request(create_stream))
        .await
        .expect("CreateLayer must succeed");
    let object_id = create_resp.into_inner().object_id;

    // Apply PRE_CHECKPOINT_UPDATE_COUNT UpdateLayers, each adding a unique
    // section. After this the layer is at seq=1 + N (== 4 by default) and
    // the materialised ObjectState contains N distinct sections.
    let expected_section_ids: Vec<u64> =
        (0..PRE_CHECKPOINT_UPDATE_COUNT).map(|i| 1000 + i).collect();
    for sid in &expected_section_ids {
        apply_section_update(&harness, &session_id, object_id, *sid).await;
    }

    // Snapshot the pre-wipe materialised state from Valkey so we have a
    // ground-truth reference to compare the post-recovery subscriber view
    // against. We do this BEFORE the wipe so we capture exactly what the
    // checkpoint should preserve.
    let pre_wipe_snapshot = read_object_state_for_test(
        &harness.svc.valkey,
        &harness.svc.key_schema,
        &object_id.to_string(),
    )
    .await
    .expect("read_object_state_for_test must succeed before wipe");
    let pre_wipe_state = snapshot_to_object_state(&pre_wipe_snapshot);
    let pre_wipe_section_ids: HashSet<u64> = pre_wipe_state.sections.keys().copied().collect();
    assert_eq!(
        pre_wipe_section_ids,
        expected_section_ids.iter().copied().collect::<HashSet<_>>(),
        "pre-wipe state must contain exactly the sections we wrote"
    );

    // Wait for the periodic checkpoint loop to land a generation that captures
    // ALL the section updates — i.e. `snap_seq >= pre_wipe_state.sequence_num`,
    // not merely the first generation. With interval=0/quiesce=0 and tick=1s the
    // loop can fire between CreateLayer and the section updates, landing an early
    // generation at seq=1 with zero sections; recovering THAT after the wipe
    // yields an empty layer. Gating on the post-update seq makes the wait
    // deterministic under heavy parallel load. The 10 s deadline carries ample
    // headroom (the loop ticks every 1 s).
    let _handle =
        wait_for_checkpoint_at_least(&harness.store, object_id, pre_wipe_state.sequence_num, 10)
            .await;

    // Wipe every `le:{oid}:obj:*` key — the production data-loss surface
    // a failover-to-fresh-replica or persistence-wipe would present. This
    // ALSO wipes `le:{oid}:obj:stream`, so post-recovery XRANGE returns
    // empty and the catch-up path falls through to `out_of_window`
    // (Regime 3) — the SDD §12.3 coalesced-diff path AC #7b requires.
    wipe_layer_state(&harness.tv, object_id).await;

    // Sanity: confirm the wipe actually removed the obj:hdr so the
    // subscriber path naturally hits NotFound on `read_object_header`.
    // `read_object_state_for_test` surfaces ServiceError::NotFound when the
    // header is missing — exactly the symptom the subscriber path
    // observes and recovers from. If a wipe regression left the header
    // intact this would return Ok and the assertions below would be
    // vacuous.
    let post_wipe_probe = read_object_state_for_test(
        &harness.svc.valkey,
        &harness.svc.key_schema,
        &object_id.to_string(),
    )
    .await;
    match &post_wipe_probe {
        Err(live_edit::error::ServiceError::NotFound(_)) => {}
        Ok(_) => panic!(
            "wipe_layer_state must leave the object header missing — \
             read_object_state_for_test returned Ok unexpectedly"
        ),
        Err(other) => panic!(
            "wipe_layer_state must surface NotFound from read_object_state_for_test, \
             got error: {other}"
        ),
    }

    // Open SubscribeReadLayer carrying have_object_id=object_id and
    // have_version=0. This drives:
    //   a) the non-brand-new path — `is_brand_new` requires BOTH knobs at
    //      zero, and we set have_object_id != 0, so the same-object branch
    //      runs (the wired site for the SR-2.4 recovery hook).
    //   b) `read_object_header` returns NotFound (wipe took obj:hdr),
    //   c) the engineer's SR-2.4 recovery branch fires → Restored,
    //   d) the retry `read_object_header` succeeds (apply re-wrote obj:hdr),
    //   e) `have_version=0 < current_seq` → XRANGE from `1-0` on the empty
    //      stream returns nothing → fall through to `catch_up_out_of_window`.
    // `have_version=0` is the cold-cache shape a client sends post-failover,
    // and it is the one this test owns: `DiffEngine::plan` short-circuits
    // `have_version == 0` to `FullContent`, so the whole restored state is
    // emitted in DFS order and the assertions below can compare it whole.
    //
    // The version-FILTERED shape — a reconnector at a version it really did
    // observe, which is the majority of post-failover subscribers — is a
    // different claim and lives in the two siblings below
    // (`a_reconnector_at_the_checkpoint_sequence_receives_the_full_restored_state`
    // and `a_reconnector_ahead_of_the_checkpoint_receives_the_full_restored_state`).
    // Those are where b.2ti bit: pre-fix the restore stamped every entry at
    // `version = 1` while the header sat at `seeded_seq`, so an `Incremental` /
    // coalesced plan filtered every restored entry out and the reconnector was
    // acked caught-up having received nothing. Do NOT read this test's
    // `have_version=0` as evidence about that path — it deliberately does not
    // exercise the filter.
    let sub_req = subscribe_request(&session_id, object_id, 0);
    let response = handle_subscribe_read_layer(&harness.svc, sub_req)
        .await
        .expect("SubscribeReadLayer must return Ok after recovery");
    let mut stream = response.into_inner();

    // Drain the bootstrap message: ONE coalesced diff covering 1..=current_seq.
    let (bootstrap_header, bootstrap_body) = recv_message(&mut stream).await;

    // AC#7b core: gRPC stream emitted a header (not an error). object_id
    // is preserved across recovery (AC #9 is its own subtask but the
    // invariant is implicit here too).
    assert_eq!(
        bootstrap_header.object_id, object_id,
        "post-recovery bootstrap header must carry the same object_id"
    );
    // sequence_num must equal the *seeded* seq from apply_checkpoint:
    // `max(snap_seq, max_seq_observed.unwrap_or(0)) + safety_margin`
    // (SR-2.5 step 4). Two terms resolve here:
    //
    //   - `max_seq_observed` is `None`. The recovery hook probes the
    //     per-object stream's `last-generated-id` for a high-water hint,
    //     but `wipe_layer_state` took `le:{oid}:obj:stream` along with the
    //     rest of `obj:*`, so the probe finds no key and folds the floor
    //     back to `snap_seq`.
    //   - `snap_seq == pre_wipe_state.sequence_num`. The
    //     `wait_for_checkpoint_at_least` gate above blocked until a
    //     generation captured every section update, and nothing writes
    //     after it, so the newest generation sits exactly there.
    //
    // The margin is ADDITIVE above that floor at every input (b.8y5), so
    // the expectation is a sum, not a max. Computing it from the live
    // config means a future change to the default surfaces as a clean
    // assertion failure rather than a hand-tuned 1004.
    let cfg = crate::common::test_config();
    let expected_seq = pre_wipe_state.sequence_num + cfg.checkpoint_seq_safety_margin;
    assert_eq!(
        bootstrap_header.sequence_num,
        expected_seq,
        "post-recovery bootstrap header sequence_num must equal snap_seq + \
         safety_margin = {} + {} = {expected_seq}, got {}",
        pre_wipe_state.sequence_num,
        cfg.checkpoint_seq_safety_margin,
        bootstrap_header.sequence_num
    );
    assert!(
        !bootstrap_body.is_empty(),
        "out_of_window coalesced diff body must be non-empty after recovery — \
         the restored state has 3 sections"
    );

    // Decode the bootstrap body into a fresh ObjectState and verify it
    // contains every pre-wipe section. apply_delta on a reset diff with
    // base_version=0 wipes the local state and rebuilds it from the diff
    // — exactly the coalesced "Regime 3" reset contract.
    let mut subscriber_state = ObjectState::default();
    apply_ompf_to_state(&mut subscriber_state, &bootstrap_body);

    let subscriber_section_ids: HashSet<u64> = subscriber_state.sections.keys().copied().collect();
    assert_eq!(
        subscriber_section_ids, pre_wipe_section_ids,
        "subscriber's reconstructed section set must match the pre-wipe set \
         (recovery preserved every section); got {subscriber_section_ids:?}, \
         expected {pre_wipe_section_ids:?}"
    );

    // Cross-check with the post-recovery materialised state — the
    // subscriber's view of every section's structural identity
    // (parent_id, name, section_type, section_order) must match what
    // read_object_state_for_test reports for the restored Valkey state. Version
    // annotations on the subscriber side are legitimately
    // apply_delta-bumped (coalesced-diff reset semantics; same caveat as
    // sibling `subscribe_regimes_test::out_of_window_reconnect_*`), so we
    // compare the structural identity fields only.
    let restored_snapshot = read_object_state_for_test(
        &harness.svc.valkey,
        &harness.svc.key_schema,
        &object_id.to_string(),
    )
    .await
    .expect("post-recovery read_object_state_for_test must succeed");
    let restored_state = snapshot_to_object_state(&restored_snapshot);
    for sid in &expected_section_ids {
        let restored = restored_state
            .sections
            .get(sid)
            .unwrap_or_else(|| panic!("restored state missing section {sid}"));
        let subscriber = subscriber_state
            .sections
            .get(sid)
            .unwrap_or_else(|| panic!("subscriber state missing section {sid}"));
        assert_eq!(
            (
                subscriber.parent_id,
                subscriber.name.as_str(),
                subscriber.section_type,
                subscriber.section_order,
            ),
            (
                restored.parent_id,
                restored.name.as_str(),
                restored.section_type,
                restored.section_order,
            ),
            "section {sid} structural identity must match the restored snapshot"
        );
    }

    // b.dy5: the out_of_window catch-up now terminates with the exactly-once
    // caught-up ack (`total_size==0`) emitted centrally by `run_catch_up`,
    // carrying `tracked_version` (= current_seq = expected_seq for Regime 3).
    // It lands on the stream immediately after the coalesced diff and BEFORE
    // any live update, so it must be drained here or the live-update read
    // below would mis-read this trailer as the per-commit diff. Before b.dy5
    // the out_of_window path emitted no terminator and this drain was absent.
    let (caught_up_header, caught_up_body) = recv_message(&mut stream).await;
    assert_eq!(
        caught_up_header.total_size, 0,
        "out_of_window catch-up trailer must be a zero-body caught-up ack \
         (total_size==0), got total_size={}",
        caught_up_header.total_size
    );
    assert!(
        caught_up_body.is_empty(),
        "caught-up ack header must carry no body bytes"
    );
    assert_eq!(
        caught_up_header.object_id, object_id,
        "caught-up ack object_id must match the recovered object"
    );
    assert_eq!(
        caught_up_header.sequence_num, expected_seq,
        "caught-up ack must carry the recovered current_seq (tracked_version) = {expected_seq}"
    );

    // AC#7b recovery-metric pin: exactly one Restored attempt landed
    // during this subscribe RPC. The at-most-once contract from sibling
    // Task `t2.fgj.ya.qm` already pins the retry-once shape; here we
    // verify the integration: a single subscribe RPC triggers exactly
    // one Restored recovery and no others.
    assert_eq!(
        recovery_attempts(&harness, "restored"),
        1,
        "exactly ONE Restored recovery attempt must have fired for this RPC"
    );
    assert_eq!(
        recovery_attempts(&harness, "not_recoverable"),
        0,
        "no NotRecoverable attempts expected — checkpoint was seeded"
    );
    assert_eq!(
        recovery_attempts(&harness, "error"),
        0,
        "no Error attempts expected — LocalDisk store is healthy"
    );

    // PM-review patch: the catch-up regime selected after recovery is
    // `out_of_window` (NOT the original Subtask body's mis-spelled
    // `regime3`). Metric is `live_subscribe_regime_total` — no `_edit_`
    // infix.
    let out_of_window_observations = subscribe_regime_count(&harness, "out_of_window");
    assert_eq!(
        out_of_window_observations, 1,
        "exactly ONE out_of_window catch-up observation must have fired \
         (recovery + wiped stream forces Regime 3); got {out_of_window_observations}"
    );
    // Sanity: no other regime cell moved on this RPC.
    assert_eq!(
        subscribe_regime_count(&harness, "brand_new"),
        0,
        "no brand_new observations: subscriber sent non-zero have_object_id"
    );
    assert_eq!(
        subscribe_regime_count(&harness, "in_window"),
        0,
        "no in_window observations: stream was wiped, XRANGE empty"
    );
    assert_eq!(
        subscribe_regime_count(&harness, "up_to_date"),
        0,
        "no up_to_date observations: have_version=1 < current_seq"
    );

    // AC#7b final pin: a subsequent live UpdateLayer must surface to the
    // subscriber via the steady-state in-window per-commit path. This
    // confirms (a) the open_consumer position survived the recovery
    // boundary, and (b) the writer half of the service can still XADD
    // against the just-restored object_id. The recovery hook does NOT
    // re-XADD an entry on the stream (see apply.rs step 5 comment), so
    // the next live write is the FIRST entry on the post-recovery
    // stream — which is exactly what an in-window per-commit subscriber
    // expects.
    let post_recovery_section_id = 2000_u64;
    apply_section_update(&harness, &session_id, object_id, post_recovery_section_id).await;

    let (live_header, live_body) = recv_message(&mut stream).await;
    assert_eq!(
        live_header.object_id, object_id,
        "post-recovery live update must carry the same object_id"
    );
    assert_eq!(
        live_header.sequence_num,
        expected_seq + 1,
        "post-recovery live update must bump seq exactly one past the \
         restored state's seq"
    );
    assert!(
        !live_body.is_empty(),
        "post-recovery live update body must be non-empty (added a section)"
    );

    // Apply the live update to the subscriber's local mirror and confirm
    // the new section is now visible — proving the post-recovery
    // steady-state path is functioning end-to-end.
    apply_ompf_to_state(&mut subscriber_state, &live_body);
    assert!(
        subscriber_state
            .sections
            .contains_key(&post_recovery_section_id),
        "subscriber's local mirror must contain the post-recovery \
         section (id={post_recovery_section_id}) after applying the \
         live per-commit diff"
    );

    // Tear the periodic loop down cleanly to avoid noisy panics if the
    // tokio runtime shutdown races the loop's outstanding awaits.
    harness.cancel.cancel();
    let _ = tokio::time::timeout(Duration::from_secs(5), harness.loop_handle).await;
}

// ═══════════════════════════════════════════════════════════════════════════
// b.2ti — a RECONNECTOR (have_version > 0) must receive the restored state.
//
// The test above subscribes at `have_version = 0`, which short-circuits every
// catch-up builder to a version-BLIND full-content plan. Every other
// post-restore subscriber takes a version-FILTERED path, and that is where the
// defect lived:
//
//   * a restore preserves `object_id` (`apply.rs`'s SR-2.4 identity guard), so
//     a subscriber that survives it does NOT re-bootstrap — `is_brand_new`
//     needs `have_object_id == 0 && have_version == 0`
//     (`src/grpc/subscribe.rs`);
//   * both reconnect regimes therefore route through a filter on
//     `version > have_version` — `plan_faithful_coalesced` under the shipped
//     `LIVE_EDIT_CATCHUP_SNAPSHOT_ENABLED=true`, `plan_incremental` (via
//     `compute_diff_at_cut`) under the legacy `false` (`src/domain/diff.rs`);
//   * pre-b.2ti the restore left every entry at the version
//     `decode_checkpoint_ompf` produced (`1`) while stamping the HEADER at
//     `seeded_seq`, so no restored entry cleared the filter. The subscriber
//     received a payload carrying no content and was then acked caught-up —
//     "a restored layer served an empty catch-up", indistinguishable at the
//     client from a healthy no-op.
//
// The fix stamps the three versions those builders FILTER on —
// `Section::last_version`, `Field::version`, `TimeSample::version` — at
// `seeded_seq`, which dominates every sequence number the object ever issued
// (the guard band), so a reconnector at ANY version it could legitimately hold
// is owed the whole restored state.
//
// `Section::first_version` is deliberately NOT re-stamped (`apply.rs` step 4a):
// `plan_faithful_coalesced`'s abandoned-prim retraction arm fires on
// `first_version <= have_version && !live`, so raising it would strand the
// orphan sections a restored base carries as permanently unretractable. That
// exemption is invisible to the assertions below, and the reason is worth
// stating because it is the shape a reader would expect to be load-bearing: a
// restored section arrives via the MODIFIED bucket (`first_version <=
// have_version && last_version > have_version`) rather than the NEW one
// (`first_version > have_version`), and `src/storage/diff_read.rs::build_delta`
// emits the same `DiffChangeSection` upsert for both. Content delivery — which
// is all these tests read — is identical either way; what selects a bucket is
// the flattened `last_version`, which is the column the fix moves.
//
// ## The three positions driven below
//
// [`ReconnectPosition`] enumerates them, and each arm pins the catch-up REGIME
// it lands in via `live_subscribe_regime_total` rather than leaving that to
// prose — the regime is what selects the builder, so an arm that silently
// changed regime would stop covering the builder it claims:
//
//   1. `AtCheckpointSequence` (`out_of_window`) — the plain shape.
//   2. `AheadOfCheckpoint` (`out_of_window`) — the DISCRIMINATING position: it
//      is what separates a `seeded_seq` stamp from a `snap_seq` one.
//   3. `InWindowAfterPartialLoss` (`in_window`) — the third REGIME. Regime
//      selection is a property of the surviving stream, not of the
//      subscriber's position alone: `run_catch_up_inner`
//      (`src/grpc/subscribe.rs`) picks `in_window` only when the stream holds
//      an entry at EXACTLY `{have_version + 1}-0`, and both arms above sit at
//      or above the stream's high-water mark (arm 1 wipes the stream outright;
//      arm 2's `have_version` IS the mark). `catch_up_in_window_snapshot`
//      reaches `compute_faithful_coalesced_at_cut` through a different call site with
//      a different cursor pin, so the fix has to hold there on its own.
//
// ## Which positions run on which catch-up path
//
// Positions 1 and 2 each run TWICE — once on the shipped path
// (`LIVE_EDIT_CATCHUP_SNAPSHOT_ENABLED=true`, `plan_faithful_coalesced`) and
// once on the legacy rollback path (`false`, `plan_incremental` via
// `compute_diff_at_cut`); see the two `..._on_the_legacy_path` siblings. Both builders
// filter on the same persisted `version` columns, so both were blind to a
// restored layer pre-fix, and the legacy knob is a supported rollback.
//
// Position 3 runs on the SHIPPED path only, and that is a deliberate boundary
// rather than a gap. `in_window` under the legacy path is not the same code at
// all: `run_catch_up_inner` routes it to `catch_up_in_window`, which
// RECONSTRUCTS each retained stream entry commit-by-commit
// (`replay_commit_diff`, each commit's diff version-bounded to that commit)
// instead of building one coalesced diff off persisted state. After a DR
// restore the retained
// entries are pre-loss commits against state that no longer exists, and nothing
// in b.2ti's fix — which changes what `apply_restore` PERSISTS — reaches that
// path. Adding a legacy `in_window` arm that expected the restored content
// would therefore assert a claim this fix does not make and never intended to;
// the honest coverage statement is that the legacy in-window replay after a
// restore is its own unfixed subject.
// ═══════════════════════════════════════════════════════════════════════════

/// Updates issued AFTER the checkpoint lands in the two arms that spare the
/// stream (`AheadOfCheckpoint`, `InWindowAfterPartialLoss`).
const POST_CHECKPOINT_UPDATE_COUNT: u64 = 2;

/// Where the reconnecting subscriber's tracked version sits relative to the
/// checkpoint the restore will read, and — for the third variant — what the
/// surviving stream does to the catch-up REGIME that position lands in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ReconnectPosition {
    /// `have_version == snap_seq`, stream wiped. The plain shape: the
    /// subscriber saw everything the checkpoint captured and nothing more.
    /// Regime `out_of_window` — the wiped stream has no entry at
    /// `{have_version + 1}-0`.
    AtCheckpointSequence,
    /// `have_version > snap_seq` — commits landed after the checkpoint and the
    /// subscriber observed them before the loss. The **discriminating** case:
    /// the restore floor rises to the surviving stream's high-water mark, so a
    /// hypothetical `snap_seq` stamp would sit BELOW `have_version` and be
    /// filtered out, while `seeded_seq` (= floor + band) sits above it.
    /// Regime `out_of_window`: `have_version` IS the stream's high-water mark,
    /// so the XRANGE from `{have_version + 1}-0` comes back empty.
    AheadOfCheckpoint,
    /// The same partial loss as `AheadOfCheckpoint` — stream spared, floor at
    /// its high-water mark — but the subscriber's tracked version sits at
    /// `snap_seq`, i.e. BELOW the surviving top. `run_catch_up_inner`'s XRANGE
    /// from `{snap_seq + 1}-0` therefore hits the first post-checkpoint entry
    /// exactly, and the catch-up routes through
    /// `catch_up_in_window_snapshot` → `compute_faithful_coalesced_at_cut`.
    ///
    /// Realistic as a shape: a Valkey data-key loss that spares the stream
    /// (the stream is a distinct key family) while a subscriber is only a
    /// couple of commits behind. Distinct from the two arms above in the CALL
    /// SITE it exercises, not in the subscriber's arithmetic: the in-window
    /// arm opens its steady-state consumer at `{current_seq}-0` and computes
    /// the coalesced diff from its own site, so the `version > have_version`
    /// filter has to clear there independently.
    InWindowAfterPartialLoss,
}

/// Issue [`POST_CHECKPOINT_UPDATE_COUNT`] commits past the checkpoint and
/// return the layer's resulting sequence number.
///
/// Shared by the two arms that spare the stream. The returned sequence is both
/// the surviving stream's high-water mark and (therefore) the restore floor
/// those arms expect.
async fn commit_past_the_checkpoint(
    harness: &CheckpointHarness,
    session_id: &str,
    object_id: u64,
    snap_seq: u64,
) -> u64 {
    for i in 0..POST_CHECKPOINT_UPDATE_COUNT {
        apply_section_and_field_update(harness, session_id, object_id, 2000 + i).await;
    }
    let post_checkpoint_seq = snapshot_to_object_state(
        &read_object_state_for_test(
            &harness.svc.valkey,
            &harness.svc.key_schema,
            &object_id.to_string(),
        )
        .await
        .expect("read_object_state_for_test must succeed after the post-checkpoint commits"),
    )
    .sequence_num;
    assert!(
        post_checkpoint_seq > snap_seq,
        "fixture precondition: the post-checkpoint commits must advance the layer past the \
         checkpoint ({post_checkpoint_seq} vs snap_seq {snap_seq}); without that the surviving \
         stream's high-water mark collapses onto the checkpoint and the arm stops testing the \
         shape it claims"
    );
    post_checkpoint_seq
}

/// Assert the surviving stream carries an entry at EXACTLY `{start_seq}-0`.
///
/// The mechanical precondition for `run_catch_up_inner`'s in-window branch
/// (`src/grpc/subscribe.rs`: `entries.first().id == start_id`). Asserted rather
/// than assumed so that a change to how commits are XADDed shows up here as a
/// fixture failure instead of silently demoting the in-window arm to
/// `out_of_window` — where it would duplicate its two siblings and pin nothing
/// new. The regime metric assertion downstream is the other half of that guard.
async fn assert_stream_holds_entry_at(harness: &CheckpointHarness, object_id: u64, start_seq: u64) {
    let stream_key = harness.tv.key_schema().layer_stream(object_id);
    let start_id = format!("{start_seq}-0");
    let probe: Vec<(String, HashMap<String, Bytes>)> = harness
        .tv
        .client()
        .pool
        .xrange_values(&stream_key, start_id.clone(), "+", Some(1))
        .await
        .expect("XRANGE probe must succeed");
    // Values decode as `Bytes`, never `String`: gRPC-path entries carry the
    // binary `ompf` field (b.gze), which is not valid UTF-8.
    assert_eq!(
        probe.first().map(|(id, _)| id.as_str()),
        Some(start_id.as_str()),
        "fixture precondition: the surviving stream must hold an entry at exactly {start_id}, \
         which is what makes `run_catch_up_inner` select the in-window regime. Got {probe:?}"
    );
}

/// A non-reset update carrying one new section AND one field on it, so the
/// catch-up filter is exercised on the SECTION columns and on `Field::version`
/// rather than on fields alone.
///
/// Which section column does the work depends on when the commit lands. Authored
/// pre-checkpoint (the fixture's usual use), the section goes through the
/// restore, which preserves `first_version` and flattens `last_version` to
/// `seeded_seq` — so the section clears the filter as MODIFIED, on
/// `last_version`. Authored post-restore, it clears it as NEW, on
/// `first_version`. Both buckets emit the same `DiffChangeSection`, so the
/// assertions read content either way.
fn incremental_section_and_field(section_id: u64) -> OwnedDelta {
    OwnedDelta {
        is_diff: false,
        base_version: 0,
        struct_commands: vec![StructCommand::DiffChangeSection(CommandDiffChangeSection {
            parent_id: 0,
            section_id,
            section_name: format!("sec-{section_id}"),
            section_type: 0,
            section_order: 0,
        })],
        set_fields: vec![SetField {
            section_id,
            key_name: field_key(section_id),
            value_or_ext_hash: field_value(section_id),
            ext_value_size: 0,
            ext_value_index: 0,
            set_order: 0,
        }],
        ..Default::default()
    }
}

fn field_key(section_id: u64) -> String {
    format!("k-{section_id}")
}

fn field_value(section_id: u64) -> Vec<u8> {
    format!("v-{section_id}").into_bytes()
}

/// [`apply_section_update`]'s sibling for [`incremental_section_and_field`].
async fn apply_section_and_field_update(
    harness: &CheckpointHarness,
    session_id: &str,
    object_id: u64,
    section_id: u64,
) {
    let delta = incremental_section_and_field(section_id);
    let ompf_bytes = delta_to_ompf_bytes(&delta, &[]);
    let stream = streaming_update(vec![
        make_update_header(
            session_id,
            LAYER_NAME,
            object_id,
            section_id * 10,
            WRITER_CLIENT_ID,
        ),
        make_update_data(ompf_bytes),
    ]);
    handle_update_layer(&harness.svc, make_request(stream))
        .await
        .unwrap_or_else(|e| panic!("UpdateLayer (section_id={section_id}) failed: {e}"));
}

/// Decode a catch-up body into the OODF delta it carries, so the assertions can
/// read the COMMANDS the server chose to emit rather than only the folded
/// result. The distinction matters here: a retraction-only diff and a
/// content-bearing one both fold onto a fresh state without erroring, and both
/// arrive as a non-empty OMPF.
fn decode_catch_up_delta(body: &[u8]) -> OwnedDelta {
    let container = OmpfContainer::parse(body).expect("parse catch-up OMPF");
    assert!(
        !container.delta_bytes.is_empty(),
        "the catch-up body must carry an OODF delta part"
    );
    oodf_ffi::parse_delta(container.delta_bytes).expect("parse OODF delta")
}

/// The shared body of the b.2ti reconnect tests — three
/// [`ReconnectPosition`]s, the first two run on both catch-up builders.
///
/// `catchup_snapshot_enabled` selects the catch-up builder under test: `true`
/// is the shipped production default (`plan_faithful_coalesced`), `false` the
/// legacy rollback path (`plan_incremental` via `compute_diff_at_cut`). Both filter on
/// `version > have_version`, so both were broken by the same stamp; the harness
/// default is `false` (`tests/common/mod.rs::test_config`), which is why the
/// production path has to be selected explicitly.
async fn assert_post_restore_reconnector_receives_the_restored_state(
    position: ReconnectPosition,
    catchup_snapshot_enabled: bool,
) {
    let mut harness = setup_checkpoint_harness(1).await;
    harness.svc.catchup_snapshot_enabled = catchup_snapshot_enabled;

    let session_store =
        SessionStore::new(harness.tv.client().clone(), harness.tv.key_schema().clone());
    let session_id = session_store
        .create(SESSION_NAME, "/stage", "test-user")
        .await
        .expect("session creation must succeed");

    let create_ompf = delta_to_ompf_bytes(&empty_delta(), &[]);
    let create_stream = streaming_create(vec![
        make_create_header(&session_id, LAYER_NAME),
        make_create_data(create_ompf),
    ]);
    let object_id = handle_create_layer(&harness.svc, make_request(create_stream))
        .await
        .expect("CreateLayer must succeed")
        .into_inner()
        .object_id;

    // The content the checkpoint will capture, and therefore the content the
    // restore must serve back.
    let checkpointed_section_ids: Vec<u64> =
        (0..PRE_CHECKPOINT_UPDATE_COUNT).map(|i| 1000 + i).collect();
    for sid in &checkpointed_section_ids {
        apply_section_and_field_update(&harness, &session_id, object_id, *sid).await;
    }

    let pre_checkpoint_state = snapshot_to_object_state(
        &read_object_state_for_test(
            &harness.svc.valkey,
            &harness.svc.key_schema,
            &object_id.to_string(),
        )
        .await
        .expect("read_object_state_for_test must succeed before the checkpoint"),
    );

    let handle = wait_for_checkpoint_at_least(
        &harness.store,
        object_id,
        pre_checkpoint_state.sequence_num,
        10,
    )
    .await;
    let snap_seq = handle.snap_seq;

    // Stop the periodic loop HERE. Everything below reasons about `snap_seq`
    // being the newest generation — in the `AheadOfCheckpoint` arm a second
    // generation landing after the post-checkpoint commits would capture them
    // and collapse the very gap that arm exists to open.
    harness.cancel.cancel();
    let _ = tokio::time::timeout(Duration::from_secs(5), &mut harness.loop_handle).await;

    let cfg = crate::common::test_config();
    let margin = cfg.checkpoint_seq_safety_margin;

    // The subscriber's tracked version, the restore floor the loss shape
    // implies, and the catch-up regime that combination must select. All three
    // arms leave `obj:hdr` absent, which is the symptom the SR-2.4 recovery
    // branch keys on.
    let (have_version, expected_restore_floor, expected_regime) = match position {
        ReconnectPosition::AtCheckpointSequence => {
            // Wipes the stream too, so the high-water probe folds to `None` and
            // the restore floor is `snap_seq`. No entry survives at
            // `{snap_seq + 1}-0`, so the catch-up is out-of-window.
            wipe_layer_state(&harness.tv, object_id).await;
            (snap_seq, snap_seq, "out_of_window")
        }
        ReconnectPosition::AheadOfCheckpoint => {
            let post_checkpoint_seq =
                commit_past_the_checkpoint(&harness, &session_id, object_id, snap_seq).await;
            // SPARES the stream, so the high-water probe returns
            // `Some(post_checkpoint_seq)` and the restore floor rises above
            // `snap_seq` — which is what makes `seeded_seq` and a hypothetical
            // `snap_seq` stamp land on opposite sides of `have_version`.
            wipe_layer_state_preserving_stream(&harness.tv, object_id).await;
            // `have_version` IS the surviving stream's top, so the in-window
            // XRANGE from `{have_version + 1}-0` returns nothing: out-of-window.
            (post_checkpoint_seq, post_checkpoint_seq, "out_of_window")
        }
        ReconnectPosition::InWindowAfterPartialLoss => {
            let post_checkpoint_seq =
                commit_past_the_checkpoint(&harness, &session_id, object_id, snap_seq).await;
            wipe_layer_state_preserving_stream(&harness.tv, object_id).await;
            // Same surviving stream, same floor — but the subscriber sits at
            // `snap_seq`, BELOW the top, so `{snap_seq + 1}-0` is on the stream
            // and the in-window branch is selected.
            assert_stream_holds_entry_at(&harness, object_id, snap_seq + 1).await;
            (snap_seq, post_checkpoint_seq, "in_window")
        }
    };

    // Sanity: the wipe really did remove the header, so the recovery branch
    // fires rather than the assertions running against surviving state.
    match read_object_state_for_test(
        &harness.svc.valkey,
        &harness.svc.key_schema,
        &object_id.to_string(),
    )
    .await
    {
        Err(live_edit::error::ServiceError::NotFound(_)) => {}
        Ok(_) => panic!("the wipe must leave the object header missing"),
        Err(other) => panic!("the wipe must surface NotFound, got: {other}"),
    }

    // ── The reconnect ────────────────────────────────────────────────────
    let response = handle_subscribe_read_layer(
        &harness.svc,
        subscribe_request(&session_id, object_id, have_version),
    )
    .await
    .expect("SubscribeReadLayer must return Ok after recovery");
    let mut stream = response.into_inner();
    let (content_header, content_body) = recv_message(&mut stream).await;

    // What the restore actually materialized — the oracle for everything below.
    // Read AFTER the subscribe, because the subscribe is what triggered it.
    let restored = snapshot_to_object_state(
        &read_object_state_for_test(
            &harness.svc.valkey,
            &harness.svc.key_schema,
            &object_id.to_string(),
        )
        .await
        .expect("post-recovery read_object_state_for_test must succeed"),
    );
    assert_eq!(
        restored.sections.keys().copied().collect::<HashSet<_>>(),
        checkpointed_section_ids.iter().copied().collect(),
        "fixture precondition: the restore must materialize exactly the checkpoint's content. \
         Anything else means a later generation landed and the arm is no longer testing the \
         position it claims"
    );
    assert_eq!(
        restored.sequence_num,
        expected_restore_floor + margin,
        "the seeded writer cursor is `restore_floor + band`: the floor is `snap_seq` when the \
         stream was wiped along with the rest of `obj:*`, and the surviving stream's high-water \
         mark when it was not"
    );
    if position == ReconnectPosition::AheadOfCheckpoint {
        assert!(
            have_version > snap_seq,
            "the discriminating property of this arm: the subscriber's tracked version \
             ({have_version}) must sit ABOVE the checkpoint's snap_seq ({snap_seq}), so an entry \
             stamped at `snap_seq` would be filtered OUT by `version > have_version` while one \
             stamped at `seeded_seq` is not"
        );
    }

    // ── The claim ────────────────────────────────────────────────────────
    //
    // Message #1 must carry the whole restored state. Pre-b.2ti it carried an
    // EMPTY delta: every restored entry sat at `version = 1`, below
    // `have_version`, so `plan_faithful_coalesced` / `plan_incremental` emitted
    // no section, no field and (the sections being live and reachable) not even
    // a retraction — and `deliver_catch_up_content` shipped the empty payload
    // and acked the subscriber caught-up.
    assert!(
        content_header.total_size > 0,
        "the first message must be a CONTENT message, not the zero-body caught-up terminator"
    );
    let delta = decode_catch_up_delta(&content_body);
    assert!(
        delta.is_diff && delta.base_version == have_version,
        "a reconnector must be served an INCREMENTAL envelope based at its own version \
         (is_diff=true, base_version={have_version}), never a from-empty reset it would reject; \
         got is_diff={}, base_version={}",
        delta.is_diff,
        delta.base_version
    );

    let emitted_sections: HashSet<u64> = delta
        .struct_commands
        .iter()
        .filter_map(|cmd| match cmd {
            StructCommand::DiffChangeSection(s) => Some(s.section_id),
            _ => None,
        })
        .collect();
    assert_eq!(
        emitted_sections,
        restored.sections.keys().copied().collect::<HashSet<_>>(),
        "the catch-up MUST emit every section the restore materialized. An empty set here is the \
         b.2ti regression itself: restored entries stamped at or below the subscriber's \
         have_version ({have_version}) are invisible to the `version > have_version` filter, so \
         the subscriber is told it is caught up while holding none of the restored state"
    );
    // …and none of them may ALSO be retracted. Scoped to the restored section
    // set on purpose, rather than banning `DiffDeleteSection` outright: b.2ti's
    // claim is "the restore's own content reaches the reconnector via the
    // content arms", and a retraction naming a restored section is the exact
    // violation — the ghost arm claimed a section the content arms just emitted,
    // which would leave the subscriber folding an upsert and a delete for the
    // same id.
    //
    // A retraction naming some OTHER id is a different subject entirely and NOT
    // this test's to adjudicate. In the `AheadOfCheckpoint` arm the subscriber
    // genuinely did observe the post-checkpoint sections (2000, 2001) before the
    // loss, and the restore does not bring them back — a catch-up that retracts
    // them is arguably the CORRECT behaviour there. A blanket ban would encode
    // "a DR restore never retracts ghosts" as a b.2ti invariant and turn red on
    // a future ghost-retraction fix for a reason unrelated to this issue.
    let restored_section_ids: HashSet<u64> = restored.sections.keys().copied().collect();
    let retracted_restored: Vec<u64> = delta
        .struct_commands
        .iter()
        .filter_map(|cmd| match cmd {
            StructCommand::DiffDeleteSection(s) if restored_section_ids.contains(&s.section_id) => {
                Some(s.section_id)
            }
            _ => None,
        })
        .collect();
    assert!(
        retracted_restored.is_empty(),
        "the payload must not RETRACT a section the restore materialized — the restored sections \
         are live and reachable, so a `DiffDeleteSection` for one of them means the ghost arm \
         claimed it instead of (or as well as) the content arms. Retracted restored ids: \
         {retracted_restored:?}. Commands: {:?}",
        delta.struct_commands
    );

    let emitted_fields: BTreeMap<(u64, String), Vec<u8>> = delta
        .set_fields
        .iter()
        .map(|f| {
            (
                (f.section_id, f.key_name.clone()),
                f.value_or_ext_hash.clone(),
            )
        })
        .collect();
    // The expectation is built from what the fixture AUTHORED
    // (`incremental_section_and_field`'s `field_key` / `field_value` over the
    // checkpointed section ids), NOT from `restored.fields`. Deriving it from
    // the restored snapshot would make the oracle self-referential: a restore
    // that materialized an empty or corrupted value would satisfy an equality
    // against its own output, and this test's whole subject is what the restore
    // hands back.
    let authored_fields: BTreeMap<(u64, String), Vec<u8>> = checkpointed_section_ids
        .iter()
        .map(|sid| ((*sid, field_key(*sid)), field_value(*sid)))
        .collect();
    assert!(
        !authored_fields.is_empty(),
        "fixture precondition: the fixture must author fields, or the field half of the version \
         filter is not exercised at all"
    );

    // Cross-check the restore against the authored bytes first, so a
    // materialization defect is reported as one rather than as a catch-up
    // shortfall below.
    let restored_fields: BTreeMap<(u64, String), Vec<u8>> = restored
        .fields
        .iter()
        .map(|((section_id, key_name), field)| {
            let value = match &field.value {
                FieldValue::Inline(bytes) => bytes.clone(),
                other => panic!("this fixture writes inline values only, got {other:?}"),
            };
            ((*section_id, key_name.clone()), value)
        })
        .collect();
    assert_eq!(
        restored_fields, authored_fields,
        "the restore must materialize exactly the field keys AND VALUES the fixture authored \
         before the checkpoint — anything else is a checkpoint/restore content defect, and would \
         make the catch-up comparison below vacuous"
    );

    assert_eq!(
        emitted_fields, authored_fields,
        "the catch-up MUST emit every field the fixture authored, with its value — the same \
         `version > have_version` filter governs `Field::version`"
    );

    // Fold it the way a client does, and compare structural identity to the
    // restored state — the end-to-end form of the claim above.
    //
    // `section_order` is deliberately NOT compared: `plan_faithful_coalesced`
    // overrides it with each section's position among its parent's children
    // (the faithful walk's position, `src/domain/diff.rs`) rather than
    // relaying the stored column, so on the
    // shipped path the two legitimately differ for any fixture that authors
    // several siblings at the same stored order. That override is its own
    // contract with its own tests; nothing here turns on it.
    let mut subscriber_state = ObjectState::default();
    apply_ompf_to_state(&mut subscriber_state, &content_body);
    for (sid, restored_section) in &restored.sections {
        let folded = subscriber_state
            .sections
            .get(sid)
            .unwrap_or_else(|| panic!("the folded subscriber state is missing section {sid}"));
        assert_eq!(
            (folded.parent_id, folded.name.as_str(), folded.section_type,),
            (
                restored_section.parent_id,
                restored_section.name.as_str(),
                restored_section.section_type,
            ),
            "section {sid} structural identity must match the restored snapshot"
        );
    }

    assert_eq!(
        content_header.sequence_num, restored.sequence_num,
        "and the delivered content must be labelled with the same sequence the restored header \
         carries — entries and header agree on ONE version after a restore, which is the \
         property b.2ti restored"
    );

    // The caught-up terminator lands next; drain it so a red here cannot be
    // mistaken for a missing content message above.
    let (ack_header, ack_body) = recv_message(&mut stream).await;
    assert_eq!(
        ack_header.total_size, 0,
        "the catch-up must terminate with the zero-body caught-up ack"
    );
    assert!(ack_body.is_empty(), "the ack carries no body bytes");

    // Non-vacuity for the whole test: a restore really did run inside this RPC.
    assert_eq!(
        recovery_attempts(&harness, "restored"),
        1,
        "exactly ONE Restored recovery attempt must have fired for this subscribe"
    );
    assert_eq!(
        recovery_attempts(&harness, "error"),
        0,
        "no Error attempts expected — the LocalDisk store is healthy"
    );

    // The regime is the reconnect one either way; `brand_new` would mean the
    // filtered path was never taken and the whole test is vacuous.
    assert_eq!(
        subscribe_regime_count(&harness, "brand_new"),
        0,
        "the subscriber sent a non-zero have_object_id AND a non-zero have_version, so it MUST \
         NOT be served by the version-blind brand-new path — that path is exactly the one the \
         sibling `..._after_valkey_wipe` test covers, and it cannot see this defect"
    );

    // And WHICH filtered regime it was, pinned per arm rather than asserted in
    // prose. The regime selects the builder — `catch_up_out_of_window` vs
    // `catch_up_in_window_snapshot`, two call sites into
    // `compute_faithful_coalesced_at_cut` with different cursor pins — so an arm that
    // drifted into the other regime would silently stop covering the one it
    // names and duplicate a sibling. Exactly one subscribe RPC runs per test,
    // and `harness.metrics` is freshly constructed per harness, so the cell is
    // exactly 1 / 0.
    assert_eq!(
        subscribe_regime_count(&harness, expected_regime),
        1,
        "the {position:?} arm must be served by the `{expected_regime}` catch-up regime \
         (`run_catch_up_inner`, src/grpc/subscribe.rs)"
    );
    for other in ["in_window", "out_of_window", "up_to_date"] {
        if other == expected_regime {
            continue;
        }
        assert_eq!(
            subscribe_regime_count(&harness, other),
            0,
            "the {position:?} arm must NOT also record the `{other}` regime — one subscribe RPC \
             runs in this test, so a second regime cell means the catch-up re-entered (a \
             re-bootstrap) rather than being served once"
        );
    }
}

/// b.2ti: a subscriber that reconnects at exactly the checkpointed sequence
/// receives the whole restored state, not an empty catch-up.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_reconnector_at_the_checkpoint_sequence_receives_the_full_restored_state() {
    assert_post_restore_reconnector_receives_the_restored_state(
        ReconnectPosition::AtCheckpointSequence,
        true,
    )
    .await;
}

/// The same claim on the legacy catch-up path
/// (`LIVE_EDIT_CATCHUP_SNAPSHOT_ENABLED=false`), which filters through
/// `plan_incremental` rather than `plan_faithful_coalesced`. Both builders read
/// the same persisted `version` columns, so both were blind to a restored layer
/// pre-fix; asserting only the shipped default would leave the rollback knob
/// uncovered.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_reconnector_at_the_checkpoint_sequence_receives_the_full_restored_state_on_the_legacy_path(
) {
    assert_post_restore_reconnector_receives_the_restored_state(
        ReconnectPosition::AtCheckpointSequence,
        false,
    )
    .await;
}

/// b.2ti's discriminating case: the subscriber's tracked version sits ABOVE the
/// checkpoint the restore reads from.
///
/// The stream survives the loss, so the restore floor is its high-water mark
/// rather than `snap_seq`, and `seeded_seq = floor + band` therefore dominates
/// the subscriber's version. A restore that stamped its entries at the
/// CHECKPOINT's sequence instead — the plausible alternative to `seeded_seq`,
/// and the one the sibling arm cannot rule out because there the two coincide —
/// would leave every entry at or below `have_version` and serve this subscriber
/// nothing.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_reconnector_ahead_of_the_checkpoint_receives_the_full_restored_state() {
    assert_post_restore_reconnector_receives_the_restored_state(
        ReconnectPosition::AheadOfCheckpoint,
        true,
    )
    .await;
}

/// The discriminating position on the legacy catch-up path
/// (`LIVE_EDIT_CATCHUP_SNAPSHOT_ENABLED=false`).
///
/// Same shape as the arm above — stream spared, `have_version` at the surviving
/// high-water mark, so the restore floor rises above `snap_seq` — but served by
/// `catch_up_out_of_window`'s legacy branch: ONE coalesced `compute_diff_at_cut` /
/// `plan_incremental` body rather than `compute_faithful_coalesced_at_cut`. The regime
/// is still `out_of_window` (regime selection is the XRANGE probe's verdict and
/// is independent of the builder knob), so this arm and its shipped-path sibling
/// differ in exactly one variable.
///
/// Worth its own arm rather than folding into position 1's legacy sibling for
/// the same reason position 2 exists at all on the shipped path: position 1
/// wipes the stream, so `snap_seq` and the restore floor coincide there and a
/// hypothetical `snap_seq` stamp would still clear `version > have_version`.
/// Only here do the two candidate stamps land on opposite sides of the filter —
/// and `plan_incremental` applies that filter at a different call site than
/// `plan_faithful_coalesced`, so the rollback knob needs the discrimination
/// proven on its own builder.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_reconnector_ahead_of_the_checkpoint_receives_the_full_restored_state_on_the_legacy_path()
{
    assert_post_restore_reconnector_receives_the_restored_state(
        ReconnectPosition::AheadOfCheckpoint,
        false,
    )
    .await;
}

/// b.2ti on the OTHER catch-up regime: a reconnector whose gap falls inside the
/// surviving stream's retention window.
///
/// The two arms above both land in `out_of_window` — one because the wipe took
/// the stream with it, one because the subscriber sits AT the surviving stream's
/// top. Neither can reach `catch_up_in_window_snapshot`, which is a separate
/// call site into `compute_faithful_coalesced_at_cut` (its own consumer pin at
/// `{current_seq}-0`, its own up-front `deleted`-sentinel scan) and applies the
/// same `version > have_version` filter on its own. Sparing the stream while
/// leaving `have_version` at `snap_seq` — one commit BELOW the first surviving
/// post-checkpoint entry — is what selects it.
///
/// No steady-state relay follows the catch-up here despite the surviving
/// entries: the consumer is pinned at `{current_seq}-0` where `current_seq` is
/// the restored header (`floor + band`), which dominates every retained entry
/// id. Message #1 is the coalesced diff and message #2 the caught-up ack, both
/// emitted by the catch-up phase before the steady-state loop starts, so the
/// assertions in the shared body are ordered deterministically regardless.
///
/// Shipped path only, unlike positions 1 and 2 — under
/// `LIVE_EDIT_CATCHUP_SNAPSHOT_ENABLED=false` the in-window regime is
/// `catch_up_in_window`'s per-commit stream replay, not a persisted-state
/// diff, so b.2ti's re-stamp does not reach it. See the block header.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_in_window_reconnector_after_a_partial_loss_receives_the_full_restored_state() {
    assert_post_restore_reconnector_receives_the_restored_state(
        ReconnectPosition::InWindowAfterPartialLoss,
        true,
    )
    .await;
}
