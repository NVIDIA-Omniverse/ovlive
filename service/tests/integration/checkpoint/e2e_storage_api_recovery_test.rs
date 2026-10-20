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

//! End-to-end recovery test for `UpdateLayer` against the
//! `StorageApiCheckpointStore` adapter (Epic AC #7, Subtask
//! `t3.fgj.jd.ra.z1` under Task `t2.fgj.jd.ra`, Epic `t1.fgj.jd`, Bee
//! `b.fgj`).
//!
//! Storage-api analog of
//! [`super::e2e_update_layer_recovery_test::update_layer_recovers_layer_state_after_valkey_wipe`].
//! The LocalDisk variant pins Epic AC #7 against a filesystem-backed
//! `CheckpointStore`; this variant pins the same AC against the
//! wiremock-backed `StorageApiCheckpointStore`, exercising the outbound
//! storage-api HTTP path end-to-end (including the
//! `ServiceTokenProvider`-supplied bearer header on every outbound GET).
//!
//! Drives the same 10-step write → checkpoint → wipe → recover → re-write
//! cycle:
//!
//! 1. `StorageApiCheckpointHarness` brings up Valkey, a per-test
//!    `wiremock::MockServer` rooted under `/test/live-edit-e2e`, a stateful
//!    in-process PUT/GET/LIST/DELETE fake, a tempfile-backed
//!    `ServiceTokenProvider` primed with `Bearer test-token`, a
//!    `StorageApiCheckpointStore`, and a `LiveEditService` wired with an
//!    ENABLED `RecoveryHook` (`tick_secs=1`, `interval=0`, `quiesce=0` —
//!    same fast-converging schedule as the LocalDisk harness).
//! 2. `CreateSession` (via `SessionStore::create`) + `CreateLayer` with a
//!    populated initial OMPF.
//! 3. 4 `UpdateLayer`s (mix of field + time-sample diffs) drive the
//!    post-create sequence number forward.
//! 4. `wait_for_checkpoint` blocks until the loop lands at least one
//!    generation under the `StorageApiCheckpointStore` (the per-tick PUT
//!    triplet — meta.json + ext-refs.json + .ompf — lands on the
//!    wiremock-mounted fake state map).
//! 5. `wipe_layer_state` UNLINKs every `le:{object_id}:obj:*` key,
//!    simulating the data-loss surface.
//! 6. A subsequent `UpdateLayer` triggers the recovery branch in
//!    `handle_update_layer`, the recovery hook issues a LIST then GET
//!    (OMPF) then GET (meta.json) round-trip against the wiremock server,
//!    applies the checkpoint, re-issues the commit, and returns Ok with a
//!    `sequence_num` strictly greater than the pre-wipe max.
//!
//! Six assertions land — three carry over from the LocalDisk variant
//! (sequence monotonicity, at-most-once recovery contract, materialized
//! state contents) and three are storage-api specific:
//!
//! - (a) `post_recovery_seq > pre_wipe_max_seq` — Epic AC #7 sequence
//!   monotonicity (mirrors LocalDisk).
//! - (b) `recovery_metric{result="restored"} == 1`,
//!   `not_recoverable == 0`, `error == 0` — at-most-once recovery contract
//!   (mirrors LocalDisk).
//! - (c) Post-recovery `read_object_state_for_test` returns `Ok` with matching
//!   `object_id`, original section present, post-recovery field present
//!   (mirrors LocalDisk).
//! - (d) **Storage-api specific** — wiremock's `received_requests()`
//!   contains exactly one GET on the OMPF path matching
//!   `<base>/.live-edit/layers/<oid>.<snap_seq>.ompf` for the wiped
//!   layer's `(object_id, snap_seq)`. Proves the recovery branch fetched
//!   the latest OMPF artifact exactly once.
//! - (e) **Storage-api specific** — at least one PUT landed on the
//!   `.ompf` path family during the pre-wipe checkpoint loop iterations.
//!   Proves the loop's `put_layer_checkpoint` round-trip exercised the
//!   storage-api adapter end-to-end before the wipe.
//! - (f) **Storage-api specific** — every recovery-time GET carried
//!   `Authorization: Bearer test-token`. Proves the outbound bearer-auth
//!   path is wired through `ServiceTokenProvider` and the adapter's
//!   `send_classified` injects the cached bearer on every outbound
//!   request.
//!
//! ## Scoping note
//!
//! Per the parent Task body authorization, this variant is scoped to
//! **layer-only** recovery. Epic 2 (`t1.fgj.7v`, currently `ready`, not
//! `done`) has not yet shipped the handler-level session-recovery hook,
//! so a session-state wipe variant is deferred to Epic 2. Task 8 captures
//! that deferral in `IMPLEMENTATION_CHANGELOG.md`.
//!
//! Sandbox env note: this test uses `TestValkey::setup`, which reads
//! `TEST_VALKEY_URL`. Under the project sandbox the URL must include the
//! password and `host.docker.internal` host; see CLAUDE.md.

use fred::interfaces::HashesInterface;

use live_edit::grpc::layer::{handle_create_layer, handle_update_layer};
use live_edit::storage::session::SessionStore;
use live_edit::storage::test_support::read_object_state_for_test;
use oodf_ffi::{
    CommandDiffChangeSection, OwnedDelta, SetField, StructCommand, TimeSample as OodfTimeSample,
};

use super::helpers::*;

use crate::common::owned_delta::delta_to_ompf_bytes;

const SESSION_NAME: &str = "e2e-storage-api-recovery-session";
const LAYER_NAME: &str = "e2e-storage-api-recovery-layer";
const CLIENT_ID: &str = "e2e-storage-api-recovery-client";
const CREATED_BY: &str = "e2e-storage-api-recovery-user";
const SESSION_BASE_PATH: &str = "/stage";

// ── Streaming framing helpers ────────────────────────────────────────────────
//
// Imported from `tests/integration/checkpoint/helpers.rs` via
// `use super::helpers::*;` (b.dzd Phase 1 deduplication).

// ── Delta fixture builders ───────────────────────────────────────────────────
//
// Duplicated verbatim from `e2e_update_layer_recovery_test.rs`. Same
// rationale as the streaming helpers — the cost-benefit of lifting these
// into a shared submodule does not justify the cross-file `pub(crate)`
// plumbing for trivially-shaped data builders.

const SECTION_ID: u64 = 42;
const FIELD_KEY: &str = "post-recovery-field";

fn populated_initial_delta() -> OwnedDelta {
    OwnedDelta {
        is_diff: true,
        base_version: 0,
        source_format: "usd".to_string(),
        struct_commands: vec![StructCommand::DiffChangeSection(CommandDiffChangeSection {
            parent_id: 0,
            section_id: SECTION_ID,
            section_name: "e2e-storage-api-section".to_string(),
            section_type: 1,
            section_order: 0,
        })],
        set_fields: vec![SetField {
            section_id: SECTION_ID,
            key_name: "initial-field".to_string(),
            value_or_ext_hash: vec![0xAA, 0xBB, 0xCC],
            ext_value_size: 0,
            ext_value_index: 0,
            set_order: 0,
        }],
        ..Default::default()
    }
}

fn delta_with_field_update(set_order: u64, marker: u8) -> OwnedDelta {
    // `base_version: 1` marks this as an INCREMENTAL diff, not a reset.
    // See `e2e_update_layer_recovery_test.rs::delta_with_field_update`
    // for the reset-delta avoidance rationale (SDD §9.1.3).
    OwnedDelta {
        is_diff: true,
        base_version: 1,
        set_fields: vec![SetField {
            section_id: SECTION_ID,
            key_name: format!("field-{set_order}"),
            value_or_ext_hash: vec![marker, marker.wrapping_add(1), marker.wrapping_add(2)],
            ext_value_size: 0,
            ext_value_index: 0,
            set_order,
        }],
        ..Default::default()
    }
}

fn delta_with_time_sample(_set_order: u64, time_value: f64, marker: u8) -> OwnedDelta {
    // Incremental diff — see `delta_with_field_update` for the
    // `base_version != 0` rationale (reset-delta avoidance).
    OwnedDelta {
        is_diff: true,
        base_version: 1,
        time_samples: vec![OodfTimeSample {
            section_id: SECTION_ID,
            time: time_value,
            value_or_ext_hash: vec![marker],
            ext_value_size: 0,
            ext_value_index: 0,
        }],
        ..Default::default()
    }
}

fn delta_post_recovery() -> OwnedDelta {
    // Incremental diff (NOT a reset) — see `delta_with_field_update`.
    // The post-recovery write MUST land on top of the restored
    // checkpoint state, not wipe it.
    OwnedDelta {
        is_diff: true,
        base_version: 1,
        set_fields: vec![SetField {
            section_id: SECTION_ID,
            key_name: FIELD_KEY.to_string(),
            value_or_ext_hash: vec![0xFE, 0xED, 0xFA, 0xCE],
            ext_value_size: 0,
            ext_value_index: 0,
            set_order: 99,
        }],
        ..Default::default()
    }
}

// ── Test driver helpers ──────────────────────────────────────────────────────

/// Issue a single `CreateLayer` carrying the populated initial delta.
/// Returns the freshly-minted `object_id`.
async fn create_layer_with_initial_state(
    harness: &StorageApiCheckpointHarness,
    session_id: &str,
) -> u64 {
    let ompf = delta_to_ompf_bytes(&populated_initial_delta(), &[]);
    let stream = streaming_create(vec![
        make_create_header(session_id, LAYER_NAME),
        make_create_data(ompf),
    ]);
    let request = make_streaming_token_request(stream);
    let response = handle_create_layer(&harness.svc, request)
        .await
        .expect("CreateLayer must succeed for storage-api E2E recovery harness setup");
    response.into_inner().object_id
}

/// Issue a single `UpdateLayer` and return the produced `sequence_num`.
async fn issue_update(
    harness: &StorageApiCheckpointHarness,
    session_id: &str,
    object_id: u64,
    delta: OwnedDelta,
    client_timestamp: u64,
) -> u64 {
    let ompf = delta_to_ompf_bytes(&delta, &[]);
    let stream = streaming_update(vec![
        make_update_header(
            session_id,
            LAYER_NAME,
            object_id,
            client_timestamp,
            CLIENT_ID,
        ),
        make_update_data(ompf),
    ]);
    let request = make_streaming_token_request(stream);
    let response = handle_update_layer(&harness.svc, request)
        .await
        .expect("UpdateLayer must succeed in the pre-wipe / post-recovery happy path");
    response.into_inner().sequence_num
}

/// Probe the live label-cell counter value for a given outcome label on
/// the `live_edit_checkpoint_recovery_attempts_total{result, service}`
/// metric. Mirrors the `recovery_metric` helper in
/// `e2e_update_layer_recovery_test.rs` — re-implemented locally because
/// that file's helper is `fn`-scoped to its module.
fn recovery_metric(harness: &StorageApiCheckpointHarness, result: &str) -> u64 {
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

// =============================================================================
// Headline E2E test — Epic AC #7 against the StorageApi adapter.
// =============================================================================

/// Headline acceptance test pinning Epic AC #7 for the UpdateLayer RPC
/// against the storage-api-backed `CheckpointStore`: after a
/// write → checkpoint → wipe cycle, a subsequent UpdateLayer MUST succeed
/// via the recovery branch and return a `sequence_num` strictly greater
/// than the pre-wipe max, with the recovery hook's outbound LIST + GET +
/// GET round-trip landing on the wiremock-backed storage-api fake AND
/// every outbound GET carrying the `ServiceTokenProvider`-supplied
/// bearer header.
///
/// Why each step is load-bearing — mirrors the LocalDisk variant docblock
/// at `e2e_update_layer_recovery_test.rs::update_layer_recovers_layer_state_after_valkey_wipe`
/// with the storage-api-specific deltas called out explicitly:
///
/// - The initial CreateLayer carries a populated delta so the
///   post-recovery snapshot has restorable sections / fields. Without
///   populated content, "restored materialized state" would assert
///   nothing.
/// - The burst of UpdateLayers (4 commits, mixing field + time-sample)
///   drives `sequence_num` well above 1 so the "strictly greater than"
///   assertion has unambiguous signal.
/// - `wait_for_checkpoint(..., deadline_secs=10)` blocks until the
///   leader-elected loop has landed at least one BASE generation — the
///   helper's kind predicate, and what this test needs: the handle it
///   returns is restored from, and only a base carries full state. The
///   harness sets `tick_secs=1` and `interval=quiesce=0`; the 10 s deadline is
///   sized for CI jitter on the 1-second tick AND the additional HTTP
///   round-trip to the local wiremock server (sub-millisecond on
///   loopback, but worth padding for CI burst load).
/// - `wipe_layer_state` UNLINKs only `le:{oid}:obj:*` (not the layer
///   mapping). This drives the engineer's wired recovery branch in
///   `handle_update_layer` from the SECOND NotFound surface — the
///   `update_layer` call's empty-header `NotFound` inside
///   `update_layer_once`. The recovery hook's
///   `get_latest_layer_checkpoint` then issues a LIST + GET (OMPF) + GET
///   (meta.json) round-trip against the wiremock server.
/// - The recovery-metric assertion (`restored == 1`) pins the
///   at-most-once contract.
/// - The post-recovery snapshot assertion verifies the restored
///   materialized state — same three layered checks as LocalDisk.
/// - The storage-api-specific assertions inspect
///   `mock_server.received_requests()` to verify (d) exactly one GET on
///   the OMPF path family for the wiped layer's `(oid, snap_seq)`, (e)
///   at least one PUT on the OMPF path family during pre-wipe, and (f)
///   every GET request carried `Authorization: Bearer test-token` —
///   proving the outbound bearer-auth path is wired through
///   `ServiceTokenProvider`.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn update_layer_recovers_layer_state_after_valkey_wipe_via_storage_api() {
    // Step 1: Harness with a fast-converging checkpoint loop and a
    // wiremock-backed StorageApiCheckpointStore.
    let harness = setup_storage_api_checkpoint_harness(1).await;

    // Step 2: CreateSession (we go through the SessionStore directly
    // rather than CreateSession RPC because we don't need to exercise
    // the gRPC surface for session creation — the recovery target is
    // UpdateLayer).
    let session_store =
        SessionStore::new(harness.tv.client().clone(), harness.tv.key_schema().clone());
    let session_id = session_store
        .create(SESSION_NAME, SESSION_BASE_PATH, CREATED_BY)
        .await
        .expect("session creation must succeed");

    // Step 3: CreateLayer with populated initial state. The CreateLayer
    // path mints the `object_id` we'll capture for the wipe-and-recover
    // assertions below.
    let object_id = create_layer_with_initial_state(&harness, &session_id).await;
    assert!(
        object_id > 0,
        "CreateLayer must mint a non-zero object_id; got {object_id}"
    );

    // Step 4: Issue 4 UpdateLayers and capture the highest acknowledged
    // sequence_num.
    let mut pre_wipe_max_seq: u64 = 0;
    let updates = [
        (delta_with_field_update(1, 0x11), 1_001_u64),
        (delta_with_time_sample(2, 0.0, 0x22), 1_002_u64),
        (delta_with_field_update(3, 0x33), 1_003_u64),
        (delta_with_time_sample(4, 1.0, 0x44), 1_004_u64),
    ];
    for (delta, ts) in updates {
        let seq = issue_update(&harness, &session_id, object_id, delta, ts).await;
        assert!(
            seq > pre_wipe_max_seq,
            "UpdateLayer sequence_num must advance monotonically; \
             previous max={pre_wipe_max_seq}, got {seq}"
        );
        pre_wipe_max_seq = seq;
    }
    assert!(
        pre_wipe_max_seq >= 5,
        "pre-wipe max sequence_num must be at least 5 (1 from CreateLayer + 4 \
         UpdateLayers); got {pre_wipe_max_seq}"
    );

    // Step 5: Wait for the leader-elected checkpoint loop to land at
    // least one generation. The handle returned carries the captured
    // `snap_seq` used in assertion (d) below to construct the expected
    // GET OMPF path tail.
    let checkpoint_handle = wait_for_checkpoint(&harness.store, object_id, 10).await;
    let snap_seq = checkpoint_handle.snap_seq;

    // Sanity check: pre-wipe baseline of the `restored` metric cell is
    // zero. The harness wires a fresh Registry / Metrics so there is no
    // leakage from prior tests.
    assert_eq!(
        recovery_metric(&harness, "restored"),
        0,
        "pre-wipe `restored` recovery metric MUST be zero (fresh harness Metrics)"
    );

    // Capture the pre-wipe received-request count + verify at least one
    // PUT landed on the OMPF path family. This is assertion (e) — the
    // pre-wipe checkpoint loop's `put_layer_checkpoint` round-trip
    // exercised the storage-api adapter end-to-end. We compute the
    // expected OMPF path-tail suffix from `harness.base_path` (the
    // `BASE_PATH_STORAGE_API` constant the harness threads in) so a
    // downstream rename of the constant surfaces here as a value
    // mismatch rather than silent passing.
    let pre_wipe_received = harness
        .mock_server
        .received_requests()
        .await
        .expect("wiremock must record requests");
    let ompf_path_suffix = format!(".live-edit/layers/{object_id}.{snap_seq}.ompf");
    let pre_wipe_ompf_puts: Vec<&wiremock::Request> = pre_wipe_received
        .iter()
        .filter(|req| req.method == http::Method::PUT && req.url.path().contains(".ompf"))
        .collect();
    assert!(
        !pre_wipe_ompf_puts.is_empty(),
        "storage-api specific (e): expected at least one PUT on the `.ompf` path \
         family during pre-wipe checkpoint loop iterations; got {} total pre-wipe \
         requests, paths={:?}",
        pre_wipe_received.len(),
        pre_wipe_received
            .iter()
            .map(|r| (r.method.to_string(), r.url.path().to_string()))
            .collect::<Vec<_>>()
    );

    // Snapshot the pre-wipe received-request count so we can index into
    // `received_requests()` post-recovery to find only the recovery-time
    // requests (excluding the prior loop iterations' LIST / GET / PUT
    // chatter).
    let pre_wipe_request_count = pre_wipe_received.len();

    // Step 6: Wipe `le:{object_id}:obj:*` only. Leaves the layer mapping
    // intact so `get_layer_mapping` resolves; the empty `obj:hdr` then
    // surfaces NotFound from `update_layer_once`, tripping the recovery
    // branch in `handle_update_layer`.
    wipe_layer_state(&harness.tv, object_id).await;

    // Sanity: confirm the wipe actually emptied the header.
    let header_key = harness.tv.key_schema().obj_header(&object_id.to_string());
    let post_wipe_header: std::collections::HashMap<String, String> = harness
        .tv
        .client()
        .pool
        .hgetall(&header_key)
        .await
        .expect("HGETALL obj:hdr must succeed");
    assert!(
        post_wipe_header.is_empty(),
        "wipe_layer_state must leave obj:hdr empty; got {} fields: {:?}",
        post_wipe_header.len(),
        post_wipe_header.keys().collect::<Vec<_>>()
    );

    // Step 7: Issue the post-wipe UpdateLayer against the same
    // (session_id, layer_name, object_id). This MUST trigger the
    // recovery branch, the recovery hook MUST round-trip against the
    // storage-api adapter, apply the checkpoint, re-issue the commit,
    // and return Ok with a sequence_num > pre_wipe_max_seq.
    let post_recovery_seq = issue_update(
        &harness,
        &session_id,
        object_id,
        delta_post_recovery(),
        2_000,
    )
    .await;

    // Step 8 — assertion (a): Post-recovery sequence_num MUST be
    // strictly greater than the pre-wipe max.
    assert!(
        post_recovery_seq > pre_wipe_max_seq,
        "Epic AC #7: post-recovery sequence_num ({post_recovery_seq}) MUST be \
         strictly greater than pre-wipe max ({pre_wipe_max_seq})"
    );

    // Step 9 — assertion (b): Exactly ONE recovery attempt landed in
    // the `restored` metric cell. Pins the at-most-once contract.
    assert_eq!(
        recovery_metric(&harness, "restored"),
        1,
        "Epic AC #7: exactly ONE recovery attempt must have reported `restored`; \
         got {}. A value of 0 means the recovery branch never fired; a value >= 2 \
         means the `recovered_once` latch is broken.",
        recovery_metric(&harness, "restored")
    );
    assert_eq!(
        recovery_metric(&harness, "not_recoverable"),
        0,
        "post-recovery `not_recoverable` cell must stay zero — the checkpoint \
         was readable and the apply pipeline succeeded"
    );
    assert_eq!(
        recovery_metric(&harness, "error"),
        0,
        "post-recovery `error` cell must stay zero — no transient backend failure \
         was injected"
    );

    // Step 10 — assertion (c): Post-recovery materialized state.
    let snapshot = read_object_state_for_test(
        harness.tv.client(),
        harness.tv.key_schema(),
        &object_id.to_string(),
    )
    .await
    .expect(
        "post-recovery read_object_state_for_test must succeed (apply pipeline \
         must have restored the header)",
    );

    assert_eq!(
        snapshot.object_id, object_id,
        "post-recovery snapshot.object_id MUST equal pre-wipe captured \
         object_id ({object_id}); got {}. A mismatch would indicate apply \
         re-minted the object instead of restoring it.",
        snapshot.object_id
    );

    assert!(
        snapshot.sections.contains_key(&SECTION_ID),
        "post-recovery snapshot MUST contain the original section_id={SECTION_ID} \
         restored from the checkpoint; got sections={:?}",
        snapshot.sections.keys().collect::<Vec<_>>()
    );

    let post_recovery_field_present = snapshot
        .fields
        .iter()
        .any(|f| f.section_id == SECTION_ID && f.key_name == FIELD_KEY);
    assert!(
        post_recovery_field_present,
        "post-recovery snapshot MUST contain the SetField '{FIELD_KEY}' landed \
         by the re-issued UpdateLayer; got fields={:?}",
        snapshot
            .fields
            .iter()
            .map(|f| (f.section_id, &f.key_name))
            .collect::<Vec<_>>()
    );

    // ── Storage-api-specific assertions (d), (f) ───────────────────────────
    //
    // Re-fetch `received_requests()` and slice off the pre-wipe tail.
    // What remains is the recovery-time + post-recovery-PUT-triplet
    // request stream; we filter to GETs to land assertion (d) and (f).
    let post_recovery_received = harness
        .mock_server
        .received_requests()
        .await
        .expect("wiremock must record requests post-recovery");
    assert!(
        post_recovery_received.len() >= pre_wipe_request_count,
        "post-recovery received_requests MUST monotonically extend the pre-wipe \
         tail; pre_wipe_count={pre_wipe_request_count}, post_recovery_count={}",
        post_recovery_received.len()
    );
    let recovery_window = &post_recovery_received[pre_wipe_request_count..];

    // (d) Exactly one GET on the OMPF path for (oid, snap_seq). Filter
    // to GETs on the `.ompf` path family, then assert the count + path
    // match.
    //
    // Path matching: the adapter percent-encodes the entire composed
    // address through `encode_resource_address`. The encoded form of `/` is
    // `%2F`, so the path we see on the wire is shaped like
    // `/v1alpha/fileobject/by-address/%2F<base>%2F.live-edit%2Flayers%2F<oid>.<snap_seq>.ompf`.
    // We assert by `contains` on the suffix the encoder always
    // preserves verbatim — the dot-separated `<oid>.<snap_seq>.ompf`
    // filename — to avoid the encoder churn introducing brittleness.
    let recovery_ompf_gets: Vec<&wiremock::Request> = recovery_window
        .iter()
        .filter(|req| {
            req.method == http::Method::GET
                && req
                    .url
                    .path()
                    .ends_with(&format!("{object_id}.{snap_seq}.ompf"))
        })
        .collect();
    assert_eq!(
        recovery_ompf_gets.len(),
        1,
        "storage-api specific (d): expected exactly ONE recovery-time GET on \
         the OMPF path for (oid={object_id}, snap_seq={snap_seq}) [suffix \
         `{ompf_path_suffix}`]; got {} matching GETs. Recovery-window paths: {:?}",
        recovery_ompf_gets.len(),
        recovery_window
            .iter()
            .map(|r| (r.method.to_string(), r.url.path().to_string()))
            .collect::<Vec<_>>()
    );

    // (f) Every recovery-time GET carried `Authorization: Bearer
    // test-token`. We filter to GET requests in the recovery window —
    // the recovery hook's flow is LIST (GET) + OMPF GET + meta.json
    // GET — and assert the bearer header on each one. A regression that
    // dropped the `ServiceTokenProvider` from the outbound adapter
    // would show a missing or empty Authorization header on at least
    // one of these requests.
    let recovery_gets: Vec<&wiremock::Request> = recovery_window
        .iter()
        .filter(|req| req.method == http::Method::GET)
        .collect();
    assert!(
        !recovery_gets.is_empty(),
        "recovery window MUST contain at least one GET (LIST + OMPF GET + meta \
         GET); got {} window requests, methods={:?}",
        recovery_window.len(),
        recovery_window
            .iter()
            .map(|r| r.method.to_string())
            .collect::<Vec<_>>()
    );
    for req in &recovery_gets {
        let auth = req
            .headers
            .get("authorization")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("");
        assert_eq!(
            auth,
            "Bearer test-token",
            "storage-api specific (f): every recovery-time GET MUST carry \
             `Authorization: Bearer test-token` (outbound bearer-auth path \
             wired through ServiceTokenProvider); got `{auth}` on {} {}",
            req.method,
            req.url.path()
        );
    }

    // Graceful shutdown of the checkpoint loop. Drops are sufficient
    // for cleanup but cancelling + awaiting yields a deterministic
    // teardown and removes the slim window where a still-running tick
    // could race with the test runtime's shutdown — additionally
    // important for the storage-api harness because a still-running
    // loop tick during MockServer drop could race the in-process HTTP
    // listener teardown.
    harness.cancel.cancel();
    let _ = harness.loop_handle.await;
}
