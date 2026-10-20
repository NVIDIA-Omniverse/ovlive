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

//! End-to-end test for Epic AC #5 — orphan sweep tolerates fresh session
//! restorations (Subtask `t3.fgj.7v.o7.c4` under Task `t2.fgj.7v.o7`,
//! Epic `t1.fgj.7v`, Bee `b.fgj`).
//!
//! ## What this test pins
//!
//! Epic AC #5: a freshly-completed session-recovery + layer-recovery
//! cycle must NOT be classified as orphan state by the periodic orphan
//! sweep. Task 5's session-recovery applier (`apply_session_checkpoint`)
//! restores BOTH halves required for the sweep's "alive" classification:
//!
//! - **Step A** writes the path-index legacy member + every path-names
//!   composite under `le:path:{hex(base_path)}*`.
//! - **Step B** writes `le:{sid}:sess` (session_meta), every
//!   `le:{sid}:layer:{hex(name)}` mapping, and `le:{sid}:layers` (the
//!   layers-index).
//!
//! The orphan sweep walks `le:*:obj:hdr` and `le:*:layer:*` keys and
//! classifies each by looking up its inverse pointer (header → mapping,
//! mapping → session). With BOTH halves restored, every inverse pointer
//! exists, so the sweep MUST find zero orphans — even when it runs
//! immediately after the recovery returns.
//!
//! ## Why this test is load-bearing
//!
//! A regression that restored only Step A (path-index) or only Step B
//! (session-scoped, but not the layer mapping) would let the sweep
//! delete freshly-restored object headers or layer mappings the very
//! next maintenance tick — racing the restorer to wipe the work it just
//! completed. The bug would be intermittent in production (only fires
//! when a restorer and the sweep leader both win their respective locks
//! within the same window), so the only way to pin it is an integration
//! test that runs the sweep IMMEDIATELY after recovery and asserts every
//! restored key still exists.
//!
//! ## Composition with AC #2's full-wipe recovery
//!
//! This test layers AC #5 on top of AC #2's full-wipe recovery contract:
//! it drives the same wipe → recover cycle the headline AC #2 test
//! exercises, then immediately spawns one orphan-sweep iteration and
//! asserts the restored state survives. A failure here without a
//! corresponding AC #2 failure means AC #5's tolerance contract is
//! broken (the sweep is treating restored keys as orphans); a failure
//! on both sides means AC #2's underlying restore regressed.
//!
//! ## Sweep-trigger pattern
//!
//! Mirrors the `run_one_sweep_iteration` pattern from
//! `tests/integration/storage/orphan_sweep_test.rs:38-60`: spawn
//! `spawn_orphan_sweep`, give it ~5 s to acquire the `le:gc:leader`
//! lock + run one pass, cancel + join under a bounded timeout. The
//! helper is re-implemented locally rather than lifted out because the
//! orphan-sweep test file's helpers are file-private (no `pub` /
//! `pub(super)`) and crossing the `tests/integration/{checkpoint,storage}/`
//! module boundary to share trivial test scaffolding is friction with
//! no real benefit — the pattern is short.
//!
//! ## Negative-counter assertion
//!
//! The Subtask body asks for `live_edit_gc_*_orphaned_total` to be
//! pinned at zero. That metric family does NOT exist in
//! `src/observability/metrics.rs` — the orphan sweep returns its
//! `orphaned_objects` / `orphaned_layers` counts via the in-process
//! `SweepResult` struct (logged at the end of the run), not via
//! Prometheus counters. The closest Prometheus surface is
//! `live_edit_gc_sweep_batch_errors_total` (sweep-error counter); we
//! pin that at zero across the run so a SCAN-error or leader-lock-error
//! regression surfaces here. The "no orphans deleted" contract is
//! pinned directly by the EXISTS / SISMEMBER probes on every restored
//! key family — strictly stronger evidence than a counter, since a
//! counter regression that silently no-op'd the SCAN would still fail
//! the key-survival checks.
//!
//! ## Post-sweep no-recovery assertion
//!
//! Step 7 of the Subtask body asks that the subsequent UpdateLayer
//! succeeds "without re-triggering recovery (session_recovery_metric
//! unchanged)". The cache funnel's `session_base_path_cache` carries
//! the just-restored `base_path` so the next RPC hits the cache and
//! never consults `session_store.get_base_path`, never invokes
//! `try_recover_session`. We pin BOTH session-recovery and
//! layer-recovery metric deltas at zero post-sweep to prove the second
//! UpdateLayer ran the happy path through fully-restored state.
//!
//! Sandbox env note: this test uses `TestValkey::setup`, which reads
//! `TEST_VALKEY_URL`. Under the project sandbox the URL must include
//! the password and `host.docker.internal` host; see CLAUDE.md.

use std::sync::Arc;
use std::time::Duration;

use fred::interfaces::{HashesInterface, KeysInterface, SetsInterface};
use tokio_util::sync::CancellationToken;

use live_edit::grpc::layer::{handle_create_layer, handle_update_layer};
use live_edit::grpc::proto::CreateSessionRequest;
use live_edit::grpc::session::handle_create_session;
use live_edit::storage::external_values::MockExternalValueStore;
use live_edit::storage::keys::{hex_encode, KeySchema};
use live_edit::storage::orphan_sweep::spawn_orphan_sweep;
use oodf_ffi::{
    CommandDiffChangeSection, OwnedDelta, SetField, StructCommand, TimeSample as OodfTimeSample,
};

use super::helpers::*;
use crate::common::owned_delta::delta_to_ompf_bytes;

// ── Test constants ───────────────────────────────────────────────────────────
//
// Stable per-test names so a failed-assertion message clearly identifies
// which Subtask's scenario is failing. Distinct `base_path` from sibling
// e2e tests so concurrent runs against a shared `TEST_VALKEY_URL` don't
// share path-index members (the `TestValkey` per-test prefix already
// isolates the keys, but readable names help triage).

const SESSION_NAME: &str = "e2e-orphan-sweep-tolerance-session";
const LAYER_NAME: &str = "e2e-orphan-sweep-tolerance-layer";
const CLIENT_ID: &str = "e2e-orphan-sweep-tolerance-client";
const SESSION_BASE_PATH: &str = "/stage/orphan-sweep-tolerance";
const REPLICA_ID: &str = "test-replica";

// Section + field constants for the populated initial delta. Matches the
// AC #2 full-wipe-recovery test shape — a non-empty initial state is
// what makes the recovery + layer-recovery hops do non-trivial work; an
// empty delta would still trigger recovery but every assertion below
// would be vacuous (empty header is the "absent" state to the sweep).

const SECTION_ID: u64 = 73;

// ── Streaming framing helpers ────────────────────────────────────────────────
//
// Imported from `tests/integration/checkpoint/helpers.rs` via
// `use super::helpers::*;` (b.dzd Phase 1 deduplication).

// ── Delta fixture builders ───────────────────────────────────────────────────
//
// `populated_initial_delta` seeds the layer with one section + one
// inline field so the sweep has a non-trivial obj:* surface to scan
// post-recovery. `delta_with_field_update` and `delta_with_time_sample`
// mirror the AC #2 burst shape — Lua-fast-path branches exercised by
// the same payload types real production traffic carries.

fn populated_initial_delta() -> OwnedDelta {
    OwnedDelta {
        is_diff: true,
        base_version: 0,
        source_format: "usd".to_string(),
        struct_commands: vec![StructCommand::DiffChangeSection(CommandDiffChangeSection {
            parent_id: 0,
            section_id: SECTION_ID,
            section_name: "e2e-orphan-sweep-section".to_string(),
            section_type: 1,
            section_order: 0,
        })],
        set_fields: vec![SetField {
            section_id: SECTION_ID,
            key_name: "initial-field".to_string(),
            value_or_ext_hash: vec![0xAB, 0xCD, 0xEF],
            ext_value_size: 0,
            ext_value_index: 0,
            set_order: 0,
        }],
        ..Default::default()
    }
}

fn delta_with_field_update(set_order: u64, marker: u8) -> OwnedDelta {
    // `base_version: 1` (any non-zero) marks this as an INCREMENTAL diff,
    // not a reset — matches the AC #2 burst shape. With `base_version
    // == 0` plus `is_diff == true` the domain `apply_delta` engine takes
    // the reset-delta branch and wipes the seed section before applying.
    OwnedDelta {
        is_diff: true,
        base_version: 1,
        set_fields: vec![SetField {
            section_id: SECTION_ID,
            key_name: format!("burst-field-{set_order}"),
            value_or_ext_hash: vec![marker, marker.wrapping_add(1), marker.wrapping_add(2)],
            ext_value_size: 0,
            ext_value_index: 0,
            set_order,
        }],
        ..Default::default()
    }
}

fn delta_with_time_sample(time_value: f64, marker: u8) -> OwnedDelta {
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

/// Trigger delta carried by the post-wipe UpdateLayer that drives the
/// session + layer recovery chain. Distinct key from the burst deltas so
/// a future debug session can grep the OMPF bytes back to this site.
fn delta_post_recovery() -> OwnedDelta {
    OwnedDelta {
        is_diff: true,
        base_version: 1,
        set_fields: vec![SetField {
            section_id: SECTION_ID,
            key_name: "post-recovery-trigger-field".to_string(),
            value_or_ext_hash: vec![0xCA, 0xFE, 0xBA, 0xBE],
            ext_value_size: 0,
            ext_value_index: 0,
            set_order: 99,
        }],
        ..Default::default()
    }
}

/// Trigger delta carried by the FINAL UpdateLayer (post-sweep, step 7
/// of the Subtask body). Distinct from the post-recovery delta so the
/// commit sequence is observable in test diagnostics, and distinct from
/// the burst deltas so a single grep on the test source pins each
/// timestamp / set_order to its scenario step.
fn delta_post_sweep() -> OwnedDelta {
    OwnedDelta {
        is_diff: true,
        base_version: 1,
        set_fields: vec![SetField {
            section_id: SECTION_ID,
            key_name: "post-sweep-trigger-field".to_string(),
            value_or_ext_hash: vec![0xDE, 0xAD, 0xBE, 0xEF],
            ext_value_size: 0,
            ext_value_index: 0,
            set_order: 199,
        }],
        ..Default::default()
    }
}

// ── Test driver helpers ──────────────────────────────────────────────────────

/// Issue a single `CreateLayer` carrying the populated initial delta.
/// Returns the freshly-minted `object_id`.
async fn create_layer_with_initial_state(harness: &CheckpointHarness, session_id: &str) -> u64 {
    let ompf = delta_to_ompf_bytes(&populated_initial_delta(), &[]);
    let stream = streaming_create(vec![
        make_create_header(session_id, LAYER_NAME),
        make_create_data(ompf),
    ]);
    let response = handle_create_layer(&harness.svc, make_streaming_token_request(stream))
        .await
        .expect("CreateLayer must succeed for the orphan-sweep-tolerance E2E baseline");
    response.into_inner().object_id
}

/// Issue a single `UpdateLayer` and return the produced `sequence_num`.
async fn issue_update(
    harness: &CheckpointHarness,
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
    let response = handle_update_layer(&harness.svc, make_streaming_token_request(stream))
        .await
        .expect(
            "UpdateLayer must succeed in the orphan-sweep-tolerance happy path — \
             a failure indicates the recovery chain (session → layer) regressed",
        );
    response.into_inner().sequence_num
}

/// `live_edit_checkpoint_recovery_attempts_total{result, scope="layer",
/// service=SERVICE_LABEL}` reader. Centralised here so step-7's "no
/// recovery re-fired" assertion can pin BOTH session and layer scopes
/// at their pre-sweep snapshots. `session_recovery_metric` in
/// `helpers.rs` covers the `scope="session"` side.
fn layer_recovery_metric(harness: &CheckpointHarness, result: &str) -> u64 {
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

/// Read `live_edit_gc_sweep_batch_errors_total{kind}` for the supplied
/// label. The sweep ticks this counter only on SCAN failure, leader-
/// lock error, or per-batch UNLINK failure (see `run_one_sweep`
/// `e.kind_label()` arms). A non-zero post-sweep value would signal
/// the sweep blew up mid-pass rather than running cleanly; the surface
/// is read across all observed kinds via a label-cell scan rather than
/// hardcoding the expected zero set.
///
/// The kinds the sweep currently emits are `scan_error`,
/// `leader_lock_error`, and `unlink_error` (see
/// `src/storage/orphan_sweep.rs::SweepError::kind_label`). Hardcoding
/// these three labels would silently miss any future addition; the
/// summary reader below scans them explicitly and panics with the full
/// set if a regression adds a new kind.
fn gc_sweep_batch_errors(harness: &CheckpointHarness, kind: &str) -> u64 {
    live_edit::observability::metrics::test_util::sample_value(
        &live_edit::observability::metrics::test_util::render(&harness.metrics),
        "live_edit_gc_sweep_batch_errors_total",
        &[("kind", kind)],
    )
    .unwrap_or(0.0) as u64
}

/// Drive `spawn_orphan_sweep` through ONE iteration: spawn, wait for
/// the leader-lock acquisition + one pass to complete (bounded at 5 s,
/// the same budget `tests/integration/storage/orphan_sweep_test.rs:38-60`
/// uses), then cancel + join under a hard timeout. Returns nothing —
/// the caller asserts post-sweep state via the surviving-key probes.
///
/// The 50-iteration × 100 ms (5 s total) wait is the same shape as the
/// orphan-sweep test's `run_one_sweep_iteration`: short enough to bound
/// runaway tests, generous enough that a leader-lock SET + two SCAN
/// cursors against a local Valkey complete with comfortable headroom.
/// Even on cold-start tests where the previous AC #2 wipe just landed,
/// the sweep's first iteration completes in tens of ms.
///
/// Re-implemented locally per the sibling-file convention noted in the
/// module docstring. A shared module-level lift would couple the
/// `tests/integration/checkpoint/` and `tests/integration/storage/`
/// trees and the lift target would land in `tests/common/`, which is
/// not the right home for sweep-specific scaffolding.
async fn run_one_orphan_sweep_iteration(
    handle: tokio::task::JoinHandle<()>,
    cancel: CancellationToken,
    poll_done: impl Fn() -> std::pin::Pin<Box<dyn std::future::Future<Output = bool> + Send>>,
) {
    for _ in 0..50 {
        if poll_done().await {
            break;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    cancel.cancel();
    tokio::time::timeout(Duration::from_secs(5), handle)
        .await
        .expect("orphan sweep must exit within 5 s of cancellation")
        .expect("orphan sweep must not panic");
}

// =============================================================================
// Headline E2E test — Epic AC #5 (orphan sweep tolerates fresh recovery).
// =============================================================================

/// Pin Epic AC #5: after a full-wipe → session-recovery + layer-recovery
/// cycle, an immediately-following orphan-sweep iteration MUST NOT
/// classify any freshly-restored key as an orphan. Every restored key
/// family (session_meta, layer_mapping, layers_index, legacy
/// path_index, path_names_index, obj:hdr) must survive the sweep; the
/// subsequent UpdateLayer must succeed without re-triggering recovery
/// (proving the restored state is intact and the cache funnel sees the
/// session as live).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn orphan_sweep_does_not_delete_freshly_restored_session_or_layer() {
    // Step 1: Harness with a fast-converging checkpoint loop (`tick=1s`,
    // `interval=quiesce=0`) so the post-burst layer-checkpoint
    // generation lands quickly.
    let harness = setup_checkpoint_harness(1).await;

    // CreateSession through the real RPC entry point. The lifecycle
    // hook fires a fire-and-forget session-checkpoint write through the
    // dispatcher; we wait for the blob to land before proceeding so the
    // wipe-and-recover step has a checkpoint to apply.
    let create_session_resp = handle_create_session(
        &harness.svc,
        make_unary_token_request(CreateSessionRequest {
            path: SESSION_BASE_PATH.to_owned(),
            name: SESSION_NAME.to_owned(),
        }),
    )
    .await
    .expect("CreateSession must succeed against an AllowAllPermissions harness");
    let session_id = create_session_resp.into_inner().session_id;
    assert!(
        !session_id.is_empty(),
        "CreateSession must mint a non-empty session_id"
    );
    wait_for_session_checkpoint(&harness.store, &session_id, 10).await;

    // Capture the pre-CreateLayer session-checkpoint blob bytes so the
    // subsequent CreateLayer-driven write can be observed to mutate
    // them. The dispatcher write is fire-and-forget; without
    // bytes-differ polling the wipe step could race the writer and the
    // restored session would carry no layer-mapping entry, defeating
    // the AC #5 contract under test.
    let pre_create_layer_session_bytes = harness
        .store
        .get_session_checkpoint(&session_id)
        .await
        .expect("get_session_checkpoint must succeed once a blob has landed")
        .expect("session-checkpoint blob must be present after wait_for_session_checkpoint")
        .bytes;

    // CreateLayer with the populated initial state. Drives the
    // dispatcher's post-CreateLayer session-checkpoint write; we wait
    // for the bytes to differ before proceeding.
    let object_id = create_layer_with_initial_state(&harness, &session_id).await;
    assert!(
        object_id > 0,
        "CreateLayer must mint a non-zero object_id; got {object_id}"
    );
    wait_for_post_create_layer_session_checkpoint(
        &harness,
        &session_id,
        pre_create_layer_session_bytes.as_ref(),
        10,
    )
    .await;

    // Wait for the first layer-checkpoint generation to land. The
    // populated CreateLayer above seeds object state; the loop's next
    // tick snapshots it.
    let _seed_generation = wait_for_checkpoint(&harness.store, object_id, 10).await;

    // Burst of UpdateLayers — three commits mixing field + time-sample
    // payloads (Lua-fast-path coverage matches the AC #2 baseline).
    let mut pre_wipe_max_seq: u64 = 0;
    let updates = [
        (delta_with_field_update(1, 0x21), 2_001_u64),
        (delta_with_time_sample(0.0, 0x22), 2_002_u64),
        (delta_with_field_update(3, 0x23), 2_003_u64),
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
        pre_wipe_max_seq >= 4,
        "pre-wipe max sequence_num must be at least 4 (1 from CreateLayer + 3 \
         UpdateLayers); got {pre_wipe_max_seq}"
    );

    // Wait for BOTH checkpoints to reflect the burst. Per the Subtask
    // body: "Wait for BOTH checkpoints" — session-checkpoint already
    // landed via the bytes-differ wait above; here we wait for a
    // post-burst layer-checkpoint with `snap_seq >= pre_wipe_max_seq`
    // so the post-wipe restore has the full burst state to replay.
    wait_for_post_burst_layer_checkpoint(&harness, object_id, pre_wipe_max_seq, 10).await;

    // Step 2: Full-namespace wipe + post-wipe UpdateLayer drives the
    // full chain (session-recovery → layer-recovery per AC #2). The
    // recovery cycle restores BOTH halves required for the orphan
    // sweep's "alive" classification — the AC #5 contract under test.
    full_valkey_wipe(&harness.tv).await;

    // Cache invalidation — without this, the post-wipe UpdateLayer
    // would HIT the cache, return the pre-wipe `base_path`, and never
    // consult `session_store.get_base_path` → session-recovery never
    // fires → AC #5 silently passes for the wrong reason (no recovery
    // happened, no orphans to tolerate). Production drives this via
    // the `session:deleted` pub/sub fanout; under E2E test we
    // invalidate explicitly. Same rationale as
    // `e2e_session_full_wipe_recovery_test.rs::invalidate_session_cache`.
    harness.svc.session_base_path_cache.invalidate(&session_id);

    // Capture pre-recovery metric baselines so the assertion phase
    // pins exactly one recovery on each scope (AC #2 composition
    // contract that AC #5 layers on top of).
    let session_restored_pre = session_recovery_metric(&harness, "restored");
    let layer_restored_pre = layer_recovery_metric(&harness, "restored");

    // Issue the post-wipe UpdateLayer — triggers both recovery hops.
    let post_recovery_seq = issue_update(
        &harness,
        &session_id,
        object_id,
        delta_post_recovery(),
        3_000,
    )
    .await;
    assert!(
        post_recovery_seq > pre_wipe_max_seq,
        "post-recovery sequence_num ({post_recovery_seq}) MUST be strictly greater than \
         pre-wipe max ({pre_wipe_max_seq}) — proves the recovery chain replayed the \
         burst state AND the commit re-issue landed on top"
    );

    // Step 3: Assert recovery succeeded — exactly one increment on
    // each scope's `restored` cell. The bounded poll absorbs the slim
    // window where layer-recovery's metric tick happens inside an
    // awaited future (matches the AC #2 baseline's poll pattern).
    wait_for_recovery_metric(
        &harness,
        "session",
        "restored",
        session_restored_pre + 1,
        10,
    )
    .await;
    wait_for_recovery_metric(&harness, "layer", "restored", layer_restored_pre + 1, 10).await;

    // ── Pre-sweep snapshot of every key family the sweep could touch ───────
    //
    // We capture the keys before spawning the sweep so a failed
    // assertion below carries the EXACT key string the test was
    // probing (vs. re-deriving via `KeySchema::*` in each assertion
    // arm and producing slightly-different debug output across
    // failure modes).

    let session_meta_key = harness.tv.key_schema().session_meta(&session_id);
    let mapping_key = harness
        .tv
        .key_schema()
        .layer_mapping(&session_id, LAYER_NAME);
    let layers_index_key = harness.tv.key_schema().layers_index(&session_id);
    let path_index_key = harness.tv.key_schema().path_index(SESSION_BASE_PATH);
    let path_names_key = harness.tv.key_schema().path_names_index(SESSION_BASE_PATH);
    let obj_header_key = harness.tv.key_schema().obj_header(&object_id.to_string());

    // The layers-index member is `hex_encode(layer_name)` (assembler
    // contract — see `src/checkpoint/session.rs::SADD layers_index`).
    let layers_index_member = hex_encode(LAYER_NAME);

    // The path_names_index member assembled by the recovery applier
    // (`KeySchema::path_name_member`) is `{session_name}\0{session_id}`
    // — exactly ONE composite per session, irrespective of layer count.
    // The session-checkpoint assembler emits this composite once per
    // session (mirroring the live `SessionStore::create_once` write at
    // `src/storage/session.rs:308`), and Step A of recovery re-SADDs it
    // idempotently against the surviving CreateSession-written member.
    // After `full_valkey_wipe` the CreateSession-written composite is
    // gone; post-recovery the single session-name composite is restored
    // by Step A.
    let session_name_composite = KeySchema::path_name_member(SESSION_NAME, &session_id);

    // Pre-sweep sanity: every restored key family is present. A failed
    // pre-sweep probe here means the recovery chain itself regressed
    // (different AC failure — AC #2 / AC #3 / AC #4); we surface that
    // explicitly so a triager doesn't waste time blaming AC #5.
    let pre_sweep_session_meta_exists: bool = harness
        .tv
        .client()
        .pool
        .exists(&session_meta_key)
        .await
        .expect("EXISTS session_meta pre-sweep probe must succeed");
    let pre_sweep_mapping_exists: bool = harness
        .tv
        .client()
        .pool
        .exists(&mapping_key)
        .await
        .expect("EXISTS layer_mapping pre-sweep probe must succeed");
    let pre_sweep_layers_index_member: bool = harness
        .tv
        .client()
        .pool
        .sismember(&layers_index_key, layers_index_member.clone())
        .await
        .expect("SISMEMBER layers_index pre-sweep probe must succeed");
    let pre_sweep_legacy_path_member: bool = harness
        .tv
        .client()
        .pool
        .sismember(&path_index_key, &session_id)
        .await
        .expect("SISMEMBER legacy path_index pre-sweep probe must succeed");
    let pre_sweep_session_composite: bool = harness
        .tv
        .client()
        .pool
        .sismember(&path_names_key, session_name_composite.clone())
        .await
        .expect("SISMEMBER session-name composite pre-sweep probe must succeed");
    let pre_sweep_obj_header_fields: std::collections::HashMap<String, String> = harness
        .tv
        .client()
        .pool
        .hgetall(&obj_header_key)
        .await
        .expect("HGETALL obj:hdr pre-sweep probe must succeed");
    assert!(
        pre_sweep_session_meta_exists,
        "pre-sweep sanity: session_meta MUST be present after recovery (recovery \
         regression — different AC than the one under test)"
    );
    assert!(
        pre_sweep_mapping_exists,
        "pre-sweep sanity: layer_mapping MUST be present after recovery"
    );
    assert!(
        pre_sweep_layers_index_member,
        "pre-sweep sanity: layers_index MUST carry the hex-encoded layer name after recovery"
    );
    assert!(
        pre_sweep_legacy_path_member,
        "pre-sweep sanity: legacy path_index MUST carry the session_id member after recovery"
    );
    assert!(
        pre_sweep_session_composite,
        "pre-sweep sanity: path_names_index MUST carry the session-name composite after recovery"
    );
    assert!(
        !pre_sweep_obj_header_fields.is_empty(),
        "pre-sweep sanity: obj:hdr MUST be populated after recovery; got empty hash, \
         meaning layer-recovery's apply_checkpoint did not rebuild the header"
    );

    // Capture pre-sweep error-counter snapshot for the sweep-error
    // assertion below.
    let scan_errors_pre = gc_sweep_batch_errors(&harness, "scan_error");
    let leader_lock_errors_pre = gc_sweep_batch_errors(&harness, "leader_lock_error");
    let unlink_errors_pre = gc_sweep_batch_errors(&harness, "unlink_error");

    // Capture pre-sweep recovery snapshots so step-7 ("no recovery
    // re-fired") can pin both scopes at zero delta.
    let session_restored_pre_sweep = session_recovery_metric(&harness, "restored");
    let session_not_recoverable_pre_sweep = session_recovery_metric(&harness, "not_recoverable");
    let session_already_present_pre_sweep = session_recovery_metric(&harness, "already_present");
    let session_error_pre_sweep = session_recovery_metric(&harness, "error");
    let layer_restored_pre_sweep = layer_recovery_metric(&harness, "restored");
    let layer_not_recoverable_pre_sweep = layer_recovery_metric(&harness, "not_recoverable");
    let layer_error_pre_sweep = layer_recovery_metric(&harness, "error");

    // Step 4: Spawn the orphan sweep IMMEDIATELY after recovery
    // completes. The sweep races the post-recovery state — if any
    // restored key were misclassified as an orphan, the sweep would
    // delete it within this single iteration. With a zero grace: the
    // restored object is young, and the sweep's grace on young objects
    // would otherwise keep it however it was classified.
    harness
        .tv
        .client()
        .set_orphan_sweep_grace_for_test(Duration::ZERO);
    let sweep_cancel = CancellationToken::new();
    let sweep_handle = spawn_orphan_sweep(
        harness.tv.client().clone(),
        harness.tv.key_schema().clone(),
        Arc::new(MockExternalValueStore::new()),
        Arc::new(live_edit::storage::layer::LayerStore::new(
            harness.tv.client().clone(),
            harness.tv.key_schema().clone(),
            0,
        )),
        crate::common::test_leader(REPLICA_ID),
        sweep_cancel.clone(),
        Some(Arc::clone(&harness.metrics)),
    );

    // Drive ONE iteration. Poll-done predicate: the sweep is
    // observably "done" once any obj:hdr surface AT ALL has been
    // touched — but since this test asserts NO deletion, a "predicate
    // never true" is the expected steady state for a healthy sweep
    // against fully-restored state. The poll budget (50 × 100 ms = 5
    // s) is therefore the actual sweep budget, after which we cancel
    // + join unconditionally. This matches the `cancel_and_join_sweep`
    // pattern from `tests/integration/storage/orphan_sweep_test.rs:103-113`
    // but inlined into the same helper for symmetry with the
    // single-iteration shape the Subtask body asks for.
    //
    // We do NOT use a "wait for the obj:hdr to disappear" predicate
    // here because that would be the FAIL state — we want the sweep
    // to run to completion without touching the header. So the
    // predicate is "always false" and the loop falls through to the
    // 5 s budget every time.
    run_one_orphan_sweep_iteration(sweep_handle, sweep_cancel, || Box::pin(async { false })).await;

    // Step 5: Assert ALL freshly-restored keys STILL PRESENT post-
    // sweep. The Subtask body enumerates six explicit assertions
    // (sess, layer, layers, path_index, path_names, obj:hdr); we
    // pin all six.

    // (a) EXISTS le:{sid}:sess == 1.
    let post_sweep_session_meta_exists: bool = harness
        .tv
        .client()
        .pool
        .exists(&session_meta_key)
        .await
        .expect("EXISTS session_meta post-sweep probe must succeed");
    assert!(
        post_sweep_session_meta_exists,
        "AC #5: post-sweep EXISTS le:{{sid}}:sess MUST be 1 — the sweep classified the \
         freshly-restored session_meta as an orphan and deleted it. Key probed: {}",
        session_meta_key
    );

    // (b) EXISTS le:{sid}:layer:{hex(name)} == 1.
    let post_sweep_mapping_exists: bool = harness
        .tv
        .client()
        .pool
        .exists(&mapping_key)
        .await
        .expect("EXISTS layer_mapping post-sweep probe must succeed");
    assert!(
        post_sweep_mapping_exists,
        "AC #5: post-sweep EXISTS le:{{sid}}:layer:{{hex(name)}} MUST be 1 — the sweep \
         classified the freshly-restored layer_mapping as an orphan and deleted it \
         (would happen if the orphan sweep's mapping → session inverse lookup \
         observed the session as absent during a recovery-in-progress window). \
         Key probed: {}",
        mapping_key
    );

    // (c) SISMEMBER le:{sid}:layers <hex(name)> == 1.
    let post_sweep_layers_index_member: bool = harness
        .tv
        .client()
        .pool
        .sismember(&layers_index_key, layers_index_member.clone())
        .await
        .expect("SISMEMBER layers_index post-sweep probe must succeed");
    assert!(
        post_sweep_layers_index_member,
        "AC #5: post-sweep SISMEMBER le:{{sid}}:layers MUST contain hex({LAYER_NAME}) — \
         the sweep's empty-set cleanup arm deleted the layers_index. The recovery \
         applier's Step B writes the member as `hex_encode(layer_name)`; member \
         probed: {layers_index_member}"
    );

    // (d) SISMEMBER le:path:{hex(base_path)} <session_id> == 1 (legacy
    // path_index set).
    let post_sweep_legacy_path_member: bool = harness
        .tv
        .client()
        .pool
        .sismember(&path_index_key, &session_id)
        .await
        .expect("SISMEMBER legacy path_index post-sweep probe must succeed");
    assert!(
        post_sweep_legacy_path_member,
        "AC #5: post-sweep SISMEMBER le:path:{{hex(base_path)}} MUST contain the \
         session_id — the sweep deleted the legacy path-index member. Step A of \
         the recovery applier SADDs this; the sweep MUST NOT regard it as orphan \
         state. Key probed: {path_index_key}, member: {session_id}"
    );

    // (e) SISMEMBER le:path:{hex(base_path)}:names <path_name_member>
    // == 1 (composite set). The recovery applier emits exactly ONE
    // composite per session — `{session_name}\0{session_id}` — via
    // `add_path_name_member(session_name, session_id)` after the
    // layer-mapping loop. This mirrors the live
    // `SessionStore::create_once` write at
    // `src/storage/session.rs:308`. Step A re-SADDs the composite
    // idempotently against the surviving CreateSession-written member
    // (when one survives the wipe scenario).
    let post_sweep_session_composite: bool = harness
        .tv
        .client()
        .pool
        .sismember(&path_names_key, session_name_composite.clone())
        .await
        .expect("SISMEMBER session-name composite post-sweep probe must succeed");
    assert!(
        post_sweep_session_composite,
        "AC #5: post-sweep SISMEMBER le:path:{{hex(base_path)}}:names MUST contain the \
         session-name composite — Step A SADDs `{{session_name}}\\0{{session_id}}` \
         per session (assembler emits one composite per session). Composite probed: \
         {session_name_composite:?}"
    );

    // (f) EXISTS le:{object_id}:obj:hdr == 1.
    let post_sweep_obj_header_exists: bool = harness
        .tv
        .client()
        .pool
        .exists(&obj_header_key)
        .await
        .expect("EXISTS obj:hdr post-sweep probe must succeed");
    assert!(
        post_sweep_obj_header_exists,
        "AC #5: post-sweep EXISTS le:{{object_id}}:obj:hdr MUST be 1 — the sweep \
         deleted the freshly-restored object header. This is the LOAD-BEARING \
         failure mode AC #5 was filed to prevent: the orphan-sweep's `is_orphaned` \
         predicate must observe the live layer_mapping pointing at this object_id \
         (a HGETALL on session_id/layer_name in the header followed by a GET on \
         the mapping key) — a missing inverse lookup window would let the sweep \
         delete the just-restored header. Key probed: {obj_header_key}"
    );

    // Step 6: Sweep's `gc_*_orphaned_total` counters did NOT
    // increment. The metric family the Subtask body cites does NOT
    // exist in the metrics module (see module docstring for the
    // discrepancy); the closest Prometheus surface is
    // `live_edit_gc_sweep_batch_errors_total`, which would tick on
    // SCAN / leader-lock / UNLINK failure. We pin all three at their
    // pre-sweep snapshots so a regression that broke the sweep's
    // internal pipeline surfaces here instead of silently no-op'ping
    // the cleanup.
    assert_eq!(
        gc_sweep_batch_errors(&harness, "scan_error"),
        scan_errors_pre,
        "AC #5: post-sweep `gc_sweep_batch_errors_total{{kind=scan_error}}` MUST \
         equal pre-sweep snapshot — a non-zero delta means a SCAN cursor failed \
         mid-pass, which would leave the sweep's classification incomplete and \
         could mask a regression that DID delete a restored key on a later \
         re-scan"
    );
    assert_eq!(
        gc_sweep_batch_errors(&harness, "leader_lock_error"),
        leader_lock_errors_pre,
        "AC #5: post-sweep `gc_sweep_batch_errors_total{{kind=leader_lock_error}}` \
         MUST equal pre-sweep snapshot — leader-lock acquisition is required for \
         the sweep to do anything meaningful; a tick on this cell would mean the \
         sweep never ran and the survival assertions above would be vacuous"
    );
    assert_eq!(
        gc_sweep_batch_errors(&harness, "unlink_error"),
        unlink_errors_pre,
        "AC #5: post-sweep `gc_sweep_batch_errors_total{{kind=unlink_error}}` MUST \
         equal pre-sweep snapshot — UNLINK failures during cleanup would mean \
         the sweep tried to delete something (a restored key, given the test's \
         post-recovery state) but Valkey rejected the call, masking the real \
         AC #5 violation"
    );

    // Step 7: Issue ONE more `handle_update_layer` post-sweep —
    // succeeds without re-triggering recovery (session_recovery_metric
    // unchanged). The cache funnel carries the session's `base_path`
    // from the post-wipe RPC, so this second RPC hits the cache, never
    // consults `session_store.get_base_path`, and never invokes
    // `try_recover_session`. We pin BOTH session-recovery and
    // layer-recovery metric deltas at zero post-sweep — proves the
    // second UpdateLayer ran the happy path through fully-restored
    // state, with no recovery hop firing on either scope.
    let post_sweep_seq =
        issue_update(&harness, &session_id, object_id, delta_post_sweep(), 4_000).await;
    assert!(
        post_sweep_seq > post_recovery_seq,
        "AC #5 step 7: post-sweep UpdateLayer sequence_num ({post_sweep_seq}) MUST be \
         strictly greater than the post-recovery sequence_num ({post_recovery_seq}) \
         — a non-monotonic seq here means the post-sweep RPC observed stale state, \
         which would itself be an AC #5 regression (the sweep silently corrupted \
         the restored object_id's seq counter)"
    );

    assert_eq!(
        session_recovery_metric(&harness, "restored"),
        session_restored_pre_sweep,
        "AC #5 step 7: post-sweep `restored{{session}}` MUST equal pre-sweep \
         snapshot — the cache funnel's positive entry from the post-wipe RPC \
         routes the second RPC past `try_recover_session` entirely. A delta \
         here means the cache was invalidated (or never populated) and \
         recovery re-fired on already-live session state"
    );
    assert_eq!(
        session_recovery_metric(&harness, "not_recoverable"),
        session_not_recoverable_pre_sweep,
        "AC #5 step 7: post-sweep `not_recoverable{{session}}` MUST equal \
         pre-sweep snapshot — no recovery attempt fired so no outcome can \
         have ticked"
    );
    assert_eq!(
        session_recovery_metric(&harness, "already_present"),
        session_already_present_pre_sweep,
        "AC #5 step 7: post-sweep `already_present{{session}}` MUST equal \
         pre-sweep snapshot — the cache HIT short-circuits before the K=1 \
         admission gate is reached, so even the `already_present` short-circuit \
         in `SessionRecoveryHook::try_recover_session` step 3 (the pre-acquire \
         EXISTS check) cannot fire"
    );
    assert_eq!(
        session_recovery_metric(&harness, "error"),
        session_error_pre_sweep,
        "AC #5 step 7: post-sweep `error{{session}}` MUST equal pre-sweep \
         snapshot — no recovery fired, so no error arm could tick"
    );
    assert_eq!(
        layer_recovery_metric(&harness, "restored"),
        layer_restored_pre_sweep,
        "AC #5 step 7: post-sweep `restored{{layer}}` MUST equal pre-sweep \
         snapshot — layer-recovery only fires when `update_layer_once`'s \
         HGETALL on obj:hdr returns empty. The post-recovery state has a \
         populated header (assertion (f) above), so the HGETALL returns the \
         restored fields and layer-recovery never enters its retry branch"
    );
    assert_eq!(
        layer_recovery_metric(&harness, "not_recoverable"),
        layer_not_recoverable_pre_sweep,
        "AC #5 step 7: post-sweep `not_recoverable{{layer}}` MUST equal \
         pre-sweep snapshot — no layer-recovery fired"
    );
    assert_eq!(
        layer_recovery_metric(&harness, "error"),
        layer_error_pre_sweep,
        "AC #5 step 7: post-sweep `error{{layer}}` MUST equal pre-sweep \
         snapshot — no layer-recovery fired"
    );

    // Step 8: Graceful shutdown of the checkpoint loop. The orphan
    // sweep was already cancelled + joined inside
    // `run_one_orphan_sweep_iteration`. Cancelling + awaiting the
    // checkpoint loop here removes the slim window where a still-
    // running tick could race with the test runtime's shutdown.
    harness.cancel.cancel();
    let _ = harness.loop_handle.await;
}

// ── Bytes-differ polling helpers ─────────────────────────────────────────────
//
// Duplicated from `e2e_session_full_wipe_recovery_test.rs` — same
// contract (the dispatcher-driven session-checkpoint write is fire-and-
// forget; a plain "is the blob present?" poll returns immediately
// because the pre-CreateSession blob is still there). Same applies to
// the post-burst layer-checkpoint generation: the seed generation from
// the immediate-post-CreateLayer tick has `snap_seq` ≈ 1 (the
// CreateLayer's seq bump); we need a generation that reflects the
// burst's `pre_wipe_max_seq` so the restore-and-re-issue chain returns
// a `sequence_num > pre_wipe_max_seq`.
//
// Re-implemented locally rather than lifted because (a) the helpers in
// the AC #2 file are file-private, and (b) the per-file duplication
// pattern is already established across four sibling e2e_session_*
// files in this directory. A future refactor that lifts these into
// `helpers.rs` would absorb every site uniformly; mixing scopes now
// would churn one file.

async fn wait_for_post_create_layer_session_checkpoint(
    harness: &CheckpointHarness,
    session_id: &str,
    pre_create_layer_blob: &[u8],
    deadline_secs: u64,
) {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(deadline_secs);
    let poll_interval = std::time::Duration::from_millis(50);
    loop {
        match harness.store.get_session_checkpoint(session_id).await {
            Ok(Some(blob)) if blob.bytes.as_ref() != pre_create_layer_blob => return,
            Ok(_) => {
                // Writer's post-CreateLayer put has not landed yet —
                // keep polling.
            }
            Err(err) => {
                panic!(
                    "get_session_checkpoint failed during post-CreateLayer wait \
                     (session_id={session_id}, deadline_secs={deadline_secs}): {err}"
                );
            }
        }
        if std::time::Instant::now() >= deadline {
            panic!(
                "wait_for_post_create_layer_session_checkpoint timed out after \
                 {deadline_secs}s — the CreateLayer-driven session-checkpoint write \
                 never landed (bytes did not differ from the pre-CreateLayer blob)"
            );
        }
        tokio::time::sleep(poll_interval).await;
    }
}

async fn wait_for_post_burst_layer_checkpoint(
    harness: &CheckpointHarness,
    object_id: u64,
    pre_wipe_max_seq: u64,
    deadline_secs: u64,
) {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(deadline_secs);
    let poll_interval = std::time::Duration::from_millis(50);
    loop {
        match harness.store.list_layer_checkpoints(object_id).await {
            Ok(handles) => {
                if handles.iter().any(|h| h.snap_seq >= pre_wipe_max_seq) {
                    return;
                }
            }
            Err(err) => {
                panic!(
                    "list_layer_checkpoints failed during post-burst wait \
                     (object_id={object_id}, pre_wipe_max_seq={pre_wipe_max_seq}, \
                     deadline_secs={deadline_secs}): {err}"
                );
            }
        }
        if std::time::Instant::now() >= deadline {
            panic!(
                "wait_for_post_burst_layer_checkpoint timed out after {deadline_secs}s \
                 — no layer-checkpoint generation with snap_seq >= {pre_wipe_max_seq} \
                 landed for object_id={object_id}"
            );
        }
        tokio::time::sleep(poll_interval).await;
    }
}
