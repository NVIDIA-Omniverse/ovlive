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

// Integration tests for LayerStore::update_layer transaction behavior:
// happy path (seq increments), concurrent writes (one `Aborted`), stream entry
// format, empty update (still increments seq + appends stream entry),
// header-gone (FAILED_PRECONDITION).
//
// Tests run against a real Valkey instance via the TestValkey fixture. Each test
// gets an isolated key namespace so tests can run in parallel safely.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use bytes::Bytes;
use fred::interfaces::{
    HashesInterface, KeysInterface, ListInterface, SetsInterface, StreamsInterface,
};

use live_edit::config::UpdateLayerBackend;
use live_edit::domain::types::{FieldValue, ObjectState};
use live_edit::error::ServiceError;
use live_edit::observability::metrics::test_util::{render, sample_value};
use live_edit::observability::metrics::Metrics;
use live_edit::storage::external_values::MockExternalValueStore;
use live_edit::storage::layer::{
    LayerStore, DEFAULT_UPDATE_LAYER_PER_OBJECT_CONCURRENCY,
    DEFAULT_UPDATE_LAYER_RETRY_MAX_ATTEMPTS,
};
use live_edit::storage::session::SessionStore;
use live_edit::storage::test_support::read_object_state_for_test;
use live_edit::storage::test_support::relay::{relay_ompf, RelayedUpdates};
use live_edit::storage::test_support::TestValkey;
use oodf_ffi::{
    CommandCreateSection, CommandDeleteSection, CommandDiffChangeSection, CommandDiffDeleteSection,
    OwnedDelta, SetField, StructCommand, TimeSample as OodfTimeSample,
};

use crate::common::owned_delta::{delta_with_set_field_inline, reset_delta};
/// Mirror of the module-private `storage::layer::EMPTY_HEADER_SENTINEL`; shared
/// via `tests/common` so every module asserting it holds one literal.
use crate::common::EMPTY_HEADER_SENTINEL_MSG;

fn mock_ext_store() -> Arc<MockExternalValueStore> {
    Arc::new(MockExternalValueStore::new())
}

/// Helper: build a LayerStore from a TestValkey fixture.
///
/// **b.ohv item 3:** the historical default here delegated to `LayerStore::new`,
/// which uses the production default (`UpdateLayerBackend::Lua`). The
/// concurrent-writer assertions in this file (`update_layer_concurrent_writes_*`,
/// `update_layer_*_writers_distinct_seqs_and_xlen_parity`,
/// `update_layer_concurrent_writes_on_two_layers_no_cross_layer_interference`)
/// are pinned to **Txn** because they assert the WATCH-cancel `Aborted` literal
/// — atomic Lua never produces an abort, so a Lua-default run made the abort
/// branches structurally unreachable and the spec-quoted literal assertions
/// silently vacuous. The base `layer_store` helper now pins Txn; the Lua-pinned
/// `layer_store_lua` helper exists for the sibling tests that exercise the
/// other backend.
fn layer_store(tv: &TestValkey) -> LayerStore {
    LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096)
        .with_update_layer_backend(UpdateLayerBackend::Txn)
}

/// Lua-backend variant — used by the sibling test that asserts both writes
/// commit cleanly under atomic Lua (no aborts ever).
fn layer_store_lua(tv: &TestValkey) -> LayerStore {
    LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096)
        .with_update_layer_backend(UpdateLayerBackend::Lua)
}

/// Txn-backend variant pinned to a one-attempt retry budget — used only by the
/// `update_layer_concurrent_writes_one_aborted_one_succeeds` test (b.ohv AC6).
/// That test asserts the WATCH-conflict surface of `update_layer_once_watch`
/// directly, so the outer retry helper (default 128
/// attempts, 50 ms cap) MUST be capped at 1 attempt or else the WATCH-loser
/// silently retries past the winner and the strict `aborts == 1` assertion
/// fails as `successes == 2, aborts == 0`. Other sibling tests in this file
/// genuinely want the post-retry composite surface and continue to use
/// `layer_store(&tv)`.
fn layer_store_with_retries_one(tv: &TestValkey) -> LayerStore {
    LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096)
        .with_update_layer_backend(UpdateLayerBackend::Txn)
        .with_update_layer_retry_max_attempts(1)
}

/// Helper: create a session and return its ID.
async fn create_test_session(tv: &TestValkey) -> String {
    let ss = SessionStore::new(tv.client().clone(), tv.key_schema().clone());
    ss.create("test-session", "/stage", "test-user")
        .await
        .expect("session creation must succeed")
}

/// Helper: create a layer via LayerStore and return its object_id.
async fn create_test_layer(ls: &LayerStore, session_id: &str, layer_name: &str) -> u64 {
    let state = ObjectState {
        sequence_num: 1,
        first_version: 1,
        source_format: "usd".to_string(),
        ..Default::default()
    };
    ls.create_layer(
        session_id,
        layer_name,
        "test-user",
        &state,
        mock_ext_store(),
    )
    .await
    .expect("create_layer must succeed")
}

/// Build a valid empty OwnedDelta suitable for passing to `update_layer`.
fn build_valid_delta() -> OwnedDelta {
    OwnedDelta::default()
}

// =============================================================================
// 1. Happy path -- seq increments, server_timestamp > 0
// =============================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_happy_path_seq_increments_and_timestamp_positive() {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let session_id = create_test_session(&tv).await;
    let oid = create_test_layer(&ls, &session_id, "happy-layer").await;

    let delta = build_valid_delta();

    let (seq, server_ts) = ls
        .update_layer_relayed(
            &session_id,
            "happy-layer",
            oid,
            Some(delta),
            "client-1",
            1000,
        )
        .await
        .expect("update_layer must succeed");

    assert!(
        seq > 0,
        "sequence_num must be positive after update, got {seq}"
    );
    assert!(
        server_ts > 0,
        "server_timestamp must be positive, got {server_ts}"
    );
}

// =============================================================================
// 2. Concurrent writes -- exactly one `Aborted`, one succeeds
// =============================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_concurrent_writes_one_aborted_one_succeeds() {
    // b.ohv AC6: this test was previously in the "known-flaky" form where the
    // two writers could complete sequentially (no WATCH window overlap) and
    // both commit cleanly, weakening the assertion to "at least one succeeds".
    // We now use a `tokio::sync::Barrier::new(2)` to force both writer tasks
    // past `barrier.wait()` simultaneously, opening the WATCH overlap window
    // deterministically. Combined with a pool large enough to give each task
    // its own exclusive txn connection, this lets us tighten the assertion to
    // exactly one Ok + one Aborted — the spec-quoted outcome of a Txn-backend
    // WATCH conflict (SDD §18.2). Without the barrier or without the per-task
    // connection, the test would either pass vacuously (no race) or fail
    // intermittently (pool serialisation hides the race).
    //
    // Retry-budget pinning (DO NOT REMOVE): `LayerStore::update_layer` wraps
    // its attempts in `retry_aborted_with_jitter_and_uncounted_waits` with a
    // default of 128 attempts and a 50 ms cap. Under the barrier-forced
    // overlap the WATCH-loser would receive `Aborted` from
    // `update_layer_once_watch` (no fence is live, so its conflict re-probe
    // leaves the conflict counted), the outer helper would sleep ≤ 50 ms, and
    // by then the winner has finished — so the loser commits cleanly on
    // retry and the outer call returns `Ok` for BOTH writers, silently
    // turning the strict `aborts == 1` assertion into `successes == 2`.
    // We pin retries to 1 via `layer_store_with_retries_one` so the test
    // asserts the `update_layer_once_watch` WATCH-cancel surface directly,
    // which is what AC6 requires.
    const POOL_SIZE: u32 = 4;
    let tv = TestValkey::setup_with_pool_size(POOL_SIZE).await;
    let ls = layer_store_with_retries_one(&tv);
    let session_id = create_test_session(&tv).await;
    let oid = create_test_layer(&ls, &session_id, "concurrent-layer").await;

    let delta = build_valid_delta();

    let barrier = std::sync::Arc::new(tokio::sync::Barrier::new(2));

    let ls1 = ls.clone();
    let sid1 = session_id.clone();
    let d1 = delta.clone();
    let b1 = barrier.clone();
    let h1 = tokio::spawn(async move {
        b1.wait().await;
        ls1.update_layer_relayed(&sid1, "concurrent-layer", oid, Some(d1), "client-a", 1000)
            .await
    });

    let ls2 = ls.clone();
    let sid2 = session_id.clone();
    let d2 = delta.clone();
    let b2 = barrier.clone();
    let h2 = tokio::spawn(async move {
        b2.wait().await;
        ls2.update_layer_relayed(&sid2, "concurrent-layer", oid, Some(d2), "client-b", 2000)
            .await
    });

    let r1 = h1.await.expect("task 1 must not panic");
    let r2 = h2.await.expect("task 2 must not panic");

    let mut successes = 0u32;
    let mut aborts = 0u32;

    for result in [&r1, &r2] {
        match result {
            Ok(_) => successes += 1,
            // A genuine WATCH conflict under live contention must surface
            // the spec-quoted nil-EXEC literal "concurrent write conflict,
            // retry" (SDD §18.2), NOT the canceled-arm literal
            // "transaction canceled (connection drop), retry" from the
            // Err(is_canceled) path. A racing writer is a WATCH-cancel —
            // never a connection drop — so the two arms must not collide
            // in observability. With `with_update_layer_retry_max_attempts(1)`
            // the inner WATCH-conflict surface is wrapped by the outer
            // `retry_aborted_with_jitter_and_uncounted_waits` loop's
            // exhaustion message (`update_layer retriable failures, 1 attempts exhausted
            // (last cause: ...)`); assert that the inner cause is exactly
            // the nil-EXEC literal via substring, and explicitly forbid the
            // canceled-arm literal so the two arms can't drift into each
            // other.
            Err(ServiceError::Aborted(msg)) => {
                assert!(
                    msg.contains("concurrent write conflict, retry"),
                    "WATCH-race Aborted must carry the spec-quoted nil-EXEC literal \
                     as its inner cause; got: {msg:?}"
                );
                assert!(
                    !msg.contains("transaction canceled (connection drop), retry"),
                    "WATCH-race Aborted must NOT carry the canceled-arm literal — \
                     a racing writer is never a connection drop; got: {msg:?}"
                );
                aborts += 1;
            }
            Err(e) => panic!("unexpected error from concurrent update_layer: {e}"),
        }
    }

    // With the barrier forcing WATCH-window overlap and pool_size >= 2
    // giving each task its own exclusive txn connection, the Txn backend
    // MUST produce exactly one commit and exactly one WATCH-conflict
    // abort. Anything else indicates a regression — silent-drop bug (two
    // Oks), connection-drop misclassification, or pool serialisation
    // hiding the race.
    assert_eq!(
        successes, 1,
        "Txn-backed barrier-synchronised concurrent writers must produce exactly one success; \
         got successes={successes} aborts={aborts}"
    );
    assert_eq!(
        aborts, 1,
        "Txn-backed barrier-synchronised concurrent writers must produce exactly one abort; \
         got successes={successes} aborts={aborts}"
    );
}

// =============================================================================
// 2b. Lua sibling — under the atomic Lua backend the same concurrent pair must
//     BOTH commit cleanly. There is no WATCH window to cancel: each invocation
//     runs the `update_layer_fast.lua` script inside Valkey's single-threaded
//     script slot, so the two scripts are simply serialised by the server and
//     both return Ok with consecutive sequence numbers. Aborts here would be a
//     regression in the Lua backend's atomicity contract (b.ohv item 3).
// =============================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_concurrent_writes_under_lua_both_commit_cleanly() {
    // pool_size >= 2 so each task can hold its own connection — otherwise the
    // shared pool serializes naturally and the contention window never opens.
    let tv = TestValkey::setup_with_pool_size(4).await;
    let ls = layer_store_lua(&tv);
    let session_id = create_test_session(&tv).await;
    let oid = create_test_layer(&ls, &session_id, "lua-concurrent-layer").await;

    let delta = build_valid_delta();

    let ls1 = ls.clone();
    let sid1 = session_id.clone();
    let d1 = delta.clone();
    let h1 = tokio::spawn(async move {
        ls1.update_layer_relayed(
            &sid1,
            "lua-concurrent-layer",
            oid,
            Some(d1),
            "lua-client-a",
            1000,
        )
        .await
    });

    let ls2 = ls.clone();
    let sid2 = session_id.clone();
    let d2 = delta.clone();
    let h2 = tokio::spawn(async move {
        ls2.update_layer_relayed(
            &sid2,
            "lua-concurrent-layer",
            oid,
            Some(d2),
            "lua-client-b",
            2000,
        )
        .await
    });

    let r1 = h1.await.expect("task 1 must not panic");
    let r2 = h2.await.expect("task 2 must not panic");

    // Both must commit. Under Lua, the script slot in Valkey serializes the
    // two scripts; each one runs to completion before the other starts —
    // there is no WATCH window, so the abort path is structurally impossible.
    // A regression that re-introduced abort behavior in the Lua dispatch (e.g.
    // a future hybrid path that mistakenly falls back to txn on contention)
    // would surface here as one Err(Aborted), failing the test.
    let (seq1, _ts1) = r1.expect(
        "Lua backend: both concurrent updates must commit; first task got an unexpected error",
    );
    let (seq2, _ts2) = r2.expect(
        "Lua backend: both concurrent updates must commit; second task got an unexpected error",
    );

    // Sequence numbers must be distinct (no silent dropped writes) and
    // strictly above the CreateLayer initial 1 — so two writers land at
    // {2, 3} in some order.
    assert_ne!(
        seq1, seq2,
        "Lua backend: two committed writers must produce distinct sequence_nums; got {seq1} and {seq2}"
    );
    let mut seqs = [seq1, seq2];
    seqs.sort();
    assert_eq!(
        seqs,
        [2, 3],
        "Lua backend: two writers on top of CreateLayer(seq=1) must yield {{2, 3}}; got {seqs:?}"
    );
}

// =============================================================================
// 3. Stream entry format -- ID `{seq}-0`, a pipe-delimited `payload` field and
//    the relay `ompf` field
// =============================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_stream_entry_format_pipe_delimited_with_all_fields() {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let keys = tv.key_schema();
    let session_id = create_test_session(&tv).await;
    let oid = create_test_layer(&ls, &session_id, "notify-layer").await;

    let delta = build_valid_delta();
    let relay = relay_ompf(&delta);
    let client_id = "my-client-42";
    let client_ts: u64 = 9999;

    let (seq, server_ts) = ls
        .update_layer(
            &session_id,
            "notify-layer",
            oid,
            Some(delta),
            Some(relay.clone()),
            client_id,
            client_ts,
        )
        .await
        .expect("update_layer must succeed");

    // XADD happens inside the UpdateLayer MULTI/EXEC, so by the time
    // update_layer returns, the entry is visible via XRANGE. No subscription
    // race to manage.
    let stream_key = keys.layer_stream(oid);
    let entries: Vec<(String, HashMap<String, Bytes>)> = tv
        .client()
        .pool
        .xrange_values(&stream_key, "-", "+", None)
        .await
        .expect("XRANGE must succeed");

    // create_layer wrote `1-0`; update_layer just wrote `{seq}-0`. Expect both.
    assert_eq!(
        entries.len(),
        2,
        "stream must contain the create (1-0) and update ({seq}-0) entries, got {}: {entries:?}",
        entries.len()
    );

    // Second entry is the UpdateLayer XADD. Assert ID and payload shape.
    let (entry_id, fields) = &entries[1];
    assert_eq!(
        entry_id,
        &format!("{seq}-0"),
        "UpdateLayer XADD must use deterministic ID {{seq}}-0"
    );
    assert_eq!(
        fields.keys().map(String::as_str).collect::<HashSet<_>>(),
        HashSet::from(["payload", "ompf"]),
        "stream entry must carry exactly the `payload` and relay `ompf` fields, got {fields:?}"
    );
    assert_eq!(
        fields.get("ompf"),
        Some(&relay),
        "the `ompf` field is the call's relay bytes, verbatim"
    );
    let payload = std::str::from_utf8(
        fields
            .get("payload")
            .expect("stream entry must have a `payload` field"),
    )
    .expect("the payload is text");

    // Payload format: update|{oid}|{seq}|{client_id}|{client_ts}|{server_ts}
    let parts: Vec<&str> = payload.split('|').collect();
    assert_eq!(
        parts.len(),
        6,
        "payload must have 6 pipe-delimited fields, got {}: {payload}",
        parts.len()
    );
    assert_eq!(parts[0], "update", "first field must be 'update'");
    assert_eq!(parts[1], oid.to_string(), "second field must be object_id");
    assert_eq!(
        parts[2],
        seq.to_string(),
        "third field must be sequence_num"
    );
    assert_eq!(parts[3], client_id, "fourth field must be client_id");
    assert_eq!(
        parts[4],
        client_ts.to_string(),
        "fifth field must be client_timestamp"
    );
    assert_eq!(
        parts[5],
        server_ts.to_string(),
        "sixth field must be server_timestamp"
    );
}

// =============================================================================
// 4. Empty update -- still increments seq and appends a stream entry
// =============================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_empty_delta_increments_seq_and_appends_stream_entry() {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let keys = tv.key_schema();
    let session_id = create_test_session(&tv).await;
    let oid = create_test_layer(&ls, &session_id, "empty-update-layer").await;

    // Empty delta (None) — SDD §8.4 empty-delta semantics.
    let (seq, _server_ts) = ls
        .update_layer(
            &session_id,
            "empty-update-layer",
            oid,
            None,
            None,
            "empty-client",
            0,
        )
        .await
        .expect("empty update must succeed");

    // create_layer wrote seq=1 in the header. apply_delta increments seq, so
    // after one update we expect seq=2.
    assert_eq!(
        seq, 2,
        "empty update must increment sequence_num from 1 to 2, got {seq}"
    );

    // Verify the XADD fired inside the MULTI/EXEC: stream contains the
    // create entry `1-0` plus the update entry `2-0`.
    let stream_key = keys.layer_stream(oid);
    let entries: Vec<(String, HashMap<String, String>)> = tv
        .client()
        .pool
        .xrange_values(&stream_key, "-", "+", None)
        .await
        .expect("XRANGE must succeed");

    assert_eq!(
        entries.len(),
        2,
        "stream must contain the create (1-0) and update (2-0) entries, got {entries:?}"
    );

    let (entry_id, fields) = &entries[1];
    assert_eq!(
        entry_id, "2-0",
        "empty-delta UpdateLayer XADD must use deterministic ID 2-0"
    );
    let payload = fields
        .get("payload")
        .expect("stream entry must have a `payload` field");
    assert!(
        payload.starts_with("update|"),
        "empty update payload must start with 'update|', got: {payload}"
    );
}

// =============================================================================
// 5. Header gone → NOT_FOUND
// =============================================================================

/// The slow path's empty-`HGETALL` guard: an object whose `obj:hdr` was deleted
/// out from under a commit is ABSENT, and b.f1v made both backends say so with
/// `NotFound` rather than `FailedPrecondition`.
///
/// The variant is the contract — `grpc::layer::handle_update_layer` keys its
/// single checkpoint-recovery arm on it, so a `FailedPrecondition` here means a
/// post-wipe UpdateLayer never attempts recovery.
///
/// One corner of a four-test claim — **an ABSENT header and a PARTIAL one are
/// distinct surfaces, identically on both backends**. The other three:
/// `update_layer_lua_test::update_layer_lua_missing_header_returns_not_found`
/// (absent / Lua),
/// `b_k1r_tests::update_layer_lua_partial_header_returns_failed_precondition`
/// (partial / Lua) and
/// `b_qw6_tests::update_layer_txn_partial_header_returns_failed_precondition`
/// (partial / txn). The `matches!` below is the opposite-variant fence for this
/// corner: a `FailedPrecondition` would mean the absent case had collapsed back
/// into the partial one.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_header_deleted_returns_not_found() {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let keys = tv.key_schema();
    let session_id = create_test_session(&tv).await;
    let oid = create_test_layer(&ls, &session_id, "header-gone-layer").await;

    // Manually delete the object header key from Valkey.
    let header_key = keys.obj_header(&oid.to_string());
    let _: () = tv
        .client()
        .pool
        .del(&header_key)
        .await
        .expect("DEL header must succeed");

    let delta = build_valid_delta();

    let result = ls
        .update_layer_relayed(
            &session_id,
            "header-gone-layer",
            oid,
            Some(delta),
            "client-x",
            5000,
        )
        .await;

    let err = result.expect_err("update with deleted header must fail");
    assert!(
        matches!(err, ServiceError::NotFound(_)),
        "b.f1v: a deleted `obj:hdr` must surface as NotFound — `FailedPrecondition` here \
         disarms the gRPC handler's checkpoint-recovery arm, which matches on the variant. \
         Got: {err:?}"
    );
    let ServiceError::NotFound(msg) = err else {
        unreachable!("matches! above already established the variant")
    };
    assert_eq!(
        msg, EMPTY_HEADER_SENTINEL_MSG,
        "the slow path's empty-HGETALL guard must carry the empty-header sentinel \
         byte-for-byte — the Lua path raises the same string and the SDD §7.5 parity \
         contract is asserted on the message text"
    );
}

// =============================================================================
// 6. MAXLEN trimming — XADD on the per-object layer stream honors the
//    configured approximate MAXLEN so streams don't grow unbounded under load.
//    This is an Epic 2 acceptance criterion: the service must actively trim
//    the stream on every write, not just rely on eviction or operator action.
// =============================================================================

/// Build a LayerStore with a custom `layer_stream_maxlen`.
fn layer_store_with_maxlen(tv: &TestValkey, maxlen: u64) -> LayerStore {
    LayerStore::new(tv.client().clone(), tv.key_schema().clone(), maxlen)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_xadd_maxlen_trims_stream_to_configured_threshold() {
    // Valkey XADD `MAXLEN ~ N` is *approximate*: the server only evicts whole
    // internal stream nodes (default ~128 entries) when doing so keeps XLEN
    // above the threshold. That means very small MAXLEN values never trim
    // within a reasonable test burst — we need MAXLEN large enough that the
    // eviction threshold is crossed. With MAXLEN=100 and 300 writes, trimming
    // converges deterministically to ~MAXLEN (see manual verification).
    // FUDGE_MULTIPLIER = 2 tolerates small node-boundary overshoot while
    // still catching a regression to "no trimming at all" (which would yield
    // XLEN == UPDATES).
    const MAXLEN: u64 = 100;
    const UPDATES: u64 = 300;
    const FUDGE_MULTIPLIER: u64 = 2;

    let tv = TestValkey::setup().await;
    let ls = layer_store_with_maxlen(&tv, MAXLEN);
    let keys = tv.key_schema();
    let session_id = create_test_session(&tv).await;
    let oid = create_test_layer(&ls, &session_id, "maxlen-layer").await;

    // Issue UPDATES back-to-back. Each XADD carries `MAXLEN ~ MAXLEN`.
    // Once enough internal stream nodes accumulate, Valkey evicts whole
    // nodes whose tail falls below the threshold.
    for i in 0..UPDATES {
        ls.update_layer_relayed(
            &session_id,
            "maxlen-layer",
            oid,
            Some(build_valid_delta()),
            "maxlen-writer",
            1_000 + i,
        )
        .await
        .unwrap_or_else(|e| panic!("update #{i} must succeed: {e}"));
    }

    // XLEN on the per-object stream — must be bounded even though we issued
    // 60 updates. Approximate trimming converges to MAXLEN + small constant.
    let stream_key = keys.layer_stream(oid);
    let xlen: u64 = tv
        .client()
        .pool
        .xlen(&stream_key)
        .await
        .expect("XLEN must succeed");

    let upper_bound = MAXLEN * FUDGE_MULTIPLIER;
    assert!(
        xlen <= upper_bound,
        "XADD MAXLEN failed to trim stream: issued {UPDATES} updates with \
         MAXLEN={MAXLEN}, expected XLEN <= {upper_bound} (MAXLEN * {FUDGE_MULTIPLIER}x fudge), got XLEN={xlen}"
    );
    // Sanity: the stream also must not be empty — we did commit writes.
    assert!(
        xlen > 0,
        "stream must contain at least one entry after UPDATES={UPDATES}, got XLEN={xlen}"
    );
}

// =============================================================================
// 7. No-silent-drop guarantee at production concurrency: N concurrent writers
//    on the same object must all commit at distinct, consecutive sequence_nums
//    (landing N stream entries). Silent returns of `Ok(seq=K)` with no matching
//    XADD are not acceptable — that reports commit to the client while losing
//    the write (bee b.8q1).
//
//    b.459 widened this from N=5 (it was
//    `update_layer_five_concurrent_writers_never_silently_drop_writes`) to
//    N=50 and added the three-counter zero assertion. Five writers against a
//    16-permit admission gate never even filled it: the pre-`MULTI` guards
//    (b.ph9's stream-ID collision probe, b.459's keyspace type probe, b.bhb's
//    void-watch detection) all fire on evidence that a *healthy* race must
//    never produce, and at N=5 that claim was untested. At N=50 the gate
//    saturates, WATCH windows overlap continuously, and every retry re-reads a
//    header another writer has just moved — which is exactly the traffic a
//    false positive on any of the three would show up in.
// =============================================================================

/// The three counters that must read ZERO across a healthy race, checked as one
/// unit (b.ph9 / b.bhb / b.459).
///
/// Each is alertable precisely because its steady-state baseline is zero, and
/// each is reachable only from a condition ordinary contention cannot create:
///
/// * `stream_id_collision_total` — an entry sitting at or above the header's
///   sequence. Losing a race writes the entry AND advances the header, so a
///   loser sees an ordinary `Aborted`, never this.
/// * `keyspace_type_error_total` — a key on the `le:` keyspace holding a
///   foreign type. Nothing this service does retypes a key it owns.
/// * `storage_watch_voided_total{committed_unwatched}` — a commit proven to
///   have run on a connection that did not carry its `WATCH`.
///
/// A tick on any of them here would mean the guard misfires under load, which
/// on a real deployment is a hard `INTERNAL` handed to a client whose only
/// mistake was writing at the same time as someone else — and a page for an
/// operator with nothing to find.
fn assert_race_guard_counters_stayed_zero(metrics: &Metrics, scenario: &str) {
    let text = render(metrics);
    let read = |name: &str, labels: &[(&str, &str)]| -> u64 {
        sample_value(&text, name, labels).unwrap_or_else(|| {
            panic!(
                "{name}{labels:?} is absent from the exposition. Every cell here is zero-seeded \
                 in `Metrics::new`, so absence means the store never received this metrics \
                 handle and this whole assertion is vacuous"
            )
        }) as u64
    };

    assert_eq!(
        read(
            "live_edit_stream_id_collision_total",
            &[("backend", "txn"), ("site", "update")]
        ),
        0,
        "{scenario}: a healthy race must never trip the stream-ID collision guard (b.ph9). \
         Losing an optimistic-concurrency race writes the winner's entry AND advances the \
         header together, so a loser retries against a consistent pair — a tick here means \
         the guard fired on ordinary contention and turned a retriable loss into a \
         non-retriable `INTERNAL`"
    );
    assert_eq!(
        read(
            "live_edit_keyspace_type_error_total",
            &[("backend", "txn"), ("site", "update")]
        ),
        0,
        "{scenario}: a healthy race must never trip the pre-`MULTI` keyspace type probe \
         (b.459). Every key these writers touch is written only by this service, at one type. \
         A tick here means the probe is refusing commits over a keyspace it built itself — a \
         write outage on the layer, and a page for an operator who will find nothing wrong \
         with the key"
    );
    assert_eq!(
        read(
            "live_edit_storage_watch_voided_total",
            &[
                ("path", "update_layer"),
                ("disposition", "committed_unwatched")
            ]
        ),
        0,
        "{scenario}: no connection was disturbed, so no commit can have run unwatched \
         (b.bhb). This is the alertable cell whose entire value rests on a zero baseline; a \
         false positive under ordinary load would make it unpageable"
    );
}

/// Run `WRITERS` concurrent `update_layer` calls on one object and assert the
/// b.8q1 no-silent-drop invariants plus the three-counter zero.
///
/// `setup` deltas are committed SEQUENTIALLY first, before the barrier — they
/// are the scenario's fixture, not part of the race — and the expected sequence
/// run is derived from the header afterwards rather than hardcoded at 2.
///
/// Returns the fixture, the object id and the metrics handle the store reports
/// into, so a caller can add scenario-specific assertions (e.g. "every writer's
/// rename is materialized", or the retry-loop claims in
/// `update_layer_retry_test`).
///
/// # One scenario, one tolerance
///
/// This helper is THE 50-concurrent-writers-on-one-object scenario for the whole
/// suite, and its abort tolerance is **zero** — see the `aborts == 0` assertion
/// below for why that is the contract rather than a tight-but-arbitrary bound.
/// `update_layer_retry_test::update_layer_fifty_concurrent_writers_commit_by_
/// consuming_retries` (formerly `…_commit_rate_at_least_99pct`, which tolerated
/// one abort) now runs through here too, so the suite states one number in one
/// place: the two tests are two CLAIMS about one run, not two runs that could
/// disagree about how many dropped writes are acceptable.
pub(crate) async fn run_fifty_writer_race(
    layer: &str,
    setup: Vec<OwnedDelta>,
    deltas: Vec<OwnedDelta>,
    scenario: &str,
) -> (TestValkey, u64, Arc<Metrics>) {
    let writers = deltas.len();

    // Sized to the per-object admission gate, NOT to `writers`. At most
    // `DEFAULT_UPDATE_LAYER_PER_OBJECT_CONCURRENCY` writers can be inside
    // `update_layer_once` at once (b.3gx), so a pool of `writers * 2` would
    // buy nothing but idle connections — while a pool SMALLER than the gate
    // would move the serialization point from the gate to the pool mutex and
    // quietly reduce the overlap this test exists to create. The +8 is
    // headroom for the fixture's own non-txn traffic.
    let pool_size = DEFAULT_UPDATE_LAYER_PER_OBJECT_CONCURRENCY + 8;

    let tv = TestValkey::setup_with_pool_size(pool_size).await;
    let metrics = Arc::new(crate::common::metrics_scrape_only("test"));
    tv.client()
        .install_observability_hooks(metrics.clone())
        .await;

    // Default retry budget on purpose (no `with_update_layer_retry_max_attempts`):
    // the production budget is part of what is under test here. A capped budget
    // would convert contention into `Aborted` and hide whether the guards fire.
    let ls = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096)
        .with_update_layer_backend(UpdateLayerBackend::Txn)
        .with_metrics(metrics.clone());
    let keys = tv.key_schema();
    let session_id = create_test_session(&tv).await;
    let oid = create_test_layer(&ls, &session_id, layer).await;

    let setup_commits = setup.len();
    for (i, delta) in setup.into_iter().enumerate() {
        ls.update_layer_relayed(&session_id, layer, oid, Some(delta), "setup", 1 + i as u64)
            .await
            .unwrap_or_else(|e| panic!("{scenario}: setup commit #{i} must land: {e:?}"));
    }
    // CreateLayer took 1 and each setup commit took one after it.
    let base_seq = 1 + setup_commits as u64;

    // Barrier so every writer reaches WATCH at roughly the same instant.
    // Without it the first writers finish their whole WATCH/HGETALL/MULTI/EXEC
    // before the later ones are scheduled — no overlap, and the test passes
    // vacuously against the very regressions it is written for.
    let barrier = Arc::new(tokio::sync::Barrier::new(writers));

    let mut handles = Vec::with_capacity(writers);
    for (i, delta) in deltas.into_iter().enumerate() {
        let ls_i = ls.clone();
        let sid_i = session_id.clone();
        let barrier_i = barrier.clone();
        let layer_i = layer.to_owned();
        let client_id = format!("writer-{i}");
        let client_ts = 1000u64 + i as u64;
        handles.push(tokio::spawn(async move {
            barrier_i.wait().await;
            ls_i.update_layer_relayed(&sid_i, &layer_i, oid, Some(delta), &client_id, client_ts)
                .await
        }));
    }

    let started = std::time::Instant::now();
    let mut successes: Vec<u64> = Vec::new();
    let mut aborts: u32 = 0;
    let mut other_errors: Vec<String> = Vec::new();
    for h in handles {
        match h.await.expect("writer task must not panic") {
            Ok((seq, _ts)) => successes.push(seq),
            Err(ServiceError::Aborted(_)) => aborts += 1,
            Err(e) => other_errors.push(format!("{e:?}")),
        }
    }
    eprintln!(
        "{scenario}: {writers} writers settled in {:.3}s ({} committed, {aborts} aborted)",
        started.elapsed().as_secs_f64(),
        successes.len()
    );

    assert!(
        other_errors.is_empty(),
        "{scenario}: unexpected non-Aborted errors from {writers} concurrent writers. Ordinary \
         contention resolves as `Aborted` (retriable) or a commit; anything else is a guard \
         firing on healthy traffic: {other_errors:?}"
    );
    assert_eq!(
        successes.len() as u32 + aborts,
        writers as u32,
        "{scenario}: total outcomes must equal writers={writers}; got successes={successes:?} \
         aborts={aborts}"
    );
    assert_eq!(
        aborts, 0,
        "{scenario}: with the DEFAULT retry budget every writer must eventually commit — the \
         retry loop exists so that ordinary optimistic-concurrency losses are invisible to the \
         caller. An `Aborted` surfacing here is retry exhaustion under contention this test \
         deliberately maximises, which a client would see as a failed write. This zero is the \
         suite's ONE tolerance for this scenario (it replaced a second, looser `>= 99%` bound \
         that a separately-written copy of this race carried in \
         `update_layer_retry_test`); a budget of {DEFAULT_UPDATE_LAYER_RETRY_MAX_ATTEMPTS} \
         attempts against a {DEFAULT_UPDATE_LAYER_PER_OBJECT_CONCURRENCY}-writer admission gate \
         is not a close call, so a single abort here is a real regression and not CI weather. \
         successes={successes:?}"
    );

    // Distinct + contiguous from the post-setup sequence: `apply_delta`
    // advances the sequence by exactly one, so N commits must produce a
    // gap-free run. A repeat means one writer lied about committing; a gap
    // means a commit consumed a sequence server-side and was reported as
    // something else.
    let mut sorted = successes.clone();
    sorted.sort_unstable();
    let expected: Vec<u64> = ((base_seq + 1)..=(base_seq + writers as u64)).collect();
    assert_eq!(
        sorted,
        expected,
        "{scenario}: the {writers} commits must occupy the consecutive sequences {}..={} with \
         no repeat and no gap",
        base_seq + 1,
        base_seq + writers as u64
    );

    // XLEN parity — the authoritative on-disk state. Silent dropped writes
    // (b.8q1) manifest here as XLEN < expected: `update_layer` returned
    // `Ok(seq)` but no XADD landed.
    let entries: Vec<(String, HashMap<String, Bytes>)> = tv
        .client()
        .pool
        .xrange_values(keys.layer_stream(oid), "-", "+", None)
        .await
        .expect("XRANGE must succeed");
    assert_eq!(
        entries.len(),
        1 + setup_commits + successes.len(),
        "{scenario}: stream XLEN must equal CreateLayer's 1-0 plus {setup_commits} setup \
         commit(s) plus one entry per racing commit; successes={successes:?}, \
         entries={entries:?}"
    );
    for seq in &successes {
        let want_id = format!("{seq}-0");
        assert!(
            entries.iter().any(|(id, _)| id == &want_id),
            "{scenario}: writer reported Ok(seq={seq}) but the stream has no {want_id} entry — \
             a silent dropped write (b.8q1). entries={entries:?}"
        );
    }

    assert_race_guard_counters_stayed_zero(&metrics, scenario);

    (tv, oid, metrics)
}

/// The b.8q1 no-silent-drop claim about the shared 50-writer race.
///
/// Every assertion is inside [`run_fifty_writer_race`]: 50 concurrent writers on
/// one object, all 50 commit (zero aborts), the sequences they report are
/// distinct and gap-free, every reported sequence has its stream entry, and the
/// three race-guard counters stay at zero.
///
/// # Its sibling claim
///
/// `update_layer_retry_test::update_layer_fifty_concurrent_writers_commit_by_
/// consuming_retries` drives the SAME helper and adds the retry-loop claims on
/// top (the attempts histogram saw one sample per commit, retries were actually
/// consumed, none exhausted). The two are deliberately one scenario with two
/// claims: they used to be two independently written races whose abort
/// tolerances disagreed (0 here, ≤1 there), which is a suite that cannot say
/// what the contract is.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn update_layer_fifty_concurrent_writers_never_silently_drop_writes() {
    const WRITERS: usize = 50;

    run_fifty_writer_race(
        "race-layer",
        Vec::new(),
        (0..WRITERS).map(|_| build_valid_delta()).collect(),
        "50 empty-delta writers",
    )
    .await;
}

/// The structural twin of the race above (b.459).
///
/// The empty delta is fast-path-shaped, so under the `Txn` knob it takes the
/// `!needs_snapshot` branch: no snapshot fan-out, a four-key probe list, and a
/// WATCH window of about three round trips. A structural delta takes the branch
/// the slow path exists for — a snapshot read fanning out over the whole
/// object, a probe list that grows with the delta, and a WATCH window an order
/// of magnitude wider. That is where contention actually bites, and where a
/// guard that misfires under load would do so first.
///
/// # Shape: pre-created sections, concurrent renames
///
/// The 50 sections are created SEQUENTIALLY in one setup commit; the race is 50
/// concurrent `DiffChangeSection` renames, each writer against its own
/// pre-existing section, each paired with a field set on that same section.
/// `lua::is_fast_path` classifies `DiffChangeSection` as structural, so every
/// writer takes the snapshot branch — which is the point — and
/// `apply_diff_change_section`'s existing-section arm emits only `SetSection`
/// when the parent and the section order are unchanged, so the race mutates no
/// children list and no index.
///
/// **Why the writers do not concurrently CREATE their sections.** When this
/// test was written that shape tripped a debug-only cross-map assertion in
/// `src/domain/delta.rs` (`debug_assert_cross_map_consistent`,
/// "state.children[0] lists section_id N which is not in state.sections")
/// **before** the commit path was reached: the snapshot read takes the sections
/// index, the section records and the children lists in separate round trips,
/// so under concurrent `CreateSection` it could catch a writer's children-list
/// entry without its section record, and `apply_delta` ran on that torn state
/// before `WATCH` could cancel the `EXEC` — ticket **b.mq1** (debug-crash /
/// release-silent), out of b.459's scope. The structural path now proves its
/// read before `apply_delta` (Spec SDD SR-4.3: the read's `seq` must equal the
/// WATCHed header's, and still be the header's after the read), so a torn read
/// is `Aborted` and retried instead. That shape is now its own test,
/// [`update_layer_fifty_concurrent_create_section_writers_never_reach_the_domain_with_a_torn_read`];
/// the deterministic torn-read tests are inline in `src/storage/layer.rs`. This
/// test stays on renames: it is about commit-path guards under contention, not
/// about the read.
///
/// Renaming a pre-existing section is drift-free by construction: the sections
/// index, every section's `parent_id` and every children list are constant
/// across the race, so no interleaving of the snapshot's round trips can produce
/// a torn view.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn update_layer_fifty_concurrent_structural_writers_all_materialize() {
    const WRITERS: usize = 50;
    /// Section ids start well above anything the fixture creates, so a stray
    /// fixture section could not be mistaken for a writer's.
    const FIRST_SECTION_ID: u64 = 1_000;

    // One setup commit creating all 50 sections. `CreateSection` hardcodes
    // `section_order = 0` (it carries no order field), which is what the
    // writers' renames below pass back so that
    // `apply_diff_change_section` takes its no-children-rebuild path.
    let setup = OwnedDelta {
        struct_commands: (0..WRITERS)
            .map(|i| {
                StructCommand::CreateSection(CommandCreateSection {
                    parent_id: 0,
                    section_id: FIRST_SECTION_ID + i as u64,
                    section_name: format!("writer-{i}-section"),
                    section_type: 1,
                })
            })
            .collect(),
        ..Default::default()
    };

    let deltas: Vec<OwnedDelta> = (0..WRITERS)
        .map(|i| {
            let section_id = FIRST_SECTION_ID + i as u64;
            OwnedDelta {
                struct_commands: vec![StructCommand::DiffChangeSection(CommandDiffChangeSection {
                    parent_id: 0,
                    section_id,
                    section_name: format!("writer-{i}-renamed"),
                    section_type: 1,
                    section_order: 0,
                })],
                set_fields: vec![SetField {
                    section_id,
                    key_name: "k".to_owned(),
                    value_or_ext_hash: vec![i as u8],
                    ext_value_size: 0,
                    ext_value_index: 0,
                    set_order: 0,
                }],
                ..Default::default()
            }
        })
        .collect();

    let (tv, oid, _metrics) = run_fifty_writer_race(
        "structural-race-layer",
        vec![setup],
        deltas,
        "50 structural (DiffChangeSection rename + SetField) writers",
    )
    .await;

    // Every writer's content must be on disk, not merely acked. A commit that
    // reported `Ok` and landed its stream entry while its `HSET`s were lost is
    // exactly the tear the pre-`MULTI` guards exist to prevent, and the stream
    // assertions above cannot see it.
    let keys = tv.key_schema();
    let oid_str = oid.to_string();
    let section_ids: HashSet<String> = tv
        .client()
        .pool
        .smembers(keys.obj_sections_index(&oid_str))
        .await
        .expect("SMEMBERS sections index must succeed");

    for i in 0..WRITERS {
        let section_id = (FIRST_SECTION_ID + i as u64).to_string();
        assert!(
            section_ids.contains(&section_id),
            "writer-{i}'s section {section_id} is missing from the sections index. \
             index={section_ids:?}"
        );
        let section: HashMap<String, String> = tv
            .client()
            .pool
            .hgetall(keys.obj_section(&oid_str, &section_id))
            .await
            .expect("HGETALL section must succeed");
        assert_eq!(
            section.get("name").map(String::as_str),
            Some(format!("writer-{i}-renamed").as_str()),
            "writer-{i}'s rename must be materialized — the section still carries its setup \
             name, so the commit was acked while its `SetSection` was lost; got {section:?}"
        );
        let field: HashMap<String, String> = tv
            .client()
            .pool
            .hgetall(keys.obj_field(&oid_str, &section_id, "k"))
            .await
            .expect("HGETALL field must succeed");
        assert!(
            !field.is_empty(),
            "writer-{i}'s field on section {section_id} is missing — the `SetField` half of its \
             delta was lost while the rename landed, which is a half-commit"
        );
    }
    assert_eq!(
        section_ids.len(),
        WRITERS,
        "the sections index must carry exactly one section per writer and nothing else"
    );
}

/// b.mq1's shape, now a regression test: 50 concurrent `CreateSection` writers
/// under the root of one object, on the txn backend.
///
/// Every commit here rewrites the sections index, a new section record and the
/// root's children list, so a structural snapshot read overlapping another
/// writer's commit can see a children-list entry whose section record or index
/// member it read before the commit landed. Before the structural read was
/// proven (Spec SDD SR-4.3) that torn state reached `apply_delta`, and in a
/// debug build — which this test binary is — `debug_assert_cross_map_consistent`
/// panicked the writer ("state.children[0] lists section_id N which is not in
/// state.sections"). With the proof such an attempt is `Aborted` before the
/// domain runs and the retry loop re-runs it, so every writer must commit, none
/// may come back `InvalidArgument` (or any other non-`Aborted` error), and no
/// writer task may panic — `run_fifty_writer_race` asserts all three, plus
/// contiguous sequences and stream parity. This is the observable form of the
/// PRD's "zero torn snapshots reaching structural domain logic".
///
/// Overlap is the helper's: a barrier releases all writers together and the
/// per-object admission gate lets 16 into `update_layer_once` at once. Whether a
/// given run actually tears a read is up to scheduling, so a pass here is
/// evidence rather than proof; the deterministic torn-read tests are inline in
/// `src/storage/layer.rs`. Runs in a few seconds, far inside nextest's timeout.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn update_layer_fifty_concurrent_create_section_writers_never_reach_the_domain_with_a_torn_read(
) {
    const WRITERS: usize = 50;
    /// Section ids well above anything the fixture creates.
    const FIRST_SECTION_ID: u64 = 2_000;

    let deltas: Vec<OwnedDelta> = (0..WRITERS)
        .map(|i| OwnedDelta {
            struct_commands: vec![StructCommand::CreateSection(CommandCreateSection {
                parent_id: 0,
                section_id: FIRST_SECTION_ID + i as u64,
                // Distinct names: `apply_create_section` cascade-deletes a
                // same-named sibling, so a shared name would let a later
                // writer's commit delete an earlier writer's section.
                section_name: format!("writer-{i}-created"),
                section_type: 1,
            })],
            ..Default::default()
        })
        .collect();

    let (tv, oid, _metrics) = run_fifty_writer_race(
        "concurrent-create-layer",
        Vec::new(),
        deltas,
        "50 concurrent CreateSection writers",
    )
    .await;

    let keys = tv.key_schema();
    let oid_str = oid.to_string();
    let section_ids: HashSet<String> = tv
        .client()
        .pool
        .smembers(keys.obj_sections_index(&oid_str))
        .await
        .expect("SMEMBERS sections index must succeed");
    let root_children: Vec<String> = tv
        .client()
        .pool
        .lrange(keys.obj_children(&oid_str, "0"), 0, -1)
        .await
        .expect("LRANGE root children must succeed");
    let expected: HashSet<String> = (0..WRITERS)
        .map(|i| (FIRST_SECTION_ID + i as u64).to_string())
        .collect();
    assert_eq!(
        section_ids, expected,
        "every writer's section, and nothing else, must be in the sections index"
    );
    assert_eq!(
        root_children.iter().cloned().collect::<HashSet<_>>(),
        expected,
        "the root's children list must name every writer's section"
    );
    assert_eq!(
        root_children.len(),
        WRITERS,
        "and name each exactly once — a commit computed against a torn read could rewrite the \
         list from a view missing or repeating a sibling"
    );
}

// =============================================================================
// 8. Parametric concurrency coverage for b.8q1 — same invariants as test #7
//    but at N=2 (minimum race) and N=10 (scale). The shared helper enforces
//    the cross-cutting invariants:
//      * Every Ok(seq) has a matching {seq}-0 entry in the layer stream.
//      * All successful sequence_nums are distinct and contiguous starting at 2.
//      * XLEN == 1 (CreateLayer) + successes.len().
//      * successes + aborts == WRITERS; no spurious other errors.
//      * At least one writer commits.
//
//    Contiguity matters beyond distinctness: `update_layer` computes
//    `new_seq = state.sequence_num` after `apply_delta` bumps it by 1, so
//    successful commits must produce a gap-free 2..=K+1 run. A non-contiguous
//    gap would mean either an invisible aborted commit or a skipped counter.
// =============================================================================
//
// Why parameterize the helper rather than proptest?
// Proptest cannot live in this file (see CLAUDE.md §Proptest Isolation —
// proptest + tokio multi-thread deadlocks at process exit), and a handful of
// fixed N values exercise the same invariants more cheaply than a
// randomized harness we'd have to move to a separate binary.

async fn run_n_concurrent_writers_on_one_layer(writers: usize, pool_size: u32) {
    assert!(
        pool_size as usize >= writers,
        "pool_size ({pool_size}) must be >= writers ({writers}) — otherwise the \
         ExclusivePool mutex serializes callers and the race never happens"
    );

    let tv = TestValkey::setup_with_pool_size(pool_size).await;
    let ls = layer_store(&tv);
    let keys = tv.key_schema();
    let session_id = create_test_session(&tv).await;
    let oid = create_test_layer(&ls, &session_id, "race-layer").await;

    let delta = build_valid_delta();

    // Barrier forces every writer task past `barrier.wait()` simultaneously
    // before any of them issues WATCH. Without it, especially at small N
    // (e.g. N=2), the first task could finish its entire WATCH/HGETALL/
    // load_snapshot/MULTI/EXEC sequence before the second task even gets
    // scheduled — no WATCH overlap, no race, the test passes vacuously even
    // against the b.8q1 bug. The barrier guarantees the contention window we
    // need to catch silent-drop regressions.
    let barrier = std::sync::Arc::new(tokio::sync::Barrier::new(writers));

    let mut handles = Vec::with_capacity(writers);
    for i in 0..writers {
        let ls_i = ls.clone();
        let sid_i = session_id.clone();
        let d_i = delta.clone();
        let barrier_i = barrier.clone();
        let client_id = format!("writer-{i}");
        let client_ts = 1000u64 + i as u64;
        handles.push(tokio::spawn(async move {
            barrier_i.wait().await;
            ls_i.update_layer_relayed(&sid_i, "race-layer", oid, Some(d_i), &client_id, client_ts)
                .await
        }));
    }

    let mut successes: Vec<u64> = Vec::new();
    let mut aborts: u32 = 0;
    let mut other_errors: Vec<String> = Vec::new();
    for h in handles {
        match h.await.expect("writer task must not panic") {
            Ok((seq, _ts)) => successes.push(seq),
            Err(ServiceError::Aborted(_)) => aborts += 1,
            Err(e) => other_errors.push(format!("{e:?}")),
        }
    }

    assert!(
        other_errors.is_empty(),
        "unexpected non-Aborted errors from {writers} concurrent writers: {other_errors:?}"
    );
    assert_eq!(
        successes.len() as u32 + aborts,
        writers as u32,
        "total outcomes must equal writers={writers}; got successes={successes:?} aborts={aborts}"
    );
    assert!(
        !successes.is_empty(),
        "at least one writer must succeed; got 0 successes, {aborts} aborts ({writers} writers)"
    );

    // Distinct-seq invariant — two Ok responses claiming the same seq means
    // one of them silently lost its write.
    let mut unique: std::collections::HashSet<u64> = std::collections::HashSet::new();
    for seq in &successes {
        assert!(
            unique.insert(*seq),
            "two writers reported the same sequence_num={seq}; successes={successes:?} — \
             indicates silent dropped writes (b.8q1)"
        );
    }

    // Contiguous-seq invariant — successful commits must produce seqs
    // 2..=2+successes.len()-1 with no gaps. A gap would mean a commit
    // succeeded server-side (consumed a seq) but was reported as Aborted
    // client-side, or a false Ok skipped a seq.
    let mut sorted_seqs = successes.clone();
    sorted_seqs.sort();
    for (i, &seq) in sorted_seqs.iter().enumerate() {
        let expected = 2 + i as u64;
        assert_eq!(
            seq, expected,
            "successful sequence_nums must be contiguous starting at 2; \
             sorted={sorted_seqs:?}, got {seq} at position {i} (expected {expected})"
        );
    }

    // XLEN parity — the per-object stream must contain exactly CreateLayer's
    // 1-0 entry plus one {seq}-0 entry per successful writer.
    let stream_key = keys.layer_stream(oid);
    // Fields read as bytes: an entry's relay `ompf` field is binary.
    let entries: Vec<(String, HashMap<String, Bytes>)> = tv
        .client()
        .pool
        .xrange_values(&stream_key, "-", "+", None)
        .await
        .expect("XRANGE must succeed");

    let xlen = entries.len();
    let expected_xlen = 1 + successes.len();
    assert_eq!(
        xlen, expected_xlen,
        "layer stream XLEN must equal 1 (CreateLayer 1-0) + {} successful updates, got XLEN={xlen}; \
         successes={successes:?}, aborts={aborts}, entries={entries:?}",
        successes.len()
    );

    for seq in &successes {
        let want_id = format!("{seq}-0");
        let found = entries.iter().any(|(id, _)| id == &want_id);
        assert!(
            found,
            "writer reported Ok(seq={seq}) but stream has no {want_id} entry; entries={entries:?} — \
             silent dropped write (b.8q1)"
        );
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_two_concurrent_writers_distinct_seqs_and_xlen_parity() {
    run_n_concurrent_writers_on_one_layer(2, 4).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn update_layer_ten_concurrent_writers_distinct_seqs_and_xlen_parity() {
    run_n_concurrent_writers_on_one_layer(10, 20).await;
}

// =============================================================================
// WATCH-window contention measurement (ignored by default — manual perf run).
//
// Reports `Ok` / `Aborted` counts and wall time for N concurrent SetField-only
// writers on one object. Used to verify the field-only fast path in
// `update_layer` (skip the snapshot read when the delta has no
// structural commands and isn't a reset) actually shrinks the WATCH window
// under contention.
//
// Run:  cargo test -p live-edit --test integration -- --ignored --nocapture \
//           storage::update_layer_transaction_test::update_layer_field_only_contention_bench
// =============================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "manual: WATCH-window contention measurement, Ok/Aborted counts and wall time for \
            SetField-only writers on one object"]
async fn update_layer_field_only_contention_bench() {
    const WRITERS: usize = 20;
    const UPDATES_PER_WRITER: usize = 10;
    const POOL: u32 = 32;

    let tv = TestValkey::setup_with_pool_size(POOL).await;
    let ls = layer_store(&tv);
    let session_id = create_test_session(&tv).await;
    let oid = create_test_layer(&ls, &session_id, "bench-layer").await;

    let barrier = std::sync::Arc::new(tokio::sync::Barrier::new(WRITERS));
    let start = std::time::Instant::now();
    let mut handles = Vec::with_capacity(WRITERS);
    for wid in 0..WRITERS {
        let ls_i = ls.clone();
        let sid_i = session_id.clone();
        let barrier_i = barrier.clone();
        handles.push(tokio::spawn(async move {
            barrier_i.wait().await;
            let mut oks = 0u32;
            let mut aborts = 0u32;
            let mut others = 0u32;
            for u in 0..UPDATES_PER_WRITER {
                let delta = delta_with_set_field_inline(
                    1,
                    &format!("k{wid}-{u}"),
                    vec![wid as u8, u as u8],
                    0,
                );
                let cid = format!("w{wid}");
                match ls_i
                    .update_layer_relayed(&sid_i, "bench-layer", oid, Some(delta), &cid, 0)
                    .await
                {
                    Ok(_) => oks += 1,
                    Err(ServiceError::Aborted(_)) => aborts += 1,
                    Err(_) => others += 1,
                }
            }
            (oks, aborts, others)
        }));
    }

    let mut total_ok = 0u32;
    let mut total_abort = 0u32;
    let mut total_other = 0u32;
    for h in handles {
        let (o, a, x) = h.await.expect("task panic");
        total_ok += o;
        total_abort += a;
        total_other += x;
    }
    let elapsed = start.elapsed();
    let total = total_ok + total_abort + total_other;
    let ok_rate = 100.0 * total_ok as f64 / total as f64;
    let thrpt = total_ok as f64 / elapsed.as_secs_f64();
    println!(
        "\n── update_layer field-only contention: N={WRITERS}, M={UPDATES_PER_WRITER}, pool={POOL} ──\n\
         wall:       {:>7.3}s\n\
         total:      {total}\n\
         ok:         {total_ok}  ({ok_rate:.1}%)\n\
         aborted:    {total_abort}\n\
         other err:  {total_other}\n\
         throughput: {thrpt:.1} ok/sec\n",
        elapsed.as_secs_f64()
    );
    assert_eq!(
        total_other, 0,
        "unexpected non-Aborted errors: {total_other}"
    );
    assert!(total_ok > 0, "no writer committed");
}

// =============================================================================
// 9. Mixed-layer concurrent writes — 5 writers per layer × 2 layers in the
//    same session. Proves there is no cross-layer interference: each layer's
//    stream and header counter evolves independently, and silent dropped
//    writes cannot leak across {object_id} hash slots. The per-object XLEN
//    parity and distinct/contiguous seq invariants apply per layer.
// =============================================================================

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn update_layer_concurrent_writes_on_two_layers_no_cross_layer_interference() {
    const WRITERS_PER_LAYER: usize = 5;
    const POOL: u32 = 2 * WRITERS_PER_LAYER as u32 * 2;

    let tv = TestValkey::setup_with_pool_size(POOL).await;
    let ls = layer_store(&tv);
    let keys = tv.key_schema();
    let session_id = create_test_session(&tv).await;

    let oid_a = create_test_layer(&ls, &session_id, "layer-a").await;
    let oid_b = create_test_layer(&ls, &session_id, "layer-b").await;
    assert_ne!(
        oid_a, oid_b,
        "the two layers must have distinct object_ids so stream/header keys don't collide"
    );

    let delta = build_valid_delta();

    // Barrier so all WRITERS_PER_LAYER * 2 tasks issue their first WATCH at
    // the same instant — otherwise tasks could serialize naturally and the
    // race window (needed to exercise the b.8q1 nil-EXEC path) might not
    // open at all.
    let total_writers = WRITERS_PER_LAYER * 2;
    let barrier = std::sync::Arc::new(tokio::sync::Barrier::new(total_writers));

    // Spawn all 10 writers together. They race across both layers so that
    // each layer sees real contention and the two races interleave.
    let mut handles = Vec::with_capacity(total_writers);
    for (layer_name, oid) in [("layer-a", oid_a), ("layer-b", oid_b)] {
        for i in 0..WRITERS_PER_LAYER {
            let ls_i = ls.clone();
            let sid_i = session_id.clone();
            let d_i = delta.clone();
            let barrier_i = barrier.clone();
            let client_id = format!("{layer_name}-writer-{i}");
            let client_ts = 1000u64 + i as u64;
            let name = layer_name.to_string();
            handles.push(tokio::spawn(async move {
                barrier_i.wait().await;
                let res = ls_i
                    .update_layer_relayed(&sid_i, &name, oid, Some(d_i), &client_id, client_ts)
                    .await;
                (name, oid, res)
            }));
        }
    }

    // Bucket results by layer so we can check each layer's invariants in
    // isolation.
    let mut successes_a: Vec<u64> = Vec::new();
    let mut successes_b: Vec<u64> = Vec::new();
    let mut aborts_a: u32 = 0;
    let mut aborts_b: u32 = 0;
    let mut other_errors: Vec<String> = Vec::new();

    for h in handles {
        let (layer_name, _oid, res) = h.await.expect("task must not panic");
        match res {
            Ok((seq, _ts)) => {
                if layer_name == "layer-a" {
                    successes_a.push(seq);
                } else {
                    successes_b.push(seq);
                }
            }
            Err(ServiceError::Aborted(_)) => {
                if layer_name == "layer-a" {
                    aborts_a += 1;
                } else {
                    aborts_b += 1;
                }
            }
            Err(e) => other_errors.push(format!("{layer_name}: {e:?}")),
        }
    }

    assert!(
        other_errors.is_empty(),
        "unexpected non-Aborted errors in mixed-layer race: {other_errors:?}"
    );
    assert_eq!(
        successes_a.len() as u32 + aborts_a,
        WRITERS_PER_LAYER as u32,
        "layer-a outcome count: successes={successes_a:?}, aborts={aborts_a}"
    );
    assert_eq!(
        successes_b.len() as u32 + aborts_b,
        WRITERS_PER_LAYER as u32,
        "layer-b outcome count: successes={successes_b:?}, aborts={aborts_b}"
    );
    assert!(
        !successes_a.is_empty() && !successes_b.is_empty(),
        "each layer must see at least one committed writer; got \
         layer-a successes={successes_a:?}, layer-b successes={successes_b:?}"
    );

    // Per-layer invariants — distinct, contiguous seqs + XLEN parity.
    for (oid, successes, aborts, layer_name) in [
        (oid_a, &successes_a, aborts_a, "layer-a"),
        (oid_b, &successes_b, aborts_b, "layer-b"),
    ] {
        // Distinct seqs within the layer.
        let mut unique: std::collections::HashSet<u64> = std::collections::HashSet::new();
        for seq in successes {
            assert!(
                unique.insert(*seq),
                "{layer_name}: duplicate sequence_num={seq} in successes={successes:?}"
            );
        }

        // Contiguous starting at 2 within the layer.
        let mut sorted = successes.clone();
        sorted.sort();
        for (i, &seq) in sorted.iter().enumerate() {
            let expected = 2 + i as u64;
            assert_eq!(
                seq, expected,
                "{layer_name}: successful seqs must be contiguous from 2; \
                 sorted={sorted:?}, seq={seq} at position {i} (expected {expected})"
            );
        }

        // XLEN parity per layer — CreateLayer 1-0 plus successes.
        let stream_key = keys.layer_stream(oid);
        // Fields read as bytes: an entry's relay `ompf` field is binary.
        let entries: Vec<(String, HashMap<String, Bytes>)> = tv
            .client()
            .pool
            .xrange_values(&stream_key, "-", "+", None)
            .await
            .expect("XRANGE must succeed");

        let xlen = entries.len();
        let expected_xlen = 1 + successes.len();
        assert_eq!(
            xlen,
            expected_xlen,
            "{layer_name} stream XLEN must equal 1 + {} successful writers, got {xlen}; \
             entries={entries:?}, successes={successes:?}, aborts={aborts}",
            successes.len()
        );

        for seq in successes {
            let want_id = format!("{seq}-0");
            assert!(
                entries.iter().any(|(id, _)| id == &want_id),
                "{layer_name}: Ok(seq={seq}) has no matching {want_id} in stream; entries={entries:?}"
            );
        }
    }

    // Cross-layer isolation check — no layer-a update should appear in
    // layer-b's stream or vice versa. The payload carries the object_id, so
    // we can verify every entry belongs to the layer whose stream we read.
    for (oid, layer_name) in [(oid_a, "layer-a"), (oid_b, "layer-b")] {
        let stream_key = keys.layer_stream(oid);
        // Fields read as bytes: an entry's relay `ompf` field is binary.
        let entries: Vec<(String, HashMap<String, Bytes>)> = tv
            .client()
            .pool
            .xrange_values(&stream_key, "-", "+", None)
            .await
            .expect("XRANGE must succeed");

        for (id, fields) in &entries {
            // Every entry's payload carries the object id, CreateLayer's
            // `create|` marker at 1-0 included.
            let payload = fields
                .get("payload")
                .map(|p| std::str::from_utf8(p).expect("a stream payload is UTF-8"))
                .unwrap_or_else(|| {
                    panic!("{layer_name}: entry {id} missing payload field: {fields:?}")
                });
            let parts: Vec<&str> = payload.split('|').collect();
            let entry_oid: u64 = parts
                .get(1)
                .and_then(|s| s.parse().ok())
                .unwrap_or_else(|| {
                    panic!("{layer_name}: entry {id} payload has no valid oid: {payload}")
                });
            assert_eq!(
                entry_oid, oid,
                "{layer_name}: stream contains foreign entry — oid in payload {entry_oid} != \
                 expected {oid}; payload={payload}"
            );
        }
    }
}

// =============================================================================
// 10. Field-only / time-sample-only fast path (b.mnu).
//
// `update_layer_once_watch` skips the snapshot read when the delta has no
// structural commands and is not a reset, building a minimal `ObjectState`
// from just the WATCHed-connection header HGETALL (see
// `src/storage/diff_read.rs::header_to_object_state`). These tests assert
// round-trip correctness of that fast path and that the slow-path delta
// shapes still work end-to-end. Any regression in how the fast-path
// ObjectState is populated (missing `next_set_order`, missing `seq`, etc.)
// surfaces here as a wrong `set_order`, wrong `version`, wrong stored
// value, or a missing field in the round-tripped snapshot.
// =============================================================================

/// Round-trip a single SetField through `update_layer` (fast path) and
/// verify the stored field is correct when read back via
/// `read_object_state_for_test` (a plain read, independent of the write path).
///
/// Invariants verified:
///   * field value bytes match what was written
///   * field's `version` equals the post-update sequence_num
///   * field's `set_order` derives from the header's `next_set_order` at
///     the time of the update (fast path reads this from the header)
///   * the object header's `next_set_order` is bumped exactly by one
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_field_only_fast_path_round_trips_correctly() {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let session_id = create_test_session(&tv).await;
    let oid = create_test_layer(&ls, &session_id, "fastpath-field-layer").await;

    // One SetField, no struct commands, not a reset → fast path. Pass a
    // non-zero delta set_order so a regression that misparses the header's
    // next_set_order (e.g. defaults to 0) produces a visibly-wrong stored
    // set_order.
    let delta = delta_with_set_field_inline(7, "color", vec![0xDE, 0xAD, 0xBE, 0xEF], 3);

    let (seq, _server_ts) = ls
        .update_layer_relayed(
            &session_id,
            "fastpath-field-layer",
            oid,
            Some(delta),
            "fastpath-client",
            1000,
        )
        .await
        .expect("fast-path update_layer must succeed");

    // create_layer wrote seq=1; one update bumps to seq=2.
    assert_eq!(seq, 2, "seq must increment exactly once");

    let snapshot = read_object_state_for_test(tv.client(), tv.key_schema(), &oid.to_string())
        .await
        .expect("read_object_state_for_test after fast-path update must succeed");

    // Header invariants — snapshot-independent because they come from the
    // header hash (the same place header_to_object_state reads).
    assert_eq!(snapshot.sequence_num, seq, "stored seq must match ack");

    // apply_set_field: stored_set_order = next_set_order(=0) + delta_set_order(=3) = 3
    //                   next_set_order = max(0, 3+1) = 4
    assert_eq!(
        snapshot.next_set_order, 4,
        "next_set_order must advance by delta_set_order+1 (fast path still \
         read-modify-writes the header field)"
    );

    // Field round-trip — exactly one field, with the exact bytes and
    // derived metadata the delta specified.
    assert_eq!(
        snapshot.fields.len(),
        1,
        "exactly one field must land; got {:?}",
        snapshot.fields
    );
    let f = &snapshot.fields[0];
    assert_eq!(f.section_id, 7, "section_id must be preserved");
    assert_eq!(f.key_name, "color", "key_name must be preserved");
    assert_eq!(
        f.value,
        FieldValue::Inline(vec![0xDE, 0xAD, 0xBE, 0xEF]),
        "value bytes must round-trip exactly"
    );
    assert_eq!(
        f.set_order, 3,
        "stored set_order = header.next_set_order(0) + delta.set_order(3) = 3"
    );
    assert_eq!(f.version, seq, "field version must equal post-update seq");
}

/// Two sequential SetField updates against the same object — verifies that
/// the fast path reads the *latest* next_set_order on each attempt (not a
/// stale cached value) and that `set_order` values are monotonic.
///
/// If the fast path incorrectly reconstructed state on the second call
/// (e.g. by re-using a cached zero-valued next_set_order), the second
/// field's stored set_order would collide with the first.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_sequential_field_only_updates_have_monotonic_set_orders() {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let session_id = create_test_session(&tv).await;
    let oid = create_test_layer(&ls, &session_id, "fastpath-seq-layer").await;

    // Update 1 — set_order delta 0 → stored 0, header next_set_order=1.
    let delta1 = delta_with_set_field_inline(1, "a", vec![0x01], 0);
    let (seq1, _) = ls
        .update_layer_relayed(&session_id, "fastpath-seq-layer", oid, Some(delta1), "c", 1)
        .await
        .expect("first fast-path update must succeed");

    // Update 2 — set_order delta 0 → stored 1 (header already at 1),
    // header next_set_order=2.
    let delta2 = delta_with_set_field_inline(1, "b", vec![0x02], 0);
    let (seq2, _) = ls
        .update_layer_relayed(&session_id, "fastpath-seq-layer", oid, Some(delta2), "c", 2)
        .await
        .expect("second fast-path update must succeed");

    assert_eq!(seq1, 2);
    assert_eq!(seq2, 3);

    let snapshot = read_object_state_for_test(tv.client(), tv.key_schema(), &oid.to_string())
        .await
        .expect("read_object_state_for_test must succeed");

    assert_eq!(snapshot.next_set_order, 2, "header must advance to 2");
    assert_eq!(snapshot.fields.len(), 2, "both fields must land");

    let fa = snapshot
        .fields
        .iter()
        .find(|f| f.key_name == "a")
        .expect("field 'a' must exist");
    let fb = snapshot
        .fields
        .iter()
        .find(|f| f.key_name == "b")
        .expect("field 'b' must exist");

    assert_eq!(fa.set_order, 0, "first stored set_order must be 0");
    assert_eq!(
        fb.set_order, 1,
        "second stored set_order must be 1 (header was at 1 when delta2 ran)"
    );
    assert_eq!(fa.version, seq1, "field a version matches first commit");
    assert_eq!(fb.version, seq2, "field b version matches second commit");
}

/// TimeSample-only delta — also eligible for the fast path since
/// `apply_time_sample` only writes to `state.time_samples` and never
/// reads existing state beyond the header.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_time_sample_only_fast_path_round_trips_correctly() {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let session_id = create_test_session(&tv).await;
    let oid = create_test_layer(&ls, &session_id, "fastpath-ts-layer").await;

    let delta = OwnedDelta {
        time_samples: vec![OodfTimeSample {
            section_id: 4,
            time: 2.5,
            value_or_ext_hash: vec![0xAA, 0xBB],
            ext_value_size: 0,
            ext_value_index: 0,
        }],
        ..Default::default()
    };

    let (seq, _) = ls
        .update_layer_relayed(
            &session_id,
            "fastpath-ts-layer",
            oid,
            Some(delta),
            "ts-client",
            1000,
        )
        .await
        .expect("time-sample fast-path update must succeed");

    let snapshot = read_object_state_for_test(tv.client(), tv.key_schema(), &oid.to_string())
        .await
        .expect("read_object_state_for_test must succeed");

    assert_eq!(snapshot.sequence_num, seq);
    assert_eq!(
        snapshot.time_samples.len(),
        1,
        "exactly one time sample must land; got {:?}",
        snapshot.time_samples
    );
    let ts = &snapshot.time_samples[0];
    assert_eq!(ts.section_id, 4);
    assert!(
        (ts.time.into_inner() - 2.5).abs() < f64::EPSILON,
        "time must round-trip"
    );
    assert_eq!(ts.value, FieldValue::Inline(vec![0xAA, 0xBB]));
    assert_eq!(ts.version, seq);
}

/// Slow-path regression — a delta with a `DeleteSection` on an existing
/// section forces `needs_snapshot=true`. `apply_delete_section` reads
/// `state.sections[section_id].parent_id` and walks the children tree to
/// cascade-delete descendants (SDD §9.1.4). If the fast-path predicate
/// incorrectly routed a DeleteSection through the fast path, state.sections
/// would be empty and `apply_delete_section` would be a no-op — the
/// section would NOT be deleted, and this test's final snapshot would
/// still contain the section (read_object_state_for_test returns `sections.len()
/// == 1`). This is a load-bearing fast-path-misroute regression check.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_delete_section_slow_path_actually_deletes() {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let session_id = create_test_session(&tv).await;
    let oid = create_test_layer(&ls, &session_id, "slowpath-delete-layer").await;

    // First: create a section (slow-path update, persists the section).
    let create_delta = OwnedDelta {
        struct_commands: vec![StructCommand::CreateSection(CommandCreateSection {
            parent_id: 0,
            section_id: 42,
            section_name: "ToDelete".to_owned(),
            section_type: 1,
        })],
        ..Default::default()
    };
    ls.update_layer_relayed(
        &session_id,
        "slowpath-delete-layer",
        oid,
        Some(create_delta),
        "create",
        1,
    )
    .await
    .expect("CreateSection update must succeed");

    // Sanity: section is there after the first update.
    let before = read_object_state_for_test(tv.client(), tv.key_schema(), &oid.to_string())
        .await
        .expect("read_object_state_for_test must succeed");
    assert_eq!(
        before.sections.len(),
        1,
        "setup failed: section was not created"
    );

    // Second: delete it. This MUST route through the slow path
    // (apply_delete_section needs the section's parent_id from state).
    let delete_delta = OwnedDelta {
        struct_commands: vec![StructCommand::DeleteSection(CommandDeleteSection {
            parent_id: 0,
            section_id: 42,
        })],
        ..Default::default()
    };
    let (seq, _) = ls
        .update_layer_relayed(
            &session_id,
            "slowpath-delete-layer",
            oid,
            Some(delete_delta),
            "delete",
            2,
        )
        .await
        .expect("DeleteSection update must succeed");

    let after = read_object_state_for_test(tv.client(), tv.key_schema(), &oid.to_string())
        .await
        .expect("read_object_state_for_test must succeed");

    assert_eq!(after.sequence_num, seq);
    assert_eq!(
        after.sections.len(),
        0,
        "DeleteSection must actually remove the section; fast-path misroute \
         would silently no-op because apply_delete_section returns early \
         when state.sections is empty. Remaining sections: {:?}",
        after.sections
    );
    // Tombstone must be written for diff reconstruction.
    assert_eq!(
        after.tombstones.len(),
        1,
        "DeleteSection must write a tombstone; got {:?}",
        after.tombstones
    );
    assert_eq!(after.tombstones[0].0, 42, "tombstone section_id must be 42");
}

/// Mixed delta — CreateSection + SetField in the same update. The
/// fast-path predicate must detect any non-ignored struct command and
/// take the slow path, so both the section and the field must persist.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_mixed_struct_and_field_slow_path_persists_both() {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let session_id = create_test_session(&tv).await;
    let oid = create_test_layer(&ls, &session_id, "mixed-layer").await;

    let delta = OwnedDelta {
        struct_commands: vec![StructCommand::CreateSection(CommandCreateSection {
            parent_id: 0,
            section_id: 5,
            section_name: "Mixed".to_owned(),
            section_type: 2,
        })],
        set_fields: vec![SetField {
            section_id: 5,
            key_name: "k".to_owned(),
            value_or_ext_hash: vec![0x7F],
            ext_value_size: 0,
            ext_value_index: 0,
            set_order: 0,
        }],
        ..Default::default()
    };

    let (seq, _) = ls
        .update_layer_relayed(&session_id, "mixed-layer", oid, Some(delta), "m", 1)
        .await
        .expect("mixed slow-path update must succeed");

    let snapshot = read_object_state_for_test(tv.client(), tv.key_schema(), &oid.to_string())
        .await
        .expect("read_object_state_for_test must succeed");

    assert_eq!(snapshot.sequence_num, seq);
    assert_eq!(snapshot.sections.len(), 1, "section must be created");
    assert_eq!(snapshot.sections[&5].name, "Mixed");
    assert_eq!(snapshot.fields.len(), 1, "field must be set");
    let f = &snapshot.fields[0];
    assert_eq!(f.section_id, 5);
    assert_eq!(f.key_name, "k");
    assert_eq!(f.value, FieldValue::Inline(vec![0x7F]));
}

/// Reset-delta slow-path regression — a delta with `is_diff=true,
/// base_version=0` triggers `apply_reset` (SDD §9.1.3), which walks
/// existing `state.sections` / `state.fields` to emit tombstones and
/// delete mutations. This path REQUIRES the snapshot (not just the
/// header), so `needs_snapshot` must include the `(is_diff && base_version
/// == 0)` branch. If that branch were removed, reset would silently
/// fast-path against an empty header-derived state — no tombstones, no
/// field deletes would be emitted, and pre-existing sections/fields would
/// linger in storage.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_reset_delta_slow_path_rebuilds_state() {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let session_id = create_test_session(&tv).await;
    let oid = create_test_layer(&ls, &session_id, "reset-layer").await;

    // Seed pre-existing state: one section + one field. This is the state
    // that must be torn down by the reset. We use a mixed delta (struct
    // command + field) so that `apply_delete_section` (slow path) can walk
    // it down on the reset.
    let seed_delta = OwnedDelta {
        struct_commands: vec![StructCommand::CreateSection(CommandCreateSection {
            parent_id: 0,
            section_id: 11,
            section_name: "Seed".to_owned(),
            section_type: 1,
        })],
        set_fields: vec![SetField {
            section_id: 11,
            key_name: "seed_key".to_owned(),
            value_or_ext_hash: vec![0x11],
            ext_value_size: 0,
            ext_value_index: 0,
            set_order: 0,
        }],
        ..Default::default()
    };
    let (seed_seq, _) = ls
        .update_layer_relayed(&session_id, "reset-layer", oid, Some(seed_delta), "seed", 1)
        .await
        .expect("seed update must succeed");

    // Sanity: seed landed.
    let before = read_object_state_for_test(tv.client(), tv.key_schema(), &oid.to_string())
        .await
        .expect("read_object_state_for_test must succeed");
    assert_eq!(before.sections.len(), 1, "setup: section must exist");
    assert_eq!(before.fields.len(), 1, "setup: field must exist");
    assert!(
        before.next_set_order >= 1,
        "setup: next_set_order must have advanced past 0; got {}",
        before.next_set_order
    );

    // Reset — `is_diff=true, base_version=0`. Must route through slow path
    // so `apply_reset` can walk pre-existing state.
    let (reset_seq, _) = ls
        .update_layer_relayed(
            &session_id,
            "reset-layer",
            oid,
            Some(reset_delta("usda")),
            "reset",
            2,
        )
        .await
        .expect("reset update must succeed");

    assert_eq!(
        reset_seq,
        seed_seq + 1,
        "reset must advance header sequence_num by exactly 1"
    );

    let after = read_object_state_for_test(tv.client(), tv.key_schema(), &oid.to_string())
        .await
        .expect("read_object_state_for_test after reset must succeed");

    assert_eq!(after.sequence_num, reset_seq);
    assert_eq!(
        after.sections.len(),
        0,
        "reset must erase all sections; fast-path misroute would leave them in place. \
         Remaining sections: {:?}",
        after.sections
    );
    assert_eq!(
        after.fields.len(),
        0,
        "reset must erase all fields; fast-path misroute would leave them in place. \
         Remaining fields: {:?}",
        after.fields
    );
    assert_eq!(
        after.next_set_order, 0,
        "reset must zero next_set_order per SDD §9.1.3d"
    );
    // apply_reset emits a tombstone for every pre-existing section.
    assert_eq!(
        after.tombstones.len(),
        1,
        "reset must emit a tombstone for the pre-existing section; got {:?}",
        after.tombstones
    );
    assert_eq!(
        after.tombstones[0].0, 11,
        "tombstone must cover the seeded section_id"
    );
}

/// DiffDeleteSection-only fast-path regression — a delta whose only
/// struct command is `DiffDeleteSection` is silently ignored on ingest
/// (SDD §9.1.4f). The fast-path predicate in `update_layer_once` filters
/// these out via `!matches!(..., DiffDeleteSection(_))`, so the update
/// takes the fast path even though there's a struct command. Nothing is
/// deleted, no tombstone is written, but `sequence_num` advances by 1
/// (every update bumps the header).
///
/// If the negation in the fast-path predicate is flipped, a
/// DiffDeleteSection-only delta would incorrectly route to the slow path
/// — correctness would still hold (apply_delta is a no-op for this
/// variant) but the fast-path optimization would be defeated for this
/// case. This test's value is documentary: it exercises the branch and
/// locks in the ingest-time ignore behavior.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_diff_delete_section_only_fast_path_increments_seq_but_nothing_deleted() {
    let tv = TestValkey::setup().await;
    let ls = layer_store(&tv);
    let session_id = create_test_session(&tv).await;
    let oid = create_test_layer(&ls, &session_id, "diff-delete-only-layer").await;

    let delta = OwnedDelta {
        struct_commands: vec![StructCommand::DiffDeleteSection(CommandDiffDeleteSection {
            section_id: 99,
        })],
        ..Default::default()
    };

    let (seq, _) = ls
        .update_layer_relayed(
            &session_id,
            "diff-delete-only-layer",
            oid,
            Some(delta),
            "diff-del",
            1,
        )
        .await
        .expect("DiffDeleteSection-only update must succeed");

    // create_layer wrote seq=1; one update bumps to seq=2.
    assert_eq!(seq, 2, "header sequence_num must advance by exactly 1");

    let snapshot = read_object_state_for_test(tv.client(), tv.key_schema(), &oid.to_string())
        .await
        .expect("read_object_state_for_test must succeed");

    assert_eq!(snapshot.sequence_num, seq);
    assert_eq!(
        snapshot.sections.len(),
        0,
        "DiffDeleteSection must not touch state; got sections: {:?}",
        snapshot.sections
    );
    assert_eq!(
        snapshot.tombstones.len(),
        0,
        "DiffDeleteSection must not emit a tombstone on ingest (SDD §9.1.4f); \
         got {:?}",
        snapshot.tombstones
    );
}

/// Raw `LRANGE 0 -1` of a parent's persisted children list
/// (`le:{<oid>}:obj:ch:<parent_id>`), read straight off the pool so the
/// assertion sees the stored key layout rather than a reconstruction of it.
/// `parent_id` 0 is the synthetic root.
async fn read_children_list(tv: &TestValkey, oid: u64, parent_id: u64) -> Vec<String> {
    let key = tv
        .key_schema()
        .obj_children(&oid.to_string(), &parent_id.to_string());
    tv.client()
        .pool
        .lrange(&key, 0, -1)
        .await
        .unwrap_or_else(|e| panic!("LRANGE {key} must succeed: {e}"))
}

/// Children-list keying regression (b.mq1) — each parent's children list must
/// be persisted under its **own** `obj:ch:{parent_id}` key.
///
/// The slow-path `StorageMutation::SetChildren` handler used to key the list by
/// the server-ignored, always-zero `children_list_id` carried on the mutation
/// instead of the parent `section_id`. Every parent's list therefore collapsed
/// onto the single key `obj:ch:0`, each commit's `DEL`+`RPUSH` rebuild
/// clobbering the previous parent's entry. A snapshot read then rebuilt a
/// flattened tree, so any nested scene authored incrementally through
/// UpdateLayer lost sections on read (catch-up served a scene missing real
/// prims).
///
/// The tree here is deliberately two parents wide and two levels deep — a
/// single-parent fixture cannot distinguish "keyed per parent" from "collapsed
/// onto obj:ch:0", because with one parent both layouts agree. Section names
/// are all distinct: `apply_create_section` cascade-deletes a name-conflicting
/// sibling under the same parent, which would silently gut the fixture.
///
/// The fixture runs on two backends, because the structural commands it sends
/// reach a different `SetChildren` / `DeleteChildrenList` writer on each:
///
/// - [`update_layer_slow_path_children_lists_keyed_per_parent_not_collapsed`]
///   pins the store to `lua-watch-structural`, so every commit takes the
///   WATCH/MULTI/EXEC slow path whose `apply_storage_mutation` handler b.mq1
///   fixed. That handler stays live under the `lua` backend too — for an
///   uncacheable object and a repeat-divergence fallback — and under both
///   rollback backends.
/// - [`update_layer_structural_route_children_lists_keyed_per_parent_not_collapsed`]
///   runs on the production-default `lua` backend, where the commits take the
///   cached structural route and the commit script's structural mode writes
///   the lists.
///
/// Commit 3 carries the same fixture across `StorageMutation::DeleteChildrenList`
/// — the adjacent `apply_storage_mutation` arm, keyed off the same `section_id`
/// and emitted once per cascade-deleted section. A mis-keyed `DEL` there is the
/// same class of defect with the same observable shape (a parent's list
/// surviving its section, or the root list clobbered wholesale). The raw
/// `LRANGE` assertions are the only thing that can see the surviving-list half
/// of that: a snapshot read (and `read_object_state_for_test`, which this test
/// reads state with) reads `obj:ch:{sid}` only for *live* section ids plus root,
/// so a children list orphaned by a mis-keyed `DEL` never reaches the snapshot.
///
/// Reproducing the pre-fix state takes more than reverting `src/storage/layer.rs`:
/// the always-zero `children_list_id` field this handler used to key on no longer
/// exists on `StorageMutation::SetChildren` (fixed in commit 92e63a7, field
/// deleted as vestigial in 0fc113b). Commit 92e63a7's diff is the only record of
/// the pre-fix shape.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_slow_path_children_lists_keyed_per_parent_not_collapsed() {
    children_lists_keyed_per_parent_not_collapsed(UpdateLayerBackend::LuaWatchStructural).await;
}

/// [`update_layer_slow_path_children_lists_keyed_per_parent_not_collapsed`]'s
/// fixture on the default `lua` backend: the cached structural route.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_structural_route_children_lists_keyed_per_parent_not_collapsed() {
    children_lists_keyed_per_parent_not_collapsed(UpdateLayerBackend::Lua).await;
}

/// The b.mq1 children-list keying fixture (the doc on
/// [`update_layer_slow_path_children_lists_keyed_per_parent_not_collapsed`]),
/// on a store with `backend`.
async fn children_lists_keyed_per_parent_not_collapsed(backend: UpdateLayerBackend) {
    // Two sibling subtrees: A(A1, A2) and B(B1, B2), both under root 0.
    const PARENT_A: u64 = 10;
    const CHILD_A1: u64 = 11;
    const CHILD_A2: u64 = 12;
    const PARENT_B: u64 = 20;
    const CHILD_B1: u64 = 21;
    const CHILD_B2: u64 = 22;
    const ROOT: u64 = 0;
    const LAYER: &str = "nested-children-layer";

    let tv = TestValkey::setup().await;
    let metrics = Arc::new(crate::common::metrics_scrape_only(LAYER));
    let ls = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096)
        .with_update_layer_backend(backend)
        .with_metrics(Arc::clone(&metrics));
    let session_id = create_test_session(&tv).await;
    let oid = create_test_layer(&ls, &session_id, LAYER).await;

    let create_section = |parent_id: u64, section_id: u64, name: &str| {
        StructCommand::CreateSection(CommandCreateSection {
            parent_id,
            section_id,
            section_name: name.to_owned(),
            section_type: 1,
        })
    };

    // ── Commit 1: build both subtrees one level deep ────────────────────────
    let commit_one = OwnedDelta {
        struct_commands: vec![
            create_section(ROOT, PARENT_A, "ParentA"),
            create_section(PARENT_A, CHILD_A1, "ChildA1"),
            create_section(ROOT, PARENT_B, "ParentB"),
            create_section(PARENT_B, CHILD_B1, "ChildB1"),
        ],
        ..Default::default()
    };
    ls.update_layer_relayed(&session_id, LAYER, oid, Some(commit_one), "c1", 1)
        .await
        .expect("commit 1 (nested CreateSections) must succeed");

    // Assert before commit 2 (load-bearing ordering): pre-fix, commit 2's
    // snapshot re-read reads back the collapsed tree and trips
    // `debug_assert_cross_map_consistent` in `src/domain/delta.rs`, which would
    // surface as a panic there instead of as the key-layout mismatch that is
    // the actual defect.
    assert_eq!(
        read_children_list(&tv, oid, PARENT_A).await,
        vec![CHILD_A1.to_string()],
        "ParentA's children list must live under its own obj:ch:{PARENT_A} key"
    );
    assert_eq!(
        read_children_list(&tv, oid, PARENT_B).await,
        vec![CHILD_B1.to_string()],
        "ParentB's children list must live under its own obj:ch:{PARENT_B} key"
    );
    // The collapse symptom, stated positively: root holds exactly its own two
    // children and nothing that belongs to a deeper parent. Order is
    // deterministic — CommandCreateSection carries no sectionOrder, so every
    // section gets section_order 0 and `insert_child_into_parent` appends on
    // ties, preserving authoring order.
    assert_eq!(
        read_children_list(&tv, oid, ROOT).await,
        vec![PARENT_A.to_string(), PARENT_B.to_string()],
        "root's children list must hold exactly [ParentA, ParentB]; extra or \
         replaced entries mean per-parent lists collapsed onto obj:ch:0"
    );

    // ── Commit 2: extend both subtrees (persist → snapshot re-read → persist) ─
    let commit_two = OwnedDelta {
        struct_commands: vec![
            create_section(PARENT_A, CHILD_A2, "ChildA2"),
            create_section(PARENT_B, CHILD_B2, "ChildB2"),
        ],
        ..Default::default()
    };
    ls.update_layer_relayed(&session_id, LAYER, oid, Some(commit_two), "c2", 2)
        .await
        .expect("commit 2 (second-round CreateSections) must succeed");

    assert_eq!(
        read_children_list(&tv, oid, PARENT_A).await,
        vec![CHILD_A1.to_string(), CHILD_A2.to_string()],
        "ParentA must accumulate both children across commits"
    );
    assert_eq!(
        read_children_list(&tv, oid, PARENT_B).await,
        vec![CHILD_B1.to_string(), CHILD_B2.to_string()],
        "ParentB must accumulate both children across commits"
    );
    assert_eq!(
        read_children_list(&tv, oid, ROOT).await,
        vec![PARENT_A.to_string(), PARENT_B.to_string()],
        "root's children list must be unchanged by the second-level writes"
    );

    // The reader's view of the same tree: `read_object_state_for_test` must rebuild
    // three distinct parent entries, not the single collapsed `0` entry the
    // pre-fix key layout produced.
    let snapshot = read_object_state_for_test(tv.client(), tv.key_schema(), &oid.to_string())
        .await
        .expect("read_object_state_for_test must succeed");
    assert_eq!(
        snapshot.children.len(),
        3,
        "snapshot must carry one children entry per parent (root, ParentA, \
         ParentB); got {:?}",
        snapshot.children
    );
    assert_eq!(
        snapshot.children.get(&ROOT),
        Some(&vec![PARENT_A, PARENT_B]),
        "snapshot root children mismatch; got {:?}",
        snapshot.children
    );
    assert_eq!(
        snapshot.children.get(&PARENT_A),
        Some(&vec![CHILD_A1, CHILD_A2]),
        "snapshot ParentA children mismatch; got {:?}",
        snapshot.children
    );
    assert_eq!(
        snapshot.children.get(&PARENT_B),
        Some(&vec![CHILD_B1, CHILD_B2]),
        "snapshot ParentB children mismatch; got {:?}",
        snapshot.children
    );

    // ── Commit 3: delete subtree A (exercises DeleteChildrenList) ────────────
    // `apply_delete_section` ignores the command's `parent_id` and derives the
    // parent from stored state, then `cascade_delete_section` emits one
    // `SetChildren` for the surviving root list followed by one
    // `DeleteChildrenList { section_id }` per removed section — ParentA, ChildA1
    // and ChildA2 (src/domain/delta.rs).
    let commit_three = OwnedDelta {
        struct_commands: vec![StructCommand::DeleteSection(CommandDeleteSection {
            parent_id: ROOT,
            section_id: PARENT_A,
        })],
        ..Default::default()
    };
    ls.update_layer_relayed(&session_id, LAYER, oid, Some(commit_three), "c3", 3)
        .await
        .expect("commit 3 (DeleteSection on ParentA) must succeed");

    assert_eq!(
        read_children_list(&tv, oid, PARENT_A).await,
        Vec::<String>::new(),
        "ParentA's obj:ch:{PARENT_A} key must be gone after the cascade; a \
         surviving list means DeleteChildrenList DEL'd some other key. Only \
         this raw read can catch it — the snapshot skips children lists whose \
         section_id is no longer live"
    );
    assert_eq!(
        read_children_list(&tv, oid, ROOT).await,
        vec![PARENT_B.to_string()],
        "root must retain exactly [ParentB]: the cascade removes ParentA from \
         the root list, and a DeleteChildrenList keyed by anything other than \
         the section_id would DEL obj:ch:0 wholesale"
    );
    assert_eq!(
        read_children_list(&tv, oid, PARENT_B).await,
        vec![CHILD_B1.to_string(), CHILD_B2.to_string()],
        "the sibling subtree must survive the cascade untouched"
    );

    // Reader's view of the post-delete tree.
    let after_delete = read_object_state_for_test(tv.client(), tv.key_schema(), &oid.to_string())
        .await
        .expect("read_object_state_for_test must succeed");
    assert_eq!(
        after_delete
            .sections
            .keys()
            .copied()
            .collect::<HashSet<_>>(),
        HashSet::from([PARENT_B, CHILD_B1, CHILD_B2]),
        "only subtree B may survive the cascade delete"
    );
    assert_eq!(
        after_delete.children.len(),
        2,
        "snapshot must carry exactly the root and ParentB children entries \
         after the cascade; got {:?}",
        after_delete.children
    );
    assert_eq!(
        after_delete.children.get(&ROOT),
        Some(&vec![PARENT_B]),
        "post-delete snapshot root children mismatch; got {:?}",
        after_delete.children
    );
    assert_eq!(
        after_delete.children.get(&PARENT_B),
        Some(&vec![CHILD_B1, CHILD_B2]),
        "post-delete snapshot ParentB children mismatch; got {:?}",
        after_delete.children
    );
    assert_eq!(
        after_delete
            .tombstones
            .iter()
            .map(|&(sid, _, _)| sid)
            .collect::<HashSet<_>>(),
        HashSet::from([PARENT_A, CHILD_A1, CHILD_A2]),
        "the cascade must tombstone the whole A subtree, not just its root; \
         got {:?}",
        after_delete.tombstones
    );

    // The writer under test is the one `backend` routes to: the three commits
    // took the structural route on `lua` (commit 1 a cold start, commits 2
    // and 3 cache hits) and the WATCH path otherwise.
    let committed_on = |path: &str| {
        sample_value(
            &render(&metrics),
            "live_edit_update_layer_attempts_histogram_count",
            &[("path", path)],
        )
        .unwrap_or_default() as u64
    };
    let expected = match backend {
        UpdateLayerBackend::Lua => [("structural_full", 1), ("structural", 2), ("watch", 0)],
        UpdateLayerBackend::LuaWatchStructural | UpdateLayerBackend::Txn => {
            [("structural_full", 0), ("structural", 0), ("watch", 3)]
        }
    };
    for (path, n) in expected {
        assert_eq!(
            committed_on(path),
            n,
            "{backend:?}: commits on path {path:?}"
        );
    }
}
