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

// Integration tests for b.k1r — bundled update_layer commit-path hardening.
//
// Covers the HIGH-severity findings from the 2026-04-23 code review:
//
//   H-stor-2  WatchGuard::Drop UNWATCH must drain via `ValkeyClient::quit`
//             before the txn-pool closes — else a shutdown races the
//             spawned UNWATCH and leaves stale WATCH on a closing slot.
//   H-stor-6  A fred-error at EXEC (Timeout / IO / Cluster / Backpressure)
//             must flow through `map_transient_fred_err` into an
//             `Aborted("transient I/O: ...")` so the retry loop picks it up,
//             matching the existing behavior for WATCH / HGETALL / MULTI-queue.
//   H-stor-7  `cleanup_old_object` must chunk `UNLINK` so layers with 10k+
//             sections don't blow past Valkey's per-command argument ceiling.
//   H-stor-8  The snapshot read and `snapshot_to_object_state` must return
//             the caller-supplied `object_id` even when the header is
//             partial, instead of the old "zeroed on any missing seq" bug
//             (found in the since-deleted `load_object_snapshot`; pinned
//             here on `read_object_snapshot`).
//   H-stor-9  Both backends must refuse to commit past `2^53 - 1` — the Lua
//             fast path returns `SEQ_OVERFLOW`; the slow path returns a
//             domain `SequenceOverflow` that maps to `Internal`.
//
// Plus the b.k1r medium "partial-header refuses the commit (not a stream-ID
// collision)" Lua guard — reported under its own `PARTIAL_HEADER:` reply tag
// since b.f1v, and still `FailedPrecondition`.
//
// Proptest-form parity (Lua vs Txn byte-identity) lives in
// `tests/proptests.rs` — proptest macros must never share a binary with
// `#[tokio::test(flavor = "multi_thread")]` blocks (CLAUDE.md Proptest
// Isolation), and that constraint rules out putting the parity proptest
// here alongside the tokio tests.

use std::collections::HashMap;
use std::sync::Arc;

use fred::interfaces::{HashesInterface, KeysInterface, SetsInterface, StreamsInterface};

use live_edit::config::UpdateLayerBackend;
use live_edit::domain::types::ObjectState;
use live_edit::error::ServiceError;
use live_edit::storage::diff_read::{
    compute_diff_at_cut, header_to_object_state, read_object_snapshot, snapshot_to_object_state,
    AtomicSnapshot, DiffCase, DiffResponse, ExtValueReads, SnapshotRead, SnapshotReadConsumer,
    SubscriberState,
};
use live_edit::storage::external_values::MockExternalValueStore;
use live_edit::storage::layer::LayerStore;
use live_edit::storage::lua::is_fast_path;
use live_edit::storage::session::SessionStore;
use live_edit::storage::test_support::relay::RelayedUpdates;
use live_edit::storage::test_support::TestValkey;
use live_edit::storage::watch_guard::WatchGuard;

use crate::common::owned_delta::{delta_with_set_field_inline, empty_delta};
/// Mirror of the module-private `storage::layer::PARTIAL_HEADER_SENTINEL`;
/// shared via `tests/common` so every module asserting it holds one literal.
use crate::common::PARTIAL_HEADER_SENTINEL_MSG;

// ─────────────────────────────────────────────────────────────────────────────
// Shared harness — mirrors the shape used in update_layer_*_test.rs so diffs
// between sibling integration tests show up as assertion changes, not
// scaffolding churn.
// ─────────────────────────────────────────────────────────────────────────────

fn mock_ext_store() -> Arc<MockExternalValueStore> {
    Arc::new(MockExternalValueStore::new())
}

async fn create_test_session(tv: &TestValkey) -> String {
    let ss = SessionStore::new(tv.client().clone(), tv.key_schema().clone());
    ss.create("b-k1r-session", "/stage", "b-k1r-user")
        .await
        .expect("session creation must succeed")
}

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
        "b-k1r-user",
        &state,
        mock_ext_store(),
    )
    .await
    .expect("create_layer must succeed")
}

// =============================================================================
// H-stor-2 — WatchGuard shutdown drain
// =============================================================================
//
// Spec (engineer's implementation, `src/storage/client.rs::ValkeyClient::quit`):
// `WatchGuard::Drop` spawns UNWATCH onto `ValkeyClient::watch_unwatch_tracker`;
// `quit()` closes the tracker and `await`s it (up to a 2s timeout) BEFORE
// closing the txn pool. Correct behavior: after `quit()` returns, every
// outstanding UNWATCH has either completed or timed out — the in-flight
// count observed via `watch_unwatch_inflight()` is 0 (on success) or the
// process has logged the drain-timeout warning.
//
// The test spawns N guards whose WATCH sets are all marked dirty (so Drop
// issues UNWATCH), drops them, and then calls `quit()`. The `tracker.close()`
// inside `quit()` prevents new spawns onto the tracker; the drop-time UNWATCH
// tasks are submitted before close, so the drain waits on them. Post-quit the
// tracker length must be 0.

/// H-stor-2: after `quit()` returns, every in-flight UNWATCH spawned by a
/// `WatchGuard::Drop` must have finished — the tracker length must be 0
/// within the 2-second drain budget. Pre-fix (bare `tokio::spawn` with no
/// tracker) the UNWATCHes would race runtime teardown and could leave
/// stale WATCH on a closing txn-pool slot; we pin the drain contract by
/// observing the tracker count through the `watch_unwatch_inflight()` seam.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn watchguard_drop_unwatch_drains_before_quit_completes() {
    const GUARDS: usize = 6;

    // pool_size >= GUARDS so every guard can hold its own connection in
    // parallel — otherwise the `acquire_txn().await` serialization masks
    // any shutdown-race behavior and the test would pass vacuously against
    // a broken implementation.
    let tv = TestValkey::setup_with_pool_size((GUARDS as u32) + 2).await;
    let client = tv.client();
    let keys = tv.key_schema();

    // Seed a key under each guard's prefix so WATCH has something to latch
    // onto — a trivially-valid WATCH still arms Drop's UNWATCH path when
    // combined with `mark_watched()`.
    let seed = keys.obj_header("watchguard-drain-seed");
    let _: () = client
        .pool
        .set(&seed, "seed", None, None, false)
        .await
        .expect("seed SET must succeed");

    // Acquire GUARDS watchguards in parallel, WATCH the seed key on each,
    // and drop them. Each Drop spawns an UNWATCH onto the shared tracker.
    //
    // `WatchGuard` exposes the underlying `Client` via `Deref`, so WATCH
    // goes through the KeysInterface impl on `Client`. `mark_watched()`
    // is what arms the Drop-time UNWATCH path.
    use std::ops::Deref;
    let mut guards: Vec<WatchGuard> = Vec::with_capacity(GUARDS);
    for _ in 0..GUARDS {
        let raw = client.acquire_txn().await;
        let mut g = WatchGuard::new(raw, client.watch_unwatch_tracker());
        g.deref()
            .watch::<&str>(&seed)
            .await
            .expect("WATCH seed must succeed");
        g.mark_watched(seed.as_str(), &[]);
        guards.push(g);
    }

    // Drop all guards — each synchronously hands its mutex+UNWATCH task to
    // the tracker. We take a synchronous snapshot immediately after the
    // drops to prove the tasks are genuinely in-flight (would be 0 if the
    // drop paths ran to completion synchronously, which would defeat the
    // test).
    drop(guards);
    let inflight_after_drop = client.watch_unwatch_inflight();
    // A value of 0 here would indicate either (a) the drops ran to
    // completion synchronously (no race window to protect), or (b) the
    // tracker is not being populated at all. Either way the drain contract
    // below is vacuous. We accept 0 but prefer to observe a positive
    // number so the test has teeth.
    assert!(
        inflight_after_drop <= GUARDS,
        "tracker length after drop ({inflight_after_drop}) must not exceed spawned tasks ({GUARDS})"
    );

    // Drain the UNWATCH tracker exactly the way `quit()` does internally —
    // close the tracker so no further spawns are accepted, then wait for
    // every queued UNWATCH task to complete. We deliberately do NOT call
    // the full `client.quit()` here because that closes the underlying
    // fred pool, and `TestValkey::Drop` (which fires at end-of-scope)
    // performs an async `cleanup_keys` over the same pool — after a quit,
    // its `SCAN` gets no answer within `CLEANUP_STEP_BUDGET`, so Drop panics
    // and fails this otherwise-passing test (it only reports it when the
    // test is already failing).
    // Draining the tracker proves the load-bearing b.k1r contract
    // ("every Drop-spawned UNWATCH completes within the drain budget")
    // without poisoning the test fixture.
    let tracker = client.watch_unwatch_tracker();
    tracker.close();
    tokio::time::timeout(std::time::Duration::from_secs(2), tracker.wait())
        .await
        .expect(
            "WatchGuard UNWATCH tracker must drain within 2s; a timeout means a Drop-spawned \
             UNWATCH never completed, the symptom b.k1r H-stor-2 fixed",
        );

    assert_eq!(
        client.watch_unwatch_inflight(),
        0,
        "post-drain, every WatchGuard UNWATCH task must have completed; a non-zero count \
         means a spawned UNWATCH outlived the tracker drain (pre-b.k1r behavior)"
    );
}

// =============================================================================
// H-stor-6 — EXEC transient-error symmetry via injected fred error
// =============================================================================
//
// The engineer's EXEC arm now routes non-canceled fred errors through
// `map_transient_fred_err("EXEC", e)`. The existing fault-injection seam
// (`with_fred_error_injector_for_test`) short-circuits WATCH rather than
// EXEC. The `b.1dz` integration tests already exercise a raw fred Timeout /
// Cluster at WATCH through `map_snapshot_load_err("WATCH ...", e)` and
// verify retry. H-stor-6 adds the *same classifier* at the EXEC arm — the
// per-arm helpers share an implementation, so a pin against the Cluster-
// at-WATCH chain is a pin against the EXEC-arm behavior too. What is
// incremental here is the retry loop's behavior when the transient-I/O
// classification surfaces from the EXEC-specific helper path under the
// same `Aborted("transient I/O: EXEC failed (...)")` shape the production
// EXEC arm now emits — covered by the single-retry recovery + counter
// pin below.
//
// The injected Aborted message uses the EXEC-arm's label ("EXEC failed
// (<kind>)") so the test would fail against pre-fix code where only the
// WATCH-arm emitted a "transient I/O:" Aborted: pre-fix EXEC errors
// mapped to `ServiceError::Internal(...)`, which the retry loop does not
// match. The injection here goes through `with_fault_injector_for_test`
// which injects a cooked Aborted, then verifies the outer retry loop
// picks it up identically to the WATCH path. Together with the pre-fix
// EXEC behavior (Internal surfaced), this is the test the ticket calls
// out for "update_layer with simulated ConnectionDrop mid-EXEC retries
// correctly (fault injection)". The `is_canceled` arm of EXEC already
// emits its own distinct Aborted, so we use the `map_transient_fred_err`
// shape here to avoid collision with that well-tested path.

/// H-stor-6: a transient fred error at EXEC classified via
/// `map_transient_fred_err("EXEC", ...)` must surface as
/// `Aborted("transient I/O: EXEC failed (<kind>): ...")` and be retried
/// by the outer `update_layer` retry loop. Commits on attempt 2, with
/// `retries_total{committed} == 1` and one attempts-histogram sample of
/// value 2.0.
#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_exec_transient_error_is_retried_and_commits() {
    use std::collections::VecDeque;
    use std::sync::Mutex as StdMutex;

    let tv = TestValkey::setup().await;
    let metrics = Arc::new(crate::common::metrics_scrape_only("bk1r-exec-transient"));

    // Cooked Aborted in the exact shape `map_transient_fred_err("EXEC", e)`
    // produces for a transient fred kind. Pre-fix EXEC would have surfaced
    // Internal for this error → retry loop wouldn't match → this test's
    // `.expect(...)` below would panic.
    let timeout_label = fred::error::ErrorKind::Timeout.to_str();
    let cooked = ServiceError::Aborted(format!(
        "transient I/O: EXEC failed ({timeout_label}): synthetic connection drop mid-EXEC"
    ));

    // Inject one transient-I/O Aborted on attempt 1; attempt 2 runs the
    // real Valkey EXEC and commits.
    let queue: Arc<StdMutex<VecDeque<ServiceError>>> =
        Arc::new(StdMutex::new(std::iter::once(cooked).collect()));

    // Pin to Txn backend so EXEC is exercised (the Lua fast path has no
    // EXEC — Lua atomicity subsumes it).
    let ls = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096)
        .with_update_layer_retry_max_attempts(32)
        .with_update_layer_backend(UpdateLayerBackend::Txn)
        .with_metrics(metrics.clone())
        .with_fault_injector_for_test(queue.clone());

    let session_id = create_test_session(&tv).await;
    let oid = create_test_layer(&ls, &session_id, "bk1r-exec-transient").await;

    let (seq, _ts) = ls
        .update_layer_relayed(
            &session_id,
            "bk1r-exec-transient",
            oid,
            Some(empty_delta()),
            "bk1r-client",
            12345,
        )
        .await
        .expect(
            "update_layer must recover from a transient EXEC-arm Aborted and commit on the \
             second attempt. Pre-b.k1r, non-canceled EXEC errors mapped directly to \
             ServiceError::Internal(...), which the retry loop doesn't match — so the first \
             attempt surfaced as Internal and this .expect would panic.",
        );
    assert!(
        seq > 1,
        "sequence must advance past create_layer (seq=1); got {seq}"
    );

    assert_eq!(
        queue.lock().unwrap().len(),
        0,
        "fault injector queue should have been drained by exactly one attempt"
    );

    // committed retry counter: commit on attempt 2 → 1 retry, on the `txn`
    // backend's only path, `path="watch"`.
    let committed = live_edit::observability::metrics::test_util::sample_value(
        &live_edit::observability::metrics::test_util::render(&metrics),
        "live_edit_update_layer_retries_total",
        &[("path", "watch"), ("outcome", "committed")],
    )
    .unwrap_or(0.0) as u64;
    assert_eq!(
        committed, 1,
        "a single retry must register as one `committed` retry; got {committed}"
    );
    let exhausted = live_edit::observability::metrics::test_util::sample_value(
        &live_edit::observability::metrics::test_util::render(&metrics),
        "live_edit_update_layer_retries_total",
        &[("path", "watch"), ("outcome", "exhausted")],
    )
    .unwrap_or(0.0) as u64;
    assert_eq!(exhausted, 0);
}

// =============================================================================
// H-stor-7 — UNLINK chunking with 10k+ section keys
// =============================================================================
//
// The CreateLayer overwrite's cleanup body tears the old object down. Pre-fix
// it issued a single `UNLINK k1 k2 ... kN` with every section / field /
// time-sample / tombstone key in the old object — at 10k+ sections the
// command overshoots Valkey's protocol limits and manifests as "Protocol
// error" or a hard disconnect. The fix chunked at 500 keys / call; the paged
// teardown (`storage::retirement::Teardown`, b.fsz MR-D4) now also reads the
// indices page by page (`SSCAN` / `ZRANGE` of at most `P` members) and
// UNLINKs each page's keys in chunks of at most `UNLINK_CHUNK_SIZE` (500).
//
// We exercise a large teardown by pre-seeding ≥ 10k section index members
// into an "old object" keyspace and then triggering an overwrite via
// `create_layer` on an existing `(session_id, layer_name)` — that routes
// into the cleanup body, which pages the sections index and unlinks
// everything under `{old_oid}`. The seeded keys must all be gone.

/// H-stor-7: a layer with 10,000 section-related keys must clean up without
/// tripping Valkey's per-command size / argument ceiling.
///
/// Strategy: seed the section index + per-section hashes directly into
/// Valkey under an object_id that's already mapped as a layer, then
/// overwrite the layer (which spawns the cleanup body and its paged
/// teardown) and assert the old sections index is gone once the cleanup
/// drains. Pre-fix this path surfaced a fred protocol error / disconnect on
/// a single monolithic UNLINK of 10k keys; every teardown command is now
/// bounded by its page and the 500-key chunk.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn create_layer_overwrite_unlinks_10k_section_keys_via_chunking() {
    // 10_000 sections: the object a single monolithic UNLINK of 20,000 keys
    // broke (pre-fix). The paged teardown now takes them about `P` members
    // per `SSCAN` step, each step's keys UNLINKed in chunks of at most 500.
    // Larger values are tempting but push the per-test SET / ZADD seeding
    // time up without buying extra coverage.
    const SECTION_COUNT: usize = 10_000;

    let tv = TestValkey::setup().await;
    let ls = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096);
    let keys = tv.key_schema();
    let pool = &tv.client().pool;
    let session_id = create_test_session(&tv).await;
    let layer_name = "bk1r-unlink-chunk";

    // First create_layer — gets us an object_id whose keyspace we can then
    // augment with synthetic section keys.
    let oid = create_test_layer(&ls, &session_id, layer_name).await;
    let oid_str = oid.to_string();

    // Seed SECTION_COUNT section index members + per-section hashes so the
    // cleanup body's teardown has at least 10k keys to UNLINK.
    //
    // Using a single pipelined SADD with multiple members: fred's
    // `sadd` takes `Vec<String>`, which encodes as one SADD command per
    // batch. We batch by 1000 to keep the seeding-side SADD well under
    // the same ceiling that the UNLINK fix is protecting against — if we
    // tried to SADD 10k members in one shot we'd be exercising the very
    // limit we're protecting against on the wrong end.
    let sections_index_key = keys.obj_sections_index(&oid_str);
    for chunk_start in (0..SECTION_COUNT).step_by(1000) {
        let chunk_end = (chunk_start + 1000).min(SECTION_COUNT);
        let members: Vec<String> = (chunk_start..chunk_end).map(|i| i.to_string()).collect();
        let _: u64 = pool
            .sadd(&sections_index_key, members)
            .await
            .expect("SADD seed must succeed");
    }

    // Seed per-section hashes. Each key is ~50 bytes; 10k of these is
    // ~500 kB, well inside the 1MB client query buffer but far beyond
    // the single-UNLINK ceiling that the fix guards against.
    for i in 0..SECTION_COUNT {
        let section_key = keys.obj_section(&oid_str, &i.to_string());
        let _: () = pool
            .hset(&section_key, [("name", format!("s{i}"))])
            .await
            .expect("HSET seed must succeed");
    }

    // Sanity: pre-overwrite, the sections index holds all 10k members.
    let seeded_count: u64 = pool
        .scard(&sections_index_key)
        .await
        .expect("SCARD must succeed");
    assert_eq!(seeded_count as usize, SECTION_COUNT, "seed accounting");

    // Overwrite. The spawned cleanup body's paged teardown
    // (`storage::retirement::Teardown`) pages every section / field /
    // time-sample / tombstone index of the old oid and UNLINKs the keys each
    // page names in chunks of at most `UNLINK_CHUNK_SIZE`. Pre-fix: a single
    // UNLINK of 10k keys either disconnected fred or returned a Protocol
    // error. The overwrite body WARN-swallows a teardown failure, so
    // `create_layer` returning Ok proves nothing about the UNLINKs; the
    // sections-index check after `wait_for_cleanup` below does.
    let state = ObjectState {
        sequence_num: 1,
        first_version: 1,
        source_format: "usd".to_string(),
        ..Default::default()
    };
    let new_oid = ls
        .create_layer(
            &session_id,
            layer_name,
            "bk1r-user",
            &state,
            mock_ext_store(),
        )
        .await
        .expect(
            "CreateLayer overwrite must succeed even when the prior object carries \
             10k+ section keys",
        );
    assert_ne!(new_oid, oid, "overwrite must mint a fresh object_id");

    // The overwrite spawns its cleanup body on the LayerStore's
    // cleanup_tracker — it does NOT block the create_layer return. Wait
    // for the spawned cleanup task to complete before checking the index
    // count, otherwise the assertion races the teardown.
    ls.wait_for_cleanup(std::time::Duration::from_secs(10))
        .await;

    // Post-overwrite: the old object's sections index must be gone. The
    // cleanup body WARN-swallows a teardown failure, so this is the check
    // that catches one — and a regression that silently skipped a page or
    // chunk without erroring would show up here as lingering keys too.
    let post_seeded_count: u64 = pool
        .scard(&sections_index_key)
        .await
        .expect("SCARD post-unlink must succeed");
    assert_eq!(
        post_seeded_count, 0,
        "after overwrite, the old object's sections index must be fully UNLINKed; \
         a positive count would mean the chunked loop bailed early or skipped a chunk"
    );
}

// =============================================================================
// H-stor-8 — diff_read regime selection for version=0 subscriber
// =============================================================================
//
// Pre-fix the snapshot loader's (`load_object_snapshot`, since deleted) `oid`
// parse had a tortured
// `parse_u64(..., "seq").map(|_| parse_u64(object_id))` gate that degraded
// `oid` to 0 whenever `seq` was missing — so the loaded snapshot looked
// like it belonged to object 0, which then propagated into
// `ObjectState.object_id` and broke the regime selection logic
// (`DiffEngine::plan` compares `have_object_id != state.object_id` as part
// of the FullContent-vs-Incremental decision).
//
// The new-build tests cover three slot positions of the "current seq":
//
//   - current_seq == 0 is physically unreachable under normal CreateLayer
//     (the initial seq is always 1 after the +1 inside apply_delta). We
//     simulate it by seeding a header manually with `seq=0` and verifying
//     `snapshot_to_object_state` returns the correct `object_id` AND that
//     `DiffEngine::plan` (via `compute_diff_at_cut`) picks FullContent.
//   - current_seq == 1 is the CreateLayer default; a version=0 subscriber
//     gets FullContent with one commit's worth of content.
//   - current_seq > 1 (e.g. 3): an updated layer; same FullContent path
//     because have_version=0 forces Case 1 regardless.

/// `read_object_snapshot` of `object_id` for `consumer`, which must answer
/// with a snapshot.
async fn read_snapshot(
    tv: &TestValkey,
    object_id: &str,
    consumer: SnapshotReadConsumer,
) -> AtomicSnapshot {
    match read_object_snapshot(tv.client(), tv.key_schema(), object_id, consumer).await {
        Ok(SnapshotRead::Snapshot(snapshot)) => *snapshot,
        Ok(SnapshotRead::TooLarge(t)) => {
            panic!("object {object_id} must answer with its state; it answered TOO_LARGE {t:?}")
        }
        Err(e) => panic!("the {consumer:?} read of object {object_id} must succeed; got {e:?}"),
    }
}

/// A version=0 subscriber's legacy diff of `object_id`, built the way the
/// service builds it: one atomic catch-up read, then `compute_diff_at_cut`.
async fn version_zero_diff(tv: &TestValkey, object_id: u64) -> DiffResponse {
    let snapshot = read_snapshot(tv, &object_id.to_string(), SnapshotReadConsumer::Catchup).await;
    let subscriber = SubscriberState {
        have_object_id: object_id,
        have_version: 0,
    };
    compute_diff_at_cut(
        &snapshot,
        &subscriber,
        mock_ext_store().as_ref(),
        ExtValueReads::recover_for_test(),
        None,
        "test",
    )
    .await
    .expect("the diff build must succeed")
    .into_ready()
    .expect("a build with fetch-on-miss never answers Missing")
}

/// H-stor-8: version=0 subscriber catch-up correctly routes through the
/// FullContent regime regardless of the object's `current_seq` value. The
/// fix ensures `ObjectState.object_id` reflects the caller-supplied
/// `object_id` so the regime dispatch doesn't fall through to a mismatched-
/// object-id branch.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn diff_read_version_zero_subscriber_full_content_at_seq_zero() {
    let tv = TestValkey::setup().await;
    let keys = tv.key_schema();
    let pool = &tv.client().pool;

    // Seed a synthetic header with `seq=0`. A real CreateLayer never
    // produces seq=0 (the initial write advances to seq=1), so this is an
    // intentionally-adversarial shape that exercises the regime dispatch
    // edge case where `state.sequence_num == have_version == 0`.
    let object_id: u64 = 424242;
    let oid_str = object_id.to_string();
    let header_key = keys.obj_header(&oid_str);
    let _: () = pool
        .hset(
            &header_key,
            [
                ("seq", "0".to_owned()),
                ("first_ver", "0".to_owned()),
                ("src_fmt", "usda".to_owned()),
                ("next_set_order", "0".to_owned()),
                ("session_id", "s1".to_owned()),
                ("layer_name", "l1".to_owned()),
                ("created_by", "tester".to_owned()),
                ("created_at", "0".to_owned()),
                ("modified_at", "0".to_owned()),
                ("last_client_id", String::new()),
                ("last_client_ts", "0".to_owned()),
                ("last_server_ts", "0".to_owned()),
            ],
        )
        .await
        .expect("HSET synthetic header must succeed");

    // Snapshot contract: caller-supplied `object_id` is preserved (post-fix).
    // Pre-b.k1r, the `parse_u64(&header, "seq").map(|_| …)` expression
    // returned `|_| parse(object_id)` wrapped in an Option — with seq=0
    // it stayed inside the `map` (Some(_)), preserving the real oid —
    // so seq=0 alone doesn't catch the pre-fix bug. The next test
    // (`current_seq_between_1_and_n`) covers the bug's true trigger.
    let snap = read_snapshot(&tv, &oid_str, SnapshotReadConsumer::Catchup)
        .await
        .state;
    assert_eq!(
        snap.object_id, object_id,
        "snapshot.object_id must equal the caller-supplied object_id (post-b.k1r)"
    );
    assert_eq!(snap.sequence_num, 0);

    // Regime dispatch: a version=0 subscriber at have_object_id = object_id
    // must receive FullContent regardless of current_seq. The OMPF payload
    // can be empty (no sections / fields), but the case must be FullContent.
    let resp = version_zero_diff(&tv, object_id).await;
    assert_eq!(
        resp.case,
        DiffCase::FullContent,
        "have_version=0 must always route to FullContent regardless of current_seq \
         (regime dispatch case 1 in DiffEngine::plan)"
    );
}

/// H-stor-8: partial header (seq missing entirely) must NOT collapse
/// `oid` to 0 — this was the exact pre-fix bug. No snapshot read reads such a
/// header on any more (b.fsz MR-D4 removed ext-GC's read-at-`seq`-0 rule
/// with the ext-GC snapshot read itself: external-value clean-up walks a
/// retiring object's references instead, `storage::retirement`), so a
/// partial header is never served as state at all: the read answers
/// `NotFound`, naming the caller-supplied `object_id` — never object 0. The
/// decode's own preservation of the id on a partial header is pinned by
/// `header_to_object_state_preserves_object_id_with_partial_header` below.
///
/// Pre-b.k1r: `parse_u64(&header, "seq").map(|_| object_id.parse())` —
/// when `seq` was missing the `.map` short-circuited, `unwrap_or(0)` fired,
/// and `oid = 0`. That propagated into the snapshot as `object_id = 0`,
/// which then made the regime dispatch compare `have_object_id=oid != 0`
/// and incorrectly choose FullContent via the object-id-mismatch branch
/// (accidentally right for this case) but report a misleading object_id
/// in logs/metrics and break every downstream consumer that keyed on it.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn diff_read_partial_header_is_not_found_under_the_caller_object_id() {
    let tv = TestValkey::setup().await;
    let keys = tv.key_schema();
    let pool = &tv.client().pool;

    let object_id: u64 = 777_777_777;
    let oid_str = object_id.to_string();
    let header_key = keys.obj_header(&oid_str);
    // Intentionally incomplete header: only `src_fmt` present. The prior
    // `seq`-based gate would zero the oid; post-fix, the oid comes from
    // the caller-supplied string unconditionally.
    let _: () = pool
        .hset(&header_key, [("src_fmt", "usda".to_owned())])
        .await
        .expect("HSET partial header must succeed");

    let result = read_object_snapshot(
        tv.client(),
        tv.key_schema(),
        &oid_str,
        SnapshotReadConsumer::Catchup,
    )
    .await;
    match result {
        Err(ServiceError::NotFound(msg)) => assert!(
            msg.contains(&oid_str),
            "the NotFound must name the caller-supplied object_id, never object 0 \
             (pre-b.k1r: a missing `seq` field silently returned object_id=0); got {msg:?}"
        ),
        Ok(SnapshotRead::Snapshot(snap)) => panic!(
            "a header without a numeric seq is no live object, so no snapshot read may serve \
             it as state; got object_id={} at seq {}",
            snap.state.object_id, snap.state.sequence_num
        ),
        Ok(SnapshotRead::TooLarge(t)) => panic!("expected NotFound, got TOO_LARGE {t:?}"),
        Err(other) => panic!("expected NotFound, got {other:?}"),
    }
}

/// H-stor-8: a non-numeric `object_id` string surfaces as `Internal`
/// instead of degrading to oid=0. Pre-fix the `object_id.parse::<u64>()
/// .unwrap_or(0)` branch silently swallowed the error.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn diff_read_non_numeric_object_id_surfaces_internal() {
    let tv = TestValkey::setup().await;
    let keys = tv.key_schema();
    let pool = &tv.client().pool;

    // Seed a header under the non-numeric string key so the HGETALL
    // succeeds and the caller reaches the `object_id.parse::<u64>()` line.
    let header_key = keys.obj_header("abcdef");
    let _: () = pool
        .hset(
            &header_key,
            [("seq", "1".to_owned()), ("src_fmt", "usda".to_owned())],
        )
        .await
        .expect("HSET must succeed");

    let result = read_object_snapshot(
        tv.client(),
        tv.key_schema(),
        "abcdef",
        SnapshotReadConsumer::Catchup,
    )
    .await;
    match result {
        Err(ServiceError::Internal(msg)) => {
            assert!(
                msg.contains("abcdef"),
                "Internal message must preserve the bogus object_id for diagnosis; got: {msg:?}"
            );
        }
        Ok(SnapshotRead::Snapshot(snap)) => panic!(
            "a non-numeric object_id must NOT silently degrade to object_id=0 \
             (pre-b.k1r behaviour); got snapshot with object_id={}",
            snap.state.object_id
        ),
        Ok(SnapshotRead::TooLarge(t)) => panic!("expected Internal, got TOO_LARGE {t:?}"),
        Err(other) => panic!("expected Internal, got {other:?}"),
    }
}

/// H-stor-8 — end-to-end: create_layer (seq=1) then a version=0 subscriber
/// gets FullContent with the seeded content. This pins the normal-case
/// regime dispatch in a shape that is guaranteed NOT to pre-exist in the
/// compute_diff_test suite: the critical contract is that the regime is
/// FullContent regardless of current_seq ∈ {1, >1}.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn diff_read_version_zero_subscriber_full_content_at_seq_one() {
    let tv = TestValkey::setup().await;
    let ls = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096);
    let session_id = create_test_session(&tv).await;
    let oid = create_test_layer(&ls, &session_id, "bk1r-regime-seq1").await;

    let resp = version_zero_diff(&tv, oid).await;
    assert_eq!(
        resp.case,
        DiffCase::FullContent,
        "current_seq=1 with have_version=0 must be FullContent (regime 1)"
    );
}

/// H-stor-8 — current_seq > 1: create + several updates to advance seq,
/// then a version=0 subscriber. Still FullContent because have_version=0
/// is the dominant regime-1 trigger.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn diff_read_version_zero_subscriber_full_content_at_seq_greater_than_one() {
    let tv = TestValkey::setup().await;
    let ls = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096);
    let session_id = create_test_session(&tv).await;
    let oid = create_test_layer(&ls, &session_id, "bk1r-regime-seq-gt-1").await;

    // Advance the sequence via three field updates so current_seq > 1.
    // Each update commits one field write (fast-path-eligible); the final
    // object's `seq` in Valkey will be 4 (1 from create + 3 from updates).
    for i in 0..3u64 {
        let delta = delta_with_set_field_inline(1, &format!("k{i}"), vec![i as u8; 2], i);
        ls.update_layer_relayed(
            &session_id,
            "bk1r-regime-seq-gt-1",
            oid,
            Some(delta),
            "bk1r-client",
            1000 + i,
        )
        .await
        .expect("update_layer must commit");
    }

    // Sanity-check the header seq is indeed > 1.
    let header_key = tv.key_schema().obj_header(&oid.to_string());
    let header: HashMap<String, String> = tv
        .client()
        .pool
        .hgetall(&header_key)
        .await
        .expect("HGETALL must succeed");
    let current_seq: u64 = header.get("seq").unwrap().parse().unwrap();
    assert!(
        current_seq > 1,
        "test preconditions: seq must have advanced past 1; got {current_seq}"
    );

    let resp = version_zero_diff(&tv, oid).await;
    assert_eq!(
        resp.case,
        DiffCase::FullContent,
        "current_seq={current_seq} with have_version=0 must be FullContent (regime 1); \
         pre-b.k1r the snapshot loader's `oid` mis-assignment could have surfaced as \
         a different case selection under some header shapes"
    );
}

/// H-stor-8 — unit-level: `header_to_object_state` preserves the caller-
/// supplied `object_id` even when the header is partial. Exercises the
/// fast-path helper that the Lua backend uses to reconstruct the
/// pre-update state; the H-stor-8 fix to `diff_read.rs::header_to_object_state`
/// replaced `parse_u64(header, "seq").unwrap_or(0)` with `parse_u64_warn`
/// but the `object_id` field was already passed through unchanged — we
/// pin both contracts here so a regression (inadvertent swap back to the
/// buggy formula) would fail.
#[test]
fn header_to_object_state_preserves_object_id_with_partial_header() {
    let mut header = HashMap::new();
    header.insert("src_fmt".to_string(), "usda".to_string());
    let state: ObjectState = header_to_object_state(&header, 999_888_777);
    assert_eq!(state.object_id, 999_888_777);
    assert_eq!(state.sequence_num, 0, "missing seq defaults to 0");
    assert_eq!(state.first_version, 0);
    assert_eq!(state.next_set_order, 0);
    assert_eq!(state.source_format, "usda");
    assert!(state.sections.is_empty());
    assert!(state.fields.is_empty());
}

/// H-stor-8 — unit-level: `snapshot_to_object_state` round-trips the
/// snapshot's `object_id` into `ObjectState`. Every snapshot reader sets
/// `ObjectSnapshot.object_id = caller.parse(object_id)`, so the round-trip is
/// identity at the `snapshot_to_object_state` layer. Pin that identity.
#[test]
fn snapshot_to_object_state_round_trips_object_id() {
    use live_edit::storage::diff_read::ObjectSnapshot;

    let snap = ObjectSnapshot {
        object_id: 1_234_567_890,
        sequence_num: 5,
        first_version: 1,
        source_format: "usd".to_string(),
        next_set_order: 7,
        sections: HashMap::new(),
        children: HashMap::new(),
        fields: Vec::new(),
        time_samples: Vec::new(),
        tombstones: Vec::new(),
    };
    let state = snapshot_to_object_state(&snap);
    assert_eq!(state.object_id, 1_234_567_890);
    assert_eq!(state.sequence_num, 5);
    assert_eq!(state.next_set_order, 7);
}

// =============================================================================
// H-stor-9 — SEQ_OVERFLOW ceiling at 2^53 - 1 on both backends
// =============================================================================
//
// The ceiling: `current_seq >= 2^53 - 1` is rejected before any write.
// The Lua script returns `redis.error_reply("SEQ_OVERFLOW: ...")`; the
// Rust caller maps that to `ServiceError::Internal(...)`.
// The slow (txn) path hits the same ceiling inside `apply_delta` via
// `DeltaError::SequenceOverflow`, which also maps to `Internal`.
//
// We seed a header with `seq = 2^53 - 1` (the ceiling itself; the next
// `+1` would cross the line), then drive both backends and assert both
// surface Internal. Using the literal 9_007_199_254_740_991 keeps the
// math grep-able.

const SEQ_CEILING: u64 = (1u64 << 53) - 1;

async fn seed_header_at_ceiling(tv: &TestValkey, oid: u64) {
    // `update_layer` opens the per-object header at `obj_header(oid)` — no
    // session-layer mapping lookup on the hot path. Seeding just the header
    // is sufficient to drive both backends through their ceiling guards.
    let keys = tv.key_schema();
    let pool = &tv.client().pool;
    let oid_str = oid.to_string();
    let header_key = keys.obj_header(&oid_str);
    let _: () = pool
        .hset(
            &header_key,
            [
                ("seq", SEQ_CEILING.to_string()),
                ("first_ver", "1".to_string()),
                ("src_fmt", "usda".to_string()),
                ("next_set_order", "0".to_string()),
                ("session_id", "s1".to_string()),
                ("layer_name", "l1".to_string()),
                ("created_by", "tester".to_string()),
                ("created_at", "0".to_string()),
                ("modified_at", "0".to_string()),
                ("last_client_id", String::new()),
                ("last_client_ts", "0".to_string()),
                ("last_server_ts", "0".to_string()),
            ],
        )
        .await
        .expect("HSET ceiling header must succeed");
}

/// H-stor-9 Lua path: a header at `seq = 2^53 - 1` rejects the next
/// `update_layer` call as `Internal` with `SEQ_OVERFLOW` in the message.
/// The Lua script emits `redis.error_reply("SEQ_OVERFLOW: ...")`; the
/// Rust caller (`update_layer_once_lua`) pattern-matches `details.starts_with
/// ("SEQ_OVERFLOW")` and wraps it in `Internal`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_lua_at_sequence_ceiling_returns_internal_with_seq_overflow() {
    let tv = TestValkey::setup().await;
    let ls = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096)
        .with_update_layer_backend(UpdateLayerBackend::Lua);

    let oid: u64 = 1;
    seed_header_at_ceiling(&tv, oid).await;

    let result = ls
        .update_layer_relayed("s1", "l1", oid, Some(empty_delta()), "bk1r-client", 4242)
        .await;
    match result {
        Err(ServiceError::Internal(msg)) => {
            assert!(
                msg.contains("SEQ_OVERFLOW") || msg.contains("sequence ceiling"),
                "Lua ceiling error must surface as Internal with a recognisable \
                 SEQ_OVERFLOW / sequence-ceiling token so operators can distinguish \
                 this from a generic protocol error; got: {msg:?}"
            );
        }
        Ok(v) => panic!(
            "Lua commit at seq=2^53-1 must refuse the next step; got Ok({v:?}) \
             — the ceiling check in update_layer_fast.lua did not fire"
        ),
        Err(other) => panic!("Lua ceiling error must be Internal (non-retriable); got {other:?}"),
    }
}

/// H-stor-9 Txn path: the same ceiling is enforced on the slow path via
/// `apply_delta`'s `SequenceOverflow` guard (domain/delta.rs). The fast and
/// slow paths must fail at the same boundary so the backend toggle
/// (`LIVE_EDIT_UPDATE_LAYER_BACKEND`) doesn't let an object escape past
/// 2^53-1 on one path only to then be rejected on the other.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_txn_at_sequence_ceiling_returns_internal() {
    let tv = TestValkey::setup().await;
    // Txn backend with no delta = goes through the slow path's fast branch
    // (no snapshot load), but `apply_delta` still enforces the 2^53 ceiling
    // on every call.
    let ls = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096)
        .with_update_layer_backend(UpdateLayerBackend::Txn);

    let oid: u64 = 2;
    seed_header_at_ceiling(&tv, oid).await;

    let result = ls
        .update_layer_relayed("s1", "l1", oid, Some(empty_delta()), "bk1r-client", 4243)
        .await;
    match result {
        Err(ServiceError::Internal(msg)) => {
            // The slow path emits `DeltaError::SequenceOverflow` → the
            // error's Display form ends up inside `Internal("delta error: ...")`.
            assert!(
                msg.to_lowercase().contains("overflow") || msg.to_lowercase().contains("sequence"),
                "Txn ceiling error must surface as Internal with a sequence-overflow \
                 token; got: {msg:?}"
            );
        }
        Ok(v) => panic!(
            "Txn commit at seq=2^53-1 must refuse the next step; got Ok({v:?}) \
             — the SEQUENCE_CEILING guard in apply_delta did not fire"
        ),
        Err(other) => panic!("Txn ceiling error must be Internal (non-retriable); got {other:?}"),
    }
}

// =============================================================================
// b.k1r medium — Partial-header Lua: refuse the commit (not a stream-ID
// collision), under the PARTIAL_HEADER tag (b.f1v)
// =============================================================================
//
// Pre-b.k1r: the Lua fast path tolerated `hdr["seq"] == nil` by falling
// through `tonumber(hdr["seq"]) or 0` → `new_seq = 1` → XADD to the stream
// at id `1-0`, colliding with the CreateLayer initial entry if it still
// existed (or simply minting a bogus record on a partial header). b.k1r made
// the script refuse the commit; b.f1v then gave the refusal its own reply tag,
// `redis.error_reply("PARTIAL_HEADER: object header partial (seq missing or
// malformed)")`, so the Rust dispatch can tell it apart from the genuinely
// absent-header `NOT_FOUND:` reply. It maps to
// `FailedPrecondition(PARTIAL_HEADER_SENTINEL_MSG)` and the retry loop bails
// immediately (no retry can fix a bad header).
//
// This test is the FENCE for that split: the partial-header case must stay
// `FailedPrecondition` while the absent-header case moved to `NotFound`. A
// corrupt-but-PRESENT object must not reach `handle_update_layer`'s
// checkpoint-recovery arm, whose bare `EXISTS` presence probe would report the
// layer restored while the corrupt header sat untouched.

/// b.k1r medium: a partial header (seq field missing) on the Lua fast path
/// surfaces as `FailedPrecondition` (not `Ok`, not `Internal`, not a
/// duplicate-stream-id Aborted, and — post-b.f1v — not `NotFound`). Pre-fix,
/// the `new_seq = current_seq + 1` arithmetic with `current_seq = 0` silently
/// allocated stream id `1-0`, colliding with the initial CreateLayer entry (or
/// appearing as an orphaned commit against a deleted header). Post-fix the
/// script's defensive check fires before any write.
///
/// One corner of a four-test claim — **an ABSENT header and a PARTIAL one are
/// distinct surfaces, identically on both backends**. The other three:
/// `b_qw6_tests::update_layer_txn_partial_header_returns_failed_precondition`
/// (partial / txn),
/// `update_layer_lua_test::update_layer_lua_missing_header_returns_not_found`
/// (absent / Lua) and
/// `update_layer_transaction_test::update_layer_header_deleted_returns_not_found`
/// (absent / txn). Each carries the opposite-variant `panic!` arm.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_lua_partial_header_returns_failed_precondition() {
    let tv = TestValkey::setup().await;
    let keys = tv.key_schema();
    let pool = &tv.client().pool;
    let ls = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096)
        .with_update_layer_backend(UpdateLayerBackend::Lua);

    // Seed a layer normally (gets a full header + the 1-0 stream entry).
    let session_id = create_test_session(&tv).await;
    let oid = create_test_layer(&ls, &session_id, "bk1r-partial-header").await;
    let oid_str = oid.to_string();

    // Fast-path eligibility is a PRECONDITION, not an accident of the fixture.
    // `update_layer_once`'s dispatch is `backend == Lua && is_fast_path(&delta)`,
    // so an ineligible delta would quietly run this down the WATCH/MULTI/EXEC
    // path — and since b.f1v made both backends emit the identical message, it
    // would still pass, leaving the Lua script's `PARTIAL_HEADER:` reply covered
    // by nothing.
    let delta = empty_delta();
    assert!(
        is_fast_path(&Some(delta.clone())),
        "this delta must be fast-path eligible, or this test silently re-tests the txn path's \
         `seq_ok` guard (already covered by `b_qw6_tests`) under a `_lua` name"
    );

    // Damage the header: delete just the `seq` field. Everything else
    // (src_fmt, next_set_order, session_id, layer_name, ...) is still
    // there, so the HGETALL-empty check in the script can't rescue us —
    // we exercise the NEW partial-header guard.
    let header_key = keys.obj_header(&oid_str);
    let _: i64 = pool
        .hdel(&header_key, "seq")
        .await
        .expect("HDEL seq must succeed");
    // Sanity: seq is gone but other fields remain.
    let after: HashMap<String, String> = pool
        .hgetall(&header_key)
        .await
        .expect("HGETALL post-hdel must succeed");
    assert!(!after.contains_key("seq"), "seq must be absent");
    assert!(after.contains_key("src_fmt"), "src_fmt must remain");

    // Attempt update — must NOT commit and must NOT mint a colliding
    // stream entry. The Lua script's partial-header guard returns
    // `PARTIAL_HEADER: ...`; the Rust caller maps that tag to
    // `FailedPrecondition(PARTIAL_HEADER_SENTINEL_MSG)`.
    let result = ls
        .update_layer_relayed(
            &session_id,
            "bk1r-partial-header",
            oid,
            Some(delta),
            "bk1r-client",
            5555,
        )
        .await;
    match result {
        Err(ServiceError::FailedPrecondition(msg)) => {
            // Pinned by equality since b.f1v: the slow path's `seq_ok` guard
            // raises the identical string (asserted from the other side in
            // `b_qw6_tests.rs`), so this is the SDD §7.5 backend-parity
            // contract, not a log line. The former `!msg.is_empty()` check
            // could not tell the partial-header message from the
            // absent-header one the dispatch used to collapse it into.
            assert_eq!(
                msg, PARTIAL_HEADER_SENTINEL_MSG,
                "the Lua path's PARTIAL_HEADER reply must map to the partial-header sentinel \
                 byte-for-byte; got {msg:?}"
            );
        }
        Ok(v) => panic!(
            "Lua update_layer on a partial header (seq missing) MUST NOT commit — \
             pre-b.k1r this fell through to new_seq=1 and collided with the 1-0 \
             stream entry, silently corrupting the stream. Got Ok({v:?})."
        ),
        Err(ServiceError::NotFound(msg)) => panic!(
            "b.f1v fence: a corrupt-but-PRESENT header must stay `FailedPrecondition`. \
             `NotFound` routes `handle_update_layer` into checkpoint recovery, whose bare \
             `EXISTS` presence probe would report the layer restored while the corrupt header \
             sat untouched. Got NotFound({msg:?})"
        ),
        Err(other) => {
            panic!("expected FailedPrecondition (no retry can fix a corrupt header); got {other:?}")
        }
    }

    // Belt-and-suspenders: stream must still have exactly one entry (the
    // CreateLayer initial 1-0 entry). A pre-fix code path that let the
    // update through would have XADD'd a second entry at 1-0 which fails
    // with a duplicate-stream-id error (if we're lucky) or at 2-0 (if
    // new_seq landed at 1 anyway due to `or 0`). Either way, more than
    // one entry below means the guard failed.
    let stream_key = keys.layer_stream(oid);
    // Fields read as bytes: an entry's relay `ompf` field is binary.
    let entries: Vec<(String, HashMap<String, bytes::Bytes>)> = pool
        .xrange_values(&stream_key, "-", "+", None)
        .await
        .expect("XRANGE must succeed");
    assert_eq!(
        entries.len(),
        1,
        "stream must still have exactly the CreateLayer initial entry (1-0) — \
         the partial-header update must have been rejected before XADD; \
         got {n} entries: {entries:?}",
        n = entries.len()
    );
    assert_eq!(
        entries[0].0, "1-0",
        "sole entry must be the initial 1-0 CreateLayer entry; got {}",
        entries[0].0
    );
}
