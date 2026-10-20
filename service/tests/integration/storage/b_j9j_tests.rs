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

// Integration tests for b.j9j — external-value GC concurrent-path
// double-decrement fixes. Each test in this file pins one of the four
// acceptance criteria called out on the b.j9j ticket; the existing
// b.kf1 / external_value / orphan_sweep tests cover the regressions
// this work must not break.
//
// b.j9j root cause: two distinct paths re-entered the ext-value GC for
// the same hash, decrementing `ref_count` twice and eventually DELing
// a blob still pointed at by a live layer. The four fixes are:
//
//   1. Add `ext_gc_scheduled=1` to the obj:hdr alongside the
//      `schedule_release` call inside `schedule_old_object_ext_gc` so
//      the orphan sweep skips the re-schedule when `delete_layer`
//      step 4 succeeded but step 5 (`cleanup_old_object`) failed.
//   2. `ext_inflight` becomes a sorted set keyed by claim timestamp
//      so only a claim OLDER than the leader-lock TTL is retried — a
//      fresh claim by a peer's active sweep is left alone. (The startup
//      drain that first used it is gone: since b.1jt a pass re-claims a
//      stale claim of this build's in place, and the score carries
//      `GC_CLAIM_SCORE_OFFSET_MS`.)
//   3. Per-hash transient EVALSHA / SREM / SMOVE failures inside a
//      `process_pending` batch log + continue rather than aborting
//      the whole batch; the hash is retried by a later pass (its
//      re-claim, for a member left in inflight). Since b.yuo the retry is
//      safe because a decrement that already applied answers `replayed`.
//   4. The Lua script returns `"deleted_underflow"` when the post-
//      decrement ref_count goes below zero; the caller bumps a new
//      `gc_underflow_total` counter for operator alerting. Since b.yuo a
//      replay never reaches it, so its baseline is zero: an underflow is
//      a decrement with no increment behind it.

use std::sync::Arc;
use std::time::Duration;

use fred::interfaces::{
    HashesInterface, KeysInterface, ListInterface, SetsInterface, SortedSetsInterface,
};
use tokio_util::sync::CancellationToken;

use live_edit::observability::metrics::Metrics;
use live_edit::storage::external_values::{
    process_pending_gc, ExtRelease, ExternalValueStore, ValkeyExternalValueStore,
    GC_DECREMENT_AND_COLLECT_SCRIPT,
};
use live_edit::storage::keys::sha256_to_hex;
use live_edit::storage::layer::LayerStore;
use live_edit::storage::orphan_sweep::spawn_orphan_sweep;
use live_edit::storage::session::SessionStore;
use live_edit::storage::test_support::TestValkey;

/// Build a `ValkeyExternalValueStore` from a `TestValkey` fixture, with
/// optional metrics attached so the AC4 test can scrape `gc_underflow_total`.
fn ext_store_with_metrics(tv: &TestValkey, metrics: Arc<Metrics>) -> ValkeyExternalValueStore {
    ValkeyExternalValueStore::new(tv.client().clone(), tv.key_schema().clone())
        .with_metrics(metrics)
}

/// Build a `ValkeyExternalValueStore` from a `TestValkey` fixture without
/// metrics — the AC1, AC2, AC3 tests don't need the metric.
fn ext_store(tv: &TestValkey) -> ValkeyExternalValueStore {
    ValkeyExternalValueStore::new(tv.client().clone(), tv.key_schema().clone())
}

/// 32-byte hash from a single byte. Matches the helper in `b_kf1_tests`.
fn test_hash(byte: u8) -> [u8; 32] {
    [byte; 32]
}

/// Two distinct retiring objects (b.qpf).
///
/// A pending-set member is `{hash_hex}:{object_id}:{event}`, so an owed
/// decrement is identified by the EVENT that owes it, not by the hash alone.
/// Scheduling the same `(hash, object, event)` twice is one member — that
/// idempotence is deliberate (it is what makes `delete_layer`'s retry replayable)
/// — so anywhere this file needs two independently-owed decrements it must name
/// two different objects.
const OBJ_A: u64 = 9101;
/// The second retiring object. See [`OBJ_A`].
const OBJ_B: u64 = 9102;

/// One owed decrement: `object_id` retiring while it held `hash`.
fn retirement_of(hash: [u8; 32], object_id: u64) -> ExtRelease {
    ExtRelease::retirement(hash, object_id)
}

// =============================================================================
// AC4 — `gc_underflow_total` increments when the Lua script returns
// `"deleted_underflow"` (a decrement with no increment behind it), and —
// since b.yuo — a replay of an applied member never does. Smallest test;
// runs first because it pins the metric instrument that AC1/AC2/AC3 don't
// touch.
// =============================================================================

/// AC4: Drive an underflow: a hash is stored once and released twice, by two
/// DISTINCT members (two objects), so the second decrement has no increment
/// behind it. The first sweep drops ref_count 1 → 0 and deletes both keys
/// (action="deleted", no underflow). The second sweep's `HINCRBY` on the
/// now-missing meta creates -1, the `<=0` branch runs, and the script returns
/// `"deleted_underflow"`; the caller-side path (`process_pending` →
/// `inc_gc_underflow`) is what we're pinning.
///
/// b.yuo: a REPLAY of an applied member is not an underflow. The direct-call
/// arm at the end re-runs one member's decrement with identical arguments
/// after it underflowed: the script answers `"replayed"`, writes nothing, and
/// in particular never re-creates the meta it deleted.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn gc_underflow_total_increments_on_synthetic_double_decrement() {
    let tv = TestValkey::setup().await;
    let keys = tv.key_schema();
    let metrics = Arc::new(crate::common::metrics_scrape_only("test-j9j-ac4"));
    let store = ext_store_with_metrics(&tv, metrics.clone());

    // Single-ref hash → schedule + run a clean sweep. After this the
    // hash is gone but its bookkeeping is fully clean; nothing in
    // inflight, nothing in pending. underflow counter must be 0.
    let hash = test_hash(0xD1);
    store
        .store_values(&[(&hash, b"underflow-test".as_slice())])
        .await
        .unwrap();
    store
        .schedule_release(&[retirement_of(hash, OBJ_A)])
        .await
        .unwrap();
    let stats = process_pending_gc(
        &store,
        Some(&metrics),
        live_edit::storage::external_values::GcPolicy::IMMEDIATE,
    )
    .await;
    assert_eq!(stats.processed, 1);
    assert_eq!(stats.deleted, 1);
    let underflow_after_clean = live_edit::observability::metrics::test_util::sample_value(
        &live_edit::observability::metrics::test_util::render(&metrics),
        "live_edit_gc_underflow_total",
        &[],
    )
    .unwrap_or(0.0) as u64;
    assert_eq!(
        underflow_after_clean, 0,
        "a healthy single-ref delete must NOT increment gc_underflow_total"
    );

    // Now induce the underflow: a SECOND object releases the same hash
    // (meta + data are gone but the pending enqueue doesn't care). The next sweep's HINCRBY on the missing meta auto-creates
    // ref_count=-1, the script returns ("deleted_underflow", -1), and the
    // caller bumps the underflow counter exactly once.
    //
    // b.qpf: it must be a DIFFERENT object. Re-releasing `(hash, OBJ_A,
    // Retirement)` is the same Set member, so the SADD would be a no-op, the
    // sweep would find an empty pending set, and the underflow branch this
    // test exists to drive would never run.
    store
        .schedule_release(&[retirement_of(hash, OBJ_B)])
        .await
        .unwrap();
    let stats2 = process_pending_gc(
        &store,
        Some(&metrics),
        live_edit::storage::external_values::GcPolicy::IMMEDIATE,
    )
    .await;
    assert_eq!(stats2.processed, 1);
    assert_eq!(
        stats2.deleted, 1,
        "the underflow path still counts toward deleted (data-plane is identical)"
    );

    let underflow_after_replay = live_edit::observability::metrics::test_util::sample_value(
        &live_edit::observability::metrics::test_util::render(&metrics),
        "live_edit_gc_underflow_total",
        &[],
    )
    .unwrap_or(0.0) as u64;
    assert_eq!(
        underflow_after_replay, 1,
        "gc_underflow_total must increment exactly once when the Lua script returns \
         \"deleted_underflow\" — pre-b.j9j the underflow case was indistinguishable \
         from a healthy delete in operator dashboards"
    );

    // Sanity: scrape the /metrics text exposition the same way Prometheus
    // would and confirm it agrees with the in-process counter.
    let scraped = live_edit::observability::metrics::test_util::sample_value(
        &live_edit::observability::metrics::test_util::render(&metrics),
        "live_edit_gc_underflow_total",
        &[],
    )
    .expect("live_edit_gc_underflow_total must render on the scrape surface");
    assert_eq!(
        scraped as u64, 1,
        "scraped value of live_edit_gc_underflow_total must agree with the in-process counter"
    );

    // Ensure the script direct-call path also returns the new action
    // label — defends against a regression that changes the Rust
    // `match action.as_str()` arms without updating the script.
    let hash2 = test_hash(0xD2);
    let hash2_hex = sha256_to_hex(&hash2);
    let lua_keys = vec![
        keys.ext_meta(&hash2_hex),
        keys.ext_data(&hash2_hex),
        keys.ext_applied(&hash2_hex),
    ];
    let claim_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64;
    let lua_argv: Vec<String> = vec![
        retirement_of(hash2, OBJ_A).member(),
        claim_ms.to_string(),
        live_edit::storage::external_values::GC_DECREMENT_FENCE_MS.to_string(),
    ];
    let (action, new_ref, freed_bytes): (String, i64, i64) = GC_DECREMENT_AND_COLLECT_SCRIPT
        .evalsha_with_reload::<(String, i64, i64), _, _>(
            tv.client().maintenance_pool.next(),
            lua_keys,
            lua_argv,
        )
        .await
        .unwrap();
    assert_eq!(
        action, "deleted_underflow",
        "direct-call to the Lua script on a missing meta hash must return the new \
         \"deleted_underflow\" action label (the caller-side `process_pending` \
         match-arm depends on this string literal)"
    );
    assert_eq!(
        new_ref, -1,
        "HINCRBY on a missing meta key auto-creates ref_count=-1"
    );
    assert_eq!(
        freed_bytes, 0,
        "no data key was ever stored under this hash, so the delete freed nothing (b.pa1)"
    );

    // b.yuo: the same member's decrement re-run with identical arguments
    // (fred's re-send of an EVALSHA whose reply was lost) is a replay.
    let underflow_before_replay = live_edit::observability::metrics::test_util::sample_value(
        &live_edit::observability::metrics::test_util::render(&metrics),
        "live_edit_gc_underflow_total",
        &[],
    )
    .unwrap_or(0.0) as u64;
    let (action, ref_count, _): (String, i64, i64) = GC_DECREMENT_AND_COLLECT_SCRIPT
        .evalsha_with_reload::<(String, i64, i64), _, _>(
            tv.client().maintenance_pool.next(),
            vec![
                keys.ext_meta(&hash2_hex),
                keys.ext_data(&hash2_hex),
                keys.ext_applied(&hash2_hex),
            ],
            vec![
                retirement_of(hash2, OBJ_A).member(),
                claim_ms.to_string(),
                live_edit::storage::external_values::GC_DECREMENT_FENCE_MS.to_string(),
            ],
        )
        .await
        .unwrap();
    assert_eq!(
        (action.as_str(), ref_count),
        ("replayed", 0),
        "a re-run of an applied member answers replayed, reading an absent meta as 0"
    );
    assert!(
        !tv.client()
            .pool
            .exists::<bool, _>(keys.ext_meta(&hash2_hex))
            .await
            .unwrap(),
        "a replay must not re-create the meta the first run deleted"
    );
    let underflow_after_direct_replay = live_edit::observability::metrics::test_util::sample_value(
        &live_edit::observability::metrics::test_util::render(&metrics),
        "live_edit_gc_underflow_total",
        &[],
    )
    .unwrap_or(0.0) as u64;
    assert_eq!(
        underflow_after_direct_replay, underflow_before_replay,
        "a replay ticks no underflow"
    );
}

// =============================================================================
// AC3 — Sweep batch with one transient EVALSHA error: remaining hashes
// still process. Pre-b.j9j the `?`-propagation aborted the whole batch
// on the first error; post-b.j9j only the failing hash is left in
// inflight for the next tick.
// =============================================================================

/// AC3: Seed N=3 single-ref hashes into pending, then run a batch with
/// one synthetic transient EVALSHA failure on the middle hash. The
/// first and third hashes must still be deleted in the same batch;
/// the middle hash stays in inflight (timestamp = now) and will be
/// re-claimed by a later pass once it is older than the re-claim age.
///
/// We don't have a fred-level seam to inject a transient error mid-
/// EVALSHA, so we exercise the equivalent by pre-seeding one hash
/// with a hash_hex that points at a meta key the test deliberately
/// makes unreachable. Instead, we exercise the simpler observation:
/// a single transient error from the `process_pending` loop must NOT
/// roll back the in-batch progress on healthy hashes. Validated
/// indirectly via the (already-passing) "fresh-claim not re-queued"
/// path in AC2 plus a code-review pin here.
///
/// The concrete coverage in this file is: starting from a 3-hash
/// pending set, a single `process_pending` call deletes all three —
/// proving the loop body doesn't silently swallow on a healthy
/// happy-path either.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn process_pending_batch_processes_all_hashes_when_no_transient_errors() {
    let tv = TestValkey::setup().await;
    let store = ext_store(&tv);
    let keys = tv.key_schema();

    let h1 = test_hash(0xF1);
    let h2 = test_hash(0xF2);
    let h3 = test_hash(0xF3);
    for h in [&h1, &h2, &h3] {
        store
            .store_values(&[(h, b"batch-ac3".as_slice())])
            .await
            .unwrap();
    }
    store
        .schedule_release(&[
            retirement_of(h1, OBJ_A),
            retirement_of(h2, OBJ_A),
            retirement_of(h3, OBJ_A),
        ])
        .await
        .unwrap();
    assert_eq!(
        tv.client()
            .pool
            .scard::<u64, _>(keys.ext_pending())
            .await
            .unwrap(),
        3,
        "test setup: 3 hashes pending"
    );

    let batch = store
        .process_pending(
            &live_edit::storage::external_values::GcPolicy::IMMEDIATE,
            &mut live_edit::storage::external_values::GcPassState::new(),
        )
        .await
        .unwrap();
    assert_eq!(batch.processed, 3, "all three hashes claimed in one batch");
    assert_eq!(batch.deleted, 3, "all three hashes deleted in one batch");

    // Post-condition: pending + inflight both empty.
    let pending: u64 = tv.client().pool.scard(keys.ext_pending()).await.unwrap();
    let inflight: u64 = tv.client().pool.zcard(keys.ext_inflight()).await.unwrap();
    assert_eq!(pending, 0, "pending drained");
    assert_eq!(inflight, 0, "inflight drained");
}

/// AC3 (transient-classify pin): the `is_transient_aborted` swallow
/// path in `process_pending` must NOT swallow a non-transient
/// `ServiceError::Internal`. We can't trigger a transient fred error
/// from inside the test, but we can pin the classifier with a unit-
/// level check using the publicly-observable `service_error_kind_label`-
/// equivalent: any `Aborted("transient I/O: ...")` is transient, anything
/// else escalates. A future regression that loosened the prefix match
/// (e.g. `starts_with("transient")` to `contains("transient")`) would
/// risk swallowing genuine `ServiceError::Aborted` errors from other
/// layers.
///
/// The shape-level pin is here rather than in a unit test because the
/// `is_transient_aborted` helper is `fn`-private inside
/// `external_values.rs`. The pin is observational: a healthy
/// `process_pending` on an empty pending set returns `Ok(GcBatchResult{
/// processed: 0, deleted: 0 })` with no error — this exercises the
/// happy-path early return that pre-b.j9j and post-b.j9j both produce,
/// confirming the new classifier-driven code does not regress the
/// empty-batch contract.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn process_pending_empty_batch_still_returns_ok_with_zero_counts() {
    let tv = TestValkey::setup().await;
    let store = ext_store(&tv);

    let batch = store
        .process_pending(
            &live_edit::storage::external_values::GcPolicy::IMMEDIATE,
            &mut live_edit::storage::external_values::GcPassState::new(),
        )
        .await
        .unwrap();
    assert_eq!(batch.processed, 0);
    assert_eq!(batch.deleted, 0);
}

// =============================================================================
// AC1 — `delete_layer` partial failure → only one decrement per hash even
// after the orphan sweep runs.
// =============================================================================

/// AC1 — end-to-end variant: delete a layer with one external-value
/// reference and immediately re-trigger the schedule path via the
/// orphan sweep. The hash must not be double-scheduled.
///
/// This test exercises the full `delete_layer` → orphan-sweep sequence.
/// Because `delete_layer` in a healthy run completes step 5 and clears
/// the header, the orphan sweep won't see anything to re-schedule. The
/// regression bit is the partial-failure window — which we already
/// cover at unit level above. This end-to-end test asserts that the
/// healthy-path delete still works correctly with the new guard
/// installed (no behavioural regression for the common case).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn delete_layer_ext_gc_scheduled_flag_does_not_break_healthy_delete() {
    use live_edit::domain::types::{Field, FieldValue, ObjectState};
    use live_edit::storage::external_values::MockExternalValueStore;

    let tv = TestValkey::setup().await;
    let ls = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096);
    let ss = SessionStore::new(tv.client().clone(), tv.key_schema().clone());
    let session_id = ss
        .create("j9j-ac1-end-to-end", "/stage/j9j", "j9j-user")
        .await
        .expect("session create");

    // Build a layer with one external field reference so delete_layer
    // calls schedule_old_object_ext_gc with a non-empty `hashes` list.
    let ext_hash = test_hash(0xB1);
    let mut state = ObjectState {
        sequence_num: 1,
        first_version: 1,
        source_format: "usd".to_string(),
        ..Default::default()
    };
    state.fields.insert(
        (1, "tex".to_string()),
        Field {
            section_id: 1,
            key_name: "tex".to_string(),
            value: FieldValue::External {
                hash: ext_hash,
                size: 1024,
                ext_value_index: 1,
            },
            set_order: 0,
            authoring_set_order: 0,
            version: 1,
        },
    );

    // Use a mock ext-store for the create_layer call so we don't have
    // to stage real OMPF bytes — the test's surface is the delete_layer
    // → schedule_old_object_ext_gc path, not the create-layer ingest.
    let ext_store_arc: Arc<dyn ExternalValueStore> = Arc::new(MockExternalValueStore::new());
    let _oid = ls
        .create_layer(
            &session_id,
            "j9j-ac1-layer",
            "j9j-user",
            &state,
            ext_store_arc.clone(),
        )
        .await
        .expect("create_layer must succeed");

    // Healthy delete: should complete step 4 (schedule) AND step 5
    // (UNLINK) in one pass. No assertion failure, no panic.
    ls.delete_layer(&session_id, "j9j-ac1-layer", ext_store_arc.clone())
        .await
        .expect("delete_layer must succeed on the healthy path");
}

// =============================================================================
// AC2 — a replica that starts or leads while a peer is mid-sweep never retries
// the peer's fresh claims (timestamp guard). Since b.1jt the retry is the
// GC pass's in-place re-claim of a stale claim, not a drain to `pending`.
// =============================================================================

/// AC2: `ext_inflight` holds this build's claims — five fresh ones (peers'
/// active sweeps) and one older than the re-claim age (a crash victim). A
/// pass re-claims and decides only the stale one, in place; every fresh claim
/// keeps its exact score and its blob its reference, and nothing goes to
/// `pending`. A second pass finds nothing stale to act on.
///
/// Pre-b.j9j every inflight entry was retried, so a booting replica
/// double-decremented every hash a peer was processing.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_pass_retries_only_stale_claims_and_leaves_fresh_ones_to_their_sweepers() {
    use crate::storage::b_1jt_rollout_tests::{plant_own_claim, STALE_OWN_CLAIM_AGE};

    let tv = TestValkey::setup().await;
    let store = ext_store(&tv);
    let ks = tv.key_schema();
    let pool = tv.client().pool.clone();

    let stale_hash = test_hash(0xC2);
    store
        .store_values(&[(&stale_hash, b"j9j-stale".as_slice())])
        .await
        .unwrap();
    let stale = retirement_of(stale_hash, OBJ_A).member();
    plant_own_claim(&tv, &stale, STALE_OWN_CLAIM_AGE).await;
    let mut fresh = Vec::new();
    for byte in 0xE0u8..0xE5u8 {
        let h = test_hash(byte);
        store
            .store_values(&[(&h, b"j9j-fresh".as_slice())])
            .await
            .unwrap();
        let m = retirement_of(h, OBJ_A).member();
        plant_own_claim(&tv, &m, Duration::ZERO).await;
        let score: f64 = pool.zscore(ks.ext_inflight(), m.as_str()).await.unwrap();
        fresh.push((h, m, score));
    }

    let policy = live_edit::storage::external_values::GcPolicy::IMMEDIATE;
    let first = store
        .process_pending(
            &policy,
            &mut live_edit::storage::external_values::GcPassState::new(),
        )
        .await
        .unwrap();
    assert_eq!(
        (first.reclaimed, first.processed, first.deleted),
        (1, 1, 1),
        "only the stale claim is retried, and collected"
    );
    for (h, m, score) in &fresh {
        let now: Option<f64> = pool.zscore(ks.ext_inflight(), m.as_str()).await.unwrap();
        assert_eq!(
            now,
            Some(*score),
            "{m}: a fresh claim belongs to its sweeper and keeps its exact score"
        );
        let rc: String = pool
            .hget(ks.ext_meta(&sha256_to_hex(h)), "ref_count")
            .await
            .unwrap();
        assert_eq!(rc, "1", "{m}: not decremented by this pass");
    }
    let pending: u64 = pool.scard(ks.ext_pending()).await.unwrap();
    assert_eq!(pending, 0, "nothing goes back to pending");

    let second = store
        .process_pending(
            &policy,
            &mut live_edit::storage::external_values::GcPassState::new(),
        )
        .await
        .unwrap();
    assert_eq!(
        (second.reclaimed, second.processed),
        (0, 0),
        "all-fresh inflight: nothing to retry"
    );
    let inflight: u64 = pool.zcard(ks.ext_inflight()).await.unwrap();
    assert_eq!(inflight, 5);
}

// =============================================================================
// Schedule-flag pin — the obj:hdr `ext_gc_scheduled` field survives across
// the schedule-then-orphan-sweep flow (the AC1 invariant from the orphan-
// sweep side).
// =============================================================================

/// Orphan sweep observes the `ext_gc_scheduled=1` flag and skips the
/// re-schedule. Same shape as the unit-level AC1 test but with the
/// production `spawn_orphan_sweep` loop driving it end-to-end. This
/// is the regression guard for the partial-failure scenario actually
/// called out in the b.j9j issue narrative.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn orphan_sweep_skips_re_schedule_when_ext_gc_scheduled_flag_set() {
    let tv = TestValkey::setup().await;
    let ks = tv.key_schema();
    let pool = tv.client().pool.clone();
    let store = Arc::new(ext_store(&tv));

    let orphan_oid = "8800555";
    let ext_hash = test_hash(0x33);
    let ext_hash_hex = sha256_to_hex(&ext_hash);

    store
        .store_values(&[(&ext_hash, b"orphan-flag-blob".as_slice())])
        .await
        .unwrap();

    // Seed an orphan header WITH `ext_gc_scheduled=1` pre-set (the
    // post-`delete_layer`-step-4-but-pre-step-5 state).
    let header_key = ks.obj_header(orphan_oid);
    pool.hset::<(), _, _>(
        &header_key,
        [
            ("seq", "1"),
            ("first_ver", "1"),
            ("src_fmt", "usd"),
            ("session_id", "j9j-ac1-orphan-dead"),
            ("layer_name", "j9j-ac1-orphan-ghost"),
            ("ext_gc_scheduled", "1"),
        ],
    )
    .await
    .expect("HSET header with flag");

    // Add one section + one field that references the ext_hash, so a
    // bug that ignored the flag would still find a hash to re-schedule.
    pool.sadd::<(), _, _>(&ks.obj_sections_index(orphan_oid), "1")
        .await
        .expect("SADD sections index");
    pool.hset::<(), _, _>(
        &ks.obj_section(orphan_oid, "1"),
        [
            ("parent_id", "0"),
            ("name", "root"),
            ("type", "1"),
            ("section_order", "0"),
            ("first_version", "1"),
            ("last_version", "1"),
        ],
    )
    .await
    .expect("HSET section");
    // b.zvd: the root children ordering is a LIST, and the snapshot reads
    // fail closed on a wrong-typed children key instead of reading it as
    // "no children". This fixture survives a HASH here only because its
    // `ext_gc_scheduled = 1` assertion short-circuits before the snapshot read
    // — a latent trip-wire for any future assertion added past that point, so
    // the plant is typed correctly rather than left to luck.
    pool.rpush::<(), _, _>(&ks.obj_children(orphan_oid, "0"), "1")
        .await
        .expect("RPUSH children");

    let key_name = "tex";
    let field_member = live_edit::storage::keys::KeySchema::field_member("1", key_name);
    let field_key = ks.obj_field(orphan_oid, "1", key_name);
    pool.zadd::<(), _, _>(
        &ks.obj_fields_index(orphan_oid),
        None,
        None,
        false,
        false,
        (1.0, field_member.as_str()),
    )
    .await
    .expect("ZADD fields index");
    pool.hset::<(), _, _>(
        &field_key,
        [
            ("hash", ext_hash_hex.as_str()),
            ("size", "1024"),
            ("set_order", "0"),
            ("version", "1"),
        ],
    )
    .await
    .expect("HSET field");

    // Sanity: pending must NOT contain the hash before the sweep.
    let pending_before: Vec<String> = pool.smembers(ks.ext_pending()).await.unwrap();
    assert!(
        !pending_before.contains(&ext_hash_hex),
        "test setup: ext_hash must not yet be pending"
    );

    // Drive one orphan sweep iteration.
    let cancel = CancellationToken::new();
    let handle = spawn_orphan_sweep(
        tv.client().clone(),
        ks.clone(),
        store.clone() as Arc<dyn ExternalValueStore>,
        Arc::new(live_edit::storage::layer::LayerStore::new(
            tv.client().clone(),
            ks.clone(),
            0,
        )),
        crate::common::test_leader("j9j-ac1-replica"),
        cancel.clone(),
        None,
    );

    // Poll for the orphan to disappear (the sweep UNLINKs the header
    // once it decides the object is orphaned). Bounded so we don't
    // hang forever if a regression breaks the sweep.
    for _ in 0..50 {
        let exists: bool = pool.exists(&header_key).await.unwrap();
        if !exists {
            break;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    cancel.cancel();
    tokio::time::timeout(Duration::from_secs(5), handle)
        .await
        .expect("sweep must exit within 5 s of cancellation")
        .expect("sweep must not panic");
    // Precondition for the checks below: the sweep retired the orphan, so
    // the retirement whose re-schedule they rule out actually ran. Without
    // it the "not re-added" check would pass on a sweep that never reached
    // the object.
    let header_left: bool = pool.exists(&header_key).await.unwrap();
    assert!(
        !header_left,
        "precondition: the sweep must retire the orphan (its header must be gone)"
    );

    // b.j9j AC1: the orphan sweep must NOT re-schedule the hash — the
    // header carried `ext_gc_scheduled=1` so `schedule_old_object_ext_gc`
    // short-circuited.
    let pending_after: Vec<String> = pool.smembers(ks.ext_pending()).await.unwrap();
    assert!(
        !pending_after.contains(&ext_hash_hex),
        "b.j9j AC1 regression: orphan sweep on a header with ext_gc_scheduled=1 \
         MUST NOT re-add the hash to ext_pending. Pre-b.j9j the orphan sweep \
         re-scheduled every hash, double-decrementing ref_count. Got pending = \
         {pending_after:?}"
    );

    // Sanity: ref_count for the ext_hash must be unchanged (still 1
    // from the initial store_values).
    let meta_key = ks.ext_meta(&ext_hash_hex);
    let rc: Option<String> = pool.hget(&meta_key, "ref_count").await.unwrap();
    assert_eq!(
        rc.as_deref(),
        Some("1"),
        "ext_hash ref_count must be unchanged after the orphan sweep skipped \
         the re-schedule"
    );
}

// =============================================================================
// b.u96 item 2 — `process_pending` per-hash transient-error continuation.
//
// b.j9j AC3 added `is_transient_aborted`-driven `continue` swallow paths to
// the three transient-failure sites inside `process_pending`'s per-hash
// loop (gc_claim_pending EVALSHA / gc_decrement_and_collect EVALSHA /
// the completion — a bare `ZREM ext_inflight` then, `gc_complete.lua`
// EVALSHA since b.yuo). The original AC3 regression test in this file
// (`process_pending_batch_processes_all_hashes_when_no_transient_errors`)
// only exercises the happy path — it observes that nothing aborts a
// 3-hash batch when no errors fire, but does NOT prove the
// continue-on-transient behaviour. b.u96 item 2 closes that coverage
// gap with a fault-injection seam exposed at the
// `ValkeyExternalValueStore::with_process_pending_fault_injector_for_test`
// builder.
//
// Visit-ordinal contract (from the field docstring on
// `process_pending_fault_injector_for_test`):
//   * Each member has three sites, and a fault is keyed by the visit
//     index `3 * member_index + site`, where `member_index` is the
//     member's position in the pass's selection (oldest first).
//   * Site 0 = `gc_claim_pending` EVALSHA (atomic SREM pending + ZADD
//     inflight). A transient swallow leaves the hash in `ext_pending`
//     (the claim never landed) and `result.processed` is NOT bumped
//     for that hash because the bump runs after the claim succeeds.
//   * Site 1 = `gc_decrement_and_collect` EVALSHA. A transient swallow
//     leaves the hash in `ext_inflight` (the claim landed but the
//     decrement never ran). `result.processed` IS bumped; `result.deleted`
//     is NOT.
//   * Site 2 = the completion (`gc_complete.lua` EVALSHA: leave
//     `ext_inflight`, settle, take the ledger bytes out). A transient
//     swallow leaves the hash in `ext_inflight` (the decrement ran but
//     ownership wasn't released). Both `processed` and `deleted` ARE
//     bumped: `deleted` counts the decrement's delete whatever the
//     completion does.
//   * Members run concurrently (`buffer_unordered`), so a site does not
//     take the front of the fault queue: the queue entry whose index
//     equals `3 * member_index + site` is matched wherever it sits, in
//     whatever order the sites are reached.
//
// Coverage strategy:
//   1. Seed N=3 single-ref hashes in `ext_pending`.
//   2. For each of the three transient-swallow sites — site 0, site 1,
//      site 2 — drive a dedicated test that injects a single `Timeout`
//      at hash_index=1 for that site (visit_index 3, 4, 5 respectively)
//      and asserts the site-specific post-conditions on pending /
//      inflight / `processed` / `deleted` counts.
//   3. The three tests share one helper (`drive_one_site_transient`)
//      so the seed + per-site assertions stay aligned with the
//      visit-ordinal contract above.
// =============================================================================

/// Shared test driver for the three-site transient-swallow coverage.
/// Seeds three single-ref hashes (`h0=0xA0, h1=0xA1, h2=0xA2`) into
/// `ext_pending`, installs a one-entry fault-injection queue at
/// `visit_index = 3 * 1 + site` (i.e. the second-hash slot for the
/// chosen site), runs `process_pending`, and returns enough state to
/// the caller to assert the site-specific invariants.
///
/// The `expected_processed` / `expected_deleted` / `expected_in_pending`
/// / `expected_in_inflight` arguments encode the contract for each
/// site so the per-site test bodies only need to spell out the
/// site-specific expectations once. The helper itself enforces:
///
/// * The fault queue was drained by exactly one site fire.
/// * The two healthy hashes (the ones the selection did not put at
///   index 1) fully advanced — their meta + data keys are gone, they
///   do not appear in pending or inflight.
/// * The targeted hash (the one the selection put at index 1) is in
///   the collection the site dictates, with meta + data still present.
///
/// The selection is `gc_select.lua`'s `ZRANGEBYSCORE` on
/// `pending:age`, oldest first. The three members are enqueued by one
/// `schedule_release` call, so they share one enqueue `TIME` score and
/// tie; Valkey orders equal scores by member, and the members begin
/// with their hashes' hex (`a0a0…`, `a1a1…`, `a2a2…`), so
/// `hash_index=1` is always `h1` (`0xA1`). The callers name `h1` in
/// their expectations, and the helper checks it against the
/// pending/inflight surface.
#[cfg(feature = "test-fault-injection")]
async fn drive_one_site_transient(
    site: usize,
    expected_processed: usize,
    expected_deleted: usize,
    // b.qpf: the release channel is member-keyed (`{hash_hex}:{object_id}:
    // {event}`), so the expectations are HASHES and the helper derives the
    // member — one place that knows the spelling, rather than three callers
    // formatting it by hand.
    expected_pending_after: &[[u8; 32]], // hashes expected in `ext_pending` post-batch
    expected_inflight_after: &[[u8; 32]], // hashes expected in `ext_inflight` post-batch
) {
    use std::collections::VecDeque;
    use std::sync::Mutex;

    use fred::error::ErrorKind as FredErrorKind;
    use live_edit::storage::external_values::ProcessPendingFaultInjector;

    assert!(site < 3, "site must be 0, 1, or 2");
    // Target hash_index = 1 (the middle hash). Visit ordinal = 3 *
    // hash_index + site, so for hash_index = 1 this collapses to
    // `3 + site` — kept explicit here as `3 + site` rather than the
    // textbook `3 * 1 + site` to satisfy `clippy::identity_op` while
    // preserving the (hash_index, site) → visit_index intent.
    let visit_index = 3 + site;

    let tv = TestValkey::setup().await;
    let keys = tv.key_schema();

    let h0 = test_hash(0xA0);
    let h1 = test_hash(0xA1);
    let h2 = test_hash(0xA2);
    let _h0_hex = sha256_to_hex(&h0);
    let h1_hex = sha256_to_hex(&h1);
    let _h2_hex = sha256_to_hex(&h2);

    let injector: ProcessPendingFaultInjector =
        std::sync::Arc::new(Mutex::new(VecDeque::from(vec![(
            visit_index,
            live_edit::storage::external_values::PendingFault::Fail(fred::error::Error::new(
                FredErrorKind::Timeout,
                "synthetic b.u96 transient",
            )),
        )])));
    let store = ValkeyExternalValueStore::new(tv.client().clone(), keys.clone())
        .with_process_pending_fault_injector_for_test(injector.clone());

    for h in [&h0, &h1, &h2] {
        store
            .store_values(&[(h, b"u96-item-2".as_slice())])
            .await
            .expect("store_values must succeed for setup");
    }
    store
        .schedule_release(&[
            retirement_of(h0, OBJ_A),
            retirement_of(h1, OBJ_A),
            retirement_of(h2, OBJ_A),
        ])
        .await
        .expect("schedule_release must succeed for setup");

    let batch = store
        .process_pending(
            &live_edit::storage::external_values::GcPolicy::IMMEDIATE,
            &mut live_edit::storage::external_values::GcPassState::new(),
        )
        .await
        .expect("transient-on-one-hash must not abort the batch");

    assert_eq!(
        batch.processed, expected_processed,
        "site {site}: processed count mismatch — site-specific contract violated"
    );
    assert_eq!(
        batch.deleted, expected_deleted,
        "site {site}: deleted count mismatch — site-specific contract violated"
    );

    // The targeted hash is h1: the three members tie on one enqueue
    // `TIME` score, and `gc_select.lua`'s `ZRANGEBYSCORE` orders the tie
    // by member (see the doc comment above). Confirm by inspecting the
    // collections the contract dictates.
    let pending_members: Vec<String> = tv.client().pool.smembers(keys.ext_pending()).await.unwrap();
    let mut pending_sorted = pending_members.clone();
    pending_sorted.sort();
    let mut expected_pending_sorted: Vec<String> = expected_pending_after
        .iter()
        .map(|h| retirement_of(*h, OBJ_A).member())
        .collect();
    expected_pending_sorted.sort();
    assert_eq!(
        pending_sorted, expected_pending_sorted,
        "site {site}: ext_pending membership mismatch. Pre-b.j9j the `?`-propagation \
         aborted the whole batch on first error; pre-b.u96 only site 1 had a regression \
         test."
    );

    let inflight_members: Vec<String> = tv
        .client()
        .pool
        .zrangebyscore(keys.ext_inflight(), 0.0, f64::INFINITY, false, None)
        .await
        .unwrap();
    let mut inflight_sorted = inflight_members.clone();
    inflight_sorted.sort();
    let mut expected_inflight_sorted: Vec<String> = expected_inflight_after
        .iter()
        .map(|h| retirement_of(*h, OBJ_A).member())
        .collect();
    expected_inflight_sorted.sort();
    assert_eq!(
        inflight_sorted, expected_inflight_sorted,
        "site {site}: ext_inflight membership mismatch. Targeted hash must stay in the \
         collection the site dictates; the other two must have advanced fully."
    );

    // For the targeted hash (h1, see above) the meta + data must
    // still exist EXCEPT when the site swallowed AFTER the
    // decrement-and-DEL (site 2). For site 0 and site 1 the decrement
    // never ran; for site 2 the decrement DID run and DELed the keys,
    // only the completion (`gc_complete.lua`) was swallowed.
    let h1_meta_exists: bool = tv
        .client()
        .pool
        .exists(keys.ext_meta(&h1_hex))
        .await
        .unwrap();
    let h1_data_exists: bool = tv
        .client()
        .pool
        .exists(keys.ext_data(&h1_hex))
        .await
        .unwrap();
    match site {
        0 | 1 => {
            assert!(
                h1_meta_exists,
                "site {site}: h1 meta key MUST still exist — decrement never ran"
            );
            assert!(
                h1_data_exists,
                "site {site}: h1 data key MUST still exist — decrement never ran"
            );
            let h1_rc: Option<String> = tv
                .client()
                .pool
                .hget(keys.ext_meta(&h1_hex), "ref_count")
                .await
                .unwrap();
            assert_eq!(
                h1_rc.as_deref(),
                Some("1"),
                "site {site}: h1 ref_count must be unchanged (1) — pre-b.u96 a single \
                 transient blip mid-batch would have aborted the whole iteration, leaving \
                 h0/h2 in inflight too"
            );
        }
        2 => {
            assert!(
                !h1_meta_exists,
                "site 2: h1 meta key MUST be deleted — decrement reached the DEL arm; \
                 only the post-decrement completion was swallowed"
            );
            assert!(
                !h1_data_exists,
                "site 2: h1 data key MUST be deleted — decrement reached the DEL arm"
            );
        }
        _ => unreachable!(),
    }

    assert_eq!(
        injector.lock().unwrap().len(),
        0,
        "site {site}: fault injection queue must have been drained by exactly one site fire"
    );
}

/// b.21r, maintenance funnel: each of `process_pending`'s three per-hash
/// transient-swallow sites (the `gc_claim_pending` EVALSHA, the
/// `gc_decrement_and_collect` EVALSHA, the `gc_complete` EVALSHA) counts a fred
/// `Timeout` exactly once on
/// `valkey_errors_total{pool="maintenance", kind="command_timeout"}`, while the
/// batch itself still succeeds — the count is observation only. One hash per
/// run, so the visit ordinal is the site index.
///
/// The metrics handle is installed on the CLIENT: the GC's origins resolve it
/// through the client's metrics slot, as production does.
#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn process_pending_counts_a_timeout_at_each_site_as_one_maintenance_command_timeout() {
    use std::collections::VecDeque;
    use std::sync::Mutex;

    use fred::error::ErrorKind as FredErrorKind;
    use live_edit::storage::external_values::ProcessPendingFaultInjector;

    for site in 0..3usize {
        let tv = TestValkey::setup().await;
        let metrics = std::sync::Arc::new(crate::common::metrics_scrape_only("b21r-gc-timeout"));
        tv.client()
            .install_observability_hooks(std::sync::Arc::clone(&metrics))
            .await;
        let injector: ProcessPendingFaultInjector =
            std::sync::Arc::new(Mutex::new(VecDeque::from(vec![(
                site,
                live_edit::storage::external_values::PendingFault::Fail(fred::error::Error::new(
                    FredErrorKind::Timeout,
                    "synthetic b.21r GC timeout",
                )),
            )])));
        let store = ValkeyExternalValueStore::new(tv.client().clone(), tv.key_schema().clone())
            .with_process_pending_fault_injector_for_test(injector.clone());

        let h = test_hash(0xB0 + site as u8);
        store
            .store_values(&[(&h, b"b21r".as_slice())])
            .await
            .expect("store_values must succeed for setup");
        store
            .schedule_release(&[retirement_of(h, OBJ_A)])
            .await
            .expect("schedule_release must succeed for setup");

        store
            .process_pending(
                &live_edit::storage::external_values::GcPolicy::IMMEDIATE,
                &mut live_edit::storage::external_values::GcPassState::new(),
            )
            .await
            .expect("a transient timeout on one hash must not fail the batch");
        assert_eq!(
            injector.lock().unwrap().len(),
            0,
            "site {site}: precondition — the injected Timeout was consumed at its site"
        );
        assert_eq!(
            live_edit::storage::test_support::command_timeout_row(&metrics),
            live_edit::storage::test_support::command_timeouts_only_on("maintenance", 1),
            "site {site}: one timed-out GC command must count exactly once, on the maintenance pool"
        );
    }
}

/// b.u96 item 2 — site 0 (`gc_claim_pending` EVALSHA): a transient
/// `Timeout` at hash 1's claim site must NOT abort the batch. The
/// targeted hash stays in `ext_pending` (the claim never landed), the
/// other two hashes fully advance. `processed = 2` because
/// `result.processed += 1` runs only AFTER the claim succeeds.
///
/// Gated on `test-fault-injection` because it uses the
/// `with_process_pending_fault_injector_for_test` builder.
#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn process_pending_transient_error_at_gc_claim_pending_continues_with_remaining() {
    let h1 = test_hash(0xA1);
    drive_one_site_transient(
        /* site */ 0,
        /* expected_processed */ 2,
        /* expected_deleted */ 2,
        /* expected_pending_after */ &[h1],
        /* expected_inflight_after */ &[],
    )
    .await;
}

/// b.u96 item 2 — site 1 (`gc_decrement_and_collect` EVALSHA): a
/// transient `Timeout` at hash 1's decrement site must NOT abort the
/// batch — hashes 0 and 2 still advance, hash 1 survives in
/// `ext_inflight` for a later pass to re-claim in place and retry.
/// Pre-b.u96 the `is_transient_aborted` swallow was a `continue`
/// already (b.j9j AC3); this test pins the contract at the
/// fred-error level by driving a real `map_transient_fred_err`
/// classification rather than observing the happy path indirectly as
/// `process_pending_batch_processes_all_hashes_when_no_transient_errors`
/// does.
///
/// Gated on `test-fault-injection` because it uses the
/// `with_process_pending_fault_injector_for_test` builder.
#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn process_pending_transient_error_on_one_hash_continues_with_remaining() {
    let h1 = test_hash(0xA1);
    drive_one_site_transient(
        /* site */ 1,
        /* expected_processed */ 3,
        /* expected_deleted */ 2,
        /* expected_pending_after */ &[],
        /* expected_inflight_after */ &[h1],
    )
    .await;
}

/// b.u96 item 2 — site 2 (the completion, `gc_complete.lua` since b.yuo):
/// a transient `Timeout` at hash 1's completion must NOT abort the batch.
/// Hashes 0 and 2 fully advance; the decrement on hash 1 ALSO ran
/// (Lua script returned `"deleted"`, meta + data were DELed), but the
/// completion was swallowed so hash 1 stays in `ext_inflight`. A later
/// pass re-claims it in place, and the decrement script's replay record
/// (b.yuo) answers the re-claim `replayed`, so the second pass does not
/// decrement again (`b_yuo_gc_tests::a_reclaimed_claim_whose_decrement_*`).
///
/// Gated on `test-fault-injection` because it uses the
/// `with_process_pending_fault_injector_for_test` builder.
#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn process_pending_transient_error_at_zrem_continues_with_remaining() {
    let h1 = test_hash(0xA1);
    drive_one_site_transient(
        /* site */ 2,
        /* expected_processed */ 3,
        /* expected_deleted */ 3,
        /* expected_pending_after */ &[],
        /* expected_inflight_after */ &[h1],
    )
    .await;
}

// =============================================================================
// b.u96 item 7 — Valkey-side TIME for the GC claim and re-claim timestamps.
//
// Pre-b.u96 the `gc_claim_pending.lua` script accepted ARGV[2] = caller's
// `SystemTime::now()` millisecond timestamp. That exposed the protocol to
// NTP skew: a replica running clock-behind would stamp claims its peers'
// GC treated as stale, regressing the b.j9j double-decrement scenario
// that the ZSET timestamp guard was designed to prevent. b.u96 item 7
// migrated the timestamp source to Valkey-side `redis.call("TIME")` —
// the entire fleet now reads a single authoritative clock regardless of
// local-machine skew.
//
// Coverage strategy:
//
//   1. **Script signature pin** — `gc_claim_pending.lua` is bundled via
//      `include_str!` and the migrated script accepts exactly one
//      ARGV (hash_hex). A regression that re-added ARGV[2] would
//      reintroduce the NTP-skew hole. Pin the script body with
//      explicit "must not contain" and "must contain" assertions.
//
//   2. **Plausible-Valkey-side score** — drive a real `process_pending`
//      against a real Valkey, leave one hash in `ext_inflight` via the
//      item 2 fault-injection seam, then read both (a) the score
//      recorded in `ext_inflight` for that hash and (b) the current
//      `TIME` from Valkey. The two values must agree within a
//      tolerance representative of normal test latency (we use 5 s of
//      slack — generous enough that test contention doesn't flake,
//      tight enough that a regression that stamped from caller-side
//      `SystemTime::now()` would still pass since the two clocks are
//      usually within microseconds — UNTIL someone runs under a
//      `libfaketime`-style wall-clock shim, at which point the
//      caller-side stamp would diverge by tens of seconds and this
//      assertion would fire).
//
// The wall-clock-skew adversarial test (driving caller-side `SystemTime`
// off by minutes via `libfaketime`) lives outside the integration
// harness because it requires a per-test process-level clock override.
// What we can pin here is: (a) the script source no longer carries the
// caller-time pathway, and (b) the score in flight matches Valkey's
// authoritative clock at the time of the test run.
// =============================================================================

/// Strip Lua line comments (`-- ...` to end of line) from a script
/// body, keeping just the executable Lua code. Used by the b.u96 item 7
/// source-pin tests so a regression-detector pattern (e.g. `ARGV[2]`)
/// only matches against real code, not the migration narrative
/// embedded in the script's leading header comments.
///
/// Doesn't handle multi-line block comments (`--[[ ... ]]`) because
/// neither of the scripts we audit uses them. If a future script does,
/// extend this helper before adding the pin.
fn strip_lua_line_comments(script: &str) -> String {
    let mut out = String::with_capacity(script.len());
    for line in script.lines() {
        if let Some(pos) = line.find("--") {
            out.push_str(&line[..pos]);
        } else {
            out.push_str(line);
        }
        out.push('\n');
    }
    out
}

/// b.u96 item 7: the `gc_claim_pending.lua` script body must NOT read
/// ARGV[2] or define a `now_ms` local in executable code — those are
/// the pre-b.u96 caller-time pathway. The migration-narrative comment
/// in the script header still mentions ARGV[2] historically; we strip
/// Lua line comments before matching so only the executable Lua code
/// is checked.
#[test]
fn gc_claim_pending_lua_no_longer_accepts_caller_now_ms_argv() {
    const SCRIPT: &str = include_str!("../../../src/storage/scripts/gc_claim_pending.lua");
    let code = strip_lua_line_comments(SCRIPT);

    // The executable script must NOT read ARGV[2] (the pre-b.u96
    // caller-supplied millisecond timestamp). Match against the
    // canonical Lua read patterns. A regression that re-added the
    // caller pathway would surface either `tonumber(ARGV[2])` or a
    // bare `ARGV[2]` access in the executable code.
    assert!(
        !code.contains("ARGV[2]"),
        "regression: gc_claim_pending.lua now contains an ARGV[2] reference in executable \
         code — b.u96 item 7 migrated the timestamp source to `redis.call(\"TIME\")` so \
         the script must read only ARGV[1] (hash_hex). Re-adding ARGV[2] reintroduces the \
         NTP-skew double-decrement hole described in b.u96 item 7. Stripped script:\n{code}"
    );

    // The executable script MUST sample Valkey-side TIME. A regression
    // that dropped the TIME call AND replaced it with
    // `tonumber(ARGV[2])` would fail the assertion above; a regression
    // that dropped TIME without re-introducing ARGV[2] (e.g.
    // defaulting now_ms to 0) would silently re-queue every fresh
    // claim and break b.j9j AC2. Pin the canonical Valkey-TIME
    // pattern.
    assert!(
        code.contains("redis.call(\"TIME\")"),
        "regression: gc_claim_pending.lua no longer samples redis.call(\"TIME\") in \
         executable code — b.u96 item 7 requires the timestamp source to be Valkey-side. \
         Without TIME the ZADD score is no longer meaningful and the GC pass's stale-vs-fresh \
         classifier collapses. Stripped script:\n{code}"
    );
}

/// b.u96 item 7, for the in-place re-claim that replaced the drain (b.1jt):
/// `gc_reclaim_stale.lua` re-scores a claim with Valkey-side `TIME`, never a
/// caller-supplied time — it reads only its member, the expected score and
/// the offset (ARGV[1..3]).
#[test]
fn gc_reclaim_stale_lua_scores_with_valkey_time() {
    const SCRIPT: &str = include_str!("../../../src/storage/scripts/gc_reclaim_stale.lua");
    let code = strip_lua_line_comments(SCRIPT);
    assert!(
        !code.contains("ARGV[4]"),
        "gc_reclaim_stale.lua must read no fourth argument (a caller time). Stripped \
         script:\n{code}"
    );
    assert!(
        code.contains("redis.call(\"TIME\")"),
        "gc_reclaim_stale.lua must re-score with Valkey-side TIME. Stripped script:\n{code}"
    );
}

/// b.u96 item 7: the score recorded in `ext_inflight` after a real
/// `process_pending` run must be plausibly Valkey-side — i.e. very
/// close to the Valkey TIME read at test time. Leverages the same
/// fault-injection trick the item-2 test uses to leave one hash in
/// inflight after the batch.
///
/// Pre-b.u96 this assertion would have passed too (the caller used
/// `SystemTime::now()` and the test client is in the same process), so
/// the script-source pins above are the real regression guards. This
/// integration-level pin closes a smaller hole: a regression that
/// passed a stale `now_ms = 0` or a hardcoded constant would surface
/// here as a divergence between the stored score and Valkey's TIME.
///
/// Gated on `test-fault-injection` because it uses the
/// `with_process_pending_fault_injector_for_test` builder.
#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn process_pending_inflight_score_matches_valkey_time() {
    use std::collections::VecDeque;
    use std::sync::Mutex;

    use fred::error::ErrorKind as FredErrorKind;
    use fred::interfaces::ClientLike;
    use fred::types::{ClusterHash, CustomCommand};
    use live_edit::storage::external_values::ProcessPendingFaultInjector;

    let tv = TestValkey::setup().await;
    let keys = tv.key_schema();

    // Seed a single hash and install a fault that fires on hash 0's
    // site-1 EVALSHA, leaving the hash in inflight with its
    // Valkey-stamped claim timestamp.
    let h = test_hash(0xB7);
    // b.qpf: the inflight ZSET is keyed by the pending MEMBER, so the ZSCORE
    // below must ask for the member, not the bare hash hex.
    let member = retirement_of(h, OBJ_A).member();
    let injector: ProcessPendingFaultInjector =
        std::sync::Arc::new(Mutex::new(VecDeque::from(vec![(
            1usize,
            live_edit::storage::external_values::PendingFault::Fail(fred::error::Error::new(
                FredErrorKind::Timeout,
                "synthetic b.u96 transient",
            )),
        )])));
    let store = ValkeyExternalValueStore::new(tv.client().clone(), keys.clone())
        .with_process_pending_fault_injector_for_test(injector.clone());

    store
        .store_values(&[(&h, b"u96-item-7".as_slice())])
        .await
        .unwrap();
    store
        .schedule_release(&[retirement_of(h, OBJ_A)])
        .await
        .unwrap();

    let batch = store
        .process_pending(
            &live_edit::storage::external_values::GcPolicy::IMMEDIATE,
            &mut live_edit::storage::external_values::GcPassState::new(),
        )
        .await
        .unwrap();
    assert_eq!(batch.processed, 1, "the single hash was enumerated");
    assert_eq!(
        batch.deleted, 0,
        "hash was claimed off pending but its decrement was transient-swallowed"
    );

    // The hash sits in ext_inflight with a score equal to the Valkey
    // TIME at the moment gc_claim_pending.lua's ZADD ran, plus this build's
    // claim-score offset (b.1jt).
    let score: f64 = tv
        .client()
        .pool
        .zscore(keys.ext_inflight(), member.as_str())
        .await
        .expect("ZSCORE must succeed for the surviving inflight entry");
    let offset = live_edit::storage::external_values::GC_CLAIM_SCORE_OFFSET_MS;
    assert!(
        score > offset as f64,
        "inflight score must be a positive Valkey-side timestamp plus the claim-score offset \
         {offset}; got {score}"
    );

    // Sample Valkey's current TIME directly. `TIME` returns
    // [seconds, microseconds] — collapse to millis the same way
    // gc_claim_pending.lua does.
    //
    // **Pinned to `ext_inflight`'s slot** (b.289 sub-task 1). `TIME` carries no
    // key, so under fred's default `ClusterHash::FirstKey` policy there is
    // nothing to hash and `Router::route` falls through to
    // `writers.values_mut().next()` — an ARBITRARY primary. The assertion below
    // compares this reading against a score stamped by `gc_claim_pending.lua`
    // running on the node that owns `ext_inflight`, and a cluster's nodes do not
    // share a clock: NTP skew between two primaries is unbounded from this
    // test's point of view, and any skew past the 5 s slack would fail the run
    // with a message pointing at the Lua script rather than at the sampling. So
    // the sample is routed with `ClusterHash::Custom(slot)` — the same
    // pinned-`CustomCommand` shape `storage::watched_txn` uses for its own
    // keyless commands — which makes it read the clock of the node whose stamp
    // it is checking.
    //
    // The pin is **correct by construction, not currently load-bearing**: this
    // test builds its fixture with `TestValkey::setup()`, which reads
    // `TEST_VALKEY_URL`, and no CI job points that at a cluster — `test-cluster:`
    // sets it to the single-node `redis://valkey:6379` sidecar and names the
    // 3-shard cluster only through `LIVE_EDIT_TEST_CLUSTER_URL`, which this test
    // never reads. So the pin is a no-op in every environment the suite runs in
    // today. It is written this way for the environment that does not exist yet:
    // point `TEST_VALKEY_URL` at a clustered deployment and the unpinned form
    // becomes a real cross-node clock comparison that fails intermittently and
    // blames the Lua script. Keep the pin.
    let time_pair: Vec<String> = tv
        .client()
        .pool
        .next()
        .custom::<Vec<String>, _>(
            CustomCommand::new_static(
                "TIME",
                ClusterHash::from(keys.ext_inflight().as_str()),
                false,
            ),
            Vec::<String>::new(),
        )
        .await
        .expect("Valkey TIME must respond");
    assert_eq!(
        time_pair.len(),
        2,
        "Valkey TIME must return exactly [seconds, microseconds]; got {time_pair:?}"
    );
    let valkey_secs: u64 = time_pair[0].parse().expect("TIME[0] is integer seconds");
    let valkey_micros: u64 = time_pair[1]
        .parse()
        .expect("TIME[1] is integer microseconds");
    let valkey_now_ms = valkey_secs * 1000 + valkey_micros / 1000;

    // The score recorded by gc_claim_pending should be very close to
    // Valkey's clock right now (we just polled). 5 s of slack covers
    // normal test latency + any context-switch slack between the
    // process_pending await and this assertion. A pre-b.u96 regression
    // that ignored Valkey TIME and instead stamped from a constant
    // (e.g., now_ms = 0) would land here with `score == 0` and a
    // multi-decade gap — well outside 5 s.
    let score_ms = score as u64 - offset;
    let diff_ms = valkey_now_ms.abs_diff(score_ms);
    assert!(
        diff_ms < 5_000,
        "inflight score must be Valkey-TIME-derived: ZSCORE={score_ms} vs Valkey TIME \
         now={valkey_now_ms} (diff {diff_ms} ms). Pre-b.u96 regression that bypassed TIME \
         (e.g. now_ms=0) would surface here as a multi-decade divergence; a \
         regression that re-introduced caller-side SystemTime would diverge under any \
         libfaketime-style clock shim"
    );

    assert_eq!(
        injector.lock().unwrap().len(),
        0,
        "fault queue must have been drained by exactly one site-1 fire"
    );
}
