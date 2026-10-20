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

//! The structural cache's fenced cold start seen through `update_layer` on the
//! `lua` backend (structural-edits C1: `storage::structural_cache_fenced`, the
//! route in `storage::structural_cache_cold_start`, and the WATCH fallback's
//! classification in `LayerStore::update_layer_once_structural`).
//!
//! Every object is `snapshot_read_atomic_test`'s `every_shape_state`, created
//! through a store on the fixture's default-limit client. The store under test
//! runs on a second client whose plain item ceiling is half the object's
//! pre-read item bound (read off an early-out `TOO_LARGE` answer at run time),
//! with no byte ceiling to speak of, so the probe puts the object above the
//! plain ceilings and every cold start goes straight to the fenced read. That
//! store's configured RPC deadline is [`FENCE_WAIT_CAP`], and its
//! remaining-deadline function reports whatever the test sets
//! ([`Fx::set_remaining`]); `None` is "no deadline in scope". The M9 tests
//! build a second store on the same client and cache whose configured
//! deadline is their own ([`Fx::store_with_rpc_deadline`]).
//!
//! | Test | What it pins |
//! |---|---|
//! | `fenced_cold_start_*_commits_through_watch_for_that_call_only_*` | T9: a fenced read that fails for its call — three broken attempts (a raw header `seq` change while each attempt's lease is live, the attempt's first fan-out command held back by the fault seam), the waiting cap (no deadline left), or a capacity wait timing out (every fenced-class read permit held by the test) — commits through the WATCH path, ticks `uncacheable_total{reason="fenced_read_failed"}` once and no cold-start route, logs one DEBUG verdict line naming its cause, records no negative entry (the next lookup is a miss), gives every permit and grant back, and the next call caches the object on the fenced route |
//! | `fenced_cold_start_with_its_capacity_closed_*` | T9: the memory budget or the read gate closed (shutdown) answers the shutdown error, never the WATCH path, writing nothing |
//! | [`fenced_bound_layer_over_the_configured_deadline_is_recorded_over_deadline_and_its_next_edit_takes_watch_without_a_probe`] | M9: on a store whose configured RPC deadline its fenced read cannot fit (the deadline is the read's reserve at the probe's estimate), the first structural edit records `OverDeadline` before any read or acquire — WATCH commit, one `over_deadline` tick, one `uncacheable` route tick, one INFO line with the size and the deadline — and the second edit takes the WATCH path from the negative entry with no probe (`MONITOR`), no `cold` lookup, no read and no further tick or line. The M9 regression test |
//! | [`fenced_cold_start_short_of_its_calls_own_deadline_fails_for_that_call_only_and_records_no_over_deadline`] | M9: a call whose own deadline left is short, on an object whose fenced read fits the configured deadline, fails that read for itself (`fenced_read_failed`, cause `cap`), records nothing, and the next call cold-starts and caches it |
//! | [`plain_routed_layer_is_never_judged_over_deadline_and_loads_under_a_deadline_no_fenced_read_fits`] | M9: an object routed to the plain atomic read loads and commits on `structural_full` under a configured deadline the check would refuse a fenced-bound object at |
//! | [`fenced_cold_start_observes_its_read_permit_wait_under_the_structural_cache_consumer`] | T18's dynamic half: one fenced cold start records one `catchup_snapshot_read_wait_seconds{consumer="structural_cache",class="fenced"}` observation, reads once on `path="fenced"` and never on `path="atomic"`, and gives everything back |
//! | [`neither_a_fenced_cold_start_nor_a_cached_commit_writes_a_writer_intent`] | T12's cache half: a fenced cold start with a lingering fence key, its `structural_full` commit, and a cached commit write nothing to `obj:snapintent` (`MONITOR`) and register no intent |
//! | `*_registers_its_writer_intent_before_watch` / [`no_chs_complete_fallback_on_a_small_layer_registers_no_writer_intent`] | T15: with no fence key, an `OverWeight` above the item ceiling, a `FencedReadFailed` and an unmarked layer's `NoChsComplete` above the item ceiling classify the WATCH fallback, whose first attempt registers its intent before its `WATCH` (`MONITOR` order, `fence_intent_total{result="registered"}` +1); a small unmarked layer's `NoChsComplete` registers none |
//!
//! Elsewhere: the cold start's own routes, permits and grant, the deadline
//! check's predicate (`fenced_read_over_deadline`) and its verdict on a plain
//! read's `TOO_LARGE` answer in `structural_cache_cold_start_test`; the read
//! side's `above_ceiling` answers
//! in `structural_cache_read_test`; the WATCH path's intent rules on a
//! recently fenced layer in `snapshot_fence_writer_test`; the zero-seeded
//! vocabularies in `observability_metrics_test.rs`.
//!
//! # Wait budget
//!
//! Every call is under [`CALL_BUDGET`] (20 s). The broken case holds each of
//! its three attempts' first fan-out command back [`HELD_WINDOW`] (250 ms) and
//! sleeps out the writers' gap each broken release opens (about as long), so
//! its failing call takes under 2 s; the capacity-timeout case waits out a
//! cap of [`CAPACITY_CAP`] (600 ms). No test makes more than three calls. All
//! well under nextest's 120 s hang-killer.
//!
//! Only `fenced_cold_start_broken_on_every_attempt_*` and its `SeqBumper`
//! are gated behind `test-fault-injection`: the
//! snapshot-read delay seam exists only under that feature. Every other test
//! here runs without it.

use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

use bytes::Bytes;
use fred::interfaces::{HashesInterface, KeysInterface, StreamsInterface};
use oodf_ffi::OwnedDelta;
use serde_json::Value;

use live_edit::config::UpdateLayerBackend;
use live_edit::error::ServiceError;
use live_edit::observability::metrics::test_util::{render, sample_value};
use live_edit::observability::metrics::Metrics;
use live_edit::storage::client::ValkeyClient;
use live_edit::storage::diff_read::{
    read_object_snapshot_slim_with_ceilings, snapshot_item_bound, SlimSnapshotReadConsumer,
    SnapshotCeilings, SnapshotReadConsumer, SnapshotReadOf,
};
use live_edit::storage::layer::LayerStore;
use live_edit::storage::lua::STRUCTURAL_CACHE_SYNC_SCRIPT;
use live_edit::storage::snapshot_fence::writer::RemainingDeadlineFn;
use live_edit::storage::snapshot_fence::{self, ms_to_duration, LayerSize};
use live_edit::storage::structural_cache::{
    CacheLookup, ColdStartReadsInFlight, StructuralCache, UncacheableReason,
};
use live_edit::storage::structural_cache_cold_start::{
    fenced_read_over_deadline, STRUCTURAL_COLD_START_ROUTE_BYTES_PER_ITEM,
};
use live_edit::storage::structural_cache_fenced::wait_reserve;
use live_edit::storage::subscriber_admission::CatchUpReadPermit;
use live_edit::storage::test_support::TestValkey;

use crate::common::owned_delta::delta_to_ompf_bytes;
use crate::grpc::log_capture::{capture_debug, Captured};
use crate::storage::snapshot_read_atomic_test::{
    create, create_cmd, create_session, every_shape_state, layer_store, structural, unmark, ROOT,
};
use crate::storage::snapshot_read_support::{
    client_with_limits, monitor_fence, test_valkey_addr, MonitorTap, MonitoredCommand,
};
use crate::storage::structural_cache_cold_start_test::{
    assert_one_verdict_line, CEILING_FIELDS, WEIGHT_FIELDS,
};

// ═════════════════════════════════════════════════════════════════════════════
// Budgets and constants
// ═════════════════════════════════════════════════════════════════════════════

/// Deadline on one `update_layer` call: the healthy path is milliseconds; the
/// broken case's failing call is under 2 s.
const CALL_BUDGET: Duration = Duration::from_secs(20);
/// How long the fault seam holds each broken attempt's first fan-out command
/// back: long enough that the seq bumper, polling every [`BUMP_POLL`], sees
/// the attempt's live lease before the pass reads anything.
#[cfg(feature = "test-fault-injection")]
const HELD_WINDOW: Duration = Duration::from_millis(250);
/// The seq bumper's poll step.
#[cfg(feature = "test-fault-injection")]
const BUMP_POLL: Duration = Duration::from_millis(5);
/// The waiting cap the capacity-timeout case leaves its cold start: the call's
/// remaining deadline is set to `hold + S` at the estimate plus this.
const CAPACITY_CAP: Duration = Duration::from_millis(600);
/// How long the test waits for a fenced-class read permit it takes for itself:
/// they are all free.
const PERMIT_WAIT: Duration = Duration::from_secs(5);
/// A byte ceiling no fixture object comes near: the item ceiling alone puts
/// the object above the plain limits.
const NO_BYTE_LIMIT: u64 = 1 << 40;
/// A cache budget no fixture object comes near.
const BUDGET_BYTES: u64 = 64 * 1024 * 1024;
/// A cache budget whose per-entry cap every fixture object's weight lower
/// bound is over.
const TINY_BUDGET_BYTES: u64 = 1_024;
/// Retry budget of every store under test.
const MAX_ATTEMPTS: u32 = 8;
/// The fence-wait cap of every store under test (its own RPC deadline).
const FENCE_WAIT_CAP: Duration = Duration::from_secs(15);

const WRITER: &str = "c1-fenced-writer";
const CLIENT_TS: u64 = 11_000;
/// Section ids the seeded tree does not use, one per call.
const NEW_SECTIONS: [u64; 3] = [80, 81, 82];

/// The verdict line's message (`UNCACHEABLE_VERDICT_MSG`), restated: the
/// string operators search for.
const UNCACHEABLE_VERDICT_MSG: &str =
    "structural cache found the object uncacheable; its structural edits take the WATCH path";

/// `live_edit_update_layer_*` `path` labels and the cold-start vocabularies,
/// restated: they are the operator-facing contract.
const WATCH: &str = "watch";
const STRUCTURAL: &str = "structural";
const STRUCTURAL_FULL: &str = "structural_full";
const ROUTES: [&str; 3] = ["atomic", "fenced", "uncacheable"];
const REASONS: [&str; 5] = [
    "over_weight",
    "no_chs_complete",
    "reload_limit",
    "fenced_read_failed",
    "over_deadline",
];

// ═════════════════════════════════════════════════════════════════════════════
// Fixture
// ═════════════════════════════════════════════════════════════════════════════

/// One object, created on the default-limit client, and a `lua` store over it
/// on a client whose plain item ceiling is half the object's item bound.
struct Fx {
    tv: TestValkey,
    /// The lowered-limit client the store under test runs on, hooked onto
    /// [`Self::metrics`].
    client: ValkeyClient,
    metrics: Arc<Metrics>,
    cache: StructuralCache,
    ls: LayerStore,
    /// What the store's remaining-deadline function reports, in ns;
    /// [`NO_DEADLINE`] for `None`.
    remaining_ns: Arc<AtomicU64>,
    session_id: String,
    layer: String,
    oid: u64,
    /// The object's pre-read item bound as created.
    bound: u64,
}

/// [`Fx::remaining_ns`]'s encoding of "no deadline in scope".
const NO_DEADLINE: u64 = u64::MAX;

/// A `lua` store on `client` and `cache`, reporting to `metrics`, whose
/// configured RPC deadline is `rpc_deadline` and whose remaining-deadline
/// function reports `remaining_ns` ([`NO_DEADLINE`] as `None`).
fn lua_store(
    client: &ValkeyClient,
    tv: &TestValkey,
    cache: &StructuralCache,
    metrics: &Arc<Metrics>,
    rpc_deadline: Duration,
    remaining_ns: &Arc<AtomicU64>,
) -> LayerStore {
    let reads = Arc::clone(remaining_ns);
    let remaining: RemainingDeadlineFn = Arc::new(move || match reads.load(Ordering::SeqCst) {
        NO_DEADLINE => None,
        ns => Some(Duration::from_nanos(ns)),
    });
    LayerStore::new(client.clone(), tv.key_schema().clone(), 4096)
        .with_update_layer_backend(UpdateLayerBackend::Lua)
        .with_update_layer_retry_max_attempts(MAX_ATTEMPTS)
        .with_structural_cache(cache.clone())
        .with_metrics(Arc::clone(metrics))
        .with_fence_wait_deadline(rpc_deadline, remaining)
}

impl Fx {
    /// A marked object, above the store's plain item ceiling, on a cache that
    /// holds it.
    async fn new(layer: &str) -> Self {
        Self::build(layer, BUDGET_BYTES, |bound| bound / 2).await
    }

    /// `every_shape_state` as `layer`, and a store under test on a cache of
    /// `budget_bytes` over a client whose item ceiling is `max_items(bound)`.
    async fn build(layer: &str, budget_bytes: u64, max_items: impl FnOnce(u64) -> u64) -> Self {
        let tv = TestValkey::setup().await;
        let session_id = create_session(&tv, layer).await;
        let oid = create(&layer_store(&tv), &session_id, layer, &every_shape_state()).await;
        let bound = item_bound(&tv, oid).await;
        let max_items = max_items(bound);

        let metrics = Arc::new(crate::common::metrics_scrape_only(layer));
        let client = client_with_limits(max_items, NO_BYTE_LIMIT).await;
        client
            .install_observability_hooks(Arc::clone(&metrics))
            .await;
        let remaining_ns = Arc::new(AtomicU64::new(NO_DEADLINE));
        let cache = StructuralCache::new(budget_bytes);
        let ls = lua_store(
            &client,
            &tv,
            &cache,
            &metrics,
            FENCE_WAIT_CAP,
            &remaining_ns,
        );
        Self {
            tv,
            client,
            metrics,
            cache,
            ls,
            remaining_ns,
            session_id,
            layer: layer.to_owned(),
            oid,
            bound,
        }
    }

    /// Set what the store's remaining-deadline function reports from now on
    /// (every store this fixture builds reads the same setting).
    fn set_remaining(&self, remaining: Option<Duration>) {
        let ns = remaining.map_or(NO_DEADLINE, |d| {
            u64::try_from(d.as_nanos()).expect("a test deadline fits u64 ns")
        });
        self.remaining_ns.store(ns, Ordering::SeqCst);
    }

    /// A second `lua` store on this fixture's client, cache and registry
    /// whose configured RPC deadline (`LIVE_EDIT_RPC_DEADLINE_MS`, the
    /// store's fence-wait cap) is `rpc_deadline`; its remaining-deadline
    /// function reports what [`Self::set_remaining`] sets.
    fn store_with_rpc_deadline(&self, rpc_deadline: Duration) -> LayerStore {
        lua_store(
            &self.client,
            &self.tv,
            &self.cache,
            &self.metrics,
            rpc_deadline,
            &self.remaining_ns,
        )
    }

    /// The configured deadline at which this object's fenced read has no
    /// time left: exactly what its waiting cap keeps back at the probe's
    /// estimate, the read's expected time plus one WATCH attempt's
    /// (`wait_reserve`). At or below it the waiting cap is zero.
    fn no_fenced_read_fits(&self) -> Duration {
        wait_reserve(&snapshot_fence::coefficients(&self.client), self.estimate())
    }

    /// The size the probe gives the cold start's fenced read: the bound, and
    /// the bound times the route predictor's bytes per item.
    fn estimate(&self) -> LayerSize {
        LayerSize {
            items: self.bound,
            bytes: self
                .bound
                .saturating_mul(STRUCTURAL_COLD_START_ROUTE_BYTES_PER_ITEM),
        }
    }

    fn oid_str(&self) -> String {
        self.oid.to_string()
    }

    fn header_key(&self) -> String {
        self.tv.key_schema().obj_header(&self.oid_str())
    }

    fn fence_key(&self) -> String {
        self.tv.key_schema().obj_snapshot_fence(&self.oid_str())
    }

    fn intent_key(&self) -> String {
        self.tv.key_schema().obj_snapshot_intent(&self.oid_str())
    }

    /// One structural `update_layer` call creating `section` under the root,
    /// relaying the delta's OMPF, under [`CALL_BUDGET`].
    async fn update(&self, section: u64) -> Result<(u64, u64), ServiceError> {
        self.update_on(&self.ls, section).await
    }

    /// The store call is boxed: its WATCH-path future is too large to hold
    /// inline in this helper's and its callers' frames on a debug build's
    /// test-thread stack.
    async fn update_on(&self, ls: &LayerStore, section: u64) -> Result<(u64, u64), ServiceError> {
        let delta: OwnedDelta = structural(vec![create_cmd(ROOT, section, "C1")]);
        let relay = Some(Bytes::from(delta_to_ompf_bytes(&delta, &[])));
        tokio::time::timeout(
            CALL_BUDGET,
            Box::pin(ls.update_layer(
                &self.session_id,
                &self.layer,
                self.oid,
                Some(delta),
                relay,
                WRITER,
                CLIENT_TS,
            )),
        )
        .await
        .expect("update_layer did not settle within the call budget")
    }

    async fn header_seq(&self) -> u64 {
        let seq: Option<String> = self
            .tv
            .client()
            .pool
            .hget(self.header_key(), "seq")
            .await
            .expect("HGET obj:hdr seq must succeed");
        seq.expect("the header has a seq")
            .parse()
            .expect("the header seq is decimal")
    }

    /// The layer stream's commits by [`WRITER`].
    async fn commits(&self) -> usize {
        let entries: Vec<(String, HashMap<String, Bytes>)> = self
            .tv
            .client()
            .pool
            .xrange_values(self.tv.key_schema().layer_stream(self.oid), "-", "+", None)
            .await
            .expect("XRANGE over the layer stream must succeed");
        entries
            .iter()
            .filter_map(|(_, fields)| fields.get("payload"))
            .map(|p| String::from_utf8_lossy(p).into_owned())
            .filter(|p| p.starts_with("update|") && p.split('|').nth(3) == Some(WRITER))
            .count()
    }

    async fn exists(&self, key: String) -> bool {
        let n: i64 = self
            .tv
            .client()
            .pool
            .exists(key)
            .await
            .expect("EXISTS must succeed");
        n == 1
    }

    // ── Metrics ──────────────────────────────────────────────────────────────

    fn cell(&self, name: &str, labels: &[(&str, &str)]) -> u64 {
        sample_value(&render(&self.metrics), name, labels).unwrap_or(0.0) as u64
    }

    /// Commits on `path` (`live_edit_update_layer_attempts_histogram_count`).
    fn commits_on(&self, path: &str) -> u64 {
        self.cell(
            "live_edit_update_layer_attempts_histogram_count",
            &[("path", path)],
        )
    }

    fn cold_starts(&self) -> BTreeMap<&'static str, u64> {
        ROUTES
            .iter()
            .map(|&route| {
                (
                    route,
                    self.cell(
                        "live_edit_structural_cold_start_reads_total",
                        &[("route", route)],
                    ),
                )
            })
            .collect()
    }

    fn verdicts(&self) -> BTreeMap<&'static str, u64> {
        REASONS
            .iter()
            .map(|&reason| {
                (
                    reason,
                    self.cell(
                        "live_edit_structural_cache_uncacheable_total",
                        &[("reason", reason)],
                    ),
                )
            })
            .collect()
    }

    /// `live_edit_snapshot_read_total{consumer="structural_cache",path,outcome}`.
    fn reads(&self, path: &str, outcome: &str) -> u64 {
        self.cell(
            "live_edit_snapshot_read_total",
            &[
                ("consumer", "structural_cache"),
                ("path", path),
                ("outcome", outcome),
            ],
        )
    }

    /// Every `structural_cache` read on `path`, whatever its outcome.
    fn reads_on(&self, path: &str) -> u64 {
        [
            "ok",
            "not_found",
            "too_large",
            "transient_error",
            "permanent_error",
            "fence_broken",
        ]
        .iter()
        .map(|outcome| self.reads(path, outcome))
        .sum()
    }

    fn intents(&self, result: &str) -> u64 {
        self.cell(
            "live_edit_update_layer_fence_intent_total",
            &[("result", result)],
        )
    }

    /// `live_edit_structural_cache_lookups_total{outcome}`: `cold` is a
    /// lookup that found no entry and went on to probe and cold-start; a
    /// lookup routed by a negative entry ticks no outcome.
    fn lookups(&self, outcome: &str) -> u64 {
        self.cell(
            "live_edit_structural_cache_lookups_total",
            &[("outcome", outcome)],
        )
    }

    // ── Assertions ───────────────────────────────────────────────────────────

    /// The cold start holds nothing: no cache read permit, every fenced-class
    /// read permit back but `held_by_test`, no memory granted.
    fn assert_everything_released(&self, held_by_test: usize, context: &str) {
        assert_eq!(
            self.cache.cold_start_reads_in_flight_for_test(),
            ColdStartReadsInFlight {
                plain: 0,
                fenced: 0,
                raised: 0,
            },
            "{context}: the cache's cold-start permits are all back"
        );
        let gate = &self.client.catchup_fenced_read_gate;
        assert_eq!(
            gate.available() + held_by_test,
            gate.limit(),
            "{context}: every fenced-class read permit the cold start took is back"
        );
        assert_eq!(
            self.client.catchup_build_budget.reserved_bytes(),
            0,
            "{context}: the cold start's memory grant is back (T = {} bytes)",
            self.client.catchup_build_budget.total_bytes()
        );
    }

    /// One call whose fenced cold start failed with `cause`: it committed once
    /// through the WATCH path, ticked `fenced_read_failed` once and no
    /// cold-start route, logged one DEBUG verdict line naming `cause`,
    /// recorded no negative entry, and gave every permit and grant back
    /// (`held_by_test` read permits aside).
    async fn assert_failed_for_the_call(
        &self,
        result: Result<(u64, u64), ServiceError>,
        captured: &Captured,
        cause: &str,
        held_by_test: usize,
    ) {
        let (seq, _) =
            result.unwrap_or_else(|e| panic!("{cause}: the call must commit; got {e:?}"));
        assert_eq!(seq, self.header_seq().await, "{cause}: the call's commit");
        assert_eq!(self.commits().await, 1, "{cause}: exactly one commit");
        assert_eq!(
            (
                self.commits_on(WATCH),
                self.commits_on(STRUCTURAL_FULL),
                self.commits_on(STRUCTURAL)
            ),
            (1, 0, 0),
            "{cause}: the call committed through the WATCH path"
        );
        assert_eq!(
            self.verdicts(),
            BTreeMap::from([
                ("over_weight", 0),
                ("no_chs_complete", 0),
                ("reload_limit", 0),
                ("fenced_read_failed", 1),
                ("over_deadline", 0),
            ]),
            "{cause}: one fenced_read_failed verdict, for the call, and no over_deadline \
             verdict for the object"
        );
        assert_eq!(
            self.cold_starts(),
            BTreeMap::from([("atomic", 0), ("fenced", 0), ("uncacheable", 0)]),
            "{cause}: a fenced read that failed for its call ticks no cold-start route"
        );
        assert!(
            matches!(self.cache.lookup(self.oid), CacheLookup::Miss),
            "{cause}: no negative entry was recorded, so a lookup is a miss"
        );
        let lines = captured.with_message(UNCACHEABLE_VERDICT_MSG);
        assert_eq!(
            lines.len(),
            1,
            "{cause}: one verdict line; full capture:\n{}",
            captured.text
        );
        let line = lines[0];
        assert_eq!(line["level"], "DEBUG", "{cause}: its rate is unbounded");
        for (field, want) in [
            ("reason", Value::from("fenced_read_failed")),
            ("cause", Value::from(cause)),
            ("object_id", Value::from(self.oid)),
            ("session_id", Value::from(self.session_id.as_str())),
            ("layer_name", Value::from(self.layer.as_str())),
        ] {
            assert_eq!(
                line["fields"][field], want,
                "{cause}: the verdict line's {field}; line: {line}"
            );
        }
        for field in [
            "attempts",
            "size_items",
            "size_bytes",
            "waited_ms",
            "cap_ms",
        ] {
            assert!(
                line["fields"].get(field).is_some(),
                "{cause}: the verdict line carries {field}; line: {line}"
            );
        }
        self.assert_everything_released(held_by_test, cause);
    }

    /// The call after a failed one cold-starts the object afresh on the fenced
    /// route and commits on `structural_full`; its entry is then cached.
    async fn assert_next_call_caches(&self, section: u64, context: &str) {
        let (seq, _) = self
            .update(section)
            .await
            .unwrap_or_else(|e| panic!("{context}: the next call must commit; got {e:?}"));
        assert_eq!(
            (self.commits_on(STRUCTURAL_FULL), self.commits_on(WATCH)),
            (1, 1),
            "{context}: the next call cold-started and committed on the cache"
        );
        assert_eq!(
            self.cold_starts(),
            BTreeMap::from([("atomic", 0), ("fenced", 1), ("uncacheable", 0)]),
            "{context}: the next call's cold start took the fenced read"
        );
        assert_eq!(
            self.verdicts()["fenced_read_failed"],
            1,
            "{context}: the earlier verdict was the earlier call's only"
        );
        match self.cache.lookup(self.oid) {
            CacheLookup::Hit(guard) => assert_eq!(
                guard.entry().seq,
                seq,
                "{context}: the cached entry stands at the commit"
            ),
            CacheLookup::Miss => panic!("{context}: the object must be cached; got a miss"),
            CacheLookup::Uncacheable { reason } => {
                panic!("{context}: the object must be cached; got Uncacheable({reason:?})")
            }
        }
        self.assert_everything_released(0, context);
    }

    // ── MONITOR ──────────────────────────────────────────────────────────────

    /// Everything `tap` recorded up to a marker issued now.
    async fn drain_to_marker(&self, tap: &mut MonitorTap) -> Vec<MonitoredCommand> {
        let marker = monitor_fence(self.tv.prefix());
        let _: i64 = self.tv.client().pool.exists(&marker).await.unwrap();
        tap.drain_until(&marker).await
    }

    /// A `snapshot_intent_register.lua` `EVALSHA` on this object: `KEYS` =
    /// intent, fence, header.
    fn is_register_eval(&self) -> impl Fn(&MonitoredCommand) -> bool {
        let intent = self.intent_key();
        let fence = self.fence_key();
        move |c| {
            c.name() == "EVALSHA" && c.args.get(3) == Some(&intent) && c.args.get(4) == Some(&fence)
        }
    }

    /// The WATCH path's `WATCH` of this object's header.
    fn is_watch(&self) -> impl Fn(&MonitoredCommand) -> bool {
        let header = self.header_key();
        move |c| !c.issued_by_script() && c.name() == "WATCH" && c.args.get(1) == Some(&header)
    }

    /// A `structural_cache_sync.lua` `EVALSHA` on this object — the read
    /// side's probe before a cold start, or a repair read: `KEYS[1]` is the
    /// object's header.
    fn is_sync_eval(&self) -> impl Fn(&MonitoredCommand) -> bool {
        let sha = STRUCTURAL_CACHE_SYNC_SCRIPT.sha1().to_string();
        let header = self.header_key();
        move |c| {
            !c.issued_by_script()
                && c.name() == "EVALSHA"
                && c.args.get(1) == Some(&sha)
                && c.args.get(3) == Some(&header)
        }
    }
}

/// The object's pre-read item bound, as the cold start's probe computes it
/// (`snapshot_item_bound` over the five cardinalities), read off an early-out
/// `TOO_LARGE` answer at an item ceiling of 1 on a client reporting nowhere.
async fn item_bound(tv: &TestValkey, oid: u64) -> u64 {
    let client = client_with_limits(1, NO_BYTE_LIMIT).await;
    match read_object_snapshot_slim_with_ceilings(
        &client,
        tv.key_schema(),
        &oid.to_string(),
        SlimSnapshotReadConsumer::StructuralCacheAudit,
        SnapshotCeilings::of_client(&client),
    )
    .await
    .expect("the bound read must answer")
    {
        SnapshotReadOf::TooLarge(too_large) => {
            let bound = snapshot_item_bound(&too_large.cardinalities);
            assert!(
                bound >= 4,
                "fixture: a bound of {bound} cannot be halved into a ceiling below it"
            );
            bound
        }
        SnapshotReadOf::Snapshot(_) => {
            panic!("fixture: an item ceiling of 1 must refuse the object")
        }
    }
}

fn position(seen: &[MonitoredCommand], what: &str, p: impl Fn(&MonitoredCommand) -> bool) -> usize {
    seen.iter()
        .position(p)
        .unwrap_or_else(|| panic!("the MONITOR record holds no {what}: {seen:#?}"))
}

/// Takes every fenced-class read permit of `client` for the test.
async fn hold_every_fenced_read_permit(client: &ValkeyClient) -> Vec<CatchUpReadPermit> {
    let gate = &client.catchup_fenced_read_gate;
    let mut held = Vec::with_capacity(gate.limit());
    for _ in 0..gate.limit() {
        held.push(
            gate.acquire(
                tokio::time::Instant::now() + PERMIT_WAIT,
                SnapshotReadConsumer::Catchup,
            )
            .await
            .unwrap_or_else(|_| panic!("fixture: a free fenced-class read permit")),
        );
    }
    assert_eq!(gate.available(), 0, "fixture: every permit is held");
    held
}

/// A background task that moves the header's `seq` once for each new reader
/// lease it sees on the fence key, while that lease is live: the attempt
/// holding it then reads the header off its cut, or its release finds the
/// header moved — either way the attempt is broken. Needs
/// `test-fault-injection`, as its one user does.
#[cfg(feature = "test-fault-injection")]
struct SeqBumper {
    stop: tokio_util::sync::CancellationToken,
    task: tokio::task::JoinHandle<u64>,
}

#[cfg(feature = "test-fault-injection")]
impl SeqBumper {
    fn start(fx: &Fx) -> Self {
        let stop = tokio_util::sync::CancellationToken::new();
        let pool = fx.tv.client().pool.clone();
        let (fence, header) = (fx.fence_key(), fx.header_key());
        let task = tokio::spawn({
            let stop = stop.clone();
            async move {
                let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
                let mut bumps = 0;
                while !stop.is_cancelled() {
                    let fields: HashMap<String, String> = pool
                        .hgetall(&fence)
                        .await
                        .expect("the bumper's HGETALL must succeed");
                    for reader in fields.keys().filter(|k| k.starts_with("r:")) {
                        if seen.insert(reader.clone()) {
                            let _: i64 = pool
                                .hincrby(&header, "seq", 1)
                                .await
                                .expect("the bumper's HINCRBY must succeed");
                            bumps += 1;
                        }
                    }
                    tokio::time::sleep(BUMP_POLL).await;
                }
                bumps
            }
        });
        Self { stop, task }
    }

    /// Stop the task; the number of leases it broke.
    async fn stop(self) -> u64 {
        self.stop.cancel();
        self.task.await.expect("the bumper must not panic")
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// T9 — a fenced read that fails falls back to WATCH for that call only
// ═════════════════════════════════════════════════════════════════════════════

/// Three broken attempts: each attempt's first fan-out command is held back
/// [`HELD_WINDOW`] by the fault seam, and while its lease is live the header's
/// `seq` moves, so every attempt is broken (`fence_broken` three times, never
/// `ok`). After `FENCE_READ_ATTEMPTS` the read is `FencedReadFailed` with
/// cause `broken` for this call only: the call commits through the WATCH path
/// and the next call caches the object (the seam's queue is empty by then).
///
/// Needs `test-fault-injection` (the read's fault seam).
#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn fenced_cold_start_broken_on_every_attempt_commits_through_watch_for_that_call_only_and_the_next_call_caches(
) {
    use live_edit::storage::diff_read::{
        ObjectSnapshotReadFaultInjector, ObjectSnapshotReadFaultTarget, ObjectSnapshotReader,
    };
    use live_edit::storage::snapshot_fence::FENCE_READ_ATTEMPTS;
    let fx = Fx::new("c1-fenced-broken").await;
    let seam = Arc::new(ObjectSnapshotReadFaultInjector::new());
    fx.client
        .install_object_snapshot_read_fault_injector_for_test(Arc::clone(&seam));
    for _ in 0..FENCE_READ_ATTEMPTS {
        seam.push_delay_at(
            ObjectSnapshotReadFaultTarget::nth_command(ObjectSnapshotReader::FencedFanout, 0),
            HELD_WINDOW,
        );
    }
    let bumper = SeqBumper::start(&fx);

    let (result, captured) = capture_debug(Box::pin(fx.update(NEW_SECTIONS[0]))).await;

    let bumps = bumper.stop().await;
    assert_eq!(
        (bumps, seam.delays_remaining()),
        (u64::from(FENCE_READ_ATTEMPTS), 0),
        "fixture: every attempt was held back and had its lease's header moved"
    );
    assert_eq!(
        (fx.reads("fenced", "fence_broken"), fx.reads("fenced", "ok")),
        (u64::from(FENCE_READ_ATTEMPTS), 0),
        "every fenced attempt was broken, none served"
    );
    assert_eq!(
        fx.reads_on("atomic"),
        0,
        "the probe put the object above the plain ceilings: no plain read"
    );
    fx.assert_failed_for_the_call(result, &captured, "broken", 0)
        .await;
    fx.assert_next_call_caches(NEW_SECTIONS[1], "after three broken attempts")
        .await;
}

/// The waiting cap: no deadline left (`remaining` is zero, below `hold + S`),
/// so the hook refuses before any permit is taken or any acquire sent — no
/// fenced read is counted and no fence key is created. The call commits
/// through the WATCH path; with the deadline gone again the next call caches.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn fenced_cold_start_past_its_waiting_cap_commits_through_watch_for_that_call_only_and_the_next_call_caches(
) {
    let fx = Fx::new("c1-fenced-cap").await;
    fx.set_remaining(Some(Duration::ZERO));

    let (result, captured) = capture_debug(Box::pin(fx.update(NEW_SECTIONS[0]))).await;

    assert_eq!(
        fx.reads_on("fenced") + fx.reads_on("atomic"),
        0,
        "the cap refused before any read or acquire"
    );
    assert!(
        !fx.exists(fx.fence_key()).await,
        "no acquire was sent, so no fence key exists"
    );
    fx.assert_failed_for_the_call(result, &captured, "cap", 0)
        .await;
    fx.set_remaining(None);
    fx.assert_next_call_caches(NEW_SECTIONS[1], "after the cap")
        .await;
}

/// A capacity wait timing out: the test holds every fenced-class read permit
/// (`P_f`) past the cold start's cap — set to about [`CAPACITY_CAP`] by the
/// call's remaining deadline — so the wait for `P_f` times out
/// (`capacity_timeout`), with no acquire sent. The call commits through the
/// WATCH path; once the permits are back the next call caches.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn fenced_cold_start_whose_capacity_wait_times_out_commits_through_watch_for_that_call_only_and_the_next_call_caches(
) {
    let fx = Fx::new("c1-fenced-capacity").await;
    let timing = snapshot_fence::coefficients(&fx.client).timing(fx.estimate());
    let reserve = ms_to_duration(timing.hold_ms.max(0.0) + timing.structural_attempt_ms.max(0.0));
    fx.set_remaining(Some(reserve + CAPACITY_CAP));
    let held = hold_every_fenced_read_permit(&fx.client).await;

    let (result, captured) = capture_debug(Box::pin(fx.update(NEW_SECTIONS[0]))).await;

    assert_eq!(
        fx.reads_on("fenced"),
        0,
        "no acquire was sent while P_f was held"
    );
    fx.assert_failed_for_the_call(result, &captured, "capacity_timeout", held.len())
        .await;
    drop(held);
    fx.set_remaining(None);
    fx.assert_next_call_caches(NEW_SECTIONS[1], "after the capacity timeout")
        .await;
}

/// What closes a fenced cold start's capacity at shutdown.
#[derive(Debug, Clone, Copy)]
enum Closer {
    /// The catch-up memory budget `T`.
    Budget,
    /// The fenced-class read gate (`P_f`).
    ReadGate,
}

/// Closed capacity (the replica shutting down) answers the shutdown error —
/// `Unavailable`, retryable — and never falls back to the WATCH path: nothing
/// is written, no commit is counted on any path, no verdict is reported and
/// no negative entry recorded, and the cache's permits are back.
async fn closed_capacity_answers_the_shutdown_error(closer: Closer) {
    let fx = Fx::new(&format!("c1-fenced-closed-{closer:?}")).await;
    let seq_before = fx.header_seq().await;
    match closer {
        Closer::Budget => fx.client.catchup_build_budget.close(),
        Closer::ReadGate => fx.client.catchup_fenced_read_gate.close(),
    }

    let result = fx.update(NEW_SECTIONS[0]).await;

    match &result {
        Err(ServiceError::Unavailable(message)) if message.contains("shutting down") => {}
        other => panic!("{closer:?}: expected the shutdown Unavailable; got {other:?}"),
    }
    assert_eq!(
        (fx.header_seq().await, fx.commits().await),
        (seq_before, 0),
        "{closer:?}: nothing was written"
    );
    assert_eq!(
        (
            fx.commits_on(WATCH),
            fx.commits_on(STRUCTURAL_FULL),
            fx.commits_on(STRUCTURAL)
        ),
        (0, 0, 0),
        "{closer:?}: no path committed, the WATCH path included"
    );
    assert_eq!(
        fx.verdicts()["fenced_read_failed"],
        0,
        "{closer:?}: shutdown is not a failed fenced read"
    );
    assert_eq!(
        fx.cold_starts(),
        BTreeMap::from([("atomic", 0), ("fenced", 0), ("uncacheable", 0)]),
        "{closer:?}: no cold-start route"
    );
    assert!(
        matches!(fx.cache.lookup(fx.oid), CacheLookup::Miss),
        "{closer:?}: nothing recorded"
    );
    assert_eq!(
        fx.cache.cold_start_reads_in_flight_for_test(),
        ColdStartReadsInFlight {
            plain: 0,
            fenced: 0,
            raised: 0,
        },
        "{closer:?}: the cache's cold-start permits are back"
    );
    assert_eq!(
        fx.client.catchup_build_budget.reserved_bytes(),
        0,
        "{closer:?}: no memory is held"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn fenced_cold_start_with_its_capacity_closed_by_the_budget_answers_the_shutdown_error_never_watch(
) {
    closed_capacity_answers_the_shutdown_error(Closer::Budget).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn fenced_cold_start_with_its_capacity_closed_by_the_read_gate_answers_the_shutdown_error_never_watch(
) {
    closed_capacity_answers_the_shutdown_error(Closer::ReadGate).await;
}

// ═════════════════════════════════════════════════════════════════════════════
// M9 — the `over_deadline` verdict, judged at the configured RPC deadline
// ═════════════════════════════════════════════════════════════════════════════

/// A fenced-bound object whose fenced read cannot fit the store's configured
/// RPC deadline — the deadline is exactly what the read's waiting cap keeps
/// back at the probe's estimate, so the cap is zero even with the whole
/// deadline left, as it is here — is recorded `OverDeadline` by its first
/// structural edit, before any fenced read: the edit commits through the
/// WATCH path, ticks `uncacheable_total{reason="over_deadline"}` and the
/// `uncacheable` cold-start route once, logs one INFO verdict line naming the
/// size it was judged at and the deadline, sends no fence acquire and reads
/// nothing, and leaves a negative entry. The second edit is routed by that
/// entry: it commits through the WATCH path with no probe (no
/// `structural_cache_sync.lua` call on the object in the `MONITOR` record, no
/// `cold` lookup), no read, no cold start and no further verdict tick or
/// line.
///
/// The regression test of M9: without the deadline check each edit probes,
/// fails its fenced read for itself (`fenced_read_failed`, cause `cap`) and
/// records nothing, so the next edit probes again (checked by disabling the
/// check in place: the first edit ticks `fenced_read_failed`, not
/// `over_deadline`).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn fenced_bound_layer_over_the_configured_deadline_is_recorded_over_deadline_and_its_next_edit_takes_watch_without_a_probe(
) {
    let fx = Fx::new("m9-over-deadline").await;
    let rpc_deadline = fx.no_fenced_read_fits();
    let coefficients = snapshot_fence::coefficients(&fx.client);
    assert!(
        fenced_read_over_deadline(&coefficients, rpc_deadline, fx.estimate())
            && !fenced_read_over_deadline(&coefficients, FENCE_WAIT_CAP, fx.estimate()),
        "fixture: the object's fenced read fits the fixture's own deadline but not \
         {rpc_deadline:?}"
    );
    // The whole configured deadline is left, as when a call starts.
    fx.set_remaining(Some(rpc_deadline));
    let ls = fx.store_with_rpc_deadline(rpc_deadline);
    let mut tap = MonitorTap::start(&test_valkey_addr()).await;

    let (first, captured) = capture_debug(Box::pin(fx.update_on(&ls, NEW_SECTIONS[0]))).await;
    let first_commands = fx.drain_to_marker(&mut tap).await;

    first.expect("the first edit must commit");
    assert_eq!(
        (
            fx.commits_on(WATCH),
            fx.commits_on(STRUCTURAL_FULL),
            fx.commits_on(STRUCTURAL)
        ),
        (1, 0, 0),
        "the first edit committed through the WATCH path"
    );
    assert_eq!(
        fx.verdicts(),
        BTreeMap::from([
            ("over_weight", 0),
            ("no_chs_complete", 0),
            ("reload_limit", 0),
            ("fenced_read_failed", 0),
            ("over_deadline", 1),
        ]),
        "one over_deadline verdict, and no per-call fenced_read_failed"
    );
    assert_eq!(
        fx.cold_starts(),
        BTreeMap::from([("atomic", 0), ("fenced", 0), ("uncacheable", 1)]),
        "the recorded verdict ends its cold start on the uncacheable route"
    );
    assert_eq!(
        fx.reads_on("fenced") + fx.reads_on("atomic"),
        0,
        "judged before any read: no fenced read and no plain read"
    );
    assert!(
        !fx.exists(fx.fence_key()).await,
        "no fence acquire was sent, so no fence key exists"
    );
    match fx.cache.lookup(fx.oid) {
        CacheLookup::Uncacheable {
            reason: UncacheableReason::OverDeadline,
        } => {}
        other => panic!("the OverDeadline negative entry must be recorded; got {other:?}"),
    }
    assert!(
        first_commands.iter().any(|c| fx.is_sync_eval()(c)),
        "vacuity guard: the first edit probed the object before its cold start"
    );
    let reserve = wait_reserve(&coefficients, fx.estimate());
    let mut absent = WEIGHT_FIELDS.to_vec();
    absent.extend(CEILING_FIELDS);
    absent.extend(["cause", "attempts", "waited_ms", "cap_ms"]);
    assert_one_verdict_line(
        &captured,
        fx.oid,
        "over_deadline",
        &[
            ("session_id", fx.session_id.as_str().into()),
            ("layer_name", fx.layer.as_str().into()),
            ("size_items", fx.estimate().items.into()),
            ("size_bytes", fx.estimate().bytes.into()),
            ("rpc_deadline_ms", (rpc_deadline.as_millis() as u64).into()),
            ("reserve_ms", (reserve.as_millis() as u64).into()),
        ],
        &absent,
    );
    fx.assert_everything_released(0, "after the over_deadline verdict");
    assert_eq!(
        fx.lookups("cold"),
        1,
        "fixture: the first edit's cold lookup"
    );

    let (second, captured) = capture_debug(Box::pin(fx.update_on(&ls, NEW_SECTIONS[1]))).await;
    let second_commands = fx.drain_to_marker(&mut tap).await;

    second.expect("the second edit must commit");
    assert_eq!(
        (
            fx.commits_on(WATCH),
            fx.commits_on(STRUCTURAL_FULL),
            fx.commits_on(STRUCTURAL)
        ),
        (2, 0, 0),
        "the second edit committed through the WATCH path"
    );
    assert!(
        !second_commands.iter().any(|c| fx.is_sync_eval()(c)),
        "the second edit sent no structural_cache_sync.lua call on the object: no probe; \
         MONITOR record: {second_commands:#?}"
    );
    assert_eq!(
        fx.lookups("cold"),
        1,
        "the second lookup found the negative entry, not a miss"
    );
    assert_eq!(
        (
            fx.verdicts()["over_deadline"],
            fx.verdicts()["fenced_read_failed"]
        ),
        (1, 0),
        "the verdict ticks once per verdict, not per edit"
    );
    assert_eq!(
        fx.cold_starts(),
        BTreeMap::from([("atomic", 0), ("fenced", 0), ("uncacheable", 1)]),
        "the second edit ran no cold start"
    );
    assert_eq!(
        fx.reads_on("fenced") + fx.reads_on("atomic"),
        0,
        "the second edit read nothing for the cache"
    );
    assert!(
        captured.with_message(UNCACHEABLE_VERDICT_MSG).is_empty(),
        "an already reported verdict logs no second line; full capture:\n{}",
        captured.text
    );
    match fx.cache.lookup(fx.oid) {
        CacheLookup::Uncacheable {
            reason: UncacheableReason::OverDeadline,
        } => {}
        other => panic!("the negative entry must still stand; got {other:?}"),
    }
}

/// M9's per-call condition: a call left too little of its own deadline for
/// the fenced read — what remains is exactly the read's reserve, so its
/// waiting cap is zero — on an object whose fenced read fits the store's
/// configured deadline fails that read for itself (`fenced_read_failed`,
/// cause `cap`) and records nothing: no `over_deadline` tick, no negative
/// entry (a lookup is a miss), and the next call, with the whole deadline
/// left, cold-starts the object again and caches it on the fenced route.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn fenced_cold_start_short_of_its_calls_own_deadline_fails_for_that_call_only_and_records_no_over_deadline(
) {
    let fx = Fx::new("m9-short-remaining").await;
    let short = fx.no_fenced_read_fits();
    assert!(
        !fenced_read_over_deadline(
            &snapshot_fence::coefficients(&fx.client),
            FENCE_WAIT_CAP,
            fx.estimate()
        ),
        "fixture: the object's fenced read fits the configured deadline"
    );
    fx.set_remaining(Some(short));

    let (result, captured) = capture_debug(Box::pin(fx.update(NEW_SECTIONS[0]))).await;

    // The verdict map it checks includes `over_deadline: 0`.
    fx.assert_failed_for_the_call(result, &captured, "cap", 0)
        .await;
    fx.set_remaining(Some(FENCE_WAIT_CAP));
    fx.assert_next_call_caches(NEW_SECTIONS[1], "after a call short of its own deadline")
        .await;
    assert_eq!(
        fx.verdicts()["over_deadline"],
        0,
        "neither call judged the object over the configured deadline"
    );
}

/// M9's scope: the deadline check guards the fenced read only. An object the
/// route predictor sends to the plain atomic read (within the plain
/// ceilings) loads into the cache and commits on `structural_full` on a store
/// whose configured deadline is one its fenced read could not fit — so had
/// the check judged it, it would have been refused. No `over_deadline` tick,
/// no fenced read, and a cache hit afterwards.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn plain_routed_layer_is_never_judged_over_deadline_and_loads_under_a_deadline_no_fenced_read_fits(
) {
    let fx = Fx::build("m9-plain-routed", BUDGET_BYTES, |_| {
        live_edit::config::SNAPSHOT_ATOMIC_READ_MAX_ITEMS_DEFAULT
    })
    .await;
    assert!(
        fx.bound <= live_edit::config::SNAPSHOT_ATOMIC_READ_MAX_ITEMS_DEFAULT,
        "fixture: the layer is within the plain item ceiling"
    );
    let rpc_deadline = fx.no_fenced_read_fits();
    assert!(
        fenced_read_over_deadline(
            &snapshot_fence::coefficients(&fx.client),
            rpc_deadline,
            fx.estimate()
        ),
        "fixture: at {rpc_deadline:?} the deadline check would refuse the object were it \
         fenced-bound"
    );
    fx.set_remaining(Some(rpc_deadline));
    let ls = fx.store_with_rpc_deadline(rpc_deadline);

    let (seq, _) = fx
        .update_on(&ls, NEW_SECTIONS[0])
        .await
        .expect("the call must commit");

    assert_eq!(
        (fx.commits_on(STRUCTURAL_FULL), fx.commits_on(WATCH)),
        (1, 0),
        "the call cold-started and committed on the cache"
    );
    assert_eq!(
        fx.cold_starts(),
        BTreeMap::from([("atomic", 1), ("fenced", 0), ("uncacheable", 0)]),
        "the plain atomic read loaded it"
    );
    assert_eq!(
        fx.verdicts().values().sum::<u64>(),
        0,
        "no verdict, over_deadline included: {:?}",
        fx.verdicts()
    );
    assert_eq!(fx.reads_on("fenced"), 0, "no fenced read");
    match fx.cache.lookup(fx.oid) {
        CacheLookup::Hit(guard) => assert_eq!(
            guard.entry().seq,
            seq,
            "the cached entry stands at the commit"
        ),
        CacheLookup::Miss => panic!("the object must be cached; got a miss"),
        CacheLookup::Uncacheable { reason } => {
            panic!("the object must be cached; got Uncacheable({reason:?})")
        }
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// T18's dynamic half — the fenced cold start's read-permit wait
// ═════════════════════════════════════════════════════════════════════════════

/// One fenced cold start on a fresh object (no fence key, so no `RETRY_AFTER`
/// and one acquire): exactly one
/// `live_edit_catchup_snapshot_read_wait_seconds{consumer="structural_cache",class="fenced"}`
/// observation, and none under any other consumer; one `path="fenced"` read,
/// `ok`, and no `path="atomic"` read; the `fenced` route; and every permit and
/// grant given back.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn fenced_cold_start_observes_its_read_permit_wait_under_the_structural_cache_consumer() {
    const WAIT: &str = "live_edit_catchup_snapshot_read_wait_seconds_count";
    let fx = Fx::new("c1-fenced-wait").await;
    assert_eq!(
        fx.cell(
            WAIT,
            &[("consumer", "structural_cache"), ("class", "fenced")]
        ),
        0,
        "fixture: nothing observed before the call"
    );

    fx.update(NEW_SECTIONS[0])
        .await
        .expect("the call must commit");

    assert_eq!(
        fx.cell(
            WAIT,
            &[("consumer", "structural_cache"), ("class", "fenced")]
        ),
        1,
        "the fenced cold start's one P_f wait is observed under its own consumer"
    );
    let text = render(&fx.metrics);
    let others: Vec<&str> = text
        .lines()
        .filter(|l| l.starts_with(&format!("{WAIT}{{")))
        .filter(|l| !l.contains("consumer=\"structural_cache\""))
        .collect();
    assert!(others.is_empty(), "no other consumer waited: {others:?}");
    assert_eq!(
        (
            fx.reads("fenced", "ok"),
            fx.reads_on("fenced"),
            fx.reads_on("atomic")
        ),
        (1, 1, 0),
        "one fenced read, served; no plain read first"
    );
    assert_eq!(
        fx.cold_starts(),
        BTreeMap::from([("atomic", 0), ("fenced", 1), ("uncacheable", 0)])
    );
    assert_eq!(fx.commits_on(STRUCTURAL_FULL), 1);
    fx.assert_everything_released(0, "after a verified fenced cold start");
}

// ═════════════════════════════════════════════════════════════════════════════
// T12's cache half — no writer intent on the structural route
// ═════════════════════════════════════════════════════════════════════════════

/// A fenced cold start on a layer whose fence key lingers (an earlier store's
/// fenced cold start left it), its `structural_full` commit, and a cached
/// commit on the same store write nothing to `obj:snapintent`: the `MONITOR`
/// record shows only reads of that key (the update script's `TYPE`, the
/// acquire's `HGETALL`) and no registration, no intent is counted, and the
/// key does not exist afterwards.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn neither_a_fenced_cold_start_nor_a_cached_commit_writes_a_writer_intent() {
    let fx = Fx::new("c1-fenced-no-intent").await;
    fx.update(NEW_SECTIONS[0])
        .await
        .expect("the first store's call must commit");
    assert!(
        fx.exists(fx.fence_key()).await,
        "fixture: the first fenced cold start's fence key lingers"
    );
    let other = LayerStore::new(fx.client.clone(), fx.tv.key_schema().clone(), 4096)
        .with_update_layer_backend(UpdateLayerBackend::Lua)
        .with_update_layer_retry_max_attempts(MAX_ATTEMPTS)
        .with_structural_cache(StructuralCache::new(BUDGET_BYTES))
        .with_metrics(Arc::clone(&fx.metrics));
    let mut tap = MonitorTap::start(&test_valkey_addr()).await;

    fx.update_on(&other, NEW_SECTIONS[1])
        .await
        .expect("the other store's cold-start call must commit");
    fx.update_on(&other, NEW_SECTIONS[2])
        .await
        .expect("the other store's cached call must commit");
    let seen = fx.drain_to_marker(&mut tap).await;

    assert_eq!(
        (
            fx.commits_on(STRUCTURAL_FULL),
            fx.commits_on(STRUCTURAL),
            fx.commits_on(WATCH)
        ),
        (2, 1, 0),
        "fixture: two fenced cold starts and one cached commit, none on WATCH"
    );
    assert_eq!(
        fx.cold_starts()["fenced"],
        2,
        "fixture: both cold starts were fenced"
    );
    let intent = fx.intent_key();
    let on_intent: Vec<&MonitoredCommand> = seen
        .iter()
        .filter(|c| c.args.iter().skip(1).any(|a| *a == intent))
        .collect();
    assert!(
        on_intent.iter().any(|c| c.name() == "TYPE"),
        "vacuity guard: the update script's step 1d TYPE-probed the intent key: {on_intent:#?}"
    );
    // Reads are the update script's and the probe's `TYPE` and the acquire's
    // `HGETALL`. A script naming the key is a write only when the key is its
    // first (the intent scripts'); the update, probe and acquire scripts name
    // it later and only read it, which their own `lua` lines show.
    let writes: Vec<&&MonitoredCommand> = on_intent
        .iter()
        .filter(|c| match c.name().as_str() {
            "TYPE" | "HGETALL" => false,
            "EVALSHA" | "EVAL" => c.args.get(3) == Some(&intent),
            _ => true,
        })
        .collect();
    assert!(
        writes.is_empty(),
        "nothing but reads touched obj:snapintent, and no intent script ran: {writes:#?}"
    );
    assert!(
        !seen.iter().any(|c| fx.is_register_eval()(c)),
        "no writer intent was registered"
    );
    assert_eq!(
        (fx.intents("registered"), fx.intents("refused_fenced")),
        (0, 0),
        "the structural route registers no writer intent (structural-edits SR-8.2)"
    );
    assert!(!fx.exists(intent).await, "obj:snapintent was never written");
}

// ═════════════════════════════════════════════════════════════════════════════
// T15 — the WATCH fallback classified from the read side
// ═════════════════════════════════════════════════════════════════════════════

/// One structural call on `fx` whose read side answered an uncacheable verdict
/// carrying the layer's size, with no fence key: its first WATCH attempt
/// registers its writer intent BEFORE its `WATCH` (`MONITOR` order), one
/// registration, counted `registered`, cleared on commit; no fence key is
/// created (no acquire was sent).
async fn fallback_registers_its_intent_before_watch(fx: &Fx, context: &str) {
    assert!(
        !fx.exists(fx.fence_key()).await,
        "{context}: fixture: no fence key before the call"
    );
    let mut tap = MonitorTap::start(&test_valkey_addr()).await;

    fx.update(NEW_SECTIONS[0])
        .await
        .unwrap_or_else(|e| panic!("{context}: the call must commit; got {e:?}"));
    let seen = fx.drain_to_marker(&mut tap).await;

    assert_eq!(
        fx.commits_on(WATCH),
        1,
        "{context}: the call committed through the WATCH path"
    );
    let register = position(&seen, "the intent registration", fx.is_register_eval());
    let watch = position(&seen, "the WATCH of the header", fx.is_watch());
    assert!(
        register < watch,
        "{context}: the intent was registered before WATCH (registration {register}, WATCH \
         {watch})"
    );
    assert_eq!(
        seen.iter().filter(|c| fx.is_register_eval()(c)).count(),
        1,
        "{context}: one registration"
    );
    assert_eq!(
        (fx.intents("registered"), fx.intents("refused_fenced")),
        (1, 0),
        "{context}: fence_intent_total{{result=\"registered\"}} +1"
    );
    assert!(
        !fx.exists(fx.fence_key()).await,
        "{context}: no fence key: the classification came from the read side, not a probe"
    );
    assert!(
        !fx.exists(fx.intent_key()).await,
        "{context}: the intent was cleared on commit"
    );
}

/// `OverWeight` above the item ceiling: the probe's weight lower bound is
/// over the per-entry cap and its item bound over the plain item ceiling, so
/// the read side answers `OverWeight` with the layer's size and the fallback
/// is classified before its first `WATCH`.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn over_weight_fallback_above_the_item_ceiling_registers_its_writer_intent_before_watch() {
    let fx = Fx::build("c1-classify-over-weight", TINY_BUDGET_BYTES, |bound| {
        bound / 2
    })
    .await;

    fallback_registers_its_intent_before_watch(&fx, "over weight").await;

    assert_eq!(
        fx.verdicts()["over_weight"],
        1,
        "fixture: an over_weight verdict"
    );
    match fx.cache.lookup(fx.oid) {
        CacheLookup::Uncacheable {
            reason: UncacheableReason::OverWeight(_),
        } => {}
        other => panic!("fixture: the OverWeight negative entry; got {other:?}"),
    }
    assert_eq!(
        fx.reads_on("fenced") + fx.reads_on("atomic"),
        0,
        "fixture: the weight check refused before any read"
    );
}

/// `FencedReadFailed` (the waiting cap, so no acquire is sent and no fence key
/// is created): the fallback is classified from the failed read's size before
/// its first `WATCH`.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn fenced_read_failed_fallback_registers_its_writer_intent_before_watch() {
    let fx = Fx::new("c1-classify-fenced-failed").await;
    fx.set_remaining(Some(Duration::ZERO));

    fallback_registers_its_intent_before_watch(&fx, "fenced read failed").await;

    assert_eq!(
        fx.verdicts()["fenced_read_failed"],
        1,
        "fixture: a fenced_read_failed verdict"
    );
}

/// `NoChsComplete` above the item ceiling: the probe answers an unmarked
/// object before any cold start, carrying its own item bound when that is over
/// the plain item ceiling, so the fallback is classified before its first
/// `WATCH`, with no read.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn no_chs_complete_fallback_above_the_item_ceiling_registers_its_writer_intent_before_watch()
{
    let fx = Fx::new("c1-classify-unmarked-above").await;
    unmark(&fx.tv, fx.oid).await;
    let unmarked_bound = item_bound(&fx.tv, fx.oid).await;
    assert!(
        unmarked_bound > fx.bound / 2,
        "fixture: the unmarked bound {unmarked_bound} is over the item ceiling {}",
        fx.bound / 2
    );

    fallback_registers_its_intent_before_watch(&fx, "no chs_complete").await;

    assert_eq!(
        fx.verdicts()["no_chs_complete"],
        1,
        "fixture: a no_chs_complete verdict"
    );
    assert_eq!(
        fx.reads_on("fenced") + fx.reads_on("atomic"),
        0,
        "fixture: the probe answered before any read"
    );
}

/// `NoChsComplete` on a small layer (the default ceilings, an unmarked
/// object): the probe's bound is within the item ceiling, so the read side
/// knows no size and the fallback is not classified; with no fence key its
/// attempt registers no intent at all.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn no_chs_complete_fallback_on_a_small_layer_registers_no_writer_intent() {
    let fx = Fx::build("c1-classify-unmarked", BUDGET_BYTES, |_| {
        live_edit::config::SNAPSHOT_ATOMIC_READ_MAX_ITEMS_DEFAULT
    })
    .await;
    assert!(
        fx.bound <= live_edit::config::SNAPSHOT_ATOMIC_READ_MAX_ITEMS_DEFAULT,
        "fixture: the layer is within the plain item ceiling"
    );
    unmark(&fx.tv, fx.oid).await;
    let mut tap = MonitorTap::start(&test_valkey_addr()).await;

    fx.update(NEW_SECTIONS[0])
        .await
        .expect("the call must commit");
    let seen = fx.drain_to_marker(&mut tap).await;

    assert_eq!(
        (fx.commits_on(WATCH), fx.verdicts()["no_chs_complete"]),
        (1, 1),
        "fixture: the unmarked object's call committed through the WATCH path"
    );
    position(&seen, "the WATCH of the header", fx.is_watch());
    assert!(
        !seen.iter().any(|c| fx.is_register_eval()(c)),
        "no registration: a NoChsComplete verdict within the item ceiling carries no size"
    );
    assert_eq!(
        (fx.intents("registered"), fx.intents("refused_fenced")),
        (0, 0)
    );
}
