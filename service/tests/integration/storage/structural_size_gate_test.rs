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

//! The structural route's size gate (`storage::structural_size_gate`) seen
//! through `LayerStore::update_layer` on the `lua` backend: a structural or
//! reset commit whose work (planned mutations plus children-list entries ÷
//! 20) is over the work limit commits through the WATCH path instead of one
//! long structural `EVALSHA`.
//!
//! Every store under test but the WARN test's (and the read-side test, which
//! calls the read side with no store) gates at a small limit through
//! the test-support override
//! (`LayerStore::with_structural_commit_work_limit_for_test`, both gates), so
//! each edit is a few thousand records: the production limit's ~100,000-record
//! WATCH commits ride Issue b.gvf's lost-reply retries and take tens of
//! seconds. The production value (100,000) is pinned by the module's unit
//! tests. Each edit is a delta that creates one section and writes fields and
//! time samples on it, its row counts chosen at run time with the planner
//! itself (`domain::delta::apply_delta`) on the layer's state, and each test
//! self-checks its sizing before the call.
//!
//! | Test | What it pins |
//! |---|---|
//! | [`structural_edit_planned_at_the_work_limit_commits_on_the_structural_route`] | a plan of exactly the limit on a cached entry commits on `path="structural"`: no fallback tick, one `live_edit_update_layer_structural_commit_seconds` sample, no slow-commit WARN |
//! | [`structural_edit_planned_one_unit_over_the_work_limit_commits_through_watch`] | a plan of the limit + 1: `reason="over_work_limit"` +1 and one INFO line naming the work and the limit in force, committed through the WATCH path with no structural `EVALSHA` sent (no commit-seconds sample), the entry put back unchanged, the layer holding the edit |
//! | [`structural_edit_under_a_large_parent_counts_the_parents_held_list_against_the_work_limit`] | the children term: a one-section create under a parent of 2,000 children, at a limit it is over only with the parent's pre-plan list counted, falls back; its INFO line's `work` is the mutations plus (new + held entries) ÷ 20 |
//! | [`reset_of_an_uncached_layer_estimated_over_the_work_limit_goes_to_watch_with_no_cold_start_read`] | the pre-read reset gate: a reset of a cold layer whose estimate (3 × sections + fields + time samples + content rows) is the limit + 1 ticks `reason="reset_over_work_limit"`, reads nothing (no cold-start route, no `structural_cache` snapshot read on any path), commits through the WATCH path with a writer intent registered on every attempt, the first included, and leaves the object a cache miss, not uncacheable |
//! | [`structural_edit_over_the_work_limit_on_a_cached_layer_above_the_ceiling_registers_its_intent_first`] | the post-plan gate's classification from the entry's size: on a cached layer above the plain ceiling, a plan over the limit falls back with a writer intent registered on every WATCH attempt, the first included |
//! | [`reset_of_an_uncached_layer_estimated_at_the_work_limit_cold_starts_and_its_plan_is_gated`] | the boundary: an estimate of exactly the limit cold-starts as before (the fenced route), and the plan, over the limit by the records the estimate leaves out, is caught by the post-plan gate (an intent on every WATCH attempt) |
//! | [`read_side_reset_gate_without_an_override_gates_at_the_production_limit`] | the read side's own default limit, which a release build's every request uses: with no override, a cold reset estimated at exactly 100,000 cold-starts and 100,001 answers `OverWorkLimit` |
//! | [`structural_commit_slower_than_two_seconds_logs_one_warn_with_its_work_and_the_layers_size`] | the slow-commit WARN, at the production limit: a structural `EVALSHA` held 2.1 s by a reply-delaying proxy logs one WARN carrying the plan's work, the limit and the layer's rows, and its commit-seconds sample lands above the `le="2"` bucket |
//!
//! The two reset tests and the cached above-ceiling test run their store on
//! a client whose plain item ceiling is half the fixture object's item
//! bound, so a cold start of it takes the fenced route ("no fenced read" is
//! then the gate's doing, not the layer's size) and a WATCH fallback is
//! classified above the ceiling. The other tests use the fixture's
//! default-limit client.
//!
//! Elsewhere: `StructuralWork::of_plan`, `exceeds` and `reset_work_lower_bound`
//! (the one-child edit under a parent of 100,000 children, and the production
//! limit's value, among them) inline in `src/storage/structural_size_gate.rs`;
//! the fallback reasons' zero-seeded vocabulary and the commit-seconds buckets
//! in `observability_metrics_test.rs`.
//!
//! # Wait budget
//!
//! Every call is under [`CALL_BUDGET`] (15 s); no test makes more than two
//! (30 s at worst). Each call plans and commits at most a few thousand
//! records, well under a second; the WARN test's commit waits [`SLOW_REPLY`]
//! (2.1 s) in the proxy. All well under nextest's 120 s hang-killer.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use bytes::Bytes;
use fred::interfaces::{HashesInterface, KeysInterface, SetsInterface, SortedSetsInterface};
use oodf_ffi::{OwnedDelta, SetField, TimeSample as OodfTimeSample};
use serde_json::Value;

use live_edit::config::UpdateLayerBackend;
use live_edit::domain::delta::{apply_delta, StorageMutation};
use live_edit::domain::types::{
    FieldRows, ObjectState, ObjectStateOf, SampleRows, Section, ValueRows,
};
use live_edit::error::ServiceError;
use live_edit::observability::metrics::test_util::{render, sample_value};
use live_edit::observability::metrics::{Metrics, SNAPSHOT_READ_OUTCOMES, SNAPSHOT_READ_PATHS};
use live_edit::storage::client::ValkeyClient;
use live_edit::storage::layer::LayerStore;
use live_edit::storage::lua::UPDATE_LAYER_FAST_SCRIPT;
use live_edit::storage::structural_cache::{CacheEntry, CacheLookup, StructuralCache};
use live_edit::storage::structural_cache_read::{
    read_for_structural_edit, StructuralPath, StructuralRead, StructuralReadRequest,
};
use live_edit::storage::structural_size_gate::{
    StructuralWork, STRUCTURAL_COMMIT_SLOW_WARN, STRUCTURAL_COMMIT_WORK_LIMIT,
};
use live_edit::storage::test_support::TestValkey;

use crate::common::owned_delta::delta_to_ompf_bytes;
use crate::grpc::log_capture::{capture, Captured};
use crate::storage::snapshot_read_atomic_test::{
    create, create_cmd, create_session, every_shape_item_bound, every_shape_state, layer_store,
    structural, ROOT,
};
use crate::storage::snapshot_read_support::{
    client_through, client_with_limits, test_valkey_addr, ProxyAction, ProxyRule, ScriptReplyProxy,
};
use crate::storage::structural_cache_cold_start_test::normalized_fresh_state;

// ═════════════════════════════════════════════════════════════════════════════
// Budgets and constants
// ═════════════════════════════════════════════════════════════════════════════

/// Deadline on one `update_layer` call: an edit of a few thousand records
/// settles in well under a second; the WARN test's in about 2.1 s.
const CALL_BUDGET: Duration = Duration::from_secs(15);
/// How long the WARN test's proxy holds the structural commit's `EVALSHA`
/// before forwarding it: past the 2 s WARN line, well under the client's 5 s
/// command timeout.
const SLOW_REPLY: Duration = Duration::from_millis(2_100);
/// Retry budget of every store under test.
const MAX_ATTEMPTS: u32 = 8;
/// A structural cache budget no fixture object comes near.
const BUDGET_BYTES: u64 = 64 * 1024 * 1024;
/// A byte ceiling no fixture object comes near: the item ceiling alone puts
/// the object above the plain limits.
const NO_BYTE_LIMIT: u64 = 1 << 40;

/// The work limit the sized tests' stores gate at (the test-support
/// override): small enough that an edit either side of it commits in well
/// under a second, large enough that the fixed records of every delta here
/// are far below it.
const LIMIT: u64 = 2_000;
/// The production limit, restated: what a store without the override gates
/// at, and what the WARN test's line names.
const N: u64 = 100_000;
/// Children-list entries per work unit, restated.
const ENTRIES_PER_UNIT: u64 = 20;

const WRITER: &str = "size-gate-writer";
const CLIENT_TS: u64 = 12_000;
/// The section the warm-up call creates under the root.
const WARM_SECTION: u64 = 90;
/// The section every sized delta creates.
const NEW_SECTION: u64 = 91;
/// The time samples every content delta writes; its fields make up the rest.
const SAMPLES: u64 = 500;

/// The size gate's fallback INFO line and slow-commit WARN, restated: the
/// strings operators search for.
const SIZE_FALLBACK_MSG: &str = "structural edit is over the structural commit's work limit; \
     committing this call through the WATCH path instead of one long script call";
const SLOW_COMMIT_MSG: &str = "structural commit script ran long: it holds the object's shard \
     for its whole run; a hold above 2.5 s is the trigger to lower the structural commit's work \
     limit";

/// `live_edit_update_layer_*` `path` labels and the vocabularies, restated:
/// they are the operator-facing contract.
const WATCH: &str = "watch";
const STRUCTURAL: &str = "structural";
const STRUCTURAL_FULL: &str = "structural_full";
const RESET: &str = "reset";
const OVER_WORK_LIMIT: &str = "over_work_limit";
const RESET_OVER_WORK_LIMIT: &str = "reset_over_work_limit";
const FALLBACK_REASONS: [&str; 3] = ["repeat_divergence", OVER_WORK_LIMIT, RESET_OVER_WORK_LIMIT];
const COLD_START_ROUTES: [&str; 3] = ["atomic", "fenced", "uncacheable"];
const UNCACHEABLE_REASONS: [&str; 5] = [
    "over_weight",
    "no_chs_complete",
    "reload_limit",
    "fenced_read_failed",
    "over_deadline",
];
const STRUCTURAL_COMMIT_SECONDS: &str = "live_edit_update_layer_structural_commit_seconds";

// ═════════════════════════════════════════════════════════════════════════════
// Fixture
// ═════════════════════════════════════════════════════════════════════════════

/// One object, created through a default-limit store, and a `lua` store
/// under test over it with its own cache, its work limit, and one registry
/// hooked onto the store and its client.
struct Fx {
    tv: TestValkey,
    metrics: Arc<Metrics>,
    cache: StructuralCache,
    ls: LayerStore,
    /// The work limit the store gates at: the override, or the production
    /// limit when none was set.
    limit: u64,
    session_id: String,
    layer: &'static str,
    oid: u64,
}

impl Fx {
    /// `state` as `layer`; the store under test runs on the fixture's
    /// default-limit client and gates at `limit`.
    async fn new(layer: &'static str, state: &ObjectState, limit: u64) -> Self {
        let tv = TestValkey::setup().await;
        let client = tv.client().clone();
        Self::on(tv, client, layer, state, Some(limit)).await
    }

    /// [`every_shape_state`] as `layer`; the store under test runs on a client
    /// whose plain item ceiling is half the object's item bound, so a cold
    /// start of it takes the fenced route, and gates at [`LIMIT`].
    async fn above_the_plain_ceiling(layer: &'static str) -> Self {
        let tv = TestValkey::setup().await;
        let client = client_with_limits(every_shape_item_bound(true) / 2, NO_BYTE_LIMIT).await;
        Self::on(tv, client, layer, &every_shape_state(), Some(LIMIT)).await
    }

    /// `state` as `layer`; the store under test runs on `client` and gates at
    /// `limit`, or at the production limit for `None`.
    async fn on(
        tv: TestValkey,
        client: ValkeyClient,
        layer: &'static str,
        state: &ObjectState,
        limit: Option<u64>,
    ) -> Self {
        let session_id = create_session(&tv, layer).await;
        let oid = create(&layer_store(&tv), &session_id, layer, state).await;
        let metrics = Arc::new(crate::common::metrics_scrape_only(layer));
        client
            .install_observability_hooks(Arc::clone(&metrics))
            .await;
        let cache = StructuralCache::new(BUDGET_BYTES);
        let ls = LayerStore::new(client, tv.key_schema().clone(), 4096)
            .with_update_layer_backend(UpdateLayerBackend::Lua)
            .with_update_layer_retry_max_attempts(MAX_ATTEMPTS)
            .with_structural_cache(cache.clone())
            .with_metrics(Arc::clone(&metrics));
        let ls = match limit {
            Some(limit) => ls.with_structural_commit_work_limit_for_test(limit),
            None => ls,
        };
        Self {
            tv,
            metrics,
            cache,
            ls,
            limit: limit.unwrap_or(STRUCTURAL_COMMIT_WORK_LIMIT),
            session_id,
            layer,
            oid,
        }
    }

    /// One `update_layer` call by [`WRITER`], relaying the delta's OMPF,
    /// under [`CALL_BUDGET`], with its log captured. The store call is boxed:
    /// its WATCH-path future is too large to hold inline on a debug build's
    /// test-thread stack.
    async fn update(&self, delta: OwnedDelta) -> (Result<(u64, u64), ServiceError>, Captured) {
        let relay = Some(Bytes::from(delta_to_ompf_bytes(&delta, &[])));
        let call = tokio::time::timeout(
            CALL_BUDGET,
            Box::pin(self.ls.update_layer(
                &self.session_id,
                self.layer,
                self.oid,
                Some(delta),
                relay,
                WRITER,
                CLIENT_TS,
            )),
        );
        let (result, captured) = capture(call).await;
        (
            result.expect("update_layer did not settle within the call budget"),
            captured,
        )
    }

    /// The warm-up call: creates [`WARM_SECTION`] under the root, so the
    /// object's entry is cached for the call under test.
    async fn warm(&self) {
        let (result, _) = self
            .update(structural(vec![create_cmd(ROOT, WARM_SECTION, "Warm")]))
            .await;
        result.expect("the warm-up call must commit");
        assert_eq!(
            self.commits_on(STRUCTURAL_FULL),
            1,
            "fixture: the warm-up call cold-started the entry"
        );
    }

    /// The object's cached entry, which must exist (the lookup's guard puts
    /// it back on drop).
    fn cached_entry(&self, context: &str) -> CacheEntry {
        match self.cache.lookup(self.oid) {
            CacheLookup::Hit(guard) => guard.entry().clone(),
            CacheLookup::Miss => panic!("{context}: the object must be cached; got a miss"),
            CacheLookup::Uncacheable { reason } => {
                panic!("{context}: the object must be cached; got uncacheable {reason:?}")
            }
        }
    }

    async fn header_seq(&self) -> u64 {
        let seq: Option<String> = self
            .tv
            .client()
            .pool
            .hget(
                self.tv.key_schema().obj_header(&self.oid.to_string()),
                "seq",
            )
            .await
            .expect("HGET obj:hdr seq must succeed");
        seq.expect("the header has a seq")
            .parse()
            .expect("the header seq is decimal")
    }

    /// The probe's cardinalities as stored: `(SCARD obj:secs, ZCARD obj:flds,
    /// ZCARD obj:tss)`, read directly — an oracle independent of the probe.
    async fn stored_cardinalities(&self) -> (u64, u64, u64) {
        let oid = self.oid.to_string();
        let keys = self.tv.key_schema();
        let pool = &self.tv.client().pool;
        let sections: u64 = pool
            .scard(keys.obj_sections_index(&oid))
            .await
            .expect("SCARD obj:secs must succeed");
        let fields: u64 = pool
            .zcard(keys.obj_fields_index(&oid))
            .await
            .expect("ZCARD obj:flds must succeed");
        let samples: u64 = pool
            .zcard(keys.obj_time_samples_index(&oid))
            .await
            .expect("ZCARD obj:tss must succeed");
        (sections, fields, samples)
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

    /// Every `live_edit_update_layer_structural_fallbacks_total` cell.
    fn fallbacks(&self) -> BTreeMap<&'static str, u64> {
        FALLBACK_REASONS
            .iter()
            .map(|&reason| {
                (
                    reason,
                    self.cell(
                        "live_edit_update_layer_structural_fallbacks_total",
                        &[("reason", reason)],
                    ),
                )
            })
            .collect()
    }

    /// Every `live_edit_structural_cold_start_reads_total` cell.
    fn cold_starts(&self) -> BTreeMap<&'static str, u64> {
        COLD_START_ROUTES
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

    /// Every `live_edit_structural_cache_uncacheable_total` cell.
    fn verdicts(&self) -> BTreeMap<&'static str, u64> {
        UNCACHEABLE_REASONS
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

    /// `live_edit_snapshot_read_total{consumer="structural_cache"}` on each
    /// path, summed over every outcome (the exported vocabularies, so a new
    /// path or outcome is counted too).
    fn cache_reads(&self) -> BTreeMap<&'static str, u64> {
        SNAPSHOT_READ_PATHS
            .iter()
            .map(|&path| {
                let total = SNAPSHOT_READ_OUTCOMES
                    .iter()
                    .map(|&outcome| {
                        self.cell(
                            "live_edit_snapshot_read_total",
                            &[
                                ("consumer", "structural_cache"),
                                ("path", path),
                                ("outcome", outcome),
                            ],
                        )
                    })
                    .sum();
                (path, total)
            })
            .collect()
    }

    /// WATCH-path attempts of the calls that committed there
    /// (`live_edit_update_layer_attempts_histogram_sum{path="watch"}`).
    fn watch_attempts(&self) -> u64 {
        self.cell(
            "live_edit_update_layer_attempts_histogram_sum",
            &[("path", WATCH)],
        )
    }

    fn intents_registered(&self) -> u64 {
        self.cell(
            "live_edit_update_layer_fence_intent_total",
            &[("result", "registered")],
        )
    }

    /// Whether the object's snapshot-fence key exists.
    async fn fence_key_exists(&self) -> bool {
        let n: i64 = self
            .tv
            .client()
            .pool
            .exists(
                self.tv
                    .key_schema()
                    .obj_snapshot_fence(&self.oid.to_string()),
            )
            .await
            .expect("EXISTS obj:snapfence must succeed");
        n == 1
    }

    /// Samples of `live_edit_update_layer_structural_commit_seconds`.
    fn structural_commit_samples(&self) -> u64 {
        self.cell(&format!("{STRUCTURAL_COMMIT_SECONDS}_count"), &[])
    }

    // ── Assertions ───────────────────────────────────────────────────────────

    /// The one size-gate INFO line in `captured`, which must name `reason`
    /// and this store's limit.
    fn one_size_fallback_line<'a>(&self, captured: &'a Captured, reason: &str) -> &'a Value {
        let line = captured.only(SIZE_FALLBACK_MSG);
        assert_eq!(line["level"], "INFO", "the size fallback logs at INFO");
        assert_eq!(line["fields"]["reason"], reason, "line: {line}");
        assert_eq!(
            line["fields"]["work_limit"],
            Value::from(self.limit),
            "the line names the limit in force, the store's override; line: {line}"
        );
        line
    }

    /// The WATCH fallback was classified above the ceiling before its first
    /// attempt: every WATCH attempt registered a writer intent, the first
    /// included. An unclassified call's first attempt registers none before
    /// its `WATCH` (only its read's tallies classify the attempts after it),
    /// so it would show one intent fewer than attempts.
    fn assert_every_watch_attempt_registered_an_intent(&self, classified_from: &str) {
        let attempts = self.watch_attempts();
        assert!(attempts >= 1, "fixture: the call made a WATCH attempt");
        assert_eq!(
            self.intents_registered(),
            attempts,
            "the fallback, classified above the ceiling from {classified_from}, registered a \
             writer intent on every WATCH attempt, the first included"
        );
    }

    /// The stored layer holds [`NEW_SECTION`] under `parent`, listed last,
    /// with `fields` fields and `samples` time samples on it.
    async fn assert_layer_holds_the_edit(
        &self,
        parent: u64,
        fields: u64,
        samples: u64,
        context: &str,
    ) {
        let stored = normalized_fresh_state(&self.tv, self.oid).await;
        assert_eq!(
            stored.sections.get(&NEW_SECTION).map(|s| s.parent_id),
            Some(parent),
            "{context}: the created section is stored under its parent"
        );
        assert_eq!(
            stored.children.get(&parent).and_then(|c| c.last()),
            Some(&NEW_SECTION),
            "{context}: the parent's list ends with the created section"
        );
        let (mut stored_fields, mut stored_samples) = (0u64, 0u64);
        stored.fields.for_each(|s, _, _| {
            stored_fields += u64::from(s == NEW_SECTION);
        });
        stored.time_samples.for_each(|s, _, _| {
            stored_samples += u64::from(s == NEW_SECTION);
        });
        assert_eq!(
            (stored_fields, stored_samples),
            (fields, samples),
            "{context}: every field and time sample of the edit is stored"
        );
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Sized deltas
// ═════════════════════════════════════════════════════════════════════════════

/// A delta creating [`NEW_SECTION`] under `parent` and writing `fields`
/// fields and [`SAMPLES`] time samples on it; a reset (`is_diff`,
/// `base_version` 0) when `reset`.
fn content_delta(parent: u64, fields: u64, reset: bool) -> OwnedDelta {
    OwnedDelta {
        is_diff: reset,
        base_version: 0,
        struct_commands: vec![create_cmd(parent, NEW_SECTION, "Big")],
        set_fields: (0..fields)
            .map(|i| SetField {
                section_id: NEW_SECTION,
                key_name: format!("k{i}"),
                value_or_ext_hash: vec![0x5A],
                ext_value_size: 0,
                ext_value_index: 0,
                set_order: i + 1,
            })
            .collect(),
        time_samples: (0..SAMPLES)
            .map(|i| OodfTimeSample {
                section_id: NEW_SECTION,
                time: i as f64,
                value_or_ext_hash: vec![0xA5],
                ext_value_size: 0,
                ext_value_index: 0,
            })
            .collect(),
        source_format: if reset {
            "usda".to_string()
        } else {
            String::new()
        },
    }
}

/// The planner's mutations for `delta` on `pre`, as the structural route
/// plans them (on a clone).
fn plan<R: ValueRows>(pre: &ObjectStateOf<R>, delta: &OwnedDelta) -> Vec<StorageMutation> {
    let mut planned = pre.clone();
    apply_delta(&mut planned, Some(delta.clone())).expect("the sized delta plans on the entry")
}

/// The fallback cells after a call that ticked only `reason` (or nothing).
fn fallback_cells(reason: Option<&'static str>) -> BTreeMap<&'static str, u64> {
    FALLBACK_REASONS
        .iter()
        .map(|&r| (r, u64::from(Some(r) == reason)))
        .collect()
}

// ═════════════════════════════════════════════════════════════════════════════
// One each side of the limit, on a cached entry
// ═════════════════════════════════════════════════════════════════════════════

/// A cached object on a store gating at [`LIMIT`], and a delta whose plan on
/// its entry measures exactly `target` work units ([`edit_of`]).
async fn cached_with_edit_of(layer: &'static str, target: u64) -> (Fx, OwnedDelta, u64) {
    let fx = Fx::new(layer, &every_shape_state(), LIMIT).await;
    fx.warm().await;
    let (delta, fields) = edit_of(&fx, target);
    (fx, delta, fields)
}

/// A delta whose plan on `fx`'s cached entry measures exactly `target` work
/// units, self-checked with `StructuralWork::of_plan`. Each field adds
/// exactly one mutation and no children-list entry, so the work grows by one
/// per field from its value at none. Returns the delta and its field count.
fn edit_of(fx: &Fx, target: u64) -> (OwnedDelta, u64) {
    let pre = fx.cached_entry("before the sized call").state;
    let work = |delta: &OwnedDelta| StructuralWork::of_plan(&plan(&pre, delta), &pre).units();
    let fields = target
        .checked_sub(work(&content_delta(ROOT, 0, false)))
        .expect("fixture: the target is above the delta's fixed records and samples");
    let delta = content_delta(ROOT, fields, false);
    assert_eq!(
        work(&delta),
        target,
        "fixture: the delta's plan measures exactly the target"
    );
    (delta, fields)
}

/// A plan of exactly the limit is not over it: the call commits on the
/// cached structural route, with one structural `EVALSHA` sampled and no
/// fallback.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn structural_edit_planned_at_the_work_limit_commits_on_the_structural_route() {
    let (fx, delta, fields) = cached_with_edit_of("size-gate-at-limit", LIMIT).await;
    let samples_before = fx.structural_commit_samples();

    let (result, captured) = fx.update(delta).await;

    let (seq, _) = result.expect("a plan at the limit must commit");
    assert_eq!(seq, fx.header_seq().await, "the call's commit");
    assert_eq!(
        (fx.commits_on(STRUCTURAL), fx.commits_on(WATCH)),
        (1, 0),
        "the call committed on the cached structural route"
    );
    assert_eq!(fx.fallbacks(), fallback_cells(None), "no fallback ticked");
    assert_eq!(
        fx.structural_commit_samples(),
        samples_before + 1,
        "one structural EVALSHA was sent and timed"
    );
    assert!(
        captured.with_message(SIZE_FALLBACK_MSG).is_empty(),
        "no size-gate line; capture:\n{}",
        captured.text
    );
    assert!(
        captured.with_message(SLOW_COMMIT_MSG).is_empty(),
        "a commit of a few thousand records is no slow commit: no WARN; capture:\n{}",
        captured.text
    );
    fx.assert_layer_holds_the_edit(ROOT, fields, SAMPLES, "at the limit")
        .await;
}

/// A plan of the limit + 1 is over it: the call commits through the WATCH
/// path, sends no structural `EVALSHA`, ticks `over_work_limit` once, logs
/// one INFO line with the plan's work, the limit in force and the entry's
/// rows, and puts the entry back unchanged.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn structural_edit_planned_one_unit_over_the_work_limit_commits_through_watch() {
    let (fx, delta, fields) = cached_with_edit_of("size-gate-over-limit", LIMIT + 1).await;
    let samples_before = fx.structural_commit_samples();
    let seq_before = fx.header_seq().await;
    let pre = fx.cached_entry("before the sized call").state;

    let (result, captured) = fx.update(delta).await;

    let (seq, _) = result.expect("a plan over the limit must still commit, on the WATCH path");
    assert_eq!(seq, fx.header_seq().await, "the call's commit");
    assert_eq!(
        (fx.commits_on(WATCH), fx.commits_on(STRUCTURAL)),
        (1, 0),
        "the call committed through the WATCH path"
    );
    assert_eq!(
        fx.fallbacks(),
        fallback_cells(Some(OVER_WORK_LIMIT)),
        "one over_work_limit fallback"
    );
    assert_eq!(
        fx.structural_commit_samples(),
        samples_before,
        "no structural EVALSHA was sent for the call"
    );
    let line = fx.one_size_fallback_line(&captured, OVER_WORK_LIMIT);
    for (field, want) in [
        ("work", Value::from(LIMIT + 1)),
        ("layer_sections", Value::from(pre.sections.len() as u64)),
        ("layer_fields", Value::from(pre.fields.len() as u64)),
        (
            "layer_time_samples",
            Value::from(pre.time_samples.len() as u64),
        ),
        ("object_id", Value::from(fx.oid)),
        ("session_id", Value::from(fx.session_id.as_str())),
        ("layer_name", Value::from(fx.layer)),
    ] {
        assert_eq!(
            line["fields"][field], want,
            "the line's {field}; line: {line}"
        );
    }
    assert_eq!(
        fx.cached_entry("after the fallback").seq,
        seq_before,
        "the entry went back unchanged, at the cut it was planned on"
    );
    fx.assert_layer_holds_the_edit(ROOT, fields, SAMPLES, "over the limit")
        .await;
}

/// The post-plan gate classifies its WATCH fallback from the entry's size
/// (P12): on a cached layer above the plain ceiling, a plan over the limit
/// read nothing of the layer this call (a hit), and no fence key lingers for
/// the WATCH attempt's own probe to find, so only that classification makes
/// the fallback's first WATCH attempt register its writer intent before its
/// `WATCH`. Every WATCH attempt registers one, the first included.
///
/// The entry is cached by a warm-up through another store sharing the
/// cache, on the default-limit client: its cold start takes the atomic
/// route and leaves no fence key, where the store under test's own cold
/// start would take the fenced route and leave one lingering.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn structural_edit_over_the_work_limit_on_a_cached_layer_above_the_ceiling_registers_its_intent_first(
) {
    let fx = Fx::above_the_plain_ceiling("size-gate-over-limit-above-ceiling").await;
    let warm = LayerStore::new(fx.tv.client().clone(), fx.tv.key_schema().clone(), 4096)
        .with_update_layer_backend(UpdateLayerBackend::Lua)
        .with_structural_cache(fx.cache.clone());
    let warm_delta = structural(vec![create_cmd(ROOT, WARM_SECTION, "Warm")]);
    let relay = Some(Bytes::from(delta_to_ompf_bytes(&warm_delta, &[])));
    tokio::time::timeout(
        CALL_BUDGET,
        Box::pin(warm.update_layer(
            &fx.session_id,
            fx.layer,
            fx.oid,
            Some(warm_delta),
            relay,
            WRITER,
            CLIENT_TS,
        )),
    )
    .await
    .expect("the warm-up call did not settle within the call budget")
    .expect("the warm-up call must commit");
    assert!(
        !fx.fence_key_exists().await,
        "fixture: the warm-up's atomic cold start left no fence key"
    );
    let (delta, fields) = edit_of(&fx, LIMIT + 1);

    let (result, captured) = fx.update(delta).await;

    result.expect("a plan over the limit must still commit, on the WATCH path");
    assert_eq!(
        fx.fallbacks(),
        fallback_cells(Some(OVER_WORK_LIMIT)),
        "one over_work_limit fallback"
    );
    assert_eq!(
        (fx.commits_on(WATCH), fx.commits_on(STRUCTURAL)),
        (1, 0),
        "the call committed through the WATCH path"
    );
    assert_eq!(
        fx.cold_starts(),
        BTreeMap::from([("atomic", 0), ("fenced", 0), ("uncacheable", 0)]),
        "the call under test planned on a hit: no cold start of its own"
    );
    fx.assert_every_watch_attempt_registered_an_intent("the cached entry");
    fx.one_size_fallback_line(&captured, OVER_WORK_LIMIT);
    fx.assert_layer_holds_the_edit(ROOT, fields, SAMPLES, "over the limit, above the ceiling")
        .await;
}

// ═════════════════════════════════════════════════════════════════════════════
// The children term
// ═════════════════════════════════════════════════════════════════════════════

/// The large parent's id and its child count.
const BIG_PARENT: u64 = 1;
const BIG_PARENT_CHILDREN: u64 = 2_000;

/// A layer whose root lists [`BIG_PARENT`], which lists
/// [`BIG_PARENT_CHILDREN`] leaf sections.
fn large_parent_state() -> ObjectState {
    let section = |section_id: u64, parent_id: u64, order: u32| Section {
        section_id,
        parent_id,
        name: format!("s{section_id}"),
        section_type: 1,
        section_order: order,
        first_version: 1,
        last_version: 1,
    };
    let mut state = ObjectState {
        sequence_num: 1,
        first_version: 1,
        source_format: "usda".to_string(),
        next_set_order: 10,
        ..Default::default()
    };
    state
        .sections
        .insert(BIG_PARENT, section(BIG_PARENT, ROOT, 0));
    state.children.insert(ROOT, vec![BIG_PARENT]);
    let children: Vec<u64> = (1_000..1_000 + BIG_PARENT_CHILDREN).collect();
    for (order, &id) in children.iter().enumerate() {
        state.sections.insert(
            id,
            section(id, BIG_PARENT, u32::try_from(order).expect("order fits")),
        );
    }
    state.children.insert(BIG_PARENT, children);
    state
}

/// The children term counts a rewritten list's pre-plan entries: a create of
/// one section under a parent of 2,000 children rewrites that list (2,001
/// new entries, 100 units) and hashes the stored one (2,000 more: 200 units
/// for the two together). The store gates at the plan's mutations + 150, so
/// the one-section edit is over the limit only when the held list is
/// counted; counting the new entries alone it would be 50 under and commit
/// on the structural route.
///
/// The mutations and the list record are read off the planner's own output,
/// so the expected work here does not come from `StructuralWork::of_plan`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn structural_edit_under_a_large_parent_counts_the_parents_held_list_against_the_work_limit()
{
    let state = large_parent_state();
    let delta = structural(vec![create_cmd(BIG_PARENT, NEW_SECTION, "One")]);
    // The limit is fixed before the store exists, from the plan on the state
    // as created; the self-checks below re-plan on the cached entry.
    let mutations = plan(&state, &delta).len() as u64;
    let new_entries = BIG_PARENT_CHILDREN + 1;
    let limit = mutations + (new_entries + BIG_PARENT_CHILDREN / 2) / ENTRIES_PER_UNIT;
    let fx = Fx::new("size-gate-children-term", &state, limit).await;
    fx.warm().await;
    let pre = fx.cached_entry("before the call").state;
    let held = pre.children.get(&BIG_PARENT).map_or(0, Vec::len) as u64;
    assert_eq!(
        held, BIG_PARENT_CHILDREN,
        "fixture: the parent's list as cached"
    );
    let planned = plan(&pre, &delta);
    assert_eq!(
        planned.len() as u64,
        mutations,
        "fixture: the plan on the entry"
    );
    let lists: Vec<(u64, Option<usize>)> = planned
        .iter()
        .filter_map(|m| match m {
            StorageMutation::SetChildren {
                section_id,
                children,
            } => Some((*section_id, Some(children.len()))),
            StorageMutation::DeleteChildrenList { section_id } => Some((*section_id, None)),
            _ => None,
        })
        .collect();
    assert_eq!(
        lists,
        vec![(BIG_PARENT, Some(new_entries as usize))],
        "fixture: the plan's only list record rewrites the large parent's list"
    );
    let work = mutations + (new_entries + held) / ENTRIES_PER_UNIT;
    assert!(
        mutations + new_entries / ENTRIES_PER_UNIT <= limit && work > limit,
        "fixture: only the held list's entries put the plan over the limit"
    );

    let (result, captured) = fx.update(delta).await;

    result.expect("the call must commit, on the WATCH path");
    assert_eq!(
        fx.fallbacks(),
        fallback_cells(Some(OVER_WORK_LIMIT)),
        "the held list's entries put the plan over the limit"
    );
    assert_eq!(
        (fx.commits_on(WATCH), fx.commits_on(STRUCTURAL)),
        (1, 0),
        "the call committed through the WATCH path"
    );
    let line = fx.one_size_fallback_line(&captured, OVER_WORK_LIMIT);
    assert_eq!(
        line["fields"]["work"],
        Value::from(work),
        "the work counts the mutations plus the new and the held entries over 20; line: {line}"
    );
    fx.assert_layer_holds_the_edit(BIG_PARENT, 0, 0, "under the large parent")
        .await;
}

// ═════════════════════════════════════════════════════════════════════════════
// The pre-read reset gate
// ═════════════════════════════════════════════════════════════════════════════

/// A cold object above the plain ceiling, on a store gating at [`LIMIT`],
/// and a reset of it whose pre-read estimate — 3 × sections + fields + time
/// samples, as stored, plus the reset's content rows — is exactly
/// `estimate`.
async fn cold_with_reset_estimated_at(layer: &'static str, estimate: u64) -> (Fx, OwnedDelta, u64) {
    let fx = Fx::above_the_plain_ceiling(layer).await;
    let (sections, fields_stored, samples_stored) = fx.stored_cardinalities().await;
    assert_eq!(
        (sections, fields_stored, samples_stored),
        (4, 2, 2),
        "fixture: every_shape_state's live sections, fields and time samples"
    );
    let content_rows = estimate - (3 * sections + fields_stored + samples_stored);
    let fields = content_rows - SAMPLES;
    let delta = content_delta(ROOT, fields, true);
    assert_eq!(
        (delta.set_fields.len() + delta.time_samples.len()) as u64,
        content_rows,
        "fixture: the reset's content rows"
    );
    assert!(
        matches!(fx.cache.lookup(fx.oid), CacheLookup::Miss),
        "fixture: the object starts uncached"
    );
    (fx, delta, fields)
}

/// A reset of an uncached layer estimated at the limit + 1 goes straight to
/// the WATCH path before the cold start: `reset_over_work_limit` +1, no
/// cold-start route and no `structural_cache` snapshot read on any path (so
/// no fenced read, although the layer is above the plain ceiling), a commit
/// through the WATCH path with a writer intent registered on every attempt,
/// the first included (classified from the probe's size), and the object
/// left a plain miss — no entry, no uncacheable verdict.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn reset_of_an_uncached_layer_estimated_over_the_work_limit_goes_to_watch_with_no_cold_start_read(
) {
    let (fx, delta, fields) = cold_with_reset_estimated_at("size-gate-reset-over", LIMIT + 1).await;

    let (result, captured) = fx.update(delta).await;

    let (seq, _) = result.expect("the reset must commit, on the WATCH path");
    assert_eq!(seq, fx.header_seq().await, "the call's commit");
    assert_eq!(
        fx.fallbacks(),
        fallback_cells(Some(RESET_OVER_WORK_LIMIT)),
        "one reset_over_work_limit fallback, and no post-plan one"
    );
    assert_eq!(
        fx.cold_starts(),
        BTreeMap::from([("atomic", 0), ("fenced", 0), ("uncacheable", 0)]),
        "no cold start ran"
    );
    assert_eq!(
        fx.cache_reads().values().sum::<u64>(),
        0,
        "the structural cache read nothing of the layer, on any path: {:?}",
        fx.cache_reads()
    );
    assert_eq!(
        (
            fx.commits_on(WATCH),
            fx.commits_on(RESET),
            fx.commits_on(STRUCTURAL_FULL)
        ),
        (1, 0, 0),
        "the reset committed through the WATCH path"
    );
    assert_eq!(fx.structural_commit_samples(), 0, "no structural EVALSHA");
    fx.assert_every_watch_attempt_registered_an_intent("the probe");
    assert!(
        matches!(fx.cache.lookup(fx.oid), CacheLookup::Miss),
        "nothing was recorded: the object is a miss, still cacheable"
    );
    assert_eq!(
        fx.verdicts().values().sum::<u64>(),
        0,
        "no uncacheable verdict: {:?}",
        fx.verdicts()
    );
    let line = fx.one_size_fallback_line(&captured, RESET_OVER_WORK_LIMIT);
    for (field, want) in [
        ("work", Value::from(LIMIT + 1)),
        ("layer_sections", Value::from(4)),
        ("layer_fields", Value::from(2)),
        ("layer_time_samples", Value::from(2)),
    ] {
        assert_eq!(
            line["fields"][field], want,
            "the line's {field}; line: {line}"
        );
    }
    let stored = normalized_fresh_state(&fx.tv, fx.oid).await;
    assert_eq!(
        stored.sections.keys().copied().collect::<Vec<_>>(),
        vec![NEW_SECTION],
        "the reset cleared the layer and created its one section"
    );
    fx.assert_layer_holds_the_edit(ROOT, fields, SAMPLES, "the reset over the estimate")
        .await;
}

/// The pre-read gate's boundary: a reset estimated at exactly the limit is
/// not over it, so the cold start runs as before — the fenced route, the
/// layer being above the plain ceiling — and the plan, which also creates
/// the reset's section and rewrites the root's list and the header beyond
/// the estimate, is over the limit and caught by the post-plan gate:
/// `over_work_limit` +1, never `reset_over_work_limit`. The call's fenced
/// cold start already classified it above the ceiling, so every WATCH
/// attempt, the first included, registers a writer intent (the post-plan
/// gate's own classification from the entry's size is pinned on a cached
/// layer, where nothing else classifies it); the cold-started entry goes
/// back to the cache.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn reset_of_an_uncached_layer_estimated_at_the_work_limit_cold_starts_and_its_plan_is_gated()
{
    let (fx, delta, fields) = cold_with_reset_estimated_at("size-gate-reset-at", LIMIT).await;
    let seq_before = fx.header_seq().await;

    let (result, captured) = fx.update(delta).await;

    result.expect("the reset must commit");
    assert_eq!(
        fx.cold_starts(),
        BTreeMap::from([("atomic", 0), ("fenced", 1), ("uncacheable", 0)]),
        "an estimate of exactly the limit cold-starts, on the fenced route"
    );
    assert!(
        fx.cache_reads()["fenced"] >= 1,
        "the cold start read the layer fenced: {:?}",
        fx.cache_reads()
    );
    assert_eq!(
        fx.fallbacks(),
        fallback_cells(Some(OVER_WORK_LIMIT)),
        "the post-plan gate, not the pre-read one, sent the reset to the WATCH path"
    );
    assert_eq!(
        (fx.commits_on(WATCH), fx.commits_on(RESET)),
        (1, 0),
        "the reset committed through the WATCH path"
    );
    assert_eq!(fx.structural_commit_samples(), 0, "no structural EVALSHA");
    fx.assert_every_watch_attempt_registered_an_intent("its fenced cold start");
    let line = fx.one_size_fallback_line(&captured, OVER_WORK_LIMIT);
    let work = line["fields"]["work"]
        .as_u64()
        .unwrap_or_else(|| panic!("the line's work is a number; line: {line}"));
    assert!(
        work > LIMIT,
        "the plan's work is over the limit; line: {line}"
    );
    assert_eq!(
        fx.cached_entry("after the post-plan fallback").seq,
        seq_before,
        "the cold-started entry went back, at the cut it was read at"
    );
    fx.assert_layer_holds_the_edit(ROOT, fields, SAMPLES, "the reset at the estimate")
        .await;
}

/// The pre-read gate's production limit, read straight off the read side: a
/// request with no override (as a release build's every request is) gates a
/// cold reset at 100,000 units. The reset's content rows are only a count on
/// the request, so the estimate reaches the limit with nothing large built:
/// `every_shape_state`'s 4 sections, 2 fields and 2 time samples give 16, and
/// the rest are rows. An estimate of exactly 100,000 cold-starts (`Ready` on
/// `structural_full`); 100,001 answers `OverWorkLimit` with that estimate.
/// The over case runs first, on the cold object, and records nothing, so the
/// at-limit read that follows is still a cold start.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn read_side_reset_gate_without_an_override_gates_at_the_production_limit() {
    const TOKEN: &str = "size-gate-read-token";
    let tv = TestValkey::setup().await;
    let layer = "size-gate-read-default-limit";
    let session_id = create_session(&tv, layer).await;
    let oid = create(&layer_store(&tv), &session_id, layer, &every_shape_state()).await;
    let cache = StructuralCache::new(BUDGET_BYTES);
    let base = 3 * 4 + 2 + 2;
    let read = |rows: u64| {
        let cache = cache.clone();
        let tv = &tv;
        async move {
            let request =
                StructuralReadRequest::new(oid, WRITER, TOKEN).with_reset_content_rows(rows);
            tokio::time::timeout(
                CALL_BUDGET,
                Box::pin(read_for_structural_edit(
                    tv.client(),
                    tv.key_schema(),
                    &cache,
                    &request,
                )),
            )
            .await
            .expect("the read did not settle within the call budget")
            .expect("the read must answer")
        }
    };

    match read(N - base + 1).await {
        StructuralRead::OverWorkLimit { estimate, .. } => assert_eq!(
            estimate,
            N + 1,
            "the estimate: 3 × 4 sections + 2 fields + 2 time samples + the rows"
        ),
        other => panic!("one unit over the production limit must be gated; got {other:?}"),
    }
    assert!(
        matches!(cache.lookup(oid), CacheLookup::Miss),
        "the gated read recorded nothing"
    );

    match read(N - base).await {
        StructuralRead::Ready(ready) => assert_eq!(
            ready.path,
            StructuralPath::StructuralFull,
            "an estimate of exactly the production limit cold-starts"
        ),
        other => panic!("an estimate at the production limit must cold-start; got {other:?}"),
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// The slow-commit WARN
// ═════════════════════════════════════════════════════════════════════════════

/// The `le` bucket of [`STRUCTURAL_COMMIT_SECONDS`] whose bound is `bound`,
/// compared as a number so the exporter's float formatting does not matter.
fn commit_seconds_bucket(text: &str, bound: f64) -> Option<u64> {
    let prefix = format!("{STRUCTURAL_COMMIT_SECONDS}_bucket{{");
    text.lines()
        .filter(|l| l.starts_with(&prefix))
        .find_map(|l| {
            let (series, value) = l.rsplit_once(' ')?;
            let start = series.find("le=\"")? + 4;
            let len = series[start..].find('"')?;
            let le: f64 = series[start..start + len].parse().ok()?;
            (le == bound).then(|| value.parse::<f64>().ok().map(|v| v as u64))?
        })
}

/// A structural commit whose `EVALSHA` round trip passes 2 s logs one WARN
/// naming the plan's work (`work`, `mutations`, `children_entries`), the
/// limit in force (the production limit: this store sets no override), the
/// layer's rows and the WARN line, and its elapsed time; and its
/// commit-seconds sample lands above the 2 s bucket. The round trip is held
/// by a proxy that delays every `EVALSHA` of the update script by
/// [`SLOW_REPLY`] before forwarding it — the only script of that SHA the call
/// sends is its structural commit (and, on a server without the script
/// cached, its reload's retry, delayed the same).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn structural_commit_slower_than_two_seconds_logs_one_warn_with_its_work_and_the_layers_size()
{
    assert_eq!(
        STRUCTURAL_COMMIT_SLOW_WARN,
        Duration::from_secs(2),
        "the WARN line the delay must pass"
    );
    let sha = UPDATE_LAYER_FAST_SCRIPT.sha1().to_owned();
    let delayed = Arc::new(AtomicUsize::new(0));
    let seen = Arc::clone(&delayed);
    let rule: ProxyRule = Arc::new(move |args: &[Vec<u8>]| {
        let is_evalsha = args.first()?.eq_ignore_ascii_case(b"EVALSHA");
        (is_evalsha && args.get(1).is_some_and(|a| a.as_slice() == sha.as_bytes())).then(|| {
            seen.fetch_add(1, Ordering::SeqCst);
            ProxyAction::Delay(SLOW_REPLY)
        })
    });
    let proxy = ScriptReplyProxy::start(test_valkey_addr(), rule).await;
    let tv = TestValkey::setup().await;
    let client = client_through(&proxy).await;
    let fx = Fx::on(
        tv,
        client,
        "size-gate-slow-commit",
        &every_shape_state(),
        None,
    )
    .await;
    let pre = normalized_fresh_state(&fx.tv, fx.oid).await;
    let delta = structural(vec![create_cmd(ROOT, NEW_SECTION, "Slow")]);
    let work = StructuralWork::of_plan(&plan(&pre, &delta), &pre);
    assert_eq!(
        work.children_entries,
        2 * pre.children[&ROOT].len() as u64 + 1,
        "fixture: the root's list rewritten (one entry more) and its held entries hashed"
    );

    let (result, captured) = fx.update(delta).await;

    result.expect("the slow commit must still land");
    assert!(
        delayed.load(Ordering::SeqCst) >= 1,
        "fixture: the proxy held the structural commit's EVALSHA"
    );
    assert_eq!(
        fx.commits_on(STRUCTURAL_FULL),
        1,
        "the call committed on the structural route"
    );
    let warns = captured.with_message(SLOW_COMMIT_MSG);
    assert_eq!(
        warns.len(),
        1,
        "one WARN for the one slow commit; capture:\n{}",
        captured.text
    );
    let line = warns[0];
    assert_eq!(line["level"], "WARN", "line: {line}");
    for (field, want) in [
        ("work", Value::from(work.units())),
        ("mutations", Value::from(work.mutations)),
        ("children_entries", Value::from(work.children_entries)),
        ("work_limit", Value::from(N)),
        ("slow_warn_ms", Value::from(2_000)),
        ("layer_sections", Value::from(pre.sections.len() as u64)),
        ("layer_fields", Value::from(pre.fields.len() as u64)),
        (
            "layer_time_samples",
            Value::from(pre.time_samples.len() as u64),
        ),
        ("object_id", Value::from(fx.oid)),
        ("session_id", Value::from(fx.session_id.as_str())),
        ("layer_name", Value::from(fx.layer)),
    ] {
        assert_eq!(
            line["fields"][field], want,
            "the WARN's {field}; line: {line}"
        );
    }
    let elapsed_ms = line["fields"]["elapsed_ms"]
        .as_u64()
        .unwrap_or_else(|| panic!("the WARN's elapsed_ms is a number; line: {line}"));
    assert!(
        u128::from(elapsed_ms) >= SLOW_REPLY.as_millis(),
        "elapsed_ms covers the held round trip; line: {line}"
    );

    let text = render(&fx.metrics);
    assert_eq!(
        fx.structural_commit_samples(),
        1,
        "one structural EVALSHA sampled"
    );
    assert_eq!(
        commit_seconds_bucket(&text, 2.0),
        Some(0),
        "the sample is above the 2 s bucket boundary"
    );
}
