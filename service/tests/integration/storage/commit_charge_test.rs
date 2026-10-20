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

//! b.26e Stage 2: a structural `UpdateLayer`'s commit charge through the
//! storage seam (`storage::commit_charge`, SDD §5.4), driven through
//! `LayerStore::update_layer_with_summary` with a charge of the test's own
//! ([`ScriptedCharge`]) or the upload implementation over a real gate
//! (`grpc::upload_admission::UploadCommitCharge`).
//!
//! | Test | What it pins |
//! |---|---|
//! | [`watch_structural_attempt_first_holds_its_fixed_cost_b26e`] | a WATCH structural attempt's first charge point (WA0) prices `ObjectStateNeed::watch_fixed()` and holds it, on `txn` and `lua-watch-structural`, with no handler hint in front of it |
//! | [`uncacheable_fallback_charges_the_watch_attempt_on_the_calls_charge_b26e`] | the cached route's fallback to WATCH (an object too large for the store's structural cache) keeps the call's charge: WA0 is held and WA1 follows on it; with room for WA0 alone, one recharge on `path="watch"` and one commit |
//! | `refused_growth_*_frees_its_permits_and_checkout_while_it_waits_b26e` (two) | a refused in-commit growth (WATCH on `txn`, the cached route on `lua`) waits with both per-object permits free and the txn checkout returned (a commit through the one-connection txn pool lands during the wait), then commits once under its token, the attempts histogram at 1 and `recharges_total{path}` at 1 |
//! | [`refused_growth_dropped_at_its_deadline_applies_nothing_b26e`] | a call dropped (its deadline) while it waits applies nothing and leaves the permits free |
//! | [`refused_growth_at_shutdown_answers_server_shutting_down_b26e`] | the upload charge's wait ended by shutdown answers `UNAVAILABLE "server shutting down"` with the `SERVER_SHUTTING_DOWN` detail and retry hint, nothing applied |
//! | [`slot_holder_and_budget_waiter_on_one_object_both_commit_b26e`] | the overflow-slot deadlock: X's growth beyond the budget's room waits on the gate while Z, holding the slot, edits the same object; both commit before the deadline |
//! | [`refused_cold_start_charge_stores_the_entry_and_the_rerun_is_a_hit_b26e`] | a refused cold-start charge (CR0, at seam A) whose hold is within the plain read's hold allowance (on every weighed shape at the default) stores the entry and holds no cold-start read permit at the wait; the re-run is a hit that commits once |
//! | [`refused_cold_start_charge_over_its_hold_allowance_stores_nothing_and_rereads_ug2`] | a refused CR0 on a plain read whose state and index build are over its hold allowance (a test override) stores nothing and holds no read permit at the wait; the re-run cold-starts again, its CR0 granted with no second wait, and commits once: two cold lookups, no hit, two atomic reads, one `atomic` route tick, one recharge |
//! | [`refused_repair_read_on_a_hit_waits_with_the_entry_at_its_cut_and_recharges_on_the_hits_path_fsz`] | a cache hit's first charge point is its repair's first read (RP0: the entry at its stored weight, state and plan indexes, and the first read's reply at its `STRUCTURAL_REPAIR_INITIAL_ENTRIES` and `STRUCTURAL_REPAIR_INITIAL_BYTES` bounds); refused, it waits once for that whole target with the permits free and the entry put back at its cut, two commits behind, then commits once at its first attempt, the recharge counted on `structural` (or `reset` for a reset), never `structural_full` |
//! | [`refused_record_growth_mid_chunk_waits_at_the_records_cut_and_resumes_from_it_fsz`] | A4 (i): a hit's per-record repair charge point (RP2) refused mid-chunk — its record a child added under a parent of 4,000 children, its two predecessors replayed — puts the entry back at the second record's cut (`seq` and `commit_chain`, its settle passing) and waits once; the resumed repair replays only the last three records and commits once, the entry then equal to a cold start |
//! | [`refused_plan_on_a_hit_waits_for_the_resumed_repairs_first_read_and_is_never_refused_there_fsz`] | F1: a refused plan growth on a hit whose repair doubled the sections table (so the plan's rewrite allowance is the first read's reply term less that growth), its CR1 form (the entry as put back plus the plan's exact need) below the resumed RP0, waits first for that RP0 on the entry as put back, which the resumed attempt then holds without a refusal under contention; rolled back at the wait; at most one more wait, for the resumed plan's own growth; one commit |
//! | [`refused_plan_on_a_repairing_hit_rolls_back_and_waits_for_its_cr1_form_under_production_pricing_fsz`] | F2: the plan's rollback and the repair's transient together under production pricing: a repairing hit's plan refused once rolls back to the repaired cut and waits once, for its CR1 form (above the RP0 floor); the resumed plan never asks again; one commit |
//! | [`watch_read_stops_before_its_next_window_and_unwatches_before_it_waits_b26e`] | a read whose growth is refused stops (at WA1, before any round-1 page; and after WA1, at WA2, on a long-named base), counts `too_large` on `structural_update`/`unfenced`, and has `UNWATCH`ed before the call waits |
//! | [`cached_plan_grows_its_charge_and_plans_once_b26e`], [`watch_plan_grows_its_charge_and_reads_once_b26e`] | a plan whose rewrite exceeds what is held grows the charge through its budget and plans once: one plan in place; one `structural_update` read |
//! | [`refused_plan_growth_waits_for_the_exact_need_and_replans_within_it_b26e`] | a refused in-plan growth waits for the plan's exact rows and list entries; the held entry is unchanged; the resumed plan never asks its budget; one commit, one recharge; on the cached route (T5a) the wait is CR1 of the entry as put back — heavier than before the plan, a kept parent's grown name vector — plus that exact need, that CR1 priced by the test from the entry ([`independent_cr1`]) and the resumed attempt's logged CR1 pinned to it |
//! | [`refused_plan_growth_waits_on_a_put_back_entry_lighter_than_before_the_plan_ug2`] | T5b: an `Overflow` whose rollback re-creates a dropped name vector at its exact length below its capacity waits for CR1 of the lighter entry as put back plus the exact need, below the pre-plan form, the CR1 priced by the test from the entry and the logged one pinned to it; one recharge |
//! | [`refused_cr2_waits_for_its_own_target_ug2`] | T5h: a refused CR2 (long-named creates whose names neither CR1 nor the rewrite charges) waits for CR2's own target, the pre-plan cut held meanwhile; one recharge |
//! | [`stale_commit_on_a_full_table_holds_its_rollback_rebuild_at_cr2_ug2`], [`refused_plan_on_a_full_table_waits_for_its_rollback_rebuild_ug2`] | R-T5: a delete on a full sections table — its CR2 holds the table's rollback rebuild when the commit answers `STALE` (on an object whose erase leaves the table no room, a hard precondition met within a sized bound of fresh objects; its rollback rebuilds the table), and a refused plan's wait carries it |
//! | `*_resumes_within_two_recharges_ug2` (four) | T5c (a dropped grown name vector), T5d (move away, move back, grow), T5e (delete then create the same id, grow) and T5g (delete and insert on a table with one entry of room, the resumed plan projecting a rebuild the refused one did not; precondition met within a sized bound): under a competitor that takes whatever the call gives back, one commit within two recharges, each wait covering the CR1 the test prices on the entry as put back ([`independent_cr1`]) — which the resumed attempt's logged CR1 must then equal — no growth taken past a refusal, and the named layout hard-checked as having occurred in the refused plan |
//! | [`resumed_attempt_keeps_what_its_recharge_waited_for_b26e`] | after a recharge, with a competitor taking whatever the call gives back, the resumed attempt's grow-only WA0 / RP0 and CR1 give nothing back: one recharge |
//! | [`watch_read_charges_every_window_before_it_is_issued_b26e`] | on a layer with a full window of 4 KiB inline value rows and a 5,000-entry list, every window and page after the count windows follows a priced need whose window bytes cover what the window then receives |
//! | [`watch_read_ranges_only_the_lists_with_entries_b26e`] | an empty list issues no `LRANGE` (marked: not read; unmarked: its `LLEN` only); the long list one `LRANGE 0 n−1`; on both of the read's transports, whichever issued the command (the client or a window script) |
//! | [`watch_read_of_a_list_grown_after_its_length_reads_that_length_and_retries_b26e`] | a list a competitor's commit grows between the read's `LLEN` and its held `LRANGE` is ranged at the `LLEN` (n ids); the `seq` proof rejects the read (one `watch_read_moved`), and the retry ranges the grown list and commits |
//! | [`watch_read_refused_before_a_script_window_sends_no_window_script_b_gvf`] | a refused growth before the read's first script window (b.gvf) stops the read with no `EVALSHA` of the window script sent |
//! | `watch_read_falling_back_charges_its_plain_window_before_sending_it_b_gvf` (`test-fault-injection`) | a script window that falls back on a cluster-state refusal (b.gvf) grows the charge for its wire commands as plain commands in flight before it sends any of them; refused, it sends none |
//!
//! The handler-level half of the commit charge is in
//! `grpc::upload_admission_test`: its object-state hint
//! (`update_layer_cached_edit_waits_for_its_hint_at_the_processing_step_b26e`),
//! the hint held in budget while the hit's first repair read (RP0) is
//! refused
//! (`update_layer_cached_edit_with_its_hint_in_budget_recharges_once_at_its_first_repair_read_fsz`)
//! and the checkpoint-recovery re-issue's reuse of the call's charge
//! (`update_layer_recovery_reissue_recharges_through_the_calls_commit_charge_b26e`).
//! The heap of a refused plan — the envelope outside it and the documented
//! residual inside it — is measured by the upload memory envelope's
//! `slow_path_refused_plan_on_a_full_budget_stays_within_its_charge_b26e`.
//!
//! Every wait is bounded by [`STEP_BUDGET`]; the healthy path meets each in
//! milliseconds.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::Duration;

use fred::interfaces::{HashesInterface, StreamsInterface};
use futures::future::BoxFuture;
use futures::FutureExt;
use live_edit::config::UpdateLayerBackend;
use live_edit::domain::alloc_model::{allocation_bytes, table_growth_bytes, vec_growth_bytes};
use live_edit::domain::delta::{MutationAllowance, RewriteCount};
use live_edit::domain::plan_index::PlanIndexes;
use live_edit::domain::types::{
    CompactFields, CompactSamples, CompactState, NameInterner, ObjectState, Section,
};
use live_edit::error::ServiceError;
use live_edit::grpc::graceful_shutdown::GracefulShutdown;
use live_edit::grpc::upload_admission::{
    object_state_charge_for_test, object_state_rewrite_allowance_for_test, UploadAdmissionConfig,
    UploadAdmissionGate, UploadCommitCharge, UploadReservation, UploadRpc,
};
use live_edit::observability::metrics::test_util::{render, sample_value};
use live_edit::observability::metrics::Metrics;
use live_edit::storage::commit_charge::{
    CommitCharge, NoCommitCharge, ObjectStateNeed, RepairTransient,
};
use live_edit::storage::external_values::MockExternalValueStore;
use live_edit::storage::layer::{LayerStore, HEADER_FIELD_COMMIT_CHAIN};
use live_edit::storage::session::SessionStore;
use live_edit::storage::structural_cache::{estimate_state_bytes, PlannedInserts, StructuralCache};
use live_edit::storage::structural_cache_read::{
    read_for_structural_edit, StructuralRead, StructuralReadRequest,
};
use live_edit::storage::structural_cache_sync::{
    STRUCTURAL_REPAIR_INITIAL_BYTES, STRUCTURAL_REPAIR_INITIAL_ENTRIES,
    STRUCTURAL_REPAIR_MAX_ENTRIES,
};
use live_edit::storage::test_support::relay::relay_for;
use live_edit::storage::test_support::TestValkey;

use crate::storage::structural_cache_read_test::{repair_points, stream_entries, RepairPoint};
use oodf_ffi::{
    CommandCreateSection, CommandDeleteSection, CommandMoveSection, OwnedDelta, StructCommand,
};
use tokio::sync::Notify;
use tokio_util::sync::CancellationToken;

/// Bound on any one wait below. Each healthy step returns in milliseconds;
/// this only turns a hang into a failure with a message.
const STEP_BUDGET: Duration = Duration::from_secs(15);

/// Child sections of the root in every fixture's base layer: enough that a
/// WATCH read has more than its header to count, few enough that every call
/// is a few round trips.
const BASE_CHILDREN: u64 = 20;

/// The first section id of the base layer's children.
const FIRST_CHILD: u64 = 100;

/// The `writers` permits per object the fixtures admit
/// (`with_update_layer_per_object_concurrency`).
const WRITERS: usize = 4;

/// The client id every call commits under.
const CLIENT: &str = "commit-charge-writer";

// ─────────────────────────────────────────────────────────────────────────────
// Fixture
// ─────────────────────────────────────────────────────────────────────────────

struct Fixture {
    tv: TestValkey,
    ls: Arc<LayerStore>,
    metrics: Arc<Metrics>,
    session_id: String,
    layer: String,
    oid: u64,
}

/// The base layer: [`BASE_CHILDREN`] named children of the root.
fn base_state() -> ObjectState {
    named_base_state(|id| format!("child{id}"))
}

/// A layer with no sections: the header every fixture state starts from.
fn empty_state() -> ObjectState {
    ObjectState {
        sequence_num: 1,
        first_version: 1,
        source_format: "usd".to_owned(),
        ..Default::default()
    }
}

/// Section `id`, named `s{id}`, under `parent` at `order`.
fn section(id: u64, parent: u64, order: u32) -> Section {
    Section {
        section_id: id,
        parent_id: parent,
        name: format!("s{id}"),
        section_type: 1,
        section_order: order,
        first_version: 1,
        last_version: 1,
    }
}

/// [`base_state`] with each child named `name(id)`.
fn named_base_state(name: impl Fn(u64) -> String) -> ObjectState {
    let mut state = empty_state();
    let ids: Vec<u64> = (FIRST_CHILD..FIRST_CHILD + BASE_CHILDREN).collect();
    for (order, id) in ids.iter().enumerate() {
        state.sections.insert(
            *id,
            Section {
                name: name(*id),
                ..section(*id, 0, order as u32)
            },
        );
    }
    state.children.insert(0, ids);
    state
}

/// A session and one base layer on a store of `backend`, with
/// `pool_size` connections per pool, metrics hooked on the store and its
/// client, and `configure` applied last.
async fn fixture_with(
    layer: &str,
    backend: UpdateLayerBackend,
    pool_size: u32,
    configure: impl FnOnce(LayerStore) -> LayerStore,
) -> Fixture {
    let tv = TestValkey::setup_with_pool_size(pool_size).await;
    let metrics = Arc::new(crate::common::metrics_scrape_only(layer));
    tv.client()
        .install_observability_hooks(Arc::clone(&metrics))
        .await;
    let ls = Arc::new(configure(
        LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096)
            .with_update_layer_backend(backend)
            .with_update_layer_per_object_concurrency(WRITERS as u32)
            .with_metrics(Arc::clone(&metrics)),
    ));
    let session_id = SessionStore::new(tv.client().clone(), tv.key_schema().clone())
        .create("commit-charge", "/stage", "test-user")
        .await
        .expect("session creation must succeed");
    let oid = create_base(&ls, &session_id, layer).await;
    Fixture {
        tv,
        ls,
        metrics,
        session_id,
        layer: layer.to_owned(),
        oid,
    }
}

async fn fixture(layer: &str, backend: UpdateLayerBackend) -> Fixture {
    fixture_with(layer, backend, 4, |ls| ls).await
}

async fn create_base(ls: &LayerStore, session_id: &str, layer: &str) -> u64 {
    create_state(ls, session_id, layer, &base_state()).await
}

async fn create_state(ls: &LayerStore, session_id: &str, layer: &str, state: &ObjectState) -> u64 {
    ls.create_layer(
        session_id,
        layer,
        "test-user",
        state,
        Arc::new(MockExternalValueStore::new()),
    )
    .await
    .expect("create_layer must succeed")
}

/// [`base_state`] with names of [`LONG_NAME_BYTES`]: longer than the WATCH
/// read's sizing floor (`NAME_BYTES_SIZING_FLOOR`, 64 B), so the window that
/// brings them receives more than WA1 projected and the read grows past it.
fn long_named_state() -> ObjectState {
    named_base_state(|id| format!("{id:0>width$}", width = LONG_NAME_BYTES))
}

/// The long-named base's name length.
const LONG_NAME_BYTES: usize = 1_000;

/// One `CreateSection` of `section_id` under the root: a structural delta —
/// the cached structural route under `lua`, the WATCH path under the two
/// rollback values.
fn structural_delta(section_id: u64) -> OwnedDelta {
    OwnedDelta {
        struct_commands: vec![StructCommand::CreateSection(CommandCreateSection {
            parent_id: 0,
            section_id,
            section_name: format!("new{section_id}"),
            section_type: 1,
        })],
        ..Default::default()
    }
}

impl Fixture {
    /// `update_layer_with_summary` of `delta` on object `oid` under `charge`.
    async fn update_on(
        &self,
        oid: u64,
        delta: OwnedDelta,
        charge: &mut dyn CommitCharge,
    ) -> Result<u64, ServiceError> {
        let delta = Some(delta);
        let relay = relay_for(&delta);
        let (result, _summary) = self
            .ls
            .update_layer_with_summary(
                &self.session_id,
                &self.layer,
                oid,
                Arc::new(delta),
                relay,
                CLIENT,
                1,
                None,
                charge,
            )
            .await;
        result.map(|commit| commit.seq)
    }

    /// [`Self::update_on`] the fixture's object.
    async fn update(
        &self,
        delta: OwnedDelta,
        charge: &mut dyn CommitCharge,
    ) -> Result<u64, ServiceError> {
        self.update_on(self.oid, delta, charge).await
    }

    /// The object's committed `seq` (`obj:hdr`).
    async fn seq(&self) -> u64 {
        self.seq_of(self.oid).await
    }

    /// Object `oid`'s committed `seq` (`obj:hdr`).
    async fn seq_of(&self, oid: u64) -> u64 {
        let raw: String = self
            .tv
            .client()
            .pool
            .hget(self.tv.key_schema().obj_header(&oid.to_string()), "seq")
            .await
            .expect("HGET obj:hdr seq must succeed");
        raw.parse().expect("seq is numeric")
    }

    /// The object's stream length.
    async fn stream_len(&self) -> u64 {
        self.tv
            .client()
            .pool
            .xlen(self.tv.key_schema().layer_stream(self.oid))
            .await
            .expect("XLEN of the object's stream must succeed")
    }

    /// `wtok[CLIENT]`: `{token:032x}:{seq}:{ts}`.
    async fn write_token(&self) -> Option<String> {
        self.tv
            .client()
            .pool
            .hget(
                self.tv.key_schema().obj_write_token(&self.oid.to_string()),
                CLIENT,
            )
            .await
            .expect("HGET obj:wtok must succeed")
    }

    /// A counter or histogram sample of the store's metrics, 0 when absent.
    fn sample(&self, name: &str, labels: &[(&str, &str)]) -> f64 {
        sample_value(&render(&self.metrics), name, labels).unwrap_or(0.0)
    }

    fn recharges(&self, path: &str) -> u64 {
        self.sample("live_edit_update_layer_recharges_total", &[("path", path)]) as u64
    }

    /// `live_edit_update_layer_attempts_histogram{path}`: (count, sum).
    fn attempts(&self, path: &str) -> (u64, u64) {
        (
            self.sample(
                "live_edit_update_layer_attempts_histogram_count",
                &[("path", path)],
            ) as u64,
            self.sample(
                "live_edit_update_layer_attempts_histogram_sum",
                &[("path", path)],
            ) as u64,
        )
    }
}

/// Whether `oid`'s per-object permits are all free on `ls`: no admission
/// entry, or `structural` at 1 and `writers` at [`WRITERS`].
fn permits_free(ls: &LayerStore, oid: u64) -> bool {
    ls.admission_permits_available_for_test(oid)
        .is_none_or(|available| available == (1, WRITERS))
}

/// A cache entry for the fixture's object on a `lua` store: one structural
/// commit outside any upload, so the next structural call is a cache hit.
async fn warm(fx: &Fixture, section_id: u64) {
    warm_on(fx, fx.oid, section_id).await;
}

/// [`warm`] for object `oid` of the fixture's layer store.
async fn warm_on(fx: &Fixture, oid: u64, section_id: u64) {
    fx.update_on(oid, structural_delta(section_id), &mut NoCommitCharge)
        .await
        .expect("the warming commit must land");
    assert!(
        fx.ls.structural_cache().weight_hint(oid).is_some(),
        "fixture: the warming commit left a cache entry"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// The cached entry at a wait (b.ug2 revision 3)
// ─────────────────────────────────────────────────────────────────────────────

/// The kept parent of the plan-in-place fixtures: the base's first child,
/// which holds one child of its own, so its name-index vector is one entry at
/// capacity 1 before any plan.
const KEPT_PARENT: u64 = FIRST_CHILD;

/// The kept parent's one child.
const KEPT_CHILD: u64 = 9_000;

/// A base of `children` children of the root, named `name(id)`, the first
/// of which ([`KEPT_PARENT`]) lists [`KEPT_CHILD`].
fn kept_parent_state(children: u64, name: impl Fn(u64) -> String) -> ObjectState {
    let mut state = empty_state();
    let ids: Vec<u64> = (FIRST_CHILD..FIRST_CHILD + children).collect();
    for (order, id) in ids.iter().enumerate() {
        state.sections.insert(
            *id,
            Section {
                name: name(*id),
                ..section(*id, 0, order as u32)
            },
        );
    }
    state.children.insert(0, ids);
    state
        .sections
        .insert(KEPT_CHILD, section(KEPT_CHILD, KEPT_PARENT, 0));
    state.children.insert(KEPT_PARENT, vec![KEPT_CHILD]);
    state
}

/// `n` `CreateSection`s under `parent` from id `first`, named
/// `{prefix}{id}`.
fn creates_under(parent: u64, first: u64, n: u64, prefix: &str) -> OwnedDelta {
    OwnedDelta {
        struct_commands: (first..first + n)
            .map(|id| {
                StructCommand::CreateSection(CommandCreateSection {
                    parent_id: parent,
                    section_id: id,
                    section_name: format!("{prefix}{id}"),
                    section_type: 1,
                })
            })
            .collect(),
        ..Default::default()
    }
}

/// Each table's room on a cached entry — the entries it takes before its next
/// insert call would grow it — or the insert calls and removals a plan makes
/// on it: the state's four tables, the field-name interner's name → id
/// table, and the plan indexes' two maps (`domain::delta_write::RebuildTable`).
#[derive(Debug, Clone, Copy, Default)]
struct TableCalls {
    sections: usize,
    children: usize,
    field_rows: usize,
    sample_rows: usize,
    name_ids: usize,
    name_index: usize,
    ext_counts: usize,
}

/// One table's `(len, capacity)` on a cached entry, read from the entry
/// itself (a clone of a table can be allocated at another capacity).
type Layout = (usize, usize);

/// Each table's `(len, capacity)` on a cached entry, as the CR1 growth term
/// reads them: the state's four tables, the field-name interner's name → id
/// table and its id → name vector, and the plan indexes' two maps.
#[derive(Debug, Clone, Copy)]
struct TableLayouts {
    sections: Layout,
    children: Layout,
    field_rows: Layout,
    /// The interner's live names and its name → id table's capacity.
    name_ids: Layout,
    /// The interner's live names and its id → name vector's capacity.
    name_slots: Layout,
    sample_rows: Layout,
    name_index: Layout,
    ext_counts: Layout,
}

/// The cached entry of an object, read by a checkout that puts it straight
/// back (unchanged: it was settled when it went back).
#[derive(Debug, Clone)]
struct HeldSnapshot {
    seq: u64,
    state: CompactState,
    rooms: TableCalls,
    /// `estimate_state_bytes` of the entry's state, walked over the entry
    /// itself — not its stored `state_weight_bytes`.
    state_bytes: u64,
    /// `PlanIndexes::weight_bytes_recomputed` of the entry's indexes — not
    /// its stored `index_weight_bytes` nor the indexes' running sum.
    index_bytes: u64,
    /// Whether the entry's indexes answer the depth check incrementally.
    depth_ok: bool,
    layouts: TableLayouts,
}

/// [`HeldSnapshot`] of `oid`'s entry on `ls`.
fn held_snapshot(ls: &LayerStore, oid: u64) -> HeldSnapshot {
    let held = ls
        .structural_cache()
        .checkout(oid)
        .expect("the object's entry is cached");
    let entry = held.entry();
    let indexes = entry
        .plan
        .as_ref()
        .expect("a cached entry holds its plan indexes");
    let room = |capacity: usize, len: usize| capacity.saturating_sub(len);
    let state = &entry.state;
    let names = state.fields.names();
    let field_rows = state.fields.capacity();
    let sample_rows = state.time_samples.capacity();
    let snapshot = HeldSnapshot {
        seq: entry.seq,
        state: state.clone(),
        rooms: TableCalls {
            sections: room(state.sections.capacity(), state.sections.len()),
            children: room(state.children.capacity(), state.children.len()),
            field_rows: state.fields.row_room(),
            sample_rows: state.time_samples.row_room(),
            name_ids: state.fields.name_room(),
            name_index: indexes.parent_key_room(),
            ext_counts: indexes.hash_key_room(),
        },
        state_bytes: estimate_state_bytes(state),
        index_bytes: indexes.weight_bytes_recomputed(),
        depth_ok: indexes.depth_ok(),
        layouts: TableLayouts {
            sections: (state.sections.len(), state.sections.capacity()),
            children: (state.children.len(), state.children.capacity()),
            field_rows: (field_rows - state.fields.row_room(), field_rows),
            name_ids: (names.len(), names.ids_capacity()),
            name_slots: (names.len(), names.slots_capacity()),
            sample_rows: (sample_rows - state.time_samples.row_room(), sample_rows),
            name_index: (
                indexes.parent_keys(),
                indexes.parent_keys() + indexes.parent_key_room(),
            ),
            ext_counts: (
                indexes.hash_keys(),
                indexes.hash_keys() + indexes.hash_key_room(),
            ),
        },
    };
    held.put_back_held();
    snapshot
}

/// The CR1 need of the entry `snapshot` read, for a delta of `inserts`,
/// computed by the test on its own from what it observed on the entry — the
/// independent pricing a wait is checked against, never the need the
/// implementation logged:
///
/// - **held**: the entry's state walked by `estimate_state_bytes` plus its
///   plan indexes' recomputed weight;
/// - **growth**: per table, `alloc_model::table_growth_bytes` (a hash table)
///   or `vec_growth_bytes` (the interner's id → name vector) over the
///   observed length and capacity for the delta's insert calls on it
///   (`PlannedInserts::of`) — the rule `ObjectStateNeed::Cached`'s `growth`
///   states for CR1, applied here from the observed layout;
/// - **rewrite**: the full depth walk's sections when the entry's depth flag
///   is clear, else nothing.
///
/// Not a reset (the fixtures here plan none), and the entry holds its plan
/// indexes (`held_snapshot` requires them), so no index-build term.
fn independent_cr1(snapshot: &HeldSnapshot, inserts: PlannedInserts) -> ObjectStateNeed {
    assert!(!inserts.reset, "independent_cr1 prices no reset");
    let table = |(len, capacity): Layout, calls: u64, slot: usize| {
        table_growth_bytes(len, capacity, calls, slot as u64)
    };
    let layouts = snapshot.layouts;
    let growth = table(
        layouts.sections,
        inserts.sections,
        size_of::<(u64, Section)>(),
    ) + table(
        layouts.children,
        inserts.children_list_keys,
        size_of::<(u64, Vec<u64>)>(),
    ) + table(
        layouts.field_rows,
        inserts.set_fields,
        CompactFields::ROW_SLOT_BYTES,
    ) + table(
        layouts.name_ids,
        inserts.set_fields,
        NameInterner::ID_ENTRY_BYTES,
    ) + vec_growth_bytes(
        layouts.name_slots.0,
        layouts.name_slots.1,
        inserts.set_fields,
        NameInterner::SLOT_BYTES as u64,
    ) + table(
        layouts.sample_rows,
        inserts.time_samples,
        CompactSamples::ROW_SLOT_BYTES,
    ) + table(
        layouts.name_index,
        inserts.name_key_inserts,
        PlanIndexes::NAME_SLOT_BYTES,
    ) + table(
        layouts.ext_counts,
        inserts.external_rows,
        PlanIndexes::EXT_SLOT_BYTES,
    );
    let walked_sections = if snapshot.depth_ok {
        0
    } else {
        snapshot.state.sections.len() as u64
    };
    ObjectStateNeed::cached_growing(
        snapshot.state_bytes + snapshot.index_bytes,
        growth,
        RewriteCount {
            walked_sections,
            ..RewriteCount::default()
        },
    )
}

/// The "tables ample" precondition (b.ug2 revision 3), hard: every table of
/// the entry `snapshot` read has room for `calls`, the plan's removals plus
/// its insert calls on it. So no plan of the delta on that entry projects a
/// rollback rebuild, and the resumed plan's need is the refused plan's.
fn assert_tables_ample(snapshot: &HeldSnapshot, calls: TableCalls, context: &str) {
    let rooms = snapshot.rooms;
    for (table, room, needed) in [
        ("sections", rooms.sections, calls.sections),
        ("children", rooms.children, calls.children),
        ("field rows", rooms.field_rows, calls.field_rows),
        ("sample rows", rooms.sample_rows, calls.sample_rows),
        ("name ids", rooms.name_ids, calls.name_ids),
        ("name index", rooms.name_index, calls.name_index),
        ("external-hash counts", rooms.ext_counts, calls.ext_counts),
    ] {
        assert!(
            room >= needed,
            "{context}: fixture: tables ample — the {table} table as put back has room for \
             {room}, below the plan's {needed} insert calls and removals; {rooms:?}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ScriptedCharge: a commit charge of the test's own
// ─────────────────────────────────────────────────────────────────────────────

/// What a [`ScriptedCharge`] saw, in order.
#[derive(Default, Debug)]
struct ChargeLog {
    /// Every need a charge point priced (`target`).
    needs: Vec<ObjectStateNeed>,
    /// Every non-waiting hold: its target and whether it was granted.
    tries: Vec<(u64, bool)>,
    /// Every waiting hold's target (a recharge).
    holds: Vec<u64>,
    /// How many needs and non-waiting holds had been recorded when each
    /// waiting hold started: where the resumed attempt's begin.
    at_hold: Vec<(usize, usize)>,
}

type SharedLog = Arc<Mutex<ChargeLog>>;

fn lock(log: &SharedLog) -> MutexGuard<'_, ChargeLog> {
    log.lock().unwrap_or_else(PoisonError::into_inner)
}

/// What a [`ScriptedCharge`] runs inside its waiting `hold`, before it
/// grants: the wait point, where the call holds nothing else.
type HoldProbe = Box<dyn FnMut(u64) -> BoxFuture<'static, ()> + Send>;

/// The message of the DEBUG event a [`ScriptedCharge`] logs for each WATCH
/// need it prices.
const SCRIPTED_PRICED_MSG: &str = "b.gvf test charge priced";

/// A commit charge that prices every need as production does
/// (`object_state_charge_for_test`, the production allowance on what it
/// holds), grants a non-waiting hold up to `room` and refuses one above it,
/// and in its waiting `hold` runs its probe, then grants the target and
/// sets its room to the larger of the target and `room_after_hold`. For
/// every WATCH need it prices it also logs, at DEBUG, the target (`charge`)
/// and the plain commands in flight (`plain_in_flight`) as
/// [`SCRIPTED_PRICED_MSG`], so a captured log interleaves the needs with the
/// read's `fan-out window` events, in issue order.
/// [`Self::without_repair_transient`] prices the stream repair's transient
/// at nothing instead.
struct ScriptedCharge {
    log: SharedLog,
    held: u64,
    room: u64,
    room_after_hold: u64,
    probe: Option<HoldProbe>,
    /// Refuse, once, the first growth whose priced need carries a rewrite:
    /// a plan's first ask of its budget (it precedes CR2 and WA3).
    refuse_first_plan_growth: bool,
    /// Refuse, once, the first growth whose priced need is a stream repair's
    /// per-record point (RP2) pricing a record's own growth or rewrite
    /// ([`is_record_need`]).
    refuse_first_record_growth: bool,
    /// A competitor takes every byte the charge gives back: a shrink lowers
    /// the room by what it gave back, as a gate whose freed room another
    /// upload took at once.
    competitor: bool,
    /// Whether the competitor arrives with the first wait's grant.
    contend_after_hold: bool,
    /// Price every need with its stream-repair transient at zero
    /// ([`Self::without_repair_transient`]).
    repair_unpriced: bool,
}

impl ScriptedCharge {
    /// A charge holding 0 that grants non-waiting holds up to `room`, and
    /// every one after its first wait.
    fn new(room: u64) -> (Self, SharedLog) {
        let log = SharedLog::default();
        (
            Self {
                log: Arc::clone(&log),
                held: 0,
                room,
                room_after_hold: u64::MAX,
                probe: None,
                refuse_first_plan_growth: false,
                refuse_first_record_growth: false,
                competitor: false,
                contend_after_hold: false,
                repair_unpriced: false,
            },
            log,
        )
    }

    fn with_probe(mut self, probe: HoldProbe) -> Self {
        self.probe = Some(probe);
        self
    }

    fn refusing_first_plan_growth(mut self) -> Self {
        self.refuse_first_plan_growth = true;
        self
    }

    fn refusing_first_record_growth(mut self) -> Self {
        self.refuse_first_record_growth = true;
        self
    }

    /// After its wait the room is the larger of the target and `room`, and a
    /// competitor takes what the charge gives back from then on.
    fn contended_after_hold(mut self, room: u64) -> Self {
        self.room_after_hold = room;
        self.contend_after_hold = true;
        self
    }

    /// Price the stream repair's transient (RP0, RP1, RP2) at zero; every
    /// other term as production does. On a hit the repair's first read (RP0)
    /// holds that read's reply term at `STRUCTURAL_REPAIR_INITIAL_ENTRIES`
    /// and `STRUCTURAL_REPAIR_INITIAL_BYTES` (the first read's reply term,
    /// within 1 MiB by the first-read rule), and being grow-only leaves it
    /// held into the plan as rewrite allowance, so a plan with a smaller
    /// rewrite never asks its budget: neither the plan's first growth this
    /// charge refuses ([`Self::refusing_first_plan_growth`]) nor a CR2 above
    /// what was held before it would happen. The tests of the plan's
    /// rollback and resume on small objects (b.ug2) price it so, as before
    /// the repair was charged; the repair's own charge points are tested
    /// with production pricing.
    fn without_repair_transient(mut self) -> Self {
        self.repair_unpriced = true;
        self
    }

    /// `need` as this charge prices it: unchanged, or with its repair
    /// transient at zero ([`Self::without_repair_transient`]).
    fn priced(&self, need: &ObjectStateNeed) -> ObjectStateNeed {
        match *need {
            ObjectStateNeed::Cached {
                held,
                growth,
                rewrite,
                ..
            } if self.repair_unpriced => ObjectStateNeed::Cached {
                held,
                growth,
                rewrite,
                repair: RepairTransient::default(),
            },
            other => other,
        }
    }
}

#[async_trait::async_trait]
impl CommitCharge for ScriptedCharge {
    fn held(&self) -> u64 {
        self.held
    }

    fn target(&self, need: &ObjectStateNeed) -> u64 {
        lock(&self.log).needs.push(*need);
        let charge = object_state_charge_for_test(&self.priced(need));
        if let ObjectStateNeed::Watch { snapshot, .. } = need {
            tracing::debug!(
                charge,
                plain_in_flight = snapshot.plain_in_flight,
                "{}",
                SCRIPTED_PRICED_MSG
            );
        }
        charge
    }

    fn rewrite_allowance(&self, need: &ObjectStateNeed) -> MutationAllowance {
        object_state_rewrite_allowance_for_test(self.held, &self.priced(need))
    }

    fn try_hold(&mut self, target: u64) -> bool {
        let mut log = lock(&self.log);
        // A plan's ask prices a need with a rewrite and no repair transient:
        // a stream repair's per-record point (RP2) prices a structural
        // record's listings and walk as a rewrite too, but carries the
        // round's reply, so it is never taken for the plan's first ask.
        let plan_growth = target > self.held && log.needs.last().is_some_and(is_plan_need);
        let record_growth = target > self.held && log.needs.last().is_some_and(is_record_need);
        let granted = if self.refuse_first_plan_growth && plan_growth {
            self.refuse_first_plan_growth = false;
            false
        } else if self.refuse_first_record_growth && record_growth {
            self.refuse_first_record_growth = false;
            false
        } else {
            target <= self.room
        };
        if granted {
            if self.competitor && target < self.held {
                self.room = self.room.saturating_sub(self.held - target);
            }
            self.held = target;
        }
        log.tries.push((target, granted));
        granted
    }

    async fn hold(&mut self, target: u64) -> Result<(), ServiceError> {
        {
            let mut log = lock(&self.log);
            log.holds.push(target);
            let at = (log.needs.len(), log.tries.len());
            log.at_hold.push(at);
        }
        if let Some(probe) = self.probe.as_mut() {
            probe(target).await;
        }
        self.room = self.room_after_hold.max(target);
        self.held = target;
        self.competitor |= self.contend_after_hold;
        Ok(())
    }
}

/// Whether `need` is one of the cached route's stream-repair charge points
/// (RP0, RP1, RP2): a need carrying a repair transient.
fn is_repair_need(need: &ObjectStateNeed) -> bool {
    matches!(need, ObjectStateNeed::Cached { repair, .. } if *repair != RepairTransient::default())
}

/// Whether `need` is a stream repair's per-record point (RP2) that prices
/// its record's own growth or rewrite: a repair transient beside a growth or
/// rewrite term, which RP0 and RP1 never carry.
fn is_record_need(need: &ObjectStateNeed) -> bool {
    match need {
        ObjectStateNeed::Cached {
            growth, rewrite, ..
        } => is_repair_need(need) && (*growth > 0 || *rewrite != RewriteCount::default()),
        _ => false,
    }
}

/// Whether `need` prices a plan's rewrite — a plan's ask or its exact trim
/// (CR2, WA3) — rather than a stream repair's per-record point (RP2), whose
/// structural record's listings and depth walk are a rewrite count too.
fn is_plan_need(need: &ObjectStateNeed) -> bool {
    need.rewrite() != RewriteCount::default() && !is_repair_need(need)
}

// ─────────────────────────────────────────────────────────────────────────────
// WA0, directly
// ─────────────────────────────────────────────────────────────────────────────

/// WA0 alone (b.26e Stage 2): on the WATCH path a structural attempt's first
/// charge point, before its intent registration, its checkout and its read,
/// prices the attempt's fixed cost alone — `ObjectStateNeed::watch_fixed()`
/// — and holds it. Called with no handler hint in front of it, as a storage
/// caller and the cached route's WATCH fallback are, so nothing pre-charges
/// it (through the `UpdateLayer` handler its hint already holds it and WA0
/// is a no-op). Discriminated by disabling WA0: the first need priced is
/// then the read's count window (WA1).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn watch_structural_attempt_first_holds_its_fixed_cost_b26e() {
    for (backend, layer) in [
        (UpdateLayerBackend::Txn, "wa0-txn"),
        (UpdateLayerBackend::LuaWatchStructural, "wa0-lua-watch"),
    ] {
        let fx = fixture(layer, backend).await;
        let (mut charge, log) = ScriptedCharge::new(u64::MAX);
        let before = fx.seq().await;
        let seq = fx
            .update(structural_delta(1_000), &mut charge)
            .await
            .unwrap_or_else(|e| panic!("{backend:?}: the structural call commits; got {e:?}"));
        assert_eq!(seq, before + 1, "{backend:?}: one commit");

        let log = lock(&log);
        assert_eq!(
            log.needs.first(),
            Some(&ObjectStateNeed::watch_fixed()),
            "{backend:?}: the attempt's first charge point prices its fixed cost alone; needs: \
             {:?}",
            log.needs
        );
        assert_eq!(
            log.tries.first(),
            Some(&(
                object_state_charge_for_test(&ObjectStateNeed::watch_fixed()),
                true
            )),
            "{backend:?}: and holds exactly that, first; holds: {:?}",
            log.tries
        );
        assert!(
            log.holds.is_empty(),
            "{backend:?}: nothing was refused, so nothing waited"
        );
    }
}

/// The cached route's fallback to the WATCH path keeps the call's charge.
/// On `lua` with a structural cache of one byte the object can never be
/// cached, so a structural call falls back to the WATCH path, and that
/// path's charge points run on the call's own charge: its first WATCH
/// charge point (WA0) prices the attempt's fixed cost alone and holds it,
/// and its read's count window (WA1) follows. With room for WA0 alone, WA1
/// is refused: the read stops, the call waits once
/// (`recharges_total{path="watch"}` 1) and then commits once. A fallback
/// handed no charge would price nothing on the WATCH path and never wait.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn uncacheable_fallback_charges_the_watch_attempt_on_the_calls_charge_b26e() {
    use live_edit::storage::structural_cache::StructuralCache;

    let wa0 = object_state_charge_for_test(&ObjectStateNeed::watch_fixed());
    for (arm, room) in [("unrefused", u64::MAX), ("WA1 refused", wa0)] {
        let layer = format!("fallback-{}", room == wa0);
        let fx = fixture_with(&layer, UpdateLayerBackend::Lua, 4, |ls| {
            ls.with_structural_cache(StructuralCache::new(1))
        })
        .await;
        let before = fx.seq().await;
        let (mut charge, log) = ScriptedCharge::new(room);
        let seq =
            tokio::time::timeout(STEP_BUDGET, fx.update(structural_delta(1_000), &mut charge))
                .await
                .unwrap_or_else(|_| panic!("{arm}: the call must finish"))
                .unwrap_or_else(|e| panic!("{arm}: the fallback commits; got {e:?}"));
        assert_eq!(seq, before + 1, "{arm}: one commit");
        assert!(
            fx.ls.structural_cache().weight_hint(fx.oid).is_none(),
            "{arm}: fixture: the object was never cached"
        );

        let log = lock(&log);
        let wa0_at = log
            .needs
            .iter()
            .position(|need| *need == ObjectStateNeed::watch_fixed())
            .unwrap_or_else(|| {
                panic!("{arm}: the WATCH attempt prices its fixed cost (WA0); {log:?}")
            });
        assert!(
            log.tries.contains(&(wa0, true)),
            "{arm}: and holds it; {log:?}"
        );
        let wa1_at = log
            .needs
            .iter()
            .position(|need| {
                matches!(need, ObjectStateNeed::Watch { snapshot, fixed: true, .. }
                    if snapshot.entries > 0)
            })
            .unwrap_or_else(|| panic!("{arm}: the read prices its count window (WA1); {log:?}"));
        assert!(
            wa0_at < wa1_at,
            "{arm}: WA0 before WA1 on the call's charge; {log:?}"
        );
        if room == wa0 {
            assert_eq!(log.holds.len(), 1, "{arm}: one wait; {log:?}");
            assert_eq!(fx.recharges("watch"), 1, "{arm}: one recharge on `watch`");
        } else {
            assert!(log.holds.is_empty(), "{arm}: nothing waited; {log:?}");
            assert_eq!(fx.recharges("watch"), 0, "{arm}: no recharge");
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Release and recharge
// ─────────────────────────────────────────────────────────────────────────────

/// The pinned commit token of the release-and-recharge calls.
const RECHARGE_TOKEN: u128 = 0x0b26_e000_0000_0000_0000_0000_0000_0003;

/// The probe of a recharge's wait point: the waiting call's object has both
/// per-object permits free on `ls`, and a commit on `other` through
/// `other_ls` — a second store on the same client, whose txn pool has one
/// connection, so the commit needs the checkout the waiting call held; and
/// with no metrics, so it leaves the measured call's counts alone — lands
/// within [`STEP_BUDGET`]. Records each wait's verdict on `seen`.
fn release_probe(
    ls: Arc<LayerStore>,
    other_ls: Arc<LayerStore>,
    session_id: String,
    other_layer: String,
    oid: u64,
    other: u64,
    seen: Arc<Mutex<Vec<(bool, bool)>>>,
) -> HoldProbe {
    Box::new(move |_target| {
        let (ls, other_ls, session_id, other_layer, seen) = (
            Arc::clone(&ls),
            Arc::clone(&other_ls),
            session_id.clone(),
            other_layer.clone(),
            Arc::clone(&seen),
        );
        async move {
            let free = permits_free(&ls, oid);
            let delta = Some(structural_delta(7_000));
            let relay = relay_for(&delta);
            let other_commit = tokio::time::timeout(
                STEP_BUDGET,
                other_ls.update_layer_with_summary(
                    &session_id,
                    &other_layer,
                    other,
                    Arc::new(delta),
                    relay,
                    "other-writer",
                    1,
                    None,
                    &mut NoCommitCharge,
                ),
            )
            .await;
            let landed = matches!(other_commit, Ok((Ok(_), _)));
            seen.lock()
                .unwrap_or_else(PoisonError::into_inner)
                .push((free, landed));
        }
        .boxed()
    })
}

/// Release and recharge on one route: `room` refuses the call's first growth past it;
/// at the wait both permits are free and the one txn connection is not held
/// (a commit on a second object lands), then the call commits once, under
/// its pinned token, on its first attempt, with one recharge on `path`.
async fn refused_growth_frees_and_recharges(
    layer: &str,
    backend: UpdateLayerBackend,
    path: &str,
    room: u64,
    warm_first: bool,
) {
    // An empty queue mints a random token per call; the measured call, the
    // next to start once its token is queued, pops the pinned one.
    let tokens = Arc::new(Mutex::new(VecDeque::new()));
    let fx = fixture_with(layer, backend, 1, {
        let tokens = Arc::clone(&tokens);
        |ls| ls.with_commit_tokens_for_test(tokens)
    })
    .await;
    let other_layer = format!("{layer}-other");
    let other = create_base(&fx.ls, &fx.session_id, &other_layer).await;
    if warm_first {
        warm(&fx, 5_000).await;
    }
    tokens
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .push_back(RECHARGE_TOKEN);
    let (before_seq, before_len) = (fx.seq().await, fx.stream_len().await);
    let (before_recharges, before_attempts) = (fx.recharges(path), fx.attempts(path));

    let seen = Arc::new(Mutex::new(Vec::new()));
    let (charge, log) = ScriptedCharge::new(room);
    let other_ls = Arc::new(
        LayerStore::new(fx.tv.client().clone(), fx.tv.key_schema().clone(), 4096)
            .with_update_layer_backend(UpdateLayerBackend::Txn),
    );
    let mut charge = charge.with_probe(release_probe(
        Arc::clone(&fx.ls),
        other_ls,
        fx.session_id.clone(),
        other_layer,
        fx.oid,
        other,
        Arc::clone(&seen),
    ));
    let seq = tokio::time::timeout(STEP_BUDGET, fx.update(structural_delta(1_000), &mut charge))
        .await
        .unwrap_or_else(|_| panic!("{layer}: the recharged call must finish"))
        .unwrap_or_else(|e| panic!("{layer}: the recharged call commits; got {e:?}"));

    {
        let log = lock(&log);
        assert_eq!(
            log.holds.len(),
            1,
            "{layer}: fixture: the growth past the room was refused once, and the call waited \
             once; {log:?}"
        );
        assert!(
            log.tries.iter().any(|(_, granted)| !granted),
            "{layer}: the wait followed a refused growth; {log:?}"
        );
    }
    assert_eq!(
        *seen.lock().unwrap_or_else(PoisonError::into_inner),
        vec![(true, true)],
        "{layer}: while the call waited, its object's per-object permits were free and a commit \
         through the one-connection txn pool landed (the checkout was returned)"
    );
    assert_eq!(seq, before_seq + 1, "{layer}: the call commits once");
    assert_eq!(fx.seq().await, before_seq + 1, "{layer}: one commit landed");
    assert_eq!(
        fx.stream_len().await,
        before_len + 1,
        "{layer}: one stream entry"
    );
    let token = fx
        .write_token()
        .await
        .expect("the commit records its token");
    assert!(
        token.starts_with(&format!("{RECHARGE_TOKEN:032x}:{seq}:")),
        "{layer}: the commit is the call's, under its one token: {token}"
    );
    assert_eq!(
        fx.recharges(path),
        before_recharges + 1,
        "{layer}: recharges_total{{path={path}}} ticks once"
    );
    let (count, sum) = fx.attempts(path);
    assert_eq!(
        (count - before_attempts.0, sum - before_attempts.1),
        (1, 1),
        "{layer}: the commit is counted on its first attempt: a recharge spends no retry"
    );
    assert!(
        permits_free(&fx.ls, fx.oid),
        "{layer}: the permits are back"
    );
}

/// Release and recharge, the WATCH path on `txn`: the attempt's fixed cost (WA0) fits,
/// its read's count window (WA1) does not, so the read stops, the attempt
/// `UNWATCH`es and returns its checkout, and the call waits holding nothing.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn refused_growth_on_the_watch_path_frees_its_permits_and_checkout_while_it_waits_b26e() {
    refused_growth_frees_and_recharges(
        "recharge-watch",
        UpdateLayerBackend::Txn,
        "watch",
        object_state_charge_for_test(&ObjectStateNeed::watch_fixed()),
        false,
    )
    .await;
}

/// Release and recharge, the cached route on `lua`: with no room at all, a
/// cache hit's first charge point — its repair's first read (RP0) — does
/// not fit, so the entry goes back and the call waits holding nothing.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn refused_growth_on_the_cached_route_frees_its_permits_while_it_waits_b26e() {
    refused_growth_frees_and_recharges(
        "recharge-cached",
        UpdateLayerBackend::Lua,
        "structural",
        0,
        true,
    )
    .await;
}

/// A call dropped while it waits for its recharge — the RPC
/// deadline drops the handler's future — applies nothing: the object's
/// `seq` and stream are unchanged, its per-object permits are free, and a
/// later call commits.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn refused_growth_dropped_at_its_deadline_applies_nothing_b26e() {
    for (backend, layer) in [
        (UpdateLayerBackend::Txn, "recharge-deadline-watch"),
        (UpdateLayerBackend::Lua, "recharge-deadline-cached"),
    ] {
        let fx = Arc::new(fixture(layer, backend).await);
        if backend == UpdateLayerBackend::Lua {
            warm(&fx, 5_000).await;
        }
        let (before_seq, before_len) = (fx.seq().await, fx.stream_len().await);
        let waiting = Arc::new(Notify::new());
        let probe: HoldProbe = {
            let waiting = Arc::clone(&waiting);
            Box::new(move |_target| {
                let waiting = Arc::clone(&waiting);
                async move {
                    waiting.notify_one();
                    // The deadline arrives first.
                    futures::future::pending::<()>().await;
                }
                .boxed()
            })
        };
        let room = match backend {
            UpdateLayerBackend::Lua => 0,
            _ => object_state_charge_for_test(&ObjectStateNeed::watch_fixed()),
        };
        let call = {
            let fx = Arc::clone(&fx);
            tokio::spawn(async move {
                let (charge, _log) = ScriptedCharge::new(room);
                let mut charge = charge.with_probe(probe);
                fx.update(structural_delta(1_000), &mut charge).await
            })
        };
        tokio::time::timeout(STEP_BUDGET, waiting.notified())
            .await
            .unwrap_or_else(|_| panic!("{layer}: the call must reach its recharge wait"));
        call.abort();
        let _ = call.await;

        assert_eq!(fx.seq().await, before_seq, "{layer}: nothing applied");
        assert_eq!(
            fx.stream_len().await,
            before_len,
            "{layer}: no stream entry"
        );
        assert!(
            permits_free(&fx.ls, fx.oid),
            "{layer}: the permits are free"
        );
        tokio::time::timeout(
            STEP_BUDGET,
            fx.update(structural_delta(1_001), &mut NoCommitCharge),
        )
        .await
        .unwrap_or_else(|_| panic!("{layer}: a later call must not wait on the dropped one"))
        .unwrap_or_else(|e| panic!("{layer}: a later call commits; got {e:?}"));
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// The upload charge over a real gate
// ─────────────────────────────────────────────────────────────────────────────

/// The gate's budget in the real-gate tests: room for three admissions
/// (`F + A` each) at once, above the floor of a 1 MiB size cap.
const GATE_BUDGET: u64 = 32 * 1024 * 1024;

fn small_gate() -> Arc<UploadAdmissionGate> {
    Arc::new(UploadAdmissionGate::new(
        UploadAdmissionConfig {
            budget_bytes: GATE_BUDGET,
            wait: Duration::from_secs(5),
            max_waiters: 16,
            max_upload_bytes: 1024 * 1024,
        },
        None,
    ))
}

/// An `UpdateLayer` reservation on `gate` past its receive phase, holding
/// its admission's fixed charge: the base the commit charge holds beyond.
/// Each on its own session, so no session's receiving cap binds.
async fn processing_reservation(
    gate: &Arc<UploadAdmissionGate>,
    session: &str,
    shutdown: &CancellationToken,
) -> UploadReservation {
    let mut reservation = gate
        .admit(UploadRpc::UpdateLayer, session, shutdown)
        .await
        .unwrap_or_else(|e| panic!("{session}: admitted; got {e:?}"));
    reservation.end_receive();
    reservation
}

/// Fill what is left of `gate`'s budget with `filler`'s growth.
fn fill_budget(gate: &UploadAdmissionGate, filler: &mut UploadReservation) {
    let others = gate.reserved_bytes() - filler.reserved();
    filler
        .try_grow_in_budget(GATE_BUDGET - others)
        .expect("the filler grows into the rest of the budget");
    assert_eq!(
        gate.reserved_bytes(),
        GATE_BUDGET,
        "fixture: the budget is full"
    );
}

/// Take the overflow slot with `holder`: a processing step above the whole
/// budget, granted at once while the slot is free and nothing is queued.
async fn take_slot(holder: &mut UploadReservation, shutdown: &CancellationToken) {
    let grant = holder
        .reserve_processing(GATE_BUDGET + 1, shutdown)
        .await
        .expect("the slot is free");
    assert!(
        grant.entered_overflow && holder.in_overflow(),
        "fixture: the holder runs in the overflow slot"
    );
}

/// A commit charge that delegates to `inner` and notifies `waiting` when its
/// waiting `hold` starts: the wait point, observed.
struct NotifyingCharge<'a> {
    inner: UploadCommitCharge<'a>,
    waiting: Arc<Notify>,
}

#[async_trait::async_trait]
impl CommitCharge for NotifyingCharge<'_> {
    fn held(&self) -> u64 {
        self.inner.held()
    }

    fn target(&self, need: &ObjectStateNeed) -> u64 {
        self.inner.target(need)
    }

    fn rewrite_allowance(&self, need: &ObjectStateNeed) -> MutationAllowance {
        self.inner.rewrite_allowance(need)
    }

    fn try_hold(&mut self, target: u64) -> bool {
        self.inner.try_hold(target)
    }

    async fn hold(&mut self, target: u64) -> Result<(), ServiceError> {
        self.waiting.notify_one();
        self.inner.hold(target).await
    }
}

/// Shutdown while the call waits for its recharge on the gate —
/// the budget full and the overflow slot held, so the wait queues — answers
/// the upload charge's shared shutdown status, `UNAVAILABLE "server shutting
/// down"` with the `SERVER_SHUTTING_DOWN` detail and its retry hint (on the
/// call's first attempt; the handler logs it at INFO at stage `storage`),
/// with nothing applied and the permits free.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn refused_growth_at_shutdown_answers_server_shutting_down_b26e() {
    for (backend, layer) in [
        (UpdateLayerBackend::Txn, "recharge-shutdown-watch"),
        (UpdateLayerBackend::Lua, "recharge-shutdown-cached"),
    ] {
        let fx = Arc::new(fixture(layer, backend).await);
        if backend == UpdateLayerBackend::Lua {
            warm(&fx, 5_000).await;
        }
        let (before_seq, before_len) = (fx.seq().await, fx.stream_len().await);
        let gate = small_gate();
        let gs = GracefulShutdown::new();
        let mut caller = processing_reservation(&gate, "caller", gs.begun()).await;
        let mut slot = processing_reservation(&gate, "slot", gs.begun()).await;
        let mut filler = processing_reservation(&gate, "filler", gs.begun()).await;
        take_slot(&mut slot, gs.begun()).await;
        fill_budget(&gate, &mut filler);

        let waiting = Arc::new(Notify::new());
        let base = caller.reserved();
        let call = {
            let (fx, waiting, gs) = (Arc::clone(&fx), Arc::clone(&waiting), gs.clone());
            tokio::spawn(async move {
                let mut charge = NotifyingCharge {
                    inner: UploadCommitCharge::new(
                        &mut caller,
                        base,
                        &gs,
                        &fx.session_id,
                        &fx.layer,
                    ),
                    waiting,
                };
                let result = fx.update(structural_delta(1_000), &mut charge).await;
                let shut_down = charge.inner.shut_down();
                (result, shut_down)
            })
        };
        tokio::time::timeout(STEP_BUDGET, waiting.notified())
            .await
            .unwrap_or_else(|_| panic!("{layer}: the call must reach its recharge wait"));
        gs.cancel();
        let (result, shut_down) = tokio::time::timeout(STEP_BUDGET, call)
            .await
            .unwrap_or_else(|_| panic!("{layer}: shutdown must end the wait"))
            .expect("the call's task must not panic");

        match result {
            Err(ServiceError::UnavailableWithDetail { message, detail }) => {
                assert_eq!(
                    message, "server shutting down",
                    "{layer}: the shutdown answer"
                );
                assert_eq!(
                    *detail,
                    live_edit::error::RefusalDetail::server_shutting_down(),
                    "{layer}: the shared shutdown detail and its retry hint, on attempt 1"
                );
            }
            other => panic!(
                "{layer}: UNAVAILABLE server shutting down with its refusal detail; got {other:?}"
            ),
        }
        assert!(
            shut_down,
            "{layer}: the charge records the shutdown, so the handler marks its line"
        );
        assert_eq!(fx.seq().await, before_seq, "{layer}: nothing applied");
        assert_eq!(
            fx.stream_len().await,
            before_len,
            "{layer}: no stream entry"
        );
        assert!(
            permits_free(&fx.ls, fx.oid),
            "{layer}: the permits are free"
        );
        drop((slot, filler));
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// The overflow-slot deadlock
// ─────────────────────────────────────────────────────────────────────────────

/// The deadlock a recharge holding its per-object permits would
/// make: X, a structural call on object O, needs growth beyond what the full
/// budget has room for, while Z, another structural call on O, holds the
/// overflow slot. X waits on the gate; Z must then be able to take O's
/// permits — X gave them back for its wait — commit, and free the slot, so
/// X's wait is served and X commits too. Holding the permits across the wait
/// (as before Stage 2) deadlocks it: X waits on the slot Z holds and Z on the
/// permit X holds, until the deadline. On `lua` (the `structural` permit,
/// one structural call per object) and on `txn` (the `writers` permits, here
/// one per object).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn slot_holder_and_budget_waiter_on_one_object_both_commit_b26e() {
    for (backend, layer) in [
        (UpdateLayerBackend::Lua, "slot-deadlock-cached"),
        (UpdateLayerBackend::Txn, "slot-deadlock-watch"),
    ] {
        // One writer per object, so on `txn` too the two calls cannot both
        // hold O's permits.
        let fx = Arc::new(
            fixture_with(layer, backend, 4, |ls| {
                ls.with_update_layer_per_object_concurrency(1)
            })
            .await,
        );
        if backend == UpdateLayerBackend::Lua {
            warm(&fx, 5_000).await;
        }
        let before_seq = fx.seq().await;
        let gate = small_gate();
        let gs = GracefulShutdown::new();
        let mut x_reservation = processing_reservation(&gate, "x", gs.begun()).await;
        let mut z_reservation = processing_reservation(&gate, "z", gs.begun()).await;
        let mut filler = processing_reservation(&gate, "filler", gs.begun()).await;
        take_slot(&mut z_reservation, gs.begun()).await;
        fill_budget(&gate, &mut filler);

        let x_waiting = Arc::new(Notify::new());
        let x = {
            let (fx, waiting, gs) = (Arc::clone(&fx), Arc::clone(&x_waiting), gs.clone());
            tokio::spawn(async move {
                let base = x_reservation.reserved();
                let mut charge = NotifyingCharge {
                    inner: UploadCommitCharge::new(
                        &mut x_reservation,
                        base,
                        &gs,
                        &fx.session_id,
                        &fx.layer,
                    ),
                    waiting,
                };
                fx.update(structural_delta(1_000), &mut charge).await
            })
        };
        tokio::time::timeout(STEP_BUDGET, x_waiting.notified())
            .await
            .unwrap_or_else(|_| panic!("{layer}: X must reach its wait on the gate"));

        let z = {
            let (fx, gs) = (Arc::clone(&fx), gs.clone());
            tokio::spawn(async move {
                let base = z_reservation.reserved();
                let result = {
                    let mut charge = UploadCommitCharge::new(
                        &mut z_reservation,
                        base,
                        &gs,
                        &fx.session_id,
                        &fx.layer,
                    );
                    fx.update(structural_delta(1_001), &mut charge).await
                };
                // Z's call is over: its reservation, and the slot, go.
                drop(z_reservation);
                result
            })
        };
        let z_seq = tokio::time::timeout(STEP_BUDGET, z)
            .await
            .unwrap_or_else(|_| {
                panic!(
                    "{layer}: Z, holding the slot, must commit while X waits on the gate — a \
                     wait that kept O's permits would deadlock here"
                )
            })
            .expect("Z's task must not panic")
            .unwrap_or_else(|e| panic!("{layer}: Z commits; got {e:?}"));
        let x_seq = tokio::time::timeout(STEP_BUDGET, x)
            .await
            .unwrap_or_else(|_| panic!("{layer}: X must commit once Z freed the slot"))
            .expect("X's task must not panic")
            .unwrap_or_else(|e| panic!("{layer}: X commits; got {e:?}"));
        assert_eq!(
            (z_seq, x_seq),
            (before_seq + 1, before_seq + 2),
            "{layer}: Z committed first, X after it"
        );
        drop(filler);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CR0, the cold start's seam-A charge
// ─────────────────────────────────────────────────────────────────────────────

/// CR0: a structural call on `lua` that cold-starts its object
/// and cannot hold the decoded entry's charge at seam A stores the entry —
/// when, as here and on every weighed shape at the default, its decoded
/// state and index build are within the plain read's hold allowance (over
/// it, nothing is
/// stored: [`refused_cold_start_charge_over_its_hold_allowance_stores_nothing_and_rereads_ug2`]) —
/// gives back the cold start's read permits — at the wait no cold-start read
/// holds a permit, and the entry is in the cache — and its re-run after the
/// wait is a cache hit that commits once: the recharge counted on
/// `structural_full`, the commit on `structural` at its first attempt.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn refused_cold_start_charge_stores_the_entry_and_the_rerun_is_a_hit_b26e() {
    let fx = fixture("cr0-refused", UpdateLayerBackend::Lua).await;
    assert!(
        fx.ls.structural_cache().weight_hint(fx.oid).is_none(),
        "fixture: the object is not cached, so the call cold-starts it"
    );
    let lookups = |outcome: &str| {
        fx.sample(
            "live_edit_structural_cache_lookups_total",
            &[("outcome", outcome)],
        ) as u64
    };
    let (before_seq, before_cold, before_hit) = (fx.seq().await, lookups("cold"), lookups("hit"));

    let seen = Arc::new(Mutex::new(Vec::new()));
    let probe: HoldProbe = {
        let (ls, seen, oid) = (Arc::clone(&fx.ls), Arc::clone(&seen), fx.oid);
        Box::new(move |_target| {
            let cache = ls.structural_cache();
            seen.lock().unwrap_or_else(PoisonError::into_inner).push((
                cache.cold_start_reads_in_flight_for_test(),
                cache.weight_hint(oid).is_some(),
                permits_free(&ls, oid),
            ));
            async {}.boxed()
        })
    };
    let (charge, log) = ScriptedCharge::new(0);
    let mut charge = charge.with_probe(probe);
    let seq = tokio::time::timeout(STEP_BUDGET, fx.update(structural_delta(1_000), &mut charge))
        .await
        .expect("the call must finish")
        .unwrap_or_else(|e| panic!("the call commits after its recharge; got {e:?}"));

    let log = lock(&log);
    assert!(
        matches!(log.needs.first(), Some(ObjectStateNeed::Cached { .. })),
        "fixture: the first charge point is the cold start's (a cached need); {log:?}"
    );
    assert_eq!(
        log.tries.first().map(|(_, granted)| *granted),
        Some(false),
        "fixture: seam A's charge is refused; {log:?}"
    );
    let seen = seen.lock().unwrap_or_else(PoisonError::into_inner).clone();
    assert_eq!(seen.len(), 1, "one wait");
    let (in_flight, stored, free) = seen[0];
    assert_eq!(
        (in_flight.plain, in_flight.fenced, in_flight.raised),
        (0, 0, 0),
        "at the wait the cold start holds no read permit"
    );
    assert!(stored, "the refused cold start stored its entry");
    assert!(free, "and the per-object permits are free");
    assert_eq!(seq, before_seq + 1, "one commit");
    assert_eq!(
        (lookups("cold") - before_cold, lookups("hit") - before_hit),
        (1, 1),
        "one cold start, then the re-run's hit"
    );
    assert_eq!(
        fx.recharges("structural_full"),
        1,
        "the cold start's recharge"
    );
    assert_eq!(
        fx.attempts("structural"),
        (1, 1),
        "the hit commits on the call's first attempt"
    );
}

/// RP0: a structural call on `lua` whose object is cached starts at its
/// repair's first read, the held entry at its stored weight (state and plan
/// indexes) and the first read's reply at its entry and byte bounds
/// (`STRUCTURAL_REPAIR_INITIAL_ENTRIES` and `STRUCTURAL_REPAIR_INITIAL_BYTES`)
/// — the first charge point a cache hit prices. Refused (no
/// room), the read is never issued: the call waits once, for that point's
/// whole target, with both per-object permits free and the entry put back
/// at its cut — still two commits behind, as another store left it — and the
/// recharge is counted on the hit's path, `structural`, or `reset` for a
/// reset, never on the cold start's `structural_full`. The resumed attempt is
/// a hit that repairs from that cut and commits once, at its first attempt.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn refused_repair_read_on_a_hit_waits_with_the_entry_at_its_cut_and_recharges_on_the_hits_path_fsz(
) {
    let reset = OwnedDelta {
        is_diff: true,
        base_version: 0,
        ..structural_delta(1_000)
    };
    for (layer, delta, path) in [
        ("rp0-structural", structural_delta(1_000), "structural"),
        ("rp0-reset", reset, "reset"),
    ] {
        let fx = fixture(layer, UpdateLayerBackend::Lua).await;
        warm(&fx, 5_000).await;
        let (cut_seq, cut_state, weight) = {
            let held = fx
                .ls
                .structural_cache()
                .checkout(fx.oid)
                .expect("fixture: the entry is cached");
            let cut = (
                held.entry().seq,
                held.entry().state.clone(),
                held.entry().weight_bytes(),
            );
            held.put_back_held();
            cut
        };
        // Two commits by another store: the hit has entries to repair.
        let other = LayerStore::new(fx.tv.client().clone(), fx.tv.key_schema().clone(), 4096)
            .with_update_layer_backend(UpdateLayerBackend::Txn);
        for id in [6_000, 6_001] {
            let delta = Some(structural_delta(id));
            let relay = relay_for(&delta);
            let (landed, _) = other
                .update_layer_with_summary(
                    &fx.session_id,
                    &fx.layer,
                    fx.oid,
                    Arc::new(delta),
                    relay,
                    "other-writer",
                    1,
                    None,
                    &mut NoCommitCharge,
                )
                .await;
            landed.expect("fixture: the other store's commit lands");
        }
        let before_seq = fx.seq().await;
        assert_eq!(
            before_seq,
            cut_seq + 2,
            "fixture: the entry is two commits behind"
        );
        let (before_recharges, before_full, before_attempts) = (
            fx.recharges(path),
            fx.recharges("structural_full"),
            fx.attempts(path),
        );

        let seen = Arc::new(Mutex::new(Vec::new()));
        let probe: HoldProbe = {
            let (ls, seen, oid, cut_state) = (
                Arc::clone(&fx.ls),
                Arc::clone(&seen),
                fx.oid,
                cut_state.clone(),
            );
            Box::new(move |_target| {
                let held = ls
                    .structural_cache()
                    .checkout(oid)
                    .expect("the waiting call put its entry back");
                let at_cut = (held.entry().seq, held.entry().state == cut_state);
                held.put_back_held();
                seen.lock()
                    .unwrap_or_else(PoisonError::into_inner)
                    .push((at_cut, permits_free(&ls, oid)));
                async {}.boxed()
            })
        };
        let (charge, log) = ScriptedCharge::new(0);
        let mut charge = charge.with_probe(probe);
        let seq = tokio::time::timeout(STEP_BUDGET, fx.update(delta, &mut charge))
            .await
            .unwrap_or_else(|_| panic!("{layer}: the call must finish"))
            .unwrap_or_else(|e| panic!("{layer}: the call commits after its recharge; got {e:?}"));

        let rp0 = ObjectStateNeed::cached_repair(
            weight,
            RepairTransient {
                reply_entry_count: STRUCTURAL_REPAIR_INITIAL_ENTRIES
                    .min(STRUCTURAL_REPAIR_MAX_ENTRIES),
                reply_bytes: STRUCTURAL_REPAIR_INITIAL_BYTES,
                ..RepairTransient::default()
            },
        );
        let rp0_target = object_state_charge_for_test(&rp0);
        {
            let log = lock(&log);
            assert_eq!(
                log.needs.first(),
                Some(&rp0),
                "{layer}: the hit's first charge point is RP0; {log:?}"
            );
            assert_eq!(
                log.tries.first(),
                Some(&(rp0_target, false)),
                "{layer}: RP0 is refused; {log:?}"
            );
            assert_eq!(
                log.holds,
                vec![rp0_target],
                "{layer}: one wait, for RP0's whole target; {log:?}"
            );
        }
        assert_eq!(
            *seen.lock().unwrap_or_else(PoisonError::into_inner),
            vec![((cut_seq, true), true)],
            "{layer}: at the wait the entry is back at its cut, nothing replayed, and the \
             per-object permits are free"
        );
        assert_eq!(seq, before_seq + 1, "{layer}: one commit");
        assert_eq!(
            (
                fx.recharges(path) - before_recharges,
                fx.recharges("structural_full") - before_full
            ),
            (1, 0),
            "{layer}: the recharge is counted on the hit's path, `{path}`"
        );
        let (count, sum) = fx.attempts(path);
        assert_eq!(
            (count - before_attempts.0, sum - before_attempts.1),
            (1, 1),
            "{layer}: the hit commits on the call's first attempt"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// RP2, the per-record repair charge, and the resumed RP0 floor (b.fsz)
// ─────────────────────────────────────────────────────────────────────────────

/// The big parent of [`big_parent_state`]: the root's one child.
const BIG_PARENT: u64 = 50;

/// The first id of [`BIG_PARENT`]'s children.
const BIG_PARENT_FIRST_CHILD: u64 = 100_000;

/// A base whose root lists [`BIG_PARENT`] alone, which lists `children`
/// children (ids from [`BIG_PARENT_FIRST_CHILD`]): a long list a record or a
/// plan adds one child to.
fn big_parent_state(children: u64) -> ObjectState {
    let mut state = empty_state();
    state.sections.insert(BIG_PARENT, section(BIG_PARENT, 0, 0));
    state.children.insert(0, vec![BIG_PARENT]);
    let kids: Vec<u64> = (BIG_PARENT_FIRST_CHILD..BIG_PARENT_FIRST_CHILD + children).collect();
    for (order, id) in kids.iter().enumerate() {
        state
            .sections
            .insert(*id, section(*id, BIG_PARENT, order as u32));
    }
    state.children.insert(BIG_PARENT, kids);
    state
}

/// One field write on [`BIG_PARENT`]: a pending record with no structure.
fn field_write(key: &str) -> OwnedDelta {
    OwnedDelta {
        set_fields: vec![oodf_ffi::SetField {
            section_id: BIG_PARENT,
            key_name: key.to_owned(),
            value_or_ext_hash: b"value".to_vec(),
            ext_value_size: 0,
            ext_value_index: 0,
            set_order: 1,
        }],
        ..Default::default()
    }
}

/// One `CreateSection` of `section_id` under `parent_id`.
fn create_under(parent_id: u64, section_id: u64) -> OwnedDelta {
    OwnedDelta {
        struct_commands: vec![StructCommand::CreateSection(CommandCreateSection {
            parent_id,
            section_id,
            section_name: format!("new{section_id}"),
            section_type: 1,
        })],
        ..Default::default()
    }
}

/// Commit `deltas` to `oid` in order through another store, on the WATCH
/// path, whose commits never touch the fixture store's structural cache: the
/// records the fixture's next cache hit repairs.
async fn commit_elsewhere(fx: &Fixture, oid: u64, deltas: Vec<OwnedDelta>) {
    let other = LayerStore::new(fx.tv.client().clone(), fx.tv.key_schema().clone(), 4096)
        .with_update_layer_backend(UpdateLayerBackend::Txn);
    for delta in deltas {
        let delta = Some(delta);
        let relay = relay_for(&delta);
        let (landed, _) = other
            .update_layer_with_summary(
                &fx.session_id,
                &fx.layer,
                oid,
                Arc::new(delta),
                relay,
                "other-writer",
                1,
                None,
                &mut NoCommitCharge,
            )
            .await;
        landed.expect("fixture: the other store's commit lands");
    }
}

/// `oid`'s header `commit_chain`.
async fn commit_chain_of(fx: &Fixture, oid: u64) -> String {
    fx.tv
        .client()
        .pool
        .hget(
            fx.tv.key_schema().obj_header(&oid.to_string()),
            HEADER_FIELD_COMMIT_CHAIN,
        )
        .await
        .expect("HGET obj:hdr commit_chain must succeed")
}

/// `oid`'s state as a cold start loads it, on a cache of its own: the
/// oracle a repaired entry must equal.
async fn cold_start_state(fx: &Fixture, oid: u64) -> (u64, String, CompactState) {
    let cache = StructuralCache::new(1 << 30);
    let request = StructuralReadRequest::new(oid, "cold-start-oracle", COLD_START_TOKEN);
    match read_for_structural_edit(fx.tv.client(), fx.tv.key_schema(), &cache, &request).await {
        Ok(StructuralRead::Ready(ready)) => {
            let entry = ready.entry.entry();
            (entry.seq, entry.commit_chain.clone(), entry.state.clone())
        }
        other => panic!("the cold start loads the object; got {other:?}"),
    }
}

/// The cold-start oracle's commit token; no commit records it.
const COLD_START_TOKEN: &str = "c01d57a2c01d57a2c01d57a2c01d57a2";

/// The repair charge point of each of the needs `log` priced in `needs` —
/// one attempt's, which begins at its RP0 — by the shared classifier
/// (`structural_cache_read_test::repair_points`), with the need.
fn classified(
    log: &ChargeLog,
    needs: std::ops::Range<usize>,
) -> Vec<(RepairPoint, ObjectStateNeed)> {
    let needs = &log.needs[needs];
    repair_points(needs)
        .into_iter()
        .zip(needs.iter().copied())
        .filter_map(|(point, need)| point.map(|point| (point, need)))
        .collect()
}

/// The first repair read's reply at its production bounds: the first read's
/// entry bound (`STRUCTURAL_REPAIR_INITIAL_ENTRIES`, at most the per-read
/// `STRUCTURAL_REPAIR_MAX_ENTRIES`) and byte bound.
fn first_read_transient() -> RepairTransient {
    RepairTransient {
        reply_entry_count: STRUCTURAL_REPAIR_INITIAL_ENTRIES.min(STRUCTURAL_REPAIR_MAX_ENTRIES),
        reply_bytes: STRUCTURAL_REPAIR_INITIAL_BYTES,
        ..RepairTransient::default()
    }
}

/// The `replay_delta_bytes` of each RP2 among `needs`, in order: on an entry
/// holding its plan indexes, each record's own delta part.
fn record_deltas(needs: &[(RepairPoint, ObjectStateNeed)]) -> Vec<u64> {
    needs
        .iter()
        .filter(|(point, _)| *point == RepairPoint::Record)
        .map(|(_, need)| match need {
            ObjectStateNeed::Cached { repair, .. } => repair.replay_delta_bytes,
            other => panic!("an RP2 is a cached need; got {other:?}"),
        })
        .collect()
}

/// A4 (i), the per-record charge point (RP2) refused mid-chunk, and the
/// resume from its cut. A cache hit's repair reads five pending records in
/// one chunk — two field writes, a child added under a parent of 4,000
/// children (whose list doubles: the growth no per-byte term covers) and two
/// more field writes — under a charge with production pricing whose first
/// growth for a record's own growth or rewrite is refused once. The repair
/// stops between the second and third records: at the wait the entry is put
/// back (and its settle, with the shadow check on, passes) at the second
/// record's cut — `seq` two past the base, the header's `commit_chain`
/// there — without the third record's section, and the call holds nothing.
/// After the grant only the last three records replay: RP2 before the wait
/// prices the first three records (the third refused), after it the last
/// three. One wait, one commit, and the entry the commit leaves equals a
/// cold start of the object.
///
/// Fails without RP2: the record's growth would not be charged, so nothing
/// would be refused (no wait).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn refused_record_growth_mid_chunk_waits_at_the_records_cut_and_resumes_from_it_fsz() {
    const CHILDREN: u64 = 4_000;
    const NEW_CHILD: u64 = 200_000;
    let fx = fixture("rp2-mid-chunk", UpdateLayerBackend::Lua).await;
    assert!(
        fx.ls.structural_cache().shadow_check_for_test(),
        "fixture: the settle's shadow check is on"
    );
    let oid = create_state(
        &fx.ls,
        &fx.session_id,
        "rp2-mid-chunk-base",
        &big_parent_state(CHILDREN),
    )
    .await;
    warm_on(&fx, oid, 5_000).await;
    let base_seq = fx.seq_of(oid).await;
    commit_elsewhere(&fx, oid, vec![field_write("r1"), field_write("r2")]).await;
    let cut_chain = commit_chain_of(&fx, oid).await;
    commit_elsewhere(
        &fx,
        oid,
        vec![
            create_under(BIG_PARENT, NEW_CHILD),
            field_write("r4"),
            field_write("r5"),
        ],
    )
    .await;
    let pending_seq = fx.seq_of(oid).await;
    assert_eq!(pending_seq, base_seq + 5, "fixture: five pending records");
    let deltas: Vec<u64> = stream_entries(&fx.tv, oid, base_seq, pending_seq)
        .await
        .into_iter()
        .map(|(_, delta)| delta.expect("a relayed record's OMPF parses"))
        .collect();

    let seen = Arc::new(Mutex::new(Vec::new()));
    let probe: HoldProbe = {
        let (ls, seen) = (Arc::clone(&fx.ls), Arc::clone(&seen));
        Box::new(move |_target| {
            let held = ls
                .structural_cache()
                .checkout(oid)
                .expect("the waiting call put its entry back");
            let cut = (
                held.entry().seq,
                held.entry().commit_chain.clone(),
                held.entry().state.sections.contains_key(&NEW_CHILD),
            );
            let stored = held.put_back_held().is_stored();
            seen.lock().unwrap_or_else(PoisonError::into_inner).push((
                cut,
                stored,
                permits_free(&ls, oid),
            ));
            async {}.boxed()
        })
    };
    let (charge, log) = ScriptedCharge::new(u64::MAX);
    let mut charge = charge.with_probe(probe).refusing_first_record_growth();
    let seq = tokio::time::timeout(
        STEP_BUDGET,
        fx.update_on(oid, structural_delta(1_000), &mut charge),
    )
    .await
    .expect("the call must finish")
    .unwrap_or_else(|e| panic!("the call commits after its wait; got {e:?}"));

    let log = std::mem::take(&mut *lock(&log));
    assert_eq!(log.holds.len(), 1, "one wait; {log:?}");
    assert_eq!(
        *seen.lock().unwrap_or_else(PoisonError::into_inner),
        vec![((base_seq + 2, cut_chain, false), true, true)],
        "at the wait the entry is back at the second record's cut, without the third record, \
         its settle passed, and the permits are free"
    );
    let (needs_at_hold, tries_at_hold) = log.at_hold[0];
    let (refused_target, granted) = log.tries[tries_at_hold - 1];
    assert!(
        !granted && refused_target == log.holds[0],
        "the call waited for the refused point's target; {log:?}"
    );
    assert!(
        is_record_need(&log.needs[needs_at_hold - 1]),
        "the refused point is a record's RP2; {log:?}"
    );
    assert_eq!(
        record_deltas(&classified(&log, 0..needs_at_hold)),
        deltas[..3].to_vec(),
        "before the wait RP2 priced the first three records, the third refused"
    );
    assert_eq!(
        record_deltas(&classified(&log, needs_at_hold..log.needs.len())),
        deltas[2..].to_vec(),
        "after it only the last three records replay"
    );
    assert_eq!(seq, pending_seq + 1, "one commit");
    assert_eq!(
        fx.recharges("structural"),
        1,
        "one recharge, on the hit's path"
    );
    let after = held_snapshot(&fx.ls, oid);
    let (cold_seq, _, cold_state) = cold_start_state(&fx, oid).await;
    assert_eq!(
        (after.seq, after.state),
        (cold_seq, cold_state),
        "the entry the commit left is a cold start's"
    );
}

/// Production prices, read back through the charge: what one list entry of
/// a plan's rewrite costs on the cached route, and the first repair read's
/// reply term (its entry and byte bounds), in bytes.
fn list_entry_and_first_read_prices() -> (u64, u64) {
    let nothing =
        object_state_charge_for_test(&ObjectStateNeed::cached(0, RewriteCount::default()));
    let per_list_entry = (object_state_charge_for_test(&ObjectStateNeed::cached(
        0,
        RewriteCount {
            list_entries: 1_000_000,
            ..RewriteCount::default()
        },
    )) - nothing)
        .div_ceil(1_000_000);
    let first_read =
        object_state_charge_for_test(&ObjectStateNeed::cached_repair(0, first_read_transient()))
            - nothing;
    (per_list_entry, first_read)
}

/// One call, under production pricing with its first plan growth refused,
/// that adds a child under [`BIG_PARENT`] of `children` children on a hit
/// that first repairs `pending`, one record another store committed after
/// the cache warm-up (which itself creates one section under the root): the
/// call's charge log, the entry as put back at each wait, the object's cut
/// before the call as a cold start loads it (the pending record in it: what
/// the repaired, rolled-back entry must equal), the delta's inserts and the
/// new child's id. `contended`: a competitor takes whatever the call gives
/// back after its wait.
async fn refused_plan_on_a_repairing_hit(
    fx: &Fixture,
    layer: &str,
    children: u64,
    pending: OwnedDelta,
    contended: bool,
) -> (
    ChargeLog,
    Vec<HeldSnapshot>,
    (u64, String, CompactState),
    PlannedInserts,
    u64,
) {
    const NEW_CHILD: u64 = 300_000;
    let oid = create_state(&fx.ls, &fx.session_id, layer, &big_parent_state(children)).await;
    warm_on(fx, oid, 5_000).await;
    commit_elsewhere(fx, oid, vec![pending]).await;
    let before_seq = fx.seq_of(oid).await;
    let before = cold_start_state(fx, oid).await;
    let delta = create_under(BIG_PARENT, NEW_CHILD);
    let inserts = PlannedInserts::of(Some(&delta));
    let at_wait = Arc::new(Mutex::new(Vec::new()));
    let (charge, log) = ScriptedCharge::new(u64::MAX);
    let charge = charge
        .with_probe(snapshot_at_each_wait(&fx.ls, oid, &at_wait))
        .refusing_first_plan_growth();
    let mut charge = if contended {
        charge.contended_after_hold(0)
    } else {
        charge
    };
    let rollbacks = fx.ls.structural_cache().rollbacks_for_test();
    let seq = tokio::time::timeout(STEP_BUDGET, fx.update_on(oid, delta, &mut charge))
        .await
        .unwrap_or_else(|_| panic!("{layer}: the call must finish"))
        .unwrap_or_else(|e| panic!("{layer}: the call commits; got {e:?}"));
    assert_eq!(seq, before_seq + 1, "{layer}: one commit");
    let log = std::mem::take(&mut *lock(&log));
    let rolled_back = fx.ls.structural_cache().rollbacks_for_test() - rollbacks;
    assert_eq!(
        rolled_back,
        log.holds.len() as u64,
        "{layer}: each wait followed a refused plan, rolled back: holds {:?}, tries {:?}",
        log.holds,
        log.tries
    );
    let at_wait = std::mem::take(&mut *at_wait.lock().unwrap_or_else(PoisonError::into_inner));
    (log, at_wait, before, inserts, NEW_CHILD)
}

/// The sections of [`refused_plan_on_a_hit_waits_for_the_resumed_repairs_first_read_and_is_never_refused_there_fsz`]'s
/// base once warmed — [`BIG_PARENT`], its children and the warm-up's section
/// — so its compacted sections table is at its load limit: 3,584 entries in
/// 4,096 buckets (hashbrown's capacity for 4,096 buckets is 7/8 of them,
/// whatever its group width: at a capacity of 15 or more the bucket count
/// does not depend on it).
const F1_SECTIONS: u64 = 3_584;

/// F1 (b.fsz): a refused plan growth on a cache hit waits for the larger of
/// CR1 of the entry as put back plus the plan's exact need and the resumed
/// hit's first charge point, its repair's first RP0 on that entry — here the
/// RP0 — so a contended resume is never refused at that point. The hit's
/// repair replays one pending creation that doubles the entry's sections
/// table (at its load limit, [`F1_SECTIONS`]), priced 1:1 by its RP2: the
/// entry grows by the new table's bytes while what the call holds grows past
/// the first read's RP0 only by the RP2's small rest, so the plan's rewrite
/// allowance is the first read's reply term less that growth. The plan — a
/// child added under a parent of 3,582 children, its listing about 0.98 of
/// that reply term — asks past the allowance and is refused once, though
/// its exact need on the entry as put back is below the resumed RP0, a
/// self-checked premise. The charge is production-priced, and after each
/// wait a competitor takes whatever the call gives back, so the resumed
/// attempt holds only what it waited for. The first wait is for the resumed
/// RP0 on the entry as put back — the test's own pricing of that entry: its
/// state and index weight plus the first read's reply at its entry and byte
/// bounds — and the resumed attempt's first charge point is exactly that
/// need, held without a refusal. At the wait the plan is rolled back: the
/// pre-plan cut, the pending record replayed into it and the new child
/// absent. The resumed plan may ask once more for its own growth — its
/// rollback-rebuild projections follow the layout the rollback left, which
/// tombstones can thin (`storage::layer`, the `Overflow` arm: capacity,
/// never memory) — so at most two waits, a second one only for a plan's
/// need, and one commit. The sections table's doubling holds at either
/// group width: insert-only, at a capacity past 15.
///
/// Fails with the wait on the CR1 form alone: the resumed RP0 is refused.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn refused_plan_on_a_hit_waits_for_the_resumed_repairs_first_read_and_is_never_refused_there_fsz(
) {
    const DOUBLING_SECTION: u64 = 400_000;
    let children = F1_SECTIONS - 2;
    let fx = fixture("f1-floor", UpdateLayerBackend::Lua).await;
    let (log, at_wait, before, inserts, new_child) = refused_plan_on_a_repairing_hit(
        &fx,
        "f1-floor-base",
        children,
        create_under(0, DOUBLING_SECTION),
        true,
    )
    .await;

    assert!(
        (1..=2).contains(&log.holds.len()),
        "one wait, or two with the resumed plan's own; {log:?}"
    );
    assert_eq!(
        fx.recharges("structural"),
        log.holds.len() as u64,
        "each wait a recharge"
    );
    let (needs_at_hold, tries_at_hold) = log.at_hold[0];
    assert!(
        classified(&log, 0..needs_at_hold)
            .iter()
            .any(|(point, need)| {
                *point == RepairPoint::Record
                    && matches!(need, ObjectStateNeed::Cached { growth, .. } if *growth > 0)
            }),
        "fixture: the pending creation's RP2 priced the sections table's doubling; {log:?}"
    );
    let put_back = &at_wait[0];
    assert_eq!(
        put_back.layouts.sections.0,
        F1_SECTIONS as usize + 1,
        "fixture: the replayed creation is in the entry"
    );
    assert_eq!(
        (put_back.seq, &put_back.state),
        (before.0, &before.2),
        "at the wait the plan is rolled back: the pre-plan cut, its pending record replayed"
    );
    assert!(
        !put_back.state.sections.contains_key(&new_child),
        "the refused plan's child is not in the entry"
    );
    let needed = suspended_need(&log, 0).rewrite();
    let cr1_form =
        object_state_charge_for_test(&independent_cr1(put_back, inserts).with_rewrite(needed));
    let floor = resumed_rp0_floor(&log, 0).expect("the cached route prices the resumed RP0");
    let independent_floor = ObjectStateNeed::cached_repair(
        put_back.state_bytes + put_back.index_bytes,
        first_read_transient(),
    );
    assert_eq!(
        floor, independent_floor,
        "the floor is the resumed RP0 on the entry as put back"
    );
    let floor_target = object_state_charge_for_test(&floor);
    assert!(
        needed.list_entries > 0 && cr1_form < floor_target,
        "fixture: the CR1 form ({cr1_form} B, list entries {}) is below the resumed RP0 \
         ({floor_target} B)",
        needed.list_entries
    );
    assert_eq!(
        log.holds[0], floor_target,
        "the first wait is for the larger of the two: the resumed RP0; {log:?}"
    );
    assert_eq!(
        log.needs.get(needs_at_hold),
        Some(&floor),
        "the resumed attempt's first charge point is that RP0; {log:?}"
    );
    assert!(
        log.tries[tries_at_hold..]
            .iter()
            .all(|&(target, granted)| granted || target != floor_target),
        "the resumed RP0 is never refused; {log:?}"
    );
    if log.holds.len() == 2 {
        assert!(
            is_plan_need(&suspended_need(&log, 1)),
            "a second wait is only for the resumed plan's own growth; {log:?}"
        );
    }
}

/// F2 (b.fsz): the plan's rollback and the stream repair's transient
/// together, under production pricing. A cache hit repairs one pending
/// record (its RP0, RP1 and RP2 priced in full), then its plan — a child
/// added under a parent of `L` children, `L` sized from the production
/// prices so the list rewrite is 1.1 times the first read's reply term — asks
/// past what the repair left held and is refused once. The plan is rolled
/// back, and the call waits once, for the larger of CR1 of the entry as put
/// back (the test's own pricing, [`independent_cr1`]) plus the plan's exact
/// need and the resumed RP0 floor: here the CR1 form. At the wait the entry
/// is the repaired pre-plan cut; the resumed plan fits what the call waited
/// for (it never asks again) and commits once.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn refused_plan_on_a_repairing_hit_rolls_back_and_waits_for_its_cr1_form_under_production_pricing_fsz(
) {
    let (per_list_entry, first_read) = list_entry_and_first_read_prices();
    let children = 11 * first_read / (10 * per_list_entry);
    let fx = fixture("f2-rollback-repair", UpdateLayerBackend::Lua).await;
    let (log, at_wait, before, inserts, new_child) = refused_plan_on_a_repairing_hit(
        &fx,
        "f2-rollback-repair-base",
        children,
        field_write("pending"),
        false,
    )
    .await;

    assert_eq!(log.holds.len(), 1, "one wait; {log:?}");
    let (needs_at_hold, _) = log.at_hold[0];
    let before_wait = classified(&log, 0..needs_at_hold);
    for point in [RepairPoint::Read, RepairPoint::Replay, RepairPoint::Record] {
        assert!(
            before_wait.iter().any(|(p, _)| *p == point),
            "the hit's repair priced its {point:?} point in full before the plan; {log:?}"
        );
    }
    let put_back = &at_wait[0];
    assert_eq!(
        (put_back.seq, &put_back.state),
        (before.0, &before.2),
        "at the wait the plan is rolled back to the repaired pre-plan cut"
    );
    assert!(!put_back.state.sections.contains_key(&new_child));
    let needed = suspended_need(&log, 0).rewrite();
    let cr1_form =
        object_state_charge_for_test(&independent_cr1(put_back, inserts).with_rewrite(needed));
    let floor = object_state_charge_for_test(
        &resumed_rp0_floor(&log, 0).expect("the cached route prices the resumed RP0"),
    );
    assert!(
        cr1_form > floor,
        "fixture: the CR1 form ({cr1_form} B) is above the resumed RP0 ({floor} B)"
    );
    assert_eq!(
        log.holds[0], cr1_form,
        "the wait is for the larger of the two: CR1 of the entry as put back plus the plan's \
         exact need; {log:?}"
    );
    let resumed_asks = log.needs[needs_at_hold..]
        .iter()
        .filter(|need| is_plan_need(need))
        .count();
    assert_eq!(
        resumed_asks, 1,
        "the resumed plan never asked: only its exact trim prices a rewrite; {log:?}"
    );
    assert_eq!(fx.recharges("structural"), 1, "one recharge");
}

/// W4-T4: CR0 refused on a plain cold start whose decoded state and index
/// build are over its hold allowance (here a test override of 1 B, under
/// any object's hold) loads nothing: at the wait no cold-start read holds a
/// permit and nothing is stored, so the re-run is a second cold start —
/// the same CR0 need priced again, granted under the charge the call waited
/// for (no second wait) — that commits once. Two cold lookups and no hit,
/// one recharge on `structural_full`, two atomic snapshot reads and one
/// `atomic` route tick in all (the first read, which loaded nothing, ticks
/// none). Against [`refused_cold_start_charge_stores_the_entry_and_the_rerun_is_a_hit_b26e`],
/// where the same refusal within the allowance stores the entry and the
/// re-run hits.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn refused_cold_start_charge_over_its_hold_allowance_stores_nothing_and_rereads_ug2() {
    let fx = fixture_with("cr0-over-allowance", UpdateLayerBackend::Lua, 4, |ls| {
        ls.with_plain_cold_start_hold_allowance_for_test(1)
    })
    .await;
    assert!(
        fx.ls.structural_cache().weight_hint(fx.oid).is_none(),
        "fixture: the object is not cached, so the call cold-starts it"
    );
    let lookups = |outcome: &str| {
        fx.sample(
            "live_edit_structural_cache_lookups_total",
            &[("outcome", outcome)],
        ) as u64
    };
    let atomic_route = || {
        fx.sample(
            "live_edit_structural_cold_start_reads_total",
            &[("route", "atomic")],
        ) as u64
    };
    let atomic_reads = || {
        fx.sample(
            "live_edit_snapshot_read_total",
            &[
                ("consumer", "structural_cache"),
                ("path", "atomic"),
                ("outcome", "ok"),
            ],
        ) as u64
    };
    let (before_seq, before_cold, before_hit) = (fx.seq().await, lookups("cold"), lookups("hit"));
    let (before_route, before_reads) = (atomic_route(), atomic_reads());

    let seen = Arc::new(Mutex::new(Vec::new()));
    let probe: HoldProbe = {
        let (ls, seen, oid) = (Arc::clone(&fx.ls), Arc::clone(&seen), fx.oid);
        Box::new(move |_target| {
            let cache = ls.structural_cache();
            seen.lock().unwrap_or_else(PoisonError::into_inner).push((
                cache.cold_start_reads_in_flight_for_test(),
                cache.weight_hint(oid).is_some(),
                permits_free(&ls, oid),
            ));
            async {}.boxed()
        })
    };
    let (charge, log) = ScriptedCharge::new(0);
    let mut charge = charge.with_probe(probe);
    let seq = tokio::time::timeout(STEP_BUDGET, fx.update(structural_delta(1_000), &mut charge))
        .await
        .expect("the call must finish")
        .unwrap_or_else(|e| panic!("the call commits after its recharge; got {e:?}"));

    let log = lock(&log);
    let cr0 = match log.needs.first() {
        Some(need @ ObjectStateNeed::Cached { .. }) => *need,
        other => panic!("fixture: the first charge point is the cold start's; got {other:?}"),
    };
    assert_eq!(
        log.tries.first(),
        Some(&(object_state_charge_for_test(&cr0), false)),
        "fixture: seam A's charge is refused; {log:?}"
    );
    let seen = seen.lock().unwrap_or_else(PoisonError::into_inner).clone();
    assert_eq!(seen.len(), 1, "one wait");
    let (in_flight, stored, free) = seen[0];
    assert_eq!(
        (in_flight.plain, in_flight.fenced, in_flight.raised),
        (0, 0, 0),
        "at the wait the cold start holds no read permit"
    );
    assert!(
        !stored,
        "over its allowance the refused cold start stored nothing"
    );
    assert!(free, "and the per-object permits are free");
    assert_eq!(
        log.holds.len(),
        1,
        "the re-run's CR0 is granted: no second wait"
    );
    let (needs_at_hold, _) = log.at_hold[0];
    assert_eq!(
        log.needs.get(needs_at_hold),
        Some(&cr0),
        "the re-run cold-starts again and prices the same CR0; {log:?}"
    );
    assert_eq!(seq, before_seq + 1, "one commit");
    assert_eq!(
        (lookups("cold") - before_cold, lookups("hit") - before_hit),
        (2, 0),
        "two cold starts, no hit"
    );
    assert_eq!(
        (atomic_reads() - before_reads, atomic_route() - before_route),
        (2, 1),
        "two atomic reads, one atomic route tick: the read that loaded nothing ticks none"
    );
    assert_eq!(
        fx.recharges("structural_full"),
        1,
        "the cold start's recharge"
    );
    assert!(
        fx.ls.structural_cache().weight_hint(fx.oid).is_some(),
        "the re-run's entry is cached after the commit"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// The WATCH read's allowance stop (WA1, WA2)
// ─────────────────────────────────────────────────────────────────────────────

/// Every non-waiting hold target of one uncharged-shaped run of `delta` on a
/// fresh base object of `fx`'s store (a charge that never refuses), in
/// order: the charge points' growth, against which a test sizes a room.
async fn calibrated_tries(
    fx: &Fixture,
    layer: &str,
    state: &ObjectState,
    delta: OwnedDelta,
) -> Vec<u64> {
    let oid = create_state(&fx.ls, &fx.session_id, layer, state).await;
    let (mut charge, log) = ScriptedCharge::new(u64::MAX);
    fx.update_on(oid, delta, &mut charge)
        .await
        .expect("the calibration run commits");
    let tries = lock(&log).tries.iter().map(|(t, _)| *t).collect();
    tries
}

/// `live_edit_snapshot_read_total{consumer="structural_update",path="unfenced",outcome}`.
fn structural_update_reads(fx: &Fixture, outcome: &str) -> u64 {
    fx.sample(
        "live_edit_snapshot_read_total",
        &[
            ("consumer", "structural_update"),
            ("path", "unfenced"),
            ("outcome", outcome),
        ],
    ) as u64
}

/// WA1, WA2: a WATCH attempt whose read cannot grow its charge
/// stops before its next window: at the count windows (WA1, the room is the
/// attempt's fixed cost alone), and at a later growth (WA2: on a base whose
/// names are longer than the read's sizing floor, so a window receives more
/// than WA1 projected; the room is exactly WA1's target, from a calibration
/// run of the same edit on an identical object, so WA1 is granted and the
/// read's next growth is not). Each stop records
/// `snapshot_read_total{consumer="structural_update",path="unfenced",outcome="too_large"}`,
/// and before the call waits the attempt has `UNWATCH`ed on the connection
/// that held its `WATCH` — a `MONITOR` record up to the wait; the WA1 stop
/// has issued no index page (`SSCAN`). The call then commits.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn watch_read_stops_before_its_next_window_and_unwatches_before_it_waits_b26e() {
    use crate::storage::snapshot_read_support::{monitor_fence, test_valkey_addr, MonitorTap};

    let fx = fixture("watch-stop", UpdateLayerBackend::Txn).await;
    let wa0 = object_state_charge_for_test(&ObjectStateNeed::watch_fixed());
    let long_named = long_named_state();
    let calibration = calibrated_tries(
        &fx,
        "watch-stop-calibration",
        &long_named,
        structural_delta(1_000),
    )
    .await;
    assert_eq!(calibration[0], wa0, "fixture: the first growth is WA0's");
    assert!(
        calibration.len() >= 3 && calibration[2] > calibration[1],
        "fixture: on the long-named base the read grows past WA1's target; {calibration:?}"
    );
    for (i, (stop, room, state)) in [
        ("WA1", wa0, base_state()),
        ("WA2", calibration[1], long_named.clone()),
    ]
    .into_iter()
    .enumerate()
    {
        // A fresh object identical to its calibration's, so WA1's target is
        // the calibrated one.
        let oid = create_state(&fx.ls, &fx.session_id, &format!("watch-stop-{i}"), &state).await;
        let before_too_large = structural_update_reads(&fx, "too_large");
        let before_seq = fx.seq_of(oid).await;
        let tap = Arc::new(tokio::sync::Mutex::new(
            MonitorTap::start(&test_valkey_addr()).await,
        ));
        let seen = Arc::new(Mutex::new(Vec::new()));
        let probe: HoldProbe = {
            let (tap, seen, client, prefix) = (
                Arc::clone(&tap),
                Arc::clone(&seen),
                fx.tv.client().clone(),
                fx.tv.prefix().to_owned(),
            );
            Box::new(move |_target| {
                let (tap, seen, client, prefix) = (
                    Arc::clone(&tap),
                    Arc::clone(&seen),
                    client.clone(),
                    prefix.clone(),
                );
                async move {
                    use fred::interfaces::KeysInterface;
                    let fence = monitor_fence(&prefix);
                    let _: i64 = client.pool.exists(&fence).await.expect("EXISTS the fence");
                    let commands = tap.lock().await.drain_until(&fence).await;
                    *seen.lock().unwrap_or_else(PoisonError::into_inner) = commands;
                }
                .boxed()
            })
        };
        let (charge, log) = ScriptedCharge::new(room);
        let mut charge = charge.with_probe(probe);
        let seq = tokio::time::timeout(
            STEP_BUDGET,
            fx.update_on(oid, structural_delta(1_000), &mut charge),
        )
        .await
        .unwrap_or_else(|_| panic!("{stop}: the call must finish"))
        .unwrap_or_else(|e| panic!("{stop}: the call commits after its recharge; got {e:?}"));
        assert_eq!(seq, before_seq + 1, "{stop}: one commit");
        {
            let log = lock(&log);
            assert_eq!(log.holds.len(), 1, "{stop}: fixture: the read stopped once");
            let refused = log
                .tries
                .iter()
                .position(|(_, granted)| !granted)
                .expect("a growth was refused");
            assert_eq!(
                refused,
                i + 1,
                "{stop}: the read's growth refused is the one after the room's: {:?}",
                log.tries
            );
        }
        assert_eq!(
            structural_update_reads(&fx, "too_large") - before_too_large,
            1,
            "{stop}: the stopped read is counted too_large"
        );

        let commands = seen.lock().unwrap_or_else(PoisonError::into_inner).clone();
        let header = fx.tv.key_schema().obj_header(&oid.to_string());
        let sections = fx.tv.key_schema().obj_sections_index(&oid.to_string());
        let watch_at = commands
            .iter()
            .rposition(|c| {
                c.name() == "WATCH" && c.args.get(1).map(String::as_str) == Some(header.as_str())
            })
            .unwrap_or_else(|| panic!("{stop}: the attempt WATCHed the header; {commands:?}"));
        let watcher = commands[watch_at].client.clone();
        assert!(
            commands[watch_at..]
                .iter()
                .any(|c| c.name() == "UNWATCH" && c.client == watcher),
            "{stop}: the attempt UNWATCHed on its connection before the wait; {:?}",
            &commands[watch_at..]
        );
        if stop == "WA1" {
            let pages = commands
                .iter()
                .filter(|c| {
                    c.name() == "SSCAN"
                        && c.args.get(1).map(String::as_str) == Some(sections.as_str())
                })
                .count();
            assert_eq!(
                pages, 0,
                "{stop}: the read stopped before its round-1 pages: no SSCAN of the sections \
                 index"
            );
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// The bounded plan
// ─────────────────────────────────────────────────────────────────────────────

/// `n` `CreateSection`s under the root from `first`: a plan whose rewrite
/// (two rows per section, and the root's list) outgrows any slack a small
/// object's read or entry leaves.
fn many_creates(first: u64, n: u64) -> OwnedDelta {
    OwnedDelta {
        struct_commands: (first..first + n)
            .map(|id| {
                StructCommand::CreateSection(CommandCreateSection {
                    parent_id: 0,
                    section_id: id,
                    section_name: format!("bulk{id}"),
                    section_type: 1,
                })
            })
            .collect(),
        ..Default::default()
    }
}

/// Descendants of the cascade fixture's one child: enough that deleting it
/// projects more rows (3 per descendant, at the row price) than a WATCH
/// read of the object leaves its plan beyond the read's own need (the
/// read's per-command term, capped at one window of commands, and its
/// per-entry term), so the WATCH plan asks its budget.
const CASCADE_DESCENDANTS: u64 = 4_000;

/// The cascade fixture's child of the root.
const CASCADE_ROOT: u64 = 50;

/// A layer whose one child of the root, [`CASCADE_ROOT`], has
/// [`CASCADE_DESCENDANTS`] children.
fn cascade_state() -> ObjectState {
    let mut state = empty_state();
    state
        .sections
        .insert(CASCADE_ROOT, section(CASCADE_ROOT, 0, 0));
    state.children.insert(0, vec![CASCADE_ROOT]);
    let ids: Vec<u64> = (1_000..1_000 + CASCADE_DESCENDANTS).collect();
    for (order, id) in ids.iter().enumerate() {
        state
            .sections
            .insert(*id, section(*id, CASCADE_ROOT, order as u32));
    }
    state.children.insert(CASCADE_ROOT, ids);
    state
}

/// Delete [`CASCADE_ROOT`] and so its whole subtree.
fn cascade_delta() -> OwnedDelta {
    OwnedDelta {
        struct_commands: vec![StructCommand::DeleteSection(CommandDeleteSection {
            parent_id: 0,
            section_id: CASCADE_ROOT,
        })],
        ..Default::default()
    }
}

/// The first section id [`cached_plan_creates`] creates: clear of the base's
/// children and of the warming commits' sections.
const CACHED_PLAN_FIRST: u64 = 10_000;

/// The cached route's bounded-plan edit: enough new sections under the root
/// that the root's listing — one list entry per section, at the list-entry
/// price — outgrows twice the reply term the hit's first repair read (RP0)
/// leaves held beyond the entry and its CR1 growth. RP0 is grow-only, so that term
/// stays held into the plan, which spends it as rewrite allowance before it
/// asks its budget. Both prices are read through the production charge, so
/// a refit of either keeps the plan past the allowance.
fn cached_plan_creates() -> OwnedDelta {
    use live_edit::storage::structural_cache_sync::STRUCTURAL_REPAIR_INITIAL_BYTES;

    let price = |need: ObjectStateNeed| object_state_charge_for_test(&need);
    let first_read_term = price(ObjectStateNeed::cached_repair(
        0,
        RepairTransient {
            reply_bytes: STRUCTURAL_REPAIR_INITIAL_BYTES,
            ..RepairTransient::default()
        },
    )) - price(ObjectStateNeed::cached_repair(
        0,
        RepairTransient::default(),
    ));
    let list_entry = price(ObjectStateNeed::cached(
        0,
        RewriteCount {
            list_entries: 1,
            ..RewriteCount::default()
        },
    )) - price(ObjectStateNeed::cached(0, RewriteCount::default()));
    many_creates(CACHED_PLAN_FIRST, 2 * first_read_term / list_entry)
}

/// The object and the edit of a bounded-plan test on `backend`: on the
/// cached route the fixture's object and [`cached_plan_creates`] (its plan
/// outgrows what the entry, its in-place growth term and the repair's first
/// read leave it); on WATCH a cascade object and its subtree's delete (its
/// plan outgrows the read's slack).
async fn bounded_plan_case(fx: &Fixture, backend: UpdateLayerBackend) -> (u64, OwnedDelta) {
    match backend {
        UpdateLayerBackend::Lua => (fx.oid, cached_plan_creates()),
        _ => (
            create_state(
                &fx.ls,
                &fx.session_id,
                &format!("{}-cascade", fx.layer),
                &cascade_state(),
            )
            .await,
            cascade_delta(),
        ),
    }
}

/// Whether the plan logged in `log` grew the charge through its budget: a
/// plan need with a rewrite ([`is_plan_need`]) priced before the last one
/// (the exact trim, CR2 or WA3), every growth granted.
fn plan_grew(log: &ChargeLog) -> bool {
    log.needs.iter().filter(|need| is_plan_need(need)).count() >= 2
        && log.tries.iter().all(|(_, granted)| *granted)
}

/// A cached structural edit under a charge whose plan's
/// rewrite exceeds what the charge holds beyond the entry and its CR1
/// growth (and the reply term the hit's first repair read left held) grows
/// the charge as it plans and plans once — one plan in place
/// (`in_place_plans_for_test`), no copy of the entry
/// (`plan_state_clones_for_test`) — and commits once, with no wait.
/// Discriminated by a budget that never extends (the plan overflows,
/// suspends and plans again).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn cached_plan_grows_its_charge_and_plans_once_b26e() {
    let fx = fixture("bounded-plan-cached", UpdateLayerBackend::Lua).await;
    warm(&fx, 5_000).await;
    let before_seq = fx.seq().await;
    let plans = || fx.ls.structural_cache().in_place_plans_for_test();
    let copies = || fx.ls.structural_cache().plan_state_clones_for_test();
    let (before_plans, before_copies) = (plans(), copies());
    let (mut charge, log) = ScriptedCharge::new(u64::MAX);
    tokio::time::timeout(STEP_BUDGET, fx.update(cached_plan_creates(), &mut charge))
        .await
        .expect("the call must finish")
        .expect("the call commits");
    assert_eq!(fx.seq().await, before_seq + 1, "one commit");
    let log = lock(&log);
    assert!(
        plan_grew(&log),
        "fixture: the plan's rewrite exceeded the held charge and grew it; {log:?}"
    );
    assert!(log.holds.is_empty(), "nothing waited; {log:?}");
    assert_eq!(plans() - before_plans, 1, "one plan in place: one plan");
    assert_eq!(copies(), before_copies, "the cached route copied no entry");
}

/// On the WATCH path the same: the plan grows the charge, the
/// attempt reads the object once (one `structural_update` read, `ok`) and
/// commits once, with no wait.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn watch_plan_grows_its_charge_and_reads_once_b26e() {
    let fx = fixture("bounded-plan-watch", UpdateLayerBackend::Txn).await;
    let (oid, delta) = bounded_plan_case(&fx, UpdateLayerBackend::Txn).await;
    let before_seq = fx.seq_of(oid).await;
    let before_reads = structural_update_reads(&fx, "ok");
    let (mut charge, log) = ScriptedCharge::new(u64::MAX);
    tokio::time::timeout(STEP_BUDGET, fx.update_on(oid, delta, &mut charge))
        .await
        .expect("the call must finish")
        .expect("the call commits");
    assert_eq!(fx.seq_of(oid).await, before_seq + 1, "one commit");
    let log = lock(&log);
    assert!(
        plan_grew(&log),
        "fixture: the plan's rewrite exceeded the read's slack and grew the charge; {log:?}"
    );
    assert!(log.holds.is_empty(), "nothing waited; {log:?}");
    assert_eq!(
        structural_update_reads(&fx, "ok") - before_reads,
        1,
        "one structural_update read: one plan"
    );
}

/// A plan whose first growth is refused suspends for the need with the
/// plan's exact (unscaled) rewrite; while it waits the held entry is
/// unchanged (cached route); after the wait, holding that target, the
/// resumed plan fits its allowance — no growth past the waited-for target —
/// and one commit lands, with one recharge. On the cached route and on
/// WATCH. The refused plan's heap — within its charge outside the plan, and
/// within the documented residual inside it — is measured by the upload
/// memory envelope's
/// `slow_path_refused_plan_on_a_full_budget_stays_within_its_charge_b26e`.
///
/// T5a, the cached arm (b.ug2 revision 3): the plan's creates go under a
/// parent whose name-index vector they push past its capacity, which the
/// rollback keeps (contents, not layout), so the entry as put back weighs
/// more than before the plan. The wait is for CR1 of the entry as put back
/// plus the plan's exact need, that CR1 priced by the test from the entry as
/// its probe read it at the wait ([`independent_cr1`]), not from anything the
/// call logged; the CR1 the resumed attempt logged is then pinned to it, and
/// the wait is above the same pricing of the pre-plan entry. Hard
/// preconditions, read at the wait: the
/// index weight as put back is above the pre-plan one, and every table of
/// the entry as put back has room for the plan's insert calls and removals
/// ([`assert_tables_ample`]), so no rollback-rebuild projection differs
/// between the two plans.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn refused_plan_growth_waits_for_the_exact_need_and_replans_within_it_b26e() {
    for (backend, layer, path) in [
        (
            UpdateLayerBackend::Lua,
            "bounded-plan-refused-cached",
            "structural",
        ),
        (
            UpdateLayerBackend::Txn,
            "bounded-plan-refused-watch",
            "watch",
        ),
    ] {
        let fx = fixture(layer, backend).await;
        let cached = backend == UpdateLayerBackend::Lua;
        let (oid, delta) = if cached {
            let oid = create_state(
                &fx.ls,
                &fx.session_id,
                &format!("{layer}-kept-parent"),
                &kept_parent_state(BASE_CHILDREN, |id| format!("child{id}")),
            )
            .await;
            warm_on(&fx, oid, 5_000).await;
            (oid, creates_under(KEPT_PARENT, 1_000, T5A_CREATES, "bulk"))
        } else {
            bounded_plan_case(&fx, backend).await
        };
        let before = cached.then(|| held_snapshot(&fx.ls, oid));
        let before_seq = fx.seq_of(oid).await;
        let inserts = PlannedInserts::of(Some(&delta));

        let unchanged = Arc::new(Mutex::new(Vec::new()));
        let at_wait = Arc::new(Mutex::new(Vec::new()));
        let probe: HoldProbe = {
            let (ls, before, unchanged, at_wait) = (
                Arc::clone(&fx.ls),
                before.clone(),
                Arc::clone(&unchanged),
                Arc::clone(&at_wait),
            );
            Box::new(move |_target| {
                if let Some(before) = &before {
                    let now = held_snapshot(&ls, oid);
                    unchanged
                        .lock()
                        .unwrap_or_else(PoisonError::into_inner)
                        .push(now.seq == before.seq && now.state == before.state);
                    at_wait
                        .lock()
                        .unwrap_or_else(PoisonError::into_inner)
                        .push(now);
                }
                async {}.boxed()
            })
        };
        let (charge, log) = ScriptedCharge::new(u64::MAX);
        let mut charge = charge
            .with_probe(probe)
            .refusing_first_plan_growth()
            .without_repair_transient();
        let seq = tokio::time::timeout(STEP_BUDGET, fx.update_on(oid, delta, &mut charge))
            .await
            .unwrap_or_else(|_| panic!("{layer}: the call must finish"))
            .unwrap_or_else(|e| panic!("{layer}: the call commits; got {e:?}"));

        let log = lock(&log);
        assert_eq!(log.holds.len(), 1, "{layer}: one wait; {log:?}");
        let waited_for = log.holds[0];
        let (needs_at_hold, _) = log.at_hold[0];
        // The suspension's target is the plan's need with its exact rewrite,
        // the last plan need priced before the wait (on the cached route the
        // resumed hit's first RP0 is priced after it, as the wait's floor;
        // this charge prices that floor at the entry alone, below the plan's).
        let suspended = suspended_need(&log, 0);
        assert_eq!(
            object_state_charge_for_test(&suspended),
            waited_for,
            "{layer}: the wait is for the plan's need; {log:?}"
        );
        assert!(
            suspended.rewrite() != RewriteCount::default(),
            "{layer}: the suspension carries the plan's rewrite; {suspended:?}"
        );
        // After the wait the resumed plan fits its allowance: it never asks
        // its budget, so the only need with a rewrite priced after the wait
        // is the exact trim's (CR2 / WA3), whose rows and list entries are
        // the plan's exact ones — those the call waited for, unscaled — and
        // which drops the plan-only name index.
        let resumed: Vec<ObjectStateNeed> = log.needs[needs_at_hold..]
            .iter()
            .filter(|need| is_plan_need(need))
            .copied()
            .collect();
        assert_eq!(
            resumed.len(),
            1,
            "{layer}: after the wait only the exact trim prices a rewrite: the resumed plan \
             never asked for more; {resumed:?}"
        );
        let (waited, trimmed) = (suspended.rewrite(), resumed[0].rewrite());
        assert_eq!(
            (trimmed.rows, trimmed.list_entries),
            (waited.rows, waited.list_entries),
            "{layer}: the call waited for the plan's exact rows and list entries, unscaled"
        );
        assert_eq!(
            (trimmed.indexed_sections, trimmed.adoption_entries),
            (0, 0),
            "{layer}: the exact trim keeps no plan-only scratch"
        );
        if let Some(before) = &before {
            assert_eq!(
                *unchanged.lock().unwrap_or_else(PoisonError::into_inner),
                vec![true],
                "{layer}: the held entry was put back with its pre-plan contents"
            );
            let at_wait = at_wait.lock().unwrap_or_else(PoisonError::into_inner);
            let put_back = &at_wait[0];
            // T5a's hard preconditions.
            assert!(
                put_back.index_bytes > before.index_bytes,
                "{layer}: fixture: the creates grew the kept parent's name-index vector, which \
                 the rollback keeps (index weight {} as put back, {} before the plan)",
                put_back.index_bytes,
                before.index_bytes
            );
            assert_tables_ample(
                put_back,
                TableCalls {
                    sections: T5A_CREATES as usize,
                    ..TableCalls::default()
                },
                layer,
            );
            // The wait is CR1 of the entry as put back, priced by the test
            // from that entry alone, plus the plan's exact need.
            let cr1 = independent_cr1(put_back, inserts);
            assert_eq!(
                waited_for,
                object_state_charge_for_test(&cr1.with_rewrite(waited)),
                "{layer}: the wait is CR1 of the entry as put back ({cr1:?}, priced from the \
                 entry) plus the plan's exact need; {log:?}"
            );
            // Then the implementation is pinned to that pricing: the CR1 the
            // resumed attempt logged is the independent one.
            assert_eq!(
                resumed_cr1(&log, 0),
                cr1,
                "{layer}: the resumed attempt's CR1 is the entry as put back's"
            );
            let pre_plan_form = object_state_charge_for_test(
                &independent_cr1(before, inserts).with_rewrite(waited),
            );
            assert!(
                waited_for > pre_plan_form,
                "{layer}: the entry as put back weighs more than the pre-plan one, so its wait \
                 ({waited_for}) is above the pre-plan form's ({pre_plan_form})"
            );
        }
        assert_eq!(seq, before_seq + 1, "{layer}: one commit");
        assert_eq!(fx.recharges(path), 1, "{layer}: one recharge");
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Grow-only charge points before the peak
// ─────────────────────────────────────────────────────────────────────────────

/// After a recharge, a competitor takes every
/// byte the call gives back. The resumed attempt's charge points before its
/// peak only grow (WA0 on WATCH; on the cached route its repair's RP0 and
/// CR1), so it gives nothing back before it reaches the memory it waited
/// for, and it causes no second suspension: `recharges_total{path}` is 1. An
/// exact WA0, RP0 or CR1 would shrink the charge to its own need, the
/// competitor would take the rest, and the read or the plan would be refused
/// again.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn resumed_attempt_keeps_what_its_recharge_waited_for_b26e() {
    for (backend, layer, path) in [
        (UpdateLayerBackend::Lua, "grow-only-cached", "structural"),
        (UpdateLayerBackend::Txn, "grow-only-watch", "watch"),
    ] {
        let fx = fixture(layer, backend).await;
        let cached = backend == UpdateLayerBackend::Lua;
        if cached {
            warm(&fx, 5_000).await;
        }
        let before_seq = fx.seq().await;
        // The first attempt suspends: on WATCH at WA1 (the room is the fixed
        // cost alone), on the cached route at its plan's first growth. After
        // the wait the room is exactly the target waited for — the
        // competitor holds the rest of the gate — and it takes whatever the
        // call gives back.
        let (charge, log) = if cached {
            let (charge, log) = ScriptedCharge::new(u64::MAX);
            (charge.refusing_first_plan_growth(), log)
        } else {
            ScriptedCharge::new(object_state_charge_for_test(&ObjectStateNeed::watch_fixed()))
        };
        let mut charge = charge.contended_after_hold(0);
        // On the cached route a plan that grows its charge, past what the
        // hit's first repair read leaves held.
        let delta = if cached {
            cached_plan_creates()
        } else {
            structural_delta(1_000)
        };
        let seq = tokio::time::timeout(STEP_BUDGET, fx.update(delta, &mut charge))
            .await
            .unwrap_or_else(|_| panic!("{layer}: the call must finish"))
            .unwrap_or_else(|e| panic!("{layer}: the call commits; got {e:?}"));
        let log = lock(&log);
        assert_eq!(seq, before_seq + 1, "{layer}: one commit");
        assert_eq!(
            log.holds.len(),
            1,
            "{layer}: one wait: the resumed attempt gave nothing back before its peak, so the \
             competitor took nothing it needed; {log:?}"
        );
        assert_eq!(
            fx.recharges(path),
            1,
            "{layer}: recharges_total{{path={path}}}"
        );
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Resumed attempts after a plan in place (b.ug2 revision 3)
// ─────────────────────────────────────────────────────────────────────────────
//
// A rollback restores the entry's contents and state weight, not its layout:
// a name-index vector the plan grew keeps its capacity, one the plan dropped
// comes back at its exact length, and a table can keep tombstones. So a call
// that suspends after a plan in place waits for CR1 of the entry as put back
// plus the plan's exact need (the `Overflow` arm), or for CR2's own target (a
// refused CR2). T5a is the cached arm of
// `refused_plan_growth_waits_for_the_exact_need_and_replans_within_it_b26e`.

/// T5a's creates under [`KEPT_PARENT`]: its one-entry name-index vector
/// (capacity 1) grows to capacity 4, within the sections table's room (22
/// sections at capacity 28).
const T5A_CREATES: u64 = 3;

/// T5b's dropped parent's children: its name-index vector, built by pushes
/// from capacity 1, sits at 2,048 entries (32,768 B) before the plan, and a
/// rollback that re-creates it allocates it at its 1,025 entries (20,480 B).
const T5B_CHILDREN: u64 = 1_025;

/// T5b's other children of the root: enough that the sections table, at
/// 1,826 rows (3,584 slots once compacted), has room for the plan's 1,026
/// removals.
const T5B_OTHER_CHILDREN: u64 = 800;

/// T5b's dropped parent.
const T5B_PARENT: u64 = 50;

/// T5b's base: [`T5B_PARENT`] under the root holding [`T5B_CHILDREN`]
/// children (ids from 10,000), beside [`T5B_OTHER_CHILDREN`] childless
/// children of the root.
fn dropped_vector_state() -> ObjectState {
    let mut state = empty_state();
    let mut root = vec![T5B_PARENT];
    state.sections.insert(T5B_PARENT, section(T5B_PARENT, 0, 0));
    for (order, id) in (FIRST_CHILD..FIRST_CHILD + T5B_OTHER_CHILDREN).enumerate() {
        state.sections.insert(id, section(id, 0, order as u32 + 1));
        root.push(id);
    }
    state.children.insert(0, root);
    let kids: Vec<u64> = (10_000..10_000 + T5B_CHILDREN).collect();
    for (order, id) in kids.iter().enumerate() {
        state
            .sections
            .insert(*id, section(*id, T5B_PARENT, order as u32));
    }
    state.children.insert(T5B_PARENT, kids);
    state
}

/// The last need the call priced before its `wait`-th wait that carries no
/// repair transient: the need it suspended for — a refused plan's exact
/// count, counted on to the plan's end, or a refused CR2's. A refused plan
/// on the cached route prices one more need after its own: the resumed
/// hit's first RP0 on the entry as put back, the wait's floor
/// ([`resumed_rp0_floor`]), which this skips.
fn suspended_need(log: &ChargeLog, wait: usize) -> ObjectStateNeed {
    let (needs_at_hold, _) = log.at_hold[wait];
    *log.needs[..needs_at_hold]
        .iter()
        .rev()
        .find(|need| !is_repair_need(need))
        .unwrap_or_else(|| panic!("the call priced a need before wait #{wait}; {log:?}"))
}

/// The resumed hit's first RP0 the cached route's refused plan priced after
/// its own need, as the floor of its `wait`-th wait (`storage::layer`, the
/// `Overflow` arm): the last need before that wait, when it carries a repair
/// transient and follows the plan's need. `None` on the WATCH path and for a
/// refused CR2.
fn resumed_rp0_floor(log: &ChargeLog, wait: usize) -> Option<ObjectStateNeed> {
    let (needs_at_hold, _) = log.at_hold[wait];
    match log.needs[..needs_at_hold] {
        [.., plan, floor] if is_plan_need(&plan) && is_repair_need(&floor) => Some(floor),
        _ => None,
    }
}

/// One name-index entry, `(name hash, section id)`, in bytes
/// (`domain::plan_index`).
const NAME_ENTRY_BYTES: u64 = size_of::<(u32, u64)>() as u64;

/// The resumed attempt's CR1 after the call's `wait`-th wait: the first
/// need it priced that carries no repair transient (on the cached route only
/// its hit's repair charge points, RP0, RP1 and RP2, come before it).
fn resumed_cr1(log: &ChargeLog, wait: usize) -> ObjectStateNeed {
    let (needs_at_hold, _) = log.at_hold[wait];
    let cr1 = *log.needs[needs_at_hold..]
        .iter()
        .find(|need| {
            !matches!(need, ObjectStateNeed::Cached { repair, .. }
                if *repair != RepairTransient::default())
        })
        .unwrap_or_else(|| panic!("the resumed attempt prices its CR1; {log:?}"));
    assert!(
        cr1.rewrite() == RewriteCount::default(),
        "the resumed attempt's first need past its repair is its CR1; {cr1:?}"
    );
    cr1
}

/// T5b: an `Overflow` whose entry as put back weighs less than before the
/// plan. The plan deletes [`T5B_PARENT`], so its 1,025 children go and the
/// parent's name-index key is dropped with its vector at capacity 2,048; the
/// rollback re-creates the vector at its exact length, 1,025 entries, below
/// its power-of-two capacity (20,480 B against 32,768 B). The one wait is
/// for CR1 of the entry as put back plus the plan's exact need — that CR1
/// priced by the test from the entry its probe read at the wait
/// ([`independent_cr1`]), and the resumed attempt's logged CR1 pinned to it —
/// below the same pricing of the pre-plan entry, and one recharge. Hard
/// preconditions: the index
/// weight as put back is below the pre-plan one, and the tables are ample
/// for the plan's removals (1,026 sections, a children list, a name-index
/// key). Discriminates the earlier pricing at the larger of the two CR1s.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn refused_plan_growth_waits_on_a_put_back_entry_lighter_than_before_the_plan_ug2() {
    let fx = fixture("t5b-dropped-vector", UpdateLayerBackend::Lua).await;
    let oid = create_state(&fx.ls, &fx.session_id, "t5b-base", &dropped_vector_state()).await;
    warm_on(&fx, oid, 5_000).await;
    let before = held_snapshot(&fx.ls, oid);
    let before_seq = fx.seq_of(oid).await;
    let at_wait = Arc::new(Mutex::new(Vec::new()));
    let probe: HoldProbe = {
        let (ls, at_wait) = (Arc::clone(&fx.ls), Arc::clone(&at_wait));
        Box::new(move |_target| {
            at_wait
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .push(held_snapshot(&ls, oid));
            async {}.boxed()
        })
    };
    let (charge, log) = ScriptedCharge::new(u64::MAX);
    let mut charge = charge.with_probe(probe).refusing_first_plan_growth();
    let delta = OwnedDelta {
        struct_commands: vec![StructCommand::DeleteSection(CommandDeleteSection {
            parent_id: 0,
            section_id: T5B_PARENT,
        })],
        ..Default::default()
    };
    let inserts = PlannedInserts::of(Some(&delta));
    let seq = tokio::time::timeout(STEP_BUDGET, fx.update_on(oid, delta, &mut charge))
        .await
        .expect("the call must finish")
        .unwrap_or_else(|e| panic!("the call commits; got {e:?}"));

    let log = lock(&log);
    assert_eq!(log.holds.len(), 1, "one wait; {log:?}");
    let at_wait = at_wait.lock().unwrap_or_else(PoisonError::into_inner);
    let put_back = &at_wait[0];
    assert_eq!(
        put_back.state, before.state,
        "the entry went back with its pre-plan contents"
    );
    assert!(
        put_back.index_bytes < before.index_bytes,
        "fixture: the rollback re-created the dropped vector at its exact length (index weight \
         {} as put back, {} before the plan)",
        put_back.index_bytes,
        before.index_bytes
    );
    assert_tables_ample(
        put_back,
        TableCalls {
            sections: T5B_CHILDREN as usize + 1,
            children: 1,
            name_index: 1,
            ..TableCalls::default()
        },
        "t5b",
    );
    let (needs_at_hold, _) = log.at_hold[0];
    let suspended = suspended_need(&log, 0);
    let needed = suspended.rewrite();
    assert!(
        needed != RewriteCount::default(),
        "the suspension carries the plan's need"
    );
    // The wait is CR1 of the entry as put back, priced by the test from that
    // entry alone, plus the plan's exact need — above the resumed hit's first
    // RP0 on that entry, the wait's floor, which the delete's 1,026 rows
    // outweigh; then the implementation's resumed CR1 is pinned to that
    // pricing.
    let cr1 = independent_cr1(put_back, inserts);
    let floor = resumed_rp0_floor(&log, 0).expect("the cached route prices the wait's floor");
    assert!(
        object_state_charge_for_test(&floor)
            < object_state_charge_for_test(&cr1.with_rewrite(needed)),
        "fixture: the plan's need is above the resumed RP0 floor ({floor:?}); {log:?}"
    );
    assert_eq!(
        log.holds[0],
        object_state_charge_for_test(&cr1.with_rewrite(needed)),
        "the wait is CR1 of the entry as put back ({cr1:?}, priced from the entry) plus the \
         plan's exact need; {log:?}"
    );
    assert_eq!(
        resumed_cr1(&log, 0),
        cr1,
        "the resumed attempt's CR1 is the entry as put back's"
    );
    let pre_plan_form =
        object_state_charge_for_test(&independent_cr1(&before, inserts).with_rewrite(needed));
    assert!(
        log.holds[0] < pre_plan_form,
        "the wait ({}) is below the pre-plan CR1's form ({pre_plan_form})",
        log.holds[0]
    );
    let resumed_asks = log.needs[needs_at_hold..]
        .iter()
        .filter(|need| is_plan_need(need))
        .count();
    assert_eq!(
        resumed_asks, 1,
        "the resumed plan never asked: only its exact trim prices a rewrite; {log:?}"
    );
    assert_eq!(seq, before_seq + 1, "one commit");
    assert_eq!(fx.recharges("structural"), 1, "one recharge");
}

/// T5h's base: root children with names of [`LONG_NAME_BYTES`], so the
/// sections table, at 122 rows once warmed (224 slots once compacted), has
/// room for [`T5H_CREATES`] even after the tombstones a rollback of them
/// leaves (a 62-row table, 112 slots, was left with 22 of room).
const T5H_ROOT_CHILDREN: u64 = 120;

/// T5h's long-named creates under [`KEPT_PARENT`]: its name-index vector
/// grows from capacity 1 to 64, and their names (about 1 KiB each), which
/// neither CR1 nor the plan's rewrite charges (they are the delta's own
/// elements, charged per element by the upload), put CR2 above what the
/// plan held — above CR1's growth bound and the plan's 5/4 headroom, which
/// ten creates did not outweigh.
const T5H_CREATES: u64 = 40;

/// T5h: a refused CR2. Long-named creates under [`KEPT_PARENT`] grow its
/// name-index vector past its capacity and put CR2 — the planned entry with
/// its new names, its possible rollback rebuilds and the plan's exact rewrite
/// — above everything the attempt held before it: a calibration run of the
/// same edit on an identical object gives CR2's target and what was held
/// before it, and the measured call's room is the latter. The one wait is
/// for CR2's own target, and one recharge. Hard preconditions: CR2's target
/// is above what was held by at least the parent vector's growth (the
/// rollback keeps that growth, so the resumed attempt's CR1 holds it), and
/// the tables are ample. Discriminates a wait for CR2's target plus the
/// shift in CR1.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn refused_cr2_waits_for_its_own_target_ug2() {
    let fx = fixture("t5h-refused-cr2", UpdateLayerBackend::Lua).await;
    let state = kept_parent_state(T5H_ROOT_CHILDREN, |id| {
        format!("{id:0>width$}", width = LONG_NAME_BYTES)
    });
    let delta = || {
        let mut delta = creates_under(KEPT_PARENT, 20_000, T5H_CREATES, "");
        for command in &mut delta.struct_commands {
            if let StructCommand::CreateSection(create) = command {
                create.section_name =
                    format!("{:0>width$}", create.section_id, width = LONG_NAME_BYTES);
            }
        }
        delta
    };
    // Calibration: the same edit on an identical, identically warmed object,
    // with every growth granted.
    let calibration = create_state(&fx.ls, &fx.session_id, "t5h-calibration", &state).await;
    warm_on(&fx, calibration, 5_000).await;
    let (free, calibration_log) = ScriptedCharge::new(u64::MAX);
    let mut free = free.without_repair_transient();
    fx.update_on(calibration, delta(), &mut free)
        .await
        .expect("the calibration run commits");
    let tries: Vec<u64> = lock(&calibration_log)
        .tries
        .iter()
        .map(|(t, _)| *t)
        .collect();
    let (&cr2, before_cr2) = tries.split_last().expect("the calibration charged");
    let held = before_cr2.iter().copied().max().expect("CR1 came first");

    let oid = create_state(&fx.ls, &fx.session_id, "t5h-measured", &state).await;
    warm_on(&fx, oid, 5_000).await;
    let before = held_snapshot(&fx.ls, oid);
    let before_seq = fx.seq_of(oid).await;
    let at_wait = Arc::new(Mutex::new(Vec::new()));
    let probe: HoldProbe = {
        let (ls, at_wait) = (Arc::clone(&fx.ls), Arc::clone(&at_wait));
        Box::new(move |_target| {
            at_wait
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .push(held_snapshot(&ls, oid));
            async {}.boxed()
        })
    };
    let (charge, log) = ScriptedCharge::new(held);
    let mut charge = charge.with_probe(probe).without_repair_transient();
    let seq = tokio::time::timeout(STEP_BUDGET, fx.update_on(oid, delta(), &mut charge))
        .await
        .expect("the call must finish")
        .unwrap_or_else(|e| panic!("the call commits; got {e:?}"));

    let log = lock(&log);
    let at_wait = at_wait.lock().unwrap_or_else(PoisonError::into_inner);
    assert_eq!(log.holds.len(), 1, "one wait; {log:?}");
    let put_back = &at_wait[0];
    assert_eq!(
        (put_back.seq, &put_back.state),
        (before.seq, &before.state),
        "while the call waits on its refused CR2 the cache holds the pre-plan cut (T9)"
    );
    let vector_growth = put_back.index_bytes.saturating_sub(before.index_bytes);
    assert!(
        vector_growth > 0,
        "fixture: the creates grew the kept parent's name-index vector, which the rollback keeps"
    );
    assert!(
        cr2 > held && cr2 - held >= vector_growth,
        "fixture: CR2's target ({cr2}) is above what the attempt held ({held}) by at least the \
         parent vector's growth ({vector_growth} B)"
    );
    assert_tables_ample(
        put_back,
        TableCalls {
            sections: T5H_CREATES as usize,
            ..TableCalls::default()
        },
        "t5h",
    );
    let refused: Vec<u64> = log
        .tries
        .iter()
        .filter(|(_, granted)| !granted)
        .map(|(target, _)| *target)
        .collect();
    assert_eq!(
        refused,
        vec![cr2],
        "the one refused growth is CR2, at the calibrated target; {log:?}"
    );
    assert_eq!(
        log.holds[0], cr2,
        "the wait is for CR2's own target; {log:?}"
    );
    let (needs_at_hold, _) = log.at_hold[0];
    let resumed_asks = log.needs[needs_at_hold..]
        .iter()
        .filter(|need| is_plan_need(need))
        .count();
    assert_eq!(
        resumed_asks, 1,
        "the resumed plan never asked past CR2's target: only its exact trim prices a \
         rewrite; {log:?}"
    );
    assert_eq!(seq, before_seq + 1, "one commit");
    assert_eq!(fx.recharges("structural"), 1, "one recharge");
}

/// One contended call (T5c–T5g): its charge log, the cached entry before it,
/// the entry as put back at each of its waits (read by the charge's
/// [`HoldProbe`], where the call holds nothing), and the delta's inserts.
struct ContendedResume {
    log: ChargeLog,
    before: HeldSnapshot,
    at_wait: Vec<HeldSnapshot>,
    inserts: PlannedInserts,
}

/// A [`HoldProbe`] that reads `oid`'s entry on `ls` at each wait into
/// `at_wait`.
fn snapshot_at_each_wait(
    ls: &Arc<LayerStore>,
    oid: u64,
    at_wait: &Arc<Mutex<Vec<HeldSnapshot>>>,
) -> HoldProbe {
    let (ls, at_wait) = (Arc::clone(ls), Arc::clone(at_wait));
    Box::new(move |_target| {
        at_wait
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(held_snapshot(&ls, oid));
        async {}.boxed()
    })
}

/// The checks of a contended resume (T5c–T5g), given the call and its
/// `recharges_total{path="structural"}` ticks:
///
/// - one or two waits, each a recharge;
/// - at each wait the entry as put back holds the pre-plan contents, and the
///   wait covers the CR1 the test prices on that entry on its own
///   ([`independent_cr1`]) — so the resumed attempt's first charge point
///   takes nothing past what it waited for; then the CR1 the resumed attempt
///   logged is pinned to that pricing;
/// - every growth taken was a granted ask: a refused growth is followed at
///   once by the call's wait, so no attempt went on past a refusal.
///
/// The waits' targets need not rise: a second wait can be below the first
/// when the second rollback rebuilt a table the first left without room,
/// which takes the CR1 growth term for that table off the entry as put back
/// (measured: about a quarter of T5g's two-wait runs). Each is still what its
/// resumed attempt needs.
fn assert_contended_resume(resume: &ContendedResume, recharges: u64, context: &str) {
    let log = &resume.log;
    assert!(
        (1..=2).contains(&log.holds.len()),
        "{context}: one or two waits; {log:?}"
    );
    assert_eq!(
        recharges,
        log.holds.len() as u64,
        "{context}: each wait a recharge"
    );
    assert_eq!(
        resume.at_wait.len(),
        log.holds.len(),
        "{context}: the probe read the entry at every wait"
    );
    for (wait, (&target, put_back)) in log.holds.iter().zip(&resume.at_wait).enumerate() {
        assert_eq!(
            (put_back.seq, &put_back.state),
            (resume.before.seq, &resume.before.state),
            "{context}: at wait #{wait} the entry is the pre-plan cut"
        );
        let cr1 = independent_cr1(put_back, resume.inserts);
        let priced = object_state_charge_for_test(&cr1);
        assert!(
            priced <= target,
            "{context}: wait #{wait} ({target}) covers CR1 of the entry as put back ({priced}, \
             {cr1:?}, priced from the entry); {log:?}"
        );
        assert_eq!(
            resumed_cr1(log, wait),
            cr1,
            "{context}: the CR1 the attempt resumed after wait #{wait} with is the entry as put \
             back's; {log:?}"
        );
    }
    for (i, _) in log
        .tries
        .iter()
        .enumerate()
        .filter(|(_, (_, granted))| !granted)
    {
        assert!(
            log.at_hold.iter().any(|&(_, tries)| tries == i + 1),
            "{context}: the refused growth #{i} was followed at once by the call's wait; {log:?}"
        );
    }
}

/// One call of `delta` on `oid` whose first plan growth is refused and whose
/// charge, after its wait, is contended: a competitor takes whatever the call
/// gives back. One commit, and the checks of [`assert_contended_resume`].
/// Returns the call, for the caller's check that its named layout occurred.
async fn assert_at_most_two_recharges(
    fx: &Fixture,
    oid: u64,
    delta: OwnedDelta,
    context: &str,
) -> ContendedResume {
    let before_seq = fx.seq_of(oid).await;
    let before_recharges = fx.recharges("structural");
    let before = held_snapshot(&fx.ls, oid);
    let inserts = PlannedInserts::of(Some(&delta));
    let at_wait = Arc::new(Mutex::new(Vec::new()));
    let (charge, log) = ScriptedCharge::new(u64::MAX);
    let mut charge = charge
        .with_probe(snapshot_at_each_wait(&fx.ls, oid, &at_wait))
        .refusing_first_plan_growth()
        .contended_after_hold(0)
        .without_repair_transient();
    let seq = tokio::time::timeout(STEP_BUDGET, fx.update_on(oid, delta, &mut charge))
        .await
        .unwrap_or_else(|_| panic!("{context}: the call must finish"))
        .unwrap_or_else(|e| panic!("{context}: the call commits; got {e:?}"));
    assert_eq!(seq, before_seq + 1, "{context}: one commit");
    let resume = ContendedResume {
        log: std::mem::take(&mut *lock(&log)),
        before,
        at_wait: std::mem::take(&mut *at_wait.lock().unwrap_or_else(PoisonError::into_inner)),
        inserts,
    };
    assert_contended_resume(
        &resume,
        fx.recharges("structural") - before_recharges,
        context,
    );
    resume
}

/// A cached object of [`kept_parent_state`] with `children` root children,
/// warmed.
async fn kept_parent_object(fx: &Fixture, layer: &str, children: u64) -> u64 {
    let oid = create_state(
        &fx.ls,
        &fx.session_id,
        layer,
        &kept_parent_state(children, |id| format!("child{id}")),
    )
    .await;
    warm_on(fx, oid, 5_000).await;
    oid
}

/// T5c: the plan grows [`KEPT_PARENT`]'s name-index vector with creates, then
/// deletes the parent, dropping that grown vector with its key; the refused
/// plan's rollback re-creates the parent's vector at its exact length. At
/// most two recharges, never an under-charge. The layout is hard-checked:
/// the refused plan projected the grown vector's rebuild, and the entry as
/// put back has the pre-plan index weight.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn dropped_grown_name_vector_resumes_within_two_recharges_ug2() {
    let fx = fixture("t5c-dropped-grown", UpdateLayerBackend::Lua).await;
    let oid = kept_parent_object(&fx, "t5c-base", BASE_CHILDREN).await;
    let mut delta = creates_under(KEPT_PARENT, 20_000, 20, "grown");
    delta
        .struct_commands
        .push(StructCommand::DeleteSection(CommandDeleteSection {
            parent_id: 0,
            section_id: KEPT_PARENT,
        }));
    let resume = assert_at_most_two_recharges(&fx, oid, delta, "t5c").await;
    // The layout happened in the refused plan: the creates grew the parent's
    // vector to their 20 entries plus the kept child's, and the delete then
    // dropped it with its key, so the refused plan's need projects that
    // grown vector's rebuild; the rollback re-created the vector at its
    // pre-plan length, so the entry as put back weighs what it did before.
    let grown_vector = allocation_bytes(21 * NAME_ENTRY_BYTES);
    let refused = suspended_need(&resume.log, 0).rewrite();
    assert!(
        refused.rollback_rebuild_bytes >= grown_vector,
        "t5c: fixture: the refused plan dropped the grown vector ({grown_vector} B at least); \
         it projected {refused:?}"
    );
    assert_eq!(
        resume.at_wait[0].index_bytes, resume.before.index_bytes,
        "t5c: fixture: the rollback re-created the dropped vector at its pre-plan length"
    );
}

/// T5d: move away, move back, grow — [`KEPT_PARENT`]'s only child moved to
/// another parent and back (the parent's key dropped and re-created), then
/// 100 creates under the parent (its vector grown to capacity 128, which the
/// rollback keeps, so the resumed plan's move-away drops a vector larger
/// than the refused plan's did). At most two recharges, never an
/// under-charge. The layout is hard-checked: the refused plan projected a
/// dropped vector (the move away's), and the entry as put back weighs more
/// than before the plan (the grown vector kept).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn move_away_back_and_grow_resumes_within_two_recharges_ug2() {
    let fx = fixture("t5d-move-away-back", UpdateLayerBackend::Lua).await;
    let oid = kept_parent_object(&fx, "t5d-base", 110).await;
    let other = FIRST_CHILD + 1;
    let mut struct_commands = vec![
        StructCommand::MoveSection(CommandMoveSection {
            old_parent_id: KEPT_PARENT,
            new_parent_id: other,
            section_id: KEPT_CHILD,
            new_name: format!("s{KEPT_CHILD}"),
        }),
        StructCommand::MoveSection(CommandMoveSection {
            old_parent_id: other,
            new_parent_id: KEPT_PARENT,
            section_id: KEPT_CHILD,
            new_name: format!("s{KEPT_CHILD}"),
        }),
    ];
    struct_commands.extend(creates_under(KEPT_PARENT, 20_000, 100, "grow").struct_commands);
    let delta = OwnedDelta {
        struct_commands,
        ..Default::default()
    };
    let resume = assert_at_most_two_recharges(&fx, oid, delta, "t5d").await;
    // The layout happened in the refused plan: the move away dropped the
    // parent's key (the only key the delta drops), so its need projects a
    // dropped vector; and the creates after the move back grew the parent's
    // vector, which the rollback keeps, so the entry as put back weighs more
    // than before the plan.
    let refused = suspended_need(&resume.log, 0).rewrite();
    assert!(
        refused.rollback_rebuild_bytes > 0,
        "t5d: fixture: the move away dropped the parent's key; the refused plan projected \
         {refused:?}"
    );
    assert!(
        resume.at_wait[0].index_bytes > resume.before.index_bytes,
        "t5d: fixture: the rollback kept the grown vector (index weight {} as put back, {} \
         before the plan)",
        resume.at_wait[0].index_bytes,
        resume.before.index_bytes
    );
}

/// T5e: delete then create the same id under the same parent with the same
/// name ([`KEPT_CHILD`] under [`KEPT_PARENT`]: the parent's key dropped with
/// its capacity-1 vector, re-created), then grow the parent with creates. At
/// most two recharges, never an under-charge. The layout is hard-checked:
/// the refused plan projected its removals' rebuilds, and the entry as put
/// back weighs more than before the plan (the re-created vector kept grown).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn delete_then_create_the_same_id_and_grow_resumes_within_two_recharges_ug2() {
    let fx = fixture("t5e-delete-create", UpdateLayerBackend::Lua).await;
    let oid = kept_parent_object(&fx, "t5e-base", BASE_CHILDREN).await;
    let mut struct_commands = vec![
        StructCommand::DeleteSection(CommandDeleteSection {
            parent_id: KEPT_PARENT,
            section_id: KEPT_CHILD,
        }),
        StructCommand::CreateSection(CommandCreateSection {
            parent_id: KEPT_PARENT,
            section_id: KEPT_CHILD,
            section_name: format!("s{KEPT_CHILD}"),
            section_type: 1,
        }),
    ];
    struct_commands.extend(creates_under(KEPT_PARENT, 20_000, 6, "grow").struct_commands);
    let delta = OwnedDelta {
        struct_commands,
        ..Default::default()
    };
    let resume = assert_at_most_two_recharges(&fx, oid, delta, "t5e").await;
    // The layout happened in the refused plan: the delete removed pre-plan
    // keys (the child's row, and the parent's name-index key with its
    // one-entry vector) — the only removals the delta makes — so its need
    // projects their rollback rebuilds; and the creates grew the re-created
    // vector, which the rollback keeps (a vector the plan only dropped comes
    // back at its pre-plan length), so the entry as put back weighs more
    // than before the plan.
    let refused = suspended_need(&resume.log, 0).rewrite();
    assert!(
        refused.rollback_rebuild_bytes > 0,
        "t5e: fixture: the delete removed pre-plan keys; the refused plan projected {refused:?}"
    );
    assert!(
        resume.at_wait[0].index_bytes > resume.before.index_bytes,
        "t5e: fixture: the rollback kept the re-created vector grown (index weight {} as put \
         back, {} before the plan)",
        resume.at_wait[0].index_bytes,
        resume.before.index_bytes
    );
}

/// R-T5's base: root children, so the sections table holds 56 rows once
/// warmed, at capacity 56: no room, so a delete projects its rollback's
/// rebuild of the table.
const R_T5_ROOT_CHILDREN: u64 = 55;

/// A cached object of [`R_T5_ROOT_CHILDREN`] children of the root, warmed,
/// its sections table full; and the bytes of that table's rebuild at its
/// pre-plan length.
async fn full_sections_object(fx: &Fixture, layer: &str) -> (u64, u64) {
    let mut state = empty_state();
    let ids: Vec<u64> = (FIRST_CHILD..FIRST_CHILD + R_T5_ROOT_CHILDREN).collect();
    for (order, id) in ids.iter().enumerate() {
        state.sections.insert(*id, section(*id, 0, order as u32));
    }
    state.children.insert(0, ids);
    let oid = create_state(&fx.ls, &fx.session_id, layer, &state).await;
    warm_on(fx, oid, 5_000).await;
    let held = held_snapshot(&fx.ls, oid);
    assert_eq!(
        held.rooms.sections, 0,
        "fixture: the sections table is full"
    );
    let rebuild = live_edit::domain::alloc_model::table_bytes(
        held.state.sections.len(),
        std::mem::size_of::<(u64, Section)>() as u64,
    );
    (oid, rebuild)
}

/// The base's last child of the root, a leaf: what [`delete_leaf`] deletes.
const R_T5_LEAF: u64 = FIRST_CHILD + R_T5_ROOT_CHILDREN - 1;

/// The delete of the base's last child of the root, a leaf.
fn delete_leaf() -> OwnedDelta {
    OwnedDelta {
        struct_commands: vec![StructCommand::DeleteSection(CommandDeleteSection {
            parent_id: 0,
            section_id: R_T5_LEAF,
        })],
        ..Default::default()
    }
}

/// The room [`delete_leaf`]'s plan leaves in the sections table of the entry
/// `held` read: its length less its capacity once the leaf's row is erased,
/// as `PlanJournal::rollback_rebuild_bytes` reads it. The erase runs on a
/// clone, which keeps the table's buckets, control bytes and hasher, so it
/// marks the same slot the same way: empty (the room comes back) or a
/// tombstone (it does not), as hashbrown decides from the slot's neighbours.
fn room_after_deleting_the_leaf(held: &HeldSnapshot) -> usize {
    let mut sections = held.state.sections.clone();
    sections
        .remove(&R_T5_LEAF)
        .expect("fixture: the leaf is a section of the entry");
    sections.capacity() - sections.len()
}

/// The most fresh objects R-T5's STALE test warms before one's leaf delete
/// leaves its full sections table no room.
///
/// hashbrown's erase writes a tombstone (DELETED), leaving the table no
/// room, exactly when the run of non-EMPTY control bytes through the erased
/// slot — those before it plus it and those after it — is at least the
/// control group's width; otherwise it writes EMPTY and the room comes
/// back. So the rate depends on the architecture's group width:
///
/// - aarch64 (NEON, width 8), where this was measured: the room came back
///   in about one object in six, so P(no room) is about 5/6 per object;
/// - x86_64 (SSE2, width 16; presumably CI): runs of 16 are rarer. Review
///   estimated P(no room) at about 0.3 to 0.45 per object; a simulation of
///   hashbrown's real probing (the group window plus the triangular stride;
///   test review) gives about 0.70, so that estimate is conservative.
///
/// Each object's table is decoded under a hasher of its own, so objects are
/// independent. At the estimate's low end, 0.3 per object, all 64 objects give
/// their room back with probability 0.7^64, about 1e-10 (about 7e-10 even
/// at 0.28); on aarch64, about 1e-50. 64 is ample on both, and an attempt
/// costs a few milliseconds.
const R_T5_STALE_ATTEMPTS_BOUND: u64 = 64;

/// R-T5: a deleting delta on a full sections table, whose commit answers
/// `STALE` (another replica commits between the plan and its `EVALSHA`): CR2's
/// need holds the table's rollback rebuild, at its pre-plan length, in its
/// growth term, so the rollback after the `STALE` runs inside the held
/// charge; the retry repairs and commits.
///
/// The hard precondition, met within [`R_T5_STALE_ATTEMPTS_BOUND`] fresh
/// objects (never skipped): the leaf's erase leaves the table no room
/// ([`room_after_deleting_the_leaf`]), so the rollback must rebuild it — and
/// does, the rebuild counter shows. When the erase gives its slot back the
/// rollback re-inserts in place and CR2 rightly holds no rebuild, so such an
/// object proves nothing here and the next one is warmed.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn stale_commit_on_a_full_table_holds_its_rollback_rebuild_at_cr2_ug2() {
    let competing = Arc::new(Mutex::new(None::<(Arc<LayerStore>, String, String, u64)>));
    let hook: live_edit::storage::watched_txn::WatchWindowHook = {
        let competing = Arc::clone(&competing);
        Arc::new(move |_client: &fred::clients::Client| {
            let armed = competing
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .take();
            async move {
                if let Some((other, session, layer, oid)) = armed {
                    let poke = OwnedDelta {
                        set_fields: vec![oodf_ffi::SetField {
                            section_id: FIRST_CHILD,
                            key_name: "stale-poke".to_owned(),
                            value_or_ext_hash: vec![7; 8],
                            ext_value_size: 0,
                            ext_value_index: 0,
                            set_order: 1,
                        }],
                        ..Default::default()
                    };
                    let relay = relay_for(&Some(poke.clone()));
                    other
                        .update_layer(&session, &layer, oid, Some(poke), relay, "r-t5-other", 1)
                        .await
                        .expect("the competing commit lands");
                }
            }
            .boxed()
        })
    };
    let fx = fixture_with("r-t5-stale-fixture", UpdateLayerBackend::Lua, 4, |ls| {
        ls.with_structural_commit_pre_dispatch_hook_for_test(hook)
    })
    .await;
    let mut chosen = None;
    for attempt in 0..R_T5_STALE_ATTEMPTS_BOUND {
        let layer = format!("r-t5-stale-{attempt}");
        let (oid, rebuild) = full_sections_object(&fx, &layer).await;
        if room_after_deleting_the_leaf(&held_snapshot(&fx.ls, oid)) == 0 {
            chosen = Some((layer, oid, rebuild));
            break;
        }
    }
    let (layer, oid, rebuild) = chosen.unwrap_or_else(|| {
        panic!(
            "precondition never met: in {R_T5_STALE_ATTEMPTS_BOUND} fresh objects the leaf's \
             erase always gave its slot back"
        )
    });
    let before_seq = fx.seq_of(oid).await;
    let other = Arc::new(
        LayerStore::new(fx.tv.client().clone(), fx.tv.key_schema().clone(), 4096)
            .with_update_layer_backend(UpdateLayerBackend::Lua),
    );
    *competing.lock().unwrap_or_else(PoisonError::into_inner) =
        Some((other, fx.session_id.clone(), layer, oid));
    let cache = fx.ls.structural_cache();
    let (rollbacks, rebuilds) = (
        cache.rollbacks_for_test(),
        cache.rollback_rebuilds_for_test(),
    );

    let (mut charge, log) = ScriptedCharge::new(u64::MAX);
    tokio::time::timeout(STEP_BUDGET, fx.update_on(oid, delete_leaf(), &mut charge))
        .await
        .expect("the call must finish")
        .expect("the retry commits");

    assert!(
        competing
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .is_none(),
        "fixture: the competing commit ran"
    );
    assert_eq!(
        fx.seq_of(oid).await,
        before_seq + 2,
        "the competing commit and the call's"
    );
    assert!(
        cache.rollbacks_for_test() - rollbacks >= 1,
        "the STALE attempt's plan was rolled back"
    );
    assert!(
        cache.rollback_rebuilds_for_test() - rebuilds >= 1,
        "and its rollback rebuilt the table the erase left no room in"
    );
    let log = lock(&log);
    // CR2 of the first attempt: the first need priced on another entry
    // weight than CR1's — the planned entry's — after the plan's asks, which
    // price CR1's need with their rewrite. CR1 is the first need with no
    // repair transient: the hit's repair charge points (RP0, RP1, RP2) come first,
    // on the same entry weight.
    let cr1_held = match log.needs.iter().find(|need| {
        !matches!(need, ObjectStateNeed::Cached { repair, .. }
            if *repair != RepairTransient::default())
    }) {
        Some(ObjectStateNeed::Cached { held, .. }) => *held,
        other => panic!("the first need past the repair is the cached route's CR1; {other:?}"),
    };
    let cr2 = log
        .needs
        .iter()
        .find_map(|need| match *need {
            ObjectStateNeed::Cached { held, growth, .. } if held != cr1_held => Some(growth),
            _ => None,
        })
        .expect("the first attempt priced its CR2");
    assert!(
        cr2 >= rebuild,
        "CR2's growth term ({cr2} B) holds the sections table's rollback rebuild ({rebuild} B); \
         {log:?}"
    );
}

/// R-T5: a plan refused on a full sections table suspends for a need that
/// carries the table's rollback rebuild, which the delete projected before
/// it removed the row: the call waits for it, and the resumed plan fits.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn refused_plan_on_a_full_table_waits_for_its_rollback_rebuild_ug2() {
    let fx = fixture("r-t5-refused", UpdateLayerBackend::Lua).await;
    let (oid, rebuild) = full_sections_object(&fx, "r-t5-refused-base").await;
    let (charge, log) = ScriptedCharge::new(u64::MAX);
    let mut charge = charge
        .refusing_first_plan_growth()
        .without_repair_transient();
    tokio::time::timeout(STEP_BUDGET, fx.update_on(oid, delete_leaf(), &mut charge))
        .await
        .expect("the call must finish")
        .expect("the call commits after its wait");
    let log = lock(&log);
    assert_eq!(log.holds.len(), 1, "one wait; {log:?}");
    let suspended = suspended_need(&log, 0).rewrite();
    assert!(
        suspended.rollback_rebuild_bytes >= rebuild,
        "the wait carries the sections table's rollback rebuild ({rebuild} B); the plan's need \
         was {suspended:?}"
    );
    assert_eq!(fx.recharges("structural"), 1, "one recharge");
}

/// T5g's base: root children, so the sections table holds 55 rows once
/// warmed, at capacity 56 (64 buckets): one entry of room. 64 buckets, not
/// 32: a tombstone needs a run of non-EMPTY control bytes of at least the
/// group width through the erased slot (8 on aarch64, 16 on x86_64), and 64
/// buckets span at least four control groups at either width, which leaves
/// a wide margin ([`T5G_ATTEMPTS_BOUND`]).
const T5G_ROOT_CHILDREN: u64 = 54;

/// One T5g attempt: a fresh object of [`T5G_ROOT_CHILDREN`] children of the
/// root (ids from `first`), warmed, and a call that deletes one child and
/// creates one, its first plan refused, contended after its wait, held to
/// the checks of [`assert_at_most_two_recharges`]. Returns the call and the
/// rebuild bytes of the sections table at its pre-plan length.
async fn delete_and_insert_attempt(
    fx: &Fixture,
    attempt: u64,
    context: &str,
) -> (ContendedResume, u64) {
    let first = 100_000 + attempt * 1_000;
    let mut state = empty_state();
    let ids: Vec<u64> = (first..first + T5G_ROOT_CHILDREN).collect();
    for (order, id) in ids.iter().enumerate() {
        state.sections.insert(*id, section(*id, 0, order as u32));
    }
    state.children.insert(0, ids);
    let oid = create_state(&fx.ls, &fx.session_id, &format!("t5g-{attempt}"), &state).await;
    warm_on(fx, oid, first + 900).await;
    let before = held_snapshot(&fx.ls, oid);
    assert_eq!(
        before.rooms.sections, 1,
        "fixture: the sections table has one entry of room"
    );
    let rebuild = live_edit::domain::alloc_model::table_bytes(
        before.state.sections.len(),
        std::mem::size_of::<(u64, Section)>() as u64,
    );
    let delta = OwnedDelta {
        struct_commands: vec![
            StructCommand::DeleteSection(CommandDeleteSection {
                parent_id: 0,
                section_id: first + (attempt % T5G_ROOT_CHILDREN),
            }),
            StructCommand::CreateSection(CommandCreateSection {
                parent_id: 0,
                section_id: first + 950,
                section_name: format!("new{attempt}"),
                section_type: 1,
            }),
        ],
        ..Default::default()
    };
    let resume = assert_at_most_two_recharges(fx, oid, delta, context).await;
    (resume, rebuild)
}

/// Whether the call's resumed plan (after its first wait) projected one
/// table rebuild more than the refused plan's need, the first wait's.
fn resumed_need_carries_one_rebuild_more(log: &ChargeLog, rebuild: u64) -> bool {
    let Some(&(needs_at_hold, _)) = log.at_hold.first() else {
        return false;
    };
    let refused = suspended_need(log, 0).rewrite().rollback_rebuild_bytes;
    let resumed = log
        .needs
        .iter()
        .skip(needs_at_hold)
        .map(|need| need.rewrite().rollback_rebuild_bytes)
        .max()
        .unwrap_or(0);
    resumed == refused + rebuild
}

/// The most T5g attempts before its precondition must have been met.
///
/// Whether an attempt meets it depends on how hashbrown marks the plan's and
/// the rollback's erases (empty or tombstone), which the section ids and the
/// table's random hash keys decide. An erase leaves a tombstone only when the
/// run of non-EMPTY control bytes through the slot is at least the control
/// group's width: 8 on aarch64 (NEON), 16 on x86_64 (SSE2). At the load
/// limit, a 64-bucket table's `b`/8 EMPTY bytes, placed at random, leave
/// such a run through an erased slot with probability about 0.74 at width 8
/// and 0.39 at width 16 (`q^(w-1)·(1 + (w-1)(1-q))`, `q` = 7/8). hashbrown's
/// probing clusters the full slots, which lengthens runs: a simulation of
/// its real probing (the group window plus the triangular stride; test
/// review) gives about 0.85 at width 8 and 0.70 at width 16.
///
/// Measured 2026-10-10 on aarch64 with the 64-bucket base, every attempt
/// run: 17, 31 and 14 of 200 (62 of 600, 10.3 %; the lowest run 7 %). On
/// x86_64 the earlier base of 26 children (32 buckets), bounded at 200
/// attempts, passed in both jobs of pipeline 72956772 (0.206 s and
/// 0.173 s). A model of hashbrown's probing calibrated to the aarch64
/// measurements (32 buckets: 14.8 % measured, 14.4 % modelled; 64 buckets:
/// 10.3 % and 12.4 %) puts the x86_64 rate at about 18–20 %, above
/// aarch64's: the precondition is lost when the refused plan's rollback
/// itself rebuilds the table, and at width 8 it does so more often (about
/// 52 % of attempts against 30 %). Sized at a design rate of 2 %, below
/// every rate measured or modelled: a run misses all 1,000 with probability
/// 0.98^1000 ≈ 1.7e-9, negligible at either width. An attempt takes about
/// 35 ms in a debug build, so the bound costs at most about 35 s, under
/// CI's 120 s per-test kill; the expected run stops after about 10 attempts
/// on aarch64 and about 5 on x86_64.
const T5G_ATTEMPTS_BOUND: u64 = 1_000;

/// T5g: delete and insert in one table, end to end. On a sections table
/// with one entry of room, a call deletes one child of the root and creates
/// one; its first plan's growth is refused, and after its wait a competitor
/// takes whatever the call gives back. The refused plan's rollback can leave
/// the table with less room than before the plan (a tombstone), so the
/// resumed plan of the same delta projects a sections rebuild the refused
/// plan did not, asks past what the call waited for, and waits once more —
/// and the third attempt fits: one commit, two recharges, never three, each
/// wait covering the CR1 the test prices on the entry as put back, which the
/// resumed attempt's logged CR1 equals, and no growth taken past a refusal
/// ([`assert_contended_resume`]).
///
/// The hard precondition, met within [`T5G_ATTEMPTS_BOUND`] attempts on fresh
/// objects (never skipped): the resumed plan's need carries exactly one
/// table rebuild (the sections table at its pre-plan length) more than the
/// refused plan's. Every other table takes no insert call or removal, so is
/// ample. Every attempt, the precondition met or not, is held to the same
/// checks.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn delete_and_insert_at_one_entry_of_room_resumes_within_two_recharges_ug2() {
    let fx = fixture("t5g-delete-insert", UpdateLayerBackend::Lua).await;
    let mut met = None;
    for attempt in 0..T5G_ATTEMPTS_BOUND {
        let context = format!("t5g attempt {attempt}");
        let (resume, rebuild) = delete_and_insert_attempt(&fx, attempt, &context).await;
        let log = &resume.log;
        if resumed_need_carries_one_rebuild_more(log, rebuild) {
            assert_eq!(
                log.holds.len(),
                2,
                "{context}: the resumed plan's extra rebuild was asked for past the wait, \
                 refused by the competitor, and waited for once more; {log:?}"
            );
            met = Some(attempt);
            break;
        }
    }
    assert!(
        met.is_some(),
        "precondition never met: in {T5G_ATTEMPTS_BOUND} attempts no resumed plan projected a \
         rebuild the refused plan did not (expected about 10 % per attempt on aarch64, about \
         18 % on x86_64 by a model of hashbrown's probing)"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// Every window charged before it is issued; lists
// ─────────────────────────────────────────────────────────────────────────────

/// Inline value rows of the pre-issue fixture: one full window of value-row
/// commands (`FANOUT_WINDOW`).
const VALUE_ROWS: u64 = 2_141;

/// Each value row's inline value: large enough that a whole-row read of one
/// window receives megabytes, far past the bounded read's per-row bound.
const VALUE_ROW_BYTES: usize = 4 * 1024;

/// Children of the pre-issue fixture's long list.
const LONG_LIST: u64 = 5_000;

/// The pre-issue fixture: section 1 holding [`VALUE_ROWS`] inline fields of
/// [`VALUE_ROW_BYTES`], and section 2 with a [`LONG_LIST`]-entry children
/// list; plus an empty section 3, whose list is empty.
fn value_rows_and_long_list_state() -> ObjectState {
    let mut state = empty_state();
    for (order, id) in [1u64, 2, 3].into_iter().enumerate() {
        state.sections.insert(id, section(id, 0, order as u32));
    }
    state.children.insert(0, vec![1, 2, 3]);
    let kids: Vec<u64> = (10_000..10_000 + LONG_LIST).collect();
    for (order, id) in kids.iter().enumerate() {
        state.sections.insert(*id, section(*id, 2, order as u32));
    }
    state.children.insert(2, kids);
    for i in 0..VALUE_ROWS {
        let key = format!("v{i:05}");
        state.fields.insert(
            (1, key.clone()),
            live_edit::domain::types::Field {
                section_id: 1,
                key_name: key,
                value: live_edit::domain::types::FieldValue::Inline(vec![
                    (i % 251) as u8;
                    VALUE_ROW_BYTES
                ]),
                set_order: 1,
                authoring_set_order: 1,
                version: 1,
            },
        );
    }
    state
}

/// The message of [`PricingCharge`]'s event.
const PRICED_MSG: &str = "b26e test charge priced";

/// A charge that never refuses and logs, at DEBUG, the window bytes of every
/// need a charge point prices, so a captured log interleaves them with the
/// read's `fan-out window` events.
struct PricingCharge {
    held: u64,
}

#[async_trait::async_trait]
impl CommitCharge for PricingCharge {
    fn held(&self) -> u64 {
        self.held
    }

    fn target(&self, need: &ObjectStateNeed) -> u64 {
        if let ObjectStateNeed::Watch { snapshot, .. } = need {
            tracing::debug!(window_bytes = snapshot.window_bytes, "{}", PRICED_MSG);
        }
        object_state_charge_for_test(need)
    }

    fn rewrite_allowance(&self, need: &ObjectStateNeed) -> MutationAllowance {
        object_state_rewrite_allowance_for_test(self.held, need)
    }

    fn try_hold(&mut self, target: u64) -> bool {
        self.held = target;
        true
    }

    async fn hold(&mut self, target: u64) -> Result<(), ServiceError> {
        self.held = target;
        Ok(())
    }
}

/// On the WATCH path's charged read of a
/// layer with a full window of inline value rows and a long children list,
/// every window and round-1 page is preceded by a priced need whose window
/// bytes are at least the reply bytes that window then receives: the read
/// charges each window's bound before it is issued. A read that took the
/// value rows whole would receive megabytes per window against a pre-issue
/// bound of 144 B per row.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn watch_read_charges_every_window_before_it_is_issued_b26e() {
    use crate::grpc::log_capture::capture_debug;
    use live_edit::storage::commit_charge::CommitChargeCell;
    use live_edit::storage::test_support::read_object_snapshot_slim_charged_for_test;

    let fx = fixture("pre-issue", UpdateLayerBackend::Txn).await;
    let oid = create_state(
        &fx.ls,
        &fx.session_id,
        "pre-issue-values",
        &value_rows_and_long_list_state(),
    )
    .await;
    let oid_str = oid.to_string();
    let mut charge = PricingCharge { held: 0 };
    let (read, captured) = {
        let cell = CommitChargeCell::new(&mut charge);
        capture_debug(read_object_snapshot_slim_charged_for_test(
            fx.tv.client(),
            fx.tv.key_schema(),
            &oid_str,
            &cell,
        ))
        .await
    };
    let snapshot = read
        .expect("the charged read succeeds")
        .expect("a charge that never refuses never stops the read");
    assert_eq!(
        live_edit::domain::types::FieldRows::len(&snapshot.state.fields) as u64,
        VALUE_ROWS,
        "fixture: every value row is read"
    );

    let mut last_priced: Option<u64> = None;
    let mut windows = 0;
    let mut count_windows = 0;
    let mut received_total = 0u64;
    for event in &captured.events {
        let fields = &event["fields"];
        match fields["message"].as_str() {
            Some(PRICED_MSG) => last_priced = fields["window_bytes"].as_u64(),
            Some("fan-out window") => {
                let received = fields["reply_bytes"].as_u64().unwrap_or_default();
                // The count windows (the header and the indices' sizes) run
                // before WA1, under the attempt's fixed cost (WA0); every
                // later window and page after a priced need.
                let Some(priced) = last_priced else {
                    count_windows += 1;
                    continue;
                };
                assert!(
                    priced >= received,
                    "window {windows} received {received} B after a hold priced at {priced} \
                     window bytes; event: {event}"
                );
                windows += 1;
                received_total += received;
            }
            _ => {}
        }
    }
    assert!(
        count_windows <= 2,
        "only the count windows precede WA1 ({count_windows} windows did)"
    );
    assert!(
        windows >= 4,
        "fixture: the read ran its pages and several round-2/3 windows after WA1 ({windows})"
    );
    assert!(
        received_total < VALUE_ROWS * VALUE_ROW_BYTES as u64,
        "fixture: no inline value was transferred ({received_total} B received)"
    );
}

/// The WATCH read issues a ranged `LRANGE` only for a list
/// with entries, the long list one `LRANGE 0 n−1` after its `LLEN`. On an
/// unmarked object, whose lists it reads by section, the empty section's
/// list gets its `LLEN` (0) and no `LRANGE`; on a marked one, which reads
/// the lists `obj:chs` names, the empty list is not read at all. The same on
/// both of the read's transports (b.gvf): inside its window scripts, and as
/// plain commands (the test-support switch; the transport a read falls back
/// to). The `MONITOR` record counts each command whichever issued it — the
/// client, or a window script (the feed's `lua` client).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn watch_read_ranges_only_the_lists_with_entries_b26e() {
    use live_edit::storage::diff_read::WindowTransport;

    let fx = fixture("lists", UpdateLayerBackend::Txn).await;
    for marked in [true, false] {
        let oid = create_state(
            &fx.ls,
            &fx.session_id,
            &format!("lists-values-{marked}"),
            &value_rows_and_long_list_state(),
        )
        .await;
        if !marked {
            crate::storage::snapshot_read_atomic_test::unmark(&fx.tv, oid).await;
        }
        for transport in [WindowTransport::Script, WindowTransport::Plain] {
            fx.tv.client().set_watch_read_transport_for_test(transport);
            ranged_lists_case(&fx, oid, marked, &format!("{transport:?}")).await;
            fx.tv
                .client()
                .set_watch_read_transport_for_test(WindowTransport::Script);
        }
    }
}

async fn ranged_lists_case(fx: &Fixture, oid: u64, marked: bool, transport: &str) {
    use crate::storage::snapshot_read_support::{monitor_fence, test_valkey_addr, MonitorTap};
    use fred::interfaces::KeysInterface;
    use live_edit::storage::commit_charge::CommitChargeCell;
    use live_edit::storage::test_support::read_object_snapshot_slim_charged_for_test;

    let oid_str = oid.to_string();
    let mut tap = MonitorTap::start(&test_valkey_addr()).await;
    let mut charge = PricingCharge { held: 0 };
    {
        let cell = CommitChargeCell::new(&mut charge);
        read_object_snapshot_slim_charged_for_test(
            fx.tv.client(),
            fx.tv.key_schema(),
            &oid_str,
            &cell,
        )
        .await
        .expect("the charged read succeeds")
        .expect("never stopped");
    }
    let fence = monitor_fence(fx.tv.prefix());
    let _: i64 = fx
        .tv
        .client()
        .pool
        .exists(&fence)
        .await
        .expect("EXISTS the fence");
    let seen = tap.drain_until(&fence).await;
    // Whichever issued it: the client, or a window script.
    let on = |name: &str, key: &str| -> Vec<Vec<String>> {
        seen.iter()
            .filter(|c| c.name() == name && c.args.get(1).map(String::as_str) == Some(key))
            .map(|c| c.args.clone())
            .collect()
    };
    let list = |parent: u64| {
        fx.tv
            .key_schema()
            .obj_children(&oid_str, &parent.to_string())
    };
    let (empty, long) = (list(3), list(2));
    assert_eq!(
        on("LLEN", &empty).len(),
        usize::from(!marked),
        "{transport}, marked {marked}: the empty list's length is read only by section"
    );
    assert!(
        on("LRANGE", &empty).is_empty(),
        "{transport}, marked {marked}: an empty list issues no LRANGE"
    );
    assert_eq!(
        on("LLEN", &long).len(),
        1,
        "{transport}, marked {marked}: the long list's length is read"
    );
    let ranges: Vec<Vec<String>> = on("LRANGE", &long)
        .into_iter()
        .map(|args| args[2..].to_vec())
        .collect();
    assert_eq!(
        ranges,
        vec![vec!["0".to_owned(), (LONG_LIST - 1).to_string()]],
        "{transport}, marked {marked}: the long list is read by one LRANGE 0 n−1"
    );
}

/// The children of the grown-list fixture's section 2.
const GROWN_LIST_KIDS: u64 = 1_000;

/// The grown-list fixture: section 2 under the root, with
/// [`GROWN_LIST_KIDS`] children.
fn one_long_list_state() -> ObjectState {
    let mut state = empty_state();
    state.sections.insert(2, section(2, 0, 0));
    state.children.insert(0, vec![2]);
    let kids: Vec<u64> = (10_000..10_000 + GROWN_LIST_KIDS).collect();
    for (order, id) in kids.iter().enumerate() {
        state.sections.insert(*id, section(*id, 2, order as u32));
    }
    state.children.insert(2, kids);
    state
}

/// One `CreateSection` of `section_id` under `parent_id`.
fn child_delta(parent_id: u64, section_id: u64) -> OwnedDelta {
    OwnedDelta {
        struct_commands: vec![StructCommand::CreateSection(CommandCreateSection {
            parent_id,
            section_id,
            section_name: format!("new{section_id}"),
            section_type: 1,
        })],
        ..Default::default()
    }
}

/// A list that grows between the WATCH read's `LLEN` and
/// its `LRANGE` is read at the length its `LLEN` returned, and the read is
/// rejected. The read's two round-3 `LRANGE`s (the root's list and section
/// 2's, after their `LLEN`s on this marked object) are held back by the
/// fault seam while a competitor's commit adds a child to section 2 (an
/// `RPUSH`ed list, and a `seq` that moved). The held `LRANGE` asks `0 n−1`
/// — n ids, not the n + 1 the list now holds — and the read's `seq` proof
/// rejects it: one `watch_read_moved` conflict; the retry ranges the grown
/// list (`0 n`) and commits, so both children land.
///
/// Needs `test-fault-injection` (the read's fault seam).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn watch_read_of_a_list_grown_after_its_length_reads_that_length_and_retries_b26e() {
    use crate::storage::snapshot_read_support::{monitor_fence, test_valkey_addr, MonitorTap};
    use fred::interfaces::{KeysInterface, ListInterface};
    use live_edit::storage::diff_read::{
        ObjectSnapshotReadFaultInjector, ObjectSnapshotReadFaultTarget, ObjectSnapshotReader,
    };

    /// How long each round-3 `LRANGE` is held back: past the competitor's
    /// whole commit.
    const HELD: Duration = Duration::from_secs(4);
    let fx = fixture("grown-list", UpdateLayerBackend::Txn).await;
    let oid = create_state(
        &fx.ls,
        &fx.session_id,
        "grown-list-layer",
        &one_long_list_state(),
    )
    .await;
    let oid_str = oid.to_string();
    let before = fx.seq_of(oid).await;
    let list = fx.tv.key_schema().obj_children(&oid_str, "2");
    let injector = Arc::new(ObjectSnapshotReadFaultInjector::new());
    // The marked read's children commands: its two LLENs (round 2), then
    // its two LRANGEs (round 3), held back.
    for nth in [2, 3] {
        injector.push_delay_at(
            ObjectSnapshotReadFaultTarget::nth_of_family(
                ObjectSnapshotReader::UnfencedFanout,
                "children",
                nth,
            ),
            HELD,
        );
    }
    fx.tv
        .client()
        .install_object_snapshot_read_fault_injector_for_test(Arc::clone(&injector));
    let mut tap = MonitorTap::start(&test_valkey_addr()).await;

    let started = tokio::time::Instant::now();
    let mut held_charge = NoCommitCharge;
    let held = fx.update_on(oid, child_delta(2, 90_001), &mut held_charge);
    let competitor = async {
        let deadline = started + STEP_BUDGET;
        while injector.delays_remaining() > 0 {
            assert!(
                tokio::time::Instant::now() < deadline,
                "fixture: the read must reach its LRANGEs"
            );
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        let delta = Some(child_delta(2, 90_002));
        let relay = relay_for(&delta);
        let (result, _summary) = fx
            .ls
            .update_layer_with_summary(
                &fx.session_id,
                "grown-list-layer",
                oid,
                Arc::new(delta),
                relay,
                "grown-list-competitor",
                1,
                None,
                &mut NoCommitCharge,
            )
            .await;
        result.expect("the competitor commits while the LRANGEs are held");
        let grown: u64 = fx.tv.client().pool.llen(&list).await.expect("LLEN");
        (grown, started.elapsed())
    };
    let (held_result, (grown, competitor_done)) = tokio::join!(held, competitor);
    let seq = held_result.expect("the held call commits on its retry");
    assert!(
        competitor_done < HELD,
        "fixture: the competitor committed inside the hold ({competitor_done:?})"
    );
    assert_eq!(
        grown,
        GROWN_LIST_KIDS + 1,
        "fixture: the list grew by one under the held read"
    );
    assert_eq!(
        seq,
        before + 2,
        "the competitor's commit, then the held call's"
    );
    assert_eq!(
        fx.sample(
            "live_edit_update_layer_conflicts_total",
            &[("path", "watch"), ("kind", "watch_read_moved")]
        ) as u64,
        1,
        "the grown list's read was rejected once by its seq proof"
    );

    let fence = monitor_fence(fx.tv.prefix());
    let _: i64 = fx
        .tv
        .client()
        .pool
        .exists(&fence)
        .await
        .expect("EXISTS the fence");
    let seen = tap.drain_until(&fence).await;
    // Each read's `LRANGE` of the list, whichever issued it: the read's
    // window script (the feed's `lua` client) on the production transport.
    let ranges: Vec<Vec<String>> = seen
        .iter()
        .filter(|c| {
            c.name() == "LRANGE" && c.args.get(1).map(String::as_str) == Some(list.as_str())
        })
        .map(|c| c.args[2..].to_vec())
        .collect();
    let range = |n: u64| vec!["0".to_owned(), (n - 1).to_string()];
    assert_eq!(
        ranges,
        vec![
            // The competitor's own read, before its commit.
            range(GROWN_LIST_KIDS),
            // The held read: the length its LLEN returned, issued after the
            // competitor's commit grew the list.
            range(GROWN_LIST_KIDS),
            // The retry's read of the grown list.
            range(GROWN_LIST_KIDS + 1),
        ],
        "the held LRANGE ranges the list at its LLEN; the retry at its new length"
    );
    let final_list: Vec<String> = fx
        .tv
        .client()
        .pool
        .lrange(&list, 0, -1)
        .await
        .expect("LRANGE");
    assert_eq!(
        final_list.len() as u64,
        GROWN_LIST_KIDS + 2,
        "both children land"
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// The window script's allowance (b.gvf)
// ─────────────────────────────────────────────────────────────────────────────
//
// Each window of the WATCH read's rounds 2 and 3 is one `EVALSHA` of
// `fanout_window_read.lua`, except the windows that stay plain (round 1, a
// lone children list over τ, every window after a script fallback). A script
// window is charged before it is sent, like a plain one; a refused growth
// sends nothing. A window that falls back is charged again as plain commands,
// each holding its own future (`W_PLAIN_CMD` per wire command), before any of
// them is sent.

/// One charged WATCH read of `oid` under a [`ScriptedCharge`] of `room` (a
/// non-waiting growth above `room` is refused, and the read then stops),
/// with a `MONITOR` tap: whether it completed (`false`: it stopped), the
/// [`SCRIPTED_PRICED_MSG`] and `fan-out window` events in order, and the store's
/// feed.
async fn read_in_room(
    fx: &Fixture,
    oid: u64,
    room: u64,
) -> (
    bool,
    Vec<serde_json::Value>,
    Vec<crate::storage::snapshot_read_support::MonitoredCommand>,
) {
    use crate::storage::snapshot_read_support::{monitor_fence, test_valkey_addr, MonitorTap};
    use fred::interfaces::KeysInterface;
    use live_edit::storage::commit_charge::CommitChargeCell;
    use live_edit::storage::test_support::read_object_snapshot_slim_charged_for_test;

    let oid_str = oid.to_string();
    let mut tap = MonitorTap::start(&test_valkey_addr()).await;
    let (mut charge, log) = ScriptedCharge::new(room);
    let (read, captured) = {
        let cell = CommitChargeCell::new(&mut charge);
        crate::grpc::log_capture::capture_debug(read_object_snapshot_slim_charged_for_test(
            fx.tv.client(),
            fx.tv.key_schema(),
            &oid_str,
            &cell,
        ))
        .await
    };
    let completed = read.expect("the charged read does not fail").is_some();
    assert!(
        lock(&log).holds.is_empty(),
        "the read grows its charge without waiting: a refused growth stops it"
    );
    let fence = monitor_fence(fx.tv.prefix());
    let _: i64 = fx
        .tv
        .client()
        .pool
        .exists(&fence)
        .await
        .expect("EXISTS the fence");
    let seen = tap.drain_until(&fence).await;
    let events = captured
        .events
        .iter()
        .filter(|e| {
            let message = e["fields"]["message"].as_str();
            message == Some(SCRIPTED_PRICED_MSG)
                || (message == Some("fan-out window")
                    && e["fields"]["object_id"].as_str() == Some(oid_str.as_str()))
        })
        .cloned()
        .collect();
    (completed, events, seen)
}

/// The priced target a [`SCRIPTED_PRICED_MSG`] event carries.
fn priced(event: &serde_json::Value) -> Option<u64> {
    (event["fields"]["message"].as_str() == Some(SCRIPTED_PRICED_MSG))
        .then(|| event["fields"]["charge"].as_u64())
        .flatten()
}

/// Whether `event` is a `fan-out window` event issued as `transport`.
fn window_as(event: &serde_json::Value, transport: &str) -> bool {
    event["fields"]["message"].as_str() == Some("fan-out window")
        && event["fields"]["transport"].as_str() == Some(transport)
}

/// The first `transport` window of a calibration run's `events` whose
/// pre-issue growth (the last need priced before it) is above every need
/// priced earlier: that growth's target, and the largest target priced
/// before it — a room that refuses exactly that growth, every earlier one
/// held. A window whose growth is not above an earlier one is already
/// covered by what the read holds, and no room refuses it alone.
fn room_before_first(events: &[serde_json::Value], transport: &str) -> (u64, u64) {
    for (window, _) in events
        .iter()
        .enumerate()
        .filter(|(_, e)| window_as(e, transport))
    {
        let growth = events[..window]
            .iter()
            .rposition(|e| priced(e).is_some())
            .expect("fixture: a growth precedes the window");
        let target = priced(&events[growth]).expect("a priced event");
        let room = events[..growth]
            .iter()
            .filter_map(priced)
            .max()
            .expect("fixture: growths precede it");
        if target > room {
            return (target, room);
        }
    }
    panic!(
        "fixture: no {transport} window of the calibration read is preceded by a growth above \
         every earlier one: {events:?}"
    );
}

/// Ids pushed onto the root's children list beyond its live children, so
/// the read's round-3 window (its `LRANGE`, charged exactly from its `LLEN`)
/// carries more reply bytes than the count windows project (`sections + 2`
/// list entries): its pre-issue growth is above every earlier one. Half of
/// what a lone script window may read within τ at the wired window model
/// (`snapshot_read_unfenced_test::one_list_tau_capacity`), so the window is
/// a script window.
fn root_list_padding() -> u64 {
    crate::storage::snapshot_read_unfenced_test::one_list_tau_capacity() / 2
}

/// Push [`root_list_padding`] ids (none a section of the object) onto object
/// `oid`'s root children list.
async fn pad_root_list(fx: &Fixture, oid: u64) {
    use fred::interfaces::ListInterface;
    let list = fx.tv.key_schema().obj_children(&oid.to_string(), "0");
    let ids: Vec<String> = (0..root_list_padding())
        .map(|i| (900_000 + i).to_string())
        .collect();
    for chunk in ids.chunks(1_000) {
        let _: i64 = fx
            .tv
            .client()
            .pool
            .rpush(&list, chunk.to_vec())
            .await
            .expect("fixture: RPUSH the root list's padding");
    }
}

/// A refused growth before a script window sends no window script (b.gvf):
/// on a base layer whose root list is padded past its projection
/// ([`pad_root_list`]), calibrated by a run that never refuses, the first
/// script window whose pre-issue growth is above every earlier growth (the
/// round-3 `LRANGE` of the padded list) is found; a read of an identical
/// layer whose room is every growth but that one stops there. It sent that
/// window no `EVALSHA`: the store saw as many window scripts naming the
/// object as the read issued script windows, all before the refusal and
/// fewer than the calibration run sent. It ran its round 1 (plain windows
/// were issued).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn watch_read_refused_before_a_script_window_sends_no_window_script_b_gvf() {
    let fx = fixture("script-allowance", UpdateLayerBackend::Txn).await;
    pad_root_list(&fx, fx.oid).await;
    let header = fx.tv.key_schema().obj_header(&fx.oid.to_string());
    let sha = live_edit::storage::lua::FANOUT_WINDOW_READ_SCRIPT.sha1();
    // Distinct by their arguments: a window re-sent after a `NOSCRIPT` (a
    // concurrent test's `SCRIPT FLUSH` on the shared store) counts once.
    let window_scripts = |seen: &[crate::storage::snapshot_read_support::MonitoredCommand],
                          header: &str| {
        seen.iter()
            .filter(|c| {
                c.name() == "EVALSHA"
                    && c.args.get(1).is_some_and(|s| s.eq_ignore_ascii_case(sha))
                    && c.args.get(3).map(String::as_str) == Some(header)
            })
            .map(|c| c.args.as_slice())
            .collect::<std::collections::BTreeSet<_>>()
            .len()
    };

    let (completed, calibration, seen) = read_in_room(&fx, fx.oid, u64::MAX).await;
    assert!(completed, "fixture: the calibration read completes");
    let calibration_scripts = window_scripts(&seen, &header);
    assert_eq!(
        calibration_scripts,
        calibration
            .iter()
            .filter(|e| window_as(e, "script"))
            .count(),
        "control: the calibration read sends one window script per script window"
    );
    let (_, room) = room_before_first(&calibration, "script");

    let oid = create_base(&fx.ls, &fx.session_id, "script-allowance-refused").await;
    pad_root_list(&fx, oid).await;
    let header = fx.tv.key_schema().obj_header(&oid.to_string());
    let (completed, events, seen) = read_in_room(&fx, oid, room).await;
    assert!(!completed, "the read stops at the refused growth");
    assert!(
        events.iter().any(|e| window_as(e, "plain")),
        "the read ran its round 1 before the refusal"
    );
    let issued = events.iter().filter(|e| window_as(e, "script")).count();
    assert_eq!(
        window_scripts(&seen, &header),
        issued,
        "the store saw a window script for each script window issued, none for the refused \
         one: {events:?}"
    );
    assert!(
        issued < calibration_scripts,
        "the read stopped at the window whose growth was refused, short of the calibration's \
         {calibration_scripts} script windows ({issued} issued)"
    );
}

/// A window that falls back asks for its plain commands before it sends
/// them (b.gvf; needs `test-fault-injection`): a cluster-state refusal
/// queued on the fault seam for the base layer's first section answers in
/// place of its script window. In a run that never refuses, the growth
/// before the first `plain_fallback` window carries that window's wire
/// commands as plain commands in flight, above every earlier growth's. A read
/// whose room is every growth but that one stops there: it sent neither the
/// window script (the seam answered it) nor any of the plain window's
/// commands (no client `HGETALL` of a child section), and it counted its
/// fallback.
#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn watch_read_falling_back_charges_its_plain_window_before_sending_it_b_gvf() {
    use fred::error::{Error as FredError, ErrorKind as FredErrorKind};
    use live_edit::storage::diff_read::{
        ObjectSnapshotReadFaultInjector, ObjectSnapshotReadFaultTarget, ObjectSnapshotReader,
    };

    let fx = fixture("fallback-allowance", UpdateLayerBackend::Txn).await;
    let injector = Arc::new(ObjectSnapshotReadFaultInjector::new());
    fx.tv
        .client()
        .install_object_snapshot_read_fault_injector_for_test(Arc::clone(&injector));
    let refuse_first_section = || {
        injector.push_at(
            ObjectSnapshotReadFaultTarget::nth_of_family(
                ObjectSnapshotReader::UnfencedFanout,
                "section",
                0,
            ),
            FredError::new(
                FredErrorKind::Unknown,
                "ERR Script attempted to access a slot not served",
            ),
        );
    };
    let fallbacks = || fx.sample("live_edit_snapshot_read_window_fallback_total", &[]);

    refuse_first_section();
    let (completed, calibration, _) = read_in_room(&fx, fx.oid, u64::MAX).await;
    assert!(
        completed,
        "fixture: the calibration read completes on plain windows"
    );
    assert_eq!(injector.remaining(), 0, "fixture: the refusal was taken");
    let window = calibration
        .iter()
        .position(|e| window_as(e, "plain_fallback"))
        .expect("fixture: the calibration read fell back");
    let growth = calibration[..window]
        .iter()
        .rposition(|e| priced(e).is_some())
        .expect("a growth precedes the fallback window");
    let plain = |e: &serde_json::Value| e["fields"]["plain_in_flight"].as_u64().unwrap_or(0);
    let fallback_commands = calibration[window]["fields"]["commands"]
        .as_u64()
        .expect("a window event carries its commands");
    assert!(
        plain(&calibration[growth]) >= fallback_commands
            && calibration[..growth]
                .iter()
                .all(|e| plain(e) < plain(&calibration[growth])),
        "the fallback's growth asks for the window's wire commands as plain commands in flight \
         ({} for a window of {fallback_commands} commands), more than any growth before it: \
         {calibration:?}",
        plain(&calibration[growth])
    );
    let (_, room) = room_before_first(&calibration, "plain_fallback");
    let before = fallbacks();

    let oid = create_base(&fx.ls, &fx.session_id, "fallback-allowance-refused").await;
    let oid_str = oid.to_string();
    refuse_first_section();
    let (completed, events, seen) = read_in_room(&fx, oid, room).await;
    assert_eq!(injector.remaining(), 0, "the refusal was taken");
    assert!(
        !completed,
        "the read stops at the fallback's refused growth"
    );
    assert!(
        !events
            .iter()
            .any(|e| window_as(e, "script") || window_as(e, "plain_fallback")),
        "no round-2 window was issued: {events:?}"
    );
    let child_reads = seen
        .iter()
        .filter(|c| {
            c.name() == "HGETALL"
                && (FIRST_CHILD..FIRST_CHILD + BASE_CHILDREN).any(|id| {
                    c.args.get(1).map(String::as_str)
                        == Some(
                            fx.tv
                                .key_schema()
                                .obj_section(&oid_str, &id.to_string())
                                .as_str(),
                        )
                })
        })
        .count();
    assert_eq!(
        child_reads, 0,
        "none of the plain window's commands was sent before its growth was granted"
    );
    assert_eq!(fallbacks() - before, 1.0, "the read counted its fallback");
}
