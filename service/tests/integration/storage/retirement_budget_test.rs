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

//! The retirement allowance `R_ref` on every retirement route (b.fsz MR-D4;
//! `storage::retirement`): `DeleteLayer`, the `CreateLayer` overwrite's
//! cleanup body and the orphan sweep each size their retirement
//! (`HGET obj:hdr ext_gc_scheduled`, `HLEN obj:ext`, and the five structural
//! indices' cardinalities), reserve the object's charge
//! (`retirement_charge_bytes`: its page, one member past its largest index
//! and at most `P`, widened to a row per size read in flight, plus `h × H`)
//! on the `LayerStore`'s own `MemoryBudget` (FIFO, an oversized request
//! clamped to run alone, top-ups forced), walk the object's references —
//! widening to the full page `W` before reading past an index that grew
//! after the size read — shrink to the page once the release returns, and
//! give the reservation back on every exit.
//!
//! Every store here runs at the smallest pod's allowance, 5 MiB, unless a
//! test names another, so `P` is 64 members and `W` 2.5 MiB: two retirements
//! at the full page fit side by side, and `obj:ext` counting more than 2,560
//! references makes one oversized. [`Rig::charge`] re-derives what a
//! retirement of a given object reserves. A test that needs the allowance
//! full holds it itself, through a grant on the store's
//! `retirement_budget()`; one that needs a retirement
//! parked mid-walk arms a hold on the snapshot-read seam at the walk's first
//! `HSCAN obj:ext` step (`ext_index`, which no other read issues).
//!
//! | Test | Design row | What it pins |
//! |---|---|---|
//! | [`the_test_allowances_are_the_pod_memory_plans_at_the_smallest_and_the_reference_pod`] | F3 | [`R_REF`] and [`R_REF_5_GIB`] are the plan's allowances at the 2,709 MiB minimum and the 5 GiB reference pod |
//! | [`concurrent_retirements_on_every_route_hold_at_most_the_allowance_and_give_it_all_back`] | D4-T3 | `DeleteLayer`s, call bodies, sweep tail arms and generic-arm orphans at once: reserved plus over-commit never above the allowance, every object gone, and the allowance, its over-commit and its waiters back to zero |
//! | [`retirements_are_granted_in_arrival_order`] | D4-T3 | FIFO: a small reservation queued behind an oversized `DeleteLayer` is granted only once that delete has finished, though it would have fitted beside the one before it |
//! | [`a_retirement_whose_teardown_page_fails_gives_its_reservation_back`] | D4-T3 | an injected page error ending a `DeleteLayer` |
//! | [`a_delete_dropped_mid_walk_gives_its_reservation_back`] | D4-T3 | a cancel mid-walk |
//! | [`wait_for_cleanup_ends_a_queued_body_and_it_keeps_its_object`] | D4-T3 | `end_waits()` first in `wait_for_cleanup`: a queued overwrite body ends with the header kept and no begin run; the allowance is not closed, so a reservation that fits is granted at once afterwards |
//! | [`a_body_reaching_the_allowance_after_shutdown_began_that_fits_completes_within_the_drain`] | D4-T3 | a body parked in its size read when shutdown begins and that fits is granted `immediate` and runs within the drain |
//! | [`a_body_reaching_the_allowance_after_shutdown_began_that_would_wait_ends_at_once`] | D4-T3 | the same body with the allowance full ends at once, never queued: the drain returns within `BUDGET / 2`, the header kept |
//! | [`an_overwrite_body_whose_wake_up_never_lands_reserves_nothing`] | D4-T3 | a wake that never lands reserves nothing, checked once the body has ended |
//! | [`a_delete_whose_deadline_passes_while_it_waits_changes_nothing_and_a_retry_works`] | D4-T4 | nothing written; this replica's admission entry and structural-cache entry kept (premise-check amendment); the wait counted `cancelled`; a retry deletes |
//! | [`a_delete_at_shutdown_that_would_wait_answers_unavailable_and_retires_nothing`] | I-D4e | shutdown's exit for a delete that would wait: `UNAVAILABLE` with `SERVER_SHUTTING_DOWN`, never queued, nothing written, both entries kept |
//! | [`a_delete_at_shutdown_that_fits_deletes_the_layer`] | I-D4e | a delete at shutdown whose reservation fits deletes the layer |
//! | [`an_oversized_retirement_is_clamped_to_the_allowance_and_runs_alone`] | D4-T5 | the clamped reservation holds the whole allowance, with no over-commit, and a second retirement waits for it |
//! | [`a_retirement_with_no_obj_ext_tops_up_for_the_references_its_walk_finds`] | D4-T6 | `H` = 0, so the walk's references are covered by a forced top-up, repaid at the shrink |
//! | [`a_clean_up_with_a_handful_of_large_values_runs_beside_a_large_one_at_the_5_gib_pods_allowance`] | F3 | the reservation follows the object: a retirement owing 5 references runs beside one owing 1,000 at `R_ref` = 10 MiB, both reserved at once |
//! | [`a_clean_up_with_a_handful_of_large_values_runs_beside_a_large_one_at_the_smallest_pods_allowance`] | F3 | the same at the 2,709 MiB pod's `R_ref` = 5 MiB |
//! | [`a_delete_whose_index_grew_after_its_size_read_widens_its_page_before_reading_past_it`] | F3 | the estimate low: an index grown while the delete queued is read past the object's page only after a forced top-up to the full page `W` |
//! | [`a_delete_whose_layer_is_replaced_while_it_waits_deletes_the_replacement_on_this_replica`] | D4-T11 | the delete-vs-replace retarget on one replica |
//! | [`a_delete_whose_layer_is_replaced_on_another_replica_retires_the_old_objects_entries_here`] | D4-T11 | the cross-replica row: the retarget retires the replaced object's admission and cache entries on the deleting replica (the one-shot guard drop), and leaves none for the replacement |
//! | [`an_overwrite_body_wakes_its_stream_before_it_queues_for_the_allowance`] | D4-T12 | a wake-up (here the body's retry of a failed inline one) is never queued behind `R_ref` |
//! | [`the_orphan_sweep_runs_one_retirement_at_a_time`] | D4-T13 | the sweep awaits each tail body: one in flight at a time (`OverwriteCleanupProbe`), one reservation at a time |
//! | [`a_duplicate_run_of_an_overwrite_tail_pays_its_releases_once`] | D4-T14 | the call's body and a sweep's tail arm both retire one replaced object; the releases are paid once |
//! | [`a_delete_meeting_a_foreign_header_or_obj_ext_counts_it_once_skips_the_release_and_removes_it`] | D4-T15 | `DeleteLayer`: answered OK, the foreign key gone, one tick on `header` / `ext_index`, nothing released, `W` reserved |
//! | [`an_overwrite_body_meeting_a_foreign_header_or_obj_ext_counts_it_once_skips_the_release_and_removes_it`] | D4-T15 | the overwrite body, the same |
//! | [`a_sweep_retirement_meeting_a_foreign_obj_ext_counts_it_once_skips_the_release_and_removes_it`] | D4-T15 | the sweep's generic arm, a foreign `obj:ext` |
//! | [`a_delete_meeting_a_foreign_structural_index_counts_it_under_its_family_reserves_the_full_page_and_stops_its_teardown_there`] | D4-T15 | `DeleteLayer`, each of the five indices foreign: the tick under its family (a value index's walk ticks once more), `W` reserved, a set index's release scheduled and a value index's not begun, the teardown stopped at the index after taking the header, and the delete answered `INTERNAL` with the mapping deleted and the layers-index member left |
//! | [`an_overwrite_body_meeting_a_foreign_structural_index_counts_it_under_its_family_reserves_the_full_page_and_stops_its_teardown_there`] | D4-T15 | the overwrite body: the tick, `W`, the release and the teardown's stop, as on `DeleteLayer` |
//! | [`a_sweep_retirement_meeting_a_foreign_structural_index_counts_it_under_its_family_reserves_the_full_page_and_stops_its_teardown_there`] | D4-T15 | the sweep's generic arm: the tick, `W`, the release and the teardown's stop, as on `DeleteLayer` |
//! | [`a_delete_whose_largest_index_is_any_one_of_the_five_reserves_its_page_one_member_past_it`] | D4-T15 | each of the five counts sizes the page: 10 members in one index alone reserve 11 rows |
//! | [`a_delete_whose_size_read_meets_a_wrong_type_on_one_index_count_counts_it_reserves_the_full_page_and_deletes_the_layer`] | D4-T15 | a `WRONGTYPE` on one count alone (the per-command seam): one tick under its family, `W` reserved, the release scheduled, the layer deleted |
//! | [`a_delete_whose_size_read_fails_permanently_on_one_index_count_answers_internal_and_changes_nothing`] | D4-T15 | a permanent non-`WRONGTYPE` failure on one count: a `protocol` refusal, `INTERNAL`, nothing done or reserved |
//! | [`a_delete_whose_size_read_fails_transiently_answers_unavailable_and_changes_nothing`] | D4-T15 | `UNAVAILABLE`, the mapping, layers index and object intact, the admission and structural-cache entries kept |
//! | [`a_retirement_ticks_the_retirement_series_and_leaves_the_catch_up_series_flat`] | D4-T16 | the per-instance metric sink |
//! | [`a_delete_whose_retirement_panics_gives_its_reservation_back_and_retires_its_entries`] | D4-T3 | a panic inside the retirement (its release's enqueue): the unwind gives the reservation back and the admission guard retires both entries |
//! | [`a_delete_whose_mapping_re_read_fails_changes_nothing_and_a_retry_works`] | D4-T4 | a failed mapping re-read: `INTERNAL`, nothing written, both entries kept, the reservation given back; a retry deletes |
//! | [`two_deletes_of_one_layer_waiting_together_both_answer_ok`] | D4-T11 | the re-read's nil branch: the second delete finds the mapping gone and answers OK |
//! | [`an_overwrite_body_whose_size_read_fails_keeps_its_object_for_a_later_sweep`] | D4-T15 | the body's size-read failure: the header kept, nothing reserved, no begin; a later sweep retires it and releases its references |
//! | [`a_sweep_retirement_whose_size_read_fails_keeps_its_orphan_for_a_later_pass`] | D4-T15 | the sweep's, the same |
//! | [`a_sweep_retirement_cancelled_while_it_waits_keeps_its_orphan_and_reserves_nothing`] | D4-T3 | the sweep's wait raced against its cancel: off the queue, `cancelled`, nothing reserved, the orphan kept |
//!
//! Not covered here: the overwrite finisher's spawn of the body (every spawn
//! goes through the same tail and body as the call's, covered by the call).
//! D4-T7's and D4-T17's teardown rows are `retirement_teardown_test.rs`'s;
//! D4-T1, T9 the walk's (and its failure handling),
//! `retirement_walk_test.rs`; D4-T2 the heap bound's,
//! `tests/upload_memory_envelope`.
//!
//! Spawned overwrite bodies are drained either by polling the object's keys
//! or through `LayerStore::wait_for_cleanup`, which ends the allowance's
//! waits first (`MemoryBudget::end_waits`): a body that fits still runs to
//! the end, so a drain waits for it, and a test that needs a body queued
//! drains it only where its ending is what the test pins.
//!
//! # Worst case per test
//!
//! Every wait for a condition is bounded by [`BUDGET`] (20 s), and none is
//! reached on a healthy store. A test's worst case is its number of such
//! waits × `BUDGET` — a wait on several objects at once is one wait
//! ([`until_objects_gone`]), [`Rig::sweep_until`] is two (its condition
//! and its join), and a test that runs its cases at once (the
//! foreign-structural-index tests) counts one case's waits — and each stays
//! inside nextest's 120 s hang-killer:
//!
//! | Test | Bounded waits | Worst case |
//! |---|---|---|
//! | `the_test_allowances_…` | 0 | — |
//! | `concurrent_retirements_…` | 5 | 100 s |
//! | `retirements_are_granted_in_arrival_order` | 4 | 80 s |
//! | `a_retirement_whose_teardown_page_fails_…` | 0 | — |
//! | `a_delete_dropped_mid_walk_…` | 2 | 40 s |
//! | `wait_for_cleanup_ends_a_queued_body_…` | 3 | 60 s |
//! | `a_body_reaching_…_that_fits_…` | 4 | 80 s |
//! | `a_body_reaching_…_that_would_wait_…` | 3½ | 70 s |
//! | `an_overwrite_body_whose_wake_up_never_lands_…` | 2 | 40 s |
//! | `a_delete_whose_deadline_passes_…` | 2 | 40 s |
//! | `a_delete_at_shutdown_that_would_wait_…` | 1 | 20 s |
//! | `a_delete_at_shutdown_that_fits_…` | 0 | — |
//! | `an_oversized_retirement_…` | 3 | 60 s |
//! | `a_retirement_with_no_obj_ext_…` | 1 | 20 s |
//! | `a_clean_up_with_a_handful_…_5_gib_…` | 3 | 60 s |
//! | `a_clean_up_with_a_handful_…_smallest_…` | 3 | 60 s |
//! | `a_delete_whose_index_grew_…` | 3 | 60 s |
//! | `…_replaced_while_it_waits_…_on_this_replica` | 3 | 60 s |
//! | `…_replaced_on_another_replica_…` | 2 | 40 s |
//! | `an_overwrite_body_wakes_its_stream_…` | 3 | 60 s |
//! | `the_orphan_sweep_runs_one_retirement_at_a_time` | 3 | 60 s |
//! | `a_duplicate_run_of_an_overwrite_tail_…` | 5 | 100 s |
//! | `a_delete_meeting_a_foreign_…` | 0 | — |
//! | `an_overwrite_body_meeting_a_foreign_…` | 3 | 60 s |
//! | `a_sweep_retirement_meeting_a_foreign_…` | 2 | 40 s |
//! | `a_delete_meeting_a_foreign_structural_index_…` | 1 | 20 s |
//! | `an_overwrite_body_meeting_a_foreign_structural_index_…` | 3 | 60 s |
//! | `a_sweep_retirement_meeting_a_foreign_structural_index_…` | 2 | 40 s |
//! | `a_delete_whose_largest_index_is_any_one_…` | 2 | 40 s |
//! | `a_delete_whose_size_read_meets_a_wrong_type_…` | 1 | 20 s |
//! | `a_delete_whose_size_read_fails_permanently_…` | 0 | — |
//! | `a_delete_whose_size_read_fails_…` | 1 | 20 s |
//! | `a_retirement_ticks_the_retirement_series_…` | 1 | 20 s |
//! | `a_delete_whose_retirement_panics_…` | 2 | 40 s |
//! | `a_delete_whose_mapping_re_read_fails_…` | 2 | 40 s |
//! | `two_deletes_of_one_layer_…` | 3 | 60 s |
//! | `an_overwrite_body_whose_size_read_fails_…` | 4 | 80 s |
//! | `a_sweep_retirement_whose_size_read_fails_…` | 4 | 80 s |
//! | `a_sweep_retirement_cancelled_while_it_waits_…` | 2 | 40 s |
//!
//! The joins of spawned deletes that the test has already seen finish their
//! wait are not counted: a hang there is the hang-killer's to report.
//!
//! Requires Valkey (`TEST_VALKEY_URL`) and the `test-fault-injection`
//! feature: the module is compiled out without it.

use std::collections::{BTreeSet, VecDeque};
use std::future::Future;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use fred::interfaces::{HashesInterface, KeysInterface, SetsInterface, StreamsInterface};
use tokio_util::sync::CancellationToken;

use live_edit::domain::types::ObjectState;
use live_edit::error::ServiceError;
use live_edit::observability::metrics::test_util::{render, sample_value};
use live_edit::observability::metrics::Metrics;
use live_edit::storage::diff_read::{
    ObjectSnapshotReadFaultInjector, ObjectSnapshotReadFaultTarget, ObjectSnapshotReader,
    SnapshotReadHold,
};
use live_edit::storage::external_values::{ExternalValueStore, MockExternalValueStore};
use live_edit::storage::keys;
use live_edit::storage::layer::{LayerStore, OverwriteCleanupProbe};
use live_edit::storage::memory_budget::MemoryBudget;
use live_edit::storage::notifications::{DELETED_STREAM_PAYLOAD, NOTIFICATION_PAYLOAD_FIELD};
use live_edit::storage::orphan_sweep::spawn_orphan_sweep;
use live_edit::storage::retirement::{
    retirement_charge_bytes, RetirementPaging, RETIREMENT_REFERENCE_CHARGE_BYTES,
};
use live_edit::storage::session::SessionStore;
use live_edit::storage::structural_cache::{CheckedOut, PutBack};
use live_edit::storage::test_support::relay::RelayedUpdates;
use live_edit::storage::test_support::TestValkey;

use crate::common::owned_delta::delta_with_set_field_inline;
use crate::storage::structural_cache_retirement_test::{
    assert_retired, cache_entry_for, entry, hold_checkout,
};

/// The retirement allowance every store here runs at unless a test names
/// another: the smallest pod's (`L / 500` at 2,709 MiB), so `P` = 64 and
/// `W` = 2.5 MiB. Checked against the pod memory plan by
/// [`the_test_allowances_are_the_pod_memory_plans_at_the_smallest_and_the_reference_pod`].
pub(crate) const R_REF: u64 = 5 * 1024 * 1024;

/// The default 5 GiB pod's retirement allowance (`L / 500`, whole MiB),
/// checked against the plan by the same test.
const R_REF_5_GIB: u64 = 10 * 1024 * 1024;

/// Bound on every wait for a condition: unreachable on a healthy store. A
/// test's worst case is its count of these, each a deadline rather than a
/// sleep, and stays inside nextest's 120 s hang-killer (the module docs'
/// table).
pub(crate) const BUDGET: Duration = Duration::from_secs(20);

/// The poll interval inside [`BUDGET`] — not an additional wait.
const POLL: Duration = Duration::from_millis(5);

/// `W`, the full page bound of a store at [`R_REF`]: what a retirement of an
/// object with an index of `P` members or more reserves before its
/// references.
pub(crate) fn page_bytes() -> u64 {
    RetirementPaging::for_allowance(R_REF).page_bytes()
}

/// Poll `condition` until it holds, failing after [`BUDGET`].
pub(crate) async fn eventually<F, Fut>(what: &str, mut condition: F)
where
    F: FnMut() -> Fut,
    Fut: Future<Output = bool>,
{
    let deadline = Instant::now() + BUDGET;
    loop {
        if condition().await {
            return;
        }
        assert!(
            Instant::now() < deadline,
            "timed out after {BUDGET:?} waiting for: {what}"
        );
        tokio::time::sleep(POLL).await;
    }
}

/// One test's fixture: a Valkey namespace, a metrics registry installed on
/// its client, the mock external-value store every route releases into, and
/// one session.
pub(crate) struct Rig {
    pub(crate) tv: TestValkey,
    pub(crate) metrics: Arc<Metrics>,
    pub(crate) ext: Arc<MockExternalValueStore>,
    pub(crate) session_id: String,
}

impl Rig {
    pub(crate) async fn new(label: &str) -> Self {
        let tv = TestValkey::setup().await;
        let metrics = Arc::new(crate::common::metrics_scrape_only(label));
        tv.client()
            .install_observability_hooks(Arc::clone(&metrics))
            .await;
        let session_id = SessionStore::new(tv.client().clone(), tv.key_schema().clone())
            .create(label, &format!("/{label}"), "retirement-tester")
            .await
            .expect("session creation must succeed");
        Self {
            tv,
            metrics,
            ext: Arc::new(MockExternalValueStore::new()),
            session_id,
        }
    }

    /// A store on this rig's client, at [`R_REF`], ticking this rig's metrics.
    pub(crate) fn store(&self) -> LayerStore {
        self.store_at(R_REF)
    }

    /// A store on this rig's client, at an allowance of `r_ref` bytes,
    /// ticking this rig's metrics.
    pub(crate) fn store_at(&self, r_ref: u64) -> LayerStore {
        LayerStore::new(self.tv.client().clone(), self.tv.key_schema().clone(), 4096)
            .with_metrics(Arc::clone(&self.metrics))
            .with_retirement_budget_bytes(r_ref)
    }

    /// The largest cardinality of `oid`'s five structural indices, as the
    /// size read counts it.
    pub(crate) async fn largest_index(&self, oid: u64) -> u64 {
        use fred::interfaces::SortedSetsInterface;
        let (keys, oid, pool) = (
            self.tv.key_schema(),
            oid.to_string(),
            &self.tv.client().pool,
        );
        let mut largest = 0;
        for set in [
            keys.obj_sections_index(&oid),
            keys.obj_deleted_sections_index(&oid),
            keys.obj_children_index(&oid),
        ] {
            let members: u64 = pool.scard(set).await.expect("SCARD must succeed");
            largest = largest.max(members);
        }
        for sorted in [
            keys.obj_fields_index(&oid),
            keys.obj_time_samples_index(&oid),
        ] {
            let members: u64 = pool.zcard(sorted).await.expect("ZCARD must succeed");
            largest = largest.max(members);
        }
        largest
    }

    /// What a retirement of `oid` owing `refs` indexed references reserves at
    /// an allowance of `r_ref`: its charge at its page, one member past its
    /// largest index and at most `P`.
    pub(crate) async fn charge_at(&self, r_ref: u64, oid: u64, refs: u32) -> u64 {
        let page =
            RetirementPaging::for_allowance(r_ref).for_rows(self.largest_index(oid).await + 1);
        retirement_charge_bytes(page.page_len(), u64::from(refs))
    }

    /// [`Self::charge_at`] at [`R_REF`].
    pub(crate) async fn charge(&self, oid: u64, refs: u32) -> u64 {
        self.charge_at(R_REF, oid, refs).await
    }

    /// Create (or overwrite) `layer` through `ls`, empty, and record `refs`
    /// distinct references on its `obj:ext` (tagged `tag`): `H` = `refs`.
    pub(crate) async fn layer(&self, ls: &LayerStore, layer: &str, tag: u32, refs: u32) -> u64 {
        let state = ObjectState {
            sequence_num: 1,
            first_version: 1,
            source_format: "usda".to_owned(),
            ..Default::default()
        };
        let oid = ls
            .create_layer(
                &self.session_id,
                layer,
                "retirement-tester",
                &state,
                self.ext.clone(),
            )
            .await
            .expect("create_layer must succeed");
        if refs > 0 {
            let fields: Vec<(String, String)> = (0..refs)
                .map(|i| (ref_hex(tag, i), "1".to_owned()))
                .collect();
            let _: () = self
                .tv
                .client()
                .pool
                .hset(self.tv.key_schema().obj_ext_index(&oid.to_string()), fields)
                .await
                .expect("HSET obj:ext must succeed");
        }
        oid
    }

    /// The layer mapping of `layer`, `None` once deleted.
    pub(crate) async fn mapping(&self, layer: &str) -> Option<String> {
        self.tv
            .client()
            .pool
            .get(self.tv.key_schema().layer_mapping(&self.session_id, layer))
            .await
            .expect("GET mapping must succeed")
    }

    /// Whether `layer` is a member of the session's layers index.
    pub(crate) async fn indexed(&self, layer: &str) -> bool {
        self.tv
            .client()
            .pool
            .sismember(
                self.tv.key_schema().layers_index(&self.session_id),
                keys::hex_encode(layer),
            )
            .await
            .expect("SISMEMBER layers index must succeed")
    }

    pub(crate) async fn exists(&self, key: &str) -> bool {
        let n: u64 = self
            .tv
            .client()
            .pool
            .exists(key)
            .await
            .expect("EXISTS must succeed");
        n == 1
    }

    pub(crate) async fn header_exists(&self, oid: u64) -> bool {
        self.exists(&self.tv.key_schema().obj_header(&oid.to_string()))
            .await
    }

    /// Every `obj:*` key of `oid` still standing, but its stream (which
    /// expires on its own) and its release carrier (which outlives a
    /// teardown by design).
    pub(crate) async fn object_keys(&self, oid: u64) -> Vec<String> {
        crate::common::scan_object_keys(self.tv.client(), self.tv.prefix(), oid)
            .await
            .into_iter()
            .map(|k| k.as_str_lossy().into_owned())
            .filter(|k| !k.ends_with(":obj:stream") && !k.contains(":obj:extret"))
            .collect()
    }

    /// Wait until every `obj:*` key of `oid` is gone: one bounded wait
    /// ([`until_objects_gone`]).
    pub(crate) async fn until_object_gone(&self, oid: u64) {
        until_objects_gone(self, &[oid]).await;
    }

    /// The `payload=deleted` entries on `oid`'s stream.
    pub(crate) async fn deleted_entries(&self, oid: u64) -> usize {
        let entries: Vec<(String, std::collections::HashMap<String, bytes::Bytes>)> = self
            .tv
            .client()
            .pool
            .xrange_values(self.tv.key_schema().layer_stream(oid), "-", "+", None)
            .await
            .expect("XRANGE must succeed");
        entries
            .iter()
            .filter(|(_, fields)| {
                fields.get(NOTIFICATION_PAYLOAD_FIELD).map(|p| &p[..])
                    == Some(DELETED_STREAM_PAYLOAD.as_bytes())
            })
            .count()
    }

    /// An unlabelled sample of this rig's metrics, `None` when absent.
    pub(crate) fn sample(&self, name: &str, labels: &[(&str, &str)]) -> Option<f64> {
        sample_value(&render(&self.metrics), name, labels)
    }

    /// `live_edit_retirement_wait_seconds_count{outcome}`, 0 when the series
    /// has no observation yet (a histogram is not pre-registered).
    pub(crate) fn waits(&self, outcome: &str) -> f64 {
        self.sample(
            "live_edit_retirement_wait_seconds_count",
            &[("outcome", outcome)],
        )
        .unwrap_or(0.0)
    }

    /// `live_edit_snapshot_read_type_error_total{key_family}`.
    pub(crate) fn type_errors(&self, family: &str) -> f64 {
        self.sample(
            "live_edit_snapshot_read_type_error_total",
            &[("key_family", family)],
        )
        .unwrap_or_else(|| panic!("key_family={family} is zero-seeded"))
    }

    /// `live_edit_ext_ref_released_total{reason="retirement"}`.
    pub(crate) fn retirement_releases(&self) -> f64 {
        self.sample(
            "live_edit_ext_ref_released_total",
            &[("reason", "retirement")],
        )
        .unwrap_or_else(|| panic!("ext_ref_released_total{{retirement}} is zero-seeded"))
    }

    /// A seam injector installed on this rig's client.
    pub(crate) fn injector(&self) -> Arc<ObjectSnapshotReadFaultInjector> {
        let injector = Arc::new(ObjectSnapshotReadFaultInjector::new());
        self.tv
            .client()
            .install_object_snapshot_read_fault_injector_for_test(Arc::clone(&injector));
        injector
    }

    /// Run the orphan sweep through `ls` until `done` holds, then cancel and
    /// join it.
    pub(crate) async fn sweep_until<F, Fut>(&self, ls: LayerStore, what: &str, done: F)
    where
        F: FnMut() -> Fut,
        Fut: Future<Output = bool>,
    {
        let cancel = CancellationToken::new();
        let handle = spawn_orphan_sweep(
            self.tv.client().clone(),
            self.tv.key_schema().clone(),
            self.ext.clone(),
            Arc::new(ls),
            crate::common::test_leader("retirement-sweep"),
            cancel.clone(),
            Some(Arc::clone(&self.metrics)),
        );
        eventually(what, done).await;
        cancel.cancel();
        tokio::time::timeout(BUDGET, handle)
            .await
            .expect("the orphan sweep must exit once cancelled")
            .expect("the orphan sweep must not panic");
    }
}

/// The `obj:ext` field of reference `i` under `tag`: 64 hex characters.
pub(crate) fn ref_hex(tag: u32, i: u32) -> String {
    format!("{tag:032x}{i:032x}")
}

/// A hold on the next reference walk's first `HSCAN obj:ext` step: the walk
/// parks there with its reservation held and every value row read.
pub(crate) fn hold_walk_at_obj_ext(
    injector: &ObjectSnapshotReadFaultInjector,
) -> Arc<SnapshotReadHold> {
    injector.push_hold_at(ObjectSnapshotReadFaultTarget::nth_of_family(
        ObjectSnapshotReader::UnfencedFanout,
        "ext_index",
        0,
    ))
}

/// Wait for `hold` to park, bounded.
pub(crate) async fn parked(hold: &SnapshotReadHold) {
    tokio::time::timeout(BUDGET, hold.parked())
        .await
        .expect("the retirement must reach the held command");
}

/// Wait until `budget` has exactly `n` queued reservations.
pub(crate) async fn waiters(budget: &MemoryBudget, n: usize) {
    eventually(&format!("{n} queued reservations"), || async {
        budget.waiters() == n
    })
    .await;
}

/// The overwrite fault-injection queue of a store whose old-stream wake-up
/// fails `failures` times.
fn failing_wakes(failures: usize) -> Arc<Mutex<VecDeque<fred::error::Error>>> {
    Arc::new(Mutex::new(
        (0..failures)
            .map(|_| fred::error::Error::new(fred::error::ErrorKind::IO, "injected wake failure"))
            .collect(),
    ))
}

/// The wake-ups an overwrite takes: the tail's inline one and the cleanup
/// body's three retries (`OVERWRITE_WAKE_RETRIES`).
const EVERY_WAKE: usize = 4;

/// Leave replaced objects stranded for the sweep: overwrite each `(layer,
/// tag, refs)` through a store of its own whose every wake-up of the old
/// stream fails, so its cleanup body gives up with the old object fenced and
/// unmapped. One bounded wait covers every body's last wake-up failure.
/// Returns each old and new object with the store its body runs on.
async fn stranded_overwrites(rig: &Rig, specs: &[(&str, u32, u32)]) -> Vec<(u64, u64, LayerStore)> {
    let mut stranded = Vec::new();
    let mut every_wake = Vec::new();
    for &(layer, tag, refs) in specs {
        let wakes = failing_wakes(EVERY_WAKE);
        let ls = rig
            .store()
            .with_create_layer_overwrite_xadd_fault_injector_for_test(Arc::clone(&wakes));
        let old = rig.layer(&ls, layer, tag, refs).await;
        let new = rig.layer(&ls, layer, tag + 1, 0).await;
        every_wake.push(wakes);
        stranded.push((old, new, ls));
    }
    eventually("every wake-up of every old stream failed", || async {
        every_wake
            .iter()
            .all(|wakes| wakes.lock().unwrap().is_empty())
    })
    .await;
    for (old, _, _) in &stranded {
        assert!(
            rig.header_exists(*old).await,
            "fixture: the replaced object is left for the sweep"
        );
    }
    stranded
}

/// Wait until every `obj:*` key of every object in `oids` is gone: one
/// bounded wait, however many objects.
pub(crate) async fn until_objects_gone(rig: &Rig, oids: &[u64]) {
    eventually(&format!("objects {oids:?}'s keys gone"), || async {
        for &oid in oids {
            if !rig.object_keys(oid).await.is_empty() {
                return false;
            }
        }
        true
    })
    .await;
}

/// An orphan of the generic arm: `layer`'s object with its mapping and
/// layers-index member removed, as a `DeleteLayer` that stopped after its
/// mapping delete leaves it.
async fn generic_orphan(rig: &Rig, ls: &LayerStore, layer: &str, tag: u32, refs: u32) -> u64 {
    let oid = rig.layer(ls, layer, tag, refs).await;
    let keys = rig.tv.key_schema();
    let _: i64 = rig
        .tv
        .client()
        .pool
        .del(keys.layer_mapping(&rig.session_id, layer))
        .await
        .expect("DEL mapping must succeed");
    let _: i64 = rig
        .tv
        .client()
        .pool
        .srem(keys.layers_index(&rig.session_id), keys::hex_encode(layer))
        .await
        .expect("SREM layers index must succeed");
    oid
}

/// The largest `reserved + overcommit` and `reserved` a sampler saw on a
/// budget, sampled until stopped.
struct Sampler {
    stop: Arc<AtomicBool>,
    held: Arc<AtomicU64>,
    reserved: Arc<AtomicU64>,
    handle: tokio::task::JoinHandle<()>,
}

impl Sampler {
    fn start(budget: Arc<MemoryBudget>) -> Self {
        let (stop, held, reserved) = (
            Arc::new(AtomicBool::new(false)),
            Arc::new(AtomicU64::new(0)),
            Arc::new(AtomicU64::new(0)),
        );
        let handle = tokio::spawn({
            let (stop, held, reserved) = (stop.clone(), held.clone(), reserved.clone());
            async move {
                while !stop.load(Ordering::SeqCst) {
                    let r = budget.reserved_bytes();
                    reserved.fetch_max(r, Ordering::SeqCst);
                    held.fetch_max(r + budget.overcommit_bytes(), Ordering::SeqCst);
                    tokio::time::sleep(Duration::from_millis(1)).await;
                }
            }
        });
        Self {
            stop,
            held,
            reserved,
            handle,
        }
    }

    /// `(largest reserved + overcommit, largest reserved)`.
    async fn stop(self) -> (u64, u64) {
        self.stop.store(true, Ordering::SeqCst);
        self.handle.await.expect("the sampler must not panic");
        (
            self.held.load(Ordering::SeqCst),
            self.reserved.load(Ordering::SeqCst),
        )
    }
}

/// Assert `budget` holds nothing and queues nothing.
fn assert_returned(budget: &MemoryBudget, context: &str) {
    assert_eq!(
        (
            budget.reserved_bytes(),
            budget.overcommit_bytes(),
            budget.waiters()
        ),
        (0, 0, 0),
        "{context}: every reservation and top-up must be given back, and nothing left queued"
    );
}

/// The allowances the stores here run at are the pod memory plan's
/// (`L / 500`, whole MiB): [`R_REF`] at the smallest limit the chart's
/// configuration allows, 2,709 MiB, and [`R_REF_5_GIB`] at the 5 GiB
/// reference limit. If the plan's retirement term or its minimum moves, this
/// fails rather than leaving the tests below at allowances no pod has.
#[test]
fn the_test_allowances_are_the_pod_memory_plans_at_the_smallest_and_the_reference_pod() {
    use live_edit::config::{PlanInputs, PodMemoryPlan, PodMemorySource};
    let reference = PodMemoryPlan::reference();
    assert_eq!(
        reference.limit().mib(),
        5 * 1024,
        "the reference pod is 5 GiB"
    );
    assert_eq!(R_REF_5_GIB, reference.retirement_budget_bytes());
    let minimum_mib = reference
        .minimum_limit_mib()
        .expect("the chart's configuration has a minimum limit");
    assert_eq!(minimum_mib, 2_709, "the smallest pod this module names");
    let smallest = PodMemoryPlan::derive(
        PodMemorySource::Fixed(minimum_mib * 1024 * 1024).resolve(),
        &PlanInputs::reference(),
    )
    .expect("the plan closes at its minimum limit");
    assert_eq!(R_REF, smallest.retirement_budget_bytes());
}

// ═════════════════════════════════════════════════════════════════════════════
// D4-T3 — the bound, FIFO, and every exit giving the reservation back
// ═════════════════════════════════════════════════════════════════════════════

/// D4-T3: retirements on every route at once — four `DeleteLayer`s, two
/// overwrites' call bodies, two sweep tail arms (replaced objects stranded by
/// failed wake-ups) and two generic-arm orphans, the sweep's grace at 0 —
/// each with 1,000 indexed references, so each reserves a row per size read
/// in flight (`W` here) + 1,000 KiB and no two fit side by side: they queue
/// for one another. A sampler on the allowance sees
/// `reserved + overcommit` never above it (nothing here clamps, and every
/// reference is counted, so nothing tops up), and sees something reserved;
/// every retired object's keys are gone; and the allowance holds and queues
/// nothing at the end.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn concurrent_retirements_on_every_route_hold_at_most_the_allowance_and_give_it_all_back() {
    const REFS: u32 = 1_000;
    let rig = Rig::new("rref-concurrent").await;
    let ls = Arc::new(rig.store().with_orphan_sweep_grace(Duration::ZERO));
    let budget = Arc::clone(ls.retirement_budget());

    let mut deletes = Vec::new();
    for i in 0..4u32 {
        let name = format!("delete-{i}");
        deletes.push((name.clone(), rig.layer(&ls, &name, 100 + i, REFS).await));
    }
    let mut replaced = Vec::new();
    for i in 0..2u32 {
        let name = format!("overwritten-{i}");
        replaced.push(rig.layer(&ls, &name, 200 + i, REFS).await);
    }
    let stranded: Vec<u64> = stranded_overwrites(
        &rig,
        &[("stranded-0", 300, REFS), ("stranded-1", 302, REFS)],
    )
    .await
    .into_iter()
    .map(|(old, _, _)| old)
    .collect();
    let mut orphans = Vec::new();
    for i in 0..2u32 {
        orphans.push(generic_orphan(&rig, &ls, &format!("orphan-{i}"), 400 + i, REFS).await);
    }

    let sampler = Sampler::start(Arc::clone(&budget));
    let mut tasks = Vec::new();
    for (name, _) in &deletes {
        let (ls, ext, session_id, name) = (
            Arc::clone(&ls),
            rig.ext.clone(),
            rig.session_id.clone(),
            name.clone(),
        );
        tasks.push(tokio::spawn(async move {
            ls.delete_layer(&session_id, &name, ext).await
        }));
    }
    for i in 0..2u32 {
        rig.layer(&ls, &format!("overwritten-{i}"), 500 + i, 0)
            .await;
    }
    let mut every = stranded.clone();
    every.extend(orphans.iter().copied());
    let sweep_store = (*ls).clone();
    rig.sweep_until(sweep_store, "the sweep retired every orphan", || {
        let rig = &rig;
        let every = every.clone();
        async move {
            for oid in every {
                if rig.header_exists(oid).await {
                    return false;
                }
            }
            true
        }
    })
    .await;
    tokio::time::timeout(BUDGET, async {
        for task in tasks {
            task.await
                .expect("the delete task must not panic")
                .expect("every DeleteLayer must succeed");
        }
    })
    .await
    .expect("every DeleteLayer returns");
    let every_object: Vec<u64> = deletes
        .iter()
        .map(|(_, oid)| *oid)
        .chain(replaced.iter().copied())
        .chain(stranded.iter().copied())
        .chain(orphans.iter().copied())
        .collect();
    eventually(
        "every retired object's keys gone and every reservation given back",
        || async {
            if budget.reserved_bytes() != 0 {
                return false;
            }
            for &oid in &every_object {
                if !rig.object_keys(oid).await.is_empty() {
                    return false;
                }
            }
            true
        },
    )
    .await;
    let (held, reserved) = sampler.stop().await;

    assert!(
        held <= R_REF,
        "reserved + overcommit must never exceed the allowance ({R_REF} B); the sampler saw \
         {held} B"
    );
    assert!(
        reserved >= page_bytes(),
        "the sampler must have seen at least one reservation, or the bound above is vacuous; it \
         saw {reserved} B"
    );
    assert_returned(&budget, "after every retirement");
}

/// D4-T3, FIFO: with the allowance held, an oversized `DeleteLayer` A queues
/// first, an oversized `DeleteLayer` B second, and a small reservation C (a
/// bare `W`, taken through the budget itself) third. Released, the allowance
/// goes to A alone (clamped, it is the whole). Once A's release returns A
/// shrinks to its page, and C would fit beside it — but B is ahead of C and needs
/// the whole, so C waits until A has finished and B has been granted, walked
/// and shrunk. A budget that let C jump the queue would grant it beside A's
/// shrunk reservation, before B had released anything: when C is granted, A's
/// object must already be gone and B's references already released.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn retirements_are_granted_in_arrival_order() {
    const OVERSIZED: u32 = 3_000;
    let rig = Rig::new("rref-fifo").await;
    let ls = Arc::new(rig.store());
    let budget = Arc::clone(ls.retirement_budget());
    let a = rig.layer(&ls, "fifo-a", 1, OVERSIZED).await;
    rig.layer(&ls, "fifo-b", 2, OVERSIZED).await;

    let full = budget
        .reserve(R_REF, None)
        .await
        .expect("the test takes the whole allowance");
    let spawn_delete = |name: &'static str| {
        let (ls, ext, session_id) = (Arc::clone(&ls), rig.ext.clone(), rig.session_id.clone());
        tokio::spawn(async move { ls.delete_layer(&session_id, name, ext).await })
    };
    let delete_a = spawn_delete("fifo-a");
    waiters(&budget, 1).await;
    let delete_b = spawn_delete("fifo-b");
    waiters(&budget, 2).await;
    let c = tokio::spawn({
        let budget = Arc::clone(&budget);
        async move { budget.reserve(page_bytes(), None).await }
    });
    waiters(&budget, 3).await;

    drop(full);
    let c_grant = tokio::time::timeout(BUDGET, c)
        .await
        .expect("C must be granted once B has walked")
        .expect("C's task must not panic")
        .expect("C's reservation must be granted");
    assert!(
        rig.object_keys(a).await.is_empty(),
        "A holds its reservation through its teardown, and B (ahead of C) was granted only once \
         A gave it all back, so A is gone when C is granted; A's keys: {:?}",
        rig.object_keys(a).await
    );
    for i in [0, OVERSIZED / 2, OVERSIZED - 1] {
        let mut hash = [0u8; 32];
        hash[12..16].copy_from_slice(&2u32.to_be_bytes());
        hash[28..32].copy_from_slice(&i.to_be_bytes());
        assert!(
            rig.ext.is_pending(&hash),
            "C was queued after B, so it is granted only once B has been granted and its \
             release has returned: B's reference {i} must already be released"
        );
    }
    drop(c_grant);
    for task in [delete_a, delete_b] {
        task.await
            .expect("the delete task must not panic")
            .expect("DeleteLayer must succeed");
    }
    assert_returned(&budget, "after the FIFO run");
}

/// D4-T3: an injected page error ending a `DeleteLayer` (its retry budget at
/// one attempt, the teardown's first index page failing) gives the
/// reservation back.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_retirement_whose_teardown_page_fails_gives_its_reservation_back() {
    let rig = Rig::new("rref-page-error").await;
    let faults = Arc::new(Mutex::new(VecDeque::from([Some(fred::error::Error::new(
        fred::error::ErrorKind::Timeout,
        "injected teardown page failure",
    ))])));
    let ls = rig
        .store()
        .with_update_layer_retry_max_attempts(1)
        .with_retirement_teardown_faults_for_test(Arc::clone(&faults));
    let oid = rig.layer(&ls, "page-error", 1, 10).await;

    let result = ls
        .delete_layer(&rig.session_id, "page-error", rig.ext.clone())
        .await;

    assert!(
        matches!(result, Err(ServiceError::Aborted(_))),
        "fixture: the injected page failure ends the one attempt; got {result:?}"
    );
    assert!(
        faults.lock().unwrap().is_empty(),
        "fixture: the teardown took the fault"
    );
    assert!(
        !rig.header_exists(oid).await,
        "fixture: the failure came after the first UNLINK"
    );
    assert_returned(ls.retirement_budget(), "after the failed DeleteLayer");
}

/// D4-T3: a `DeleteLayer` dropped while its walk is parked (an RPC deadline
/// or a cancel mid-walk) gives its reservation back, its walk's pass
/// included. The walk runs inside the retry loop, after the mapping's `DEL`,
/// so the dropped delete leaves an unmapped object with its header for the
/// orphan sweep.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_delete_dropped_mid_walk_gives_its_reservation_back() {
    let rig = Rig::new("rref-cancel-walk").await;
    let ls = rig.store();
    let oid = rig.layer(&ls, "cancel-walk", 1, 10).await;
    let injector = rig.injector();
    let hold = hold_walk_at_obj_ext(&injector);

    {
        let delete = ls.delete_layer(&rig.session_id, "cancel-walk", rig.ext.clone());
        tokio::pin!(delete);
        tokio::select! {
            result = &mut delete => panic!("the delete must park on its walk; got {result:?}"),
            () = parked(&hold) => {}
        }
        assert_eq!(
            ls.retirement_budget().reserved_bytes(),
            rig.charge(oid, 10).await,
            "fixture: the parked walk holds its reservation, the object's charge"
        );
    }
    hold.release();

    // The walk's pass, on its own thread, holds the reservation too; it is
    // aborted with the dropped future at its next await.
    eventually("the dropped delete's reservation given back", || async {
        ls.retirement_budget().reserved_bytes() == 0
    })
    .await;
    assert_returned(
        ls.retirement_budget(),
        "after the walk's future was dropped",
    );
    assert!(
        rig.mapping("cancel-walk").await.is_none() && rig.header_exists(oid).await,
        "the walk ran after the mapping's DEL, and the dropped delete never reached its \
         teardown: the object is left, unmapped, for the orphan sweep"
    );
    assert_eq!(
        rig.ext.pending_count(),
        0,
        "the dropped walk released nothing"
    );
}

/// D4-T3: `wait_for_cleanup` ends the allowance's waits first
/// (`MemoryBudget::end_waits`). An overwrite body queued behind it (the
/// allowance held by the test) ends at once: the replaced object keeps its
/// header, fenced and unmapped, for the orphan sweep, and no release was begun
/// (no `ext_gc_scheduled`, no carrier). The allowance is not closed: once the
/// held grant goes, a reservation that fits is granted at once, and nothing
/// is left reserved.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn wait_for_cleanup_ends_a_queued_body_and_it_keeps_its_object() {
    let rig = Rig::new("rref-end-waits").await;
    let ls = rig.store();
    let old = rig.layer(&ls, "ending", 1, 10).await;
    let budget = Arc::clone(ls.retirement_budget());
    let full = budget
        .reserve(R_REF, None)
        .await
        .expect("the test takes the whole allowance");
    rig.layer(&ls, "ending", 2, 0).await;
    waiters(&budget, 1).await;

    ls.wait_for_cleanup(BUDGET).await;

    assert!(
        !budget.is_closed() && budget.waits_ended(),
        "the allowance's waits are ended first; the allowance is not closed"
    );
    assert_eq!(budget.waiters(), 0, "the queued body ended");
    assert_header_kept_and_unbegun(&rig, old, "the queued body").await;
    assert_eq!(
        rig.waits("cancelled"),
        1.0,
        "the queued wait ended cancelled"
    );
    drop(full);
    let granted = tokio::time::timeout(BUDGET, budget.reserve(page_bytes(), None))
        .await
        .expect("a reservation that fits never waits")
        .expect("a reservation that fits is granted once the waits are ended");
    assert_eq!(
        rig.waits("immediate"),
        2.0,
        "granted at once, beside the test's own grant of the whole"
    );
    drop(granted);
    assert_returned(&budget, "after the held grant went");
}

/// The replaced object `oid` keeps its header, no release was begun on it
/// (no `ext_gc_scheduled`), and no release carrier was written.
async fn assert_header_kept_and_unbegun(rig: &Rig, oid: u64, context: &str) {
    let keys = rig.tv.key_schema();
    let oid_str = oid.to_string();
    let scheduled: Option<String> = rig
        .tv
        .client()
        .pool
        .hget(keys.obj_header(&oid_str), "ext_gc_scheduled")
        .await
        .expect("HGET must succeed");
    assert!(
        rig.header_exists(oid).await && scheduled.is_none(),
        "{context}: the replaced object keeps its header for the sweep, and no begin ran"
    );
    assert!(
        !rig.exists(&keys.obj_ext_retire_carrier(&oid_str)).await,
        "{context}: no release carrier was written"
    );
}

/// A store whose overwrite bodies park on `gate` before they touch anything,
/// an object with 10 indexed references under `layer` replaced on it, and a
/// hold on the replaced object's size read armed while its body is parked —
/// so the body is spawned and tracked but has not reached the allowance.
/// Returns the store, the replaced object, and the hold once the body is
/// parked on it.
async fn body_parked_in_its_size_read(
    rig: &Rig,
    layer: &str,
) -> (LayerStore, u64, Arc<SnapshotReadHold>) {
    let gate = Arc::new(tokio::sync::Barrier::new(2));
    let ls = rig
        .store()
        .with_overwrite_cleanup_gate_for_test(Arc::clone(&gate));
    let old = rig.layer(&ls, layer, 1, 10).await;
    rig.layer(&ls, layer, 2, 0).await;
    let hold = rig
        .injector()
        .push_hold_at(ObjectSnapshotReadFaultTarget::next_read());
    tokio::time::timeout(BUDGET, gate.wait())
        .await
        .expect("the body is parked on its gate");
    parked(&hold).await;
    (ls, old, hold)
}

/// D4-T3, a body that reaches the allowance after shutdown began and fits:
/// parked in its size read when `wait_for_cleanup` ends the allowance's
/// waits, then released, it is granted at once (`immediate`) and runs within
/// the drain — every `obj:*` key of the replaced object gone when the drain
/// returns — and the allowance is back to zero. Fails on a shutdown that
/// closes the allowance, which refuses even a reservation that fits.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_body_reaching_the_allowance_after_shutdown_began_that_fits_completes_within_the_drain() {
    let rig = Rig::new("rref-late-fits").await;
    let (ls, old, hold) = body_parked_in_its_size_read(&rig, "late-fits").await;
    let ls = Arc::new(ls);
    let budget = Arc::clone(ls.retirement_budget());

    let drain = tokio::spawn({
        let ls = Arc::clone(&ls);
        async move { ls.wait_for_cleanup(BUDGET).await }
    });
    eventually("shutdown reached the allowance", || async {
        budget.waits_ended() || budget.is_closed()
    })
    .await;
    hold.release();
    tokio::time::timeout(BUDGET, drain)
        .await
        .expect("the drain returns")
        .expect("the drain task must not panic");

    assert_eq!(
        rig.object_keys(old).await,
        Vec::<String>::new(),
        "the body ran within the drain: every key of the replaced object is gone when it returns"
    );
    assert_eq!(
        (
            rig.waits("immediate"),
            rig.waits("waited"),
            rig.waits("cancelled")
        ),
        (1.0, 0.0, 0.0),
        "the body was granted at once"
    );
    assert_returned(&budget, "after the drain");
}

/// D4-T3, a body that reaches the allowance after shutdown began and would
/// have to wait (the allowance held full by the test): it ends at once,
/// never queued — the drain returns within `BUDGET / 2` while the test still
/// holds the allowance, the wait is recorded `cancelled` and never `waited`,
/// and the replaced object keeps its header with no release begun. Fails on
/// a shutdown that ends only the waits it finds, which lets this body queue
/// and hold the drain until its timeout.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_body_reaching_the_allowance_after_shutdown_began_that_would_wait_ends_at_once() {
    let rig = Rig::new("rref-late-waits").await;
    let (ls, old, hold) = body_parked_in_its_size_read(&rig, "late-waits").await;
    let ls = Arc::new(ls);
    let budget = Arc::clone(ls.retirement_budget());
    let full = budget
        .reserve(R_REF, None)
        .await
        .expect("the test takes the whole allowance");

    let drain = tokio::spawn({
        let ls = Arc::clone(&ls);
        async move { ls.wait_for_cleanup(BUDGET).await }
    });
    eventually("shutdown reached the allowance", || async {
        budget.waits_ended() || budget.is_closed()
    })
    .await;
    hold.release();
    tokio::time::timeout(BUDGET / 2, drain)
        .await
        .expect(
            "the body that would wait ends at once, so the drain returns well inside its timeout",
        )
        .expect("the drain task must not panic");

    assert_eq!(budget.waiters(), 0, "the body never queued");
    assert_eq!(
        (rig.waits("waited"), rig.waits("cancelled")),
        (0.0, 1.0),
        "the body's reservation was refused at once"
    );
    assert_header_kept_and_unbegun(&rig, old, "the late body").await;
    drop(full);
    assert_returned(&budget, "after the held grant went");
}

/// D4-T3: an overwrite body whose wake-up of the old stream never lands (the
/// tail's attempt and the body's three retries all fail) reserves nothing:
/// once the body has ended (the store's drain returns), no wait is recorded
/// on the allowance, nothing is held, and the replaced object is left for the
/// sweep. Were the reservation taken before the wake-ups, or after the last
/// one failed, it would have been recorded by the time the body ended.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_overwrite_body_whose_wake_up_never_lands_reserves_nothing() {
    let rig = Rig::new("rref-no-wake").await;
    let (old, _, ls) = stranded_overwrites(&rig, &[("never-woken", 1, 10)])
        .await
        .remove(0);
    tokio::time::timeout(BUDGET, ls.wait_for_cleanup(BUDGET * 2))
        .await
        .expect("the body ends once its last wake-up failed");

    for outcome in ["immediate", "waited", "cancelled"] {
        assert_eq!(
            rig.waits(outcome),
            0.0,
            "outcome={outcome}: a body whose wake-up never lands must not reserve"
        );
    }
    assert_returned(ls.retirement_budget(), "after the body ended");
    assert!(rig.header_exists(old).await);
}

// ═════════════════════════════════════════════════════════════════════════════
// D4-T4 — a deadline while waiting changes nothing
// ═════════════════════════════════════════════════════════════════════════════

/// D4-T4: a `DeleteLayer` whose deadline passes while it waits for the
/// allowance (its future dropped once it is queued, as the RPC deadline drops
/// it) changes nothing: the mapping, the layers-index member and the object
/// stand; on this replica the object's admission entry stands, the admission
/// map is the same size, and a structural-cache checkout taken before the
/// delete is put back `Stored` — the guard that retires both is installed
/// only once the delete enters its retry loop (premise-check amendment). The
/// wait is recorded `cancelled`, and once the allowance frees a retry
/// deletes the layer.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_delete_whose_deadline_passes_while_it_waits_changes_nothing_and_a_retry_works() {
    let rig = Rig::new("rref-deadline").await;
    let ls = rig.store();
    let oid = rig.layer(&ls, "deadline", 1, 10).await;
    ls.update_layer_relayed(
        &rig.session_id,
        "deadline",
        oid,
        Some(delta_with_set_field_inline(1, "k", vec![1], 1)),
        "deadline-writer",
        1_000,
    )
    .await
    .expect("an update must commit, so this replica holds the object's admission entry");
    let entries_before = ls.semaphore_count();
    assert!(
        ls.admission_permits_available_for_test(oid).is_some(),
        "fixture: the admission entry exists"
    );
    let straggler = hold_checkout(&ls, oid);
    let budget = Arc::clone(ls.retirement_budget());
    let full = budget
        .reserve(R_REF, None)
        .await
        .expect("the test takes the whole allowance");

    {
        let delete = ls.delete_layer(&rig.session_id, "deadline", rig.ext.clone());
        tokio::pin!(delete);
        tokio::select! {
            result = &mut delete => panic!("the delete must wait for the allowance; got {result:?}"),
            () = waiters(&budget, 1) => {}
        }
    }
    assert_eq!(budget.waiters(), 0, "the dropped wait left the queue");

    assert_eq!(
        rig.mapping("deadline").await,
        Some(oid.to_string()),
        "the mapping stands"
    );
    assert!(
        rig.indexed("deadline").await,
        "the layers-index member stands"
    );
    assert!(rig.header_exists(oid).await, "the object stands");
    assert!(
        ls.admission_permits_available_for_test(oid).is_some(),
        "this replica's admission entry for the live object stands"
    );
    assert_eq!(ls.semaphore_count(), entries_before);
    assert!(
        straggler.put_back(entry(oid, 2)) == PutBack::Stored,
        "this replica's structural-cache entry stands: a checkout taken before the delete is put \
         back"
    );
    assert_eq!(rig.waits("cancelled"), 1.0, "the wait ended cancelled");

    drop(full);
    ls.delete_layer(&rig.session_id, "deadline", rig.ext.clone())
        .await
        .expect("a retry of the delete must succeed");
    assert!(rig.mapping("deadline").await.is_none());
    rig.until_object_gone(oid).await;
    assert_returned(&budget, "after the retry");
}

/// An update on `layer`'s object `oid` through `ls`, so `ls` holds the
/// object's admission entry; and a structural-cache checkout of it held as a
/// straggler. Returns the admission map's size and the checkout.
async fn admitted_and_cached(
    rig: &Rig,
    ls: &LayerStore,
    layer: &str,
    oid: u64,
) -> (usize, CheckedOut) {
    ls.update_layer_relayed(
        &rig.session_id,
        layer,
        oid,
        Some(delta_with_set_field_inline(1, "k", vec![1], 1)),
        "admitted-writer",
        1_000,
    )
    .await
    .expect("an update must commit, so this replica holds the object's admission entry");
    assert!(
        ls.admission_permits_available_for_test(oid).is_some(),
        "fixture: the admission entry exists"
    );
    (ls.semaphore_count(), hold_checkout(ls, oid))
}

/// D4-T4's sibling exit (I-D4e): a `DeleteLayer` that reaches the allowance
/// once shutdown has ended its waits, and would have to wait (the allowance
/// held full), answers `UNAVAILABLE` with the `SERVER_SHUTTING_DOWN` retry
/// hint, never queued and with nothing changed, and — like every exit before
/// its retry loop — leaves this replica's admission entry and
/// structural-cache entry for the live object.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_delete_at_shutdown_that_would_wait_answers_unavailable_and_retires_nothing() {
    let rig = Rig::new("rref-shutdown-waits").await;
    let ls = rig.store();
    let oid = rig.layer(&ls, "shutdown", 1, 3).await;
    let (entries_before, straggler) = admitted_and_cached(&rig, &ls, "shutdown", oid).await;
    let budget = Arc::clone(ls.retirement_budget());
    let full = budget
        .reserve(R_REF, None)
        .await
        .expect("the test takes the whole allowance");
    budget.end_waits();

    let result = tokio::time::timeout(
        BUDGET,
        ls.delete_layer(&rig.session_id, "shutdown", rig.ext.clone()),
    )
    .await
    .expect("a delete that would wait is refused at once, never queued");

    match &result {
        Err(ServiceError::UnavailableWithDetail { detail, .. }) => assert_eq!(
            **detail,
            live_edit::error::RefusalDetail::server_shutting_down(),
            "the refusal carries the shutdown retry hint"
        ),
        other => panic!(
            "a delete that would wait at shutdown answers UNAVAILABLE with \
             SERVER_SHUTTING_DOWN; got {other:?}"
        ),
    }
    assert_eq!(budget.waiters(), 0, "it never queued");
    assert_eq!(rig.mapping("shutdown").await, Some(oid.to_string()));
    assert!(rig.indexed("shutdown").await && rig.header_exists(oid).await);
    assert!(
        ls.admission_permits_available_for_test(oid).is_some(),
        "this replica's admission entry stands"
    );
    assert_eq!(ls.semaphore_count(), entries_before);
    assert!(
        straggler.put_back(entry(oid, 2)) == PutBack::Stored,
        "this replica's structural-cache entry stands"
    );
    drop(full);
    assert_returned(&budget, "after the held grant went");
}

/// I-D4e's other side: a `DeleteLayer` at shutdown (the allowance's waits
/// ended) whose reservation fits is granted at once and deletes the layer.
/// Fails on a shutdown that closes the allowance.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_delete_at_shutdown_that_fits_deletes_the_layer() {
    let rig = Rig::new("rref-shutdown-fits").await;
    let ls = rig.store();
    let oid = rig.layer(&ls, "shutdown-fits", 1, 3).await;
    let budget = Arc::clone(ls.retirement_budget());
    budget.end_waits();

    ls.delete_layer(&rig.session_id, "shutdown-fits", rig.ext.clone())
        .await
        .expect("a delete whose reservation fits runs at shutdown");

    assert!(rig.mapping("shutdown-fits").await.is_none() && !rig.indexed("shutdown-fits").await);
    assert_eq!(
        rig.object_keys(oid).await,
        Vec::<String>::new(),
        "the object is gone"
    );
    assert_eq!(rig.waits("immediate"), 1.0, "granted at once");
    assert_eq!(rig.ext.pending_count(), 3, "its references were released");
    assert_returned(&budget, "after the delete");
}

// ═════════════════════════════════════════════════════════════════════════════
// D4-T5, D4-T6 — the clamp and the top-up
// ═════════════════════════════════════════════════════════════════════════════

/// D4-T5: a retirement whose charge exceeds the allowance (3,000 references:
/// a row per size read in flight, 2.5 MiB, + 2.9 MiB against 5 MiB) is
/// clamped to it and runs
/// alone: parked mid-walk it holds exactly the whole allowance, with no
/// over-commit (its walk's cover never takes the grant and top-up past the
/// allowance), and a second retirement, which would fit an empty allowance,
/// waits for it.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_oversized_retirement_is_clamped_to_the_allowance_and_runs_alone() {
    let rig = Rig::new("rref-clamp").await;
    let ls = Arc::new(rig.store());
    let budget = Arc::clone(ls.retirement_budget());
    let big = rig.layer(&ls, "big", 1, 3_000).await;
    let small = rig.layer(&ls, "small", 2, 0).await;
    let injector = rig.injector();
    let hold = hold_walk_at_obj_ext(&injector);
    let spawn_delete = |name: &'static str| {
        let (ls, ext, session_id) = (Arc::clone(&ls), rig.ext.clone(), rig.session_id.clone());
        tokio::spawn(async move { ls.delete_layer(&session_id, name, ext).await })
    };

    let delete_big = spawn_delete("big");
    parked(&hold).await;
    assert_eq!(
        (budget.reserved_bytes(), budget.overcommit_bytes()),
        (R_REF, 0),
        "the oversized reservation is clamped to the whole allowance, with nothing beside it"
    );
    let delete_small = spawn_delete("small");
    waiters(&budget, 1).await;
    assert!(
        rig.header_exists(small).await,
        "the second retirement waits for the clamped one"
    );

    hold.release();
    for task in [delete_big, delete_small] {
        task.await
            .expect("the delete task must not panic")
            .expect("DeleteLayer must succeed");
    }
    until_objects_gone(&rig, &[big, small]).await;
    assert_returned(&budget, "after both deletes");
}

/// D4-T6: an object whose `obj:ext` is gone (a pre-b.qpf object) sizes
/// `H` = 0 and, its 100 rows past `P`, reserves the full page `W` alone; its
/// walk finds 100 external rows and covers them with a forced top-up of `h`
/// each (its page already holds a row per size read in flight), held beside
/// the grant while the walk runs and repaid when the reservation shrinks to
/// its page.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_retirement_with_no_obj_ext_tops_up_for_the_references_its_walk_finds() {
    use fred::interfaces::SortedSetsInterface;
    const ROWS: u32 = 100;
    let rig = Rig::new("rref-top-up").await;
    let ls = Arc::new(rig.store());
    let budget = Arc::clone(ls.retirement_budget());
    let oid = rig.layer(&ls, "top-up", 1, 0).await;
    let (keys, oid_str, pool) = (rig.tv.key_schema(), oid.to_string(), &rig.tv.client().pool);
    for i in 0..ROWS {
        let name = format!("ext{i}");
        let _: () = pool
            .hset(
                keys.obj_field(&oid_str, "1", &name),
                [
                    ("hash", ref_hex(1, i)),
                    ("size", "16".to_owned()),
                    ("set_order", (i + 1).to_string()),
                    ("version", "1".to_owned()),
                ],
            )
            .await
            .expect("HSET of an external row must succeed");
        let _: i64 = pool
            .zadd(
                keys.obj_fields_index(&oid_str),
                None,
                None,
                false,
                false,
                (f64::from(i), keys::KeySchema::field_member("1", &name)),
            )
            .await
            .expect("ZADD of the row's member must succeed");
    }
    let injector = rig.injector();
    let hold = hold_walk_at_obj_ext(&injector);

    let delete = tokio::spawn({
        let (ls, ext, session_id) = (Arc::clone(&ls), rig.ext.clone(), rig.session_id.clone());
        async move { ls.delete_layer(&session_id, "top-up", ext).await }
    });
    parked(&hold).await;
    assert_eq!(
        (budget.reserved_bytes(), budget.overcommit_bytes()),
        (
            page_bytes(),
            u64::from(ROWS) * RETIREMENT_REFERENCE_CHARGE_BYTES
        ),
        "H = 0 reserves W alone; the references the rows name are covered by a top-up of h each"
    );
    hold.release();
    delete
        .await
        .expect("the delete task must not panic")
        .expect("DeleteLayer must succeed");
    for i in 0..ROWS {
        let mut hash = [0u8; 32];
        hash[12..16].copy_from_slice(&1u32.to_be_bytes());
        hash[28..32].copy_from_slice(&i.to_be_bytes());
        assert!(
            rig.ext.is_pending(&hash),
            "reference {i} the walk found is released"
        );
    }
    assert_returned(&budget, "after the delete");
}

// ═════════════════════════════════════════════════════════════════════════════
// F3 — the reservation follows the object
// ═════════════════════════════════════════════════════════════════════════════

/// A `DeleteLayer` of a layer owing a handful of references (5) and one of a
/// layer owing 1,000, on a store at an allowance of `r_ref`: both are granted
/// at once and park mid-walk side by side, holding exactly their two charges
/// and queueing nothing; then both finish and give everything back. Each
/// reserved the full page bound whatever its object before the reservation
/// followed the object, so the two never fitted together at either allowance
/// (`W` + 5 KiB beside `W` + 1,000 KiB).
async fn a_handful_runs_beside_a_thousand(label: &str, r_ref: u64) {
    let rig = Rig::new(label).await;
    let ls = Arc::new(rig.store_at(r_ref));
    let budget = Arc::clone(ls.retirement_budget());
    let large = rig.layer(&ls, "large", 1, 1_000).await;
    let handful = rig.layer(&ls, "handful", 2, 5).await;
    let both = rig.charge_at(r_ref, large, 1_000).await + rig.charge_at(r_ref, handful, 5).await;
    assert!(
        both <= r_ref,
        "fixture: the two charges fit the allowance ({both} B against {r_ref} B)"
    );
    let injector = rig.injector();
    let (hold_large, hold_handful) = (
        hold_walk_at_obj_ext(&injector),
        hold_walk_at_obj_ext(&injector),
    );
    let spawn_delete = |name: &'static str| {
        let (ls, ext, session_id) = (Arc::clone(&ls), rig.ext.clone(), rig.session_id.clone());
        tokio::spawn(async move { ls.delete_layer(&session_id, name, ext).await })
    };

    let delete_large = spawn_delete("large");
    parked(&hold_large).await;
    let delete_handful = spawn_delete("handful");
    parked(&hold_handful).await;
    assert_eq!(
        (
            budget.reserved_bytes(),
            budget.overcommit_bytes(),
            budget.waiters()
        ),
        (both, 0, 0),
        "both retirements hold their own charge at once, and neither waits"
    );

    hold_large.release();
    hold_handful.release();
    for task in [delete_large, delete_handful] {
        task.await
            .expect("the delete task must not panic")
            .expect("DeleteLayer must succeed");
    }
    until_objects_gone(&rig, &[large, handful]).await;
    assert_returned(&budget, "after both deletes");
}

/// F3, at the default 5 GiB pod's allowance (`R_ref` = 10 MiB): a clean-up
/// owing a handful of references runs beside one owing 1,000.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_clean_up_with_a_handful_of_large_values_runs_beside_a_large_one_at_the_5_gib_pods_allowance(
) {
    a_handful_runs_beside_a_thousand("rref-beside-5gib", R_REF_5_GIB).await;
}

/// F3, at the smallest pod's allowance (2,709 MiB, `R_ref` = 5 MiB): the
/// same.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_clean_up_with_a_handful_of_large_values_runs_beside_a_large_one_at_the_smallest_pods_allowance(
) {
    a_handful_runs_beside_a_thousand("rref-beside-minimum", R_REF).await;
}

/// F3, the estimate low: a `DeleteLayer` sizes its retirement before its
/// fence, so an index can grow between its size read and its walk. Here the
/// delete sizes an empty object (its page one member past its largest index)
/// and queues behind the test's grant of the whole allowance; meanwhile
/// three full pages of inline field rows are indexed. Granted, its walk's
/// first page comes back full at the object's page, and before reading the
/// next it widens to the full page: parked before its second `obj:flds`
/// `ZRANGE` page it already holds its grant (the object's page) and a forced
/// top-up of exactly the rest of `W`, and it still does at its first
/// `HSCAN obj:ext` step. A walk that read past the object's page on its grant
/// alone, or widened only after paging the index, would show no top-up at the
/// first park. The delete then finishes, every row removed, and gives
/// everything back.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_delete_whose_index_grew_after_its_size_read_widens_its_page_before_reading_past_it() {
    use fred::interfaces::SortedSetsInterface;
    let rig = Rig::new("rref-widen").await;
    let ls = Arc::new(rig.store());
    let budget = Arc::clone(ls.retirement_budget());
    let oid = rig.layer(&ls, "grown", 1, 0).await;
    let sized = rig.charge(oid, 0).await;
    assert!(
        sized < page_bytes(),
        "fixture: the object's page is below the full page ({sized} B)"
    );

    let full = budget
        .reserve(R_REF, None)
        .await
        .expect("the test takes the whole allowance");
    let delete = tokio::spawn({
        let (ls, ext, session_id) = (Arc::clone(&ls), rig.ext.clone(), rig.session_id.clone());
        async move { ls.delete_layer(&session_id, "grown", ext).await }
    });
    waiters(&budget, 1).await;

    let rows = 3 * RetirementPaging::for_allowance(R_REF).page_len();
    let (keys, oid_str, pool) = (rig.tv.key_schema(), oid.to_string(), &rig.tv.client().pool);
    for i in 0..rows {
        let name = format!("grown{i}");
        let _: () = pool
            .hset(
                keys.obj_field(&oid_str, "1", &name),
                [
                    ("value", "x".to_owned()),
                    ("set_order", (i + 1).to_string()),
                    ("version", "1".to_owned()),
                ],
            )
            .await
            .expect("HSET of an inline row must succeed");
        let _: i64 = pool
            .zadd(
                keys.obj_fields_index(&oid_str),
                None,
                None,
                false,
                false,
                (i as f64, keys::KeySchema::field_member("1", &name)),
            )
            .await
            .expect("ZADD of the row's member must succeed");
    }
    let injector = rig.injector();
    let second_page = injector.push_hold_at(ObjectSnapshotReadFaultTarget::nth_of_family(
        ObjectSnapshotReader::UnfencedFanout,
        "fields_index",
        1,
    ));
    let hold = hold_walk_at_obj_ext(&injector);
    drop(full);
    parked(&second_page).await;
    assert_eq!(
        (budget.reserved_bytes(), budget.overcommit_bytes()),
        (sized, page_bytes() - sized),
        "parked before its second obj:flds page, the walk already holds the full page: it \
         widened before reading past the object's page"
    );
    second_page.release();
    parked(&hold).await;
    assert_eq!(
        (budget.reserved_bytes(), budget.overcommit_bytes()),
        (sized, page_bytes() - sized),
        "the walk keeps the full page to its end"
    );

    hold.release();
    delete
        .await
        .expect("the delete task must not panic")
        .expect("DeleteLayer must succeed");
    assert_eq!(
        rig.object_keys(oid).await,
        Vec::<String>::new(),
        "every grown row was torn down"
    );
    assert_returned(&budget, "after the delete");
}

// ═════════════════════════════════════════════════════════════════════════════
// D4-T11 — the delete-vs-replace retarget
// ═════════════════════════════════════════════════════════════════════════════

/// D4-T11, one replica: a `DeleteLayer` of object X waits for the allowance,
/// and meanwhile the layer is replaced by Y on the same store. Once granted,
/// the delete re-reads the mapping, retargets, and deletes Y (policy 9: a
/// delete whose wait let a replacement commit deletes the replacement). The
/// layer is gone, Y is gone, X is cleaned up by its overwrite body, and the
/// store holds no admission entry or cache entry for either.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_delete_whose_layer_is_replaced_while_it_waits_deletes_the_replacement_on_this_replica() {
    let rig = Rig::new("rref-retarget").await;
    let ls = Arc::new(rig.store());
    let budget = Arc::clone(ls.retirement_budget());
    let x = rig.layer(&ls, "replaced", 1, 10).await;
    let full = budget
        .reserve(R_REF, None)
        .await
        .expect("the test takes the whole allowance");

    let delete = tokio::spawn({
        let (ls, ext, session_id) = (Arc::clone(&ls), rig.ext.clone(), rig.session_id.clone());
        async move { ls.delete_layer(&session_id, "replaced", ext).await }
    });
    waiters(&budget, 1).await;
    let y = rig.layer(&ls, "replaced", 2, 10).await;
    assert_ne!(x, y);
    drop(full);

    delete
        .await
        .expect("the delete task must not panic")
        .expect("the delete must succeed, retargeted onto the replacement");
    assert!(rig.mapping("replaced").await.is_none(), "the layer is gone");
    assert!(!rig.indexed("replaced").await);
    until_objects_gone(&rig, &[y, x]).await;
    assert_eq!(ls.semaphore_count(), 0, "no admission entry for X or Y");
    assert_retired(&ls, x, "the replaced object");
    assert_retired(&ls, y, "the retargeted replacement");
    eventually("every reservation given back", || async {
        budget.reserved_bytes() == 0
    })
    .await;
}

/// D4-T11, the cross-replica row (premise-check amendment; the shape of
/// `b_v6d_tests::create_resend_redrive_retires_the_replaced_objects_admission_and_cache_entries`):
/// replica A holds X's admission entry (an update) and a cached entry for X;
/// A's `DeleteLayer` waits for A's allowance; replica B replaces the layer
/// with Y, and B's own overwrite guard retires X's entries on B only. Once A
/// is granted it re-reads the mapping and retargets: it must retire X's
/// admission entry and structural-cache entry on A at once (the one-shot
/// guard drop), then delete Y, after which A holds no entry for Y either.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_delete_whose_layer_is_replaced_on_another_replica_retires_the_old_objects_entries_here()
{
    let rig = Rig::new("rref-retarget-cross").await;
    let replica_a = Arc::new(rig.store());
    let replica_b = rig.store();
    let budget_a = Arc::clone(replica_a.retirement_budget());
    let x = rig.layer(&replica_a, "shared", 1, 10).await;
    replica_a
        .update_layer_relayed(
            &rig.session_id,
            "shared",
            x,
            Some(delta_with_set_field_inline(1, "k", vec![1], 1)),
            "replica-a-writer",
            1_000,
        )
        .await
        .expect("an update on A must commit, so A holds X's admission entry");
    cache_entry_for(&replica_a, x);
    assert!(
        replica_a.admission_permits_available_for_test(x).is_some(),
        "fixture: A holds X's admission entry"
    );
    let full = budget_a
        .reserve(R_REF, None)
        .await
        .expect("the test takes A's whole allowance");

    let delete = tokio::spawn({
        let (ls, ext, session_id) = (
            Arc::clone(&replica_a),
            rig.ext.clone(),
            rig.session_id.clone(),
        );
        async move { ls.delete_layer(&session_id, "shared", ext).await }
    });
    waiters(&budget_a, 1).await;
    let y = rig.layer(&replica_b, "shared", 2, 10).await;
    assert!(
        replica_a.admission_permits_available_for_test(x).is_some(),
        "fixture: B's overwrite retired X's entries on B only"
    );
    drop(full);

    delete
        .await
        .expect("the delete task must not panic")
        .expect("A's delete must succeed, retargeted onto Y");
    assert!(
        replica_a.admission_permits_available_for_test(x).is_none(),
        "A's retarget must retire X's admission entry on A: no writer resolves the layer to X \
         any more"
    );
    assert_retired(&replica_a, x, "X on A, after the retarget");
    assert_eq!(
        replica_a.semaphore_count(),
        0,
        "after deleting Y, A holds no admission entry for it either"
    );
    assert_retired(&replica_a, y, "Y on A, after the delete");
    assert!(rig.mapping("shared").await.is_none(), "the layer is gone");
    until_objects_gone(&rig, &[y, x]).await;
}

// ═════════════════════════════════════════════════════════════════════════════
// D4-T12, D4-T13, D4-T14 — the overwrite and sweep routes
// ═════════════════════════════════════════════════════════════════════════════

/// D4-T12: an overwrite body's wake-up is never queued behind the allowance.
/// The tail's inline wake-up fails (injected), so the body retries it; with
/// the allowance held by the test, by the time the body is queued for its
/// reservation the old stream already carries its `deleted` entry. A body
/// that reserved first would sit in the queue with the stream unwoken, its
/// subscribers waiting on memory.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_overwrite_body_wakes_its_stream_before_it_queues_for_the_allowance() {
    let rig = Rig::new("rref-wake-first").await;
    let wakes = failing_wakes(1);
    let ls = rig
        .store()
        .with_create_layer_overwrite_xadd_fault_injector_for_test(Arc::clone(&wakes));
    let budget = Arc::clone(ls.retirement_budget());
    let old = rig.layer(&ls, "woken", 1, 10).await;
    let full = budget
        .reserve(R_REF, None)
        .await
        .expect("the test takes the whole allowance");

    rig.layer(&ls, "woken", 2, 0).await;
    assert!(
        wakes.lock().unwrap().is_empty(),
        "fixture: the inline wake-up took the failure"
    );
    waiters(&budget, 1).await;
    assert_eq!(
        rig.deleted_entries(old).await,
        1,
        "the body's retried wake-up landed before it queued for memory"
    );

    drop(full);
    rig.until_object_gone(old).await;
    eventually("the body gave its reservation back", || async {
        budget.reserved_bytes() == 0
    })
    .await;
}

/// D4-T13: the orphan sweep awaits each retirement: two replaced objects
/// stranded for it (their wake-ups failed) run through the shared cleanup
/// body one at a time — the `OverwriteCleanupProbe`, which holds each body
/// 100 ms before it does anything so that overlapping bodies would meet in
/// it, sees both enter and never two at once — and the sweep's allowance
/// never holds more than one reservation (each the object's charge, owing 10
/// references).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn the_orphan_sweep_runs_one_retirement_at_a_time() {
    const REFS: u32 = 10;
    let rig = Rig::new("rref-sweep-serial").await;
    let stranded =
        stranded_overwrites(&rig, &[("stranded-a", 1, REFS), ("stranded-b", 3, REFS)]).await;
    let (first, second) = (stranded[0].0, stranded[1].0);
    // Each object's charge, read while its indices stand: once the sweep has
    // retired it, its indices read empty.
    let one = rig
        .charge(first, REFS)
        .await
        .max(rig.charge(second, REFS).await);
    let probe = Arc::new(OverwriteCleanupProbe::new(Duration::from_millis(100)));
    let sweep = rig
        .store()
        .with_orphan_sweep_grace(Duration::ZERO)
        .with_overwrite_cleanup_probe_for_test(Arc::clone(&probe));
    let sampler = Sampler::start(Arc::clone(sweep.retirement_budget()));

    rig.sweep_until(sweep, "the sweep retired both replaced objects", || async {
        !rig.header_exists(first).await && !rig.header_exists(second).await
    })
    .await;
    let (held, _) = sampler.stop().await;

    assert!(probe.entered() >= 2, "both stranded bodies ran");
    assert_eq!(probe.max_in_flight(), 1, "never two at once");
    assert!(
        held <= one,
        "one reservation at a time ({one} B); the sampler saw {held} B held"
    );
}

/// D4-T14: one replaced object, two runs of its overwrite tail — the call's
/// spawned body, parked mid-walk with its first `obj:ext` page of references
/// already in hand (1,000 references: past the store's compact hash encoding,
/// so `HSCAN` pages them), and the orphan sweep's tail arm, which finds the
/// object stranded meanwhile and retires it whole. Released, the call's body
/// finishes its walk and its begin finds the object gone, so the object's
/// references are released exactly once.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_duplicate_run_of_an_overwrite_tail_pays_its_releases_once() {
    const REFS: u32 = 1_000;
    let rig = Rig::new("rref-duplicate-tail").await;
    let ls = rig.store();
    let old = rig.layer(&ls, "twice", 1, REFS).await;
    let injector = rig.injector();
    let hold = injector.push_hold_at(ObjectSnapshotReadFaultTarget::nth_of_family(
        ObjectSnapshotReader::UnfencedFanout,
        "ext_index",
        1,
    ));

    rig.layer(&ls, "twice", 2, 0).await;
    parked(&hold).await;
    rig.sweep_until(
        rig.store().with_orphan_sweep_grace(Duration::ZERO),
        "the sweep retired the replaced object",
        || async { !rig.header_exists(old).await },
    )
    .await;
    assert_eq!(
        rig.retirement_releases(),
        f64::from(REFS),
        "fixture: the sweep's run paid the references"
    );
    hold.release();
    eventually("the call's body ended", || async {
        ls.retirement_budget().reserved_bytes() == 0
    })
    .await;

    assert_eq!(
        rig.retirement_releases(),
        f64::from(REFS),
        "the second run of the tail must pay nothing: each reference once"
    );
    rig.until_object_gone(old).await;
}

// ═════════════════════════════════════════════════════════════════════════════
// D4-T15 — the size read: a foreign key, and a transient error
// ═════════════════════════════════════════════════════════════════════════════

/// Replace `key` with a string.
pub(crate) async fn make_foreign(rig: &Rig, key: &str) {
    let _: i64 = rig
        .tv
        .client()
        .pool
        .del(key)
        .await
        .expect("DEL must succeed");
    let _: () = rig
        .tv
        .client()
        .pool
        .set(key, "foreign", None, None, false)
        .await
        .expect("SET of the foreign value must succeed");
}

/// Which key the D4-T15 foreign-key tests make foreign, and the family its
/// tick lands under.
#[derive(Debug, Clone, Copy)]
enum Foreign {
    Header,
    ExtIndex,
}

impl Foreign {
    fn key(self, rig: &Rig, oid: u64) -> String {
        match self {
            Foreign::Header => rig.tv.key_schema().obj_header(&oid.to_string()),
            Foreign::ExtIndex => rig.tv.key_schema().obj_ext_index(&oid.to_string()),
        }
    }

    fn family(self) -> &'static str {
        match self {
            Foreign::Header => "header",
            Foreign::ExtIndex => "ext_index",
        }
    }
}

/// Assert exactly `family` ticked once on the type-error counter, and no
/// release was enqueued or recorded for `oid`.
fn assert_counted_once_and_unreleased(rig: &Rig, family: &str, context: &str) {
    for &cell in live_edit::observability::metrics::SNAPSHOT_READ_KEY_FAMILIES {
        assert_eq!(
            rig.type_errors(cell),
            if cell == family { 1.0 } else { 0.0 },
            "{context}: exactly one tick, on key_family={family}; key_family={cell}"
        );
    }
    assert_eq!(
        rig.ext.pending_count(),
        0,
        "{context}: the release is skipped"
    );
    assert_eq!(rig.retirement_releases(), 0.0, "{context}");
}

/// D4-T15, `DeleteLayer`: a foreign `obj:hdr` or `obj:ext` meets the size
/// read, which counts it once (`header` or `ext_index`), WARNs and never
/// propagates it: the delete skips the walk and the release, reserves the
/// full page `W`,
/// answers OK, and its teardown's first `UNLINK` removes the foreign key with
/// the rest of the object (a `DeleteLayer` returns once its teardown has run,
/// so no wait is needed).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_delete_meeting_a_foreign_header_or_obj_ext_counts_it_once_skips_the_release_and_removes_it(
) {
    for foreign in [Foreign::Header, Foreign::ExtIndex] {
        let rig = Rig::new("rref-foreign-delete").await;
        let ls = rig.store();
        let oid = rig.layer(&ls, "foreign", 1, 3).await;
        let key = foreign.key(&rig, oid);
        make_foreign(&rig, &key).await;
        ls.delete_layer(&rig.session_id, "foreign", rig.ext.clone())
            .await
            .unwrap_or_else(|e| panic!("{foreign:?}: DeleteLayer must answer OK; got {e:?}"));
        assert!(
            !rig.exists(&key).await,
            "{foreign:?}: the teardown's first UNLINK removes the foreign key"
        );
        assert_eq!(
            rig.object_keys(oid).await,
            Vec::<String>::new(),
            "{foreign:?}: the object is gone"
        );
        assert_counted_once_and_unreleased(&rig, foreign.family(), &format!("{foreign:?}, delete"));
        assert_eq!(
            rig.waits("immediate"),
            1.0,
            "{foreign:?}: the full page bound alone was reserved"
        );
    }
}

/// D4-T15, the overwrite body: the same, with the key replaced while the body
/// is parked on its gate — counted once, no release, the foreign key removed
/// with the rest of the replaced object. Both bodies run at once, under one
/// bounded wait for their objects.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_overwrite_body_meeting_a_foreign_header_or_obj_ext_counts_it_once_skips_the_release_and_removes_it(
) {
    let mut runs = Vec::new();
    for foreign in [Foreign::Header, Foreign::ExtIndex] {
        let rig = Rig::new("rref-foreign-overwrite").await;
        let gate = Arc::new(tokio::sync::Barrier::new(2));
        let ls = rig
            .store()
            .with_overwrite_cleanup_gate_for_test(Arc::clone(&gate));
        let old = rig.layer(&ls, "foreign", 1, 3).await;
        rig.layer(&ls, "foreign", 2, 0).await;
        let key = foreign.key(&rig, old);
        make_foreign(&rig, &key).await;
        runs.push((foreign, rig, ls, gate, old, key));
    }
    for (_, _, _, gate, _, _) in &runs {
        tokio::time::timeout(BUDGET, gate.wait())
            .await
            .expect("the body is parked on its gate");
    }
    eventually("both replaced objects' keys gone", || async {
        for (_, rig, _, _, old, _) in &runs {
            if !rig.object_keys(*old).await.is_empty() {
                return false;
            }
        }
        true
    })
    .await;
    for (foreign, rig, _, _, _, key) in &runs {
        assert!(
            !rig.exists(key).await,
            "{foreign:?}: the foreign key is removed"
        );
        assert_counted_once_and_unreleased(
            rig,
            foreign.family(),
            &format!("{foreign:?}, overwrite body"),
        );
    }
}

/// D4-T15, the orphan sweep's generic arm: a foreign `obj:ext` (a foreign
/// header ends the sweep's pass at its counted discovery read, before any
/// retirement — `b_muu_tests`) is counted once under `ext_index`, nothing is
/// released, and the orphan is retired with the foreign key.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_sweep_retirement_meeting_a_foreign_obj_ext_counts_it_once_skips_the_release_and_removes_it(
) {
    let rig = Rig::new("rref-foreign-sweep").await;
    let ls = rig.store();
    let oid = generic_orphan(&rig, &ls, "foreign", 1, 3).await;
    let key = Foreign::ExtIndex.key(&rig, oid);
    make_foreign(&rig, &key).await;
    rig.sweep_until(
        rig.store().with_orphan_sweep_grace(Duration::ZERO),
        "the sweep removed every key of the orphan",
        || async { rig.object_keys(oid).await.is_empty() },
    )
    .await;
    assert!(!rig.exists(&key).await, "the foreign obj:ext is removed");
    assert_counted_once_and_unreleased(&rig, "ext_index", "foreign obj:ext, sweep");
}

// ═════════════════════════════════════════════════════════════════════════════
// D4-T15 — the size read: a foreign structural index
// ═════════════════════════════════════════════════════════════════════════════

/// The indexed references every foreign-index retirement owes (`H`): enough
/// that the release has something to schedule, and few enough that the full
/// page's charge differs from the empty object's own page's.
const FOREIGN_INDEX_REFS: u32 = 3;

/// Which of an object's five structural indices a foreign-index test makes
/// foreign, and the family the size read counts it under.
#[derive(Debug, Clone, Copy)]
enum ForeignIndex {
    Sections,
    DeletedSections,
    Children,
    Fields,
    TimeSamples,
}

impl ForeignIndex {
    const ALL: [Self; 5] = [
        Self::Sections,
        Self::DeletedSections,
        Self::Children,
        Self::Fields,
        Self::TimeSamples,
    ];

    fn key(self, rig: &Rig, oid: u64) -> String {
        let (keys, oid) = (rig.tv.key_schema(), oid.to_string());
        match self {
            Self::Sections => keys.obj_sections_index(&oid),
            Self::DeletedSections => keys.obj_deleted_sections_index(&oid),
            Self::Children => keys.obj_children_index(&oid),
            Self::Fields => keys.obj_fields_index(&oid),
            Self::TimeSamples => keys.obj_time_samples_index(&oid),
        }
    }

    fn family(self) -> &'static str {
        match self {
            Self::Sections => "sections_index",
            Self::DeletedSections => "deleted_sections_index",
            Self::Children => "children_index",
            Self::Fields => "fields_index",
            Self::TimeSamples => "time_samples_index",
        }
    }

    /// Whether the reference walk reads this index: it pages the value
    /// indices `obj:flds` and `obj:tss` by `ZRANGE`, and no set index.
    fn walked(self) -> bool {
        matches!(self, Self::Fields | Self::TimeSamples)
    }

    /// The type-error ticks a retirement meeting this index produces under
    /// its family: one at the size read's count, and for a value index one
    /// more at the walk's first `ZRANGE` page of it, which fails the walk.
    /// The teardown's own index pages count nothing.
    fn type_errors(self) -> f64 {
        if self.walked() {
            2.0
        } else {
            1.0
        }
    }
}

/// Queue a hold on the next retirement's reference-walk header `EXISTS`,
/// which the walk issues after the route's reservation is granted. The size
/// read's `HGET` is the route's first `header` command on the seam, so a hold
/// queued ahead of the walk's, already released, is taken there and passed.
fn hold_walk_at_header(injector: &ObjectSnapshotReadFaultInjector) -> Arc<SnapshotReadHold> {
    let header = || {
        ObjectSnapshotReadFaultTarget::nth_of_family(
            ObjectSnapshotReader::UnfencedFanout,
            "header",
            0,
        )
    };
    injector.push_hold_at(header()).release();
    injector.push_hold_at(header())
}

/// Assert what a retirement of `oid` meeting a foreign `index` (its key
/// `key`) left, `reserved` read while its walk was parked on its header:
///
/// - the type-error counter ticked [`ForeignIndex::type_errors`] times under
///   the index's family and nowhere else;
/// - the reservation was the charge at the full page `P`, not the empty
///   object's own one-member page;
/// - for a set index, which the walk does not read, the release was
///   scheduled. For a value index the walk fails on the index before the
///   release begins, so none is recorded or sent: nothing is pending, nothing
///   counted released, and neither the release carrier nor its attempt-token
///   HASH exists (both outlive the teardown). The header, which would carry
///   `ext_gc_scheduled`, is gone by then, so it is not read. The object's
///   references are left to the reference-count reconciler;
/// - the teardown's first `UNLINK` took the header and the teardown stopped
///   at the foreign index, which still stands: its page read is not counted,
///   and once the header is gone nothing finds the object to remove it.
async fn assert_met_foreign_index(
    rig: &Rig,
    injector: &ObjectSnapshotReadFaultInjector,
    index: ForeignIndex,
    oid: u64,
    key: &str,
    reserved: u64,
    context: &str,
) {
    assert_eq!(
        injector.holds_remaining(),
        0,
        "{context}: fixture: the size read and the walk each took their hold"
    );
    for &cell in live_edit::observability::metrics::SNAPSHOT_READ_KEY_FAMILIES {
        assert_eq!(
            rig.type_errors(cell),
            if cell == index.family() {
                index.type_errors()
            } else {
                0.0
            },
            "{context}: the ticks on key_family={cell}"
        );
    }
    let full = RetirementPaging::for_allowance(R_REF);
    let full_charge = retirement_charge_bytes(full.page_len(), u64::from(FOREIGN_INDEX_REFS));
    let own_page_charge =
        retirement_charge_bytes(full.for_rows(1).page_len(), u64::from(FOREIGN_INDEX_REFS));
    assert_ne!(
        full_charge, own_page_charge,
        "fixture: the full page's charge differs from the empty object's own page's"
    );
    assert_eq!(
        reserved, full_charge,
        "{context}: the index could not be counted, so the full page was reserved"
    );
    if index.walked() {
        let oid_str = oid.to_string();
        let keys = rig.tv.key_schema();
        assert_eq!(
            (rig.ext.pending_count(), rig.retirement_releases()),
            (0, 0.0),
            "{context}: the walk failed on the index, so no release was scheduled"
        );
        assert!(
            !rig.exists(&keys.obj_ext_retire_carrier(&oid_str)).await
                && !rig.exists(&keys.obj_ext_retire_attempts(&oid_str)).await,
            "{context}: no release began: no carrier and no attempt token were written"
        );
    } else {
        assert_eq!(
            rig.ext.pending_count(),
            FOREIGN_INDEX_REFS as usize,
            "{context}: the release was scheduled"
        );
    }
    assert!(
        !rig.header_exists(oid).await,
        "{context}: the teardown's first UNLINK took the header"
    );
    assert!(
        rig.exists(key).await,
        "{context}: the teardown stopped at the foreign index, which still stands"
    );
}

/// D4-T15, `DeleteLayer`: a foreign `obj:secs`, `obj:dels`, `obj:chs`,
/// `obj:flds` or `obj:tss` meets the size read, which counts it under the
/// index's own family and WARNs; the delete reserves the full page and goes
/// on past it: the mapping is deleted, and with a set index the release is
/// scheduled (with a value index the walk fails on it, and none is). The
/// teardown then stops at the foreign index after its first `UNLINK` (b.yz5:
/// the index's rows stay behind), so the delete answers `INTERNAL` and never
/// reaches its layers-index `SREM`: the member stays. One delete per index,
/// all at once.
///
/// The `key_family` tick is the assertion a size read that counts no index
/// fails: it reads 0 for a set index there, and 1 for a value index (the
/// walk's alone).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_delete_meeting_a_foreign_structural_index_counts_it_under_its_family_reserves_the_full_page_and_stops_its_teardown_there(
) {
    futures::future::join_all(ForeignIndex::ALL.map(|index| async move {
        let rig = Rig::new("rref-foreign-index-delete").await;
        let ls = rig.store();
        let oid = rig
            .layer(&ls, "foreign-index", 1, FOREIGN_INDEX_REFS)
            .await;
        let key = index.key(&rig, oid);
        make_foreign(&rig, &key).await;
        let injector = rig.injector();
        let walk = hold_walk_at_header(&injector);

        let delete = ls.delete_layer(&rig.session_id, "foreign-index", rig.ext.clone());
        tokio::pin!(delete);
        tokio::select! {
            result = &mut delete => panic!("{index:?}: the delete must park on its walk; got {result:?}"),
            () = parked(&walk) => {}
        }
        let reserved = ls.retirement_budget().reserved_bytes();
        walk.release();
        let answer = delete.await;

        let context = format!("{index:?}, delete");
        assert!(
            matches!(answer, Err(ServiceError::Internal(_))),
            "{context}: the teardown's refused index page answers INTERNAL; got {answer:?}"
        );
        assert_eq!(
            rig.mapping("foreign-index").await,
            None,
            "{context}: the delete went on past its size read and deleted the mapping"
        );
        assert!(
            rig.indexed("foreign-index").await,
            "{context}: the delete stopped before its layers-index SREM, so the member stays"
        );
        assert_met_foreign_index(&rig, &injector, index, oid, &key, reserved, &context).await;
    }))
    .await;
}

/// D4-T15, the overwrite body: the tick, the full page, the release
/// (scheduled for a set index, never begun for a value index) and the
/// teardown's stop, as on `DeleteLayer`, with the replaced object's index
/// made foreign while the body is parked on its gate; the body WARNs the
/// teardown's failure and ends. One body per index, all at once.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_overwrite_body_meeting_a_foreign_structural_index_counts_it_under_its_family_reserves_the_full_page_and_stops_its_teardown_there(
) {
    futures::future::join_all(ForeignIndex::ALL.map(|index| async move {
        let rig = Rig::new("rref-foreign-index-overwrite").await;
        let gate = Arc::new(tokio::sync::Barrier::new(2));
        let ls = rig
            .store()
            .with_overwrite_cleanup_gate_for_test(Arc::clone(&gate));
        let old = rig.layer(&ls, "foreign-index", 1, FOREIGN_INDEX_REFS).await;
        rig.layer(&ls, "foreign-index", 2, 0).await;
        let key = index.key(&rig, old);
        make_foreign(&rig, &key).await;
        let injector = rig.injector();
        let walk = hold_walk_at_header(&injector);

        tokio::time::timeout(BUDGET, gate.wait())
            .await
            .expect("the body is parked on its gate");
        parked(&walk).await;
        let reserved = ls.retirement_budget().reserved_bytes();
        walk.release();
        tokio::time::timeout(BUDGET, ls.wait_for_cleanup(BUDGET * 2))
            .await
            .expect("the body ends");

        let context = format!("{index:?}, overwrite body");
        assert_met_foreign_index(&rig, &injector, index, old, &key, reserved, &context).await;
    }))
    .await;
}

/// D4-T15, the orphan sweep's generic arm: the tick, the full page, the
/// release and the teardown's stop, as on `DeleteLayer`, for an orphan with a
/// numeric id and a header, so its fence lands and it runs the size read.
/// One sweep per index, all at once.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_sweep_retirement_meeting_a_foreign_structural_index_counts_it_under_its_family_reserves_the_full_page_and_stops_its_teardown_there(
) {
    futures::future::join_all(ForeignIndex::ALL.map(|index| async move {
        let rig = Rig::new("rref-foreign-index-sweep").await;
        let ls = rig.store();
        let oid = generic_orphan(&rig, &ls, "foreign-index", 1, FOREIGN_INDEX_REFS).await;
        let key = index.key(&rig, oid);
        make_foreign(&rig, &key).await;
        let injector = rig.injector();
        let walk = hold_walk_at_header(&injector);
        let sweep = rig.store().with_orphan_sweep_grace(Duration::ZERO);
        let budget = Arc::clone(sweep.retirement_budget());

        // Read the reservation once the walk parks, then let it go on.
        let reserved = std::sync::OnceLock::new();
        rig.sweep_until(
            sweep,
            "the walk parked, then the teardown took the header",
            || async {
                if walk.is_parked() {
                    reserved.get_or_init(|| {
                        let held = budget.reserved_bytes();
                        walk.release();
                        held
                    });
                }
                reserved.get().is_some() && !rig.header_exists(oid).await
            },
        )
        .await;
        let reserved = *reserved.get().expect("the walk parked");

        let context = format!("{index:?}, sweep");
        assert_met_foreign_index(&rig, &injector, index, oid, &key, reserved, &context).await;
    }))
    .await;
}

/// The members [`ForeignIndex::seed`] puts in one index: more than one, and
/// fewer than the full page's `P` (64 at [`R_REF`]).
const SEEDED_MEMBERS: u64 = 10;

impl ForeignIndex {
    /// Put `members` members in this index of `oid`: section or parent ids in
    /// a set index, field-style members in a value index (the value indices
    /// are the sorted sets the walk pages).
    async fn seed(self, rig: &Rig, oid: u64, members: u64) {
        use fred::interfaces::SortedSetsInterface;
        let (key, pool) = (self.key(rig, oid), &rig.tv.client().pool);
        for i in 1..=members {
            if self.walked() {
                let _: i64 = pool
                    .zadd(
                        &key,
                        None,
                        None,
                        false,
                        false,
                        (
                            i as f64,
                            keys::KeySchema::field_member("1", &format!("m{i}")),
                        ),
                    )
                    .await
                    .expect("ZADD must succeed");
            } else {
                let _: i64 = pool
                    .sadd(&key, i.to_string())
                    .await
                    .expect("SADD must succeed");
            }
        }
    }
}

/// D4-T15, which counts size the page: an object whose only non-empty
/// structural index is one of the five, holding [`SEEDED_MEMBERS`] members,
/// reserves its page one member past that index —
/// `retirement_charge_bytes(SEEDED_MEMBERS + 1, H)` — whichever index it is,
/// read with the `DeleteLayer`'s walk parked on its header. The delete is
/// then dropped, and its reservation given back.
///
/// The reservation is the assertion a size read that leaves this index out of
/// its largest count fails: it reserves the empty object's one-member page.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_delete_whose_largest_index_is_any_one_of_the_five_reserves_its_page_one_member_past_it()
{
    let full = RetirementPaging::for_allowance(R_REF);
    let refs = u64::from(FOREIGN_INDEX_REFS);
    let seeded_charge = retirement_charge_bytes(SEEDED_MEMBERS + 1, refs);
    assert!(
        SEEDED_MEMBERS + 1 < full.page_len()
            && seeded_charge != retirement_charge_bytes(full.for_rows(1).page_len(), refs),
        "fixture: the seeded page is below the full page and above the empty object's"
    );
    futures::future::join_all(ForeignIndex::ALL.map(|index| async move {
        let rig = Rig::new("rref-largest-index").await;
        let ls = rig.store();
        let oid = rig
            .layer(&ls, "largest-index", 1, FOREIGN_INDEX_REFS)
            .await;
        index.seed(&rig, oid, SEEDED_MEMBERS).await;
        let injector = rig.injector();
        let walk = hold_walk_at_header(&injector);

        let reserved = {
            let delete = ls.delete_layer(&rig.session_id, "largest-index", rig.ext.clone());
            tokio::pin!(delete);
            tokio::select! {
                result = &mut delete => panic!("{index:?}: the delete must park on its walk; got {result:?}"),
                () = parked(&walk) => {}
            }
            ls.retirement_budget().reserved_bytes()
        };
        walk.release();
        eventually("the dropped delete's reservation given back", || async {
            ls.retirement_budget().reserved_bytes() == 0
        })
        .await;

        assert_eq!(
            reserved, seeded_charge,
            "{index:?}: the page is one member past the {SEEDED_MEMBERS} members of the largest \
             index"
        );
    }))
    .await;
}

/// A `WRONGTYPE` reply, as the store answers a command on a key of another
/// type and fred hands it over.
fn wrongtype() -> fred::error::Error {
    fred::error::Error::new(
        fred::error::ErrorKind::InvalidArgument,
        "WRONGTYPE Operation against a key holding the wrong kind of value",
    )
}

/// D4-T15, a `WRONGTYPE` on one index's count alone, through the size read's
/// per-command seam (the index itself is a real, empty one): the size read
/// counts it once under that index's family, WARNs and reserves the full
/// page, and the delete goes on — the walk, the release (3 references
/// pending) and a teardown that, the index being sound, removes the whole
/// object — and answers OK. One delete per index, all at once.
///
/// The tick and the full-page reservation are the assertions a size read
/// that skips that count's `WRONGTYPE` arm fails; a size read that propagated
/// it would fail the delete instead.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_delete_whose_size_read_meets_a_wrong_type_on_one_index_count_counts_it_reserves_the_full_page_and_deletes_the_layer(
) {
    let full_charge = retirement_charge_bytes(
        RetirementPaging::for_allowance(R_REF).page_len(),
        u64::from(FOREIGN_INDEX_REFS),
    );
    futures::future::join_all(ForeignIndex::ALL.map(|index| async move {
        let rig = Rig::new("rref-count-wrongtype").await;
        let ls = rig.store();
        let oid = rig
            .layer(&ls, "count-wrongtype", 1, FOREIGN_INDEX_REFS)
            .await;
        let injector = rig.injector();
        injector.push_at(
            ObjectSnapshotReadFaultTarget::retirement_size_read(index.family()),
            wrongtype(),
        );
        let walk = hold_walk_at_header(&injector);

        let delete = ls.delete_layer(&rig.session_id, "count-wrongtype", rig.ext.clone());
        tokio::pin!(delete);
        tokio::select! {
            result = &mut delete => panic!("{index:?}: the delete must park on its walk; got {result:?}"),
            () = parked(&walk) => {}
        }
        let reserved = ls.retirement_budget().reserved_bytes();
        walk.release();
        let answer = delete.await;

        assert_eq!(
            (injector.remaining(), injector.holds_remaining()),
            (0, 0),
            "{index:?}: fixture: the size read took the error and both holds were taken"
        );
        assert!(
            answer.is_ok(),
            "{index:?}: the delete goes on and succeeds; got {answer:?}"
        );
        for &cell in live_edit::observability::metrics::SNAPSHOT_READ_KEY_FAMILIES {
            assert_eq!(
                rig.type_errors(cell),
                if cell == index.family() { 1.0 } else { 0.0 },
                "{index:?}: the ticks on key_family={cell}"
            );
        }
        assert_eq!(
            reserved, full_charge,
            "{index:?}: the count failed, so the full page was reserved"
        );
        assert_eq!(
            rig.ext.pending_count(),
            FOREIGN_INDEX_REFS as usize,
            "{index:?}: the release was scheduled"
        );
        assert_eq!(rig.mapping("count-wrongtype").await, None);
        assert!(!rig.indexed("count-wrongtype").await);
        assert_eq!(
            rig.object_keys(oid).await,
            Vec::<String>::new(),
            "{index:?}: the object is gone"
        );
    }))
    .await;
}

/// D4-T15, a permanent failure other than `WRONGTYPE` on one index's count,
/// through the size read's per-command seam (a `Protocol` error): the size
/// read counts it on `live_edit_snapshot_read_refused_total{kind="protocol"}`
/// (no type error) and returns it, and the `DeleteLayer` answers `INTERNAL`
/// with nothing done: the mapping, the layers-index member and the object
/// stand, and nothing was reserved. One delete per index, all at once.
///
/// The `INTERNAL` answer and the standing mapping are the assertions a size
/// read that treated this failure as a `WRONGTYPE` fails: it would go on and
/// delete the layer.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_delete_whose_size_read_fails_permanently_on_one_index_count_answers_internal_and_changes_nothing(
) {
    futures::future::join_all(ForeignIndex::ALL.map(|index| async move {
        let rig = Rig::new("rref-count-refused").await;
        let ls = rig.store();
        let oid = rig.layer(&ls, "count-refused", 1, FOREIGN_INDEX_REFS).await;
        let injector = rig.injector();
        injector.push_at(
            ObjectSnapshotReadFaultTarget::retirement_size_read(index.family()),
            fred::error::Error::new(fred::error::ErrorKind::Protocol, "injected protocol error"),
        );

        let answer = ls
            .delete_layer(&rig.session_id, "count-refused", rig.ext.clone())
            .await;

        assert_eq!(
            injector.remaining(),
            0,
            "{index:?}: fixture: the size read took the error"
        );
        assert!(
            matches!(answer, Err(ServiceError::Internal(_))),
            "{index:?}: a permanent size-read failure answers INTERNAL; got {answer:?}"
        );
        assert_eq!(
            rig.sample(
                "live_edit_snapshot_read_refused_total",
                &[("kind", "protocol")]
            ),
            Some(1.0),
            "{index:?}: counted once as a protocol refusal"
        );
        for &cell in live_edit::observability::metrics::SNAPSHOT_READ_KEY_FAMILIES {
            assert_eq!(rig.type_errors(cell), 0.0, "{index:?}: no type error");
        }
        assert_eq!(rig.mapping("count-refused").await, Some(oid.to_string()));
        assert!(rig.indexed("count-refused").await);
        assert!(rig.header_exists(oid).await, "{index:?}: the object stands");
        for outcome in ["immediate", "waited", "cancelled"] {
            assert_eq!(
                rig.waits(outcome),
                0.0,
                "{index:?}: {outcome}: nothing was reserved"
            );
        }
    }))
    .await;
}

/// D4-T15: a transient failure of the size read (the snapshot-read seam's
/// `next_read` entry, which on a retirement names the size read) answers the
/// `DeleteLayer` `UNAVAILABLE` with nothing done: the mapping, the
/// layers-index member and the object stand, nothing was reserved, and on
/// this replica the object's admission entry and its structural-cache entry
/// stand (premise-check amendment). A retry deletes the layer.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_delete_whose_size_read_fails_transiently_answers_unavailable_and_changes_nothing() {
    let rig = Rig::new("rref-size-read").await;
    let ls = rig.store();
    let oid = rig.layer(&ls, "size-read", 1, 3).await;
    ls.update_layer_relayed(
        &rig.session_id,
        "size-read",
        oid,
        Some(delta_with_set_field_inline(1, "k", vec![1], 1)),
        "size-read-writer",
        1_000,
    )
    .await
    .expect("an update must commit, so this replica holds the object's admission entry");
    let entries_before = ls.semaphore_count();
    let straggler = hold_checkout(&ls, oid);
    let injector = rig.injector();
    injector.push(fred::error::Error::new(
        fred::error::ErrorKind::Timeout,
        "injected size-read timeout",
    ));

    let result = ls
        .delete_layer(&rig.session_id, "size-read", rig.ext.clone())
        .await;

    assert_eq!(
        injector.remaining(),
        0,
        "fixture: the size read took the fault"
    );
    assert!(
        matches!(result, Err(ServiceError::Unavailable(_))),
        "a transient size-read failure answers as the mapping read's would: UNAVAILABLE; got \
         {result:?}"
    );
    assert_eq!(rig.mapping("size-read").await, Some(oid.to_string()));
    assert!(rig.indexed("size-read").await);
    assert!(
        rig.header_exists(oid).await
            && rig
                .exists(&rig.tv.key_schema().obj_ext_index(&oid.to_string()))
                .await,
        "the object stands"
    );
    for outcome in ["immediate", "waited", "cancelled"] {
        assert_eq!(rig.waits(outcome), 0.0, "nothing was reserved");
    }
    assert!(
        ls.admission_permits_available_for_test(oid).is_some(),
        "this replica's admission entry stands"
    );
    assert_eq!(ls.semaphore_count(), entries_before);
    assert!(
        straggler.put_back(entry(oid, 2)) == PutBack::Stored,
        "this replica's structural-cache entry stands"
    );
    for &cell in live_edit::observability::metrics::SNAPSHOT_READ_KEY_FAMILIES {
        assert_eq!(rig.type_errors(cell), 0.0, "a timeout is no type error");
    }

    ls.delete_layer(&rig.session_id, "size-read", rig.ext.clone())
        .await
        .expect("a retry must delete the layer");
    rig.until_object_gone(oid).await;
}

// ═════════════════════════════════════════════════════════════════════════════
// D4-T16 — the per-instance metric sink
// ═════════════════════════════════════════════════════════════════════════════

/// D4-T16: a retirement ticks the retirement allowance's own series and never
/// the catch-up budget's, though both instances tick the store's one metrics
/// handle. Parked mid-walk, `live_edit_retirement_budget_reserved_bytes`
/// reads the reservation while `live_edit_catchup_build_budget_reserved_bytes`
/// reads 0; afterwards both reserved gauges and both over-commit gauges read
/// 0, the wait is recorded once on `live_edit_retirement_wait_seconds`
/// (`immediate`), and the catch-up budget's wait histogram and admission
/// counter are untouched. A shared sink moves the catch-up series.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_retirement_ticks_the_retirement_series_and_leaves_the_catch_up_series_flat() {
    const REFS: u32 = 10;
    let rig = Rig::new("rref-series").await;
    let ls = Arc::new(rig.store());
    let oid = rig.layer(&ls, "series", 1, REFS).await;
    let injector = rig.injector();
    let hold = hold_walk_at_obj_ext(&injector);
    let reservation = rig.charge(oid, REFS).await;

    let delete = tokio::spawn({
        let (ls, ext, session_id) = (Arc::clone(&ls), rig.ext.clone(), rig.session_id.clone());
        async move { ls.delete_layer(&session_id, "series", ext).await }
    });
    parked(&hold).await;
    assert_eq!(
        (
            rig.sample("live_edit_retirement_budget_reserved_bytes", &[]),
            rig.sample("live_edit_catchup_build_budget_reserved_bytes", &[]),
        ),
        (Some(reservation as f64), Some(0.0)),
        "the retirement's reservation moves its own gauge and not the catch-up budget's"
    );
    hold.release();
    delete
        .await
        .expect("the delete task must not panic")
        .expect("DeleteLayer must succeed");

    for gauge in [
        "live_edit_retirement_budget_reserved_bytes",
        "live_edit_retirement_budget_overcommit_bytes",
        "live_edit_catchup_build_budget_reserved_bytes",
        "live_edit_catchup_build_budget_overcommit_bytes",
    ] {
        assert_eq!(
            rig.sample(gauge, &[]),
            Some(0.0),
            "{gauge} after the delete"
        );
    }
    assert_eq!(rig.waits("immediate"), 1.0, "one uncontended reservation");
    assert_eq!(
        rig.sample("live_edit_catchup_build_wait_seconds_count", &[])
            .unwrap_or(0.0),
        0.0,
        "the catch-up budget's wait histogram has no observation"
    );
    let catch_up_admissions: BTreeSet<String> =
        live_edit::observability::metrics::CATCHUP_BUILD_ADMISSION_RESULTS
            .iter()
            .filter(|result| {
                rig.sample(
                    "live_edit_catchup_build_admission_total",
                    &[("result", result)],
                ) != Some(0.0)
            })
            .map(|result| (*result).to_owned())
            .collect();
    assert!(
        catch_up_admissions.is_empty(),
        "no catch-up admission cell may move: {catch_up_admissions:?}"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// D4-T4, D4-T11 — DeleteLayer's remaining exits: a panic inside the
// retirement, a failed mapping re-read, and a mapping gone by the re-read
// ═════════════════════════════════════════════════════════════════════════════

/// An external-value store that panics in the release's enqueue (the call
/// `schedule_old_object_ext_gc` makes once the walk and the begin are done)
/// and delegates everything else to an in-memory store.
struct PanickingReleaseStore {
    inner: MockExternalValueStore,
}

#[async_trait::async_trait]
impl ExternalValueStore for PanickingReleaseStore {
    async fn enqueue_retirement_attempt(
        &self,
        _object_id: &str,
        _attempt: &str,
        _releases: &[live_edit::storage::external_values::ExtRelease],
        _send_by: std::time::Instant,
    ) -> Result<live_edit::storage::external_values::ReleaseAttemptOutcome, ServiceError> {
        panic!("injected panic inside a retirement's release");
    }

    async fn store_values(&self, values: &[(&[u8; 32], &[u8])]) -> Result<(), ServiceError> {
        self.inner.store_values(values).await
    }

    async fn store_verified_values(
        &self,
        values: &[(&[u8; 32], &[u8])],
    ) -> Result<(), ServiceError> {
        self.inner.store_verified_values(values).await
    }

    fn ext_value_verify(&self) -> live_edit::storage::external_values::ExtValueVerify<'_> {
        self.inner.ext_value_verify()
    }

    async fn load_values(
        &self,
        hashes: &[&[u8; 32]],
    ) -> Result<Vec<([u8; 32], Vec<u8>)>, ServiceError> {
        self.inner.load_values(hashes).await
    }

    async fn read_batch(
        &self,
        hashes: &[[u8; 32]],
    ) -> Result<std::collections::HashMap<[u8; 32], Vec<u8>>, ServiceError> {
        self.inner.read_batch(hashes).await
    }

    async fn probe_values(
        &self,
        hashes: &[[u8; 32]],
    ) -> Result<live_edit::storage::external_values::ExtValueProbe, ServiceError> {
        self.inner.probe_values(hashes).await
    }

    async fn process_pending(
        &self,
        policy: &live_edit::storage::external_values::GcPolicy,
        state: &mut live_edit::storage::external_values::GcPassState,
    ) -> Result<live_edit::storage::external_values::GcBatchResult, ServiceError> {
        self.inner.process_pending(policy, state).await
    }

    fn ext_release_gate(
        &self,
    ) -> Option<&Arc<live_edit::storage::ext_release_gate::ExtReleaseGate>> {
        self.inner.ext_release_gate()
    }
}

/// A panic inside a `DeleteLayer`'s retirement — its release's enqueue
/// panics, past the walk and the begin, with the admission guard installed —
/// unwinds the delete's task, and the unwind gives the whole reservation
/// back (the grant and the walk's top-ups are `Drop` guards) and retires this
/// replica's admission entry and structural-cache entry for the object (the
/// guard's drop).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_delete_whose_retirement_panics_gives_its_reservation_back_and_retires_its_entries() {
    let rig = Rig::new("rref-panic").await;
    let ls = Arc::new(rig.store());
    let oid = rig.layer(&ls, "panics", 1, 10).await;
    ls.update_layer_relayed(
        &rig.session_id,
        "panics",
        oid,
        Some(delta_with_set_field_inline(1, "k", vec![1], 1)),
        "panic-writer",
        1_000,
    )
    .await
    .expect("an update must commit, so this replica holds the object's admission entry");
    cache_entry_for(&ls, oid);
    assert!(
        ls.admission_permits_available_for_test(oid).is_some(),
        "fixture: the admission entry exists"
    );
    let budget = Arc::clone(ls.retirement_budget());
    let store: Arc<dyn ExternalValueStore> = Arc::new(PanickingReleaseStore {
        inner: MockExternalValueStore::new(),
    });

    let delete = tokio::spawn({
        let (ls, session_id) = (Arc::clone(&ls), rig.session_id.clone());
        async move { ls.delete_layer(&session_id, "panics", store).await }
    });
    let joined = tokio::time::timeout(BUDGET, delete)
        .await
        .expect("the delete's task ends");

    match joined {
        Err(e) if e.is_panic() => {}
        other => panic!("the release's panic must unwind the delete's task; got {other:?}"),
    }
    eventually("the unwound delete's reservation given back", || async {
        budget.reserved_bytes() == 0 && budget.overcommit_bytes() == 0
    })
    .await;
    assert_returned(&budget, "after the panic");
    assert!(
        ls.admission_permits_available_for_test(oid).is_none(),
        "the admission guard's drop retired this replica's admission entry"
    );
    assert_eq!(ls.semaphore_count(), 0);
    assert_retired(&ls, oid, "the unwound delete's object");
}

/// A `DeleteLayer` whose mapping re-read fails (the mapping replaced by a
/// value that is no object id while the delete waits for the allowance, so
/// `get_layer_mapping` answers `INTERNAL`) changes nothing: the mapping, the
/// layers-index member and the object stand, the reservation is given back,
/// and this replica's admission entry and structural-cache entry stand — the
/// guard that retires them is installed only after the re-read. Once the
/// mapping is restored, a retry deletes the layer.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_delete_whose_mapping_re_read_fails_changes_nothing_and_a_retry_works() {
    const CORRUPT: &str = "not-an-object-id";
    let rig = Rig::new("rref-re-read").await;
    let ls = Arc::new(rig.store());
    let oid = rig.layer(&ls, "re-read", 1, 10).await;
    let (entries_before, straggler) = admitted_and_cached(&rig, &ls, "re-read", oid).await;
    let budget = Arc::clone(ls.retirement_budget());
    let full = budget
        .reserve(R_REF, None)
        .await
        .expect("the test takes the whole allowance");
    let mapping_key = rig
        .tv
        .key_schema()
        .layer_mapping(&rig.session_id, "re-read");

    let delete = tokio::spawn({
        let (ls, ext, session_id) = (Arc::clone(&ls), rig.ext.clone(), rig.session_id.clone());
        async move { ls.delete_layer(&session_id, "re-read", ext).await }
    });
    waiters(&budget, 1).await;
    let _: () = rig
        .tv
        .client()
        .pool
        .set(&mapping_key, CORRUPT, None, None, false)
        .await
        .expect("SET of the corrupt mapping must succeed");
    drop(full);
    let result = tokio::time::timeout(BUDGET, delete)
        .await
        .expect("the delete returns")
        .expect("the delete task must not panic");

    match &result {
        Err(ServiceError::Internal(msg)) => assert!(
            msg.contains(CORRUPT),
            "the re-read's own error, naming the corrupt value; got {msg:?}"
        ),
        other => panic!("a failed mapping re-read answers INTERNAL; got {other:?}"),
    }
    assert_eq!(
        rig.mapping("re-read").await.as_deref(),
        Some(CORRUPT),
        "nothing was written: the retry loop's mapping DEL never ran"
    );
    assert!(rig.indexed("re-read").await && rig.header_exists(oid).await);
    assert!(
        ls.admission_permits_available_for_test(oid).is_some(),
        "this replica's admission entry stands"
    );
    assert_eq!(ls.semaphore_count(), entries_before);
    assert!(
        straggler.put_back(entry(oid, 2)) == PutBack::Stored,
        "this replica's structural-cache entry stands"
    );
    assert_returned(&budget, "after the failed re-read");

    let _: () = rig
        .tv
        .client()
        .pool
        .set(&mapping_key, oid.to_string(), None, None, false)
        .await
        .expect("SET of the restored mapping must succeed");
    ls.delete_layer(&rig.session_id, "re-read", rig.ext.clone())
        .await
        .expect("a retry of the delete must succeed");
    assert!(rig.mapping("re-read").await.is_none());
    assert_eq!(rig.object_keys(oid).await, Vec::<String>::new());
}

/// The re-read's nil branch: two `DeleteLayer`s of one layer wait for the
/// allowance together (the test holds it; two reservations of `W` + 10 KiB
/// do not fit side by side). The first is granted, deletes the layer and
/// gives its reservation back; the second, granted then, re-reads a mapping
/// that is gone and goes on for the same object, as a mapping gone before
/// the retry loop's `DEL` always has: both answer OK, and the object is gone.
/// Fails on a nil re-read answered `NOT_FOUND`.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn two_deletes_of_one_layer_waiting_together_both_answer_ok() {
    let rig = Rig::new("rref-re-read-nil").await;
    let ls = Arc::new(rig.store());
    let oid = rig.layer(&ls, "deleted-twice", 1, 10).await;
    let budget = Arc::clone(ls.retirement_budget());
    let full = budget
        .reserve(R_REF, None)
        .await
        .expect("the test takes the whole allowance");
    let spawn_delete = || {
        let (ls, ext, session_id) = (Arc::clone(&ls), rig.ext.clone(), rig.session_id.clone());
        tokio::spawn(async move { ls.delete_layer(&session_id, "deleted-twice", ext).await })
    };

    let first = spawn_delete();
    waiters(&budget, 1).await;
    let second = spawn_delete();
    waiters(&budget, 2).await;
    drop(full);

    tokio::time::timeout(BUDGET, async {
        for (which, task) in [("first", first), ("second", second)] {
            task.await
                .expect("the delete task must not panic")
                .unwrap_or_else(|e| panic!("the {which} delete must answer OK; got {e:?}"));
        }
    })
    .await
    .expect("both deletes return");
    assert_eq!(rig.waits("waited"), 2.0, "fixture: both deletes waited");
    assert!(rig.mapping("deleted-twice").await.is_none() && !rig.indexed("deleted-twice").await);
    assert_eq!(rig.object_keys(oid).await, Vec::<String>::new());
    assert_returned(&budget, "after both deletes");
}

// ═════════════════════════════════════════════════════════════════════════════
// D4-T15 — a size read that fails on the overwrite body and the sweep, and
// the sweep's wait raced against its cancel
// ═════════════════════════════════════════════════════════════════════════════

/// A transient failure of the overwrite body's size read keeps the replaced
/// object for the sweep with nothing reserved and no release begun, and a
/// later sweep pass retires it, its references released.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn an_overwrite_body_whose_size_read_fails_keeps_its_object_for_a_later_sweep() {
    const REFS: u32 = 10;
    let rig = Rig::new("rref-body-size-read").await;
    let gate = Arc::new(tokio::sync::Barrier::new(2));
    let ls = rig
        .store()
        .with_overwrite_cleanup_gate_for_test(Arc::clone(&gate));
    let old = rig.layer(&ls, "body-size-read", 1, REFS).await;
    rig.layer(&ls, "body-size-read", 2, 0).await;
    let injector = rig.injector();
    injector.push(fred::error::Error::new(
        fred::error::ErrorKind::Timeout,
        "injected size-read timeout",
    ));
    tokio::time::timeout(BUDGET, gate.wait())
        .await
        .expect("the body is parked on its gate");
    tokio::time::timeout(BUDGET, ls.wait_for_cleanup(BUDGET * 2))
        .await
        .expect("the body ends");

    assert_eq!(
        injector.remaining(),
        0,
        "fixture: the size read took the fault"
    );
    assert_header_kept_and_unbegun(&rig, old, "after the failed size read").await;
    for outcome in ["immediate", "waited", "cancelled"] {
        assert_eq!(rig.waits(outcome), 0.0, "{outcome}: nothing was reserved");
    }

    rig.sweep_until(
        rig.store().with_orphan_sweep_grace(Duration::ZERO),
        "a later sweep retired the replaced object",
        || async { rig.object_keys(old).await.is_empty() },
    )
    .await;
    assert_eq!(
        rig.ext.pending_count(),
        REFS as usize,
        "the sweep's retirement released its references"
    );
}

/// A transient failure of the sweep's size read keeps the orphan — its
/// header standing, nothing reserved, no release begun — and a later sweep
/// pass retires it, its references released: the failed pass did not go on
/// to tear it down.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_sweep_retirement_whose_size_read_fails_keeps_its_orphan_for_a_later_pass() {
    const REFS: u32 = 10;
    let rig = Rig::new("rref-sweep-size-read").await;
    let ls = rig.store();
    let oid = generic_orphan(&rig, &ls, "sweep-size-read", 1, REFS).await;
    let injector = rig.injector();
    injector.push(fred::error::Error::new(
        fred::error::ErrorKind::Timeout,
        "injected size-read timeout",
    ));

    rig.sweep_until(
        rig.store().with_orphan_sweep_grace(Duration::ZERO),
        "a sweep pass's size read took the fault",
        || async { injector.remaining() == 0 },
    )
    .await;
    assert_header_kept_and_unbegun(&rig, oid, "after the failed size read").await;
    for outcome in ["immediate", "waited", "cancelled"] {
        assert_eq!(rig.waits(outcome), 0.0, "{outcome}: nothing was reserved");
    }

    rig.sweep_until(
        rig.store().with_orphan_sweep_grace(Duration::ZERO),
        "a later pass retired the orphan",
        || async { rig.object_keys(oid).await.is_empty() },
    )
    .await;
    assert_eq!(
        rig.ext.pending_count(),
        REFS as usize,
        "the later pass ran the whole retirement, its release included"
    );
}

/// The sweep's wait for the allowance raced against its cancel: with the
/// allowance held by the test, the sweep's retirement of an orphan queues;
/// the sweep is cancelled while it waits. The wait leaves the queue, nothing
/// is reserved, the wait is recorded `cancelled`, and the orphan keeps its
/// header with no release begun, for the next sweep.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_sweep_retirement_cancelled_while_it_waits_keeps_its_orphan_and_reserves_nothing() {
    let rig = Rig::new("rref-sweep-cancel").await;
    let ls = rig.store();
    let oid = generic_orphan(&rig, &ls, "sweep-cancel", 1, 10).await;
    let sweep = rig.store().with_orphan_sweep_grace(Duration::ZERO);
    let budget = Arc::clone(sweep.retirement_budget());
    let full = budget
        .reserve(R_REF, None)
        .await
        .expect("the test takes the whole allowance");

    rig.sweep_until(sweep, "the sweep's retirement queued", || async {
        budget.waiters() == 1
    })
    .await;

    assert_eq!(
        budget.waiters(),
        0,
        "the cancel took the wait off the queue"
    );
    assert_eq!(
        budget.reserved_bytes(),
        R_REF,
        "only the test's grant is held"
    );
    assert_eq!(
        rig.waits("cancelled"),
        1.0,
        "the wait is recorded cancelled"
    );
    assert_header_kept_and_unbegun(&rig, oid, "after the cancelled wait").await;
    assert_eq!(rig.ext.pending_count(), 0, "nothing was released");
    drop(full);
    assert_returned(&budget, "after the held grant went");
}
