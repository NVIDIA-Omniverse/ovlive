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

//! b.459 — the WATCH/MULTI/EXEC commit path must refuse a wrong-typed mutated
//! key **before** it opens the `MULTI`.
//!
//! # The defect
//!
//! A Valkey `MULTI`/`EXEC` is atomic in isolation and in queueing, but **not in
//! execution**. A command that fails at execution time neither aborts the
//! transaction nor rolls back its siblings: every other queued command still
//! runs, and fred surfaces only the first error frame — so the caller never
//! learns which siblings landed. Pre-b.459 that made an out-of-band writer who
//! left a plain string on, say, one field hash enough to half-commit a
//! structural `UpdateLayer`: the `XADD` published a commit to every subscriber
//! while the `HSET` beside it was refused, and the client got one `Internal`
//! naming a single key.
//!
//! The Lua fast path never had this shape — its validation pre-pass TYPE-probes
//! every key it will write before its first mutation, and a script aborts on the
//! first failing `redis.call`. b.459 gave the slow path the same guarantee the
//! only way it can be had there: a read-only `TYPE` probe over the mutated key
//! set, issued **before** `MULTI` opens (`scripts/keyspace_type_probe.lua`,
//! `LayerStore::probe_keyspace_types`).
//!
//! # What lives here
//!
//! | Test | What it pins |
//! |---|---|
//! | [`txn_wrongtype_new_section_key_refuses_whole_commit`] | the regression, structural shape: a string on the section hash a `CreateSection` will write |
//! | [`txn_wrongtype_children_index_refuses_whole_commit`] | the `obj:chs` Set on an object without the `chs_complete` marker — the one index no read on this path touches there, so the probe is its *only* guard |
//! | [`txn_wrongtype_children_index_of_a_marked_object_is_caught_by_the_snapshot_read_not_the_probe`] | the same Set on a marked object, which the snapshot read DOES read: it refuses first, on its own `children_index` counter cell |
//! | [`txn_wrongtype_new_tombstone_key_refuses_whole_commit`] | the tombstone hash a `DeleteSection` will write |
//! | [`txn_wrongtype_new_field_key_refuses_whole_commit`] | the `!needs_snapshot` txn-knob path (`LIVE_EDIT_UPDATE_LAYER_BACKEND=txn` over a fast-path-shaped delta), per-record half |
//! | [`txn_wrongtype_fields_index_refuses_whole_commit`] | the same path, index half (`obj:flds` ZSet) |
//! | [`txn_wrongtype_new_time_sample_key_refuses_whole_commit`] | the time-sample record hash |
//! | [`txn_wrongtype_time_samples_index_refuses_whole_commit`] | the `obj:tss` ZSet |
//! | [`txn_wrongtype_ext_index_refuses_whole_commit`] | the `obj:ext` hash on the txn-knob path — a key the external-value pre-image read beside the probe also reads, so it pins that the probe's verdict decides (b.j2x) |
//! | [`txn_clean_keyspace_commits_and_never_ticks_the_type_error`] | the negative control: the identical deltas over a clean keyspace commit, and the alertable cell keeps its zero baseline |
//! | [`txn_wrongtype_children_list_key_is_caught_by_the_snapshot_read_not_the_probe`] | exclusion #1's boundary, post-b.zvd: the list key is still not probed, but on a list `obj:chs` names the snapshot read's `LRANGE` now fails closed on it first |
//! | [`txn_wrongtype_key_the_commit_only_dels_is_caught_by_the_snapshot_read_not_the_probe`] | exclusion #2's boundary, same reason: on an unmarked object the leaf's children list is read before the probe, so the `DEL`-only key never reaches it |
//! | [`txn_wrongtype_children_list_the_marked_index_does_not_name_is_healed_by_the_commit`] | exclusion #1's heal, reachable again on a marked object: a list `obj:chs` does not name is not read, and the commit's `DEL` + `RPUSH` repairs it |
//! | [`txn_wrongtype_unindexed_leaf_list_the_commit_only_dels_is_healed_on_a_marked_object`] | exclusion #2's heal on a marked object: the unread `DEL`-only key is deleted by the commit |
//! | [`txn_field_only_delta_commits_despite_a_corrupt_time_samples_index`] | negative scope: the probe list comes from the MUTATIONS, so a corrupt index this delta never writes must not refuse it |
//! | [`txn_time_sample_only_delta_commits_despite_a_corrupt_fields_index`] | the same claim with the two index classes swapped — they enter the probe list from different `mutation_probe_keys` arms |
//! | [`txn_empty_delta_commits_despite_both_indexes_corrupt`] | the heartbeat-shaped commit probes one key (`obj:hdr`) and must be indifferent to both corrupt indexes |
//! | [`txn_and_lua_backends_refuse_an_identical_keyspace_identically`] | backend parity, asserted on the message BYTES (up to the Lua path's parenthesized locator — see the test) and on both metric cells |
//! | [`txn_many_mutations_against_one_wrong_typed_index_refuse_once`] | one commit is one refusal and one tick, whatever the mutation count |
//! | [`txn_wrongtype_deleted_sections_index_is_caught_by_the_snapshot_read_not_the_probe`] | the honest boundary: three index keys are read *before* the probe, so they fail closed there instead — documented, not claimed as probe coverage |
//!
//! A second family lives in [`probe_fault_injection`] (feature-gated on
//! `test-fault-injection`): the tests above all reach the probe's *tagged* arm
//! through a real wrong-typed key, which is the only arm a live keyspace can
//! stage. The probe's other arm — a transient probe FAILURE, which must become
//! a retriable `Aborted` rather than a write outage — needs a Valkey that dies
//! mid-commit, so it is driven through `LayerStore::
//! with_keyspace_probe_fred_error_injector_for_test` instead. See that module's
//! header for its own table.
//!
//! A third, [`ext_pre_image_read_fault_injection`] (same feature gate), pins the
//! ORDER in which the knob path consults the probe and the external-value
//! pre-image read that runs beside it (b.j2x). The read touches the row keys
//! and `obj:ext` the probe also covers, so against a real wrong-typed key both
//! fail. The probe's verdict is consulted first, so the tagged-refusal rows
//! above now pass on every run. But they cannot show that ORDER is what makes
//! them pass: before b.j2x the two sat under one fail-fast join and those rows
//! also passed most of the time, whenever the probe's error happened to arrive
//! first. That module makes the read fail first, every time, through
//! `LayerStore::with_ext_pre_image_read_fred_error_injector_for_test`, and
//! asserts the probe's verdict still decides.
//!
//! The per-variant key inventory (which key each `StorageMutation` probes, and
//! at which type) is pinned as pure unit tests beside the function itself in
//! `src/storage/layer.rs`; the source-shape fence that keeps
//! `mutation_probe_keys` and `execute_mutation` naming the same variants lives
//! in `mapper_routing_test.rs`. This file is the behavioural half: it drives
//! real commits against a real Valkey and asserts the whole-commit refusal.
//!
//! # Why every test here pins the txn backend
//!
//! `LayerStore::with_update_layer_backend(UpdateLayerBackend::Txn)` is what
//! routes a fast-path-shaped delta (field-only, time-sample-only) down the
//! WATCH/MULTI/EXEC path — that is the production `LIVE_EDIT_UPDATE_LAYER_BACKEND=txn`
//! rollback knob, and the only way those two key families reach the probe. The
//! structural deltas take the slow path under `txn` (and `lua-watch-structural`)
//! too, but not under the default `lua`, whose structural route plans on the
//! structural cache and runs no probe; pinning the backend uniformly keeps
//! every test in the file exercising one dispatch.
//!
//! # Reachability, stated once
//!
//! The structural path's snapshot read — `read_object_snapshot`'s unfenced
//! fan-out (Spec SDD SR-4.3) — reads `obj:secs`, `obj:dels`, `obj:flds`,
//! `obj:tss`, the per-record hashes of everything those indices name and, since
//! b.zvd classifies the list reads, the children lists, all before the probe
//! ever runs. Which lists follows the children rule (SR-1.2): on an object
//! without the `chs_complete` marker, the root's and every live section's; on a
//! marked object — every object `create_layer` writes — `obj:chs` itself and
//! the lists it names under the root or a live section. Wrong-typing a key that
//! read reads is therefore caught by the snapshot read (also fail-closed, also
//! before `MULTI` — a different guard reaching the same verdict), so the probe
//! cells this file can drive are exactly: keys the commit CREATES (not yet in
//! any index), `obj:chs` of an UNMARKED object (read by nothing on this path
//! there), and — on the `!needs_snapshot` knob path, where no snapshot is
//! loaded at all — `obj:flds` / `obj:tss` themselves and `obj:ext`. On that
//! path the only other pre-`MULTI` read of any of these is the external-value
//! pre-image read (the written rows and `obj:ext`), which runs beside the probe
//! and is consulted only after it (b.j2x), so it never takes the refusal from
//! the probe. The boundary rows of the table above pin that rather than leaving
//! it as a comment.
//!
//! The `obj:ch:*` half of that list moved twice. Pre-b.zvd the children-list
//! `LRANGE` was `.unwrap_or_default()`, so a `WRONGTYPE` there read as "this
//! parent has no children" and the commit sailed on to the probe; b.zvd
//! classified that read like its `ZRANGE` siblings (`src/storage/diff_read.rs`),
//! which moved the two exclusion tests below from asserting a *heal* to
//! asserting the snapshot read's refusal. Then the structural path moved to the
//! marked children rule, under which a marked object's list that `obj:chs` does
//! not name is never read — so for exactly those keys the heal is reachable
//! again, and the two heal rows pin it. The exclusions themselves did not
//! change — `create_probe_keys` / `mutation_probe_keys` still leave the list
//! keys out, which the pure unit tests in `src/storage/layer.rs` pin and the
//! heal rows confirm behaviourally.
//!
//! # Pool size 1, deliberately
//!
//! Every test here is single-writer, and a one-slot `txn` pool is what makes the
//! slot-hygiene assertion exact: the follow-up `acquire_txn_for` provably probes
//! the same connection the refused attempt borrowed. Anything larger and it
//! could pick a different slot and pass vacuously (the b.bhb pattern —
//! `b_bhb_tests::layer_in_doubt_without_a_dispatch_leaves_no_open_multi_on_the_txn_slot`).

use std::collections::HashMap;
use std::sync::Arc;

use fred::interfaces::{HashesInterface, KeysInterface, StreamsInterface};

use live_edit::config::UpdateLayerBackend;
use live_edit::domain::types::ObjectState;
use live_edit::error::ServiceError;
use live_edit::observability::metrics::test_util::{render, sample_sum, sample_value};
use live_edit::observability::metrics::Metrics;
use live_edit::storage::external_values::MockExternalValueStore;
use live_edit::storage::layer::{LayerStore, HEADER_FIELD_CHS_COMPLETE};
use live_edit::storage::session::SessionStore;
use live_edit::storage::test_support::relay::RelayedUpdates;
use live_edit::storage::test_support::TestValkey;
use oodf_ffi::{
    CommandCreateSection, CommandDeleteSection, OwnedDelta, SetField, StructCommand,
    TimeSample as OodfTimeSample,
};

// The snapshot read's own counter, `live_edit_snapshot_read_type_error_total`,
// read through b.atj's asserters rather than a copy, for the reason
// `b_atj_tests` imports b.muu's: one definition of "only this family moved".
use crate::storage::b_atj_tests::{assert_no_family_ticked, assert_only_family_ticked};

// ═════════════════════════════════════════════════════════════════════════════
// Contract constants — restated, not imported
// ═════════════════════════════════════════════════════════════════════════════

/// The exact operator-visible prefix both backends produce for a wrong-typed
/// mutated key.
///
/// Restated here rather than imported because the items that build it
/// (`KEYSPACE_WRONGTYPE_TAG`, the `format!` in `probe_keyspace_types`) are
/// `pub(crate)` / private in `src/storage/layer.rs` and an integration test is a
/// separate crate. That is the same standing — and the same deliberate
/// duplication — as `update_layer_lua_test.rs`'s copy of the prefix: the string
/// IS the contract, so a test that imported it could not fail when it changed.
const KEYSPACE_TYPE_ERROR_PREFIX: &str = "update_layer keyspace type error: WRONGTYPE_KEY:";

/// Every `reason` cell of `live_edit_valkey_txn_dirty_checkouts_total` (b.7td;
/// `topology_reply` added by b.289 sub-task 3), restated for the same reason as
/// the prefix above.
const TXN_DIRTY_REASONS: [&str; 5] = [
    "leaked_multi",
    "unexpected_reply",
    "probe_timeout",
    "probe_error",
    "topology_reply",
];

/// The section id every structural test in this file creates. Chosen away from
/// 1 so a stray fixture section could not be mistaken for it.
const SECTION_ID: u64 = 7;

/// A second root child, for the tests whose setup commit already created
/// [`SECTION_ID`] so the root's children list is one `obj:chs` names.
const SECOND_SECTION_ID: u64 = 8;

/// The time carried by the time-sample tests' delta.
const SAMPLE_TIME: f64 = 12.5;

// ═════════════════════════════════════════════════════════════════════════════
// Fixture
// ═════════════════════════════════════════════════════════════════════════════

/// A layer on a one-slot pool, with the metrics handle wired into BOTH the
/// store (for `keyspace_type_error_total`) and the client's observability hooks
/// (for `valkey_txn_dirty_checkouts_total` / `valkey_pool_in_use`).
struct Fixture {
    tv: TestValkey,
    metrics: Arc<Metrics>,
    store: LayerStore,
    session_id: String,
    object_id: u64,
    /// Session name, layer name and base-path stem all at once — one string so
    /// a test can never hold two of them out of step.
    name: String,
}

impl Fixture {
    async fn new(name: &str, backend: UpdateLayerBackend) -> Self {
        Self::with_store_tuning(name, backend, |store| store).await
    }

    /// [`Fixture::new`] with a hook that tunes the `LayerStore` builder chain.
    ///
    /// Applied AFTER the backend and metrics are wired and BEFORE the fixture's
    /// session and layer are created, which matters for the fault-injection
    /// tests below: `create_layer` must run with the injector already
    /// installed and must provably not burn an entry from it.
    ///
    /// It does not, but the reason changed with b.zvd. `probe_keyspace_types`
    /// used to have exactly two call sites, both inside `update_layer_once`;
    /// `LayerStore::write_object` is now a third, so `create_layer` DOES probe.
    /// What keeps the fixture honest is [`fixture_with_probe_injector`]: it
    /// hands over an *empty* queue and arms it only after construction, so the
    /// fixture's own `create_layer` finds nothing to pop. Every injecting test
    /// then asserts the queue length it observes after its own `update`, so a
    /// call that burned an unexpected entry surfaces as a wrong queue length
    /// rather than as silence.
    ///
    /// The external-value pre-image seam (b.j2x) has the same exposure:
    /// `write_object`'s `HGETALL obj:ext` pops from it, so
    /// `ext_pre_image_read_fault_injection::fixture_with_ext_read_injector`
    /// also hands over an empty queue and arms it after construction.
    async fn with_store_tuning(
        name: &str,
        backend: UpdateLayerBackend,
        tune: impl FnOnce(LayerStore) -> LayerStore,
    ) -> Self {
        // pool_size = 1: see the module header — this is what makes the
        // slot-hygiene assertion name the same connection the refused attempt
        // borrowed.
        let tv = TestValkey::setup_with_pool_size(1).await;
        let metrics = Arc::new(crate::common::metrics_scrape_only("test"));
        tv.client()
            .install_observability_hooks(metrics.clone())
            .await;

        let store = tune(
            LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096)
                .with_update_layer_backend(backend)
                .with_metrics(metrics.clone()),
        );
        let session_id = SessionStore::new(tv.client().clone(), tv.key_schema().clone())
            .create(name, &format!("/{name}"), "b459-user")
            .await
            .expect("session creation must succeed");
        let object_id = store
            .create_layer(
                &session_id,
                name,
                "b459-user",
                &ObjectState {
                    sequence_num: 1,
                    first_version: 1,
                    source_format: "usda".to_owned(),
                    next_set_order: 1,
                    ..Default::default()
                },
                Arc::new(MockExternalValueStore::new()),
            )
            .await
            .expect("create_layer must succeed");

        Self {
            tv,
            metrics,
            store,
            session_id,
            object_id,
            name: name.to_owned(),
        }
    }

    fn oid_str(&self) -> String {
        self.object_id.to_string()
    }

    fn layer(&self) -> &str {
        &self.name
    }

    async fn update(&self, delta: OwnedDelta, client_ts: u64) -> Result<(u64, u64), ServiceError> {
        self.store
            .update_layer_relayed(
                &self.session_id,
                self.layer(),
                self.object_id,
                Some(delta),
                "b459-writer",
                client_ts,
            )
            .await
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Deltas
// ═════════════════════════════════════════════════════════════════════════════

fn create_section_delta(section_id: u64, name: &str) -> OwnedDelta {
    OwnedDelta {
        struct_commands: vec![StructCommand::CreateSection(CommandCreateSection {
            parent_id: 0,
            section_id,
            section_name: name.to_owned(),
            section_type: 1,
        })],
        ..Default::default()
    }
}

fn delete_section_delta(section_id: u64) -> OwnedDelta {
    OwnedDelta {
        struct_commands: vec![StructCommand::DeleteSection(CommandDeleteSection {
            parent_id: 0,
            section_id,
        })],
        ..Default::default()
    }
}

fn set_field_delta(section_id: u64, key: &str) -> OwnedDelta {
    OwnedDelta {
        set_fields: vec![SetField {
            section_id,
            key_name: key.to_owned(),
            value_or_ext_hash: vec![0x01, 0x02],
            ext_value_size: 0,
            ext_value_index: 0,
            set_order: 0,
        }],
        ..Default::default()
    }
}

fn set_time_sample_delta(section_id: u64, time: f64) -> OwnedDelta {
    OwnedDelta {
        time_samples: vec![OodfTimeSample {
            section_id,
            time,
            value_or_ext_hash: vec![0xAA, 0xBB],
            ext_value_size: 0,
            ext_value_index: 0,
        }],
        ..Default::default()
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Assertions
// ═════════════════════════════════════════════════════════════════════════════

/// The wrong-typed-key refusal surface, asserted in full.
///
/// Deliberately stricter than `update_layer_lua_test.rs`'s
/// `assert_wrongtype_key_error` (which checks the prefix, the tag and two
/// substrings): this one pins the message OPENING byte for byte and requires
/// the actual type as well as the expected one. Both backends are asserted
/// through this ONE helper, which is what makes the parity claim exact rather
/// than approximate — two helpers could drift and the parity test would still
/// pass.
fn assert_keyspace_type_error(err: &ServiceError, key: &str, expected_type: &str) {
    let ServiceError::Internal(msg) = err else {
        panic!(
            "a wrong-typed mutated key must be a NON-RETRIABLE `Internal`. `Aborted` would spin \
             the retry loop against a condition no retry can clear (the key is still the wrong \
             type on the next attempt) and `FailedPrecondition` would advertise a client-side \
             recovery that cannot help — only an operator deleting the key can. Got: {err:?}"
        );
    };
    assert!(
        msg.starts_with(KEYSPACE_TYPE_ERROR_PREFIX),
        "the refusal must open with {KEYSPACE_TYPE_ERROR_PREFIX:?} — that prefix is what routes \
         an operator to the keyspace-corruption runbook rather than to the stream-collision one, \
         and it is the string `mapper_routing_test`'s allowlist and the Lua-path assertions are \
         both written against; got: {msg:?}"
    );
    assert!(
        msg.contains(key),
        "the refusal must NAME the offending key — that name is the whole remediation, and \
         without it an operator has a layer that refuses every write and no way to find out \
         why; expected {key:?} in: {msg:?}"
    );
    assert!(
        msg.contains("holds a string"),
        "the refusal must say what the key actually holds, so an operator can tell an \
         out-of-band writer's leftover from a bug in this service; got: {msg:?}"
    );
    assert!(
        msg.contains(&format!("expected {expected_type}")),
        "the refusal must say which type the commit needed; expected {expected_type:?} in: \
         {msg:?}"
    );
    assert!(
        !msg.contains("STREAM_ID_COLLISION"),
        "the stream is healthy here — reporting this as a stream-ID collision would send an \
         operator at the wrong key entirely; got: {msg:?}"
    );
}

/// `live_edit_keyspace_type_error_total{backend, site}`, summed over the
/// `mode` series the `{backend="lua", site="update"}` cell is split into.
fn keyspace_type_errors(metrics: &Metrics, backend: &str, site: &str) -> u64 {
    sample_sum(
        &render(metrics),
        "live_edit_keyspace_type_error_total",
        &[("backend", backend), ("site", site)],
    )
    .unwrap_or_else(|| {
        panic!(
            "live_edit_keyspace_type_error_total{{backend=\"{backend}\", site=\"{site}\"}} is \
             absent from the exposition. Both cells are zero-seeded in `Metrics::new`, so \
             absence means the store never received this metrics handle — every count read \
             from it would be vacuous, and an alert on a series that only appears at incident \
             time has no derivative to fire on"
        )
    }) as u64
}

/// `live_edit_valkey_txn_dirty_checkouts_total`, every `reason` cell at once.
///
/// Read as a whole rather than one cell: the assertion is "the checkout probe
/// found nothing wrong", and a verdict filing itself under a sibling reason is
/// exactly what a single-cell read would miss.
fn txn_dirty_checkouts(metrics: &Metrics) -> Vec<(&'static str, u64)> {
    let exposition = render(metrics);
    TXN_DIRTY_REASONS
        .iter()
        .map(|reason| {
            let value = sample_value(
                &exposition,
                "live_edit_valkey_txn_dirty_checkouts_total",
                &[("reason", *reason)],
            )
            .unwrap_or_else(|| {
                panic!(
                    "live_edit_valkey_txn_dirty_checkouts_total{{reason=\"{reason}\"}} is absent \
                     from the exposition — this is not the registry the client reports into"
                )
            });
            (*reason, value as u64)
        })
        .collect()
}

fn pool_in_use(metrics: &Metrics, pool: &str) -> i64 {
    sample_value(
        &render(metrics),
        "live_edit_valkey_pool_in_use",
        &[("pool", pool)],
    )
    .unwrap_or_else(|| panic!("live_edit_valkey_pool_in_use{{pool=\"{pool}\"}} must be present"))
        as i64
}

/// The refused attempt must have handed its `txn` slot back clean: it
/// `UNWATCH`ed and never opened a `MULTI`.
///
/// Observed the b.bhb way — through the next borrower's b.7td checkout probe
/// (`DISCARD` then `UNWATCH`, classifying the replies). A `+OK` to `DISCARD`
/// means a `MULTI` was left open (`reason="leaked_multi"`); a reply matching
/// neither expectation means the connection's reply stream is offset
/// (`reason="unexpected_reply"`). Both tick the counter, so the counter staying
/// flat across the checkout is the assertion.
///
/// A leaked `WATCH` specifically is NOT observable this way and this helper does
/// not claim it: the probe's `UNWATCH` answers `+OK` whether or not a watch set
/// was live and nothing counts the difference.
async fn assert_txn_slot_clean(fx: &Fixture) {
    let before = txn_dirty_checkouts(&fx.metrics);
    let header_key = fx.tv.key_schema().obj_header(&fx.oid_str());
    let slot = fx.tv.client().acquire_txn_for(&header_key).await;
    let after = txn_dirty_checkouts(&fx.metrics);

    // Vacuity guard: the gauge is published through the same snapshot the
    // dirty counter ticks through, so a nonzero reading here proves the flat
    // counter means "the probe found nothing", not "this client reports into
    // no registry at all".
    assert_eq!(
        pool_in_use(&fx.metrics, "txn"),
        1,
        "the txn slot is held right now, so valkey_pool_in_use{{pool=\"txn\"}} must read 1. A 0 \
         means the observability hooks are not wired to this client and the dirty-checkout \
         assertion below cannot fail for any reason"
    );
    assert_eq!(
        after, before,
        "the checkout after the refused commit found the txn slot DIRTY. On a one-slot pool \
         this is the same connection the refusal borrowed, so the probe is reporting residue \
         that attempt left: reason=\"leaked_multi\" means it went back to the pool with a MULTI \
         open, reason=\"unexpected_reply\" means its reply stream is offset. b.459's refusal \
         path must `UNWATCH` and return WITHOUT opening a MULTI — either reading says it did \
         something else"
    );

    drop(slot);
}

// ═════════════════════════════════════════════════════════════════════════════
// Keyspace witnesses
// ═════════════════════════════════════════════════════════════════════════════

/// Every object-scoped key this file's deltas could touch, plus `extra`.
///
/// Enumerated explicitly rather than `SCAN`ned: a `SCAN` on a clustered client
/// walks one shard's slot range, and these tests are not cluster-gated. An
/// explicit list is also the more honest assertion — it names the sibling keys
/// a half-commit would have written.
fn witness_keys(fx: &Fixture, extra: &[String]) -> Vec<String> {
    let keys = fx.tv.key_schema();
    let oid = fx.oid_str();
    let mut all = vec![
        keys.obj_header(&oid),
        keys.layer_stream(fx.object_id),
        keys.obj_sections_index(&oid),
        keys.obj_children_index(&oid),
        keys.obj_deleted_sections_index(&oid),
        keys.obj_fields_index(&oid),
        keys.obj_time_samples_index(&oid),
        keys.obj_children(&oid, "0"),
    ];
    all.extend_from_slice(extra);
    all
}

/// `DUMP` every key, so the comparison is over the serialized VALUE rather than
/// over a per-type reader.
///
/// `DUMP` is type-agnostic and returns `nil` for an absent key, which is what
/// makes one witness cover "this hash did not change", "this key was not
/// created" and "no stream entry was appended" in a single before/after
/// equality — including the columns a targeted reader would forget to look at
/// (`last_server_ts`, `next_set_order`).
async fn dump_witness(fx: &Fixture, keys: &[String]) -> Vec<(String, Option<Vec<u8>>)> {
    let mut out = Vec::with_capacity(keys.len());
    for key in keys {
        let dumped: Option<Vec<u8>> = fx
            .tv
            .client()
            .pool
            .dump(key)
            .await
            .expect("DUMP must succeed");
        out.push((key.clone(), dumped));
    }
    out
}

/// The object header's `seq`, read directly — the human-readable half of the
/// witness comparison.
async fn header_seq(fx: &Fixture) -> u64 {
    let header: HashMap<String, String> = fx
        .tv
        .client()
        .pool
        .hgetall(fx.tv.key_schema().obj_header(&fx.oid_str()))
        .await
        .expect("HGETALL obj header must succeed");
    header
        .get("seq")
        .expect("a live object header must carry `seq`")
        .parse()
        .expect("header seq must be numeric")
}

async fn stream_len(fx: &Fixture) -> u64 {
    fx.tv
        .client()
        .pool
        .xlen(fx.tv.key_schema().layer_stream(fx.object_id))
        .await
        .expect("XLEN must succeed")
}

/// Every entry ID on the object's live stream, oldest first.
///
/// `stream_len` answers "how many", this answers "which" — the difference
/// matters for the retry tests below, where the claim is not just that the
/// stream grew by one but that it grew by the ONE id the commit's sequence
/// names. A retried attempt that published twice under two different ids would
/// pass a length check taken against the wrong baseline; it cannot pass this.
async fn stream_entry_ids(fx: &Fixture) -> Vec<String> {
    let entries: Vec<(String, HashMap<String, bytes::Bytes>)> = fx
        .tv
        .client()
        .pool
        .xrange_values(
            fx.tv.key_schema().layer_stream(fx.object_id),
            "-",
            "+",
            None,
        )
        .await
        .expect("XRANGE must succeed");
    entries.into_iter().map(|(id, _)| id).collect()
}

async fn plant_string(fx: &Fixture, key: &str) {
    let _: () = fx
        .tv
        .client()
        .pool
        .set(key, "planted-by-an-out-of-band-writer", None, None, false)
        .await
        .unwrap_or_else(|e| panic!("planting a string at {key} must succeed: {e}"));
}

/// Strip the `chs_complete` marker `create_layer` writes, leaving the header an
/// object materialized before the marker existed carries. On such an object
/// the structural path's snapshot read never reads `obj:chs`, and reads the
/// root's and every live section's children list whether `obj:chs` names it or
/// not (Spec SDD SR-1.2).
async fn unmark(fx: &Fixture) {
    let removed: u64 = fx
        .tv
        .client()
        .pool
        .hdel(
            fx.tv.key_schema().obj_header(&fx.oid_str()),
            HEADER_FIELD_CHS_COMPLETE,
        )
        .await
        .expect("HDEL obj:hdr chs_complete must succeed");
    assert_eq!(
        removed, 1,
        "fixture precondition: the object must have been marked, or it was never the shape this \
         test simulates"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// The shared refusal scenario
// ═════════════════════════════════════════════════════════════════════════════

/// Plant a string at `planted_key`, drive `delta`, and assert the full
/// whole-commit refusal contract.
///
/// The five claims, and why each is separately load-bearing:
///
/// 1. **The error surface** — non-retriable `Internal`, naming key / actual /
///    expected. Anything retriable spins the loop against a standing condition.
/// 2. **The keyspace is bit-for-bit unchanged** (`DUMP` over every key the
///    commit would have touched). This is the claim the ticket exists for: the
///    pre-fix failure was not "the commit errored", it was "the commit errored
///    *after* writing some of itself".
/// 3. **`seq` did not advance and no stream entry was appended** — the same
///    fact as (2) in the two places a reviewer will look first, and the one a
///    subscriber would have observed.
/// 4. **The `txn` slot came back clean** — the refusal returns through an
///    `UNWATCH`, not by abandoning an open transaction on a pooled connection.
/// 5. **The counter ticked exactly once** — once per refused commit, not once
///    per probed key, and on the cell whose alert value rests on a zero
///    baseline.
async fn assert_commit_refused_whole(
    fx: &Fixture,
    planted_key: &str,
    expected_type: &str,
    delta: OwnedDelta,
    client_ts: u64,
    backend_label: &str,
) -> ServiceError {
    plant_string(fx, planted_key).await;

    let keys = witness_keys(fx, &[planted_key.to_owned()]);
    let before = dump_witness(fx, &keys).await;
    let seq_before = header_seq(fx).await;
    let stream_before = stream_len(fx).await;
    let counter_before = keyspace_type_errors(&fx.metrics, backend_label, "update");

    let err = fx
        .update(delta, client_ts)
        .await
        .expect_err("a commit over a wrong-typed mutated key must be refused");
    assert_keyspace_type_error(&err, planted_key, expected_type);

    let after = dump_witness(fx, &keys).await;
    assert_eq!(
        after, before,
        "a refused commit must leave the object's keyspace BIT-FOR-BIT unchanged. A difference \
         here is the half-commit b.459 exists to close: some queued command landed before the \
         wrong-typed one was refused, so materialized state now disagrees with what the client \
         was told and (if the stream moved) with what subscribers were told"
    );
    assert_eq!(
        header_seq(fx).await,
        seq_before,
        "the header sequence must not advance for a commit that did not happen — a client \
         reading it back would see a version it can never fetch the content of"
    );
    assert_eq!(
        stream_len(fx).await,
        stream_before,
        "no stream entry may be appended: the probe runs before `MULTI`, so subscribers must \
         never be told about this commit"
    );

    assert_txn_slot_clean(fx).await;

    assert_eq!(
        keyspace_type_errors(&fx.metrics, backend_label, "update"),
        counter_before + 1,
        "exactly one refusal must tick \
         live_edit_keyspace_type_error_total{{backend=\"{backend_label}\", site=\"update\"}}. \
         Zero leaves operators blind to a layer that has silently stopped accepting writes; \
         more than one means the probe is counting per key rather than per commit and the \
         alert's rate is meaningless"
    );

    err
}

// ═════════════════════════════════════════════════════════════════════════════
// Structural-path families
// ═════════════════════════════════════════════════════════════════════════════

/// b.459: a string on the section hash a `CreateSection` will `HSET` refuses the
/// whole commit.
///
/// This is the regression in its original shape — it is the exact staging
/// `b_bhb_tests`'s `ExecError` test used before b.459, where it reached the
/// `EXEC` and tore the commit (the `XADD` landed, the `HSET` did not). The
/// section key is brand new, so it is in no index and nothing before the probe
/// reads it: the probe is the only thing standing between this keyspace and a
/// published half-commit.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn txn_wrongtype_new_section_key_refuses_whole_commit() {
    let fx = Fixture::new("b459-section", UpdateLayerBackend::Txn).await;
    let key = fx
        .tv
        .key_schema()
        .obj_section(&fx.oid_str(), &SECTION_ID.to_string());

    assert_commit_refused_whole(
        &fx,
        &key,
        "hash",
        create_section_delta(SECTION_ID, "Refused"),
        1_000,
        "txn",
    )
    .await;
}

/// b.459: a string on `obj:chs` refuses the whole commit.
///
/// On an object without the `chs_complete` marker, the children-list
/// parent-id index is the one key in the mutated set that **nothing** on this
/// path reads first: the snapshot read enumerates parents from the sections
/// index plus the synthetic root there and never touches `obj:chs`. So unlike
/// the section/field/time-sample indices there is no second guard behind the
/// probe — pre-b.459 a string on this key was a guaranteed tear on every
/// structural commit. Staged unmarked for that reason; on a marked object the
/// snapshot read reads `obj:chs` and refuses first
/// ([`txn_wrongtype_children_index_of_a_marked_object_is_caught_by_the_snapshot_read_not_the_probe`]).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn txn_wrongtype_children_index_refuses_whole_commit() {
    let fx = Fixture::new("b459-chs", UpdateLayerBackend::Txn).await;
    unmark(&fx).await;
    let key = fx.tv.key_schema().obj_children_index(&fx.oid_str());

    assert_commit_refused_whole(
        &fx,
        &key,
        "set",
        create_section_delta(SECTION_ID, "Refused"),
        1_000,
        "txn",
    )
    .await;
}

/// A string on `obj:chs` of a MARKED object is refused by the snapshot read,
/// not by the probe — and counted on the read's own counter.
///
/// On an object carrying the `chs_complete` marker the structural path's
/// snapshot read trusts `obj:chs` to name the children lists (Spec SDD SR-1.2),
/// so it counts the key (`SCARD`, before paging it) before the probe runs and
/// refuses on the `WRONGTYPE`. The verdict is the probe's — fail closed, before
/// `MULTI`, nothing written — but the tells are the snapshot read's: the
/// refusal names the read's `SCARD children index`, carries no `WRONGTYPE_KEY:` tag, leaves
/// `keyspace_type_error_total` flat, and ticks exactly
/// `snapshot_read_type_error_total{key_family="children_index"}` — the cell an
/// operator alerting on read-side refusals watches.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn txn_wrongtype_children_index_of_a_marked_object_is_caught_by_the_snapshot_read_not_the_probe(
) {
    let fx = Fixture::new("b459-chs-marked", UpdateLayerBackend::Txn).await;
    let key = fx.tv.key_schema().obj_children_index(&fx.oid_str());
    plant_string(&fx, &key).await;

    let keys = witness_keys(&fx, std::slice::from_ref(&key));
    let before = dump_witness(&fx, &keys).await;
    let seq_before = header_seq(&fx).await;

    let err = fx
        .update(create_section_delta(SECTION_ID, "Refused"), 1_000)
        .await
        .expect_err(
            "a marked object's snapshot read reads `obj:chs`, so a wrong-typed index must fail \
             the commit closed",
        );

    assert_refused_by_the_snapshot_read(&fx, &err, "SCARD children index");
    assert_only_family_ticked(&fx.metrics, "children_index", 1);
    assert_eq!(
        dump_witness(&fx, &keys).await,
        before,
        "the refusal must happen before `MULTI` whichever guard produced it — a difference here \
         means the commit half-landed: {err:?}"
    );
    assert_eq!(
        header_seq(&fx).await,
        seq_before,
        "the header must not advance for a commit that did not happen"
    );
}

/// b.459: a string on the tombstone hash a `DeleteSection` will `HSET` refuses
/// the whole commit.
///
/// The tombstone key of a section that has not been deleted yet is absent from
/// `obj:dels`, so — like the section key above — no read before the probe
/// touches it. The delta is a real cascade delete driven against a section this
/// test first commits, so the mutation list is the production one:
/// `SetChildren` on the parent, `DeleteSection`, `CreateTombstone`,
/// `DeleteChildrenList`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn txn_wrongtype_new_tombstone_key_refuses_whole_commit() {
    let fx = Fixture::new("b459-tomb", UpdateLayerBackend::Txn).await;
    fx.update(create_section_delta(SECTION_ID, "Doomed"), 1_000)
        .await
        .expect("the setup commit must land on a clean keyspace");

    let key = fx
        .tv
        .key_schema()
        .obj_deleted_section(&fx.oid_str(), &SECTION_ID.to_string());

    assert_commit_refused_whole(
        &fx,
        &key,
        "hash",
        delete_section_delta(SECTION_ID),
        2_000,
        "txn",
    )
    .await;
}

// ═════════════════════════════════════════════════════════════════════════════
// The `!needs_snapshot` txn-knob path
// ═════════════════════════════════════════════════════════════════════════════

/// b.459: a string on the per-field hash refuses the whole commit on the
/// txn-knob path.
///
/// A field-only delta is fast-path-shaped (`lua::is_fast_path` is true), so it
/// reaches the WATCH/MULTI/EXEC path only under
/// `LIVE_EDIT_UPDATE_LAYER_BACKEND=txn` — the rollback knob. That branch loads
/// **no** snapshot at all, so the type probe is the only pre-`MULTI` guard: it
/// is both the first and the last, and it is the branch that pays the one extra
/// round trip for it. The external-value pre-image read beside it also reads
/// this row, but it is not a guard: its result is consulted only after the
/// probe has passed (b.j2x), which
/// `ext_pre_image_read_fault_injection::txn_wrongtype_field_key_refuses_tagged_even_when_the_pre_image_read_fails_first`
/// pins.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn txn_wrongtype_new_field_key_refuses_whole_commit() {
    let fx = Fixture::new("b459-field", UpdateLayerBackend::Txn).await;
    let key = fx
        .tv
        .key_schema()
        .obj_field(&fx.oid_str(), &SECTION_ID.to_string(), "k");

    assert_commit_refused_whole(
        &fx,
        &key,
        "hash",
        set_field_delta(SECTION_ID, "k"),
        1_000,
        "txn",
    )
    .await;
}

/// b.459: a string on `obj:flds` refuses the whole commit on the txn-knob path.
///
/// The index half of the pair above. On the *structural* path this key is
/// `ZRANGE`d by the snapshot read first, so that guard would fire instead;
/// here there is no snapshot, which is precisely why the probe has to cover it.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn txn_wrongtype_fields_index_refuses_whole_commit() {
    let fx = Fixture::new("b459-flds", UpdateLayerBackend::Txn).await;
    let key = fx.tv.key_schema().obj_fields_index(&fx.oid_str());

    assert_commit_refused_whole(
        &fx,
        &key,
        "zset",
        set_field_delta(SECTION_ID, "k"),
        1_000,
        "txn",
    )
    .await;
}

/// b.459: a string on the per-time-sample hash refuses the whole commit.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn txn_wrongtype_new_time_sample_key_refuses_whole_commit() {
    let fx = Fixture::new("b459-ts", UpdateLayerBackend::Txn).await;
    let key =
        fx.tv
            .key_schema()
            .obj_time_sample(&fx.oid_str(), &SECTION_ID.to_string(), SAMPLE_TIME);

    assert_commit_refused_whole(
        &fx,
        &key,
        "hash",
        set_time_sample_delta(SECTION_ID, SAMPLE_TIME),
        1_000,
        "txn",
    )
    .await;
}

/// b.459: a string on `obj:tss` refuses the whole commit.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn txn_wrongtype_time_samples_index_refuses_whole_commit() {
    let fx = Fixture::new("b459-tss", UpdateLayerBackend::Txn).await;
    let key = fx.tv.key_schema().obj_time_samples_index(&fx.oid_str());

    assert_commit_refused_whole(
        &fx,
        &key,
        "zset",
        set_time_sample_delta(SECTION_ID, SAMPLE_TIME),
        1_000,
        "txn",
    )
    .await;
}

/// b.459 × b.j2x: a string on `obj:ext` refuses the whole commit on the
/// txn-knob path, with the probe's tagged error.
///
/// `collect_probe_keys` names `obj:ext` for any value-row mutation (b.qpf), so
/// the probe covers it. On this branch the external-value pre-image read can
/// read the same key at the same time — one `HGET obj:ext <hash>` per external
/// hash the commit touches (b.26e Stage 2; this inline write touches none, so
/// here it does not) — and such a read meets the string too. Until b.j2x the
/// two sat under one fail-fast join, so whichever error
/// arrived first was the refusal: sometimes the probe's tagged message, its
/// tick and its remediation ERROR line, sometimes the read's untagged
/// `WRONGTYPE` with none of them. The read is now consulted only after the
/// probe has passed, so the tagged refusal is the only possible outcome.
///
/// Against a real key this can only show the probe winning on this run. The
/// deterministic form, where the read is made to fail first every time, is
/// `ext_pre_image_read_fault_injection::txn_wrongtype_ext_index_refuses_tagged_even_when_the_pre_image_read_fails_first`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn txn_wrongtype_ext_index_refuses_whole_commit() {
    let fx = Fixture::new("b459-ext", UpdateLayerBackend::Txn).await;
    let key = fx.tv.key_schema().obj_ext_index(&fx.oid_str());

    assert_commit_refused_whole(
        &fx,
        &key,
        "hash",
        set_field_delta(SECTION_ID, "k"),
        1_000,
        "txn",
    )
    .await;
}

// ═════════════════════════════════════════════════════════════════════════════
// Negative control
// ═════════════════════════════════════════════════════════════════════════════

/// b.459: the identical deltas over a CLEAN keyspace commit, and the alertable
/// cell keeps its zero baseline.
///
/// Without this the whole file is unfalsifiable in one direction: a probe that
/// refused *every* commit would pass every test above. It also pins the cost
/// side of the fix — the probe adds a round trip to the structural path and one
/// to the knob path, and it must add nothing else.
///
/// All three delta shapes run against one layer so the assertion covers the
/// probe's behaviour on a keyspace that is progressively less empty (the second
/// and third commits probe index keys the first one created).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn txn_clean_keyspace_commits_and_never_ticks_the_type_error() {
    let fx = Fixture::new("b459-clean", UpdateLayerBackend::Txn).await;

    let (seq_a, _) = fx
        .update(create_section_delta(SECTION_ID, "Kept"), 1_000)
        .await
        .expect("a structural commit over a clean keyspace must land");
    let (seq_b, _) = fx
        .update(set_field_delta(SECTION_ID, "k"), 2_000)
        .await
        .expect("a field commit over a clean keyspace must land");
    let (seq_c, _) = fx
        .update(set_time_sample_delta(SECTION_ID, SAMPLE_TIME), 3_000)
        .await
        .expect("a time-sample commit over a clean keyspace must land");

    assert_eq!(
        (seq_a, seq_b, seq_c),
        (2, 3, 4),
        "the three commits must take consecutive sequences after CreateLayer's 1 — a gap would \
         mean one of them was refused and retried behind the scenes"
    );
    assert_eq!(
        stream_entry_ids(&fx).await,
        vec!["1-0", "2-0", "3-0", "4-0"],
        "CreateLayer's entry plus exactly one per commit, each at the id its own sequence names. \
         Asserted on the IDs rather than the count: a length of 4 is also what a stream with a \
         duplicate and a missing entry looks like, and the whole point of the negative control \
         is that the probe changed nothing about what subscribers see"
    );
    assert_eq!(
        header_seq(&fx).await,
        4,
        "the header must carry the last committed sequence"
    );
    assert_eq!(
        keyspace_type_errors(&fx.metrics, "txn", "update"),
        0,
        "nothing about these commits is a keyspace anomaly. The counter's entire alert value \
         rests on a zero baseline in steady state: a probe that ticked on healthy traffic would \
         make it unpageable"
    );
    assert_txn_slot_clean(&fx).await;
}

// ═════════════════════════════════════════════════════════════════════════════
// The two deliberate exclusions
// ═════════════════════════════════════════════════════════════════════════════

/// The refusal shape a *snapshot read* produces, as distinct from the probe's.
///
/// Both guards reach the same verdict — fail closed, before `MULTI`, nothing
/// written — and which one fires is an implementation detail this file does not
/// pin as a contract. What it does pin is the pair of tells that tell them
/// apart, because confusing the two would send an operator to the wrong runbook
/// and would hide a probe that had silently started covering a key it must not:
///
/// * no `WRONGTYPE_KEY:` tag — that tag is minted only by
///   `scripts/keyspace_type_probe.lua`, so its presence would mean the probe ran
///   and refused;
/// * `keyspace_type_error_total{backend="txn", site="update"}` still zero — the
///   probe is the only writer of that cell.
///
/// `expected_site` — the operation the snapshot read names in its refusal, such
/// as `LLEN children list (parent 7)` — is what makes the pair of tells
/// attributable rather than merely consistent. Both bullets above are satisfied
/// by *any* `Internal` the snapshot read can raise — a wrong-typed `obj:secs`,
/// a wrong-typed `obj:hdr`, a fault three read-tiers away — so without pinning
/// the specific read that met the planted key, a caller would still pass if
/// that key were never reached and something else refused first. For a
/// children list it mirrors `b_zvd_tests`, where the same substring is the
/// read's one piece of locating information.
fn assert_refused_by_the_snapshot_read(fx: &Fixture, err: &ServiceError, expected_site: &str) {
    let ServiceError::Internal(msg) = err else {
        panic!(
            "a wrong-typed key the structural path's snapshot read reads is a permanent \
             condition no retry can clear, so the refusal must be a non-retriable `Internal`; \
             got: {err:?}"
        );
    };
    assert!(
        msg.contains(expected_site),
        "the refusal must be the one this test staged — {expected_site:?}. Any other message \
         means the planted key was never reached and some unrelated read refused first, which \
         would leave the two tells below asserting nothing about this staging; got: {msg:?}"
    );
    assert!(
        !msg.contains("WRONGTYPE_KEY:"),
        "the `WRONGTYPE_KEY:` tag is minted only by the keyspace type probe, and this key is \
         read by the snapshot read BEFORE the probe runs — seeing the tag means the probe was \
         reached, i.e. the snapshot read swallowed the type error again; got: {msg:?}"
    );
    assert_eq!(
        keyspace_type_errors(&fx.metrics, "txn", "update"),
        0,
        "the probe is the only writer of \
         live_edit_keyspace_type_error_total{{backend=\"txn\", site=\"update\"}}, so a tick here \
         says the refusal came from the probe rather than from the snapshot read this key is \
         covered by"
    );
}

/// b.459 exclusion #1, post-b.zvd: the children LIST key is still not probed —
/// but the snapshot read now fails closed on it before the probe is reached.
///
/// # The exclusion itself is unchanged
///
/// `SetChildren` rebuilds the list from scratch — `DEL k` then `RPUSH k …` — so
/// the `RPUSH` lands on a key that no longer exists whatever type it held
/// before. Probing `obj:ch:{parent}` would convert a self-healing case into a
/// write outage, which is why `mutation_probe_keys` (and, since b.zvd,
/// `create_probe_keys`) leave it out and why `ProbedType` still has no `List`
/// variant. That claim now rests on the pure unit tests beside those functions
/// in `src/storage/layer.rs` — it is compile-cheap and staging-free there.
///
/// # Why this test no longer asserts the heal
///
/// Pre-b.zvd the commit reached the probe at all only because
/// the snapshot loader of the time (`load_object_snapshot`, since deleted)
/// read every children list as
/// `pool.lrange(...).await.unwrap_or_default()`: the planted string made that
/// `LRANGE` answer `WRONGTYPE`, the swallow turned it into "no children", and
/// the commit sailed on to the probe, which passed it over and let the
/// `DEL`-then-`RPUSH` repair the key. That swallow was the defect b.zvd fixed —
/// a *transient* fred error on that `LRANGE` was equally indistinguishable from
/// an empty list, so a structural commit could be computed against a silently
/// truncated children view and then `DEL` a real children list under a fully
/// valid `WATCH`.
///
/// The staging puts the planted key where the structural path's snapshot read
/// reads it: the setup commit gives the root a child, so on this marked object
/// `obj:chs` names the root's list and the read reads it (Spec SDD SR-1.2) —
/// its length first (`LLEN`), then a ranged `LRANGE`, so the plant is refused
/// at the `LLEN`. The read therefore refuses first, and the heal is unreachable through
/// `update_layer` for such a key: every commit that emits `SetChildren` reads
/// the snapshot. What is left to assert behaviourally is the boundary — the
/// same shape [`txn_wrongtype_deleted_sections_index_is_caught_by_the_snapshot_read_not_the_probe`]
/// asserts for the three index keys. A list the read does NOT cover — on a
/// marked object, one `obj:chs` does not name — reaches the commit, whose `DEL`
/// + `RPUSH` heals it:
/// [`txn_wrongtype_children_list_the_marked_index_does_not_name_is_healed_by_the_commit`].
///
/// Do not "fix" a failure here by re-adding `obj:ch:*` to `mutation_probe_keys`:
/// the exclusion is still correct, and probing the key would refuse the commit
/// one round trip later for the wrong reason.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn txn_wrongtype_children_list_key_is_caught_by_the_snapshot_read_not_the_probe() {
    let fx = Fixture::new("b459-chlist", UpdateLayerBackend::Txn).await;
    fx.update(create_section_delta(SECTION_ID, "First"), 1_000)
        .await
        .expect("the setup commit must land on a clean keyspace");
    let list_key = fx.tv.key_schema().obj_children(&fx.oid_str(), "0");
    plant_string(&fx, &list_key).await;

    let keys = witness_keys(&fx, std::slice::from_ref(&list_key));
    let before = dump_witness(&fx, &keys).await;
    let seq_before = header_seq(&fx).await;

    let err = fx
        .update(create_section_delta(SECOND_SECTION_ID, "Refused"), 2_000)
        .await
        .expect_err(
            "a wrong-typed children LIST key the snapshot read reads must fail the commit closed \
             (b.zvd). An `Ok` here means the snapshot read is swallowing the type error again — \
             and the same swallow turns a transient `LRANGE` failure into a silently childless \
             snapshot the commit then writes back",
        );

    assert_refused_by_the_snapshot_read(&fx, &err, "LLEN children list (parent 0)");
    assert_eq!(
        dump_witness(&fx, &keys).await,
        before,
        "the refusal must happen before `MULTI` whichever guard produced it — a difference here \
         means the commit half-landed: {err:?}"
    );
    assert_eq!(
        header_seq(&fx).await,
        seq_before,
        "the header must not advance for a commit that did not happen"
    );
}

/// b.459 exclusion #2, post-b.zvd: a key the commit only `DEL`s is still not
/// probed — but this staging's key is read by the snapshot first, so the
/// refusal is the snapshot read's.
///
/// # The exclusion itself is unchanged
///
/// `DeleteChildrenList` removes `obj:ch:{sid}` with a bare `DEL`, which succeeds
/// against a value of any type. Probing it would refuse a commit whose entire
/// effect on that key is to delete the offending value — the operator would be
/// told to clean up by hand the exact thing the service was about to clean up
/// itself. That claim rests on the pure unit test
/// `src/storage/layer.rs::tests::delete_mutations_do_not_probe_the_record_keys_they_del`.
///
/// # Why this test no longer asserts the commit
///
/// The object is unmarked, so the structural path's snapshot read walks the
/// root plus every live section (Spec SDD SR-1.2): section 7 is live in
/// `obj:secs` when the string is planted, so `obj:ch:7` IS in that walk.
/// Pre-b.zvd the walk read `pool.lrange(...).await.unwrap_or_default()` and
/// turned the `WRONGTYPE` into an empty list — which happens to be the truth
/// for a leaf — so the commit reached the probe, was passed over, and `DEL`'d
/// the planted string. b.zvd classified that read (`src/storage/diff_read.rs`),
/// because the same swallow hid a *transient* `LRANGE` failure behind an
/// identical "no children" answer, so the refusal now comes one guard earlier.
///
/// On a MARKED object the read takes its lists from `obj:chs`, which does not
/// name a leaf's list, so the planted key is never read and the commit deletes
/// it — the behavioural form, pinned by
/// [`txn_wrongtype_unindexed_leaf_list_the_commit_only_dels_is_healed_on_a_marked_object`].
/// Every other `DEL`-only key family (`DeleteSection`'s section hash,
/// `DeleteField`'s field hash, `DeleteTimeSample`'s sample hash) must already
/// be live in its index for the mutation to be emitted at all, and
/// wrong-typing a key the snapshot reads makes the snapshot read fail first
/// there.
///
/// Do not "fix" a failure here by adding the `DEL`-only keys to
/// `mutation_probe_keys` — the exclusion stays correct, only the staging changed
/// which guard answers.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn txn_wrongtype_key_the_commit_only_dels_is_caught_by_the_snapshot_read_not_the_probe() {
    let fx = Fixture::new("b459-delonly", UpdateLayerBackend::Txn).await;
    unmark(&fx).await;
    fx.update(create_section_delta(SECTION_ID, "Doomed"), 1_000)
        .await
        .expect("the setup commit must land on a clean keyspace");

    let leaf_children_key = fx
        .tv
        .key_schema()
        .obj_children(&fx.oid_str(), &SECTION_ID.to_string());
    plant_string(&fx, &leaf_children_key).await;

    let keys = witness_keys(&fx, std::slice::from_ref(&leaf_children_key));
    let before = dump_witness(&fx, &keys).await;
    let seq_before = header_seq(&fx).await;

    let err = fx
        .update(delete_section_delta(SECTION_ID), 2_000)
        .await
        .expect_err(
            "the leaf's children list is in the snapshot read's parent set, so a wrong-typed \
             value there must fail the commit closed (b.zvd). An `Ok` here means that read is \
             swallowing the type error again",
        );

    assert_refused_by_the_snapshot_read(
        &fx,
        &err,
        &format!("LLEN children list (parent {SECTION_ID})"),
    );
    assert_eq!(
        dump_witness(&fx, &keys).await,
        before,
        "the refusal must happen before `MULTI` whichever guard produced it — a difference here \
         means the commit half-landed: {err:?}"
    );
    assert_eq!(
        header_seq(&fx).await,
        seq_before,
        "the header must not advance for a commit that did not happen"
    );
}

/// The heal's two observable ends: the planted string is gone from `key` —
/// replaced by `expected_list`, or deleted when that is `None` — and no counter
/// moved, because nothing refused anything. Neither guard reached the key: the
/// snapshot read did not read it and the probe does not probe it.
async fn assert_healed(fx: &Fixture, key: &str, expected_list: Option<Vec<String>>) {
    use fred::interfaces::ListInterface;

    match expected_list {
        Some(expected) => {
            let list: Vec<String> =
                fx.tv
                    .client()
                    .pool
                    .lrange(key, 0, -1)
                    .await
                    .unwrap_or_else(|e| {
                        panic!(
                        "{key} must hold a list again — the commit's `DEL` + `RPUSH` rebuilds it \
                         whatever it held: {e}"
                    )
                    });
            assert_eq!(list, expected, "{key} must hold exactly the rebuilt list");
        }
        None => {
            let exists: u64 = fx
                .tv
                .client()
                .pool
                .exists(key)
                .await
                .expect("EXISTS must succeed");
            assert_eq!(exists, 0, "the commit's `DEL` must have removed {key}");
        }
    }
    assert_eq!(
        keyspace_type_errors(&fx.metrics, "txn", "update"),
        0,
        "the probe must not have refused anything: the list key is excluded from it"
    );
    assert_no_family_ticked(&fx.metrics);
}

/// b.459 exclusion #1's heal, reachable on a marked object: a wrong-typed
/// children list `obj:chs` does not name is not read by the structural path's
/// snapshot read, so the commit that rewrites it reaches `MULTI`, and its `DEL`
/// + `RPUSH` repairs the key.
///
/// The fixture's root has no children yet, so `obj:chs` does not name the
/// root's list — on a marked object the snapshot read takes its lists from
/// `obj:chs` (Spec SDD SR-1.2), so a string planted there is never `LRANGE`d.
/// The `CreateSection` under the root then emits `SetChildren` for the root.
/// Whether the read sees the key is the only difference from
/// [`txn_wrongtype_children_list_key_is_caught_by_the_snapshot_read_not_the_probe`],
/// which plants the same key once `obj:chs` names it and is refused.
///
/// A refusal here means either the read reads lists the marked index does not
/// name (the marked children rule broken) or the probe started probing the list
/// key (exclusion #1 broken); a commit that lands with the string still there
/// means the `DEL` before the `RPUSH` is gone.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn txn_wrongtype_children_list_the_marked_index_does_not_name_is_healed_by_the_commit() {
    let fx = Fixture::new("b459-chlist-heal", UpdateLayerBackend::Txn).await;
    let list_key = fx.tv.key_schema().obj_children(&fx.oid_str(), "0");
    plant_string(&fx, &list_key).await;

    let (seq, _) = fx
        .update(create_section_delta(SECTION_ID, "Healer"), 1_000)
        .await
        .unwrap_or_else(|e| {
            panic!(
                "a marked object's snapshot read must not read a list `obj:chs` does not name, \
                 and the probe must not probe the list key, so the commit must land and heal it; \
                 got {e:?}"
            )
        });

    assert_eq!(seq, 2, "the commit must land on the sequence after create");
    assert_healed(&fx, &list_key, Some(vec![SECTION_ID.to_string()])).await;
}

/// b.459 exclusion #2's heal on a marked object: a wrong-typed leaf list the
/// commit only `DEL`s is not read by the snapshot read, so the
/// `DeleteSection`'s `DeleteChildrenList` deletes it.
///
/// Section 7 is a leaf, so it has no children list and `obj:chs` does not name
/// one; the string planted at `obj:ch:7` is a list key the marked read never
/// `LRANGE`s. The unmarked twin, whose read does walk every live section, is
/// [`txn_wrongtype_key_the_commit_only_dels_is_caught_by_the_snapshot_read_not_the_probe`].
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn txn_wrongtype_unindexed_leaf_list_the_commit_only_dels_is_healed_on_a_marked_object() {
    let fx = Fixture::new("b459-delonly-heal", UpdateLayerBackend::Txn).await;
    fx.update(create_section_delta(SECTION_ID, "Doomed"), 1_000)
        .await
        .expect("the setup commit must land on a clean keyspace");
    let leaf_children_key = fx
        .tv
        .key_schema()
        .obj_children(&fx.oid_str(), &SECTION_ID.to_string());
    plant_string(&fx, &leaf_children_key).await;

    let (seq, _) = fx
        .update(delete_section_delta(SECTION_ID), 2_000)
        .await
        .unwrap_or_else(|e| {
            panic!(
                "the marked read must not read the leaf's unindexed list and the probe must not \
                 probe a key the commit only DELs, so the delete must land; got {e:?}"
            )
        });

    assert_eq!(
        seq, 3,
        "the delete must land on the sequence after the setup commit"
    );
    assert_healed(&fx, &leaf_children_key, None).await;
}

// ═════════════════════════════════════════════════════════════════════════════
// Negative scope — the probe must not be over-broad on the knob path
// ═════════════════════════════════════════════════════════════════════════════
//
// Every test above asks "does the probe refuse what it must?". These three ask
// the opposite and equally load-bearing question: **does it refuse only what it
// must?** They are the txn-knob mirror of the Lua path's `saw_field` / `saw_ts`
// gate trio (`update_layer_lua_test::update_layer_lua_field_only_delta_commits_
// despite_a_corrupt_ts_index` and its two siblings), asserted here against the
// SAME keyspaces so the two backends are shown to have the same scope as well as
// the same refusal.
//
// The staging is the `!needs_snapshot` branch — `LIVE_EDIT_UPDATE_LAYER_BACKEND=
// txn` over a fast-path-shaped delta — which is the only place these two index
// keys are reachable at all: on the structural branch the snapshot read
// reads both of them first and fails closed there (see the module header's
// reachability note and
// `txn_wrongtype_deleted_sections_index_is_caught_by_the_snapshot_read_not_the_probe`).
// So this is not a duplicate of the Lua trio in a different colour — it is the
// only backend/branch pair where a scope regression on `obj:flds` / `obj:tss`
// could hide.
//
// The failure mode being fenced is severe and asymmetric: `collect_probe_keys`
// derives its list from the MUTATIONS, so widening it to "every index key the
// object owns" would look tidy and would wedge every write to an object whose
// unrelated index a single out-of-band writer had corrupted — a total write
// outage on a layer, produced by the guard rather than by the corruption.

/// A delta that names nothing — the heartbeat-shaped commit. Its only mutation
/// is the object header, so its probe list is four keys — `obj:hdr` and the
/// three commit-level keys named with every header write: the commit-token
/// record `obj:wtok` (b.75i), the snapshot fence `obj:snapfence` and the
/// writer-intent key `obj:snapintent` — and neither index is on it.
fn empty_delta() -> OwnedDelta {
    OwnedDelta::default()
}

/// The planted string must survive verbatim.
///
/// Two claims in one read: the commit did not *refuse* over this key (asserted
/// by the caller) and it did not *write through* it either. A probe that dropped
/// its scope guard would fail the first; a `ZADD` that ignored the delta's shape
/// would re-type the key and fail the second.
async fn assert_still_planted(fx: &Fixture, key: &str) {
    let raw: Option<String> = fx
        .tv
        .client()
        .pool
        .get(key)
        .await
        .expect("GET on the planted string must succeed");
    assert_eq!(
        raw.as_deref(),
        Some("planted-by-an-out-of-band-writer"),
        "{key} must be untouched by a commit that does not index into it — a different value \
         means the commit wrote through a key its mutation list never named"
    );
}

/// b.459 negative scope: under the txn knob, a FIELD-only delta commits over a
/// corrupt `obj:tss`.
///
/// The delta's mutations are `SetField` + `SetObjectHeader`, so the probe list is
/// `obj:fld:7:k`, `obj:flds` and `obj:hdr`, plus the commit-level keys —
/// `obj:ext` (named for any value-row mutation, b.qpf), and the commit-token
/// record `obj:wtok` (b.75i), the snapshot fence `obj:snapfence` and the
/// writer-intent key `obj:snapintent` (all three named with the header
/// write). The time-samples
/// index is not on it and must not be consulted. This is the mirror of the Lua path's `saw_ts` gate
/// (`update_layer_lua_field_only_delta_commits_despite_a_corrupt_ts_index`),
/// which is guarded by an `if` in the script; here the equivalent guard is that
/// `collect_probe_keys` walks the mutation list rather than the key schema.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn txn_field_only_delta_commits_despite_a_corrupt_time_samples_index() {
    let fx = Fixture::new("b459-scope-ts", UpdateLayerBackend::Txn).await;
    let ts_index_key = fx.tv.key_schema().obj_time_samples_index(&fx.oid_str());
    plant_string(&fx, &ts_index_key).await;

    let (seq, _) = fx
        .update(set_field_delta(SECTION_ID, "k"), 1_000)
        .await
        .expect(
            "an SF-only delta never names the time-samples index, so a corrupt one must not \
             refuse it. Refusing here would be a write outage: every field write to the object \
             would be blocked by a key none of them touch",
        );
    assert_eq!(
        seq, 2,
        "the commit takes the next sequence after CreateLayer"
    );

    // Landed, not merely un-refused.
    let field: HashMap<String, String> = fx
        .tv
        .client()
        .pool
        .hgetall(
            fx.tv
                .key_schema()
                .obj_field(&fx.oid_str(), &SECTION_ID.to_string(), "k"),
        )
        .await
        .expect("HGETALL field must succeed");
    assert_eq!(
        field.get("version").map(String::as_str),
        Some("2"),
        "the field must carry the commit version — a commit that returned Ok without \
         materializing is the tear these guards exist to prevent; got {field:?}"
    );
    assert_eq!(
        stream_entry_ids(&fx).await,
        vec!["1-0", "2-0"],
        "CreateLayer's entry plus this commit's own"
    );

    assert_still_planted(&fx, &ts_index_key).await;
    assert_eq!(
        keyspace_type_errors(&fx.metrics, "txn", "update"),
        0,
        "a corrupt index this delta never writes must not tick the alertable cell — a tick on a \
         commit that succeeded is worse than useless, it pages an operator toward a key the \
         service correctly ignored"
    );
    assert_txn_slot_clean(&fx).await;
}

/// b.459 negative scope: under the txn knob, a TIME-SAMPLE-only delta commits
/// over a corrupt `obj:flds`.
///
/// The mirror of the test above with the classes swapped, and worth having
/// separately for the same reason the Lua pair is: the two index keys enter the
/// probe list from two different `mutation_probe_keys` arms, so one can be
/// widened without touching the other.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn txn_time_sample_only_delta_commits_despite_a_corrupt_fields_index() {
    let fx = Fixture::new("b459-scope-flds", UpdateLayerBackend::Txn).await;
    let fields_index_key = fx.tv.key_schema().obj_fields_index(&fx.oid_str());
    plant_string(&fx, &fields_index_key).await;

    let (seq, _) = fx
        .update(set_time_sample_delta(SECTION_ID, SAMPLE_TIME), 1_000)
        .await
        .expect(
            "an ST-only delta never names the fields index, so a corrupt one must not refuse it. \
             A single corrupt `obj:flds` would otherwise wedge the time-sample traffic that \
             never touches it",
        );
    assert_eq!(
        seq, 2,
        "the commit takes the next sequence after CreateLayer"
    );

    let sample: HashMap<String, String> = fx
        .tv
        .client()
        .pool
        .hgetall(fx.tv.key_schema().obj_time_sample(
            &fx.oid_str(),
            &SECTION_ID.to_string(),
            SAMPLE_TIME,
        ))
        .await
        .expect("HGETALL time sample must succeed");
    assert_eq!(
        sample.get("version").map(String::as_str),
        Some("2"),
        "the time sample must carry the commit version; got {sample:?}"
    );
    assert_eq!(
        stream_entry_ids(&fx).await,
        vec!["1-0", "2-0"],
        "CreateLayer's entry plus this commit's own"
    );

    assert_still_planted(&fx, &fields_index_key).await;
    assert_eq!(
        keyspace_type_errors(&fx.metrics, "txn", "update"),
        0,
        "a corrupt index this delta never writes must not tick the alertable cell"
    );
    assert_txn_slot_clean(&fx).await;
}

/// b.459 negative scope, third clause: an EMPTY delta commits with BOTH indexes
/// corrupt.
///
/// The empty delta's whole mutation list is `SetObjectHeader`, so its probe list
/// is `obj:hdr` plus the three keys named with the header write — the
/// commit-token record `obj:wtok` (b.75i), `obj:snapfence` and `obj:snapintent`
/// — and no index key and no `obj:ext`, since it touches no value row. It still advances the header and publishes a stream entry — the
/// heartbeat-shaped commit clients use to prove liveness — and both corrupt
/// indexes must be irrelevant to it. This is the case that fails first if the
/// probe list is ever derived from the object rather than from the delta.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn txn_empty_delta_commits_despite_both_indexes_corrupt() {
    let fx = Fixture::new("b459-scope-both", UpdateLayerBackend::Txn).await;
    let fields_index_key = fx.tv.key_schema().obj_fields_index(&fx.oid_str());
    let ts_index_key = fx.tv.key_schema().obj_time_samples_index(&fx.oid_str());
    plant_string(&fx, &fields_index_key).await;
    plant_string(&fx, &ts_index_key).await;

    let (seq, _) = fx
        .update(empty_delta(), 1_000)
        .await
        .expect("an empty delta probes neither index and must commit");
    assert_eq!(
        seq, 2,
        "the commit takes the next sequence after CreateLayer"
    );

    assert_eq!(
        header_seq(&fx).await,
        2,
        "the empty commit must advance the header — that advance IS the commit"
    );
    assert_eq!(
        stream_entry_ids(&fx).await,
        vec!["1-0", "2-0"],
        "the heartbeat commit must reach subscribers"
    );

    assert_still_planted(&fx, &fields_index_key).await;
    assert_still_planted(&fx, &ts_index_key).await;
    assert_eq!(
        keyspace_type_errors(&fx.metrics, "txn", "update"),
        0,
        "two corrupt indexes neither of which this delta writes must leave the alertable cell at \
         its zero baseline"
    );
    assert_txn_slot_clean(&fx).await;
}

// ═════════════════════════════════════════════════════════════════════════════
// Backend parity
// ═════════════════════════════════════════════════════════════════════════════

/// b.459: both commit backends refuse the SAME keyspace with the same message,
/// and each ticks its own cell of the same counter.
///
/// # What "the same message" means, exactly
///
/// Both refusals are staged against ONE layer and ONE planted key — the txn
/// probe leaves the keyspace bit-for-bit untouched, so the Lua attempt that
/// follows meets identical conditions — which is what makes a byte-level
/// comparison possible at all. The comparison is asserted as: **the txn message
/// is a prefix of the Lua message, and the remainder is at most one
/// parenthesized locator.**
///
/// That is not a hedge, it is the actual contract. Everything an operator, a log
/// grep, an alert annotation or the `update_layer_lua_test` assertions key on —
/// the `update_layer keyspace type error:` prefix, the `WRONGTYPE_KEY:` tag, the
/// key name, the actual type and the expected type — is byte-identical across
/// two independent implementations (a Rust `format!` over one Lua script's error
/// reply, and a different Lua script's error reply). The Lua path then appends a
/// locator naming *where in the delta* the refusal came from — `(mutation record
/// 1)`, `(fields index)`, `(time-samples index)` — because its probe is
/// interleaved with the apply pass and knows the record index. The txn probe
/// walks a deduped key list where the key IS the locator, so it has no
/// equivalent to add.
///
/// Asserting prefix-plus-parenthetical rather than raw equality keeps both
/// halves of that honest: a txn message that started naming a different key, a
/// different actual/expected type, or a different tag would fail, and so would a
/// Lua suffix that stopped being a parenthetical aside and started carrying
/// contract.
///
/// # Both suffix shapes
///
/// The two stagings below are the per-record key and the shared index key, which
/// is exactly the pair whose Lua locators differ (`(mutation record N)` vs
/// `(fields index)`). One staging would have pinned only one of them.
///
/// # The metric split
///
/// Both refusals were already `Internal` before b.459, so the error surface
/// alone could not tell an operator which path refused. The `{backend}` label
/// is what makes that visible, and the two cells moving in step is the parity
/// claim asserted in production rather than only here.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn txn_and_lua_backends_refuse_an_identical_keyspace_identically() {
    let fx = Fixture::new("b459-parity", UpdateLayerBackend::Txn).await;

    // A second store over the SAME client, keyspace and metrics handle —
    // everything but the backend is held constant.
    let lua_store = LayerStore::new(fx.tv.client().clone(), fx.tv.key_schema().clone(), 4096)
        .with_update_layer_backend(UpdateLayerBackend::Lua)
        .with_metrics(fx.metrics.clone());

    let stagings: Vec<(String, &str)> = vec![
        (
            fx.tv
                .key_schema()
                .obj_field(&fx.oid_str(), &SECTION_ID.to_string(), "k"),
            "hash",
        ),
        (fx.tv.key_schema().obj_fields_index(&fx.oid_str()), "zset"),
    ];

    for (i, (planted_key, expected_type)) in stagings.iter().enumerate() {
        plant_string(&fx, planted_key).await;

        let txn_err = fx
            .update(set_field_delta(SECTION_ID, "k"), 1_000 + i as u64)
            .await
            .expect_err("the txn backend must refuse");
        assert_keyspace_type_error(&txn_err, planted_key, expected_type);

        let lua_err = lua_store
            .update_layer_relayed(
                &fx.session_id,
                fx.layer(),
                fx.object_id,
                Some(set_field_delta(SECTION_ID, "k")),
                "b459-writer",
                2_000 + i as u64,
            )
            .await
            .expect_err("the lua backend must refuse the same keyspace");
        assert_keyspace_type_error(&lua_err, planted_key, expected_type);

        let (ServiceError::Internal(txn_msg), ServiceError::Internal(lua_msg)) =
            (&txn_err, &lua_err)
        else {
            unreachable!("both were asserted `Internal` above");
        };
        assert!(
            lua_msg.starts_with(txn_msg.as_str()),
            "the two backends must refuse the same keyspace with the same sentence. Everything \
             an operator's runbook, log grep or alert annotation keys on lives in that shared \
             text, and `LIVE_EDIT_UPDATE_LAYER_BACKEND` is a rollback lever that is supposed to \
             change performance, not diagnostics.\n  txn: {txn_msg}\n  lua: {lua_msg}"
        );
        let locator = &lua_msg[txn_msg.len()..];
        assert!(
            locator.is_empty() || (locator.starts_with(" (") && locator.ends_with(')')),
            "the ONLY thing the Lua path may add is a parenthesized locator saying where in the \
             delta the refusal came from — it probes interleaved with its apply pass and knows \
             the record index, which the txn probe (walking a deduped key list, where the key \
             IS the locator) has no equivalent for. Anything else in the tail is contract that \
             exists on one backend and not the other; got {locator:?}"
        );

        let expected_ticks = i as u64 + 1;
        assert_eq!(
            keyspace_type_errors(&fx.metrics, "txn", "update"),
            expected_ticks,
            "each txn refusal must tick its own cell exactly once"
        );
        assert_eq!(
            keyspace_type_errors(&fx.metrics, "lua", "update"),
            expected_ticks,
            "each lua refusal must tick its own cell exactly once — the `backend` label is what \
             tells an operator which commit path is refusing, and folding the two would lose it"
        );

        // Clear the staging so the next one is the only anomaly in the
        // keyspace. Both refusals left everything else untouched, so this DEL
        // is the whole reset.
        let _: () = fx
            .tv
            .client()
            .pool
            .del(planted_key)
            .await
            .expect("DEL of the planted key must succeed");
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// One commit, one refusal
// ═════════════════════════════════════════════════════════════════════════════

/// b.459: a delta with many mutations against one wrong-typed index is refused
/// once, ticks once, and writes nothing.
///
/// The probe key list is deduped in first-seen order, so the shared index keys
/// (`obj:flds` here) appear once no matter how many records name them — a 128-
/// field delta must not send 128 copies of `obj:flds` to the script. The
/// argument-vector size is not observable from an integration test, so what this
/// pins is the observable consequence: **one commit is one refusal and one
/// tick**, whatever the mutation count. The dedupe and its ordering are asserted
/// directly, as pure unit tests, beside `collect_probe_keys` in
/// `src/storage/layer.rs`.
///
/// The clean half runs first and is not a warm-up: it proves a 128-mutation
/// commit still succeeds, i.e. the probe added for b.459 does not fall over on a
/// large delta (the failure mode a naive per-mutation key list would have).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn txn_many_mutations_against_one_wrong_typed_index_refuse_once() {
    const FIELDS: u64 = 128;

    let fx = Fixture::new("b459-dedupe", UpdateLayerBackend::Txn).await;
    let wide = || OwnedDelta {
        set_fields: (0..FIELDS)
            .map(|i| SetField {
                section_id: SECTION_ID,
                key_name: format!("k{i}"),
                value_or_ext_hash: vec![i as u8],
                ext_value_size: 0,
                ext_value_index: 0,
                set_order: i,
            })
            .collect(),
        ..Default::default()
    };

    fx.update(wide(), 1_000)
        .await
        .expect("a wide field delta over a clean keyspace must commit");
    assert_eq!(
        keyspace_type_errors(&fx.metrics, "txn", "update"),
        0,
        "a healthy wide commit must not tick the anomaly counter"
    );

    let index_key = fx.tv.key_schema().obj_fields_index(&fx.oid_str());
    let _: () = fx
        .tv
        .client()
        .pool
        .del(&index_key)
        .await
        .expect("DEL must succeed");

    assert_commit_refused_whole(&fx, &index_key, "zset", wide(), 2_000, "txn").await;
}

// ═════════════════════════════════════════════════════════════════════════════
// The boundary the probe does NOT own
// ═════════════════════════════════════════════════════════════════════════════

/// b.459, stated honestly: `obj:dels` is read by the structural path's snapshot
/// read before the probe runs, so on the structural path that read is the
/// guard — not the probe.
///
/// `mutation_probe_keys` does name `obj:dels` (for `CreateTombstone`), so the
/// probe would catch it in isolation; it simply never gets the chance, because
/// `SMEMBERS` on a string fails first. The same is true of `obj:secs`, and of
/// `obj:flds` / `obj:tss` on the structural path.
///
/// This is worth a test rather than a comment because the two guards must reach
/// the SAME verdict: fail closed, before `MULTI`, with nothing written. If the
/// snapshot read ever started swallowing a `WRONGTYPE` (as the children-list
/// `LRANGE` did before b.zvd), the commit would proceed to a probe that has
/// already been passed over — and this test would go red rather than the defect
/// going unnoticed. It deliberately does NOT assert the error text: which of the
/// two guards fires is an implementation detail, and pinning the text would make
/// the boundary look like a contract.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn txn_wrongtype_deleted_sections_index_is_caught_by_the_snapshot_read_not_the_probe() {
    let fx = Fixture::new("b459-dels", UpdateLayerBackend::Txn).await;
    fx.update(create_section_delta(SECTION_ID, "Doomed"), 1_000)
        .await
        .expect("the setup commit must land on a clean keyspace");

    let index_key = fx.tv.key_schema().obj_deleted_sections_index(&fx.oid_str());
    plant_string(&fx, &index_key).await;

    let keys = witness_keys(&fx, std::slice::from_ref(&index_key));
    let before = dump_witness(&fx, &keys).await;
    let seq_before = header_seq(&fx).await;

    let err = fx
        .update(delete_section_delta(SECTION_ID), 2_000)
        .await
        .expect_err(
            "a wrong-typed deleted-sections index must fail the commit closed. Whichever guard \
             fires, an `Ok` here means the commit ran with a broken index and the tombstone \
             SADD was silently lost",
        );

    assert_eq!(
        dump_witness(&fx, &keys).await,
        before,
        "the refusal must happen before `MULTI` whichever guard produced it — a difference here \
         means the commit half-landed: {err:?}"
    );
    assert_eq!(
        header_seq(&fx).await,
        seq_before,
        "the header must not advance for a commit that did not happen"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// The arm a live keyspace cannot stage: a failed probe
// ═════════════════════════════════════════════════════════════════════════════

/// b.459's second classification arm, driven through the fred-layer injection
/// seam.
///
/// # Why a seam is needed at all
///
/// Everything above this module stages the probe's **tagged** arm: a real
/// string on a real key, a real `EVALSHA`, a real `WRONGTYPE_KEY:` error reply.
/// That is the arm a keyspace can hold still for. The other arm cannot be
/// staged that way — it is what happens when the probe's own round trip
/// **fails** (Valkey restarts, the connection drops, the command times out),
/// and reproducing it against a live server means killing Valkey inside the
/// window between `WATCH` and `MULTI`.
///
/// The consequence of getting that arm wrong is the whole reason it needs a
/// test. `probe_keyspace_types` is a *guard*, and the naive shape for a guard is
/// "if it did not report clean, refuse". Applied to a timed-out probe that reads
/// as a permanent `Internal`, a five-second Valkey blip becomes a five-second
/// **write outage** across every layer — reported to operators as a
/// keyspace-corruption refusal naming a key that is perfectly well typed. The
/// contract is the opposite: an inconclusive probe is `Aborted`, the retry loop
/// picks it up, and the next attempt gets a real answer.
///
/// # What the seam is, and what it is not
///
/// `LayerStore::with_keyspace_probe_fred_error_injector_for_test` substitutes a
/// queued `fred::error::Error` for the probe's `EVALSHA` result. It injects
/// *upstream* of `LayerStore::classify_keyspace_probe_error` — the same function
/// the real reply's `Err(e)` is handed to — so what these tests exercise is the
/// production classifier, not a test-only imitation of it. It is not a
/// simulation of the retry loop: the loop, its backoff, its metrics and the
/// second attempt's real probe all run for real against real Valkey.
///
/// | Test | What it pins |
/// |---|---|
/// | [`txn_transient_timeout_probe_error_is_retried_and_the_commit_lands`] | fred `Timeout` → `Aborted` → retry → the commit lands, exactly once |
/// | [`txn_transient_io_probe_error_is_retried_and_the_commit_lands`] | the same for fred `IO`, so the claim is about the classifier's kind TABLE and not one row of it |
/// | [`txn_transient_probe_error_with_a_one_attempt_budget_surfaces_aborted`] | the classification itself, read off the surfaced error rather than inferred from the retry outcome |
/// | [`txn_non_transient_probe_error_surfaces_internal_without_retry`] | the other half of that contract — a `Protocol` fault is `Internal` and takes no retry, so "retry everything" does not satisfy the rows above |
/// | [`txn_injected_wrongtype_tagged_probe_error_takes_the_tagged_arm`] | the tagged arm reached on the tag ALONE — no wrong-typed key anywhere in the keyspace |
/// | [`txn_wrongtype_tag_beats_a_transient_error_kind`] | arm ORDER: the tag is tested before transience, so a tagged error carrying a transient kind still refuses |
/// | [`txn_header_only_commit_still_probes_so_the_no_key_short_circuit_is_unreachable`] | why the "an empty probe list does not burn an entry" case has no behavioural test here |
///
/// # The vacuity guard every test here carries
///
/// A queue that was never read looks exactly like a queue whose entry was
/// classified as harmless: both leave a green test. So every test below asserts
/// the queue's length after its own call. An injection that silently did not
/// happen leaves a non-empty queue and fails — which is what keeps these from
/// being six elaborate tests of an ordinary commit.
///
/// # Two activation conditions, one of them silent
///
/// The seam is `#[cfg(any(test, feature = "test-fault-injection"))]` on
/// `LayerStore`, so this module is feature-gated to match. Without
/// `--features test-fault-injection` these tests are compiled out entirely and
/// the run reports green having covered none of the above — pass the feature
/// explicitly in any hand-rolled invocation.
#[cfg(feature = "test-fault-injection")]
mod probe_fault_injection {
    use super::*;

    use fred::error::{Error as FredError, ErrorKind as FredErrorKind};
    use std::collections::VecDeque;
    use std::sync::Mutex;

    // The constants, queue type and helpers marked `pub(super)` below are
    // shared with `ext_pre_image_read_fault_injection`, which drives a
    // different seam through the same fixture and observables.

    /// The layer's sequence after `Fixture::new` — `create_layer` takes 1.
    pub(super) const SEQ_AFTER_CREATE: u64 = 1;
    /// The sequence the single commit in each test below must take.
    pub(super) const SEQ_AFTER_ONE_COMMIT: u64 = 2;

    pub(super) type FredQueue = Arc<Mutex<VecDeque<FredError>>>;

    /// A txn-backend fixture whose keyspace-type probe is wired to an initially
    /// EMPTY injector queue, returned alongside it.
    ///
    /// Empty on purpose: several tests below need the fixture's `object_id` to
    /// build the error they inject (a realistic tagged message names a real
    /// key), and the `object_id` does not exist until the fixture does. Arming
    /// after construction also means `create_layer` provably cannot burn an
    /// entry, so the per-test queue-length assertions are attributable to the
    /// `update` call alone.
    ///
    /// `max_attempts` is passed explicitly rather than defaulted because it is
    /// the variable this family is written around: above 1 the observable is
    /// the *commit*, at exactly 1 it is the *classification*. Naming it at each
    /// call site keeps which claim a test makes visible in the call itself.
    async fn fixture_with_probe_injector(name: &str, max_attempts: u32) -> (Fixture, FredQueue) {
        let queue: FredQueue = Arc::new(Mutex::new(VecDeque::new()));
        let q = queue.clone();
        let fx = Fixture::with_store_tuning(name, UpdateLayerBackend::Txn, move |store| {
            store
                .with_update_layer_retry_max_attempts(max_attempts)
                .with_keyspace_probe_fred_error_injector_for_test(q)
        })
        .await;
        (fx, queue)
    }

    pub(super) fn arm(queue: &FredQueue, err: FredError) {
        queue.lock().unwrap().push_back(err);
    }

    pub(super) fn queue_len(queue: &FredQueue) -> usize {
        queue.lock().unwrap().len()
    }

    /// A `WRONGTYPE_KEY:`-tagged fred error shaped exactly like the one a real
    /// error reply produces.
    ///
    /// Matched to the real thing rather than guessed. `keyspace_type_probe.lua`
    /// returns `redis.error_reply("WRONGTYPE_KEY: <key> holds a <actual>,
    /// expected <expected>")`, and fred's `pretty_error` picks the `ErrorKind`
    /// from the reply's FIRST WHITESPACE TOKEN. Here that token is
    /// `WRONGTYPE_KEY:`, which matches none of its special cases, so the kind is
    /// `Unknown`. Bare `WRONGTYPE` — Valkey's own type error, a different
    /// string — maps to `InvalidArgument` instead; the two must not be
    /// conflated, which is why the kind is spelled out rather than borrowed from
    /// the neighbouring error.
    ///
    /// `kind` is a parameter and not hardcoded to `Unknown` because
    /// [`txn_wrongtype_tag_beats_a_transient_error_kind`] needs the same message
    /// under a transient kind: the classifier reads `details()`, so the kind
    /// must not be what routes this.
    fn tagged_probe_err(kind: FredErrorKind, key: &str, actual: &str, expected: &str) -> FredError {
        FredError::new(
            kind,
            format!("WRONGTYPE_KEY: {key} holds a {actual}, expected {expected}"),
        )
    }

    /// An UNtagged fred error — the details carry no `WRONGTYPE_KEY:` prefix,
    /// so `classify_keyspace_probe_error` falls through to its second arm and
    /// the verdict is decided by the kind alone. That is the whole point of the
    /// name: what routes these to `map_snapshot_load_err` is the absence of the
    /// tag, not the kind.
    fn untagged_probe_err(kind: FredErrorKind) -> FredError {
        FredError::new(kind, "injected probe failure")
    }

    /// `live_edit_update_layer_retries_total{path="watch",outcome}` — the path
    /// of every commit on the `txn` backend these fixtures pin.
    pub(super) fn retries(metrics: &Metrics, outcome: &str) -> u64 {
        sample_value(
            &render(metrics),
            "live_edit_update_layer_retries_total",
            &[("path", "watch"), ("outcome", outcome)],
        )
        .unwrap_or_else(|| {
            panic!(
                "live_edit_update_layer_retries_total{{path=\"watch\",outcome=\"{outcome}\"}} is \
                 absent from the exposition — every cell is zero-seeded, so absence means the \
                 store never received this metrics handle and every retry reading taken from it \
                 is vacuous"
            )
        }) as u64
    }

    /// A transient probe failure of `kind` must be retried, and the commit must
    /// land exactly once.
    ///
    /// Shared body for the two kinds below — parameterized rather than
    /// copy-pasted, since the kind is the only difference and the claims are
    /// identical; kept as two `#[tokio::test]` wrappers rather than a loop so
    /// nextest isolates them and a failure names the kind in the test name.
    ///
    /// The claims, and why each is separately load-bearing:
    ///
    /// 1. **`Ok(seq)`** — the injected error re-entered the retry loop. This is
    ///    what the seam exists for: `map_snapshot_load_err` produced an
    ///    `Aborted`, which is the loop's in-band retry signal, so attempt 2 ran
    ///    a real probe against a clean keyspace and committed.
    /// 2. **The header advanced exactly once** — a retry must RE-attempt the
    ///    commit, not repeat it. A `seq` past 2 would mean attempt 1 got further
    ///    than the probe before failing.
    /// 3. **Exactly one `{seq}-0` stream entry** — the subscriber-visible half
    ///    of (2). A commit published twice is a duplicate delivered to every
    ///    subscriber of the object; asserted on the entry IDs rather than the
    ///    count so a republish under a different id cannot hide in arithmetic.
    /// 4. **`keyspace_type_error_total{txn,update}` stays 0** — a Valkey blip is
    ///    not a keyspace anomaly. Ticking here would put a page-worthy counter
    ///    on the same footing as ordinary transport noise and cost it the zero
    ///    baseline its alert rests on.
    /// 5. **The queue is drained** — the vacuity guard; without it (1)-(4) are
    ///    assertions about an ordinary healthy commit.
    /// 6. **No dirty txn checkout across the call** — the abandoned attempt
    ///    handed its slot back clean, witnessed by the retry's own b.7td
    ///    checkout probe on the same one-slot pool.
    async fn transient_probe_error_is_retried(name: &str, kind: FredErrorKind) {
        let (fx, queue) = fixture_with_probe_injector(name, 32).await;
        // fred's own label, taken before the kind is moved into the error, so
        // the panic below names the injected kind without hardcoding a string.
        let kind_label = kind.to_str();
        arm(&queue, untagged_probe_err(kind));
        let dirty_before = txn_dirty_checkouts(&fx.metrics);

        // A structural delta: `CreateSection` puts the section hash, the
        // sections index and `obj:chs` on the probe list, so the probe
        // provably issues the round trip the injector substitutes for.
        let (seq, _) = fx
            .update(create_section_delta(SECTION_ID, "Retried"), 1_000)
            .await
            .unwrap_or_else(|e| {
                panic!(
                    "a fred {kind_label} on the keyspace-type probe must be classified transient \
                     and retried, and the second attempt must commit. An error here means the \
                     probe's failure arm reaches a NON-retriable classification: a Valkey blip \
                     during the pre-MULTI probe would become a write outage across every layer, \
                     reported as a keyspace-corruption refusal naming a key that is perfectly \
                     well typed. Got: {e:?}"
                )
            });

        assert_eq!(
            seq, SEQ_AFTER_ONE_COMMIT,
            "the retried commit must take the sequence its first attempt would have"
        );
        assert_eq!(
            header_seq(&fx).await,
            SEQ_AFTER_ONE_COMMIT,
            "the header must advance exactly once. Landing past {SEQ_AFTER_ONE_COMMIT} means the \
             failed attempt got further than the probe and the retry committed on top of it"
        );
        assert_eq!(
            stream_entry_ids(&fx).await,
            vec![
                format!("{SEQ_AFTER_CREATE}-0"),
                format!("{SEQ_AFTER_ONE_COMMIT}-0"),
            ],
            "CreateLayer's entry plus exactly ONE entry for the retried commit, at the id its \
             sequence names. Anything more is a duplicate update delivered to every subscriber \
             of this object"
        );
        assert_eq!(
            keyspace_type_errors(&fx.metrics, "txn", "update"),
            0,
            "a transport failure on the probe is not a keyspace anomaly. Ticking here would put \
             this counter on the same footing as ordinary Valkey noise and cost it the zero \
             baseline its alert depends on"
        );
        assert_eq!(
            queue_len(&queue),
            0,
            "the injector queue must be drained by exactly one attempt. A leftover entry means \
             the probe never issued its round trip, which would make every assertion above a \
             statement about an ordinary healthy commit"
        );
        assert_eq!(
            retries(&fx.metrics, "committed"),
            1,
            "committing on attempt 2 is exactly one retry — the operator-visible evidence that \
             the blip was absorbed rather than surfaced"
        );
        assert_eq!(
            retries(&fx.metrics, "exhausted"),
            0,
            "the budget was not exhausted"
        );
        assert_eq!(
            txn_dirty_checkouts(&fx.metrics),
            dirty_before,
            "the abandoned attempt must have handed its txn slot back clean. On a one-slot pool \
             the retry borrows the very same connection, so its b.7td checkout probe is the \
             witness: a tick here means the probe's error path returned without UNWATCHing, or \
             left a MULTI open"
        );
        assert_txn_slot_clean(&fx).await;
    }

    /// b.459: a fred `Timeout` on the probe is retried and the commit lands.
    ///
    /// `Timeout` is the shape the real failure mode takes — the probe rides the
    /// multiplexed data pool under `LIVE_EDIT_VALKEY_COMMAND_TIMEOUT_MS`, so a
    /// Valkey that stalls is what an operator actually hits.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn txn_transient_timeout_probe_error_is_retried_and_the_commit_lands() {
        transient_probe_error_is_retried("b459-probe-timeout", FredErrorKind::Timeout).await;
    }

    /// b.459: a fred `IO` on the probe is retried and the commit lands.
    ///
    /// Not redundant with the `Timeout` case: `is_transient_fred_error`
    /// classifies by a TABLE of kinds, and one kind passing proves one row of
    /// it. `IO` is the connection-dropped shape — a Valkey restart, a managed
    /// proxy invalidation — as against `Timeout`'s stalled-server shape, and the
    /// two together are what a rollout actually produces.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn txn_transient_io_probe_error_is_retried_and_the_commit_lands() {
        transient_probe_error_is_retried("b459-probe-io", FredErrorKind::IO).await;
    }

    /// b.459: a NON-transient probe failure surfaces as `Internal` without
    /// retrying.
    ///
    /// The other half of the transient contract, and the half that keeps it
    /// from being satisfied by "retry everything". `map_snapshot_load_err`
    /// classifies by a kind table, and a classifier widened until every probe
    /// failure looks retriable would spin the full budget against a
    /// deterministic bug — a protocol desync, a parse failure, a bad script —
    /// burning the budget on every commit and reporting retry exhaustion
    /// instead of the actual fault.
    ///
    /// `Protocol` is the right representative: it means the connection's reply
    /// stream is no longer trustworthy, which no amount of retrying repairs.
    ///
    /// Mirrors `update_layer_lua_test`'s
    /// `update_layer_lua_fred_protocol_error_surfaces_internal_without_retry`
    /// for this path's own call site, so the two backends' classifier wirings
    /// are pinned symmetrically.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn txn_non_transient_probe_error_surfaces_internal_without_retry() {
        let (fx, queue) = fixture_with_probe_injector("b459-probe-protocol", 32).await;
        arm(&queue, untagged_probe_err(FredErrorKind::Protocol));

        let keys = witness_keys(&fx, &[]);
        let before = dump_witness(&fx, &keys).await;

        let err = fx
            .update(create_section_delta(SECTION_ID, "NeverLands"), 1_000)
            .await
            .expect_err(
                "a non-transient fred error on the probe must surface, not be retried away. An \
                 `Ok` here means the classifier over-matched: attempt 2 probed a clean keyspace \
                 and committed, and a deterministic protocol fault would be invisible until it \
                 exhausted the budget on some later commit",
            );

        let ServiceError::Internal(msg) = &err else {
            panic!(
                "a non-transient probe failure must be the non-retriable \
                 `ServiceError::Internal`. `Aborted` would spin the retry loop against a \
                 condition no retry clears. Got: {err:?}"
            );
        };
        let kind_label = FredErrorKind::Protocol.to_str();
        assert!(
            msg.contains(kind_label),
            "the surfaced error must carry fred's own ErrorKind label (`{kind_label}`); got: \
             {msg:?}"
        );
        assert!(
            !msg.contains("transient I/O:"),
            "a non-transient error must not carry the transient prefix — that prefix is the \
             retry loop's own signal vocabulary and reading it here would mean the classifier \
             chose the wrong arm; got: {msg:?}"
        );
        assert!(
            !msg.contains(KEYSPACE_TYPE_ERROR_PREFIX),
            "a failed probe is not a wrong-typed key, whatever its kind — that message routes an \
             operator to the keyspace-corruption runbook over a clean keyspace; got: {msg:?}"
        );

        assert_eq!(
            queue_len(&queue),
            0,
            "exactly ONE attempt may have run. A drained queue plus a surfaced error is what \
             proves no retry was taken: a retried attempt would have found the queue empty, \
             probed for real, and committed"
        );
        assert_eq!(
            dump_witness(&fx, &keys).await,
            before,
            "the refusal happens before `MULTI`, so the keyspace must be bit-for-bit unchanged"
        );
        assert_eq!(
            header_seq(&fx).await,
            SEQ_AFTER_CREATE,
            "the header must not advance for a commit that did not happen"
        );
        assert_eq!(
            keyspace_type_errors(&fx.metrics, "txn", "update"),
            0,
            "a transport-layer fault is not a keyspace anomaly"
        );
        assert_eq!(
            retries(&fx.metrics, "exhausted"),
            0,
            "a non-retriable error must pass through the loop without consuming the budget"
        );
        assert_txn_slot_clean(&fx).await;
    }

    /// b.459: with a one-attempt budget, a transient probe failure surfaces as
    /// `Aborted` — and nothing is written.
    ///
    /// The two tests above assert the *consequence* of the classification (the
    /// commit landed, so the error must have been retriable). This one asserts
    /// the classification itself, which is strictly stronger and is what
    /// localizes a regression: an `Internal` here is a permanent failure, and a
    /// `FailedPrecondition` would advertise a client-side recovery that cannot
    /// help. Shrinking the budget to 1 is what makes the loop hand the verdict
    /// back rather than absorb it — the retry loop
    /// (`retry::retry_aborted_with_jitter_and_uncounted_waits`) wraps a final
    /// attempt's `Aborted` in its exhaustion sentence and returns it, so the
    /// surfaced error still carries the classifier's own words.
    ///
    /// `Aborted` is also the right thing for the CLIENT to see: it maps to a
    /// gRPC status callers retry, which is the honest answer to "the server
    /// could not find out whether your commit was safe". A permanent code would
    /// tell the caller to stop trying over a condition that clears itself.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn txn_transient_probe_error_with_a_one_attempt_budget_surfaces_aborted() {
        let (fx, queue) = fixture_with_probe_injector("b459-probe-1attempt", 1).await;
        arm(&queue, untagged_probe_err(FredErrorKind::Timeout));

        let keys = witness_keys(&fx, &[]);
        let before = dump_witness(&fx, &keys).await;
        let dirty_before = txn_dirty_checkouts(&fx.metrics);

        let err = fx
            .update(create_section_delta(SECTION_ID, "NeverLands"), 1_000)
            .await
            .expect_err("a one-attempt budget cannot absorb the injected probe failure");

        let ServiceError::Aborted(msg) = &err else {
            panic!(
                "an INCONCLUSIVE probe must classify as the retriable `ServiceError::Aborted`. \
                 The probe answers \"is a key this commit writes wrong-typed?\"; a failed probe \
                 does not answer it, and refusing the commit permanently turns a Valkey blip \
                 into a write outage carrying a keyspace-corruption message about a key that is \
                 fine. Got: {err:?}"
            );
        };
        assert!(
            msg.contains("transient I/O:"),
            "the surfaced error must carry `map_snapshot_load_err`'s transient prefix — that is \
             the string saying the classifier ran and chose the transient arm, as against an \
             `Aborted` produced elsewhere in the commit path (a lost WATCH, say); got: {msg:?}"
        );
        assert!(
            msg.contains("EVALSHA keyspace_type_probe"),
            "the surfaced error must name the probe as the failing operation. Without the op \
             label an operator cannot tell a probe blip from any other transient round trip in \
             the commit, and the probe is the round trip b.459 newly added to this path; got: \
             {msg:?}"
        );
        let kind_label = FredErrorKind::Timeout.to_str();
        assert!(
            msg.contains(kind_label),
            "the surfaced error must carry fred's own ErrorKind label (`{kind_label}`) so a \
             flood of retries can be attributed to a cause; got: {msg:?}"
        );
        assert!(
            !msg.contains(KEYSPACE_TYPE_ERROR_PREFIX),
            "a failed probe must NOT be reported as a wrong-typed key — that message routes an \
             operator to the keyspace-corruption runbook over a keyspace that is clean; got: \
             {msg:?}"
        );

        assert_eq!(
            dump_witness(&fx, &keys).await,
            before,
            "the refusal happens before `MULTI`, so the object's keyspace must be BIT-FOR-BIT \
             unchanged. A difference here means the attempt got past the probe"
        );
        assert_eq!(
            header_seq(&fx).await,
            SEQ_AFTER_CREATE,
            "the header must not advance for a commit that did not happen"
        );
        assert_eq!(
            stream_entry_ids(&fx).await,
            vec![format!("{SEQ_AFTER_CREATE}-0")],
            "no stream entry may be appended: subscribers must never be told about a commit that \
             never opened its MULTI"
        );
        assert_eq!(
            keyspace_type_errors(&fx.metrics, "txn", "update"),
            0,
            "a failed probe is not a wrong-typed key and must not tick the anomaly counter"
        );
        assert_eq!(
            queue_len(&queue),
            0,
            "the single attempt must have consumed the single queued entry — a leftover entry \
             would mean the `Aborted` came from somewhere other than the probe"
        );
        assert_eq!(retries(&fx.metrics, "committed"), 0, "nothing committed");
        assert_eq!(
            retries(&fx.metrics, "exhausted"),
            0,
            "the counter records retries TAKEN, and a one-attempt budget takes none \
             (`max_attempts - 1 == 0`). It reads oddly, so it is asserted rather than skipped: a \
             future change to the retry accounting has to come past this line deliberately"
        );
        assert_eq!(
            txn_dirty_checkouts(&fx.metrics),
            dirty_before,
            "the refused attempt must hand its txn slot back clean"
        );
        assert_txn_slot_clean(&fx).await;
    }

    /// b.459: a `WRONGTYPE_KEY:`-tagged probe error takes the tagged arm — on
    /// the tag alone, with no wrong-typed key anywhere in the keyspace.
    ///
    /// Every other tagged-arm test in this file plants a real string and lets
    /// the script find it, which couples the arm to the script. This one
    /// separates them: the keyspace is pristine and the tag arrives by
    /// injection, so what is asserted is exactly the Rust-side contract — the
    /// tag is what selects the refusal message, the metric tick and the
    /// non-retriable classification.
    ///
    /// The refusal being non-retriable is asserted through the queue rather
    /// than by inspecting the loop: one entry was queued, so a retried attempt
    /// would have probed a clean keyspace and returned `Ok`. A surfaced error
    /// with a drained queue is exactly "one attempt, no retry".
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn txn_injected_wrongtype_tagged_probe_error_takes_the_tagged_arm() {
        let (fx, queue) = fixture_with_probe_injector("b459-probe-tagged", 32).await;

        // A real key from the schema, so the message asserted here is the one
        // an operator would actually read.
        let key = fx
            .tv
            .key_schema()
            .obj_section(&fx.oid_str(), &SECTION_ID.to_string());
        arm(
            &queue,
            tagged_probe_err(FredErrorKind::Unknown, &key, "string", "hash"),
        );

        let keys = witness_keys(&fx, std::slice::from_ref(&key));
        let before = dump_witness(&fx, &keys).await;
        let dirty_before = txn_dirty_checkouts(&fx.metrics);

        let err = fx
            .update(create_section_delta(SECTION_ID, "Refused"), 1_000)
            .await
            .expect_err("a tagged probe error must refuse the commit");
        assert_keyspace_type_error(&err, &key, "hash");

        assert_eq!(
            dump_witness(&fx, &keys).await,
            before,
            "the refusal must leave the keyspace BIT-FOR-BIT unchanged — the probe runs before \
             `MULTI`, so a difference here is the half-commit b.459 exists to close"
        );
        assert_eq!(
            header_seq(&fx).await,
            SEQ_AFTER_CREATE,
            "the header must not advance for a refused commit"
        );
        assert_eq!(
            stream_entry_ids(&fx).await,
            vec![format!("{SEQ_AFTER_CREATE}-0")],
            "no stream entry may be appended for a refused commit"
        );
        assert_eq!(
            keyspace_type_errors(&fx.metrics, "txn", "update"),
            1,
            "the tagged arm must tick live_edit_keyspace_type_error_total{{backend=\"txn\", \
             site=\"update\"}} exactly once. Zero leaves operators blind to a layer that has \
             silently stopped accepting writes"
        );
        assert_eq!(
            queue_len(&queue),
            0,
            "exactly ONE attempt may have run. A drained queue plus a surfaced error is the \
             proof the refusal was NON-retriable: had it been classified retriable the loop \
             would have run attempt 2 against a clean keyspace and returned `Ok`, and had it \
             never probed at all the entry would still be sitting here"
        );
        assert_eq!(retries(&fx.metrics, "committed"), 0, "nothing committed");
        assert_eq!(
            retries(&fx.metrics, "exhausted"),
            0,
            "a non-retriable refusal must not consume the retry budget — spinning attempts \
             against a standing keyspace condition delays the operator-visible verdict and loads \
             Valkey for nothing"
        );
        assert_eq!(
            txn_dirty_checkouts(&fx.metrics),
            dirty_before,
            "the refused attempt must hand its txn slot back clean"
        );
        assert_txn_slot_clean(&fx).await;
    }

    /// b.459: the tag is tested BEFORE transience — a tagged error carrying a
    /// transient fred kind still refuses.
    ///
    /// `classify_keyspace_probe_error` is two arms in a fixed order: the
    /// `WRONGTYPE_KEY:` tag first, `map_snapshot_load_err` second. That order is
    /// load-bearing and the wrong order is silent. The kind on a real error
    /// reply is derived by fred from the reply text while the tag is chosen by
    /// the script, so the two are independent — and under a transience-first
    /// classifier a tagged error that happened to carry a transient kind would
    /// become `Aborted`. The loop would then spin its whole budget against a key
    /// that is still the wrong type on every attempt and surface a generic
    /// retry-exhaustion message: the remediation naming the offending key is
    /// lost, and a corrupt keyspace looks like Valkey flakiness.
    ///
    /// The regression is cheap to observe here precisely because the queue holds
    /// one entry — under a transience-first classifier attempt 2 probes a clean
    /// keyspace and the call returns `Ok`.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn txn_wrongtype_tag_beats_a_transient_error_kind() {
        let (fx, queue) = fixture_with_probe_injector("b459-probe-tag-order", 32).await;

        let key = fx.tv.key_schema().obj_children_index(&fx.oid_str());
        arm(
            &queue,
            // The kind the transient arm matches on, carrying the tag the
            // tagged arm matches on. Only the arm ORDER decides the verdict.
            tagged_probe_err(FredErrorKind::Timeout, &key, "string", "set"),
        );

        let err = fx
            .update(create_section_delta(SECTION_ID, "Refused"), 1_000)
            .await
            .expect_err(
                "a `WRONGTYPE_KEY:`-tagged probe error must refuse the commit whatever fred kind \
                 it carries. An `Ok` here means transience was tested first, the refusal was \
                 retried away, and a genuinely corrupt keyspace would be reported — after the \
                 whole retry budget — as generic Valkey flakiness with no key named",
            );
        assert_keyspace_type_error(&err, &key, "set");

        assert_eq!(
            keyspace_type_errors(&fx.metrics, "txn", "update"),
            1,
            "the tagged arm ticks its counter regardless of the carried kind"
        );
        assert_eq!(
            queue_len(&queue),
            0,
            "one attempt, one entry — the refusal was not retried"
        );
        assert_eq!(
            header_seq(&fx).await,
            SEQ_AFTER_CREATE,
            "the header must not advance for a refused commit"
        );
        assert_eq!(
            stream_entry_ids(&fx).await,
            vec![format!("{SEQ_AFTER_CREATE}-0")],
            "no stream entry may be appended for a refused commit"
        );
    }

    /// b.459: every commit probes at least the object header, so the probe's
    /// "no keys, no round trip" short-circuit is unreachable in production —
    /// which is why the case that would leave a queued entry unconsumed has no
    /// behavioural test in this file.
    ///
    /// # The case this stands in for
    ///
    /// `probe_keyspace_types` returns `Ok(())` on an empty key list BEFORE it
    /// consults the injector, so a commit that probes nothing leaves a queued
    /// entry pending. That is a real property of the seam, and the obvious test
    /// for it — queue an entry, drive a commit with no probed keys, assert the
    /// entry survives — cannot be written, because **no commit has an empty
    /// probe list**: every one emits `SetObjectHeader`, whose header hash
    /// `mutation_probe_keys` always names.
    ///
    /// So this pins the reason instead of asserting the unreachable case. A
    /// header-only commit — the emptiest delta the service accepts — is driven
    /// with an entry queued, and the entry MUST be consumed. If a future change
    /// stopped probing the header this goes red, the short-circuit becomes
    /// reachable, and the test above becomes writable. The empty-list behaviour
    /// itself is pinned where it is observable: `collect_probe_keys_over_no_
    /// mutations_is_empty`, a pure unit test beside `collect_probe_keys` in
    /// `src/storage/layer.rs`.
    ///
    /// A header-only delta is also fast-path-shaped, so under the `txn` knob it
    /// takes the `!needs_snapshot` branch — the one that loads no snapshot at
    /// all, where the probe is the only pre-`MULTI` read of any kind. That makes
    /// it exactly the branch on which an unprobed header would be unguarded by
    /// anything else.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn txn_header_only_commit_still_probes_so_the_no_key_short_circuit_is_unreachable() {
        let (fx, queue) = fixture_with_probe_injector("b459-probe-header-only", 32).await;
        arm(&queue, untagged_probe_err(FredErrorKind::Timeout));

        let (seq, _) = fx
            .update(OwnedDelta::default(), 1_000)
            .await
            .expect("the retried header-only commit must land");

        assert_eq!(
            queue_len(&queue),
            0,
            "a header-only commit must still issue the probe round trip. A surviving entry means \
             `collect_probe_keys` returned an empty list for a real commit: the object header \
             would be written by an unprobed `HSET`, and on this branch — no snapshot is \
             loaded — nothing else would have read it either"
        );
        assert_eq!(
            seq, SEQ_AFTER_ONE_COMMIT,
            "the commit advances the sequence exactly once after the retry"
        );
        assert_eq!(header_seq(&fx).await, SEQ_AFTER_ONE_COMMIT);
        assert_eq!(
            keyspace_type_errors(&fx.metrics, "txn", "update"),
            0,
            "a transport failure on the probe is not a keyspace anomaly"
        );
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// The race a live keyspace can only sample: the pre-image read failing first
// ═════════════════════════════════════════════════════════════════════════════

/// b.j2x: on the txn-knob path the keyspace probe's verdict decides before the
/// external-value pre-image read beside it is consulted.
///
/// # The defect
///
/// The `!needs_snapshot` branch of `update_layer_once_watch` reads its
/// external-value pre-image (`HEXISTS <row> value` + `HGET <row> hash` per
/// written row, then one `HGET obj:ext <hash>` per external hash those rows
/// hold or receive; before b.26e Stage 2 the read took the whole index with
/// one `HGETALL`)
/// at the same time as its pre-`MULTI` probes. Every key that read touches is
/// also on the probe list, so a wrong-typed field row, time-sample row or
/// `obj:ext` fails both. Before
/// b.j2x all three futures sat in one `tokio::try_join!`, which returns the
/// first `Err` to arrive and drops the rest. When the read won, the commit was
/// refused with the read's untagged `WRONGTYPE`: no `keyspace_type_error_total`
/// tick, no remediation ERROR line, no key named.
/// [`txn_wrongtype_new_field_key_refuses_whole_commit`] and its time-sample
/// sibling failed that way a few percent of the time.
///
/// # Why a seam is needed
///
/// Against a real key the race can only be sampled: a passing run shows the
/// probe won that run. The seam,
/// `LayerStore::with_ext_pre_image_read_fred_error_injector_for_test`, fails
/// the read before its first await, sending nothing, through the read's own
/// mapper and op label. Under a fail-fast join the read's error is then always
/// the first to arrive, so the old code fails the regression rows below on
/// every run.
///
/// | Test | What it pins |
/// |---|---|
/// | [`txn_wrongtype_field_key_refuses_tagged_even_when_the_pre_image_read_fails_first`] | the regression: a wrong-typed field row refuses with the probe's tagged error and tick, nothing written, though the read's `WRONGTYPE` is already in |
/// | [`txn_wrongtype_time_sample_key_refuses_tagged_even_when_the_pre_image_read_fails_first`] | the same for a time-sample row, the other row family the flaky tests planted |
/// | [`txn_wrongtype_ext_index_refuses_tagged_even_when_the_pre_image_read_fails_first`] | the same for `obj:ext`, the commit-level key the read's count reads `HGET` |
/// | [`txn_transient_pre_image_read_failure_is_retried_and_the_commit_lands`] | once the probes pass, the read's own failure still decides: a `Timeout` is `Aborted`, retried, and the commit lands once |
/// | [`txn_non_transient_pre_image_read_failure_surfaces_internal_naming_the_read`] | the same for a permanent kind: `Internal` naming the read's op label, no type-error tick, nothing written |
///
/// The first three must fail without the fix. The last two pass with or
/// without it; they catch a fix that drops or ignores the read's error once
/// the checks have passed.
///
/// # The vacuity guard every test here carries
///
/// The queue is asserted drained after each call. A read that never issued
/// its round trip leaves its entry queued: an empty `rows` list
/// short-circuits ahead of the seam, and the snapshot branch never calls the
/// read. Without the check every other assertion here could be describing an
/// ordinary probe refusal or an ordinary commit.
///
/// # Two activation conditions, one of them silent
///
/// Feature-gated on `test-fault-injection` to match the seam. Without
/// `--features test-fault-injection` these tests are compiled out and the run
/// reports green having covered none of the above.
#[cfg(feature = "test-fault-injection")]
mod ext_pre_image_read_fault_injection {
    use super::*;

    use super::probe_fault_injection::{
        arm, queue_len, retries, FredQueue, SEQ_AFTER_CREATE, SEQ_AFTER_ONE_COMMIT,
    };
    use fred::error::{Error as FredError, ErrorKind as FredErrorKind};
    use std::collections::VecDeque;
    use std::sync::Mutex;

    /// The op label a failure of the update path's pre-image read carries in
    /// its first phase, the rows' columns (`LayerStore::EXT_PRE_IMAGE_PIPELINE_OP`;
    /// the seam fails the read there, before its phase-2 count reads, labelled
    /// `"HGET ext index counts"`). Restated rather than imported for the reason
    /// [`KEYSPACE_TYPE_ERROR_PREFIX`] is: the constant is private to
    /// `src/storage/layer.rs`, and the string is what an operator reads.
    const EXT_PRE_IMAGE_PIPELINE_OP: &str = "HEXISTS row values + HGET row hashes";

    /// A txn-backend fixture whose pre-image read is wired to an initially
    /// EMPTY injector queue, returned alongside it.
    ///
    /// Empty because the fixture's own `create_layer` reaches
    /// `write_object`'s `HGETALL obj:ext`, which pops from the same queue.
    /// Arming after construction keeps each queue-length assertion about the
    /// test's own `update` alone.
    async fn fixture_with_ext_read_injector(name: &str, max_attempts: u32) -> (Fixture, FredQueue) {
        let queue: FredQueue = Arc::new(Mutex::new(VecDeque::new()));
        let q = queue.clone();
        let fx = Fixture::with_store_tuning(name, UpdateLayerBackend::Txn, move |store| {
            store
                .with_update_layer_retry_max_attempts(max_attempts)
                .with_ext_pre_image_read_fred_error_injector_for_test(q)
        })
        .await;
        (fx, queue)
    }

    /// The error a real `HGET` / `HGETALL` against a non-hash key produces.
    ///
    /// Valkey answers `-WRONGTYPE Operation against a key holding the wrong
    /// kind of value`, and fred maps the bare `WRONGTYPE` token to
    /// `InvalidArgument` (the probe's `WRONGTYPE_KEY:` tag maps to `Unknown`
    /// instead; see `probe_fault_injection::tagged_probe_err`). It carries no
    /// tag, so a refusal built from it can only be the untagged one.
    fn server_wrongtype_reply() -> FredError {
        FredError::new(
            FredErrorKind::InvalidArgument,
            "WRONGTYPE Operation against a key holding the wrong kind of value",
        )
    }

    /// Plant a string at the key `planted_key` names, make the pre-image read
    /// fail with the server's own `WRONGTYPE`, and assert the tagged refusal.
    ///
    /// [`assert_commit_refused_whole`] carries the refusal contract: the tagged
    /// message naming the key and both types, the keyspace bit-for-bit
    /// unchanged, `seq` and the stream unchanged, the txn slot clean, and
    /// `{txn,update}` ticked once. Under the old fail-fast join its first
    /// assertion fails, because the surfaced error is the read's
    /// `"HEXISTS row values + … failed: Invalid Argument: WRONGTYPE …"`. On top
    /// of that it asserts the drained queue, without which the read might never
    /// have been in the race at all, that the read's op label is absent from
    /// the refusal, and that no retry budget was spent.
    async fn tagged_refusal_beats_a_failed_pre_image_read(
        name: &str,
        planted_key: impl FnOnce(&Fixture) -> String,
        delta: OwnedDelta,
    ) {
        let (fx, queue) = fixture_with_ext_read_injector(name, 32).await;
        let key = planted_key(&fx);
        arm(&queue, server_wrongtype_reply());

        let err = assert_commit_refused_whole(&fx, &key, "hash", delta, 1_000, "txn").await;

        let ServiceError::Internal(msg) = &err else {
            unreachable!("`assert_commit_refused_whole` asserted `Internal`");
        };
        assert!(
            !msg.contains(EXT_PRE_IMAGE_PIPELINE_OP),
            "the refusal must be the probe's alone. The pre-image read's op label in it means \
             the read's own failure was reported, the one that carries no tick and names no key; \
             got: {msg:?}"
        );
        assert_eq!(
            queue_len(&queue),
            0,
            "the pre-image read must have consumed its injected `WRONGTYPE`. A leftover entry \
             means the read never issued its round trip, so it was never in the race this test \
             exists to settle and the tagged refusal above proves nothing about the ordering"
        );
        assert_eq!(
            retries(&fx.metrics, "exhausted"),
            0,
            "a wrong-typed key is a standing condition; the refusal must not spend the retry \
             budget on it"
        );
    }

    /// b.j2x: a wrong-typed field row refuses with the probe's tagged error
    /// even when the pre-image read's `WRONGTYPE` arrives first.
    ///
    /// The shape of the flaky `txn_wrongtype_new_field_key_refuses_whole_commit`,
    /// with the race decided against the probe on every run.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn txn_wrongtype_field_key_refuses_tagged_even_when_the_pre_image_read_fails_first() {
        tagged_refusal_beats_a_failed_pre_image_read(
            "b459-j2x-field",
            |fx| {
                fx.tv
                    .key_schema()
                    .obj_field(&fx.oid_str(), &SECTION_ID.to_string(), "k")
            },
            set_field_delta(SECTION_ID, "k"),
        )
        .await;
    }

    /// b.j2x: the same for a time-sample row, the shape of the flaky
    /// `txn_wrongtype_new_time_sample_key_refuses_whole_commit`.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn txn_wrongtype_time_sample_key_refuses_tagged_even_when_the_pre_image_read_fails_first()
    {
        tagged_refusal_beats_a_failed_pre_image_read(
            "b459-j2x-ts",
            |fx| {
                fx.tv.key_schema().obj_time_sample(
                    &fx.oid_str(),
                    &SECTION_ID.to_string(),
                    SAMPLE_TIME,
                )
            },
            set_time_sample_delta(SECTION_ID, SAMPLE_TIME),
        )
        .await;
    }

    /// b.j2x: the same for `obj:ext`.
    ///
    /// The key is not a row: the probe names it for any value-row mutation,
    /// and the read `HGET`s it once per external hash the commit touches (this
    /// inline write touches none; the seam fails the read before it gets that
    /// far either way). This is the case the SDD used to document as
    /// nondeterministic.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn txn_wrongtype_ext_index_refuses_tagged_even_when_the_pre_image_read_fails_first() {
        tagged_refusal_beats_a_failed_pre_image_read(
            "b459-j2x-ext",
            |fx| fx.tv.key_schema().obj_ext_index(&fx.oid_str()),
            set_field_delta(SECTION_ID, "k"),
        )
        .await;
    }

    /// b.j2x: once the probes pass, a transient failure of the pre-image read
    /// is still the attempt's outcome: `Aborted`, retried, and the commit
    /// lands exactly once.
    ///
    /// The keyspace is clean, so both probes pass and the read's injected
    /// `Timeout` is the only error. A fix that consulted the checks and then
    /// discarded the read's result would commit on attempt 1 over an empty
    /// pre-image; the one retry counted below is what rules that out.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn txn_transient_pre_image_read_failure_is_retried_and_the_commit_lands() {
        let (fx, queue) = fixture_with_ext_read_injector("b459-j2x-read-timeout", 32).await;
        arm(
            &queue,
            FredError::new(FredErrorKind::Timeout, "injected pre-image read timeout"),
        );

        let (seq, _) = fx
            .update(set_field_delta(SECTION_ID, "k"), 1_000)
            .await
            .unwrap_or_else(|e| {
                panic!(
                    "a transient failure of the pre-image read must be classified `Aborted` and \
                     retried, and the retry must commit over the clean keyspace; got {e:?}"
                )
            });

        assert_eq!(
            retries(&fx.metrics, "committed"),
            1,
            "the commit must land on attempt 2. Landing on attempt 1 means the read's `Aborted` \
             was dropped once the probes passed, and the commit was planned against a pre-image \
             that was never read"
        );
        assert_eq!(
            retries(&fx.metrics, "exhausted"),
            0,
            "the budget was not exhausted"
        );
        assert_eq!(
            queue_len(&queue),
            0,
            "the first attempt's read must have consumed the injected `Timeout`. A leftover \
             entry means the read never issued its round trip and the retry counted above came \
             from somewhere else"
        );
        assert_eq!(
            seq, SEQ_AFTER_ONE_COMMIT,
            "the retried commit must take the sequence its first attempt would have"
        );
        assert_eq!(
            header_seq(&fx).await,
            SEQ_AFTER_ONE_COMMIT,
            "the header must advance exactly once"
        );
        assert_eq!(
            stream_entry_ids(&fx).await,
            vec![
                format!("{SEQ_AFTER_CREATE}-0"),
                format!("{SEQ_AFTER_ONE_COMMIT}-0"),
            ],
            "CreateLayer's entry plus exactly one entry for the retried commit, at the id its \
             sequence names"
        );
        let field: HashMap<String, String> = fx
            .tv
            .client()
            .pool
            .hgetall(
                fx.tv
                    .key_schema()
                    .obj_field(&fx.oid_str(), &SECTION_ID.to_string(), "k"),
            )
            .await
            .expect("HGETALL field must succeed");
        assert_eq!(
            field.get("version").map(String::as_str),
            Some("2"),
            "the field must carry the commit's version; a commit that returned `Ok` without \
             materializing is a tear; got {field:?}"
        );
        assert_eq!(
            keyspace_type_errors(&fx.metrics, "txn", "update"),
            0,
            "a transport failure on the pre-image read is not a keyspace anomaly"
        );
        assert_txn_slot_clean(&fx).await;
    }

    /// b.j2x: once the probes pass, a NON-transient failure of the pre-image
    /// read surfaces as `Internal` naming that read.
    ///
    /// A one-attempt budget so the classification is read off the surfaced
    /// error: an `Aborted` here would come back wrapped in the loop's
    /// exhaustion sentence rather than being retried into a commit. A fix that
    /// dropped the read's error once the checks passed would return `Ok`.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn txn_non_transient_pre_image_read_failure_surfaces_internal_naming_the_read() {
        let (fx, queue) = fixture_with_ext_read_injector("b459-j2x-read-protocol", 1).await;
        arm(
            &queue,
            FredError::new(
                FredErrorKind::Protocol,
                "injected pre-image read protocol fault",
            ),
        );

        let field_key = fx
            .tv
            .key_schema()
            .obj_field(&fx.oid_str(), &SECTION_ID.to_string(), "k");
        let ext_key = fx.tv.key_schema().obj_ext_index(&fx.oid_str());
        let keys = witness_keys(&fx, &[field_key, ext_key]);
        let before = dump_witness(&fx, &keys).await;

        let err = fx
            .update(set_field_delta(SECTION_ID, "k"), 1_000)
            .await
            .expect_err(
                "a permanent failure of the pre-image read must refuse the commit. An `Ok` here \
                 means the read's error was dropped once the probes passed, and the commit was \
                 planned against a pre-image that was never read",
            );

        let ServiceError::Internal(msg) = &err else {
            panic!(
                "a non-transient fred error on the pre-image read must be the non-retriable \
                 `Internal`; got {err:?}"
            );
        };
        assert!(
            msg.contains(&format!("{EXT_PRE_IMAGE_PIPELINE_OP} failed")),
            "the refusal must name the pre-image read as the failing operation, so an operator \
             can tell it from the probes beside it; got: {msg:?}"
        );
        let kind_label = FredErrorKind::Protocol.to_str();
        assert!(
            msg.contains(kind_label),
            "the refusal must carry fred's own ErrorKind label (`{kind_label}`); got: {msg:?}"
        );
        assert!(
            !msg.contains(KEYSPACE_TYPE_ERROR_PREFIX),
            "a failed read over a clean keyspace is not a wrong-typed key; got: {msg:?}"
        );

        assert_eq!(
            queue_len(&queue),
            0,
            "the single attempt's read must have consumed the injected error — a leftover entry \
             means the `Internal` came from somewhere other than the read"
        );
        assert_eq!(
            keyspace_type_errors(&fx.metrics, "txn", "update"),
            0,
            "a transport-layer fault is not a keyspace anomaly"
        );
        assert_eq!(
            dump_witness(&fx, &keys).await,
            before,
            "the refusal happens before `MULTI`, so the keyspace must be bit-for-bit unchanged"
        );
        assert_eq!(
            header_seq(&fx).await,
            SEQ_AFTER_CREATE,
            "the header must not advance for a commit that did not happen"
        );
        assert_eq!(
            stream_entry_ids(&fx).await,
            vec![format!("{SEQ_AFTER_CREATE}-0")],
            "no stream entry may be appended for a refused commit"
        );
        assert_txn_slot_clean(&fx).await;
    }
}
