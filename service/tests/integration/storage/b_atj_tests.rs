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

//! b.atj — a snapshot read refused by a wrong-typed key must move a metric.
//!
//! # The gap
//!
//! b.zvd converted a silent heal into a loud refusal: the snapshot loader of
//! the time (`load_object_snapshot`, since deleted) began failing closed when
//! any key it read held a foreign Valkey type, and every snapshot read since
//! does the same. Today FOUR snapshot consumers wedge on such a
//! key, all through `read_object_snapshot`'s key set, which classifies the same
//! way: the structural `UpdateLayer` WATCH path through its unfenced fan-out at
//! every size; the subscribe
//! catch-up at every size — at or below the atomic-read limits through its
//! atomic read, above them through the fenced read's fan-out its layer's
//! shared flight runs (a legacy in-window replay of such a layer takes the
//! catch-up's coalesced route there); the checkpoint assembler at every size —
//! atomically at or below the limits, through a fenced read of its own above
//! them; and per-commit reconstruction (a legacy in-window replay) at or below
//! the limits through its atomic read. A live commit is never reconstructed:
//! one without relay bytes is an empty commit, sent with no read. But before
//! b.atj *no counter moved* on any of these reads. Four is the exhaustive
//! snapshot-consumer count, not a sample. A fifth reader counts on the same
//! counter without reading a snapshot: external-value clean-up of a deleted or
//! replaced layer (`storage::retirement`, b.fsz MR-D4), whose size read and
//! reference walk read only the header, `obj:flds`, `obj:tss`, their rows and
//! `obj:ext` (`key_family="ext_index"`). The authoritative list, and the key
//! split on an object carrying the `chs_complete` marker (a wrong-typed
//! `obj:chs` stops all four at every size, because the atomic read's size
//! check reads it before it can answer `TOO_LARGE`, while a wrong-typed list
//! `obj:chs` does not name stops none of them), lives on the
//! `snapshot_read_type_error_total` field doc in `src/observability/metrics.rs`.
//!
//! Structural `UpdateLayer` is on that WATCH path on every call under `txn`
//! and `lua-watch-structural`, and under `lua` only on its uncacheable and
//! repeat-divergence fallbacks. Under `lua` otherwise it commits through the
//! structural write-planning cache, which reads the object through
//! `read_object_snapshot_with_ceilings`'s atomic read only on a cold start or
//! reload (`storage::structural_cache_cold_start`), so only those calls stop
//! on a key that read touches; a cached call issues no snapshot read — it
//! reads the header, stream and commit token through
//! `structural_cache_sync.lua`, whose refusal this counting does not cover.
//! The cache's sampled audit (`storage::structural_cache_audit`) issues the
//! same atomic read in a background task: its `WRONGTYPE` refusal ticks
//! `live_edit_snapshot_read_type_error_total` (below) and skips the audit, and
//! stops no call.
//!
//! `live_edit_keyspace_type_error_total{backend,site}` is written only by the
//! two pre-`MULTI` type probes, which are commit-side, so an operator alerting
//! on it saw a flat line through the exact wedge b.zvd had just made loud.
//!
//! b.atj adds the read-side twin,
//! `live_edit_snapshot_read_type_error_total{key_family}`, ticked from
//! `classify_snapshot_read_err` when `storage::client::is_server_wrongtype_error`
//! matches. **Counting only.** The `ServiceError` it returns is still
//! `map_snapshot_load_err`'s, byte for byte, and nothing downstream of the
//! predicate branches on it — see the sibling file's
//! [`b_zvd_tests::snapshot_read_of_a_wrongtype_children_list_is_a_loud_internal_naming_the_parent`],
//! which pins the *refusal* this file pins the *tick* for. That split is
//! deliberate: b.zvd owns the error surface (variant, message, named parent),
//! b.atj owns the observability, and neither file restates the other's claim.
//!
//! [`b_zvd_tests::snapshot_read_of_a_wrongtype_children_list_is_a_loud_internal_naming_the_parent`]:
//!     ../b_zvd_tests/index.html
//!
//! # Why metric VALUES are asserted here at all
//!
//! `docs-internal/test-writing-guide.md` § What NOT to Test says not to assert
//! metric values, "these are observability, not contracts". This counter is the
//! exception the rule implies rather than a violation of it: the whole
//! deliverable of b.atj *is* the counter, its `increase(...) > 0` alert rule is
//! shipped in `docs/service/docs/monitoring.md`, and a wedge that ticks nothing is
//! precisely the defect. The guide's target is an incidental metric assertion
//! bolted onto a behavioural test; the established precedent for a counter that
//! IS the contract is `b_459_tests` / `b_zvd_tests` asserting
//! `keyspace_type_error_total` cell by cell, and this file follows it.
//!
//! # What lives here
//!
//! | Test | Family it stages | What only it covers |
//! |---|---|---|
//! | [`fenced_read_of_a_wrongtype_children_list_ticks_children`] | `children` | the fenced read's own wrapper on a permanent refusal: it ends on the first one, as ONE recorded `permanent_error` read with no broken or retried attempt, and ticks exactly once — on the b.zvd wedge shape itself, the only family whose op label carries a parent id |
//! | [`txn_structural_commit_over_a_wrongtype_children_list_ticks_children`] | `children` | the commit consumer: an operator's alert now fires for a write outage that used to be counter-silent |
//! | [`fenced_read_over_wrongtype_keys_it_never_reads_ticks_nothing`] | — | negative control: the vocabulary is about READS, not about the object's keyspace |
//! | [`fenced_read_of_a_missing_object_ticks_nothing`] | — | negative control: `NotFound` is not a type error |
//! | [`non_wrongtype_snapshot_read_errors::snapshot_read_refused_by_a_transient_timeout_ticks_neither_counter`] | — | the gate: an error that IS classified, and is not a `WRONGTYPE`, ticks neither this counter nor b.muu's refusal counter (`test-fault-injection`) |
//! | [`non_wrongtype_snapshot_read_errors::snapshot_read_refused_by_a_non_wrongtype_invalid_argument_ticks_the_refusal_counter_not_this_one`] | — | the `InvalidArgument` kind a `WRONGTYPE` shares, isolating the message half of the predicate: since b.muu it lands on `snapshot_read_refused_total{kind="invalid_argument"}` and STILL leaves every `key_family` cell at zero (`test-fault-injection`) |
//! | [`checkpoint_assembly_over_a_wrongtype_header_ticks_header_exactly_once`] | `header` | the CHECKPOINT consumer's own `HGET obj:hdr seq` — a read no snapshot read issues, routed through the counting helper by b.muu |
//!
//! **The per-family labels are staged on the other two readers, not here.**
//! The **fenced read**
//! (`storage::snapshot_fence::read::read_object_snapshot_fenced`) — the read of
//! a layer above the atomic-read limits for catch-up and the checkpoint
//! assembler — runs each pass through the same `fanout_snapshot_read` as the
//! unfenced read, with the same per-command `key_family` labels, so staging
//! every family on it would re-prove
//! `snapshot_read_unfenced_test::unfenced_read_of_a_wrongtype_*_ticks_*`
//! against the same code. Those tests, and the atomic read's
//! `snapshot_read_atomic_test::atomic_read_of_a_wrongtype_*_ticks_*`, stage
//! every family, the header included, on this file's [`Fixture`] and
//! asserters; every one asserts every OTHER registered cell at zero, which
//! is what makes the `key_family` label discriminating rather than decorative.
//! What only the fenced read can show is its own wrapper's handling of a
//! permanent refusal — the one fenced staging below — and the two negative
//! controls on it. (Its first touch of `obj:hdr` is the fence's acquire
//! script, not a fan-out read, so a wrong-typed header is met there, under
//! `storage::snapshot_fence`'s classification, not this counter's.)
//!
//! The two `non_wrongtype_snapshot_read_errors` arms are the only ones that
//! reach `classify_snapshot_read_err` with an error THIS counter must not
//! count, and they are consequently the only ones that would fail if the
//! `is_server_wrongtype_error` gate were deleted and the `key_family` tick made
//! unconditional — see [`non_wrongtype_snapshot_read_errors`] for why neither
//! plain negative control can carry that claim. They need
//! `--features test-fault-injection` and are silently compiled out without it.
//! Since b.muu one of the two is no longer silent *overall*: a permanent
//! non-`WRONGTYPE` refusal ticks the sibling counter
//! `live_edit_snapshot_read_refused_total{kind}`, and both arms now assert that
//! sibling explicitly — the transient one at zero, the `InvalidArgument` one at
//! its own cell. That pairing is what keeps this counter's zero baseline
//! readable as "no foreign value on the `le:` keyspace" while the other counter
//! carries the permanent-backend-failure traffic. The refusal counter's own
//! per-`kind` coverage lives in `b_muu_tests`, which also owns the two readers
//! this file imports for it.
//!
//! # What is deliberately NOT here
//!
//! * **The enum↔label-set agreement.** `SnapshotKeyFamily::ALL` and
//!   `metrics::SNAPSHOT_READ_KEY_FAMILIES` are pinned as the same set by the
//!   in-crate unit test
//!   `storage::diff_read::tests::snapshot_key_family_all_matches_the_registered_label_set`,
//!   which is compile-cheap and needs no Valkey. Both items are `pub(crate)`, so
//!   an integration test could not reach them anyway.
//! * **Zero-seeding and the bounded label set.**
//!   `observability_metrics_test::snapshot_read_type_error_total_pre_initialises_its_bounded_key_family_set`,
//!   beside its `keyspace_type_error_total` twin.
//! * **The predicate's discrimination, in isolation.** `is_server_wrongtype_error`
//!   is unit tested beside itself in `src/storage/client.rs`, where a hand-built
//!   `fred::error::Error` costs nothing. What [`non_wrongtype_snapshot_read_errors`]
//!   adds is the *wiring*: that the gate is actually consulted on the way to the
//!   counter, which a unit test of the predicate cannot see.
//! * **The whole-commit refusal contract** (nothing written, header not
//!   advanced) — `b_459_tests::txn_wrongtype_children_list_key_is_caught_by_the_snapshot_read_not_the_probe`
//!   owns it, and this file's commit-path test asserts only the tick that test
//!   does not.

use std::sync::Arc;

use fred::interfaces::KeysInterface;
use ordered_float::OrderedFloat;

use live_edit::checkpoint::adapters::InMemoryCheckpointStore;
use live_edit::checkpoint::assembler::assemble_layer_checkpoint;
use live_edit::config::UpdateLayerBackend;
use live_edit::domain::types::{Field, FieldValue, ObjectState, Section, TimeSample};
use live_edit::error::ServiceError;
use live_edit::observability::metrics::test_util::{render, sample_sum, sample_value};
use live_edit::observability::metrics::{
    Metrics, SNAPSHOT_READ_KEY_FAMILIES, SNAPSHOT_READ_OUTCOMES,
};
use live_edit::storage::client::ValkeyClient;
use live_edit::storage::diff_read::{AtomicSnapshot, SnapshotReadConsumer};
use live_edit::storage::external_values::MockExternalValueStore;
use live_edit::storage::keys::KeySchema;
use live_edit::storage::layer::LayerStore;
use live_edit::storage::session::SessionStore;
use live_edit::storage::snapshot_fence::read::{read_object_snapshot_fenced, CappedFenceWait};
use live_edit::storage::snapshot_fence::LayerSize;
use live_edit::storage::test_support::relay::RelayedUpdates;
use live_edit::storage::test_support::TestValkey;
use oodf_ffi::{CommandCreateSection, CommandDeleteSection, OwnedDelta, StructCommand};

// b.muu's refusal-counter readers live beside the tests that own that counter,
// and are imported rather than copied for the reason
// `postcomp_orphan_sweep_fence_tests` imports the b.qpf keyspace readers: one
// definition, so the two files cannot drift on what "flat" means.
// (`assert_only_refusal_kind_ticked` is imported inside
// `non_wrongtype_snapshot_read_errors` instead: its single caller lives there,
// and that module is compiled out without `test-fault-injection`.)
use crate::storage::b_muu_tests::assert_no_refusal_kind_ticked;

// ═════════════════════════════════════════════════════════════════════════════
// Contract constants and fixture parameters
// ═════════════════════════════════════════════════════════════════════════════

/// Every seeded cell of the commit-side sibling
/// `live_edit_keyspace_type_error_total{backend,site}`.
///
/// Restated rather than imported because that counter's seeded cross-product is
/// built inline in `Metrics::new` and exported nowhere — unlike b.atj's own
/// vocabulary, which this file reads straight from
/// `metrics::SNAPSHOT_READ_KEY_FAMILIES` (the *rename* guard on those twelve labels
/// lives in
/// `observability_metrics_test::snapshot_read_type_error_total_pre_initialises_its_bounded_key_family_set`,
/// which restates them beside the `const`; a second restatement here would only
/// be a second place to keep in sync).
///
/// Read by [`assert_keyspace_type_errors_flat`]: b.atj's counter is a *new*
/// counter precisely so the commit-side one keeps its meaning, and a refusal
/// that ticked both would break both baselines — the or-joined alert recipe
/// would page twice for one key and the remediation text would disagree with
/// itself about whether a write was in flight.
const KEYSPACE_TYPE_ERROR_CELLS: [(&str, &str); 3] =
    [("lua", "update"), ("txn", "update"), ("txn", "create")];

/// The section every fixture below hangs its value rows off.
///
/// This constant, the three below it, [`create_section_delta`] and
/// [`delete_section_delta`] are `pub(crate)` for `snapshot_read_atomic_test`,
/// which stages the atomic read's refusals on this same [`Fixture`].
pub(crate) const SECTION_ID: u64 = 1;

/// The section the tombstone fixture creates and then deletes. Distinct from
/// [`SECTION_ID`] so the two fixtures' plants can never be confused.
pub(crate) const DOOMED_SECTION_ID: u64 = 2;

/// The key name of the seeded field row.
pub(crate) const FIELD_NAME: &str = "color";

/// The time of the seeded time-sample row. Exactly representable in `f64` so the
/// key the test builds and the key the reader builds are the same string.
pub(crate) const SAMPLE_TIME: f64 = 2.5;

/// What an out-of-band writer left behind. The bytes are irrelevant — what
/// matters is that the key is a `string` and every read below expects something
/// else.
const PLANTED_VALUE: &str = "planted-by-an-out-of-band-writer";

/// The size estimate the fenced stagings hand their read: a handful of items,
/// the order of these fixtures. It sizes only the fence's lease and wait cap,
/// and the fence key's stored size replaces it once a reply names one.
const FENCED_READ_ESTIMATE: LayerSize = LayerSize {
    items: 8,
    bytes: 256,
};

/// Ceiling on one fenced read here. An uncontended read of these fixtures
/// takes milliseconds; the budget only turns a hang into a failure well inside
/// nextest's 120 s kill, so the failure message is still printed.
const FENCED_READ_BUDGET: std::time::Duration = std::time::Duration::from_secs(30);

/// The fenced read of `object_id` (Spec SDD SR-5.3), as catch-up runs it above
/// the atomic-read limits: no writer competes, so acquire grants at once, and
/// the plain capped wait ([`CappedFenceWait`]) never waits. The read takes no
/// ceiling check of its own, so it runs on a small fixture as it would on a
/// large one; [`FENCED_READ_ESTIMATE`] stands in for the size a consumer
/// derives from the atomic read's `TOO_LARGE`. Bounded by
/// [`FENCED_READ_BUDGET`], so a fence that never granted fails the test rather
/// than hanging it.
///
/// `pub(crate)` for `b_zvd_tests`, which stages its read-side refusals and
/// seam failures on the same read.
pub(crate) async fn fenced_read(
    client: &ValkeyClient,
    keys: &KeySchema,
    object_id: u64,
) -> Result<AtomicSnapshot, ServiceError> {
    let consumer = SnapshotReadConsumer::Catchup;
    let mut wait = CappedFenceWait::new(client, object_id, consumer);
    tokio::time::timeout(
        FENCED_READ_BUDGET,
        read_object_snapshot_fenced(
            client,
            keys,
            object_id,
            consumer,
            FENCED_READ_ESTIMATE,
            &mut wait,
        ),
    )
    .await
    .unwrap_or_else(|_| panic!("the fenced read did not finish within {FENCED_READ_BUDGET:?}"))
}

// ═════════════════════════════════════════════════════════════════════════════
// Metric readers
// ═════════════════════════════════════════════════════════════════════════════

/// `live_edit_snapshot_read_type_error_total{key_family}`.
///
/// **Panics on an absent series rather than defaulting to zero.** All twelve cells
/// are zero-seeded in `Metrics::new`, so absence has exactly two causes and both
/// make every assertion in this file vacuous: the counter does not exist (the
/// pre-b.atj tree — every test here fails on this panic, which is the intended
/// pre-fix failure), or this `Metrics` handle was never installed on the client
/// under test, in which case the tick had nowhere to land. Defaulting to zero
/// would turn both into a green run.
///
/// Shape borrowed from `b_zvd_tests::keyspace_type_errors`.
///
/// `pub(crate)` for `b_muu_tests`, which asserts this counter flat on every arm
/// of its sibling.
pub(crate) fn snapshot_read_type_errors(metrics: &Metrics, family: &str) -> u64 {
    sample_value(
        &render(metrics),
        "live_edit_snapshot_read_type_error_total",
        &[("key_family", family)],
    )
    .unwrap_or_else(|| {
        panic!(
            "live_edit_snapshot_read_type_error_total{{key_family=\"{family}\"}} is absent from \
             the exposition. Either the counter does not exist (pre-b.atj: a snapshot read \
             refused on a wrong-typed key moved nothing, which is the defect this file covers), \
             or this `Metrics` handle was never installed on the `ValkeyClient` — \
             `classify_snapshot_read_err` reads the handle through \
             `ValkeyClient::metrics_snapshot()`, so without \
             `install_observability_hooks` the tick is silently skipped and every count below \
             reads zero for the wrong reason"
        )
    }) as u64
}

/// `live_edit_keyspace_type_error_total{backend, site}`, same panic-on-absent
/// posture and the same reason. Summed over the `mode` series the
/// `{backend="lua", site="update"}` cell is split into.
fn keyspace_type_errors(metrics: &Metrics, backend: &str, site: &str) -> u64 {
    sample_sum(
        &render(metrics),
        "live_edit_keyspace_type_error_total",
        &[("backend", backend), ("site", site)],
    )
    .unwrap_or_else(|| {
        panic!(
            "live_edit_keyspace_type_error_total{{backend=\"{backend}\", site=\"{site}\"}} is \
             absent from the exposition — every reachable cell is zero-seeded in `Metrics::new`, \
             so absence means this handle holds no metrics and the flatness assertion below could \
             not fail for any reason"
        )
    }) as u64
}

/// Assert that exactly one `key_family` cell moved, to exactly `expected_ticks`,
/// and that every other one is still at its seeded zero.
///
/// The other-zeros half is what makes the label discriminating rather than
/// decorative. A `classify_snapshot_read_err` that passed a constant family, or
/// a `SnapshotKeyFamily::label` arm wired to the neighbouring constant, ticks the
/// right *counter* and the wrong *cell* — which sends an operator to the wrong
/// key family and is invisible to a test that only checks the expected cell.
pub(crate) fn assert_only_family_ticked(metrics: &Metrics, expected: &str, expected_ticks: u64) {
    assert!(
        SNAPSHOT_READ_KEY_FAMILIES.contains(&expected),
        "{expected:?} is not a registered key family — the caller is asserting on a cell that \
         cannot exist, so this test would panic in `snapshot_read_type_errors` for a reason that \
         has nothing to do with the behaviour under test"
    );
    assert_eq!(
        snapshot_read_type_errors(metrics, expected),
        expected_ticks,
        "live_edit_snapshot_read_type_error_total{{key_family=\"{expected}\"}} must read \
         {expected_ticks} after the staged refusal. This counter IS the b.atj deliverable: the \
         shipped alert recipe is `increase(...) > 0`, so a refusal that moves nothing leaves an \
         operator with the flat line b.zvd's loud failure was supposed to end"
    );
    for &family in SNAPSHOT_READ_KEY_FAMILIES {
        if family == expected {
            continue;
        }
        assert_eq!(
            snapshot_read_type_errors(metrics, family),
            0,
            "only one key was wrong-typed, so only key_family=\"{expected}\" may move — \
             key_family=\"{family}\" moved too. Either a second read was also refused (the \
             staging is not the one this test intends) or the family label does not track the \
             read that produced it, which routes an operator to a key family that is perfectly \
             healthy"
        );
    }
}

/// Assert every `key_family` cell is still at its seeded zero.
pub(crate) fn assert_no_family_ticked(metrics: &Metrics) {
    for &family in SNAPSHOT_READ_KEY_FAMILIES {
        assert_eq!(
            snapshot_read_type_errors(metrics, family),
            0,
            "no key the snapshot read touches was wrong-typed, so no cell may move — \
             key_family=\"{family}\" did. A counter that ticks without a refusal destroys the \
             zero baseline the `increase(...) > 0` alert recipe rests on, which is a page for \
             nothing at 03:00"
        );
    }
}

/// Assert the commit-side sibling counter did not move on any of its three
/// cells.
pub(crate) fn assert_keyspace_type_errors_flat(metrics: &Metrics) {
    for (backend, site) in KEYSPACE_TYPE_ERROR_CELLS {
        assert_eq!(
            keyspace_type_errors(metrics, backend, site),
            0,
            "live_edit_keyspace_type_error_total{{backend=\"{backend}\", site=\"{site}\"}} is \
             written only by the pre-MULTI type probes, which are commit-side and never run on a \
             bare read. A tick here means one refusal was counted on BOTH counters, which \
             double-counts the or-joined alert recipe and hands the operator two contradictory \
             remediations — the commit-side one says a write was about to happen, and on this \
             path nothing was going to be written"
        );
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// Fixture
// ═════════════════════════════════════════════════════════════════════════════

/// A session, a layer, and a `Metrics` handle wired into BOTH the `ValkeyClient`
/// (which is where `classify_snapshot_read_err` reads it from) and the
/// `LayerStore` (which is where the commit-side counter would tick from).
///
/// Both halves are needed even by the read-only tests: without the client hooks
/// the tick is silently skipped, and without the store handle
/// [`assert_keyspace_type_errors_flat`] would be asserting over a counter that
/// had no way to move regardless.
///
/// `pub(crate)` — with the three fields and the five methods `b_muu_tests`
/// needs — so the b.muu arms stage against exactly this keyspace rather than a
/// near-copy of it. A second fixture would be the "copy-paste tests with
/// one-line differences" the test-review guide flags, and worse: the b.muu
/// assertions are all "every OTHER cell is flat", which is only meaningful if
/// the object under test has the same key inventory this file's are staged on.
pub(crate) struct Fixture {
    pub(crate) tv: TestValkey,
    pub(crate) metrics: Arc<Metrics>,
    store: LayerStore,
    session_id: String,
    pub(crate) object_id: u64,
    name: String,
}

impl Fixture {
    /// A layer with NO sections, fields, time samples or children lists: the
    /// smallest keyspace a read can stage against, so a plant on one of its
    /// indices is the only key of its family the read meets.
    pub(crate) async fn empty(name: &str) -> Self {
        Self::new(name, UpdateLayerBackend::Lua, empty_state()).await
    }

    /// A layer with one section carrying one field and one time sample — the
    /// shape that makes the `section` / `fields_index` / `time_samples_index` /
    /// `field` / `time_sample` families reachable.
    pub(crate) async fn seeded(name: &str) -> Self {
        Self::new(name, UpdateLayerBackend::Lua, seeded_state()).await
    }

    async fn new(name: &str, backend: UpdateLayerBackend, state: ObjectState) -> Self {
        let tv = TestValkey::setup().await;
        let metrics = Arc::new(crate::common::metrics_scrape_only("test"));
        tv.client()
            .install_observability_hooks(Arc::clone(&metrics))
            .await;

        let store = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096)
            .with_update_layer_backend(backend)
            .with_metrics(Arc::clone(&metrics));

        let session_id = SessionStore::new(tv.client().clone(), tv.key_schema().clone())
            .create(name, &format!("/{name}"), "b-atj-user")
            .await
            .expect("session creation must succeed");
        let object_id = store
            .create_layer(
                &session_id,
                name,
                "b-atj-user",
                &state,
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

    pub(crate) fn oid(&self) -> String {
        self.object_id.to_string()
    }

    /// Overwrite `key` with a plain string, whatever it held before. This is the
    /// out-of-band writer the whole counter exists to surface.
    pub(crate) async fn plant_string(&self, key: &str) {
        let _: () = self
            .tv
            .client()
            .pool
            .set(key, PLANTED_VALUE, None, None, false)
            .await
            .unwrap_or_else(|e| panic!("planting a string at {key} must succeed: {e}"));
    }

    /// [`fenced_read`] of this layer.
    async fn fenced_snapshot(&self) -> Result<AtomicSnapshot, ServiceError> {
        fenced_read(self.tv.client(), self.tv.key_schema(), self.object_id).await
    }

    /// Drive one delta through the real commit path.
    ///
    /// `pub(crate)` for `snapshot_read_atomic_test`, whose tombstone staging
    /// needs a real `DeleteSection` commit on this same fixture.
    pub(crate) async fn update(&self, delta: OwnedDelta) -> Result<(u64, u64), ServiceError> {
        self.store
            .update_layer_relayed(
                &self.session_id,
                &self.name,
                self.object_id,
                Some(delta),
                "b-atj-writer",
                1_000,
            )
            .await
    }
}

/// An `ObjectState` with no sections, fields or time samples.
fn empty_state() -> ObjectState {
    ObjectState {
        sequence_num: 1,
        first_version: 1,
        source_format: "usda".to_owned(),
        next_set_order: 1,
        ..Default::default()
    }
}

/// [`empty_state`] plus one root-child section, its children entry, one inline
/// field and one inline time sample — one key of every per-entry family the
/// snapshot read fans out over.
fn seeded_state() -> ObjectState {
    let mut state = empty_state();
    state.sections.insert(
        SECTION_ID,
        Section {
            section_id: SECTION_ID,
            parent_id: 0,
            name: "World".to_owned(),
            section_type: 1,
            section_order: 0,
            first_version: 1,
            last_version: 1,
        },
    );
    state.children.insert(0, vec![SECTION_ID]);
    state.fields.insert(
        (SECTION_ID, FIELD_NAME.to_owned()),
        Field {
            section_id: SECTION_ID,
            key_name: FIELD_NAME.to_owned(),
            value: FieldValue::Inline(vec![0x01]),
            set_order: 1,
            authoring_set_order: 1,
            version: 1,
        },
    );
    state.time_samples.insert(
        (SECTION_ID, OrderedFloat(SAMPLE_TIME)),
        TimeSample {
            section_id: SECTION_ID,
            time: OrderedFloat(SAMPLE_TIME),
            value: FieldValue::Inline(vec![0x02]),
            version: 1,
        },
    );
    state
}

pub(crate) fn create_section_delta(section_id: u64, name: &str) -> OwnedDelta {
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

pub(crate) fn delete_section_delta(section_id: u64) -> OwnedDelta {
    OwnedDelta {
        struct_commands: vec![StructCommand::DeleteSection(CommandDeleteSection {
            parent_id: 0,
            section_id,
        })],
        ..Default::default()
    }
}

// ═════════════════════════════════════════════════════════════════════════════
// The fenced read's own handling of a permanent refusal
// ═════════════════════════════════════════════════════════════════════════════

/// `live_edit_snapshot_read_total{consumer="catchup", path="fenced", outcome}`;
/// `0` while absent. One sample per fenced read attempt: a completed pass, a
/// broken one, or one that ended on an error.
fn fenced_catchup_reads(metrics: &Metrics, outcome: &str) -> u64 {
    sample_value(
        &render(metrics),
        "live_edit_snapshot_read_total",
        &[
            ("consumer", "catchup"),
            ("path", "fenced"),
            ("outcome", outcome),
        ],
    )
    .unwrap_or(0.0) as u64
}

/// `children` on the fenced read — the b.zvd wedge shape, and the family an
/// operator is most likely to meet: the fenced wrapper ends on the FIRST
/// permanent refusal, ticks exactly once, and does not retry.
///
/// The per-family labels are not this test's subject (the module header says
/// where they are staged); the fenced read's wrapper is. A fenced read is up
/// to `FENCE_READ_ATTEMPTS` attempts, and an attempt whose fan-out fails ends
/// the read with that command's error — it is not a broken attempt the read
/// retries (the release that follows reads only the header and the fence
/// key). A wrapper that treated the refusal as a broken attempt would re-read
/// the wrong-typed key on every attempt, ticking the counter once per attempt
/// and paging an operator several times for one bad key.
///
/// Five claims:
///
/// 1. **It refused at all**, with the `Internal` b.zvd chose. Asserted first so
///    a staging that silently succeeded (the planted key was never read) fails
///    as "no refusal" rather than as a puzzling zero on the counter.
/// 2. **It refused on the read this test staged**, identified by the op label
///    `map_snapshot_load_err` embeds, which carries the parent id — so it also
///    proves the plant landed under the synthetic root.
/// 3. **One attempt, ended on the error**: exactly one fenced read is recorded
///    for the catch-up consumer, and it is `outcome="permanent_error"` — no
///    `fence_broken` attempt and no second pass.
/// 4. **Exactly `children` ticked, once** ([`assert_only_family_ticked`]).
/// 5. **The commit-side counter stayed flat**
///    ([`assert_keyspace_type_errors_flat`]).
///
/// One tick, not "at least one", is sound at this staging: within the one
/// attempt the fan-out's windows are `?`-propagated in sequence, and
/// `try_join_all`'s small-vec arm abandons the poll pass on the first `Err`;
/// the fixture plants exactly ONE key and stays far under the 30-future
/// threshold at which `try_join_all` switches to its `FuturesOrdered` arm.
///
/// Staged on the seeded fixture: on a marked object the fan-out reads a list
/// only when `obj:chs` names its parent, and the seeded root has a child, so
/// `obj:chs` names "0"; the root's list is then the only list it reads (the
/// section is a leaf).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn fenced_read_of_a_wrongtype_children_list_ticks_children() {
    const ROOT_LIST_OP: &str = "LRANGE children list (parent 0)";

    let fx = Fixture::seeded("atj-ch").await;
    let key = fx.tv.key_schema().obj_children(&fx.oid(), "0");
    fx.plant_string(&key).await;

    // `AtomicSnapshot` is not `Debug`, so the `Ok` arm is spelled out rather
    // than reached through `expect_err`.
    let err = match fx.fenced_snapshot().await {
        Err(e) => e,
        Ok(snapshot) => panic!(
            "a wrong-typed {key} must fail the fenced read closed (b.zvd) before there is any \
             tick to assert; got Ok with {} sections / {} fields",
            snapshot.state.sections.len(),
            snapshot.state.fields.len()
        ),
    };

    let ServiceError::Internal(msg) = &err else {
        panic!(
            "a `WRONGTYPE` is permanent — no retry can retype a key — so the refusal must be a \
             non-retriable `Internal`. b.atj changed nothing here: `classify_snapshot_read_err` \
             hands every error to the same `map_snapshot_load_err`, so a different variant means \
             the counting helper started steering control flow, which is exactly what \
             `is_server_wrongtype_error`'s doc-comment forbids; got {err:?}"
        );
    };
    assert!(
        msg.contains(ROOT_LIST_OP),
        "the refusal must be the one this test staged — {ROOT_LIST_OP:?}. Any other message \
         means the planted key was never reached and some unrelated read refused first, which \
         would leave the counter assertions below describing a different staging entirely; got \
         {msg:?}"
    );

    let recorded: Vec<(&str, u64)> = SNAPSHOT_READ_OUTCOMES
        .iter()
        .map(|&outcome| (outcome, fenced_catchup_reads(&fx.metrics, outcome)))
        .filter(|&(_, n)| n > 0)
        .collect();
    assert_eq!(
        recorded,
        vec![("permanent_error", 1)],
        "the fenced read must end on its FIRST permanent refusal: one attempt, recorded once as \
         `permanent_error`. A `fence_broken` sample, or more than one sample, means the wrapper \
         treated the refusal as a broken attempt and read the wrong-typed key again"
    );

    assert_only_family_ticked(&fx.metrics, "children", 1);
    assert_keyspace_type_errors_flat(&fx.metrics);
}

// ═════════════════════════════════════════════════════════════════════════════
// The consumers that inherit the tick
// ═════════════════════════════════════════════════════════════════════════════

/// The commit consumer: a structural `UpdateLayer` refused by a wrong-typed
/// children list now moves a counter.
///
/// This is the operator-visible incident b.atj exists for. Pre-b.atj this exact
/// scenario — writes to one layer stop, permanently, until someone deletes a key
/// — produced a loud `Internal` per attempt and a flat line on every counter an
/// operator was watching, because the commit-side
/// `keyspace_type_error_total` is ticked only by the pre-`MULTI` probe and the
/// snapshot read refuses *before* the probe is reached.
///
/// Deliberately a sibling of, not an edit to,
/// `b_459_tests::txn_wrongtype_children_list_key_is_caught_by_the_snapshot_read_not_the_probe`:
/// that test owns the refusal contract (nothing written, header not advanced,
/// no `WRONGTYPE_KEY:` tag) and this one owns only the tick, so neither file has
/// to be read to understand the other's failure. The one claim they share —
/// `keyspace_type_error_total{txn,update}` staying flat — is asserted from both
/// sides on purpose: over there it proves the probe did not answer, here it
/// proves one refusal was not counted twice.
///
/// Staged on the seeded fixture, whose root already has a child, so the root's
/// list is one `obj:chs` names. The structural path reads through
/// `read_object_snapshot`'s unfenced fan-out, which on an object carrying the
/// `chs_complete` marker (every object `create_layer` writes) reads only the
/// lists `obj:chs` names (Spec SDD SR-1.2): a wrong-typed list it does not name
/// is never read, and the commit's `DEL` + `RPUSH` heals it instead —
/// `b_459_tests` pins that heal. The structural read takes a list's length
/// first (`LLEN`, then a ranged `LRANGE`; b.26e Stage 2), so the refusal names
/// the `LLEN`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn txn_structural_commit_over_a_wrongtype_children_list_ticks_children() {
    let fx = Fixture::new("atj-txn-ch", UpdateLayerBackend::Txn, seeded_state()).await;
    let key = fx.tv.key_schema().obj_children(&fx.oid(), "0");
    fx.plant_string(&key).await;

    let err = fx
        .update(create_section_delta(DOOMED_SECTION_ID, "Refused"))
        .await
        .expect_err(
            "a wrong-typed children list must fail the structural commit closed (b.zvd) — an \
             `Ok` here means the snapshot read swallowed the type error again and there is no \
             refusal to count",
        );
    assert!(
        matches!(&err, ServiceError::Internal(m) if m.contains("LLEN children list (parent 0)")),
        "the commit must be refused by the snapshot read, not by anything else; got {err:?}"
    );

    assert_only_family_ticked(&fx.metrics, "children", 1);
    assert_keyspace_type_errors_flat(&fx.metrics);
}

// ═════════════════════════════════════════════════════════════════════════════
// Negative controls
// ═════════════════════════════════════════════════════════════════════════════

/// The vocabulary is about READS, not about the object's keyspace: a foreign
/// value on a key the fenced read never touches moves no cell.
///
/// Keys genuinely part of this object's state and all reachable by the same
/// out-of-band writer, none of which the fenced read reads:
///
/// * a children list `obj:chs` does not name — here the leaf section's. It is
///   the sharpest control available: on a marked object the fan-out reads a
///   list only through `obj:chs`, and a helper that classified by key *prefix*
///   rather than by call site would tick `children` for it anyway. (The
///   unmarked read, which probes every live section's list, is pinned on the
///   unfenced fan-out in `snapshot_read_unfenced_test`.)
/// * `obj:stream`, the per-object live feed, which is read by the subscriber
///   consumer and never by the state fan-out;
/// * the commit-token record `obj:wtok`, the external-value index `obj:ext`,
///   and the ext-retire carrier and its attempt tokens — commit-side and
///   GC-side keys.
///
/// The fence's own keys (`obj:snapfence`, `obj:snapintent`) are deliberately
/// not planted: the fenced read's acquire and release scripts DO read them.
///
/// The read must still succeed. An error here would mean the fan-out grew a
/// dependency on one of these keys, which none of its families covers.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn fenced_read_over_wrongtype_keys_it_never_reads_ticks_nothing() {
    let fx = Fixture::seeded("atj-untouched").await;
    let keys = fx.tv.key_schema();
    let oid = fx.oid();
    for key in [
        // SECTION_ID is a leaf: no list, so `obj:chs` does not name one.
        keys.obj_children(&oid, &SECTION_ID.to_string()),
        keys.layer_stream(fx.object_id),
        keys.obj_write_token(&oid),
        keys.obj_ext_index(&oid),
        keys.obj_ext_retire_carrier(&oid),
        keys.obj_ext_retire_attempts(&oid),
    ] {
        fx.plant_string(&key).await;
    }

    let snapshot = match fx.fenced_snapshot().await {
        Ok(s) => s.state,
        Err(e) => panic!(
            "none of the planted keys is read by the fenced read, so a foreign value on them must \
             be invisible to it. A refusal means the read grew a dependency on one of them that \
             none of its `key_family` labels covers; got {e:?}"
        ),
    };
    assert_eq!(
        snapshot.sections.len(),
        1,
        "the snapshot must still carry the fixture's one section — a degraded-but-Ok read would \
         make the zero counts below vacuous"
    );

    assert_no_family_ticked(&fx.metrics);
    assert_keyspace_type_errors_flat(&fx.metrics);
}

/// A plain missing object is a `NotFound`, not a type error, and moves no cell.
///
/// The consumer-visible claim: the ordinary "the layer was deleted while a
/// subscriber was catching up" race — much the commonest way a snapshot read
/// fails in production — must leave the counter's baseline intact, or the
/// shipped `increase(...) > 0` recipe pages for a keyspace that is perfectly
/// healthy.
///
/// **This is not a test of the `is_server_wrongtype_error` gate**, and it is
/// worth being explicit about why, because the shape invites the reading. The
/// fence's acquire script finds no header and answers `NOT_FOUND`, a reply and
/// not an error, so the read ends `NotFound` before any fan-out command is
/// issued and `classify_snapshot_read_err` is never entered at all. The gate's behaviour on
/// an error that is not a `WRONGTYPE` is pinned instead by
/// [`non_wrongtype_snapshot_read_errors`], which has to inject one because no
/// keyspace can stage it.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn fenced_read_of_a_missing_object_ticks_nothing() {
    let fx = Fixture::empty("atj-missing").await;
    // An id inside this fixture's own prefix but belonging to no layer, so the
    // header comes back absent rather than wrong-typed.
    let absent = fx.object_id + 1_000_000;

    let err = match fenced_read(fx.tv.client(), fx.tv.key_schema(), absent).await {
        Err(e) => e,
        Ok(_) => panic!("object {absent} was never created, so the read must be a NotFound"),
    };
    assert!(
        matches!(err, ServiceError::NotFound(_)),
        "an absent object is `NotFound` — an absent header, not a type error; got {err:?}"
    );

    assert_no_family_ticked(&fx.metrics);
    assert_keyspace_type_errors_flat(&fx.metrics);
}

/// The checkpoint consumer's own header read, counted: a wrong-typed `obj:hdr`
/// met through CHECKPOINT ASSEMBLY ticks `header` exactly once.
///
/// # What changed, and why this test did not become a new file
///
/// This test existed before b.muu asserting the *opposite*, as a deliberate
/// scope pin on a documented gap: `checkpoint::assembler::ValkeySnapshotLoader`
/// issued four reads of its own — `read_header_seq` (`HGET obj:hdr seq`),
/// `read_header_identity` (`HGETALL obj:hdr`) and `read_tombstone`'s pair —
/// straight through `map_snapshot_load_err`, bypassing
/// `classify_snapshot_read_err` and therefore the counter. Because
/// `load_consistent_snapshot` takes its header pin *before* the state read, a
/// wrong-typed header was always met by that untracked `HGET` first, which made
/// `header` unreachable from this consumer altogether. b.muu routed all four
/// through the counting helper, so the same staging now ticks — and the test
/// flipped in place rather than being deleted and re-added elsewhere, which is
/// what its old doc-comment instructed.
///
/// # Why it is the `HGET obj:hdr seq` that answers
///
/// Step 1 of `load_consistent_snapshot` is the pin read, `?`-propagated before
/// any state read is issued, so the wrong-typed header is met there and the
/// `HGETALL obj:hdr` identity read (step 6, after the snapshot) never runs.
/// That ordering is what makes `== 1` provable rather than probable, and it is
/// the claim worth keeping: an assembler refactor that moved the pin after the
/// state read would meet the same key twice — once through the state read (the
/// atomic read, which reads the header itself, for this fixture's small layer;
/// the fenced read above the atomic-read limits) and once through the
/// assembler — and page an operator twice for one bad key.
///
/// This is also the one family the CHECKPOINT consumer can reach through a
/// *direct* assembler read: its only direct reads are the header pin
/// (`HGET obj:hdr seq`) and the header identity read (`HGETALL obj:hdr`). It
/// issues no tombstone read at all — every read it runs proves its cut, so
/// there is no raced section to reconcile — and a wrong-typed tombstone key is
/// met by the proven state read's own fan-out.
///
/// What this test still does NOT assert is the error variant. A `WRONGTYPE`
/// arrives here as `CheckpointError::Transient`
/// (`service_err_to_checkpoint_err` maps everything but `NotFound` that way),
/// which is arguably wrong for a permanent condition — but that classification
/// predates b.atj, belongs to the checkpoint lane, and b.muu deliberately left
/// it byte-identical, so pinning it here would give it a second home in a file
/// that has no business owning it.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn checkpoint_assembly_over_a_wrongtype_header_ticks_header_exactly_once() {
    let fx = Fixture::seeded("atj-muu").await;
    fx.plant_string(&fx.tv.key_schema().obj_header(&fx.oid()))
        .await;

    let err = assemble_layer_checkpoint(
        fx.tv.client(),
        fx.tv.key_schema(),
        fx.object_id,
        &InMemoryCheckpointStore::new(),
        "b-atj-replica",
        &mut live_edit::checkpoint::assembler::NoLeaseToKeep,
    )
    .await
    .err()
    .unwrap_or_else(|| {
        panic!(
            "a wrong-typed `obj:hdr` must fail checkpoint assembly closed — an `Ok` would mean a \
             DR generation was stamped from a header nobody could read"
        )
    });
    assert!(
        err.to_string().contains("WRONGTYPE"),
        "the assembler's refusal must carry Valkey's own type error, or this test is not staged \
         on the read it believes it is; got {err:?}"
    );
    // WHICH of the assembler's header reads answered. Every sibling staging in
    // this file identifies its read by the `op` label
    // `map_snapshot_load_err` embeds, and this test is the one with two
    // candidates on one key: `read_header_seq`'s pin (`HGET obj:hdr seq`) and
    // `read_header_identity`'s post-snapshot `HGETALL obj:hdr`. The ordering
    // argument in the doc-comment above — the pin runs first and
    // `?`-propagates, so the identity read never issues — is exactly what makes
    // `== 1` provable rather than probable, and this is the assertion that
    // actually pins it; the count alone reads 1 either way.
    //
    // The label survives into `err.to_string()` because
    // `service_err_to_checkpoint_err` builds `CheckpointError::Transient`'s
    // source with `format!("{other}")` over the `ServiceError::Internal("{op}
    // failed: {e}")` the mapper produced.
    assert!(
        err.to_string().contains("HGET obj:hdr seq"),
        "the refusal must come from the assembler's PIN read, `HGET obj:hdr seq`. A `HGETALL \
         obj:hdr` here is the identity read at step 6 answering instead, which means the pin no \
         longer precedes the state read — the refactor this test's exactly-once claim exists to \
         catch, and the one that would meet a bad header twice (once through the state \
         read, once through the assembler) and page an operator twice for one key; got {err:?}"
    );

    assert_only_family_ticked(&fx.metrics, "header", 1);
    // A `WRONGTYPE` is claimed by `count_object_key_refusal`'s FIRST arm, which
    // returns before the permanent/transient split is consulted — so the b.muu
    // sibling counter must stay flat. The two arms partitioning the error space
    // is what keeps each counter's alert recipe from double-paging for one key.
    assert_no_refusal_kind_ticked(&fx.metrics);
    assert_keyspace_type_errors_flat(&fx.metrics);
}

// ═════════════════════════════════════════════════════════════════════════════
// The gate itself: a snapshot read refused by something that is NOT a WRONGTYPE
// ═════════════════════════════════════════════════════════════════════════════

/// The `if crate::storage::client::is_server_wrongtype_error(&e)` gate at the
/// top of `classify_snapshot_read_err`, pinned from the side that makes it
/// load-bearing.
///
/// # Why the two negative controls above do not cover this
///
/// Neither of them enters `classify_snapshot_read_err` at all.
/// [`fenced_read_over_wrongtype_keys_it_never_reads_ticks_nothing`] produces
/// no error whatsoever — the read succeeds — and
/// [`fenced_read_of_a_missing_object_ticks_nothing`] ends on the acquire
/// script's `NOT_FOUND` answer, a reply and not an error, so again no fred
/// error is ever classified. Between them they say
/// only that a read which does not fail does not tick, which is a weaker claim
/// than it looks: **delete the `if`, tick unconditionally, and every other test
/// in this file still passes.** The family tests all stage a real
/// `WRONGTYPE`, where gated and ungated agree by construction.
///
/// These two tests are the arms that tell them apart. An unconditional tick
/// reads 1 on both.
///
/// # What is at stake in the gate
///
/// The counter's whole value is its zero baseline — the shipped alert is
/// `increase(...) > 0`, so *any* non-zero is a page. A transport blip on one
/// read of the fan-out is an ordinary event (it is precisely what b.zvd's
/// transient arm made retriable, and the retry loop absorbs it invisibly), so a
/// gate that let it through would page an operator, with a
/// `SNAPSHOT_READ_TYPE_ERROR_REMEDIATION` message telling them to go hunt an
/// out-of-band writer, for a Valkey that reconnected and a commit that landed.
///
/// Each test also re-asserts the classification the error arrives with, which is
/// the other half of the b.atj safety argument: the predicate may count and log
/// but must never steer control flow, so an `Aborted` must stay `Aborted`
/// (retriable) and an `Internal` must stay `Internal` (permanent) across the
/// counting helper. `src/storage/client.rs`'s
/// `map_snapshot_load_err_wrongtype_still_maps_to_internal_unchanged` pins that
/// invariance for the matched half; this is the unmatched half, through the real
/// call chain rather than a direct call to the mapper.
///
/// # Why a seam, and the activation condition that is silent
///
/// No keyspace can hold a `Timeout` still, so this arm cannot be staged against
/// a live server —
/// `ValkeyClient::install_object_snapshot_read_fault_injector_for_test` pops one
/// canned `fred::error::Error` in place of the atomic snapshot read's `EVALSHA`
/// result and feeds it into the **same** classification a real failure of that
/// call takes (`classify_object_snapshot_read_err` →
/// `classify_snapshot_read_err`, `src/storage/diff_read.rs`), so an injected
/// error and a real one provably share one path. The seam is
/// `#[cfg(any(test, feature = "test-fault-injection"))]`, so this module is
/// feature-gated to match — and that gate is the *silent* one: without
/// `--features test-fault-injection` these tests are compiled out entirely and
/// the run reports green having covered none of the above. Pass the feature
/// explicitly in any hand-rolled invocation.
#[cfg(feature = "test-fault-injection")]
pub(crate) mod non_wrongtype_snapshot_read_errors {
    use super::*;

    use fred::error::{Error as FredError, ErrorKind as FredErrorKind};
    use live_edit::storage::diff_read::{
        read_object_snapshot, ObjectSnapshotReadFaultInjector, SnapshotReadConsumer,
    };

    use crate::storage::b_muu_tests::assert_only_refusal_kind_ticked;

    /// The op label a failure of the atomic read's `EVALSHA` itself is
    /// classified under. Asserted in both arms so the surfaced error is
    /// provably the injected one rather than some unrelated failure that
    /// happened first.
    ///
    /// `pub(crate)` alongside [`refuse_snapshot_read_with`], for `b_muu_tests`.
    pub(crate) const INJECTED_READ_OP_LABEL: &str = "EVALSHA object_snapshot_read";

    /// Arm a one-shot injector, run the atomic snapshot read, and hand back the
    /// `ServiceError` it refused with.
    ///
    /// Two claims are asserted here because every arm shares them:
    ///
    /// 1. **It refused at all.** An `Ok` means the seam never fired, and the
    ///    counter assertions in the caller would then be describing an ordinary
    ///    healthy read — which ticks nothing for reasons that have nothing to do
    ///    with the gate.
    /// 2. **The queue drained** (`remaining() == 0`) — the vacuity guard, and the
    ///    only thing that distinguishes "the injected error was classified" from
    ///    "some earlier read failed first and the canned error is still sitting
    ///    in the queue".
    ///
    /// The atomic read is one `EVALSHA`, so exactly one call reaches the seam
    /// and one queued error answers it. A canned error that is not a tagged
    /// per-key refusal is a failure of the `EVALSHA` itself, attributed to the
    /// header's family and classified by the same `map_snapshot_load_err` every
    /// snapshot read ends in.
    ///
    /// `pub(crate)` since b.muu: `b_muu_tests` drives five more kinds through
    /// this same seam and must reach them through the SAME staging, or its
    /// "exactly one cell moved" claims would be about a different call path.
    pub(crate) async fn refuse_snapshot_read_with(
        fx: &Fixture,
        injected: FredError,
    ) -> ServiceError {
        let injector = Arc::new(ObjectSnapshotReadFaultInjector::new());
        injector.push(injected);
        fx.tv
            .client()
            .install_object_snapshot_read_fault_injector_for_test(injector.clone());

        // `SnapshotRead` is not `Debug`, so the `Ok` arm is spelled out rather
        // than reached through `expect_err`.
        let err = match read_object_snapshot(
            fx.tv.client(),
            fx.tv.key_schema(),
            &fx.oid(),
            SnapshotReadConsumer::Catchup,
        )
        .await
        {
            Err(e) => e,
            Ok(_) => panic!(
                "the injected atomic-read failure must surface — an `Ok` means the seam never \
                 fired, so the counter assertions that follow describe an ordinary healthy read \
                 and cover nothing"
            ),
        };

        assert_eq!(
            injector.remaining(),
            0,
            "the canned error must have been consumed — a leftover entry means the seam never \
             fired and this test is asserting about some other error entirely"
        );
        err
    }

    /// A **transient** snapshot-read failure moves NEITHER counter, and is
    /// still classified `Aborted`.
    ///
    /// A `Timeout` on one snapshot read is the everyday Valkey blip: b.zvd made
    /// it retriable precisely so the update-layer retry loop absorbs it and the
    /// commit lands, invisibly to the caller, and a subscriber's catch-up
    /// answers `UNAVAILABLE`, which its client retries. Ticking here would put a
    /// non-zero on a counter whose alert recipe is `increase(...) > 0` — an
    /// operator paged, and pointed at an out-of-band writer, for an incident that
    /// self-healed one attempt later.
    ///
    /// **This is the only test pinning that a transient failure ticks neither
    /// counter**, and since b.muu that is two distinct claims rather than one.
    /// `count_object_key_refusal`'s third arm — the implicit one, reached when
    /// `is_server_wrongtype_error` says no AND `is_transient_fred_error` says
    /// yes — writes nothing and logs nothing, and it is the whole reason the
    /// refusal counter's own baseline is readable: every other test in either
    /// file stages a permanent error, where an arm-2 gate that had been deleted
    /// and made unconditional would agree by construction. Delete
    /// `!is_transient_fred_error(e)` from arm 2 and this test is the one that
    /// goes red.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn snapshot_read_refused_by_a_transient_timeout_ticks_neither_counter() {
        let fx = Fixture::empty("atj-timeout").await;

        let err = refuse_snapshot_read_with(
            &fx,
            FredError::new(FredErrorKind::Timeout, "injected snapshot-read timeout"),
        )
        .await;

        let ServiceError::Aborted(msg) = &err else {
            panic!(
                "a transient failure must stay RETRIABLE across `classify_snapshot_read_err`. The \
                 counting helper is licensed only because it steers no control flow — every \
                 error, matched or not, goes to the unchanged `map_snapshot_load_err` — so an \
                 `Internal` here means the predicate started classifying, which turns a Valkey \
                 blip into a permanent failure for every snapshot consumer; got {err:?}"
            );
        };
        assert!(
            msg.contains(INJECTED_READ_OP_LABEL),
            "the refusal must be the injected one, naming the read that failed; got {msg:?}"
        );

        assert_no_family_ticked(&fx.metrics);
        assert_no_refusal_kind_ticked(&fx.metrics);
        assert_keyspace_type_errors_flat(&fx.metrics);
    }

    /// A **permanent** snapshot-read failure that is not a `WRONGTYPE` still
    /// moves no `key_family` cell — it moves the b.muu sibling's
    /// `kind="invalid_argument"` cell instead — and is still classified
    /// `Internal`.
    ///
    /// The sharper of the two arms, and the reason both exist. `WRONGTYPE` is
    /// recognised by `is_server_wrongtype_error` as *kind
    /// `InvalidArgument` **and** details starting `WRONGTYPE `* — two conditions,
    /// and the transient arm above fails the kind one, so on its own it cannot
    /// tell a correct predicate from one that matched on the kind alone. This
    /// error carries the same `InvalidArgument` kind with unrelated details, so
    /// it isolates the message half.
    ///
    /// That half is what keeps the counter's meaning honest: fred raises
    /// `InvalidArgument` for client-side argument validation that never reached a
    /// server at all, and counting one of those as a wrong-typed key would send
    /// an operator hunting an out-of-band writer for what is a bug in this
    /// service.
    ///
    /// # The b.muu half, and why it is asserted from this file
    ///
    /// Before b.muu this refusal moved nothing at all, and the test said so.
    /// That was the defect: a permanent, standing failure of every snapshot
    /// consumer, invisible on every counter. It now lands on
    /// `live_edit_snapshot_read_refused_total{kind="invalid_argument"}`, and
    /// asserting THAT here — beside every `key_family` cell staying at
    /// zero — is the pairing that makes this file's baseline claim precise:
    /// `snapshot_read_type_error_total` reads zero if and only if no foreign
    /// value sits on the `le:` keyspace, even though the service has since
    /// started counting a strictly larger set of refusals next door. A single
    /// counter carrying both would have cost the type-error counter exactly that
    /// property, and with it the `increase(...) > 0` recipe.
    ///
    /// The per-`kind` mapping itself (which fred kind lands on which cell, and
    /// that the other eleven cells stay flat) belongs to `b_muu_tests`; what is
    /// pinned here is only that the two counters partition the error space
    /// rather than overlap on it.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn snapshot_read_refused_by_a_non_wrongtype_invalid_argument_ticks_the_refusal_counter_not_this_one(
    ) {
        let fx = Fixture::empty("atj-invalid-arg").await;

        let err = refuse_snapshot_read_with(
            &fx,
            // fred's own wording for a client-side argument rejection: the
            // `InvalidArgument` kind a `WRONGTYPE` also carries, without the
            // prefix the predicate anchors on.
            FredError::new(FredErrorKind::InvalidArgument, "Invalid argument(s)"),
        )
        .await;

        let ServiceError::Internal(msg) = &err else {
            panic!(
                "a non-transient failure must stay PERMANENT across \
                 `classify_snapshot_read_err` — advertising a retry would spin the update-layer \
                 retry loop against a standing condition; got {err:?}"
            );
        };
        assert!(
            msg.contains(INJECTED_READ_OP_LABEL),
            "the refusal must be the injected one, naming the read that failed; got {msg:?}"
        );

        assert_no_family_ticked(&fx.metrics);
        assert_only_refusal_kind_ticked(&fx.metrics, "invalid_argument", 1);
        assert_keyspace_type_errors_flat(&fx.metrics);
    }
}
