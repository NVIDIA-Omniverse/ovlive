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

//! Creation-time first-base bootstrap (SDD SR-7.1 / SR-7.2 / SR-7.3;
//! PRD AC-5). Task `t2.dfh.57.kq`.
//!
//! Cases 1-6 (success, shape, fences) are Subtask `t3.dfh.57.kq.xu` (TEST-A);
//! cases 7-13 (gating matrix, PUT-failure arm, duplicate run, overwrite,
//! cursor-regression guard, empty content, vanished header) are Subtask
//! `t3.dfh.57.kq.nq` (TEST-B) and start at the TEST-B banner below. Cases 14-18
//! (b.vn6 — the POST-PUT half of the delete race, its fast-path reap and that
//! reap's two abort-before-delete arms, the pre-PUT liveness re-check, and the
//! refusal's reap-dependent log level) follow TEST-B under their own banners.
//!
//! ## What is under test
//!
//! `live_edit::checkpoint::BaseBootstrap` (`src/checkpoint/base_bootstrap.rs`).
//! Without it a new layer has no base — no lineage root, so tail capture
//! refuses `no_base` — until the periodic loop first assembles it. That
//! assembly proves its cut however hot the layer is (one atomic read at or
//! below the atomic-read limits, the fenced read above them), so the gap lasts
//! until the layer's first eligible tick. SR-7.1 closes it at creation instead:
//! it persists the committed creation state as a `CheckpointKind::Base` at
//! `snap_seq = 1` from an async post-commit task, and stamps the two header
//! cursor columns.
//!
//! The unit is driven **directly** here (`BaseBootstrap::run_once` /
//! `::spawn_creation_base`) rather than through a gRPC `CreateLayer`. That is
//! seam-faithful: the handler's entire contribution is the one
//! `spawn_creation_base(object_id, state)` call, and `run_once` is a `pub async fn`
//! precisely so the arms are awaitable without a request in flight. Every case
//! materializes a real layer via `LayerStore::create_layer` against a live
//! Valkey first, so the header, stream, and ext-store state the unit reads are
//! the production ones.
//!
//! ## SR-7.3 fence
//!
//! `CreateLayer`'s `1-0` seed stream entry stays `ompf`-less; this Task must not
//! touch stream writes. Case 4 pins that as an assertion rather than leaving it
//! to review.
//!
//! ## The metric families this file reasons about
//!
//! Two, and keeping them apart is what most of the prose below is doing.
//! `checkpoint_writes_total{result,kind="layer"}` counts generation-COMPLETION
//! outcomes fleet-wide, and every case here asserts against it via
//! [`BootstrapFixture::layer_writes`]. `live_edit_checkpoint_base_bootstrap_total{outcome,site}`
//! is the family Epic 5's SR-7 adjudication (Task 5 `t2.dfh.be.s8`, landed
//! 2026-09-01) minted for THIS unit: one increment per `BaseBootstrap::run_once`
//! invocation, ten outcomes × two sites (b.vn6 added the tenth,
//! `unstamped_header_absent`), so no arm of the unit is silent any more. The two
//! are additive, never substitutes — a guard skip owns a bootstrap-outcome cell
//! and moves no `writes_total` cell, because it completed no generation.
//!
//! This file asserts the NEGATIVE side of that split only: which
//! `writes_total` cells a declined arm may not move, plus (at the gate-off arm,
//! the one path that never enters `run_once`) that the whole bootstrap family
//! stays flat. The POSITIVE per-outcome cell assertions deliberately live in
//! the SR-7 lane's own bootstrap-observability module rather than being
//! scattered here — one home for the vocabulary, no drift surface.
//!
//! ## Hard scope fences (deliberate absences — do not "fix" them)
//!
//! - **No checkpoint loop.** No `setup_checkpoint_harness`, no
//!   `CheckpointLoopConfig`, no leader lease. Every case drives
//!   `create_layer` + the unit and asserts against the adapter and `obj:hdr`.
//! - **No `refused_no_base`, no tail capture, no `live_edit_checkpoint_tail_total`
//!   cell.** Task 7 (`t2.dfh.57.1u`) owns the AC-5 end-to-end arms through the
//!   real loop. Observing the skip-if-no-commits suppression itself is likewise
//!   Task 7's; case 1 asserts only its *precondition*.
//! - **No `super::helpers` wait helper.** `wait_for_checkpoint_at_least` is
//!   base-only and polls a store the *loop* is writing to; every case here
//!   awaits `run_once` directly and reads the adapter back, so `handle.kind` is
//!   asserted inline.
//!
//! ## Why this file is UNGATED
//!
//! No `#![cfg(feature = "test-checkpoint-fault-injection")]`. The repo's
//! documented preference for failing exactly one `CheckpointStore` method is a
//! hand-rolled delegating wrapper, not the `CheckpointFaultInjector` seam —
//! rationale at `lifecycle_session_writes_error_test.rs:93-103`. Case 3's
//! `SlowLayerPutStore` is exactly that shape. Keeping the file ungated means
//! it runs in feature-free narrow invocations, and it keeps the sibling
//! TEST-B subtask's failure arm (`FailingBasePutStore`) from silently compiling
//! out.
//!
//! ## Fixture contract (shared with TEST-B, `t3.dfh.57.kq.nq`)
//!
//! [`bootstrap_fixture`] is the single wiring point; TEST-B appends cases 7-13
//! to this file and reuses it rather than forking it. Its two parameters are
//! the extension points:
//!
//! - **the store to install** — any `Arc<dyn CheckpointStore>`. Assertions read
//!   it back through [`BootstrapFixture::store`], so a counting / failing /
//!   slowing wrapper must be built on `super::helpers::DelegatingCheckpointStore`
//!   and forward **every** method, including the ones with non-trivial trait
//!   defaults (`read_ext_refs_sidecar` above all — the trait default returns an
//!   empty set, so an unforwarded wrapper makes every sidecar assertion
//!   vacuously green).
//! - **the resolved `enabled` gate** — DERIVE it, never hardcode it. Build a
//!   `Config` via [`bootstrap_config`] and pass it through
//!   [`derive_bootstrap_enabled`], which spells the same three-way conjunction
//!   `main.rs` does. A test hardcoding `false` keeps passing if production
//!   stops gating.
//!
//! Waiting is deterministic — never `sleep` for synchronization. Two sanctioned
//! idioms: [`BootstrapFixture::drain`] (`close()` + bounded `wait()`, terminal —
//! one per fixture) and `super::helpers::wait_for_writer_idle(&tracker, 10)`
//! when the tracker must stay open. `SLOW_DELAY` in case 3's wrapper is
//! stimulus, not synchronization.
//!
//! Sandbox env note: these tests use `TestValkey::setup`, which reads
//! `TEST_VALKEY_URL`. Under the project sandbox that variable is preset; see
//! CLAUDE.md.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use bytes::Bytes;
use fred::interfaces::{
    HashesInterface, KeysInterface, SetsInterface, SortedSetsInterface, StreamsInterface,
};
use ordered_float::OrderedFloat;

use tokio_util::task::TaskTracker;

use live_edit::checkpoint::adapters::InMemoryCheckpointStore;
use live_edit::checkpoint::assembler::assemble_layer_checkpoint;
use live_edit::checkpoint::{
    BaseBootstrap, BootstrapOutcome, BootstrapTarget, CheckpointAdapter, CheckpointBlob,
    CheckpointError, CheckpointHandle, CheckpointKind, CheckpointMeta, CheckpointStore,
};
use live_edit::config::{Config, PodMemorySource};
use live_edit::domain::delta::apply_delta;
use live_edit::domain::ompf::OmpfContainer;
use live_edit::domain::types::{Field, FieldValue, ObjectState, Section, TimeSample};
use live_edit::observability::metrics::{
    SharedMetrics, CHECKPOINT_BASE_BOOTSTRAP_OUTCOMES, CHECKPOINT_BASE_BOOTSTRAP_SITES,
};
use live_edit::storage::client::ValkeyClient;
use live_edit::storage::external_values::{ExternalValueStore, ValkeyExternalValueStore};
use live_edit::storage::keys::sha256_to_hex;
use live_edit::storage::layer::LayerStore;
use live_edit::storage::notifications::{NOTIFICATION_PAYLOAD_FIELD, RELAY_OMPF_FIELD};
use live_edit::storage::session::SessionStore;
use live_edit::storage::test_support::TestValkey;

use super::helpers::DelegatingCheckpointStore;

/// Replica id stamped into every persisted `CheckpointMeta` as
/// `creator_replica_id`, and handed to the assembler oracle in case 2 so both
/// generations claim the same origin. It is also the label `Metrics::new`
/// carries — but note `checkpoint_writes_total` renders no `service` label
/// (see [`BootstrapFixture::layer_writes`]), so it is not a metric selector
/// here.
const SERVICE_LABEL: &str = "first-base-bootstrap-test-replica";

/// `snap_seq` every bootstrap generation must land at. Mirrors the unit's
/// private `BOOTSTRAP_SNAP_SEQ`; named here so the assertions read as the
/// contract rather than as a magic `1`.
const BOOTSTRAP_SNAP_SEQ: u64 = 1;

/// Bound on a tracker drain. A hang here is a `BaseBootstrap` bug (a task
/// that never completes, or one spawned onto a different tracker), not a
/// test-only flake — so it panics with a diagnostic rather than waiting
/// forever.
const DRAIN_TIMEOUT: Duration = Duration::from_secs(10);

/// Stream maxlen for the fixture's `LayerStore` — the production default.
const STREAM_MAXLEN: u64 = 4096;

// =============================================================================
// Gate derivation — derive from `Config`, never hardcode.
// =============================================================================

/// Build a `Config` with the three SR-7.1 gate inputs set explicitly.
///
/// Drives the real `Config::from_env` under `temp_env::with_vars` so the parsed
/// values are the ones a deployment with these env vars would get, not a
/// hand-assembled struct that could drift from the parser. The two endpoint
/// vars are required unconditionally by `Config::from_env` and are irrelevant
/// to this file.
fn bootstrap_config(checkpoint_enabled: bool, adapter: &str, tail_enabled: bool) -> Config {
    let enabled_str = if checkpoint_enabled { "true" } else { "false" };
    let tail_str = if tail_enabled { "true" } else { "false" };
    temp_env::with_vars(
        [
            (
                "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                Some("http://auth.example.com"),
            ),
            (
                "LIVE_EDIT_USERINFO_ENDPOINT",
                Some("http://userinfo.example.com"),
            ),
            ("LIVE_EDIT_CHECKPOINT_ENABLED", Some(enabled_str)),
            ("LIVE_EDIT_CHECKPOINT_ADAPTER", Some(adapter)),
            ("LIVE_EDIT_CHECKPOINT_TAIL_ENABLED", Some(tail_str)),
        ],
        || {
            Config::from_env_with(PodMemorySource::Reference)
                .expect("Config::from_env must succeed for the bootstrap gate matrix")
        },
    )
}

/// The SR-7.1 three-way gate, spelled exactly as `src/main.rs` spells it at the
/// `BaseBootstrap::new` call site.
///
/// Deliberately stricter than the session dispatcher's `!Disabled`: this unit
/// exists to serve the tail path, so it is off whenever the tail path is.
/// Keeping the expression here (rather than a literal at each construction
/// site) means a change to the gate's shape shows up as one edit, and a
/// production regression that drops a conjunct fires the gating-matrix
/// assertions instead of quietly passing.
fn derive_bootstrap_enabled(cfg: &Config) -> bool {
    cfg.checkpoint_enabled
        && !matches!(cfg.checkpoint_adapter, CheckpointAdapter::Disabled)
        && cfg.checkpoint_tail_enabled
}

/// All three gate inputs on — the shape every TEST-A case runs under.
fn all_gates_on_config() -> Config {
    bootstrap_config(true, "InMemory", true)
}

// =============================================================================
// Fixture.
// =============================================================================

/// Everything one bootstrap case needs, owned together so `TestValkey`'s
/// prefix-scoped Drop cleanup outlives the assertions.
struct BootstrapFixture {
    tv: TestValkey,
    session_id: String,
    layers: LayerStore,
    ext_values: Arc<ValkeyExternalValueStore>,
    tracker: TaskTracker,
    metrics: SharedMetrics,
    /// The same trait object the unit holds — assertions read generations back
    /// through this handle.
    store: Arc<dyn CheckpointStore>,
    bootstrap: BaseBootstrap,
}

/// Wire a `BaseBootstrap` over a live Valkey with `store` installed and
/// the gate resolved to `enabled`.
///
/// Models `main.rs`'s construction: same collaborator order, the shared
/// `TaskTracker` the graceful-shutdown drain owns, a `SharedMetrics` over this
/// file's own registry, and `SERVICE_LABEL` as the creator replica id. The
/// layer itself is NOT created here — cases seed their own state (see
/// [`seed_state`]) and call [`BootstrapFixture::create_layer`], because the
/// overwrite / empty-content arms need different creation shapes.
async fn bootstrap_fixture(store: Arc<dyn CheckpointStore>, enabled: bool) -> BootstrapFixture {
    let tv = TestValkey::setup().await;
    let metrics: SharedMetrics = Arc::new(crate::common::metrics_scrape_only(SERVICE_LABEL));
    let tracker = TaskTracker::new();

    let session_id = SessionStore::new(tv.client().clone(), tv.key_schema().clone())
        .create(
            "first-base-bootstrap-session",
            "/first-base-bootstrap",
            "first-base-test-user",
        )
        .await
        .expect("session creation must succeed against a healthy Valkey");

    let layers = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), STREAM_MAXLEN);
    let ext_values = Arc::new(ValkeyExternalValueStore::new(
        tv.client().clone(),
        tv.key_schema().clone(),
    ));

    let bootstrap = BaseBootstrap::new(
        Arc::clone(&store),
        tv.client().clone(),
        tv.key_schema().clone(),
        tracker.clone(),
        Arc::clone(&metrics),
        SERVICE_LABEL.to_string(),
        enabled,
    );

    BootstrapFixture {
        tv,
        session_id,
        layers,
        ext_values,
        tracker,
        metrics,
        store,
        bootstrap,
    }
}

impl BootstrapFixture {
    /// Materialize a real layer from `state`, returning its freshly-allocated
    /// `object_id` — the same value `handle_create_layer` passes to
    /// `spawn_creation_base`.
    async fn create_layer(&self, layer_name: &str, state: &ObjectState) -> u64 {
        self.layers
            .create_layer(
                &self.session_id,
                layer_name,
                "first-base-test-user",
                state,
                Arc::clone(&self.ext_values) as Arc<dyn ExternalValueStore>,
            )
            .await
            .expect("create_layer must succeed against a healthy Valkey")
    }

    /// Terminal drain: close the tracker and wait, bounded, for every spawned
    /// bootstrap task to finish. `TaskTracker::close` is terminal, so this runs
    /// at most once per fixture — a case that needs the tracker to stay open
    /// uses `super::helpers::wait_for_writer_idle` instead.
    async fn drain(&self) {
        self.tracker.close();
        tokio::time::timeout(DRAIN_TIMEOUT, self.tracker.wait())
            .await
            .expect(
                "the bootstrap TaskTracker must drain inside DRAIN_TIMEOUT — a hang here points \
                 at a BaseBootstrap task that never completes, not at a test-only flake",
            );
    }

    /// `HGET le:{oid}:obj:hdr <field>`.
    async fn header_field(&self, object_id: u64, field: &str) -> Option<String> {
        let key = self.tv.key_schema().obj_header(&object_id.to_string());
        self.tv
            .client()
            .pool
            .hget(&key, field)
            .await
            .expect("HGET obj:hdr must succeed")
    }

    /// `HEXISTS le:{oid}:obj:hdr <field>`.
    async fn header_field_exists(&self, object_id: u64, field: &str) -> bool {
        let key = self.tv.key_schema().obj_header(&object_id.to_string());
        self.tv
            .client()
            .pool
            .hexists(&key, field)
            .await
            .expect("HEXISTS obj:hdr must succeed")
    }

    /// `live_edit_checkpoint_writes_total{result=<result>, kind="layer"}` off
    /// this fixture's own registry.
    ///
    /// Matched on `(result, kind)` only, deliberately: unlike
    /// `checkpoint_recovery_attempts_total`, this family carries **no
    /// `service` label** — verified against the rendered exposition, where
    /// every cell is `{result=…,kind=…}` and nothing else. A `service` term in
    /// the filter matches zero lines and `sample_value` folds that to a silent
    /// `0`, which would make every tick assertion in this file read as
    /// "counter never moved". Cross-test isolation is supplied by
    /// `metrics_scrape_only`'s per-fixture registry rather than by a label.
    ///
    /// The whole `result × kind` grid is pre-initialised at `Metrics::new`, so
    /// every cell renders from boot and an unticked arm reads a real `0`
    /// rather than an absent series.
    fn layer_writes(&self, result: &str) -> u64 {
        live_edit::observability::metrics::test_util::sample_value(
            &live_edit::observability::metrics::test_util::render(&self.metrics),
            "live_edit_checkpoint_writes_total",
            &[("result", result), ("kind", "layer")],
        )
        .unwrap_or(0.0) as u64
    }

    /// Every rendered cell of `live_edit_checkpoint_base_bootstrap_total`, as
    /// `(outcome, site) → value`, off this fixture's own registry.
    ///
    /// Parsed WHOLE rather than per-cell, for the reason
    /// [`gather_base_bootstrap_cells`](super::helpers::gather_base_bootstrap_cells)
    /// records: an out-of-vocabulary `outcome` or `site` materialised on first
    /// occurrence shows up here as an EXTRA key, where a
    /// [`BootstrapFixture::layer_writes`]-shaped per-cell probe would never look
    /// for it. This file's only consumer is the gate-off arm, whose claim is
    /// that the WHOLE family is byte-identical before and after — a claim a
    /// single-cell probe cannot make.
    ///
    /// A one-line delegation to the shared reader, kept as a method purely so
    /// call sites read `fx.base_bootstrap_cells()` alongside the fixture's other
    /// probes: the parse is the suite's single copy, and this file's opposite
    /// claim (no cell moves, where the concurrency file's names one) lives in
    /// the assertions, not in the reader.
    fn base_bootstrap_cells(&self) -> HashMap<(String, String), f64> {
        super::helpers::gather_base_bootstrap_cells(&self.metrics)
    }
}

// =============================================================================
// Creation-state seeds.
// =============================================================================

/// A NON-TRIVIAL creation state: two root sections, one inline field each.
///
/// Deliberately not `ObjectState::default()` — an empty state would let a
/// broken build path (one that emitted nothing at all) pass every assertion
/// below. `seed` varies the section ids and payload bytes so two layers in one
/// test cannot be confused for each other. Copied in shape from
/// `tests/integration/storage/checkpointer_end_to_end_test.rs::seed_state`.
///
/// Inline values never promote to the ext store, so the ext-refs sidecar for
/// this state is `Some` but empty; case 5 uses [`seed_state_with_ext`] for the
/// non-empty arm.
fn seed_state(seed: u8) -> ObjectState {
    let mut state = ObjectState {
        sequence_num: 1,
        first_version: 1,
        source_format: "usda".to_string(),
        next_set_order: 0,
        ..Default::default()
    };

    let sec_a = u64::from(seed) * 10 + 1;
    let sec_b = u64::from(seed) * 10 + 2;
    state.sections.insert(
        sec_a,
        Section {
            section_id: sec_a,
            parent_id: 0,
            name: format!("layer-{seed}-sec-a"),
            section_type: 1,
            section_order: 0,
            first_version: 1,
            last_version: 1,
        },
    );
    state.sections.insert(
        sec_b,
        Section {
            section_id: sec_b,
            parent_id: 0,
            name: format!("layer-{seed}-sec-b"),
            section_type: 1,
            section_order: 1,
            first_version: 1,
            last_version: 1,
        },
    );
    // Root children list — `DiffEngine::sections_in_dfs_order` walks from root,
    // so without it the reset delta emits no sections at all.
    state.children.insert(0, vec![sec_a, sec_b]);

    for sid in [sec_a, sec_b] {
        let key = format!("k{sid}");
        state.fields.insert(
            (sid, key.clone()),
            Field {
                section_id: sid,
                key_name: key,
                value: FieldValue::Inline(vec![seed, sid as u8, 0xAA]),
                set_order: u64::from(seed),
                authoring_set_order: u64::from(seed),
                version: 1,
            },
        );
    }

    state
}

/// [`seed_state`] plus one `FieldValue::External` field, so `ordered_hashes`
/// (and therefore the persisted ext-refs sidecar) is non-empty.
///
/// The caller must have stored the blob behind `hash` in the ext store first —
/// `create_layer` does not mint ext values, it only records the references the
/// state already carries.
fn seed_state_with_ext(seed: u8, hash: [u8; 32], size: u64) -> ObjectState {
    let mut state = seed_state(seed);
    let sec_a = u64::from(seed) * 10 + 1;
    state.fields.insert(
        (sec_a, "ext-field".to_string()),
        Field {
            section_id: sec_a,
            key_name: "ext-field".to_string(),
            value: FieldValue::External {
                hash,
                size,
                ext_value_index: 0,
            },
            set_order: u64::from(seed) + 1,
            authoring_set_order: u64::from(seed) + 1,
            version: 1,
        },
    );
    state
}

fn sha256(bytes: &[u8]) -> [u8; 32] {
    use sha2::Digest;
    let mut hasher = sha2::Sha256::new();
    hasher.update(bytes);
    hasher.finalize().into()
}

/// `now` in unix milliseconds — the upper bound `last_checkpoint_ms` is checked
/// against.
fn now_unix_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("system clock must be after the unix epoch")
        .as_millis() as u64
}

/// Decode a persisted checkpoint blob into the `ObjectState` its reset delta
/// reconstructs, plus the raw OMPF bytes.
///
/// The reset delta is applied to a fresh `ObjectState`, which is what a DR
/// restore does — so this is the semantic content of the generation, not an
/// incidental encoding of it.
fn decode_blob(blob: &CheckpointBlob) -> (ObjectState, Bytes) {
    let ompf_bytes = blob.bytes.clone();
    let state = decode_reset_ompf(&ompf_bytes);
    (state, ompf_bytes)
}

/// The body of [`decode_blob`], over RAW OMPF bytes rather than a persisted
/// blob.
///
/// Split out for TEST-B's cases 9 and 12, which decode bytes captured at the
/// `put_layer_checkpoint` call site (case 9 compares the two racing PUTs'
/// payloads, which never both reach the store — the adapter collapses them) and
/// the blob read back off the store (case 12 asserts an *empty* reset applies
/// cleanly rather than erroring). Sharing one decoder keeps "decodes cleanly"
/// meaning the same thing on both paths.
fn decode_reset_ompf(ompf_bytes: &Bytes) -> ObjectState {
    let parsed = OmpfContainer::parse(ompf_bytes).expect("OMPF parse must succeed");
    let delta = oodf_ffi::parse_delta(parsed.delta_bytes)
        .expect("OODF delta parse must succeed — the bootstrap must never emit malformed bytes");
    let mut state = ObjectState::default();
    apply_delta(&mut state, Some(delta)).expect("apply_delta on the reset must succeed");
    state
}

/// Assert two reconstructed states are semantically isomorphic: same section
/// set with same `(parent_id, name, section_type)`, same field keys with
/// identical `FieldValue` payloads, same time-sample keys with identical
/// payloads, same `source_format`.
///
/// Per-entry version numbers are deliberately NOT compared — `apply_delta`
/// normalizes them to the applied sequence, so they carry no information about
/// the builder under test. Same rationale as `assembler_test.rs`'s round-trip
/// comparison.
fn assert_states_isomorphic(left: &ObjectState, right: &ObjectState, context: &str) {
    let left_sections: HashSet<u64> = left.sections.keys().copied().collect();
    let right_sections: HashSet<u64> = right.sections.keys().copied().collect();
    assert_eq!(
        left_sections, right_sections,
        "{context}: section id sets must match"
    );
    for sid in &left_sections {
        let l = &left.sections[sid];
        let r = &right.sections[sid];
        assert_eq!(
            r.parent_id, l.parent_id,
            "{context}: parent_id on sec {sid}"
        );
        assert_eq!(r.name, l.name, "{context}: name on sec {sid}");
        assert_eq!(
            r.section_type, l.section_type,
            "{context}: section_type on sec {sid}"
        );
    }

    let left_fields: HashMap<(u64, String), FieldValue> = left
        .fields
        .iter()
        .map(|((sid, k), f)| ((*sid, k.clone()), f.value.clone()))
        .collect();
    let right_fields: HashMap<(u64, String), FieldValue> = right
        .fields
        .iter()
        .map(|((sid, k), f)| ((*sid, k.clone()), f.value.clone()))
        .collect();
    assert_eq!(
        left_fields, right_fields,
        "{context}: field keys + payloads must match"
    );

    let left_ts: HashMap<(u64, OrderedFloat<f64>), FieldValue> = left
        .time_samples
        .iter()
        .map(|((sid, t), ts): (_, &TimeSample)| ((*sid, *t), ts.value.clone()))
        .collect();
    let right_ts: HashMap<(u64, OrderedFloat<f64>), FieldValue> = right
        .time_samples
        .iter()
        .map(|((sid, t), ts): (_, &TimeSample)| ((*sid, *t), ts.value.clone()))
        .collect();
    assert_eq!(
        left_ts, right_ts,
        "{context}: time-sample keys + payloads must match"
    );

    assert_eq!(
        left.source_format, right.source_format,
        "{context}: source_format must round-trip"
    );
}

// =============================================================================
// Case 1 — the success path.
// =============================================================================

/// All three gates on: the bootstrap persists exactly one `Base` generation at
/// `snap_seq = 1` and stamps BOTH cursor columns — and only those two.
///
/// The `restore_seq_bump` absence assertion is the sharp one; see the comment
/// at the assertion for why an absent column is the truthful value and a
/// fabricated `"0"` would be a corruption.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn bootstrap_persists_first_base_at_snap_seq_one_and_stamps_both_cursors() {
    // ── Arrange ──────────────────────────────────────────────────────────
    let cfg = all_gates_on_config();
    let enabled = derive_bootstrap_enabled(&cfg);
    assert!(
        enabled,
        "the all-gates-on Config must resolve the SR-7.1 three-way gate to true — if this fires, \
         `bootstrap_config` no longer produces the enabled deployment shape and every assertion \
         below is testing the disabled unit"
    );

    let in_memory = Arc::new(InMemoryCheckpointStore::new());
    let fx = bootstrap_fixture(Arc::clone(&in_memory) as Arc<dyn CheckpointStore>, enabled).await;

    let state = seed_state(1);
    let oid = fx.create_layer("first-base-success-layer", &state).await;
    let before_ms = now_unix_ms();

    // ── Act ──────────────────────────────────────────────────────────────
    fx.bootstrap
        .run_once(oid, state, BootstrapTarget::CreationBase)
        .await;
    fx.drain().await;

    // ── Assert: exactly one Base generation at snap_seq = 1 ──────────────
    let handles = fx
        .store
        .list_layer_checkpoints(oid)
        .await
        .expect("list_layer_checkpoints must succeed");
    assert_eq!(
        handles.len(),
        1,
        "exactly ONE generation must exist after a single bootstrap run; got {handles:?}"
    );
    let handle = &handles[0];
    assert_eq!(
        handle.snap_seq, BOOTSTRAP_SNAP_SEQ,
        "the bootstrap base must land at snap_seq = 1 — the header sequence CreateLayer commits \
         by construction"
    );
    assert_eq!(
        handle.object_id, oid,
        "the generation must be keyed on the object_id create_layer returned"
    );
    // Asserted inline: this case awaits `run_once` and reads the adapter back,
    // so there is no wait helper in the path whose base-only predicate could
    // stand in for the claim.
    assert_eq!(
        handle.kind,
        CheckpointKind::Base,
        "the bootstrap writes a BASE — a lineage root. A Tail here would mean the unit lost its \
         explicit `kind:` and inherited a default"
    );

    // ── Assert: both cursor columns stamped ──────────────────────────────
    let after_ms = now_unix_ms();
    let seq_col = fx
        .header_field(oid, "last_checkpoint_seq")
        .await
        .expect("last_checkpoint_seq must be present after a successful bootstrap");
    assert_eq!(
        seq_col, "1",
        "last_checkpoint_seq must be stamped to the persisted generation's snap_seq"
    );

    let ms_col: u64 = fx
        .header_field(oid, "last_checkpoint_ms")
        .await
        .expect("last_checkpoint_ms must be present after a successful bootstrap")
        .parse()
        .expect("last_checkpoint_ms must parse as u64");
    assert!(
        ms_col > 0,
        "last_checkpoint_ms must be a real wall-clock stamp, not 0 — the interval arm reads 0 as \
         `never checkpointed`, which would defeat the stamp entirely"
    );
    assert!(
        ms_col >= before_ms && ms_col <= after_ms,
        "last_checkpoint_ms ({ms_col}) must fall inside the run window [{before_ms}, {after_ms}]"
    );

    // ── Assert: `restore_seq_bump` ABSENT ────────────────────────────────
    //
    // The periodic loop writes this column as "0" (checkpointer.rs:2780-2783)
    // to RETIRE a correction a DR restore left behind. A freshly-created layer
    // never had a restore, so the bootstrap deliberately does not write it: an
    // ABSENT column already reads as a truthful 0 with `cursor_columns_trusted
    // = true` (assembler.rs:360-370), whereas a fabricated value would assert
    // that a restore happened here — and the assembler subtracts that bump from
    // the raw span, so a fabricated one corrupts the `observed_commit_span`
    // measurement the DR restore guard band consumes.
    assert!(
        !fx.header_field_exists(oid, "restore_seq_bump").await,
        "restore_seq_bump must be ABSENT after a bootstrap stamp — the bootstrap writes TWO \
         pairs, not three. Writing it (even as \"0\") asserts a restore happened on this layer \
         and feeds a wrong baseline into the observed-commit-span measurement"
    );

    // ── Assert: the skip-if-no-commits precondition ──────────────────────
    //
    // `seq == last_checkpoint_seq` is the gate the periodic loop consults
    // (checkpointer.rs:2526) to suppress a redundant first-tick re-snapshot of
    // the state the bootstrap just persisted — the mechanism behind SR-7.1's
    // "suppresses a redundant first-tick re-snapshot". Observing the skip
    // ITSELF requires the loop and is Task 7's (t2.dfh.57.1u); what this Task
    // owns is the precondition, asserted here.
    let seq = fx
        .header_field(oid, "seq")
        .await
        .expect("obj:hdr seq must be present on a materialized layer");
    assert_eq!(
        seq, "1",
        "CreateLayer commits the creation state at sequence 1 by construction"
    );
    assert_eq!(
        seq, seq_col,
        "obj:hdr seq and last_checkpoint_seq must be equal after the bootstrap — that equality is \
         the skip-if-no-commits gate's input"
    );
}

// =============================================================================
// Case 1b — persisted meta provenance.
// =============================================================================

/// Companion to case 1, covering the `CheckpointMeta` the unit stamps — the
/// fields case 1's handle-level assertions cannot see.
///
/// Added on review of the landed diff rather than pre-planned: step 4 of
/// `run_once` populates the meta from the header-identity read it already did,
/// and nothing else in this Task's test lane asserts it. Those fields are not
/// cosmetic — `apply_checkpoint` rebuilds `obj:hdr` (`session_id`,
/// `layer_name`, `created_by`, and all three timestamp columns) out of exactly
/// this meta, so a bootstrap base with blank identity would restore a layer
/// that no session owns.
///
/// The `observed_commit_span` assertion pins the sentinel deliberately: this is
/// a layer's FIRST generation, so there is no predecessor to measure against
/// and the honest value is "no measurement" (`0`), which folds the DR restore
/// guard band back to the configured `LIVE_EDIT_CHECKPOINT_SEQ_SAFETY_MARGIN`.
/// A fabricated span here would be commits-since-creation reported in the units
/// of a per-interval rate — the exact confusion `NO_COMMIT_SPAN_MEASUREMENT`
/// exists to prevent.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn bootstrap_base_meta_carries_the_header_identity_and_declines_a_commit_span_measurement() {
    // ── Arrange ──────────────────────────────────────────────────────────
    let cfg = all_gates_on_config();
    let enabled = derive_bootstrap_enabled(&cfg);
    assert!(
        enabled,
        "meta-provenance case must run under the enabled gate"
    );

    let in_memory = Arc::new(InMemoryCheckpointStore::new());
    let fx = bootstrap_fixture(Arc::clone(&in_memory) as Arc<dyn CheckpointStore>, enabled).await;

    let state = seed_state(7);
    let oid = fx.create_layer("first-base-meta-layer", &state).await;
    let before_ms = now_unix_ms();

    // ── Act ──────────────────────────────────────────────────────────────
    fx.bootstrap
        .run_once(oid, state, BootstrapTarget::CreationBase)
        .await;
    fx.drain().await;
    let after_ms = now_unix_ms();

    // ── Assert ───────────────────────────────────────────────────────────
    let blob = fx
        .store
        .get_latest_layer_checkpoint(oid)
        .await
        .expect("get_latest_layer_checkpoint must succeed")
        .expect("the bootstrap must have persisted a generation");
    let meta = &blob.meta;

    assert_eq!(
        meta.creator_replica_id, SERVICE_LABEL,
        "the persisted meta must attribute the blob to the replica that wrote it"
    );
    assert_eq!(
        meta.snap_seq,
        Some(BOOTSTRAP_SNAP_SEQ),
        "the meta's snap_seq must agree with the handle's"
    );
    assert_eq!(
        meta.kind,
        CheckpointKind::Base,
        "the bootstrap stamps `kind` explicitly rather than inheriting a default"
    );
    assert_eq!(
        meta.base_snap_seq, None,
        "a Base carries no lineage root of its own — the CheckpointMeta kind invariant"
    );
    assert_eq!(
        meta.from_seq, None,
        "a Base carries no commit-span lower bound — the CheckpointMeta kind invariant"
    );

    // Identity, read off the persisted header rather than threaded down from
    // the caller — the reason `run_once` does the HGETALL at all.
    assert_eq!(
        meta.session_id, fx.session_id,
        "session_id must round-trip from obj:hdr into the meta; apply_checkpoint rebuilds the \
         header from it"
    );
    assert_eq!(
        meta.layer_name, "first-base-meta-layer",
        "layer_name must round-trip from obj:hdr into the meta"
    );
    assert_eq!(
        meta.created_by, "first-base-test-user",
        "created_by must round-trip from obj:hdr into the meta"
    );

    let header_server_ts: u64 = fx
        .header_field(oid, "last_server_ts")
        .await
        .expect("obj:hdr last_server_ts must be present on a materialized layer")
        .parse()
        .expect("last_server_ts must parse as u64");
    assert!(
        header_server_ts > 0,
        "create_layer must mint a real server timestamp"
    );
    assert_eq!(
        meta.server_ts, header_server_ts,
        "server_ts must come from the persisted header — it is minted inside create_layer and \
         never returned to the caller, so a value invented here would silently drift"
    );

    assert_eq!(
        meta.observed_commit_span, 0,
        "a layer's FIRST generation has no predecessor to measure against, so the honest value is \
         the no-measurement sentinel 0 — which folds the DR restore band back to the configured \
         safety margin"
    );
    assert!(
        meta.written_at_unix_ms >= before_ms && meta.written_at_unix_ms <= after_ms,
        "written_at_unix_ms ({}) must fall inside the run window [{before_ms}, {after_ms}]",
        meta.written_at_unix_ms
    );
}

// =============================================================================
// Case 2 — assembler parity.
// =============================================================================

/// SR-7.1's "consistent by construction" claim, proven against the assembler as
/// an oracle: for the same layer state, the bootstrap's base and the assembler's
/// base reconstruct the same `ObjectState`, serialize to BYTE-IDENTICAL OMPF,
/// and name the same ext hashes.
///
/// **Why this case is the strongest byte-determinism witness available.** The
/// two builders are structurally different and take differently-shaped inputs —
/// the bootstrap builds from the handler's in-hand `ObjectState`, the assembler
/// from a snapshot loaded back out of Valkey and rebuilt by
/// `snapshot_to_object_state`. Two independently-shaped input pipelines
/// converging on identical bytes proves more than any same-builder repetition
/// can, because neither side can inherit the other's map layout.
///
/// **Assertion order is load-bearing.** `assert_states_isomorphic` runs FIRST,
/// so a genuine CONTENT divergence names itself semantically (which section,
/// which field, which payload) instead of surfacing as an opaque byte diff; the
/// byte assertion then fences emission ORDER, the one thing isomorphism cannot
/// see. That is the "byte equality with semantic isomorphism as the diagnostic
/// fallback" shape the tail-checkpoint Epic mandates for every re-base
/// idempotency assertion.
///
/// **Provenance — the pre-sort measurement, recorded not dropped (2026-08-29).**
/// Byte equality was attempted on this fixture *before* the emission order was
/// made total, and it did **not** hold. Decoding both deltas side by side showed:
///
/// - `struct_commands` identical and in the same order (two `DiffChangeSection`
///   rows, sections 21 then 22).
/// - `set_fields` carrying the *same three entries* with entry-for-entry
///   identical `section_id` / `key_name` / `value_or_ext_hash` /
///   `ext_value_size` / `ext_value_index` / `set_order` — but in a DIFFERENT
///   ORDER: the bootstrap emitted `[(22,"k22"), (21,"ext-field"), (21,"k21")]`
///   where the assembler emitted `[(21,"k21"), (21,"ext-field"), (22,"k22")]`.
/// - `time_samples`, `is_diff`, `base_version`, `source_format` identical.
///
/// Total blob length moved with that order too — 456 and 480 bytes both observed
/// for the same logical content — because FlatBuffers offsets and vtable sharing
/// depend on emission order. Cause, confirmed rather than assumed:
/// `build_reset_ompf_from_state` emits whatever order the full-content diff plan
/// hands it, and that plan walked `ObjectState::fields`, a `HashMap`, so the two
/// differently-constructed maps iterated differently.
///
/// **What the sort changed (b.gcb / b.vz7).**
/// `DiffEngine::plan_full_content` now orders all three emitted collections on
/// total orders over the CONTENT of `ObjectState` — fields on
/// `(version, authoring_set_order, section_id, key_name)`, time samples on
/// `(version, section_id, time)`, orphan sections on `section_id` — so no
/// `HashMap` table layout reaches the wire. Everything except the order already
/// matched in the measurement above; the sort removes the order, and this case
/// therefore asserts byte identity rather than reporting a divergence.
///
/// Emission order remains non-load-bearing for CORRECTNESS: every field carries
/// its own `authoring_set_order`, which `build_reset_ompf_from_state` remaps into
/// the single OODF `set_order` wire slot and which the faithful catch-up builders
/// reconstruct authoring order from. The byte assertion does not claim order
/// matters semantically — it fences the determinism that SR-9.2's cross-replica
/// re-base idempotency argument now stands on (identical state ⇒ identical bytes
/// ⇒ a re-base PUT at the same key is byte-idempotent).
///
/// **Sidecar comparison scope.** `read_ext_refs_sidecar` returns a
/// `HashSet<String>`, so the sidecar comparison below is SET-semantic and cannot
/// witness first-appearance ORDER. That order property is now SETTLED rather than
/// open: b.vz7 decided it in favour of the deterministic sort, and
/// `collect_external_value_hashes` inherits the order transitively by scanning
/// the slices `plan_full_content` already ordered. Byte-level sidecar evidence
/// lives where the sidecar bytes are actually observable —
/// `ext_refs_sidecar_json_is_byte_identical_for_equal_state` (unit, in
/// `src/checkpoint/assembler.rs`) and
/// `assembler_produces_byte_identical_blobs_across_independent_assemblies`
/// (integration, in `tests/integration/checkpoint/assembler_test.rs`). This
/// case's set-semantic comparison remains adequate for what IT claims — that the
/// two builders name the same hashes — and deliberately stays that way:
/// `fx.store` is the bootstrap's own store, and the comparison shape here is a
/// two-builder set equality, not a byte oracle.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn bootstrap_base_is_semantically_identical_to_the_assembler_base_for_the_same_state() {
    // ── Arrange ──────────────────────────────────────────────────────────
    let cfg = all_gates_on_config();
    let enabled = derive_bootstrap_enabled(&cfg);
    assert!(enabled, "parity case must run under the enabled gate");

    let bootstrap_store = Arc::new(InMemoryCheckpointStore::new());
    let fx = bootstrap_fixture(
        Arc::clone(&bootstrap_store) as Arc<dyn CheckpointStore>,
        enabled,
    )
    .await;

    // A payload with one ext-value reference alongside the inline fields, so
    // the sidecar comparison is not vacuously empty-vs-empty.
    let payload = b"first-base-parity-ext-payload";
    let hash = sha256(payload);
    let hash_hex = sha256_to_hex(&hash);
    fx.ext_values
        .store_values(&[(&hash, payload.as_slice())])
        .await
        .expect("ext store_values must succeed");

    let state = seed_state_with_ext(2, hash, payload.len() as u64);
    let oid = fx.create_layer("first-base-parity-layer", &state).await;

    // ── Act: the bootstrap's base ────────────────────────────────────────
    fx.bootstrap
        .run_once(oid, state, BootstrapTarget::CreationBase)
        .await;
    fx.drain().await;

    let bootstrap_blob = fx
        .store
        .get_latest_layer_checkpoint(oid)
        .await
        .expect("get_latest_layer_checkpoint must succeed")
        .expect("the bootstrap must have persisted a generation");

    // ── Act: the assembler's base, into a SEPARATE store ─────────────────
    //
    // One proven read of the idle layer (no concurrent writer, no regression);
    // the assembler writes at the header sequence, which is also 1 here.
    let oracle_store = Arc::new(InMemoryCheckpointStore::new());
    let decision = assemble_layer_checkpoint(
        fx.tv.client(),
        fx.tv.key_schema(),
        oid,
        oracle_store.as_ref(),
        SERVICE_LABEL,
        &mut live_edit::checkpoint::assembler::NoLeaseToKeep,
    )
    .await
    .expect("the assembler must succeed against a populated, idle layer");
    let outcome = match decision {
        live_edit::checkpoint::assembler::AssemblyDecision::Written(outcome) => outcome,
        live_edit::checkpoint::assembler::AssemblyDecision::SkippedNotQuiesced => panic!(
            "the assembler must prove a cut at or above its pin against an idle layer; got \
             SkippedNotQuiesced"
        ),
        live_edit::checkpoint::assembler::AssemblyDecision::LeaseLost => {
            panic!("the assembler was handed a keeper that never loses its lease; got LeaseLost")
        }
        live_edit::checkpoint::assembler::AssemblyDecision::ShuttingDown => {
            panic!("the assembler was handed a keeper that never shuts down; got ShuttingDown")
        }
    };
    let oracle_blob = oracle_store
        .get_latest_layer_checkpoint(oid)
        .await
        .expect("oracle get must succeed")
        .expect("the assembler must have persisted a generation");
    assert_eq!(
        outcome.handle.snap_seq, BOOTSTRAP_SNAP_SEQ,
        "the oracle assembles the same idle layer at the same sequence — if this drifts, the two \
         blobs are no longer comparable"
    );

    // ── Assert: semantic isomorphism (PRIMARY) ───────────────────────────
    let (bootstrap_state, bootstrap_bytes) = decode_blob(&bootstrap_blob);
    let (oracle_state, oracle_bytes) = decode_blob(&oracle_blob);
    assert_states_isomorphic(
        &bootstrap_state,
        &oracle_state,
        "bootstrap base vs assembler base",
    );

    // ── Assert: byte identity (the emission-ORDER fence) ─────────────────
    //
    // Deliberately AFTER the isomorphism assertion: that one has already ruled
    // out a content divergence and named it semantically if there were one, so
    // a failure here is specifically about ORDER — the only property bytes
    // carry and isomorphism cannot see.
    //
    // Length is asserted separately and first because it is the sharper
    // diagnostic: on identical content a length difference is the FlatBuffers
    // signature of a different emission order (offsets and vtable sharing
    // depend on it), which is exactly what the pre-sort probe measured here.
    assert_eq!(
        bootstrap_bytes.len(),
        oracle_bytes.len(),
        "bootstrap and assembler base OMPF must be the same LENGTH for the same layer state. On \
         content already proven isomorphic, a length difference is the FlatBuffers signature of a \
         different emission order — vtable sharing and offset widths depend on it. The 2026-08-29 \
         pre-sort probe measured 456 vs 480 bytes for exactly this state; see the doc-comment"
    );
    assert_eq!(
        bootstrap_bytes, oracle_bytes,
        "the bootstrap and the assembler must build BYTE-IDENTICAL base OMPF for the same layer \
         state. The isomorphism assertion above passed, so the content matches and this is an \
         emission ORDER divergence: DiffEngine::plan_full_content's total order over ObjectState \
         CONTENT (b.gcb / b.vz7) is what makes these two independently-shaped input pipelines \
         converge, and it is the ground under SR-9.2's cross-replica re-base idempotency \
         argument. Do NOT relax this to a non-fatal probe — decode both deltas, identify which \
         vector diverged, and fix the order"
    );

    // ── Assert: sidecars equal ───────────────────────────────────────────
    //
    // Both stores are the real `InMemoryCheckpointStore`, which OVERRIDES
    // `read_ext_refs_sidecar`. That matters: the trait default returns an empty
    // set, so a wrapper store that failed to forward this method would make
    // both sides vacuously `{}` and the assertion green for the wrong reason.
    let bootstrap_sidecar = fx
        .store
        .read_ext_refs_sidecar(&bootstrap_blob.handle)
        .await
        .expect("bootstrap sidecar read must succeed");
    let oracle_sidecar = oracle_store
        .read_ext_refs_sidecar(&oracle_blob.handle)
        .await
        .expect("oracle sidecar read must succeed");
    let expected: HashSet<String> = [hash_hex.clone()].into_iter().collect();
    assert_eq!(
        bootstrap_sidecar, expected,
        "the bootstrap sidecar must name exactly the creation state's ext hash — an empty set \
         here would also be what an unforwarded trait default produces, so this pins the \
         non-vacuous case"
    );
    assert_eq!(
        bootstrap_sidecar, oracle_sidecar,
        "bootstrap and assembler must name the same ext hashes (set semantics — see the \
         doc-comment's sidecar-comparison-scope note for where the byte-level first-appearance \
         order evidence lives)"
    );
}

// =============================================================================
// Case 3 — post-commit async shape.
// =============================================================================

/// Delay the slow store inserts INSIDE `put_layer_checkpoint` before delegating.
///
/// This is **stimulus, not synchronization**: it is not used to wait for
/// anything, it exists so that a bootstrap which (incorrectly) ran
/// synchronously on the caller's stack would still be inside the PUT when the
/// "did the commit path block?" assertion is evaluated. Long enough that the
/// 0-before-drain observation cannot be a scheduling coincidence; short enough
/// to keep the case fast.
const SLOW_DELAY: Duration = Duration::from_millis(200);

/// Wraps a real `InMemoryCheckpointStore` and sleeps [`SLOW_DELAY`] inside
/// `put_layer_checkpoint` BEFORE delegating, incrementing
/// `put_completed_calls` AFTER the delegated call returns.
///
/// Counter-after-delegate is load-bearing: a regression that abandoned
/// in-flight bootstrap tasks at `tracker.close()` would cancel the sleep, the
/// increment line would never run, and the post-drain assertion would observe
/// zero even though the spawn was issued.
///
/// Every other method delegates verbatim through
/// [`DelegatingCheckpointStore`] — including `read_ext_refs_sidecar`, whose
/// non-trivial trait default would otherwise substitute behaviour silently.
struct SlowLayerPutStore {
    inner: DelegatingCheckpointStore<InMemoryCheckpointStore>,
    put_completed_calls: AtomicUsize,
}

impl SlowLayerPutStore {
    fn new() -> Self {
        Self {
            inner: DelegatingCheckpointStore::new(Arc::new(InMemoryCheckpointStore::new())),
            put_completed_calls: AtomicUsize::new(0),
        }
    }

    fn put_completed(&self) -> usize {
        self.put_completed_calls.load(Ordering::Acquire)
    }
}

#[async_trait]
impl CheckpointStore for SlowLayerPutStore {
    async fn put_layer_checkpoint(
        &self,
        object_id: u64,
        snap_seq: u64,
        ompf: Bytes,
        meta: CheckpointMeta,
    ) -> Result<CheckpointHandle, CheckpointError> {
        tokio::time::sleep(SLOW_DELAY).await;
        let handle = self
            .inner
            .put_layer_checkpoint(object_id, snap_seq, ompf, meta)
            .await;
        self.put_completed_calls.fetch_add(1, Ordering::Release);
        handle
    }

    async fn get_latest_layer_checkpoint(
        &self,
        object_id: u64,
    ) -> Result<Option<CheckpointBlob>, CheckpointError> {
        self.inner.get_latest_layer_checkpoint(object_id).await
    }

    async fn get_layer_checkpoint(
        &self,
        handle: &CheckpointHandle,
    ) -> Result<Option<CheckpointBlob>, CheckpointError> {
        self.inner.get_layer_checkpoint(handle).await
    }

    async fn list_layer_checkpoints(
        &self,
        object_id: u64,
    ) -> Result<Vec<CheckpointHandle>, CheckpointError> {
        self.inner.list_layer_checkpoints(object_id).await
    }

    async fn delete_layer_checkpoint(
        &self,
        handle: CheckpointHandle,
    ) -> Result<(), CheckpointError> {
        self.inner.delete_layer_checkpoint(handle).await
    }

    async fn put_session_checkpoint(
        &self,
        session_id: &str,
        body: Bytes,
        meta: CheckpointMeta,
    ) -> Result<CheckpointHandle, CheckpointError> {
        self.inner
            .put_session_checkpoint(session_id, body, meta)
            .await
    }

    async fn get_session_checkpoint(
        &self,
        session_id: &str,
    ) -> Result<Option<CheckpointBlob>, CheckpointError> {
        self.inner.get_session_checkpoint(session_id).await
    }

    async fn delete_session_checkpoint(&self, session_id: &str) -> Result<(), CheckpointError> {
        self.inner.delete_session_checkpoint(session_id).await
    }

    async fn list_session_checkpoint_ids(&self) -> Result<Vec<String>, CheckpointError> {
        self.inner.list_session_checkpoint_ids().await
    }

    async fn put_ext_blob(&self, hash_hex: &str, bytes: Bytes) -> Result<(), CheckpointError> {
        self.inner.put_ext_blob(hash_hex, bytes).await
    }

    async fn get_ext_blob(&self, hash_hex: &str) -> Result<Option<Bytes>, CheckpointError> {
        self.inner.get_ext_blob(hash_hex).await
    }

    async fn ext_blob_exists(&self, hash_hex: &str) -> Result<bool, CheckpointError> {
        self.inner.ext_blob_exists(hash_hex).await
    }

    async fn delete_ext_blob(&self, hash_hex: &str) -> Result<(), CheckpointError> {
        self.inner.delete_ext_blob(hash_hex).await
    }

    async fn put_counters_sidecar(&self, body: Bytes) -> Result<(), CheckpointError> {
        self.inner.put_counters_sidecar(body).await
    }

    async fn get_counters_sidecar(&self) -> Result<Option<Bytes>, CheckpointError> {
        self.inner.get_counters_sidecar().await
    }

    async fn list_all_object_ids(&self) -> Result<Vec<u64>, CheckpointError> {
        self.inner.list_all_object_ids().await
    }

    async fn list_referenced_ext_hashes(&self) -> Result<HashSet<String>, CheckpointError> {
        self.inner.list_referenced_ext_hashes().await
    }

    async fn list_live_ext_set(
        &self,
    ) -> Result<live_edit::checkpoint::LiveExtSet, CheckpointError> {
        self.inner.list_live_ext_set().await
    }

    // The four methods below carry non-trivial trait defaults. Omitting them
    // would not merely skip instrumentation, it would SUBSTITUTE behaviour
    // (`Ok(empty)` for the per-handle sidecar read, `Ok(vec![])` for the raw
    // artifact enumeration, `Ok(0)` for temp reclamation, the local bound for
    // the publish worst case) with nothing failing — see the same note on
    // `DelegatingCheckpointStore`.
    async fn read_ext_refs_sidecar(
        &self,
        handle: &CheckpointHandle,
    ) -> Result<HashSet<String>, CheckpointError> {
        self.inner.read_ext_refs_sidecar(handle).await
    }

    async fn list_layer_artifacts(
        &self,
    ) -> Result<Vec<live_edit::checkpoint::LayerArtifact>, CheckpointError> {
        self.inner.list_layer_artifacts().await
    }

    async fn delete_layer_temp_artifacts(
        &self,
        object_id: u64,
        snap_seq: u64,
    ) -> Result<usize, CheckpointError> {
        self.inner
            .delete_layer_temp_artifacts(object_id, snap_seq)
            .await
    }

    fn publish_worst_case(&self) -> Duration {
        self.inner.publish_worst_case()
    }
}

/// The zero-synchronous-work fence, asserted as SHAPE — not latency.
///
/// `spawn_creation_base` returns as soon as the task is on the tracker, so with a
/// store that takes [`SLOW_DELAY`] inside its PUT, the completion counter must
/// still read 0 the instant the call returns. After the terminal tracker drain
/// it must read exactly 1 — proving the task really was spawned on the SHARED
/// tracker (the one graceful shutdown closes and waits on), not dropped or
/// spawned somewhere unjoined.
///
/// No wall-clock assertion appears anywhere in this case: timing assertions
/// never belong in a gating suite.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn bootstrap_does_no_synchronous_work_on_the_create_layer_commit_path() {
    // ── Arrange ──────────────────────────────────────────────────────────
    let cfg = all_gates_on_config();
    let enabled = derive_bootstrap_enabled(&cfg);
    assert!(enabled, "async-shape case must run under the enabled gate");

    let slow = Arc::new(SlowLayerPutStore::new());
    let fx = bootstrap_fixture(Arc::clone(&slow) as Arc<dyn CheckpointStore>, enabled).await;

    let state = seed_state(3);
    let oid = fx
        .create_layer("first-base-async-shape-layer", &state)
        .await;

    // ── Act: the handler's entire contribution, verbatim ─────────────────
    fx.bootstrap.spawn_creation_base(oid, state);

    // ── Assert: the commit path did not block ────────────────────────────
    assert_eq!(
        slow.put_completed(),
        0,
        "`spawn_creation_base` must return before the adapter PUT completes. A non-zero counter here \
         means the base capture ran synchronously on the CreateLayer commit path — the fence this \
         Task's Success Criteria call out"
    );

    // ── Assert: it really was spawned on the shared tracker ──────────────
    fx.drain().await;
    assert_eq!(
        slow.put_completed(),
        1,
        "after closing and draining the SHARED TaskTracker the bootstrap PUT must have completed \
         exactly once — a 0 means the task was dropped or spawned onto a tracker graceful \
         shutdown never joins"
    );
}

// =============================================================================
// Case 4 — SR-7.3 stream fence.
// =============================================================================

/// SR-7.3: `CreateLayer`'s `1-0` seed stream entry stays `ompf`-less, and the
/// bootstrap does not touch stream writes at all.
///
/// Asserted by NAME on both sides — the payload field must be present and the
/// relay field must be absent — because a regression that started attaching the
/// bootstrap's reset OMPF to the seed entry would be invisible to an
/// entry-count-only check.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn bootstrap_leaves_the_create_layer_seed_stream_entry_untouched() {
    // ── Arrange ──────────────────────────────────────────────────────────
    let cfg = all_gates_on_config();
    let enabled = derive_bootstrap_enabled(&cfg);
    assert!(enabled, "stream-fence case must run under the enabled gate");

    let in_memory = Arc::new(InMemoryCheckpointStore::new());
    let fx = bootstrap_fixture(Arc::clone(&in_memory) as Arc<dyn CheckpointStore>, enabled).await;

    let state = seed_state(4);
    let oid = fx
        .create_layer("first-base-stream-fence-layer", &state)
        .await;

    // ── Act ──────────────────────────────────────────────────────────────
    fx.bootstrap
        .run_once(oid, state, BootstrapTarget::CreationBase)
        .await;
    fx.drain().await;

    // Precondition for the assertions below: the bootstrap actually ran and
    // succeeded, so an untouched stream is a fence being honoured rather than
    // a unit that silently did nothing.
    let handles = fx
        .store
        .list_layer_checkpoints(oid)
        .await
        .expect("list_layer_checkpoints must succeed");
    assert_eq!(
        handles.len(),
        1,
        "the stream fence is only meaningful if the bootstrap ran; expected one generation"
    );

    // ── Assert: exactly one `1-0` entry, payload-only ────────────────────
    let stream_key = fx.tv.key_schema().layer_stream(oid);
    let entries: Vec<(String, Vec<(String, String)>)> = fx
        .tv
        .client()
        .pool
        .xrange(&stream_key, "-", "+", None)
        .await
        .expect("XRANGE over the per-object stream must succeed");

    assert_eq!(
        entries.len(),
        1,
        "the per-object stream must carry exactly the CreateLayer seed entry after a bootstrap; \
         got {entries:?}"
    );
    let (entry_id, fields) = &entries[0];
    assert_eq!(
        entry_id, "1-0",
        "the seed entry is written at the fixed id 1-0 inside CreateLayer's MULTI/EXEC"
    );

    let field_names: HashSet<&str> = fields.iter().map(|(k, _)| k.as_str()).collect();
    assert!(
        field_names.contains(NOTIFICATION_PAYLOAD_FIELD),
        "the seed entry must carry the notification payload field `{NOTIFICATION_PAYLOAD_FIELD}`; \
         got fields {field_names:?}"
    );
    assert!(
        !field_names.contains(RELAY_OMPF_FIELD),
        "SR-7.3: the CreateLayer seed entry must stay `{RELAY_OMPF_FIELD}`-less. A relay field \
         here means something started attaching OMPF bytes to the seed entry — the Existing \
         Behavior fence this Task must not cross"
    );
    assert_eq!(
        field_names.len(),
        1,
        "the seed entry must carry ONLY the payload field; got {field_names:?}"
    );
}

// =============================================================================
// Case 5 — sidecar contents.
// =============================================================================

/// A creation state referencing an external value must produce a base whose
/// ext-refs sidecar names exactly that hash, lowercase hex.
///
/// The sidecar is what the retention cascade and the DR ext-blob recovery path
/// read to decide which content-addressed blobs a generation still needs; a
/// bootstrap base that dropped its references would let those blobs be
/// collected out from under the only generation that names them.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn bootstrap_base_sidecar_lists_the_creation_states_ext_hashes() {
    // ── Arrange ──────────────────────────────────────────────────────────
    let cfg = all_gates_on_config();
    let enabled = derive_bootstrap_enabled(&cfg);
    assert!(enabled, "sidecar case must run under the enabled gate");

    let in_memory = Arc::new(InMemoryCheckpointStore::new());
    let fx = bootstrap_fixture(Arc::clone(&in_memory) as Arc<dyn CheckpointStore>, enabled).await;

    let payload = b"first-base-sidecar-ext-payload-bytes";
    let hash = sha256(payload);
    let hash_hex = sha256_to_hex(&hash);
    fx.ext_values
        .store_values(&[(&hash, payload.as_slice())])
        .await
        .expect("ext store_values must succeed");

    // Inline fields alongside the External one, so a sidecar that naively
    // enumerated every field would be caught by the exact-set comparison.
    let state = seed_state_with_ext(5, hash, payload.len() as u64);
    let oid = fx.create_layer("first-base-sidecar-layer", &state).await;

    // ── Act ──────────────────────────────────────────────────────────────
    fx.bootstrap
        .run_once(oid, state, BootstrapTarget::CreationBase)
        .await;
    fx.drain().await;

    // ── Assert ───────────────────────────────────────────────────────────
    let handles = fx
        .store
        .list_layer_checkpoints(oid)
        .await
        .expect("list_layer_checkpoints must succeed");
    assert_eq!(
        handles.len(),
        1,
        "expected exactly one bootstrap generation"
    );

    let sidecar = fx
        .store
        .read_ext_refs_sidecar(&handles[0])
        .await
        .expect("read_ext_refs_sidecar must succeed on a bootstrap handle");
    let expected: HashSet<String> = [hash_hex.clone()].into_iter().collect();
    assert_eq!(
        sidecar, expected,
        "the base sidecar must name exactly the creation state's ext hashes, lowercase hex \
         (expected {hash_hex})"
    );
}

// =============================================================================
// Case 6 — metric.
// =============================================================================

/// A successful bootstrap ticks the EXISTING layer-write counter
/// (`checkpoint_writes_total{result="success", kind="layer"}`) — reuse of that
/// family, with no new `result` / `kind` value and no label-cardinality change
/// on it.
///
/// Scoped to that family deliberately. Epic 5's SR-7 adjudication (Task 5
/// `t2.dfh.be.s8`, 2026-09-01) later minted a SECOND, additive family for this
/// unit (`live_edit_checkpoint_base_bootstrap_total{outcome,site}`, on which the
/// same invocation ticks `{persisted, creation_base}`), so "no new family" is a
/// claim about `checkpoint_writes_total`, never about the unit as a whole. The
/// two are additive: a completed generation is the one arm that ticks both.
///
/// Asserted as a delta against a baseline read on the same registry, not as an
/// absolute value, so the case stays valid if anything else in the process ever
/// ticks the same series. The `{result="error"}` counterpart is TEST-B's
/// (case 8).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn bootstrap_ticks_the_layer_write_counter() {
    // ── Arrange ──────────────────────────────────────────────────────────
    let cfg = all_gates_on_config();
    let enabled = derive_bootstrap_enabled(&cfg);
    assert!(enabled, "metric case must run under the enabled gate");

    let in_memory = Arc::new(InMemoryCheckpointStore::new());
    let fx = bootstrap_fixture(Arc::clone(&in_memory) as Arc<dyn CheckpointStore>, enabled).await;

    let state = seed_state(6);
    let oid = fx.create_layer("first-base-metric-layer", &state).await;

    let success_before = fx.layer_writes("success");
    let error_before = fx.layer_writes("error");

    // ── Act ──────────────────────────────────────────────────────────────
    fx.bootstrap
        .run_once(oid, state, BootstrapTarget::CreationBase)
        .await;
    fx.drain().await;

    // ── Assert ───────────────────────────────────────────────────────────
    assert_eq!(
        fx.layer_writes("success"),
        success_before + 1,
        "a successful bootstrap must tick checkpoint_writes_total{{result=\"success\", \
         kind=\"layer\"}} exactly once"
    );
    assert_eq!(
        fx.layer_writes("error"),
        error_before,
        "a successful bootstrap must NOT tick the error arm — a tick on both would make a \
         fleet-wide bootstrap-failure alert unreadable"
    );
}

// #############################################################################
// #                                                                           #
// #  TEST-B — Subtask `t3.dfh.57.kq.nq`. Cases 7-13.                          #
// #                                                                           #
// #  Everything below reuses TEST-A's `bootstrap_fixture`, `bootstrap_config` #
// #  / `derive_bootstrap_enabled` gate derivation, `seed_state*` seeds, and    #
// #  `assert_states_isomorphic` / `decode_*` comparators. Same hard fences:    #
// #  no checkpoint loop, no `refused_no_base` or tail-metric assertion, no     #
// #  `super::helpers` wait helper, no stream writes, no                        #
// #  `#![cfg(feature = ...)]` gate on the file.                               #
// #                                                                           #
// #############################################################################

// =============================================================================
// TEST-B store wrappers.
// =============================================================================

/// Wraps a real `InMemoryCheckpointStore`, COUNTING every `put_layer_checkpoint`
/// call and RECORDING its `(object_id, snap_seq, ompf)` argument triple, then
/// delegating verbatim.
///
/// Counter-and-record **before** delegating, the inverse of
/// [`SlowLayerPutStore`]'s deliberate counter-after: every TEST-B consumer of
/// this type asserts an *upper* bound (usually zero) on attempts, so the
/// question is "was the call ever issued", not "did it complete". Incrementing
/// after the delegate would let an attempt that failed, hung, or was cancelled
/// read as no attempt at all — exactly the silence a gating-matrix arm must not
/// be able to fake. Case 9 additionally needs the arguments of a PUT the store
/// never surfaces (the second of two racing writes collapses into the first
/// generation), which only a call-site recording can supply.
///
/// Every other method delegates through [`DelegatingCheckpointStore`],
/// including the four with non-trivial trait defaults — see the note at those
/// methods on [`SlowLayerPutStore`].
struct CountingLayerPutStore {
    inner: DelegatingCheckpointStore<InMemoryCheckpointStore>,
    put_calls: AtomicUsize,
    puts: Mutex<Vec<(u64, u64, Bytes)>>,
}

impl CountingLayerPutStore {
    fn new() -> Self {
        Self {
            inner: DelegatingCheckpointStore::new(Arc::new(InMemoryCheckpointStore::new())),
            put_calls: AtomicUsize::new(0),
            puts: Mutex::new(Vec::new()),
        }
    }

    fn put_calls(&self) -> usize {
        self.put_calls.load(Ordering::Acquire)
    }

    /// The `(object_id, snap_seq, ompf)` triple of every attempted layer PUT,
    /// in call order.
    fn recorded_puts(&self) -> Vec<(u64, u64, Bytes)> {
        self.puts
            .lock()
            .expect("recorded-puts mutex poisoned")
            .clone()
    }
}

#[async_trait]
impl CheckpointStore for CountingLayerPutStore {
    async fn put_layer_checkpoint(
        &self,
        object_id: u64,
        snap_seq: u64,
        ompf: Bytes,
        meta: CheckpointMeta,
    ) -> Result<CheckpointHandle, CheckpointError> {
        self.put_calls.fetch_add(1, Ordering::AcqRel);
        self.puts
            .lock()
            .expect("recorded-puts mutex poisoned")
            .push((object_id, snap_seq, ompf.clone()));
        self.inner
            .put_layer_checkpoint(object_id, snap_seq, ompf, meta)
            .await
    }

    async fn get_latest_layer_checkpoint(
        &self,
        object_id: u64,
    ) -> Result<Option<CheckpointBlob>, CheckpointError> {
        self.inner.get_latest_layer_checkpoint(object_id).await
    }

    async fn get_layer_checkpoint(
        &self,
        handle: &CheckpointHandle,
    ) -> Result<Option<CheckpointBlob>, CheckpointError> {
        self.inner.get_layer_checkpoint(handle).await
    }

    async fn list_layer_checkpoints(
        &self,
        object_id: u64,
    ) -> Result<Vec<CheckpointHandle>, CheckpointError> {
        self.inner.list_layer_checkpoints(object_id).await
    }

    async fn delete_layer_checkpoint(
        &self,
        handle: CheckpointHandle,
    ) -> Result<(), CheckpointError> {
        self.inner.delete_layer_checkpoint(handle).await
    }

    async fn put_session_checkpoint(
        &self,
        session_id: &str,
        body: Bytes,
        meta: CheckpointMeta,
    ) -> Result<CheckpointHandle, CheckpointError> {
        self.inner
            .put_session_checkpoint(session_id, body, meta)
            .await
    }

    async fn get_session_checkpoint(
        &self,
        session_id: &str,
    ) -> Result<Option<CheckpointBlob>, CheckpointError> {
        self.inner.get_session_checkpoint(session_id).await
    }

    async fn delete_session_checkpoint(&self, session_id: &str) -> Result<(), CheckpointError> {
        self.inner.delete_session_checkpoint(session_id).await
    }

    async fn list_session_checkpoint_ids(&self) -> Result<Vec<String>, CheckpointError> {
        self.inner.list_session_checkpoint_ids().await
    }

    async fn put_ext_blob(&self, hash_hex: &str, bytes: Bytes) -> Result<(), CheckpointError> {
        self.inner.put_ext_blob(hash_hex, bytes).await
    }

    async fn get_ext_blob(&self, hash_hex: &str) -> Result<Option<Bytes>, CheckpointError> {
        self.inner.get_ext_blob(hash_hex).await
    }

    async fn ext_blob_exists(&self, hash_hex: &str) -> Result<bool, CheckpointError> {
        self.inner.ext_blob_exists(hash_hex).await
    }

    async fn delete_ext_blob(&self, hash_hex: &str) -> Result<(), CheckpointError> {
        self.inner.delete_ext_blob(hash_hex).await
    }

    async fn put_counters_sidecar(&self, body: Bytes) -> Result<(), CheckpointError> {
        self.inner.put_counters_sidecar(body).await
    }

    async fn get_counters_sidecar(&self) -> Result<Option<Bytes>, CheckpointError> {
        self.inner.get_counters_sidecar().await
    }

    async fn list_all_object_ids(&self) -> Result<Vec<u64>, CheckpointError> {
        self.inner.list_all_object_ids().await
    }

    async fn list_referenced_ext_hashes(&self) -> Result<HashSet<String>, CheckpointError> {
        self.inner.list_referenced_ext_hashes().await
    }

    async fn list_live_ext_set(
        &self,
    ) -> Result<live_edit::checkpoint::LiveExtSet, CheckpointError> {
        self.inner.list_live_ext_set().await
    }

    // The four non-trivial trait defaults — forwarded for the same reason
    // spelled out on `SlowLayerPutStore`: omitting them SUBSTITUTES behaviour
    // rather than merely skipping instrumentation.
    async fn read_ext_refs_sidecar(
        &self,
        handle: &CheckpointHandle,
    ) -> Result<HashSet<String>, CheckpointError> {
        self.inner.read_ext_refs_sidecar(handle).await
    }

    async fn list_layer_artifacts(
        &self,
    ) -> Result<Vec<live_edit::checkpoint::LayerArtifact>, CheckpointError> {
        self.inner.list_layer_artifacts().await
    }

    async fn delete_layer_temp_artifacts(
        &self,
        object_id: u64,
        snap_seq: u64,
    ) -> Result<usize, CheckpointError> {
        self.inner
            .delete_layer_temp_artifacts(object_id, snap_seq)
            .await
    }

    fn publish_worst_case(&self) -> Duration {
        self.inner.publish_worst_case()
    }
}

/// Classification of the synthetic adapter failure case 8 injects. Mirrors
/// `lifecycle_session_writes_error_test.rs`'s `FailureMode`: SR-7.2's
/// stamps-nothing contract is classification-independent, so both arms run.
#[derive(Debug, Clone, Copy)]
enum FailureMode {
    Transient,
    Permanent,
}

/// Wraps a real `InMemoryCheckpointStore` and fails EVERY
/// `put_layer_checkpoint` with a classified [`CheckpointError`], delegating
/// every other method verbatim.
///
/// Hand-rolled rather than routed through the `CheckpointFaultInjector` seam,
/// for the two reasons `FailingSessionPutStore` records: the injector fires
/// `maybe_inject` on every method (an op-label filter would couple this test to
/// the in-memory adapter's call-site string), and it lives behind
/// `test-checkpoint-fault-injection` — a `#![cfg(feature = ...)]` on this file
/// would silently compile the whole bootstrap suite out of feature-free narrow
/// runs.
///
/// `put_calls` counts attempts before synthesising the error, which is what
/// makes "exactly ONE attempt, no retry" assertable: the unit documents a
/// single best-effort PUT, and a retry loop appearing here would be a design
/// regression (it would put unbounded adapter traffic on a spawned task that
/// nothing joins except graceful shutdown).
struct FailingBasePutStore {
    inner: DelegatingCheckpointStore<InMemoryCheckpointStore>,
    mode: FailureMode,
    put_calls: AtomicUsize,
}

impl FailingBasePutStore {
    fn always_failing(mode: FailureMode) -> Self {
        Self {
            inner: DelegatingCheckpointStore::new(Arc::new(InMemoryCheckpointStore::new())),
            mode,
            put_calls: AtomicUsize::new(0),
        }
    }

    fn put_calls(&self) -> usize {
        self.put_calls.load(Ordering::Acquire)
    }

    fn synthesise_error(&self) -> CheckpointError {
        match self.mode {
            FailureMode::Transient => CheckpointError::Transient {
                source: "synthetic transient layer-PUT failure for the first-base bootstrap test"
                    .into(),
            },
            FailureMode::Permanent => CheckpointError::Permanent {
                source: "synthetic permanent layer-PUT failure for the first-base bootstrap test"
                    .into(),
            },
        }
    }
}

#[async_trait]
impl CheckpointStore for FailingBasePutStore {
    async fn put_layer_checkpoint(
        &self,
        _object_id: u64,
        _snap_seq: u64,
        _ompf: Bytes,
        _meta: CheckpointMeta,
    ) -> Result<CheckpointHandle, CheckpointError> {
        self.put_calls.fetch_add(1, Ordering::AcqRel);
        Err(self.synthesise_error())
    }

    async fn get_latest_layer_checkpoint(
        &self,
        object_id: u64,
    ) -> Result<Option<CheckpointBlob>, CheckpointError> {
        self.inner.get_latest_layer_checkpoint(object_id).await
    }

    async fn get_layer_checkpoint(
        &self,
        handle: &CheckpointHandle,
    ) -> Result<Option<CheckpointBlob>, CheckpointError> {
        self.inner.get_layer_checkpoint(handle).await
    }

    async fn list_layer_checkpoints(
        &self,
        object_id: u64,
    ) -> Result<Vec<CheckpointHandle>, CheckpointError> {
        self.inner.list_layer_checkpoints(object_id).await
    }

    async fn delete_layer_checkpoint(
        &self,
        handle: CheckpointHandle,
    ) -> Result<(), CheckpointError> {
        self.inner.delete_layer_checkpoint(handle).await
    }

    async fn put_session_checkpoint(
        &self,
        session_id: &str,
        body: Bytes,
        meta: CheckpointMeta,
    ) -> Result<CheckpointHandle, CheckpointError> {
        self.inner
            .put_session_checkpoint(session_id, body, meta)
            .await
    }

    async fn get_session_checkpoint(
        &self,
        session_id: &str,
    ) -> Result<Option<CheckpointBlob>, CheckpointError> {
        self.inner.get_session_checkpoint(session_id).await
    }

    async fn delete_session_checkpoint(&self, session_id: &str) -> Result<(), CheckpointError> {
        self.inner.delete_session_checkpoint(session_id).await
    }

    async fn list_session_checkpoint_ids(&self) -> Result<Vec<String>, CheckpointError> {
        self.inner.list_session_checkpoint_ids().await
    }

    async fn put_ext_blob(&self, hash_hex: &str, bytes: Bytes) -> Result<(), CheckpointError> {
        self.inner.put_ext_blob(hash_hex, bytes).await
    }

    async fn get_ext_blob(&self, hash_hex: &str) -> Result<Option<Bytes>, CheckpointError> {
        self.inner.get_ext_blob(hash_hex).await
    }

    async fn ext_blob_exists(&self, hash_hex: &str) -> Result<bool, CheckpointError> {
        self.inner.ext_blob_exists(hash_hex).await
    }

    async fn delete_ext_blob(&self, hash_hex: &str) -> Result<(), CheckpointError> {
        self.inner.delete_ext_blob(hash_hex).await
    }

    async fn put_counters_sidecar(&self, body: Bytes) -> Result<(), CheckpointError> {
        self.inner.put_counters_sidecar(body).await
    }

    async fn get_counters_sidecar(&self) -> Result<Option<Bytes>, CheckpointError> {
        self.inner.get_counters_sidecar().await
    }

    async fn list_all_object_ids(&self) -> Result<Vec<u64>, CheckpointError> {
        self.inner.list_all_object_ids().await
    }

    async fn list_referenced_ext_hashes(&self) -> Result<HashSet<String>, CheckpointError> {
        self.inner.list_referenced_ext_hashes().await
    }

    async fn list_live_ext_set(
        &self,
    ) -> Result<live_edit::checkpoint::LiveExtSet, CheckpointError> {
        self.inner.list_live_ext_set().await
    }

    // Same non-trivial-default forwarding obligation as above.
    async fn read_ext_refs_sidecar(
        &self,
        handle: &CheckpointHandle,
    ) -> Result<HashSet<String>, CheckpointError> {
        self.inner.read_ext_refs_sidecar(handle).await
    }

    async fn list_layer_artifacts(
        &self,
    ) -> Result<Vec<live_edit::checkpoint::LayerArtifact>, CheckpointError> {
        self.inner.list_layer_artifacts().await
    }

    async fn delete_layer_temp_artifacts(
        &self,
        object_id: u64,
        snap_seq: u64,
    ) -> Result<usize, CheckpointError> {
        self.inner
            .delete_layer_temp_artifacts(object_id, snap_seq)
            .await
    }

    fn publish_worst_case(&self) -> Duration {
        self.inner.publish_worst_case()
    }
}

// =============================================================================
// Case 7 — the SR-7.1 gating matrix.
// =============================================================================

/// One off-arm of the gating matrix: build the deployment shape `cfg`
/// describes, assert the DERIVED gate is false, then assert the three
/// consequences.
///
/// The gate is derived through [`derive_bootstrap_enabled`] (the same
/// conjunction `main.rs` spells at the `BaseBootstrap::new` call site) and
/// asserted false BEFORE any consequence. A test that hard-coded `enabled =
/// false` would keep passing after production stopped gating on that conjunct;
/// deriving it means the arm fires here instead.
///
/// **Why zero PUTs after a full drain is the sound witness for "did not
/// spawn".** `TaskTracker::len() == 0` immediately after `spawn_creation_base`
/// cannot distinguish "no task" from "a task that already completed", so it is
/// deliberately not asserted. The tracker is closed and drained first, and only
/// then is the counting store — which records an attempt BEFORE delegating —
/// read. A spawned task that reached its PUT would be caught whether it
/// succeeded, failed, or hung until the drain deadline.
///
/// **Why the bootstrap-outcome family must stay flat here, and why that is not
/// the same claim the four skip arms make.** Epic 5's SR-7 adjudication
/// (Task 5 `t2.dfh.be.s8`, landed 2026-09-01) made every `run_once` invocation
/// tick exactly one `live_edit_checkpoint_base_bootstrap_total{outcome,site}`
/// cell, so "declined" arms are no longer silent — a guard skip has its own
/// outcome. The gate-off path is the one exception, and structurally so: the
/// disabled short-circuit returns in the SPAWN wrapper
/// (`base_bootstrap.rs`'s `spawn_creation_base` gate), above `run_once`
/// entirely, so there is no invocation to attribute. That "no invocation, no
/// tick" property is exactly what makes the family's sum identity — one
/// increment per `run_once`, the denominator every ratio in the adjudication
/// divides by — honest, so it is asserted here rather than left to review.
/// The whole family is compared, not one cell: a gate-off arm that ticked
/// ANY outcome would inflate the denominator, and which one it picked is
/// immaterial.
async fn assert_gate_off_arm_writes_nothing(cfg: Config, arm: &str, seed: u8, layer_name: &str) {
    // ── Arrange ──────────────────────────────────────────────────────────
    let enabled = derive_bootstrap_enabled(&cfg);
    assert!(
        !enabled,
        "{arm}: the SR-7.1 three-way gate must resolve to FALSE for this deployment shape. If \
         this fires, production stopped gating on this conjunct (or `bootstrap_config` stopped \
         producing the shape) and every consequence asserted below is vacuous"
    );

    let counting = Arc::new(CountingLayerPutStore::new());
    let fx = bootstrap_fixture(Arc::clone(&counting) as Arc<dyn CheckpointStore>, enabled).await;

    let state = seed_state(seed);
    let oid = fx.create_layer(layer_name, &state).await;

    let success_before = fx.layer_writes("success");
    let error_before = fx.layer_writes("error");
    let bootstrap_cells_before = fx.base_bootstrap_cells();
    assert_eq!(
        bootstrap_cells_before.len(),
        CHECKPOINT_BASE_BOOTSTRAP_OUTCOMES.len() * CHECKPOINT_BASE_BOOTSTRAP_SITES.len(),
        "{arm}: anti-vacuity floor — the bootstrap-outcome family must render its full zero-seeded \
         grid before the act. A reader that matched nothing would make the flatness assertion \
         below trivially true (empty == empty)"
    );

    // ── Act: the handler's entire contribution, verbatim ─────────────────
    //
    // `spawn_creation_base`, NOT `run_once`: the disabled short-circuit lives in
    // the spawn wrapper (so the handler never branches on configuration), and
    // `run_once` is the task body BELOW that gate. Driving `run_once` here
    // would bypass the very thing under test.
    fx.bootstrap.spawn_creation_base(oid, state);
    fx.drain().await;

    // ── Assert: zero attempts, no generation, no cursor columns ──────────
    assert_eq!(
        counting.put_calls(),
        0,
        "{arm}: a gated-off bootstrap must issue ZERO put_layer_checkpoint calls; got {}",
        counting.put_calls()
    );

    let handles = fx
        .store
        .list_layer_checkpoints(oid)
        .await
        .expect("list_layer_checkpoints must succeed");
    assert!(
        handles.is_empty(),
        "{arm}: a gated-off bootstrap must leave the layer with NO generations; got {handles:?}"
    );

    assert!(
        !fx.header_field_exists(oid, "last_checkpoint_seq").await,
        "{arm}: a gated-off bootstrap must not stamp last_checkpoint_seq — a cursor without a \
         generation behind it would suppress the periodic loop's first real snapshot of a layer \
         that has none"
    );
    assert!(
        !fx.header_field_exists(oid, "last_checkpoint_ms").await,
        "{arm}: a gated-off bootstrap must not stamp last_checkpoint_ms"
    );

    // Not in the Subtask's required set, but free and sharp: the disabled
    // short-circuit returns before any metric call, so neither arm of the
    // layer-write counter may move either.
    assert_eq!(
        fx.layer_writes("success"),
        success_before,
        "{arm}: a gated-off bootstrap must not tick the success arm"
    );
    assert_eq!(
        fx.layer_writes("error"),
        error_before,
        "{arm}: a gated-off bootstrap must not tick the error arm — 'checkpointing is off' is not \
         a failure, and ticking it would manufacture a fleet-wide noise floor"
    );

    // ── And the bootstrap-outcome family is flat too — every cell ────────
    assert_eq!(
        fx.base_bootstrap_cells(),
        bootstrap_cells_before,
        "{arm}: a gated-off bootstrap must leave EVERY \
         live_edit_checkpoint_base_bootstrap_total cell exactly as it found it. The gate returns \
         in the spawn wrapper, above run_once, so there is no invocation to attribute — 'no \
         invocation, no tick' is what keeps the family's one-tick-per-run_once sum identity \
         honest. A tick here (any outcome, either site) would inflate the denominator every ratio \
         in the SR-7 adjudication divides by, with a population that never ran"
    );
}

/// `LIVE_EDIT_CHECKPOINT_TAIL_ENABLED=false`, all else on.
///
/// The conjunct that distinguishes this unit's gate from
/// `SessionCheckpointWriter`'s (`!Disabled` alone): the bootstrap exists to
/// serve the tail path, so it is off whenever the tail path is.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn bootstrap_does_not_spawn_when_tail_capture_is_disabled() {
    assert_gate_off_arm_writes_nothing(
        bootstrap_config(true, "InMemory", false),
        "LIVE_EDIT_CHECKPOINT_TAIL_ENABLED=false",
        20,
        "first-base-gate-tail-off-layer",
    )
    .await;
}

/// `LIVE_EDIT_CHECKPOINT_ADAPTER=Disabled`, all else on.
///
/// The three-way gate makes the `NoopCheckpointStore`'s `NotImplemented` return
/// unreachable from this unit — the PUT is never issued in the first place.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn bootstrap_does_not_spawn_when_the_adapter_is_disabled() {
    assert_gate_off_arm_writes_nothing(
        bootstrap_config(true, "Disabled", true),
        "LIVE_EDIT_CHECKPOINT_ADAPTER=Disabled",
        21,
        "first-base-gate-adapter-disabled-layer",
    )
    .await;
}

/// `LIVE_EDIT_CHECKPOINT_ENABLED=false`, all else on.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn bootstrap_does_not_spawn_when_checkpointing_is_disabled() {
    assert_gate_off_arm_writes_nothing(
        bootstrap_config(false, "InMemory", true),
        "LIVE_EDIT_CHECKPOINT_ENABLED=false",
        22,
        "first-base-gate-checkpointing-off-layer",
    )
    .await;
}

/// The CONTROL arm of the gating matrix: all three inputs on, driven through
/// the SAME `spawn_creation_base` entry point the three off-arms use.
///
/// Without it the matrix is unfalsifiable — a fixture broken in any way that
/// stops the bootstrap from running (a wrong tracker, a store never installed,
/// a layer never materialized) would make all three off-arms pass for the wrong
/// reason. Case 1 proves the success path through `run_once`; this proves the
/// *gate wrapper* lets it through.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn bootstrap_spawns_and_lands_a_base_when_all_three_gates_are_on() {
    // ── Arrange ──────────────────────────────────────────────────────────
    let cfg = all_gates_on_config();
    let enabled = derive_bootstrap_enabled(&cfg);
    assert!(
        enabled,
        "the all-gates-on Config must resolve the SR-7.1 three-way gate to TRUE — otherwise the \
         three off-arms above are comparing against nothing"
    );

    let counting = Arc::new(CountingLayerPutStore::new());
    let fx = bootstrap_fixture(Arc::clone(&counting) as Arc<dyn CheckpointStore>, enabled).await;

    let state = seed_state(23);
    let oid = fx
        .create_layer("first-base-gate-control-layer", &state)
        .await;

    // ── Act ──────────────────────────────────────────────────────────────
    fx.bootstrap.spawn_creation_base(oid, state);
    fx.drain().await;

    // ── Assert: the mirror image of every off-arm assertion ──────────────
    assert_eq!(
        counting.put_calls(),
        1,
        "the enabled gate must let exactly ONE put_layer_checkpoint through"
    );

    let handles = fx
        .store
        .list_layer_checkpoints(oid)
        .await
        .expect("list_layer_checkpoints must succeed");
    assert_eq!(
        handles.len(),
        1,
        "the control arm must land exactly one generation; got {handles:?}"
    );
    assert_eq!(handles[0].snap_seq, BOOTSTRAP_SNAP_SEQ);
    assert_eq!(handles[0].kind, CheckpointKind::Base);

    assert_eq!(
        fx.header_field(oid, "last_checkpoint_seq").await.as_deref(),
        Some("1"),
        "the control arm must stamp last_checkpoint_seq"
    );
    assert!(
        fx.header_field_exists(oid, "last_checkpoint_ms").await,
        "the control arm must stamp last_checkpoint_ms"
    );
}

// =============================================================================
// Case 8 — the PUT-failure arm (SR-7.2 stamps-nothing).
// =============================================================================

/// SR-7.2: when the single best-effort base PUT fails, the bootstrap stamps
/// NOTHING and leaves no generation — and the CreateLayer RPC is unaffected.
///
/// Run for BOTH error classifications by the two wrappers below, because the
/// stamps-nothing contract is classification-independent: `run_once` logs
/// `is_transient` / `is_permanent` for the operator but takes the identical
/// terminal arm either way, and a future "retry the transient case" change must
/// break this test rather than pass it.
///
/// **The `refused_no_base` continuation belongs to Task 7 (`t2.dfh.57.1u`),
/// not here.** The Task-level phrasing "the layer's subsequent capture attempt
/// surfaces `refused_no_base`" needs the tail-capture path plus the periodic
/// loop, both out of this Task's scope. What this case owns is the
/// *precondition* that makes that downstream refusal inevitable: no generation
/// on the store and no cursor column on the header, so nothing downstream can
/// mistake this layer for one with a lineage root.
async fn assert_put_failure_stamps_nothing(mode: FailureMode, seed: u8, layer_name: &str) {
    // ── Arrange ──────────────────────────────────────────────────────────
    let cfg = all_gates_on_config();
    let enabled = derive_bootstrap_enabled(&cfg);
    assert!(
        enabled,
        "the PUT-failure arm must run under the ENABLED gate — under a disabled gate the absence \
         of a generation would prove nothing about the failure path"
    );

    let failing = Arc::new(FailingBasePutStore::always_failing(mode));
    let fx = bootstrap_fixture(Arc::clone(&failing) as Arc<dyn CheckpointStore>, enabled).await;

    let state = seed_state(seed);
    let sec_a = u64::from(seed) * 10 + 1;
    let sec_b = u64::from(seed) * 10 + 2;

    // `create_layer` must succeed on its own terms — the bootstrap is a
    // post-commit task and cannot fail the RPC. `BootstrapFixture::create_layer`
    // `expect`s the Ok, so reaching this line at all is the "create_layer
    // returned Ok" half of the assertion; the materialization checks below are
    // the other half.
    let oid = fx.create_layer(layer_name, &state).await;

    let success_before = fx.layer_writes("success");
    let error_before = fx.layer_writes("error");

    // ── Act ──────────────────────────────────────────────────────────────
    fx.bootstrap
        .run_once(oid, state, BootstrapTarget::CreationBase)
        .await;
    fx.drain().await;

    // ── Assert: exactly ONE attempt, no retry ────────────────────────────
    assert_eq!(
        failing.put_calls(),
        1,
        "{mode:?}: the bootstrap makes exactly ONE best-effort put_layer_checkpoint attempt. A \
         count above 1 means a retry loop appeared — a design regression: the task is joined only \
         by graceful shutdown, so retries here put unbounded adapter traffic inside the shutdown \
         budget, and the documented recovery is the next eligible periodic tick"
    );

    // ── Assert: no generation ────────────────────────────────────────────
    let handles = fx
        .store
        .list_layer_checkpoints(oid)
        .await
        .expect("list_layer_checkpoints must succeed");
    assert!(
        handles.is_empty(),
        "{mode:?}: a failed PUT must leave NO generation; got {handles:?}"
    );

    // ── Assert: all THREE cursor columns absent ──────────────────────────
    //
    // `restore_seq_bump` is in the set deliberately even though the success
    // path never writes it either: the sharp regression this guards against is
    // a stamp block that moved ABOVE the PUT's error return, which would write
    // whatever set of columns it stamps regardless of the PUT's outcome.
    for column in [
        "last_checkpoint_seq",
        "last_checkpoint_ms",
        "restore_seq_bump",
    ] {
        assert!(
            !fx.header_field_exists(oid, column).await,
            "{mode:?}: a failed PUT must leave obj:hdr `{column}` ABSENT. A cursor stamped over a \
             base that never landed is strictly worse than no base at all — it suppresses the \
             periodic loop's skip-if-no-commits gate for a layer with no lineage root"
        );
    }

    // ── Assert: the RPC's own work survived intact ───────────────────────
    assert_eq!(
        fx.header_field(oid, "seq").await.as_deref(),
        Some("1"),
        "{mode:?}: CreateLayer's commit must be untouched by a bootstrap failure — the layer is \
         live and serving at sequence 1"
    );
    let sections_key = fx.tv.key_schema().obj_sections_index(&oid.to_string());
    let section_members: HashSet<String> = fx
        .tv
        .client()
        .pool
        .smembers(&sections_key)
        .await
        .expect("SMEMBERS over the sections index must succeed");
    let expected_sections: HashSet<String> =
        [sec_a.to_string(), sec_b.to_string()].into_iter().collect();
    assert_eq!(
        section_members, expected_sections,
        "{mode:?}: the layer must be fully materialized — a bootstrap failure must never roll back \
         or corrupt the committed creation state"
    );

    // ── Assert: the operator-visible signal ──────────────────────────────
    assert_eq!(
        fx.layer_writes("error"),
        error_before + 1,
        "{mode:?}: a failed base PUT must tick checkpoint_writes_total{{result=\"error\", \
         kind=\"layer\"}} exactly once — it is what makes a fleet-wide bootstrap failure visible \
         immediately rather than only later, per-layer, as a refused capture"
    );
    assert_eq!(
        fx.layer_writes("success"),
        success_before,
        "{mode:?}: a failed base PUT must not tick the success arm"
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn bootstrap_transient_put_failure_stamps_no_cursor_columns_and_leaves_no_generation() {
    assert_put_failure_stamps_nothing(
        FailureMode::Transient,
        30,
        "first-base-put-failure-transient-layer",
    )
    .await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn bootstrap_permanent_put_failure_stamps_no_cursor_columns_and_leaves_no_generation() {
    assert_put_failure_stamps_nothing(
        FailureMode::Permanent,
        31,
        "first-base-put-failure-permanent-layer",
    )
    .await;
}

// =============================================================================
// Case 9 — duplicate run: single-generation collapse.
// =============================================================================

/// Two CONCURRENT `run_once` calls against the same `(object_id, state)`
/// collapse to ONE generation carrying equivalent content, and leave the cursor
/// at exactly 1.
///
/// **Why `run_once` twice rather than two `create_layer` calls.** A second
/// `create_layer` takes the overwrite branch and mints a NEW `object_id`
/// (`allocate_object_id`, `src/storage/layer.rs`), so the two PUT keys would
/// differ and no duplicate would exist to observe — that is case 10, a
/// different property. `run_once` is a `pub async fn` precisely so this race is
/// drivable.
///
/// **Why the concurrency is deterministic enough to assert TWO attempts.**
/// `tokio::join!` drives both futures from one task: the first future is polled
/// until its first `await` (the `obj:hdr` identity read, a network round trip
/// that cannot complete on the first poll), then the second is polled to the
/// same point. Both identity reads are therefore in flight before either
/// resolves, so both observe `last_checkpoint_seq` absent, and neither can take
/// the step-1 early-abort. If this ever does collapse to one attempt, the
/// assertion below names that as the cause rather than failing obscurely.
///
/// **What this proves about the unit's deliberate simplicity.** The sibling
/// `SessionCheckpointWriter` carries a per-session serialized dispatcher, a
/// coalescing slot map, and a Valkey-side publish watermark, because its PUT is
/// keyed per session and its payload changes with every lifecycle event, so a
/// racing loser durably clobbers the winner. This unit needs none of that: the
/// key is `(object_id, 1, Base)` and the payload is idempotent-identical, so
/// last-writer-wins is harmless BY CONSTRUCTION. That is the claim under test —
/// no read-modify-write, no dedup, no per-object serialization is required, and
/// adding any of it would be pure cost.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn duplicate_bootstrap_run_collapses_to_one_generation_with_equivalent_content() {
    // ── Arrange ──────────────────────────────────────────────────────────
    let cfg = all_gates_on_config();
    let enabled = derive_bootstrap_enabled(&cfg);
    assert!(
        enabled,
        "duplicate-run case must run under the enabled gate"
    );

    let counting = Arc::new(CountingLayerPutStore::new());
    let fx = bootstrap_fixture(Arc::clone(&counting) as Arc<dyn CheckpointStore>, enabled).await;

    let state = seed_state(40);
    let oid = fx
        .create_layer("first-base-duplicate-run-layer", &state)
        .await;

    // ── Act: both futures in flight before either resolves ───────────────
    let first = fx
        .bootstrap
        .run_once(oid, state.clone(), BootstrapTarget::CreationBase);
    let second = fx
        .bootstrap
        .run_once(oid, state.clone(), BootstrapTarget::CreationBase);
    tokio::join!(first, second);
    fx.drain().await;

    // ── Assert: both really raced ────────────────────────────────────────
    assert_eq!(
        counting.put_calls(),
        2,
        "both concurrent runs must reach the PUT for the collapse to be under test. A 1 here means \
         one run took the step-1 early-abort because the other had already stamped the cursor — \
         the runs serialized instead of racing, and the assertions below would be vacuous"
    );

    // ── Assert: the adapter collapsed them to ONE generation ─────────────
    //
    // The `CheckpointStore` conformance contract (case #9): a repeat PUT at the
    // same `(object_id, snap_seq, kind)` overwrites in place rather than
    // appending a second generation.
    let handles = fx
        .store
        .list_layer_checkpoints(oid)
        .await
        .expect("list_layer_checkpoints must succeed");
    assert_eq!(
        handles.len(),
        1,
        "two bootstrap runs at the same (object_id, snap_seq, kind) must collapse to ONE \
         generation; got {handles:?}"
    );
    assert_eq!(handles[0].snap_seq, BOOTSTRAP_SNAP_SEQ);
    assert_eq!(handles[0].kind, CheckpointKind::Base);

    // ── Assert: the two payloads are interchangeable ─────────────────────
    let puts = counting.recorded_puts();
    assert_eq!(puts.len(), 2, "expected both PUT argument triples recorded");
    for (object_id, snap_seq, _) in &puts {
        assert_eq!(*object_id, oid, "both PUTs must target the same object");
        assert_eq!(
            *snap_seq, BOOTSTRAP_SNAP_SEQ,
            "both PUTs must target the same snap_seq — the key that makes the collapse safe"
        );
    }

    // Semantic equivalence is the LOAD-BEARING claim: it is what makes
    // last-writer-wins harmless whichever write lands last.
    let first_state = decode_reset_ompf(&puts[0].2);
    let second_state = decode_reset_ompf(&puts[1].2);
    assert_states_isomorphic(
        &first_state,
        &second_state,
        "the two racing bootstrap payloads",
    );

    // Byte equality is asserted too, and after b.gcb / b.vz7 it holds for a
    // reason that GENERALIZES: `DiffEngine::plan_full_content` emits a total
    // order over the CONTENT of `ObjectState` — fields on `(version,
    // authoring_set_order, section_id, key_name)`, time samples on `(version,
    // section_id, time)`, orphan sections on `section_id` — so no `HashMap`
    // table layout reaches the wire and equal state yields equal bytes however
    // each map was constructed. The clone-shaped fixture here is therefore no
    // longer what carries the assertion; case 2 proves the same equality across
    // two structurally different builders fed independently-constructed maps.
    // If this ever fails it is a defect in that sort, not a fixture quirk:
    // decode both payloads, identify which vector diverged, and fix the order.
    assert_eq!(
        puts[0].2, puts[1].2,
        "two runs over the same ObjectState must build byte-identical OMPF: emission order is a \
         total order over the state's CONTENT, independent of any map's table layout. A failure \
         here is a defect in DiffEngine::plan_full_content's ordering — see this assertion's \
         comment before relaxing it"
    );

    // ── Assert: the cursor is stamped once, not doubled ──────────────────
    assert_eq!(
        fx.header_field(oid, "last_checkpoint_seq").await.as_deref(),
        Some("1"),
        "last_checkpoint_seq must read exactly the persisted generation's snap_seq. The stamp is a \
         plain HSET of a literal, not an increment — a `2` here would mean something started \
         accumulating"
    );
    assert!(
        fx.header_field_exists(oid, "last_checkpoint_ms").await,
        "last_checkpoint_ms must be stamped"
    );
    assert!(
        !fx.header_field_exists(oid, "restore_seq_bump").await,
        "neither run may write restore_seq_bump"
    );
}

// =============================================================================
// Case 10 — the CreateLayer overwrite branch.
// =============================================================================

/// A CreateLayer OVERWRITE roots a lineage for its NEW `object_id`, and leaves
/// the old object's generation alone.
///
/// Both CreateLayer branches spawn by construction and neither needs
/// special-casing: `create_layer` returns the NEW `object_id` on the
/// fresh-create and the overwrite path alike, and the new object's own header
/// stamps `seq = 1`, so keying on the returned id at `snap_seq = 1` is correct
/// either way. This case pins that — the handler's single unconditional
/// `spawn_creation_base(returned_oid, state)` is the whole contract.
///
/// **The bounded residual, stated rather than asserted.** The old object's base
/// at `(old_object_id, 1)` is now unreachable — nothing will ever restore it,
/// because the layer mapping points at the new id. That is a checkpoint-store
/// ORPHAN, and the §22.17.5 reconciliation sweep is the existing owner of that
/// whole class; this Task introduces no new cleanup path and this case
/// deliberately asserts the orphan is still present rather than expecting
/// anything to have removed it.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn bootstrap_also_roots_a_lineage_for_a_create_layer_overwrite() {
    // ── Arrange ──────────────────────────────────────────────────────────
    let cfg = all_gates_on_config();
    let enabled = derive_bootstrap_enabled(&cfg);
    assert!(enabled, "overwrite case must run under the enabled gate");

    let counting = Arc::new(CountingLayerPutStore::new());
    let fx = bootstrap_fixture(Arc::clone(&counting) as Arc<dyn CheckpointStore>, enabled).await;

    // ── Act 1: fresh create + its bootstrap ──────────────────────────────
    let layer_name = "first-base-overwrite-layer";
    let first_state = seed_state(50);
    let old_oid = fx.create_layer(layer_name, &first_state).await;
    fx.bootstrap
        .run_once(old_oid, first_state, BootstrapTarget::CreationBase)
        .await;

    let old_handles = fx
        .store
        .list_layer_checkpoints(old_oid)
        .await
        .expect("list_layer_checkpoints must succeed");
    assert_eq!(
        old_handles.len(),
        1,
        "precondition: the first create's bootstrap must have landed a generation, otherwise the \
         untouched-old-generation assertion below is vacuous"
    );
    let old_blob_before = fx
        .store
        .get_layer_checkpoint(&old_handles[0])
        .await
        .expect("get_layer_checkpoint must succeed")
        .expect("the old object's generation must be readable")
        .bytes;

    // ── Act 2: overwrite the SAME layer name + its bootstrap ─────────────
    //
    // Different seed so the two objects' payloads are distinguishable — a
    // regression that re-persisted the OLD state under the new id would
    // otherwise pass the shape assertions below.
    let second_state = seed_state(51);
    let new_oid = fx.create_layer(layer_name, &second_state).await;
    assert_ne!(
        new_oid, old_oid,
        "the overwrite branch allocates a NEW object_id — if these are equal the fixture did not \
         take the overwrite path and this case is testing a plain second create"
    );
    fx.bootstrap
        .run_once(new_oid, second_state.clone(), BootstrapTarget::CreationBase)
        .await;
    fx.drain().await;

    // ── Assert: the NEW object has its own base + both cursors ───────────
    let new_handles = fx
        .store
        .list_layer_checkpoints(new_oid)
        .await
        .expect("list_layer_checkpoints must succeed");
    assert_eq!(
        new_handles.len(),
        1,
        "the overwrite's new object must get its OWN base; got {new_handles:?}"
    );
    assert_eq!(
        new_handles[0].snap_seq, BOOTSTRAP_SNAP_SEQ,
        "the overwrite's base lands at snap_seq = 1 — the new object's own header stamps seq = 1, \
         so the fresh-create reasoning carries over verbatim"
    );
    assert_eq!(new_handles[0].kind, CheckpointKind::Base);

    assert_eq!(
        fx.header_field(new_oid, "last_checkpoint_seq")
            .await
            .as_deref(),
        Some("1"),
        "the overwrite's new object must have last_checkpoint_seq stamped"
    );
    let new_ms: u64 = fx
        .header_field(new_oid, "last_checkpoint_ms")
        .await
        .expect("last_checkpoint_ms must be present on the overwrite's new object")
        .parse()
        .expect("last_checkpoint_ms must parse as u64");
    assert!(
        new_ms > 0,
        "last_checkpoint_ms must be a real wall-clock stamp"
    );

    // The new base must carry the NEW state, not a re-persisted old one.
    let new_blob = fx
        .store
        .get_layer_checkpoint(&new_handles[0])
        .await
        .expect("get_layer_checkpoint must succeed")
        .expect("the new object's generation must be readable");
    let (decoded_new, _) = decode_blob(&new_blob);
    assert_states_isomorphic(
        &second_state,
        &decoded_new,
        "the overwrite's base vs the state its CreateLayer committed",
    );

    // ── Assert: the OLD object's generation is untouched ─────────────────
    //
    // Store-side only, deliberately: the overwrite schedules asynchronous
    // cleanup of the OLD object's Valkey keys, so the old header is expected to
    // disappear and asserting on it would be racing that cleanup. What must
    // hold is that the second bootstrap wrote nothing under the old id and
    // removed nothing.
    let old_handles_after = fx
        .store
        .list_layer_checkpoints(old_oid)
        .await
        .expect("list_layer_checkpoints must succeed");
    assert_eq!(
        old_handles_after, old_handles,
        "the overwrite's bootstrap must neither add to nor remove from the OLD object's lineage — \
         it is keyed on the new object_id and has no reason to touch the old one"
    );
    let old_blob_after = fx
        .store
        .get_layer_checkpoint(&old_handles[0])
        .await
        .expect("get_layer_checkpoint must succeed")
        .expect(
            "the old object's generation must still be present — it is an ORPHAN owned by the \
             §22.17.5 reconciliation sweep, and nothing in this Task collects it",
        )
        .bytes;
    assert_eq!(
        old_blob_after, old_blob_before,
        "the old object's persisted bytes must be unchanged by the overwrite's bootstrap"
    );

    assert_eq!(
        counting.put_calls(),
        2,
        "exactly one PUT per CreateLayer commit — one for the fresh create, one for the overwrite"
    );
}

// =============================================================================
// Case 11 — the cursor-regression guard.
// =============================================================================

/// A generation has already landed for this object: the bootstrap declines
/// entirely and leaves BOTH cursor columns exactly as it found them.
///
/// Drives the step-1 early-abort arm (`identity.last_checkpoint_seq != 0`) by
/// pre-stamping a cursor a later generation would have written. Stamping `1`
/// back over it would be strictly harmful, and in two independent ways the unit
/// documents: it un-suppresses the periodic loop's skip-if-no-commits gate, and
/// it makes the next assembly measure `observed_commit_span` from 1 instead of
/// from the generation that actually landed — inflating the span, which widens
/// the DR restore guard band.
///
/// The PUT-count assertion is not redundant with the column assertions: it
/// distinguishes "the guard fired BEFORE the PUT" (the landed design) from "the
/// guard fired only at the pre-stamp re-read", which would leave the columns
/// correct while still depositing a stale base at `snap_seq = 1` underneath a
/// newer generation.
///
/// **The metric reading, as Epic 5's SR-7 adjudication landed it (Task 5
/// `t2.dfh.be.s8`, 2026-09-01).** This arm is no longer silent: it ticks
/// `live_edit_checkpoint_base_bootstrap_total{outcome="skipped_generation_present",
/// site="creation_base"}`, the pre-PUT half of the pair whose post-PUT sibling
/// is `unstamped_generation_present`. It still moves NO `checkpoint_writes_total`
/// cell, and that is a deliberate split rather than an omission: that family
/// counts generation-COMPLETION outcomes fleet-wide, and a guard that declines
/// to complete a generation is neither a success nor an error there. The two
/// `layer_writes` assertions below are the fence on exactly that, unedited
/// across the adjudication. The positive cell assertion is NOT made here — it
/// belongs to the SR-7 lane's own bootstrap-observability module, so that the
/// vocabulary lives in one place instead of drifting across two files.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn bootstrap_does_not_regress_a_cursor_a_later_generation_already_advanced() {
    // ── Arrange ──────────────────────────────────────────────────────────
    let cfg = all_gates_on_config();
    let enabled = derive_bootstrap_enabled(&cfg);
    assert!(
        enabled,
        "cursor-regression case must run under the enabled gate"
    );

    let counting = Arc::new(CountingLayerPutStore::new());
    let fx = bootstrap_fixture(Arc::clone(&counting) as Arc<dyn CheckpointStore>, enabled).await;

    let state = seed_state(60);
    let oid = fx
        .create_layer("first-base-cursor-regression-layer", &state)
        .await;

    // Pre-stamp the header as a later generation would have left it. The `ms`
    // value is a fixed, obviously-synthetic instant so an overwrite by the
    // bootstrap's `now_unix_ms()` is unmistakable rather than a near-miss.
    const ADVANCED_SEQ: &str = "500";
    const ADVANCED_MS: &str = "1700000000000";
    let header_key = fx.tv.key_schema().obj_header(&oid.to_string());
    let _: u64 = fx
        .tv
        .client()
        .pool
        .hset(
            &header_key,
            vec![
                ("last_checkpoint_seq", ADVANCED_SEQ),
                ("last_checkpoint_ms", ADVANCED_MS),
            ],
        )
        .await
        .expect("pre-stamping the advanced cursor must succeed");

    let success_before = fx.layer_writes("success");
    let error_before = fx.layer_writes("error");

    // ── Act ──────────────────────────────────────────────────────────────
    fx.bootstrap
        .run_once(oid, state, BootstrapTarget::CreationBase)
        .await;
    fx.drain().await;

    // ── Assert: both columns UNCHANGED ───────────────────────────────────
    assert_eq!(
        fx.header_field(oid, "last_checkpoint_seq").await.as_deref(),
        Some(ADVANCED_SEQ),
        "last_checkpoint_seq must be left at the advanced value. Regressing it to 1 un-suppresses \
         the loop's skip-if-no-commits gate AND makes the next assembly measure its commit span \
         from 1 rather than from the generation that actually landed, widening the DR restore \
         guard band"
    );
    assert_eq!(
        fx.header_field(oid, "last_checkpoint_ms").await.as_deref(),
        Some(ADVANCED_MS),
        "last_checkpoint_ms must be left at the advanced value — the two columns ride one HSET, so \
         a moved `ms` with a preserved `seq` would mean the stamp block ran partially"
    );

    // ── Assert: the guard fired BEFORE the PUT ───────────────────────────
    assert_eq!(
        counting.put_calls(),
        0,
        "the early-abort must fire on the identity read, before any adapter work. A 1 here means \
         the guard moved to the pre-stamp re-read only, which would leave a stale base at \
         snap_seq = 1 sitting underneath a newer generation"
    );
    let handles = fx
        .store
        .list_layer_checkpoints(oid)
        .await
        .expect("list_layer_checkpoints must succeed");
    assert!(
        handles.is_empty(),
        "no generation may be written for an object whose lineage already has a root; got \
         {handles:?}"
    );

    // The skip is a routine decline, not a fault: same posture as the
    // vanished-header arm — no PUT, no stamp, and no `writes_total` cell. Both
    // arms DO tick their own bootstrap-outcome cell
    // (`skipped_generation_present` here, `skipped_header_absent` there); what
    // the two assertions below pin is that neither reaches the
    // generation-completion family.
    assert_eq!(
        fx.layer_writes("success"),
        success_before,
        "a declined bootstrap must not tick the success arm"
    );
    assert_eq!(
        fx.layer_writes("error"),
        error_before,
        "a declined bootstrap must not tick the error arm — an already-rooted lineage is the \
         desired state, not a failure"
    );
}

// =============================================================================
// Case 12 — empty-content CreateLayer.
// =============================================================================

/// An empty-content `CreateLayer` gets a lineage root like any other layer.
///
/// `handle_create_layer` applies `apply_delta(&mut state, None)` to a fresh
/// `ObjectState` when the uploaded OMPF is empty (SDD §8.4 — a supported RPC
/// shape, not a degenerate one), commits that state at sequence 1, and spawns
/// the bootstrap unconditionally. The state reproduced here is exactly that.
///
/// **Why an empty BASE is legitimate although a zero-record TAIL is
/// forbidden — they are different artifact classes.** A tail is a *delta*
/// envelope: it claims "these commits happened past the base", so a tail
/// carrying zero records claims nothing and is pure cost on the lineage, which
/// is why capture refuses to write one. A base is a *full-state snapshot*: it
/// claims "this is the entire content at this sequence", and for an
/// empty-content layer the entire content genuinely IS empty. Refusing to write
/// it would leave precisely the population SR-7.1 exists to cover — a layer with
/// no lineage root, for which tail capture then refuses `no_base` forever —
/// and would do so for a layer that is perfectly legal to create.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn bootstrap_roots_a_lineage_for_an_empty_content_create_layer() {
    // ── Arrange ──────────────────────────────────────────────────────────
    let cfg = all_gates_on_config();
    let enabled = derive_bootstrap_enabled(&cfg);
    assert!(
        enabled,
        "empty-content case must run under the enabled gate"
    );

    let counting = Arc::new(CountingLayerPutStore::new());
    let fx = bootstrap_fixture(Arc::clone(&counting) as Arc<dyn CheckpointStore>, enabled).await;

    // The `src/grpc/layer.rs` empty-content shape, verbatim.
    let mut state = ObjectState::default();
    apply_delta(&mut state, None).expect("apply_delta(None) on a fresh state must succeed");
    assert_eq!(
        state.sequence_num, BOOTSTRAP_SNAP_SEQ,
        "precondition: the empty-content path still takes a version increment, which is what makes \
         the committed sequence 1 by construction"
    );
    assert!(
        state.sections.is_empty() && state.fields.is_empty() && state.time_samples.is_empty(),
        "precondition: the empty-content creation state carries no content"
    );

    let oid = fx
        .create_layer("first-base-empty-content-layer", &state)
        .await;

    // ── Act ──────────────────────────────────────────────────────────────
    fx.bootstrap
        .run_once(oid, state, BootstrapTarget::CreationBase)
        .await;
    fx.drain().await;

    // ── Assert: one Base at snap_seq = 1, both cursors stamped ───────────
    let handles = fx
        .store
        .list_layer_checkpoints(oid)
        .await
        .expect("list_layer_checkpoints must succeed");
    assert_eq!(
        handles.len(),
        1,
        "an empty-content layer must get exactly one base — declining here would leave it with no \
         lineage root at all; got {handles:?}"
    );
    assert_eq!(handles[0].snap_seq, BOOTSTRAP_SNAP_SEQ);
    assert_eq!(handles[0].kind, CheckpointKind::Base);

    assert_eq!(
        fx.header_field(oid, "last_checkpoint_seq").await.as_deref(),
        Some("1"),
        "last_checkpoint_seq must be stamped for an empty-content layer"
    );
    let ms_col: u64 = fx
        .header_field(oid, "last_checkpoint_ms")
        .await
        .expect("last_checkpoint_ms must be present")
        .parse()
        .expect("last_checkpoint_ms must parse as u64");
    assert!(
        ms_col > 0,
        "last_checkpoint_ms must be a real wall-clock stamp"
    );

    // ── Assert: the sidecar is Some-but-EMPTY, not absent and not an error ─
    //
    // `sidecar_bytes` is ALWAYS `Some` — matching the assembler — which is what
    // makes the adapter's publish-order obligation apply to bootstrap bases too.
    // An empty hash LIST inside a present sidecar is the correct encoding of
    // "this generation references no external values"; a missing sidecar would
    // instead read as "this generation predates sidecars".
    let blob = fx
        .store
        .get_layer_checkpoint(&handles[0])
        .await
        .expect("get_layer_checkpoint must succeed")
        .expect("the empty-content base must be readable");
    assert!(
        blob.meta.sidecar_bytes.is_some(),
        "the bootstrap must persist a sidecar even when the hash list is empty"
    );
    let sidecar = fx.store.read_ext_refs_sidecar(&handles[0]).await.expect(
        "read_ext_refs_sidecar must return Ok on an empty sidecar — an error here would fail \
             the retention cascade and the DR ext-blob recovery path for every empty-content layer",
    );
    assert!(
        sidecar.is_empty(),
        "an empty-content layer references no external values; got {sidecar:?}"
    );

    // ── Assert: the blob decodes AND applies cleanly ─────────────────────
    //
    // The sharp part is that this is a real reset delta, not zero bytes: a DR
    // restore replays it onto a fresh `ObjectState` and must end with an empty
    // layer rather than a parse error.
    let (restored, ompf_bytes) = decode_blob(&blob);
    assert!(
        !ompf_bytes.is_empty(),
        "the empty-content base must still carry a well-formed OMPF container with a reset delta \
         inside it — an empty byte string is not a snapshot"
    );
    assert!(
        restored.sections.is_empty()
            && restored.fields.is_empty()
            && restored.time_samples.is_empty(),
        "replaying the empty-content base onto a fresh ObjectState must reconstruct an empty layer"
    );
}

// =============================================================================
// Case 13 — the vanished-header race.
// =============================================================================

/// `obj:hdr` deleted between the commit and the bootstrap's identity read: the
/// unit writes nothing, resurrects nothing, and completes no generation.
///
/// The layer was deleted or overwritten inside the bootstrap window — a routine
/// race, not a fault, so it mirrors the assembler's `NotFound` posture. That is
/// why neither `checkpoint_writes_total{result="success"}` nor `{result="error"}`
/// may move: an `error` tick here would put a permanent alert floor under every
/// normal create-then-delete workload, and a `success` tick would claim a
/// generation that does not exist.
///
/// **What it DOES tick, since Epic 5's SR-7 adjudication (Task 5
/// `t2.dfh.be.s8`, 2026-09-01).** The arm is not silent any more — it ticks
/// `live_edit_checkpoint_base_bootstrap_total{outcome="skipped_header_absent",
/// site="creation_base"}`, one cell, on the family whose sum is one increment
/// per `run_once`. Hence the rename: the old name's "ticks no metric" claim is
/// falsified by the landed reading, while every assertion under it is still
/// exactly right, because all of them are about the OTHER family. The operator
/// story is that a rising `skipped_header_absent` reads as create-then-delete
/// churn rather than as checkpoint damage, and it is a normal, non-alertable
/// population. The positive cell is asserted in the SR-7 lane's own
/// bootstrap-observability module, not duplicated here.
///
/// The `EXISTS == 0` assertion is the sharp one. Before b.vn6 the unit's cursor
/// stamp was a bare, unguarded `HSET`, so an `HSET` reached on this path would
/// RESURRECT `obj:hdr` as a skeletal hash holding two cursor columns and no
/// `seq`. Asserting the key is still absent is what proves the early return
/// fired upstream of it. Since b.vn6 the stamp is itself conditional
/// (`stamp_cursor_if_live.lua`), so the assertion is now belt AND braces — this
/// case pins the PRE-PUT early return, and case 14 below pins the POST-PUT
/// atomic refusal for the window this arm cannot see.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn bootstrap_on_a_vanished_header_writes_nothing_and_ticks_only_the_bootstrap_outcome_cell() {
    // ── Arrange ──────────────────────────────────────────────────────────
    let cfg = all_gates_on_config();
    let enabled = derive_bootstrap_enabled(&cfg);
    assert!(
        enabled,
        "vanished-header case must run under the enabled gate — under a disabled gate the absence \
         of writes would prove nothing about this arm"
    );

    let counting = Arc::new(CountingLayerPutStore::new());
    let fx = bootstrap_fixture(Arc::clone(&counting) as Arc<dyn CheckpointStore>, enabled).await;

    let state = seed_state(70);
    let oid = fx
        .create_layer("first-base-vanished-header-layer", &state)
        .await;

    // Simulate the delete / overwrite that landed inside the bootstrap window.
    let header_key = fx.tv.key_schema().obj_header(&oid.to_string());
    let _: u64 = fx
        .tv
        .client()
        .pool
        .unlink(vec![header_key.clone()])
        .await
        .expect("UNLINK obj:hdr must succeed");
    assert!(
        !fx.tv
            .client()
            .pool
            .exists::<bool, _>(&header_key)
            .await
            .expect("EXISTS obj:hdr must succeed"),
        "precondition: obj:hdr must actually be gone before the bootstrap runs"
    );

    let success_before = fx.layer_writes("success");
    let error_before = fx.layer_writes("error");

    // ── Act ──────────────────────────────────────────────────────────────
    fx.bootstrap
        .run_once(oid, state, BootstrapTarget::CreationBase)
        .await;
    fx.drain().await;

    // ── Assert: nothing written anywhere ─────────────────────────────────
    assert_eq!(
        counting.put_calls(),
        0,
        "a vanished header must abort before any adapter work — a base for a layer that no longer \
         exists is an orphan created on purpose"
    );
    let handles = fx
        .store
        .list_layer_checkpoints(oid)
        .await
        .expect("list_layer_checkpoints must succeed");
    assert!(
        handles.is_empty(),
        "no generation may exist for a vanished layer; got {handles:?}"
    );

    assert!(
        !fx.tv
            .client()
            .pool
            .exists::<bool, _>(&header_key)
            .await
            .expect("EXISTS obj:hdr must succeed"),
        "obj:hdr must still be ABSENT. The cursor stamp is an unguarded HSET, so reaching it on \
         this path would resurrect the key as a skeletal hash carrying two cursor columns and no \
         `seq` — which the assembler then drops as NotFound and the commit paths fail closed on"
    );

    // ── Assert: no generation-completion cell moves ──────────────────────
    assert_eq!(
        fx.layer_writes("success"),
        success_before,
        "a vanished header is not a successful write"
    );
    assert_eq!(
        fx.layer_writes("error"),
        error_before,
        "a vanished header is not a FAILURE either — it is a routine delete/overwrite race, and \
         ticking `error` here would put a permanent alert floor under every normal \
         create-then-delete workload"
    );
}

// =============================================================================
// Case 13b — b.hka review round 2: the identity-empty header reaches case 13's
// arm.
// =============================================================================

/// An `obj:hdr` that EXISTS, carries `seq`, and has lost its `session_id` column
/// is refused at step 1 — same outcome, same posture, same zero writes as the
/// vanished header above.
///
/// # Why this producer needed a fix at all
///
/// The refusal is not local to this unit and never was: `read_header_identity`
/// answers `Ok(None)` for a degenerate header since b.hka review round 2, so this
/// unit's pre-existing `Ok(None)` arm covers it with no new branch. Before that
/// the loader folded the missing column to `String::new()` and this unit had NO
/// emptiness check of its own — only the periodic assembler carried one — so the
/// bootstrap built the reset OMPF, spent the PUT, and handed a meta with empty
/// identity to `validate_generation_identity`, which refuses it with a
/// `Permanent`. That is a `writes_total{result="error"}` tick on every attempt
/// for that layer, from a perfectly healthy adapter, forever.
///
/// So the assertions are the two halves of what the fix bought: the outcome cell
/// is the benign `skipped_header_absent` rather than an error, and the adapter is
/// never called. `error_before` is the sharp one — pre-fix it moved.
///
/// # Why `SkippedHeaderAbsent` is right for a header that is not absent
///
/// A layer with no identity cannot be restored: the apply path cannot rebuild
/// `obj:hdr` without inventing session metadata, and inventing it is how a
/// restore lands a layer under the wrong session. For DR purposes the object does
/// not exist, so reporting it as one is the honest reading — and it keeps the
/// vocabulary of this metric family to the four outcomes an operator already
/// knows rather than minting a fifth for a state that is not routine enough to
/// need its own dashboard row. The WARN naming the degenerate columns is emitted
/// once, at the shared read, which is where an operator can act on it.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn bootstrap_on_an_identity_empty_header_skips_before_the_put_and_writes_no_artifact() {
    // ── Arrange ──────────────────────────────────────────────────────────
    let cfg = all_gates_on_config();
    let enabled = derive_bootstrap_enabled(&cfg);
    assert!(
        enabled,
        "the refusal must be observed under the ENABLED gate — under a disabled gate the unit \
         never runs and `put_calls == 0` would prove nothing"
    );

    let counting = Arc::new(CountingLayerPutStore::new());
    let fx = bootstrap_fixture(Arc::clone(&counting) as Arc<dyn CheckpointStore>, enabled).await;

    let state = seed_state(74);
    let oid = fx
        .create_layer("hka-identity-empty-header-layer", &state)
        .await;

    // Strip ONLY `session_id` — the partially-written / hand-repaired header,
    // which is the only way a live layer's `obj:hdr` can carry `seq` and lack an
    // identity column.
    let header_key = fx.tv.key_schema().obj_header(&oid.to_string());
    let removed: u64 = fx
        .tv
        .client()
        .pool
        .hdel(&header_key, "session_id")
        .await
        .expect("HDEL obj:hdr session_id must succeed");
    assert_eq!(
        removed, 1,
        "precondition: the column must have been there to remove — if `obj:hdr` stopped carrying \
         `session_id` at all this case would pass vacuously"
    );
    assert!(
        fx.header_field_exists(oid, "seq").await,
        "precondition: `seq` must SURVIVE. A header without it is case 16's seq-less phantom, \
         which refuses at a different step, and a DELETEd key is case 13 — this case is about \
         neither"
    );
    assert!(
        fx.header_field_exists(oid, "layer_name").await,
        "precondition: and `layer_name` must survive too, so exactly ONE degenerate column is \
         under test"
    );

    let success_before = fx.layer_writes("success");
    let error_before = fx.layer_writes("error");

    // ── Act ──────────────────────────────────────────────────────────────
    let outcome = fx
        .bootstrap
        .run_once(oid, state, BootstrapTarget::CreationBase)
        .await;
    fx.drain().await;

    // ── Assert ───────────────────────────────────────────────────────────
    assert_eq!(
        outcome,
        BootstrapOutcome::SkippedHeaderAbsent,
        "a layer with no identity cannot be restored, so for DR purposes it does not exist and \
         the step-1 absent-header arm is the honest report. `Failed` here would be the pre-fix \
         shape: the PUT attempted and refused by `validate_generation_identity`"
    );
    assert_eq!(
        counting.put_calls(),
        0,
        "THE claim: the adapter is never called. Pre-b.hka the unit built the reset OMPF and PUT \
         it, and the adapter refused with a `Permanent` — a wasted multi-MB round trip per \
         attempt AND an adapter-health error tick from a healthy adapter"
    );
    let handles = fx
        .store
        .list_layer_checkpoints(oid)
        .await
        .expect("list_layer_checkpoints must succeed");
    assert!(
        handles.is_empty(),
        "no generation for a layer nothing can restore; got {handles:?}"
    );
    assert!(
        !fx.header_field_exists(oid, "last_checkpoint_seq").await,
        "and no cursor column is stamped: claiming durable coverage for a layer with no \
         restorable generation is exactly the phantom b.vn6 closed at the other arm"
    );

    assert_eq!(
        fx.layer_writes("success"),
        success_before,
        "nothing landed, so nothing may be counted as a successful write"
    );
    assert_eq!(
        fx.layer_writes("error"),
        error_before,
        "and — the sharp one — nothing may be counted as an ERROR either. Pre-fix this cell moved \
         on every attempt for such a layer, from a healthy adapter, which is what put a permanent \
         floor under the adapter-health alert"
    );
}

// #############################################################################
// #                                                                           #
// #  b.vn6 — cases 14-16. The half of the delete race case 13 CANNOT see.     #
// #                                                                           #
// #  Case 13 kills the header BEFORE `run_once` starts, so step 1's identity  #
// #  read finds nothing and the unit returns before touching the adapter.     #
// #  The incident this ticket was filed for is the other window: the header   #
// #  is alive at step 1, the PUT lands, and the layer dies while it is in     #
// #  flight — after which the (then bare) `HSET` cursor stamp CREATED a       #
// #  cursor-only `obj:hdr` phantom and the base blob was left in the store    #
// #  covering a layer that no longer existed. Eight of those in one dev run.  #
// #                                                                           #
// #  These three cases pin the three answers: the atomic post-PUT refusal     #
// #  plus its fast-path reap (14), the reap failing without the refusal       #
// #  degrading (15), and the cheap pre-PUT re-check that keeps a slow build   #
// #  from writing the blob at all (16).                                       #
// #                                                                           #
// #############################################################################

/// Wraps a real `InMemoryCheckpointStore` and KILLS the layer's `obj:hdr` from
/// inside `put_layer_checkpoint`, after the generation has landed and
/// immediately before the call returns `Ok(handle)`.
///
/// # Why a store wrapper and not the fault-injection seam
///
/// `CheckpointFaultInjector::maybe_inject` is synchronous, so it cannot issue a
/// Valkey command — it can only substitute an error. The window this models is
/// not an adapter failure at all: the PUT *succeeds*, and what changes
/// underneath it is Valkey. Only a wrapper that can await a `UNLINK` between the
/// delegate's return and its own can place the delete inside that window, and
/// placing it there deterministically — rather than racing a spawned deleter —
/// is what keeps these cases off the flaky list. Same hand-rolled-wrapper
/// posture as [`SlowLayerPutStore`] and [`FailingBasePutStore`], and the same
/// reason this file stays feature-ungated.
///
/// # Wiring
///
/// The Valkey handle only exists once [`bootstrap_fixture`] has minted its
/// `TestValkey`, which is after this store must already be constructed — hence
/// the [`OnceLock`] rather than a constructor argument. [`Self::arm`] installs
/// it; an un-armed store is inert and delegates verbatim, which is what makes a
/// forgotten `arm()` a loud `expect` rather than a silently-passing case.
///
/// Every other method delegates through [`DelegatingCheckpointStore`], including
/// the four with non-trivial trait defaults — `read_ext_refs_sidecar` above all,
/// since the fast-path reap READS it and an inherited default returning an empty
/// set would make the tombstone assertions vacuously green.
struct HeaderKillingPutStore {
    inner: DelegatingCheckpointStore<InMemoryCheckpointStore>,
    put_calls: AtomicUsize,
    delete_calls: AtomicUsize,
    /// When set, every `delete_layer_checkpoint` fails BEFORE delegating, so
    /// the generation survives — case 15's fast-path-reap failure.
    fail_deletes: bool,
    /// When set, every `read_ext_refs_sidecar` fails — case 17's first arm.
    ///
    /// A separate flag from [`Self::fail_deletes`] rather than a shared "fail
    /// everything" switch, because the two arms make OPPOSITE claims about the
    /// same counter: case 15 says the delete was attempted exactly once after
    /// the record landed, case 17 says it was never attempted at all. One flag
    /// could not express both.
    fail_sidecar_reads: bool,
    /// `(client, header_key)`, installed by [`Self::arm`].
    kill: std::sync::OnceLock<(ValkeyClient, String)>,
}

impl HeaderKillingPutStore {
    fn new() -> Self {
        Self {
            inner: DelegatingCheckpointStore::new(Arc::new(InMemoryCheckpointStore::new())),
            put_calls: AtomicUsize::new(0),
            delete_calls: AtomicUsize::new(0),
            fail_deletes: false,
            fail_sidecar_reads: false,
            kill: std::sync::OnceLock::new(),
        }
    }

    /// Same, but every generation delete fails — the arm that proves the
    /// refusal does not depend on the reap succeeding.
    fn new_with_failing_deletes() -> Self {
        Self {
            fail_deletes: true,
            ..Self::new()
        }
    }

    /// Same, but every ext-refs sidecar read fails — the arm that proves the
    /// reap ABORTS before its delete rather than proceeding without the hashes.
    fn new_with_failing_sidecar_reads() -> Self {
        Self {
            fail_sidecar_reads: true,
            ..Self::new()
        }
    }

    /// Point the wrapper at the header it must kill. Called after the fixture
    /// exists and after the layer has been created, so `object_id` is known.
    fn arm(&self, tv: &TestValkey, object_id: u64) {
        self.kill
            .set((
                tv.client().clone(),
                tv.key_schema().obj_header(&object_id.to_string()),
            ))
            .map_err(|_| ())
            .expect("HeaderKillingPutStore::arm must be called exactly once");
    }

    fn put_calls(&self) -> usize {
        self.put_calls.load(Ordering::Acquire)
    }

    fn delete_calls(&self) -> usize {
        self.delete_calls.load(Ordering::Acquire)
    }
}

#[async_trait]
impl CheckpointStore for HeaderKillingPutStore {
    async fn put_layer_checkpoint(
        &self,
        object_id: u64,
        snap_seq: u64,
        ompf: Bytes,
        meta: CheckpointMeta,
    ) -> Result<CheckpointHandle, CheckpointError> {
        self.put_calls.fetch_add(1, Ordering::AcqRel);
        let handle = self
            .inner
            .put_layer_checkpoint(object_id, snap_seq, ompf, meta)
            .await?;

        // The delete lands HERE: the generation is durable on the adapter and
        // the caller has not yet reached its cursor stamp. That ordering is the
        // whole point — a kill before the delegate would model case 13's
        // pre-PUT window instead, and a kill after the return would race the
        // stamp rather than precede it.
        let (client, header_key) = self
            .kill
            .get()
            .expect("HeaderKillingPutStore must be armed before the bootstrap runs");
        let _: u64 = client
            .pool
            .unlink(vec![header_key.clone()])
            .await
            .expect("UNLINK obj:hdr from inside the PUT must succeed");

        Ok(handle)
    }

    async fn delete_layer_checkpoint(
        &self,
        handle: CheckpointHandle,
    ) -> Result<(), CheckpointError> {
        self.delete_calls.fetch_add(1, Ordering::AcqRel);
        if self.fail_deletes {
            return Err(CheckpointError::Transient {
                source: "injected generation-delete failure \
                         (HeaderKillingPutStore::new_with_failing_deletes)"
                    .into(),
            });
        }
        self.inner.delete_layer_checkpoint(handle).await
    }

    async fn get_latest_layer_checkpoint(
        &self,
        object_id: u64,
    ) -> Result<Option<CheckpointBlob>, CheckpointError> {
        self.inner.get_latest_layer_checkpoint(object_id).await
    }

    async fn get_layer_checkpoint(
        &self,
        handle: &CheckpointHandle,
    ) -> Result<Option<CheckpointBlob>, CheckpointError> {
        self.inner.get_layer_checkpoint(handle).await
    }

    async fn list_layer_checkpoints(
        &self,
        object_id: u64,
    ) -> Result<Vec<CheckpointHandle>, CheckpointError> {
        self.inner.list_layer_checkpoints(object_id).await
    }

    async fn put_session_checkpoint(
        &self,
        session_id: &str,
        body: Bytes,
        meta: CheckpointMeta,
    ) -> Result<CheckpointHandle, CheckpointError> {
        self.inner
            .put_session_checkpoint(session_id, body, meta)
            .await
    }

    async fn get_session_checkpoint(
        &self,
        session_id: &str,
    ) -> Result<Option<CheckpointBlob>, CheckpointError> {
        self.inner.get_session_checkpoint(session_id).await
    }

    async fn delete_session_checkpoint(&self, session_id: &str) -> Result<(), CheckpointError> {
        self.inner.delete_session_checkpoint(session_id).await
    }

    async fn list_session_checkpoint_ids(&self) -> Result<Vec<String>, CheckpointError> {
        self.inner.list_session_checkpoint_ids().await
    }

    async fn put_ext_blob(&self, hash_hex: &str, bytes: Bytes) -> Result<(), CheckpointError> {
        self.inner.put_ext_blob(hash_hex, bytes).await
    }

    async fn get_ext_blob(&self, hash_hex: &str) -> Result<Option<Bytes>, CheckpointError> {
        self.inner.get_ext_blob(hash_hex).await
    }

    async fn ext_blob_exists(&self, hash_hex: &str) -> Result<bool, CheckpointError> {
        self.inner.ext_blob_exists(hash_hex).await
    }

    async fn delete_ext_blob(&self, hash_hex: &str) -> Result<(), CheckpointError> {
        self.inner.delete_ext_blob(hash_hex).await
    }

    async fn put_counters_sidecar(&self, body: Bytes) -> Result<(), CheckpointError> {
        self.inner.put_counters_sidecar(body).await
    }

    async fn get_counters_sidecar(&self) -> Result<Option<Bytes>, CheckpointError> {
        self.inner.get_counters_sidecar().await
    }

    async fn list_all_object_ids(&self) -> Result<Vec<u64>, CheckpointError> {
        self.inner.list_all_object_ids().await
    }

    async fn list_referenced_ext_hashes(&self) -> Result<HashSet<String>, CheckpointError> {
        self.inner.list_referenced_ext_hashes().await
    }

    async fn list_live_ext_set(
        &self,
    ) -> Result<live_edit::checkpoint::LiveExtSet, CheckpointError> {
        self.inner.list_live_ext_set().await
    }

    // The four non-trivial trait defaults — forwarded for the reason spelled
    // out on `SlowLayerPutStore`: omitting them SUBSTITUTES behaviour rather
    // than merely skipping instrumentation.
    async fn read_ext_refs_sidecar(
        &self,
        handle: &CheckpointHandle,
    ) -> Result<HashSet<String>, CheckpointError> {
        if self.fail_sidecar_reads {
            return Err(CheckpointError::Transient {
                source: "injected ext-refs sidecar read failure \
                         (HeaderKillingPutStore::new_with_failing_sidecar_reads)"
                    .into(),
            });
        }
        self.inner.read_ext_refs_sidecar(handle).await
    }

    async fn list_layer_artifacts(
        &self,
    ) -> Result<Vec<live_edit::checkpoint::LayerArtifact>, CheckpointError> {
        self.inner.list_layer_artifacts().await
    }

    async fn delete_layer_temp_artifacts(
        &self,
        object_id: u64,
        snap_seq: u64,
    ) -> Result<usize, CheckpointError> {
        self.inner
            .delete_layer_temp_artifacts(object_id, snap_seq)
            .await
    }

    fn publish_worst_case(&self) -> Duration {
        self.inner.publish_worst_case()
    }
}

/// Every member of `le:ckpt:{recon}:tomb`, sorted — the ext-blob tombstone
/// carrier the fast-path reap must record into BEFORE it deletes anything.
async fn tombstoned_hashes(fx: &BootstrapFixture) -> Vec<String> {
    let key = fx.tv.key_schema().checkpoint_reconcile_tombstones();
    let mut members: Vec<String> = fx
        .tv
        .client()
        .pool
        .zrange(&key, 0, -1, None, false, None, false)
        .await
        .expect("ZRANGE over le:ckpt:{recon}:tomb must succeed");
    members.sort();
    members
}

// =============================================================================
// Case 14 — b.vn6: the header dies while the base PUT is in flight.
// =============================================================================

/// The layer is deleted between the bootstrap's identity read and its cursor
/// stamp: the stamp is REFUSED atomically, the header is not resurrected, and
/// the base blob written for the now-dead layer is reaped inline.
///
/// # Why this fails without the fix
///
/// The stamp used to be a bare `HSET` on `obj:hdr`. `HSET` on a key that does
/// not exist CREATES it — so on this exact path the pre-fix unit wrote a hash
/// carrying `last_checkpoint_seq` and `last_checkpoint_ms` and *nothing else*:
/// no `seq`, no `created_by`, none of the twelve fields `create_layer` writes.
/// That phantom then made the checkpoint loop WARN "missing/unparseable obj:hdr
/// seq" on every tick and `diff_read` default fields, for the ~5 minutes until
/// the Valkey orphan sweep reaped it, and the base triplet it had just written
/// stayed in the store forever. Run this case against the pre-fix unit and the
/// `EXISTS obj:hdr == false` assertion below goes red on a resurrected key,
/// with `list_layer_checkpoints` red immediately after it on the leaked base.
///
/// `skipped_header_absent` cannot cover any of it, which is the other half of
/// the incident: that cell read a flat zero throughout, because the header was
/// still alive when the unit looked at it and died while the PUT was in flight.
/// Hence the tenth outcome — a POST-PUT cell, `unstamped_header_absent`, whose
/// distinguishing feature is that a durable blob exists behind it.
///
/// # What the reap owes, and why the order is asserted through the ZSET
///
/// `ext-refs.json` is the only durable record of which ext blobs a generation
/// references, so deleting the generation before recording those hashes strands
/// every one of them permanently. The layer therefore carries a real external
/// value ([`seed_state_with_ext`]) and the case asserts the hash is on
/// `le:ckpt:{recon}:tomb` *after* a reap that also deleted the generation — the
/// two together are only jointly satisfiable in the sidecar-first order.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn bootstrap_refuses_the_stamp_and_reaps_the_base_when_the_header_dies_during_the_put() {
    // ── Arrange ──────────────────────────────────────────────────────────
    let cfg = all_gates_on_config();
    let enabled = derive_bootstrap_enabled(&cfg);
    assert!(
        enabled,
        "the post-PUT delete race must run under the enabled gate — under a disabled gate the \
         unit never reaches the PUT and every assertion below would be vacuous"
    );

    let killer = Arc::new(HeaderKillingPutStore::new());
    let fx = bootstrap_fixture(Arc::clone(&killer) as Arc<dyn CheckpointStore>, enabled).await;

    let payload = b"vn6-post-put-race-ext-payload";
    let hash = sha256(payload);
    let hash_hex = sha256_to_hex(&hash);
    fx.ext_values
        .store_values(&[(&hash, payload.as_slice())])
        .await
        .expect("ext store_values must succeed");

    let state = seed_state_with_ext(71, hash, payload.len() as u64);
    let oid = fx.create_layer("vn6-post-put-race-layer", &state).await;
    killer.arm(&fx.tv, oid);

    let header_key = fx.tv.key_schema().obj_header(&oid.to_string());
    let success_before = fx.layer_writes("success");
    let error_before = fx.layer_writes("error");
    let cells_before = fx.base_bootstrap_cells();

    // ── Act ──────────────────────────────────────────────────────────────
    let outcome = fx
        .bootstrap
        .run_once(oid, state, BootstrapTarget::CreationBase)
        .await;
    fx.drain().await;

    // ── Assert: the arm that ran ─────────────────────────────────────────
    assert_eq!(
        outcome,
        BootstrapOutcome::UnstampedHeaderAbsent,
        "the PUT succeeded and the header died under it, so this is the POST-PUT refusal — not \
         `SkippedHeaderAbsent` (pre-PUT, adapter untouched, nothing to reclaim) and not `Failed` \
         (nothing failed; the layer stopped existing)"
    );
    assert_eq!(
        killer.put_calls(),
        1,
        "exactly one PUT: the race is only reachable through a PUT that landed, and a retry here \
         would be an unbounded adapter loop on a task nothing joins but graceful shutdown"
    );

    // ── Assert: no phantom ───────────────────────────────────────────────
    assert!(
        !fx.tv
            .client()
            .pool
            .exists::<bool, _>(&header_key)
            .await
            .expect("EXISTS obj:hdr must succeed"),
        "THE b.vn6 assertion. `obj:hdr` must still be ABSENT: the pre-fix bare `HSET` CREATED it \
         here, and what it created was a cursor-only phantom with no `seq` that the checkpoint \
         loop WARNed about every tick and `diff_read` defaulted fields around until the orphan \
         sweep reaped it minutes later"
    );

    // ── Assert: the blob was reaped, sidecar-first ───────────────────────
    assert_eq!(
        killer.delete_calls(),
        1,
        "the fast-path reap must delete the generation it just wrote — exactly one delete, for \
         the just-written handle and nothing else (older generations belong to the reconciliation \
         pass, which holds the lease this path does not)"
    );
    let handles = fx
        .store
        .list_layer_checkpoints(oid)
        .await
        .expect("list_layer_checkpoints must succeed");
    assert!(
        handles.is_empty(),
        "no generation may survive for a layer that no longer exists — leaving it is the leaked \
         base triplet the ticket reported, one per create-then-delete race, forever; got \
         {handles:?}"
    );
    assert_eq!(
        tombstoned_hashes(&fx).await,
        vec![hash_hex.clone()],
        "and its ext hashes must be on le:ckpt:{{recon}}:tomb. Ordering is the claim: the \
         ext-refs sidecar is the ONLY durable record of which blobs this generation named, so a \
         reap that deleted first would strand {hash_hex} permanently — nothing else would ever \
         know to look for it"
    );

    // ── Assert: the metric split ─────────────────────────────────────────
    assert_eq!(
        fx.layer_writes("success"),
        success_before,
        "no generation was COMPLETED — the blob exists for a moment and is then reclaimed — so \
         `success` would be a false claim of durable coverage"
    );
    assert_eq!(
        fx.layer_writes("error"),
        error_before,
        "and nothing FAILED either. Ticking `error` on a routine create-then-delete race is what \
         would put a permanent alert floor under every churny client (smoke tests, CI, editors \
         with scratch layers) — the same reason case 13 refuses it"
    );
    let cells_after = fx.base_bootstrap_cells();
    let cell = (
        "unstamped_header_absent".to_string(),
        "creation_base".to_string(),
    );
    let seeded = |cells: &HashMap<(String, String), f64>| -> f64 {
        cells.get(&cell).copied().unwrap_or_else(|| {
            panic!(
                "{cell:?} must be a ZERO-SEEDED series on this family — the whole grid is \
                 materialised at Metrics::new so that an arm that never fired reads a real 0 \
                 rather than an absent line. An absent key here means the outcome is missing from \
                 CHECKPOINT_BASE_BOOTSTRAP_OUTCOMES or from the seeding loop"
            )
        })
    };
    assert_eq!(
        seeded(&cells_after) - seeded(&cells_before),
        1.0,
        "the post-PUT refusal owns its own cell. Folding it onto `skipped_header_absent` would \
         erase the distinction the incident turned on: that cell is pre-PUT and leaves nothing \
         behind, this one has a durable blob to reclaim"
    );
}

// =============================================================================
// Case 15 — b.vn6: the fast-path reap fails.
// =============================================================================

/// Same race, but the generation delete fails: the refusal is unchanged and the
/// base is left intact for the reconciliation pass rather than half-reclaimed.
///
/// # The claim
///
/// The fast-path reap is a latency optimisation, not the correctness path. The
/// delete path that killed the layer registered its `object_id` on
/// `le:ckpt:{recon}:dead` before returning (pinned by
/// `reconcile_dead_object_test.rs`), and the reconciliation pass's
/// dead-object phase reaps every generation of every member (pinned by
/// `checkpointer.rs`'s `dead_object_*` cases). So the property this case owes is
/// that a failed reap degrades to exactly that backstop and to nothing worse:
/// same outcome, same absent header, no partial state.
///
/// The sharp assertion is the pairing at the end: the hashes are tombstoned even
/// though the generation SURVIVES. That is deliberate and it is the safe
/// direction — the collect phase re-reads the live set and spares a hash the
/// surviving sidecar still references, so recording early costs nothing, while
/// recording late (or not at all) would strand the blobs if the next pass's
/// delete succeeded. It is also the only observation that proves the reap ran
/// the sidecar step BEFORE the delete rather than skipping straight to it.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn bootstrap_still_refuses_the_stamp_when_the_fast_path_reap_fails() {
    // ── Arrange ──────────────────────────────────────────────────────────
    let cfg = all_gates_on_config();
    let enabled = derive_bootstrap_enabled(&cfg);
    assert!(
        enabled,
        "the failed-reap case must run under the enabled gate"
    );

    let killer = Arc::new(HeaderKillingPutStore::new_with_failing_deletes());
    let fx = bootstrap_fixture(Arc::clone(&killer) as Arc<dyn CheckpointStore>, enabled).await;

    let payload = b"vn6-failed-reap-ext-payload";
    let hash = sha256(payload);
    let hash_hex = sha256_to_hex(&hash);
    fx.ext_values
        .store_values(&[(&hash, payload.as_slice())])
        .await
        .expect("ext store_values must succeed");

    let state = seed_state_with_ext(72, hash, payload.len() as u64);
    let oid = fx.create_layer("vn6-failed-reap-layer", &state).await;
    killer.arm(&fx.tv, oid);

    let header_key = fx.tv.key_schema().obj_header(&oid.to_string());
    let success_before = fx.layer_writes("success");
    let error_before = fx.layer_writes("error");

    // ── Act ──────────────────────────────────────────────────────────────
    let outcome = fx
        .bootstrap
        .run_once(oid, state, BootstrapTarget::CreationBase)
        .await;
    fx.drain().await;

    // ── Assert ───────────────────────────────────────────────────────────
    assert_eq!(
        outcome,
        BootstrapOutcome::UnstampedHeaderAbsent,
        "the reap is best-effort and its failure must not change the arm — the refusal already \
         happened, atomically, and it is what the outcome names"
    );
    assert!(
        !fx.tv
            .client()
            .pool
            .exists::<bool, _>(&header_key)
            .await
            .expect("EXISTS obj:hdr must succeed"),
        "the header stays absent whatever the adapter does: the refusal is a Valkey-side \
         conditional write and has no dependency on the reap"
    );
    assert_eq!(
        killer.delete_calls(),
        1,
        "the reap was attempted exactly once and then gave up — retrying here would spend an \
         unbounded amount of a spawned task's life on work the reconciliation pass owns"
    );

    let handles = fx
        .store
        .list_layer_checkpoints(oid)
        .await
        .expect("list_layer_checkpoints must succeed");
    assert_eq!(
        handles.len(),
        1,
        "the generation SURVIVES a failed reap, and that is correct rather than tolerated: the \
         object is on le:ckpt:{{recon}}:dead, so the reconciliation pass's dead-object phase \
         reclaims it under the leader lease. Losing track of it here would be the leak; deleting \
         it half-way would be worse"
    );
    assert_eq!(
        tombstoned_hashes(&fx).await,
        vec![hash_hex.clone()],
        "and the hashes are recorded even though the artifact survives — the record-before-delete \
         order, observed at the one moment the two steps disagree. Recording early is the safe \
         direction (the collect phase re-reads the live set and spares a still-referenced hash); \
         recording late would strand {hash_hex} the moment a later delete succeeded"
    );

    assert_eq!(
        fx.layer_writes("success"),
        success_before,
        "still no completed generation"
    );
    assert_eq!(
        fx.layer_writes("error"),
        error_before,
        "and still not an error arm. The adapter DID fail here, but what it failed at is \
         reclamation, not the write this family counts — folding it in would make \
         `checkpoint_writes_total{{error,layer}}` unreadable as a write-failure signal"
    );
}

// =============================================================================
// Case 16 — b.vn6: the pre-PUT liveness re-check.
// =============================================================================

/// A header that survives step 1's identity read but carries no `seq` is
/// refused BEFORE the PUT, so no artifact is written at all.
///
/// # What this pins, and why the fixture is a seq-less header rather than a race
///
/// The unit makes no adapter call between step 1's `HGETALL obj:hdr` and the
/// PUT — the OMPF build and the sidecar serialization in between are pure — so
/// there is no seam a store wrapper could use to delete the header inside that
/// window, and racing a spawned deleter against it would be exactly the timing
/// flake this suite refuses to write.
///
/// The seq-less header reaches the same arm deterministically, and it is a real
/// state rather than a contrivance: it is precisely the cursor-only phantom the
/// pre-fix bare `HSET` left in the dev keyspace. `read_header_identity` admits
/// this header — it refuses only an ABSENT one or one whose `session_id` /
/// `layer_name` is degenerate (b.hka review round 2), and here both identity
/// columns are intact while every u64 column folds a missing value to 0 — so
/// step 1 passes it through and the cursor-regression guard does not fire. The
/// pre-PUT `HEXISTS seq` probe is then the only thing standing between it and a
/// base blob written for a layer that does not exist. That the probe is a
/// `HEXISTS ... "seq"` rather than an `EXISTS` is what makes the fix
/// self-healing over phantoms already sitting in a deployed keyspace.
///
/// The identity-degenerate header is the OTHER shape of a partially-written
/// header and refuses one step earlier, at step 1 — see
/// `bootstrap_on_an_identity_empty_header_skips_before_the_put_and_writes_no_artifact`
/// beside case 13.
///
/// The `EXISTS == true` assertion is what separates this arm from case 13's:
/// there the key is gone and step 1 returns early, here the key is present and
/// the later probe is the only possible refusal point.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn bootstrap_on_a_seq_less_header_skips_before_the_put_and_writes_no_artifact() {
    // ── Arrange ──────────────────────────────────────────────────────────
    let cfg = all_gates_on_config();
    let enabled = derive_bootstrap_enabled(&cfg);
    assert!(
        enabled,
        "the pre-PUT re-check must run under the enabled gate — under a disabled gate the unit \
         never runs and `put_calls == 0` would prove nothing"
    );

    let counting = Arc::new(CountingLayerPutStore::new());
    let fx = bootstrap_fixture(Arc::clone(&counting) as Arc<dyn CheckpointStore>, enabled).await;

    let state = seed_state(73);
    let oid = fx.create_layer("vn6-seq-less-header-layer", &state).await;

    // Strip ONLY `seq`. What is left is a non-empty header — identity columns
    // intact, no cursor columns — so step 1 reads it happily and the
    // cursor-regression guard sees `last_checkpoint_seq = 0`.
    let header_key = fx.tv.key_schema().obj_header(&oid.to_string());
    let removed: u64 = fx
        .tv
        .client()
        .pool
        .hdel(&header_key, "seq")
        .await
        .expect("HDEL obj:hdr seq must succeed");
    assert_eq!(
        removed, 1,
        "precondition: `seq` must have been present and removed — a header that never carried it \
         would make this case about a different state"
    );
    assert!(
        fx.header_field_exists(oid, "layer_name").await,
        "precondition: the header must still EXIST and be non-empty, or step 1's identity read \
         would return None and this case would be case 13 with extra steps"
    );

    let success_before = fx.layer_writes("success");
    let error_before = fx.layer_writes("error");

    // ── Act ──────────────────────────────────────────────────────────────
    let outcome = fx
        .bootstrap
        .run_once(oid, state, BootstrapTarget::CreationBase)
        .await;
    fx.drain().await;

    // ── Assert ───────────────────────────────────────────────────────────
    assert_eq!(
        outcome,
        BootstrapOutcome::SkippedHeaderAbsent,
        "`skipped_header_absent` is the right cell precisely because this refusal is PRE-PUT and \
         leaves the adapter untouched. The post-PUT refusal is a different cell \
         (`unstamped_header_absent`, case 14) for that exact reason — one has a durable blob \
         behind it and the other does not"
    );
    assert_eq!(
        counting.put_calls(),
        0,
        "nothing may be written: converting a would-be durable-garbage outcome into a clean \
         pre-PUT skip is the entire value of this probe"
    );
    let handles = fx
        .store
        .list_layer_checkpoints(oid)
        .await
        .expect("list_layer_checkpoints must succeed");
    assert!(
        handles.is_empty(),
        "no generation for a layer whose header is not live; got {handles:?}"
    );
    assert!(
        !fx.header_field_exists(oid, "last_checkpoint_seq").await,
        "and the phantom is not made worse. Stamping a cursor onto it would claim durable \
         coverage for a layer nothing can reconstruct — the self-healing direction is to leave it \
         alone until the Valkey orphan sweep reaps it"
    );

    assert_eq!(fx.layer_writes("success"), success_before);
    assert_eq!(
        fx.layer_writes("error"),
        error_before,
        "an advisory probe that declined is not a write failure"
    );
}

// =============================================================================
// Case 17 — b.vn6: the reap ABORTS before its delete when the pre-delete
// obligation fails.
// =============================================================================

/// Which step of the fast-path reap's record-before-delete sequence fails.
///
/// Both arms have the same remedy — leave the base standing and let the
/// reconciliation pass's dead-object phase reclaim it — and the same
/// prohibition, which is the one this case exists for: **the delete must not be
/// attempted at all.**
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ReapStep {
    /// `read_ext_refs_sidecar` fails: the reap does not know which blobs the
    /// base names, so it may not delete the only artifact that says.
    SidecarRead,
    /// The read succeeded and the tombstone write failed: the hashes are known
    /// and *unrecorded*, which leaves the system in the same position as not
    /// knowing them.
    TombstoneRecord,
}

/// The fast-path reap deletes NOTHING when a step before the delete fails.
///
/// # Why this is a separate case from 15
///
/// Case 15 fails the delete itself, so it can only observe what happens
/// *after* the obligation was discharged. This one fails the obligation, and
/// the assertion is the one case 15 structurally cannot make:
/// `delete_calls() == 0`. The ordering inside `reap_dead_base` is
/// `read_ext_refs_sidecar` → `ExtTombstoneSink::record` →
/// `delete_layer_checkpoint`, and each guard arm must `return false` **without
/// deleting**. A reap that deleted anyway would strand every ext blob the base
/// named, permanently: `ext-refs.json` is the only durable record of which
/// blobs a generation references, it is deleted with the generation, and no
/// later pass could enumerate what it used to name.
///
/// The layer therefore carries a real external value, so the strand is a real
/// one rather than a hypothetical over an empty hash set.
///
/// Both arms are driven through the real path. The sidecar read fails inside
/// the store wrapper — the same trait method `reap_dead_base` calls. The
/// tombstone write fails because `le:ckpt:{recon}:tomb` is poisoned with a
/// Valkey string, so the sink's script comes back `WRONGTYPE` from `fred`; the
/// same technique the reconciliation pass's twin
/// (`reap_leaves_the_orphan_in_place_when_a_record_before_delete_step_fails`)
/// uses, which is what makes the two files' claims comparable.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn bootstrap_reap_deletes_nothing_when_a_step_before_the_delete_fails() {
    for step in [ReapStep::SidecarRead, ReapStep::TombstoneRecord] {
        let enabled = derive_bootstrap_enabled(&all_gates_on_config());
        assert!(enabled, "{step:?}: the reap arms need the enabled gate");

        let killer = Arc::new(match step {
            ReapStep::SidecarRead => HeaderKillingPutStore::new_with_failing_sidecar_reads(),
            ReapStep::TombstoneRecord => HeaderKillingPutStore::new(),
        });
        let fx = bootstrap_fixture(Arc::clone(&killer) as Arc<dyn CheckpointStore>, enabled).await;

        let payload = b"vn6-abort-before-delete-payload";
        let hash = sha256(payload);
        fx.ext_values
            .store_values(&[(&hash, payload.as_slice())])
            .await
            .expect("ext store_values must succeed");

        let tomb_key = fx.tv.key_schema().checkpoint_reconcile_tombstones();
        if step == ReapStep::TombstoneRecord {
            let _: () = fx
                .tv
                .client()
                .pool
                .set(&tomb_key, "not-a-zset", None, None, false)
                .await
                .expect("poisoning the tombstone key must succeed");
        }

        let state = seed_state_with_ext(74, hash, payload.len() as u64);
        let oid = fx
            .create_layer(&format!("vn6-abort-before-delete-{step:?}"), &state)
            .await;
        killer.arm(&fx.tv, oid);

        let header_key = fx.tv.key_schema().obj_header(&oid.to_string());
        let success_before = fx.layer_writes("success");
        let error_before = fx.layer_writes("error");

        let outcome = fx
            .bootstrap
            .run_once(oid, state, BootstrapTarget::CreationBase)
            .await;
        fx.drain().await;

        assert_eq!(
            outcome,
            BootstrapOutcome::UnstampedHeaderAbsent,
            "{step:?}: the outcome names the REFUSAL, which already happened atomically \
             Valkey-side. The reap is a best-effort latency optimisation downstream of it, and \
             nothing it does or fails to do may change which arm the unit reports"
        );
        assert_eq!(
            killer.delete_calls(),
            0,
            "{step:?}: THE assertion. The delete must not be ATTEMPTED — not attempted and \
             failed, not attempted and succeeded. `ext-refs.json` is the only durable record of \
             which blobs this base names and it goes with the base, so a delete on an \
             undischarged obligation strands them with nothing left able to enumerate them"
        );
        let handles = fx
            .store
            .list_layer_checkpoints(oid)
            .await
            .expect("list_layer_checkpoints must succeed");
        assert_eq!(
            handles.len(),
            1,
            "{step:?}: so the generation survives whole, for the reconciliation pass's \
             dead-object phase to reclaim under the leader lease this path does not hold. \
             Got {handles:?}"
        );
        assert!(
            !fx.tv
                .client()
                .pool
                .exists::<bool, _>(&header_key)
                .await
                .expect("EXISTS obj:hdr must succeed"),
            "{step:?}: and the header stays absent regardless — the refusal is a Valkey-side \
             conditional write with no dependency on the reap"
        );
        if step == ReapStep::SidecarRead {
            assert!(
                tombstoned_hashes(&fx).await.is_empty(),
                "the read failed, so no hash could be recorded — and recording a hash the reap \
                 never actually read would start a collect grace against a blob whose only \
                 reference is a sidecar that is still standing"
            );
        } else {
            let kind: String = fx
                .tv
                .client()
                .pool
                .r#type(&tomb_key)
                .await
                .expect("TYPE must succeed");
            assert_eq!(
                kind, "string",
                "the poisoned key must still be a string: the sink failed on it rather than \
                 replacing it, which is what makes this arm a genuine record failure"
            );
        }

        assert_eq!(
            fx.layer_writes("success"),
            success_before,
            "{step:?}: no generation was completed"
        );
        assert_eq!(
            fx.layer_writes("error"),
            error_before,
            "{step:?}: and a failed RECLAMATION is not a failed write — folding it into \
             `{{error, layer}}` would make that cell unreadable as a write-failure signal"
        );
    }
}

// =============================================================================
// Case 18 — b.vn6: the refusal's log LEVEL follows the reap result.
// =============================================================================

/// The post-PUT refusal logs at **INFO when the base was reaped** and at **WARN
/// when it was not**, with byte-identical message text either way.
///
/// # Why the level is behaviour and not cosmetics
///
/// The refusal itself is a routine create-then-delete race — a churny client, a
/// smoke test, an editor with scratch layers — and when the inline reap
/// succeeds there is nothing left for an operator to do: no phantom header, no
/// leaked blob, no pending obligation. A WARN there puts a permanent alert floor
/// under every deployment with churn, which is the fastest way to teach an
/// operator to filter the message. `reaped = false` is the opposite: the inline
/// cleanup failed and reclamation is deferred to the reconciliation pass, which
/// is exactly the "worth following up" signal a WARN is for.
///
/// The two arms carry the **same** message text deliberately, so a log query
/// written against one finds the other; `reaped` is what distinguishes them.
/// This case asserts that too — a future edit that split the text would silently
/// break every such query.
///
/// # `flavor = "current_thread"` is load-bearing
///
/// `tracing::subscriber::set_default` installs the dispatcher in a
/// **thread-local**, and a `#[tokio::test]` future is polled on the calling
/// thread only under the current-thread runtime. On a multi-thread runtime
/// `run_once` can resume on a worker after any of its several `.await`s, the
/// event would be emitted with no dispatcher installed, and the capture buffer
/// would come back empty — reporting "the level is wrong" for a unit that logged
/// it correctly, which is the worst failure mode available. Same pattern and
/// rationale as `rebase_refusal_test.rs`'s `capture_logs`.
///
/// Spelled explicitly rather than left to `#[tokio::test]`'s default, following
/// `rebase_refusal_test.rs`'s
/// `rebase_refuses_compose_and_raises_the_suspect_lineage_tripwire`: the default
/// and the requirement happen to coincide today, so an edit that added
/// `flavor = "multi_thread"` for an unrelated reason would silently empty the
/// capture buffer instead of failing to compile. The attribute IS the assertion
/// here, so it is written down.
#[tokio::test(flavor = "current_thread")]
async fn bootstrap_refusal_logs_info_when_it_reaped_and_warn_when_it_could_not() {
    // `(fail the reap?, expected level)`.
    for (fail_reap, level) in [(false, "INFO"), (true, "WARN")] {
        let enabled = derive_bootstrap_enabled(&all_gates_on_config());
        assert!(enabled, "the log-level arms need the enabled gate");

        let killer = Arc::new(if fail_reap {
            HeaderKillingPutStore::new_with_failing_deletes()
        } else {
            HeaderKillingPutStore::new()
        });
        let fx = bootstrap_fixture(Arc::clone(&killer) as Arc<dyn CheckpointStore>, enabled).await;

        let state = seed_state(75);
        let oid = fx
            .create_layer(&format!("vn6-refusal-level-{level}"), &state)
            .await;
        killer.arm(&fx.tv, oid);

        let (outcome, logs) = capture_logs(fx.bootstrap.run_once(
            oid,
            state,
            BootstrapTarget::CreationBase,
        ))
        .await;
        fx.drain().await;

        assert_eq!(
            outcome,
            BootstrapOutcome::UnstampedHeaderAbsent,
            "{level}: precondition — both arms must reach the refusal, or the captured output is \
             about some other event"
        );

        let line = logs
            .lines()
            .find(|line| line.contains(REFUSAL_LOG_NEEDLE))
            .unwrap_or_else(|| {
                panic!(
                    "{level}: the refusal must be logged at all — it is the only record an \
                     operator has that a base was written for a layer that no longer exists. \
                     Captured:\n{logs}"
                )
            });
        assert!(
            line.contains(level),
            "{level}: the refusal's level must follow the REAP result, not the refusal. A \
             fully-reaped race leaves nothing to act on and must not raise a WARN on every \
             create-then-delete workload; an un-reaped one defers a reclamation and must. Line \
             was:\n{line}"
        );
        assert!(
            line.contains(&format!(
                "reaped={fail_reap_flag}",
                fail_reap_flag = !fail_reap
            )),
            "{level}: and the `reaped` field is what distinguishes the two arms, since the \
             message text is deliberately identical. Line was:\n{line}"
        );
    }
}

/// A substring of the refusal message shared by BOTH levels — see case 18 for
/// why the two arms must not diverge in text.
const REFUSAL_LOG_NEEDLE: &str = "obj:hdr was gone by the cursor stamp";

/// Drive `f` under a thread-local `tracing` subscriber and return its value
/// alongside everything it emitted.
///
/// `with_max_level(TRACE)` rather than `WARN`: an assertion that a line is INFO
/// must be able to see a line the unit emitted at DEBUG, or a demotion would
/// read as "the event is missing" instead of "the level is wrong". Callers must
/// be `current_thread` — see case 18's doc for why.
async fn capture_logs<T>(f: impl std::future::Future<Output = T>) -> (T, String) {
    let buf = Arc::new(std::sync::Mutex::new(Vec::new()));
    let subscriber = tracing_subscriber::fmt()
        .with_writer(CapturingWriter(Arc::clone(&buf)))
        .with_max_level(tracing::Level::TRACE)
        // ANSI off: the assertions match plain substrings, and colour escapes
        // land inside them.
        .with_ansi(false)
        .finish();
    let out = {
        let _guard = tracing::subscriber::set_default(subscriber);
        f.await
    };
    let captured = buf.lock().unwrap_or_else(|p| p.into_inner()).clone();
    (
        out,
        String::from_utf8(captured).expect("captured tracing output must be UTF-8"),
    )
}

/// A `MakeWriter` that captures everything written to it.
#[derive(Clone)]
struct CapturingWriter(Arc<std::sync::Mutex<Vec<u8>>>);

impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for CapturingWriter {
    type Writer = CapturingWriter;
    fn make_writer(&'a self) -> Self::Writer {
        self.clone()
    }
}

impl std::io::Write for CapturingWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .extend_from_slice(buf);
        Ok(buf.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}
