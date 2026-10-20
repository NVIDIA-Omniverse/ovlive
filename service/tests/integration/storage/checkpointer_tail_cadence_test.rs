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

// b.dfh Epic `t1.dfh.57` Task 7, Subtask `t3.dfh.57.1u.dg` — PRD AC-1 driven
// tail cadence, the Epic's flag-off criterion, and Task 4's four pinned
// outcome→metric / cursor-stamp mappings, all driven through the REAL periodic
// loop.
//
// Everything here goes through `spawn_checkpoint_loop`. The sibling suites
// drive `capture_layer_tail` / `attempt_tail_capture_fallback` directly and can
// therefore assert deterministically; this file's whole reason to exist is that
// the wiring in between — the `SkippedNotQuiesced` arm's flag gate, the cadence
// the interval arm imposes on successive captures, the two-pair cursor stamp,
// the kind-aware prune — is only observable end to end. That is also why these
// are the only wall-clock-bound tests in the Task's suite, and why they are
// isolated from the AC-2 matrix.
//
// ── THE OMPF TRAP ───────────────────────────────────────────────────────────
//
// Every NON-EMPTY commit driven from this module passes
// `relay_ompf: Some(Bytes::from(delta_to_ompf_bytes(&delta, &[])))`.
// `LayerStore::update_layer`'s `relay_ompf` argument is what puts the `ompf`
// field on the stream entry, and the tail classifier reads an ompf-less update
// entry as an EMPTY COMMIT by invariant. A fixture that passed `None` would
// still produce a decodable envelope — one whose records are no-ops over real
// content, i.e. the b.pw6 defect class manufactured inside the fixture, and
// invisible to every assertion below. `None` is reserved for genuinely empty
// `OwnedDelta::default()` commits, of which this module drives none.
//
// ── NO `set_gate_timestamps` COPY HERE, DELIBERATELY ────────────────────────
//
// The sibling `checkpointer_metrics_test.rs` backdates `obj:hdr` to pin gate
// ages without waiting real time; its helper writes `last_server_ts` in
// MICROSECONDS and `last_checkpoint_ms` in MILLISECONDS, and getting that pair
// wrong backdates the layer to ~1970 so every gate arm passes — the b.yut defect
// shape reproduced inside a fixture. This module needs no such copy and carries
// none: the quantity under test IS the cadence, so both clocks are driven for
// real (real commits stamp `last_server_ts`; the loop's own success arms stamp
// `last_checkpoint_ms`) and a backdated header would defeat the point. The trap
// is restated only so a later edit reaches for the verbatim helper rather than
// re-deriving one.
//
// ── EVERY LOOP RUNS WITH THE UNPROVABLE-CUT SEAM ARMED ──────────────────────
//
// Every `spawn_checkpoint_loop` here runs on `unprovable_cut_client`
// (`checkpoint::helpers`), armed for the driven layer, and
// `drive_captured_tail` does the same. Every read proves its cut — atomically
// at or below the atomic-read limits, under the snapshot fence above them
// (Spec SDD SR-4.1, SR-11.5) — so without the seam every tick writes a base,
// the `SkippedNotQuiesced` arm is never reached, and nothing in this file
// would have a subject. The seam has the layer's every assembly pin above
// every cut, so the production regression arm skips it and the tick falls
// back to the capture. A "never-quiescent" layer below means exactly that: a
// hot layer whose every assembly is skipped.
//
// The seam exists only under `test-checkpoint-fault-injection`, so this whole
// module is compiled only under that feature — compiled out, not skipped,
// without it (the CI `test:` job's feature set includes it).
//
// ── NO SNAPSHOT BASE LANDS ──────────────────────────────────────────────────
//
// With the seam armed every assembly of the driven layer is skipped, so no
// snapshot-arm base can land. Where a test's claim depends on that — the AC-1
// cadence, the cursor stamp, the retention prune — it asserts it outright
// (`snapshot_base_landed`), so a seam that stops firing fails as itself
// rather than as an unrelated assertion. The others assert only counters and
// outcome identities a snapshot base would not move; the chain-at-K test also
// lists re-base bases, which a listing cannot tell from snapshot ones, so it
// names the seam in its non-vacuity message instead. The cursor claims
// follow `case_d_unprovable_cut_ticks_unverified_and_persists_nothing`'s
// banner (`checkpointer_metrics_test.rs`): a tail landed and the cursor
// advanced across it; `last_checkpoint_seq` never runs ahead of the newest
// generation actually on the adapter.
//
// ── WAIT BUDGETS vs. nextest's HANG-KILLER ──────────────────────────────────
//
// `.config/nextest.toml` marks a test slow at 40 s and HARD-TERMINATES it at
// 120 s (`terminate-after = 3`). A SIGKILLed test prints nothing, so every
// diagnostic panic in this file — and they are most of its value — is lost
// exactly when a real failure needs reading. Each test therefore holds ONE
// deadline across its whole drive (`Instant::now() + <budget>`), and every wait
// inside is bounded by what is left of it: the worst case equals the budget
// whatever the loop does, rather than being a per-wait sum that grows silently
// when a wait is added. Sizing follows
// `interval_arm_bounds_checkpoint_write_volume_for_a_hot_layer`
// (`checkpointer_interval_cadence_test.rs`: interval 2 s, 4.2 s window, 50 ms
// tick), NOT the never-idle test's two 20 s waits.
//
// | Test | Waits | Worst case |
// |---|---|---|
// | `ac1_interval_arm_captures_successive_tails_on_a_never_quiescent_layer` | one `CADENCE_BUDGET` over warm-up + 2 tail observations + 2 cursor waits | **45 s** |
// | `a_capturing_tick_advances_both_the_skipped_layer_and_success_layer_tail_cells` | `drive_captured_tail`'s own 30 s | **30 s** |
// | `a_chain_at_k_tick_rebases_and_moves_no_layer_tail_writes_cell` | one `CHAIN_AT_K_BUDGET` over warm-up + re-base observation | **25 s** |
// | `a_failing_tail_put_moves_error_layer_tail_and_leaves_the_outcome_family_unchanged` | one `PUT_FAILURE_BUDGET` over warm-up + error observation | **25 s** |
// | `after_a_capture_the_header_carries_both_cursor_columns_and_no_restore_seq_bump` | `drive_captured_tail`'s own 30 s | **30 s** |
// | `flag_off_drives_the_same_fixture_with_zero_tail_movement` | one `FLAG_OFF_BUDGET` over warm-up + unverified observation | **25 s** |
// | `a_driven_run_keeps_ten_tail_outcome_cells_and_rebased_at_zero` | `drive_captured_tail`'s own 30 s | **30 s** |
// | `tails_survive_the_retention_prune_at_retain_generations_one` | one `PRUNE_BUDGET` over warm-up + plant + capture + prune observation | **30 s** |
//
// None approaches 120 s, and the two 30 s entries are the shared fixture's own
// deadline rather than a budget this file can shorten.
//
// ── SUITE-LEVEL FLAG-OFF ────────────────────────────────────────────────────
//
// `LIVE_EDIT_CHECKPOINT_TAIL_ENABLED` defaults to `false` and NOTHING in this
// module reads or writes it: every fixture passes `TailCaptureConfig` by value
// into `CheckpointLoopConfig`. So no test here perturbs any sibling's view of
// the flag, and the suite-level "with the flag off nothing changed" evidence is
// the orchestrator's workspace-scope Full test at the Task boundary — where
// every pre-existing checkpointer suite IS the flag-off suite, unchanged.

#![cfg(feature = "test-checkpoint-fault-injection")]

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use bytes::Bytes;
use fred::interfaces::HashesInterface;
use tokio::time::{sleep, timeout, Instant};
use tokio_util::sync::CancellationToken;

use live_edit::checkpoint::adapters::InMemoryCheckpointStore;
use live_edit::checkpoint::{
    spawn_checkpoint_loop, CheckpointBlob, CheckpointError, CheckpointHandle, CheckpointKind,
    CheckpointLoopConfig, CheckpointMeta, CheckpointStore, LayerArtifact, TailCaptureConfig,
};
use live_edit::domain::types::{Field, FieldValue, ObjectState, Section};
use live_edit::observability::metrics::{Metrics, SharedMetrics};
use live_edit::storage::external_values::MockExternalValueStore;
use live_edit::storage::layer::LayerStore;
use live_edit::storage::session::SessionStore;
use live_edit::storage::test_support::TestValkey;

use crate::checkpoint::helpers::{
    drive_captured_tail, gather_tail_outcome_cells, put_base_generation, put_tail_generation,
    read_layer_writes_counter, read_tail_captures_landed, read_tail_counter,
    read_tail_writes_counter, tail_captures_landed, unprovable_cut_client, CAPTURED_TAIL_SEED_SEQ,
};

// ────────────────────────────────────────────────────────────────────────────
// Module constants.
// ────────────────────────────────────────────────────────────────────────────

/// The ten `live_edit_checkpoint_tail_total{outcome}` cells, restated as a
/// LITERAL array.
///
/// Deliberately not an import of `CHECKPOINT_TAIL_OUTCOMES`: importing the
/// production const would make both sides of the vocabulary guard derive from
/// one definition, so adding an outcome would seed the new cell AND grow the
/// expectation in lockstep and the guard would pass by construction. A widening
/// reaches this file only through a deliberate human edit here.
/// `refused_chain_full` is the ninth. It and `rebased` are the two cells the
/// SR-4.1 chain-collapse re-base moves between: a chain-at-K tick ticks
/// `rebased` when the collapse lands and `refused_chain_full` when it does
/// not.
///
/// `refused_lineage_unrootable` is the tenth, added by Epic 5's SR-7
/// adjudication (`t2.dfh.be.s8`) — the two checkpoint-store-DAMAGE arms (a
/// tails-only listing, and a lineage head at sequence 0) split out of
/// `refused_no_base`, which is left meaning exactly "this object has no durable
/// generation at all". This restatement IS the deliberate human edit the
/// paragraph above describes.
const TAIL_OUTCOMES: [&str; 10] = [
    "captured",
    "captured_capped",
    "rebased",
    "refused_trimmed",
    "refused_missing_bytes",
    "refused_sentinel",
    "refused_seq_gap",
    "refused_no_base",
    "refused_chain_full",
    "refused_lineage_unrootable",
];

/// The three `result` values on `live_edit_checkpoint_writes_total`, restated
/// literally for the same reason as [`TAIL_OUTCOMES`].
const WRITE_RESULTS: [&str; 3] = ["success", "error", "skipped"];

/// Sections seeded into every layer this module drives.
///
/// 120 is `case_d_unprovable_cut_ticks_unverified_and_persists_nothing`'s
/// number and `drive_captured_tail`'s: content every tail record can address.
/// Which arm a tick takes is the unprovable-cut seam's, not the width's (see
/// the module header).
const WIDE_LAYER_SECTIONS: u64 = 120;

/// Stream retention for every `LayerStore` this module builds.
///
/// Far above the production default of 4096 on purpose: the writers here commit
/// continuously for the whole drive, and if `MAXLEN` trimming ever reached past
/// a capture window's floor the unit would refuse `refused_trimmed` — a fixture
/// artifact indistinguishable, at the assertion, from the wiring being broken.
const WIDE_LAYER_STREAM_MAXLEN: u64 = 1_000_000;

/// Commits a hot writer must land before a loop is allowed to start.
///
/// A state precondition polled off the writer's own counter, not a timing hack.
/// Without it the first tick races the writer's start-up and may find no commit
/// above the lineage root for a tail to cover.
const WARM_UP_COMMITS: u64 = 20;

/// `max_entries` for every [`TailCaptureConfig`] built here.
///
/// Far above the production default of 4096, and the reason is capture SHAPE
/// rather than capacity: a window that exceeds the entry cap is counted
/// `captured_capped` instead of `captured`, and this module wants the ordinary
/// uncapped spelling to be what its fixtures normally produce.
///
/// It is deliberately **not** load-bearing for any assertion. This constant
/// used to be justified as putting the cap "out of reach", so that `captured`
/// was the only success spelling the module could produce — but that is a
/// headroom argument about runner speed, and CI has since exceeded it: pipeline
/// 65897884 landed ~196k commits inside a single 25 s window, twice this
/// number. Every success assertion here now reads the capture family
/// (`read_tail_captures_landed` / [`TAIL_CAPTURE_LANDED_OUTCOMES`]), which is
/// speed-independent, so a runner fast enough to cap these windows changes the
/// label split and nothing else.
const TAIL_MAX_ENTRIES: u64 = 100_000;

/// Tick cadence for every loop spawned here. Fast enough that a budget is spent
/// waiting for the CODE, not for the next tick.
const LOOP_TICK: Duration = Duration::from_millis(50);

/// An hour — parked above every budget in this module so the gate arm it
/// disables cannot be what lets a write through, and so neither periodic
/// sidecar timer fires inside a test.
const PARKED: Duration = Duration::from_secs(3600);

/// AC-1's cadence under test. Two seconds is the sibling volume-bound test's
/// interval, chosen there for the same reason: long enough that successive
/// writes are visibly spaced, short enough that two of them fit in a budget.
const CADENCE_INTERVAL: Duration = Duration::from_secs(2);

/// One deadline across the whole AC-1 drive: warm-up, two tail observations and
/// two cursor waits all draw from it.
const CADENCE_BUDGET: Duration = Duration::from_secs(45);

/// One deadline across the chain-at-K re-base drive.
const CHAIN_AT_K_BUDGET: Duration = Duration::from_secs(25);

/// One deadline across the failing-put drive.
const PUT_FAILURE_BUDGET: Duration = Duration::from_secs(25);

/// One deadline across the flag-off drive.
const FLAG_OFF_BUDGET: Duration = Duration::from_secs(25);

/// One deadline across the retention-prune drive.
const PRUNE_BUDGET: Duration = Duration::from_secs(30);

/// Poll granularity for every bounded wait below. Also the slack term in the
/// cadence-spacing assertion — an observation can lag the event it observes by
/// at most one of these.
const POLL_INTERVAL: Duration = Duration::from_millis(50);

// ────────────────────────────────────────────────────────────────────────────
// Loop-driving plumbing (file-local by convention — see the Subtask body: only
// the tail-PRODUCING step is shared, via `helpers::drive_captured_tail`).
// ────────────────────────────────────────────────────────────────────────────

fn make_metrics() -> SharedMetrics {
    Arc::new(crate::common::metrics_scrape_only(
        "checkpointer-tail-cadence",
    ))
}

/// Bounded-poll an async predicate against an ABSOLUTE deadline shared by the
/// whole drive. Returns `true` when it holds in time.
async fn poll_until<F, Fut>(deadline: Instant, mut pred: F) -> bool
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = bool>,
{
    while Instant::now() < deadline {
        if pred().await {
            return true;
        }
        sleep(POLL_INTERVAL).await;
    }
    pred().await
}

/// Cancel the loop and bounded-join the handle. Same 5 s shutdown discipline as
/// every sibling checkpointer suite.
async fn cancel_and_join(cancel: CancellationToken, handle: tokio::task::JoinHandle<()>) {
    cancel.cancel();
    timeout(Duration::from_secs(5), handle)
        .await
        .expect("checkpoint loop must exit within 5 s of cancellation")
        .expect("checkpoint loop task must not panic on shutdown");
}

/// Build a loop config with the two periodic sidecar timers parked.
fn loop_config(
    replica_id: &str,
    interval: Duration,
    quiesce: Duration,
    retain_generations: usize,
    tail: TailCaptureConfig,
) -> CheckpointLoopConfig {
    CheckpointLoopConfig {
        tick: LOOP_TICK,
        interval,
        quiesce,
        retain_generations,
        counters_sync: PARKED,
        reconcile_interval: PARKED,
        leader: crate::common::test_leader(replica_id),
        tail,
    }
}

/// A tail config with the entry cap raised out of reach (see
/// [`TAIL_MAX_ENTRIES`]) and every other knob at its production default.
fn tail_config(enabled: bool, max_chain: usize) -> TailCaptureConfig {
    TailCaptureConfig {
        enabled,
        max_chain,
        max_entries: TAIL_MAX_ENTRIES,
        ..TailCaptureConfig::default()
    }
}

/// Seed a session plus a wide layer ([`WIDE_LAYER_SECTIONS`]), at
/// [`CAPTURED_TAIL_SEED_SEQ`] over a [`WIDE_LAYER_STREAM_MAXLEN`] stream.
/// Returns `(object_id, session_id)`.
///
/// The creation sequence is [`CAPTURED_TAIL_SEED_SEQ`] and every caller plants
/// its lineage root at that same number. The coupling is load-bearing, not
/// cosmetic — the capture unit opens its read window at `{S0 + 1}-0` where S0 is
/// the newest generation in the store listing, so pairing a base at S0 with a
/// header sequence of S0 opens the window on exactly the writer's first real
/// commit. Any other pairing leaves a hole below the floor, the classifier
/// returns `Trimmed`, and the fixture silently exercises `refused_trimmed` while
/// its author believes it covers success.
async fn seed_wide_layer(tv: &TestValkey, label: &str) -> (u64, String) {
    let session_id = SessionStore::new(tv.client().clone(), tv.key_schema().clone())
        .create(
            &format!("tail-cadence-session-{label}"),
            &format!("/stage/tail-cadence-{label}"),
            "tail-cadence-user",
        )
        .await
        .expect("session creation must succeed");

    let mut state = ObjectState {
        sequence_num: CAPTURED_TAIL_SEED_SEQ,
        first_version: 1,
        source_format: "usda".to_string(),
        ..Default::default()
    };
    let ids: Vec<u64> = (1..=WIDE_LAYER_SECTIONS).collect();
    for (order, sid) in ids.iter().enumerate() {
        state.sections.insert(
            *sid,
            Section {
                section_id: *sid,
                parent_id: 0,
                name: format!("seed-{sid}"),
                section_type: 1,
                section_order: order as u32,
                first_version: 1,
                last_version: 1,
            },
        );
        state.fields.insert(
            (*sid, "seed-field".to_string()),
            Field {
                section_id: *sid,
                key_name: "seed-field".to_string(),
                value: FieldValue::Inline(vec![0x5E, 0xED]),
                set_order: 0,
                authoring_set_order: 0,
                version: 1,
            },
        );
    }
    state.children.insert(0, ids);

    let object_id = LayerStore::new(
        tv.client().clone(),
        tv.key_schema().clone(),
        WIDE_LAYER_STREAM_MAXLEN,
    )
    .create_layer(
        &session_id,
        &layer_name(label),
        "tail-cadence-user",
        &state,
        Arc::new(MockExternalValueStore::new()),
    )
    .await
    .expect("create_layer must succeed");

    (object_id, session_id)
}

/// The layer name a `label` maps to. One function so the seeder and the writer
/// cannot disagree.
fn layer_name(label: &str) -> String {
    format!("tail-cadence-layer-{label}")
}

/// A background task committing field-set deltas as fast as the store will take
/// them, each carrying a REAL `relay_ompf`.
///
/// Both halves are load-bearing. A field-only delta rides `update_layer`'s Lua
/// fast path — a structural command takes a slower path (the structural route
/// on `lua`, or the WATCH/MULTI/EXEC slow path on the rollback backends, which
/// commits roughly an order of magnitude less often) — and the `relay_ompf` is
/// the module header's OMPF trap.
///
/// There is deliberately NO pacing sleep: every tick finds commits above the
/// cursor to capture, and the layer stays due.
struct HotWriter {
    stop: Arc<AtomicBool>,
    commits: Arc<AtomicU64>,
    error: Arc<Mutex<Option<String>>>,
    handle: tokio::task::JoinHandle<()>,
}

impl HotWriter {
    fn spawn(tv: &TestValkey, session_id: &str, label: &str, object_id: u64) -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let commits = Arc::new(AtomicU64::new(0));
        let error: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));
        let layer_store = Arc::new(LayerStore::new(
            tv.client().clone(),
            tv.key_schema().clone(),
            WIDE_LAYER_STREAM_MAXLEN,
        ));
        let handle = {
            let stop = Arc::clone(&stop);
            let commits = Arc::clone(&commits);
            let error = Arc::clone(&error);
            let session_id = session_id.to_string();
            let layer = layer_name(label);
            tokio::spawn(async move {
                let mut nonce = 10_000u64;
                while !stop.load(Ordering::Relaxed) {
                    let delta = oodf_ffi::OwnedDelta {
                        set_fields: vec![oodf_ffi::SetField {
                            // Section 1 exists in every layer `seed_wide_layer`
                            // creates.
                            section_id: 1,
                            key_name: "hot-field".to_string(),
                            // Distinct per commit, so a silently-dropped delta
                            // cannot be mistaken for one that was applied.
                            value_or_ext_hash: nonce.to_le_bytes().to_vec(),
                            ext_value_size: 0,
                            ext_value_index: 0,
                            set_order: nonce,
                        }],
                        ..Default::default()
                    };
                    let ompf =
                        Bytes::from(crate::common::owned_delta::delta_to_ompf_bytes(&delta, &[]));
                    if let Err(e) = layer_store
                        .update_layer(
                            &session_id,
                            &layer,
                            object_id,
                            Some(delta),
                            Some(ompf),
                            "tail-cadence-hot-writer",
                            0,
                        )
                        .await
                    {
                        // Normal at the END of a test (the pool is torn down at
                        // exit), but recorded either way: a writer that died on
                        // its second commit and one whose arm was simply not
                        // reached produce the same "the arm was never reached"
                        // symptom, and only this string tells them apart.
                        *error.lock().unwrap_or_else(|p| p.into_inner()) = Some(e.to_string());
                        return;
                    }
                    commits.fetch_add(1, Ordering::Relaxed);
                    nonce += 1;
                }
            })
        };
        Self {
            stop,
            commits,
            error,
            handle,
        }
    }

    fn commits(&self) -> u64 {
        self.commits.load(Ordering::Relaxed)
    }

    /// Commit count plus the writer's cause of death, for panic messages.
    fn report(&self) -> String {
        match self.error.lock().unwrap_or_else(|p| p.into_inner()).clone() {
            Some(e) => format!("{} writer commits, then died with: {e}", self.commits()),
            None => format!("{} writer commits, no writer error", self.commits()),
        }
    }

    async fn stop(self) {
        self.stop.store(true, Ordering::Relaxed);
        let _ = self.handle.await;
    }
}

/// Meet the loop with an already-hot layer. Panics with the writer's report on
/// deadline, because every downstream wait would otherwise time out for a reason
/// this one names precisely.
async fn wait_for_warm_up(writer: &HotWriter, deadline: Instant) {
    let reached = poll_until(deadline, || async { writer.commits() >= WARM_UP_COMMITS }).await;
    assert!(
        reached,
        "the hot writer must land {WARM_UP_COMMITS} warm-up commits before the loop starts; \
         without them the first tick races writer start-up and may find no commit above the \
         lineage root to capture ({})",
        writer.report()
    );
}

// ────────────────────────────────────────────────────────────────────────────
// Metric + header readers.
//
// The writes/tail-family readers come from `helpers.rs`
// (`read_layer_writes_counter`, `read_tail_counter`, `read_tail_writes_counter`,
// `gather_tail_outcome_cells`) and are delta-based and panic-on-absent by design
// — see the label-discipline banner there. `read_layer_writes_counter` reads the
// SNAPSHOT form's `kind="layer"` cell, the one a capturing tick must ALSO move
// (Task 4's first pinned decision: the `kind` dimension is what disambiguates
// the two). It was a file-local copy here until Epic 5 Task 2 collapsed the
// three copies of it onto the shared reader. The one below is the family
// helpers.rs does not own.
// ────────────────────────────────────────────────────────────────────────────

/// Read `live_edit_checkpoint_stable_cut_total{outcome, service}`.
///
/// The `service` label is REQUIRED in the match: `Metrics::new` pre-materializes
/// both outcomes at zero under the *metrics* service label, so an outcome-only
/// (subset) match can return that pre-init 0 series instead of the loop's. Folds
/// to 0 when absent — unlike the two families above, the loop's own
/// replica-labelled series does not exist until its first tick.
fn read_stable_cut_counter(metrics: &Metrics, outcome: &str, service: &str) -> u64 {
    live_edit::observability::metrics::test_util::sample_value(
        &live_edit::observability::metrics::test_util::render(metrics),
        "live_edit_checkpoint_stable_cut_total",
        &[("outcome", outcome), ("service", service)],
    )
    .unwrap_or(0.0) as u64
}

/// `HGETALL le:{object_id}:obj:hdr`. One round-trip, so the cursor columns are
/// read atomically by the server rather than assembled from two reads.
async fn read_header(tv: &TestValkey, object_id: u64) -> HashMap<String, String> {
    tv.client()
        .pool
        .hgetall::<HashMap<String, String>, _>(tv.key_schema().obj_header(&object_id.to_string()))
        .await
        .expect("HGETALL obj:hdr must succeed")
}

/// One `obj:hdr` column as a `u64`, `None` when absent. Panics on a
/// non-numeric value — a column the loop writes is either absent or a number.
async fn read_header_u64(tv: &TestValkey, object_id: u64, field: &str) -> Option<u64> {
    read_header(tv, object_id).await.get(field).map(|raw| {
        raw.parse()
            .unwrap_or_else(|e| panic!("obj:hdr {field} must be numeric; got {raw:?} ({e})"))
    })
}

/// Every `live_edit_checkpoint_writes_total{result, kind}` label pair present.
///
/// Parses the family whole rather than probing named cells so an out-of-band
/// `kind` (one the pre-init loop never seeded, materialised on its first
/// occurrence) shows up as an EXTRA pair instead of being invisible to a
/// per-name lookup.
fn gather_writes_label_pairs(metrics: &Metrics) -> HashSet<(String, String)> {
    let text = live_edit::observability::metrics::test_util::render(metrics);
    let mut out: HashSet<(String, String)> = HashSet::new();
    for line in text.lines() {
        let Some(rest) = line.strip_prefix("live_edit_checkpoint_writes_total") else {
            continue;
        };
        if !rest.starts_with('{') {
            continue;
        }
        let extract = |label: &str| -> Option<String> {
            let needle = format!("{label}=\"");
            let i = line.find(&needle)?;
            let after = &line[i + needle.len()..];
            let end = after.find('"')?;
            Some(after[..end].to_owned())
        };
        let result = extract("result").expect("`result` label must be present on every series");
        let kind = extract("kind").expect("`kind` label must be present on every series");
        out.insert((result, kind));
    }
    out
}

/// The `{result, kind="layer_tail"}` row of the writes counter, as
/// `result → value`. Read before and after a drive so "unchanged" is a delta
/// claim rather than an absolute one (the family is zero-seeded, so an absolute
/// zero is also what a feature that never ran produces).
fn tail_writes_row(metrics: &Metrics) -> HashMap<&'static str, u64> {
    WRITE_RESULTS
        .iter()
        .map(|r| (*r, read_tail_writes_counter(metrics, r)))
        .collect()
}

/// Handles of one kind in a listing, oldest-first by sequence.
fn of_kind(handles: &[CheckpointHandle], kind: CheckpointKind) -> Vec<CheckpointHandle> {
    let mut out: Vec<CheckpointHandle> =
        handles.iter().filter(|h| h.kind == kind).cloned().collect();
    out.sort_by_key(|h| h.snap_seq);
    out
}

/// Did a snapshot-arm BASE land during the drive — i.e. did some tick's
/// assembly escape the unprovable-cut seam?
///
/// `planted` names every base the test put by hand (each fixture plants its
/// lineage root at [`CAPTURED_TAIL_SEED_SEQ`]; the retention test plants a
/// second), so any OTHER base in the listing came from the snapshot arm. A tick
/// that wrote one would stamp all THREE cursor columns, restart the cadence,
/// and root a new lineage unit above every tail, so a test whose claim rests
/// on the capture arm alone asserts this is `false`.
///
/// Not usable where the loop re-bases: a re-base writes a base too, and a
/// listing cannot tell the two apart.
fn snapshot_base_landed(handles: &[CheckpointHandle], planted: &[u64]) -> bool {
    handles
        .iter()
        .any(|h| h.kind == CheckpointKind::Base && !planted.contains(&h.snap_seq))
}

// ────────────────────────────────────────────────────────────────────────────
// `TailAwareCheckpointStore` — records every layer put's kind, and optionally
// fails the Tail ones.
//
// Composed atop `helpers::DelegatingCheckpointStore` (the newtype pattern
// `checkpointer_leader_race_test.rs::CountingCheckpointStore` establishes).
// EVERY trait method is implemented explicitly, including the four carrying
// non-trivial defaults (`read_ext_refs_sidecar`, `list_layer_artifacts`,
// `delete_layer_temp_artifacts`, `publish_worst_case`) — the delegation skeleton
// is a held FIELD, not a blanket impl, so an omission here is not an
// un-instrumented pass-through but a silent behaviour SUBSTITUTION that no test
// fails on.
//
// Two jobs, one type, because they are the same instrumentation seen from two
// sides:
//   - Recording makes "no store call carried a tail" a POSITIVE assertion for
//     the flag-off test, rather than the absence of an artifact — which an
//     adapter that silently dropped the put would also produce.
//   - Failing on `kind == Tail` drives Task 4's third pinned decision: the
//     capture unit's put fails, so `capture_layer_tail` returns `Err`, the
//     fallback ticks `writes_total{error, layer_tail}`, and the outcome family
//     stays untouched (an error is not a decision, so it has no cell there).
// ────────────────────────────────────────────────────────────────────────────

struct TailAwareCheckpointStore {
    delegate: crate::checkpoint::helpers::DelegatingCheckpointStore<InMemoryCheckpointStore>,
    fail_tail_puts: bool,
    layer_put_kinds: Mutex<Vec<CheckpointKind>>,
}

impl TailAwareCheckpointStore {
    fn recording() -> Self {
        Self::new(false)
    }

    fn failing_tail_puts() -> Self {
        Self::new(true)
    }

    fn new(fail_tail_puts: bool) -> Self {
        Self {
            delegate: crate::checkpoint::helpers::DelegatingCheckpointStore::new(Arc::new(
                InMemoryCheckpointStore::new(),
            )),
            fail_tail_puts,
            layer_put_kinds: Mutex::new(Vec::new()),
        }
    }

    /// Every `put_layer_checkpoint` ATTEMPT's kind, in call order — attempts,
    /// not successes, so a failed tail put is still visible.
    fn layer_put_kinds(&self) -> Vec<CheckpointKind> {
        self.layer_put_kinds
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .clone()
    }

    fn tail_put_attempts(&self) -> usize {
        self.layer_put_kinds()
            .into_iter()
            .filter(|k| *k == CheckpointKind::Tail)
            .count()
    }
}

#[async_trait]
impl CheckpointStore for TailAwareCheckpointStore {
    async fn put_layer_checkpoint(
        &self,
        object_id: u64,
        snap_seq: u64,
        ompf: Bytes,
        meta: CheckpointMeta,
    ) -> Result<CheckpointHandle, CheckpointError> {
        // Recorded BEFORE the fail branch: the test wants to know a tail put was
        // ATTEMPTED even when this wrapper is the thing refusing it.
        let kind = meta.kind;
        self.layer_put_kinds
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .push(kind);
        if self.fail_tail_puts && kind == CheckpointKind::Tail {
            return Err(CheckpointError::Transient {
                source: "test-injected transient failure on a Tail put"
                    .to_string()
                    .into(),
            });
        }
        self.delegate
            .put_layer_checkpoint(object_id, snap_seq, ompf, meta)
            .await
    }

    async fn get_latest_layer_checkpoint(
        &self,
        object_id: u64,
    ) -> Result<Option<CheckpointBlob>, CheckpointError> {
        self.delegate.get_latest_layer_checkpoint(object_id).await
    }

    // Forwarded per `CheckpointStore::get_layer_checkpoint`'s MUST — its trait
    // default returns `NotImplemented`, which would mask a generation the
    // delegate holds rather than merely leaving the call un-instrumented.
    async fn get_layer_checkpoint(
        &self,
        handle: &CheckpointHandle,
    ) -> Result<Option<CheckpointBlob>, CheckpointError> {
        self.delegate.get_layer_checkpoint(handle).await
    }

    async fn list_layer_checkpoints(
        &self,
        object_id: u64,
    ) -> Result<Vec<CheckpointHandle>, CheckpointError> {
        self.delegate.list_layer_checkpoints(object_id).await
    }

    async fn delete_layer_checkpoint(
        &self,
        handle: CheckpointHandle,
    ) -> Result<(), CheckpointError> {
        self.delegate.delete_layer_checkpoint(handle).await
    }

    async fn put_session_checkpoint(
        &self,
        session_id: &str,
        body: Bytes,
        meta: CheckpointMeta,
    ) -> Result<CheckpointHandle, CheckpointError> {
        self.delegate
            .put_session_checkpoint(session_id, body, meta)
            .await
    }

    async fn get_session_checkpoint(
        &self,
        session_id: &str,
    ) -> Result<Option<CheckpointBlob>, CheckpointError> {
        self.delegate.get_session_checkpoint(session_id).await
    }

    async fn delete_session_checkpoint(&self, session_id: &str) -> Result<(), CheckpointError> {
        self.delegate.delete_session_checkpoint(session_id).await
    }

    async fn list_session_checkpoint_ids(&self) -> Result<Vec<String>, CheckpointError> {
        self.delegate.list_session_checkpoint_ids().await
    }

    async fn put_ext_blob(&self, hash_hex: &str, bytes: Bytes) -> Result<(), CheckpointError> {
        self.delegate.put_ext_blob(hash_hex, bytes).await
    }

    async fn get_ext_blob(&self, hash_hex: &str) -> Result<Option<Bytes>, CheckpointError> {
        self.delegate.get_ext_blob(hash_hex).await
    }

    async fn ext_blob_exists(&self, hash_hex: &str) -> Result<bool, CheckpointError> {
        self.delegate.ext_blob_exists(hash_hex).await
    }

    async fn delete_ext_blob(&self, hash_hex: &str) -> Result<(), CheckpointError> {
        self.delegate.delete_ext_blob(hash_hex).await
    }

    async fn put_counters_sidecar(&self, body: Bytes) -> Result<(), CheckpointError> {
        self.delegate.put_counters_sidecar(body).await
    }

    async fn get_counters_sidecar(&self) -> Result<Option<Bytes>, CheckpointError> {
        self.delegate.get_counters_sidecar().await
    }

    async fn list_all_object_ids(&self) -> Result<Vec<u64>, CheckpointError> {
        self.delegate.list_all_object_ids().await
    }

    async fn list_referenced_ext_hashes(&self) -> Result<HashSet<String>, CheckpointError> {
        self.delegate.list_referenced_ext_hashes().await
    }

    async fn list_live_ext_set(
        &self,
    ) -> Result<live_edit::checkpoint::LiveExtSet, CheckpointError> {
        self.delegate.list_live_ext_set().await
    }

    // The four below carry non-trivial trait defaults: omitting them would
    // substitute `Ok(empty)`, `Ok(vec![])`, `Ok(0)` and the local publish bound
    // rather than merely leaving them un-instrumented.
    async fn read_ext_refs_sidecar(
        &self,
        handle: &CheckpointHandle,
    ) -> Result<HashSet<String>, CheckpointError> {
        self.delegate.read_ext_refs_sidecar(handle).await
    }

    async fn list_layer_artifacts(&self) -> Result<Vec<LayerArtifact>, CheckpointError> {
        self.delegate.list_layer_artifacts().await
    }

    async fn delete_layer_temp_artifacts(
        &self,
        object_id: u64,
        snap_seq: u64,
    ) -> Result<usize, CheckpointError> {
        self.delegate
            .delete_layer_temp_artifacts(object_id, snap_seq)
            .await
    }

    fn publish_worst_case(&self) -> Duration {
        self.delegate.publish_worst_case()
    }
}

// ────────────────────────────────────────────────────────────────────────────
// AC-1 — successive tails at interval cadence on a never-quiescent layer.
//
// The layer is committed continuously and the unprovable-cut seam skips its
// every assembly: pre-tail, such a layer produced NOTHING durable, ever.
// `quiesce` is parked at an hour precisely so the quiesce arm
// CANNOT be what lets a capture through — every tail observed below is the
// interval arm's doing.
//
// The claim is a CADENCE, not a single event, so two tails are required: the
// first is explainable on its own by `last_checkpoint_ms` being absent ("never
// checkpointed"), and only the second can come from the interval arm measuring
// time since the last CHECKPOINT. That second capture is also where the
// `{unverified rising, captured rising}` pair earns its keep (SR-8.2): a tail
// success does NOT suppress the DR-exposure signal, so both must move together
// and they are asserted in one statement.
// ────────────────────────────────────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn ac1_interval_arm_captures_successive_tails_on_a_never_quiescent_layer() {
    const REPLICA: &str = "tail-cadence-ac1-replica";
    let deadline = Instant::now() + CADENCE_BUDGET;

    let tv = TestValkey::setup_with_pool_size(8).await;
    let metrics = make_metrics();
    let (object_id, session_id) = seed_wide_layer(&tv, "ac1").await;
    let store: Arc<dyn CheckpointStore> = Arc::new(InMemoryCheckpointStore::new());

    // The lineage root, at the layer's own creation sequence. Without it every
    // capture refuses `refused_no_base` forever.
    let root = put_base_generation(
        &store,
        object_id,
        &session_id,
        &layer_name("ac1"),
        CAPTURED_TAIL_SEED_SEQ,
    )
    .await;

    let writer = HotWriter::spawn(&tv, &session_id, "ac1", object_id);
    wait_for_warm_up(&writer, deadline).await;

    let unverified_before = read_stable_cut_counter(&metrics, "unverified", REPLICA);
    let captured_before = read_tail_captures_landed(&metrics);

    let cancel = CancellationToken::new();
    let loop_handle = spawn_checkpoint_loop(
        unprovable_cut_client(&tv, object_id),
        tv.key_schema().clone(),
        Arc::clone(&store),
        loop_config(
            REPLICA,
            CADENCE_INTERVAL,
            // An hour: the quiesce arm is structurally unable to fire on a layer
            // committed continuously, so every capture below is attributable to
            // the interval arm.
            PARKED,
            // Well above anything this test plants: the landed prune counts
            // LINEAGE UNITS (one per base, with the tails grouped above it),
            // and a low quota would be one more variable in a fixture whose
            // failures are already hard to read — the more so now that an
            // over-run quota can delete a tail as well as a base.
            8,
            tail_config(true, 8),
        ),
        Some(Arc::clone(&metrics)),
        cancel.clone(),
    );

    // ── First capture ───────────────────────────────────────────────────────
    let first = wait_for_tail_above(&store, object_id, CAPTURED_TAIL_SEED_SEQ, deadline).await;
    let first_at = Instant::now();
    let first_tail = first.unwrap_or_else(|| {
        panic!(
            "a never-quiescent layer must produce a captured TAIL within the drive budget \
             ({CADENCE_BUDGET:?}); ({}). Root planted at {root:?}. Pre-tail this layer produced \
             nothing durable at all — that is the gap AC-1 closes",
            writer.report()
        )
    });
    let (first_seq, first_ms) =
        wait_for_cursor_pair(&tv, object_id, first_tail.snap_seq, deadline).await;
    let commits_at_first = writer.commits();

    // ── Second capture ──────────────────────────────────────────────────────
    let second = wait_for_tail_above(&store, object_id, first_tail.snap_seq, deadline).await;
    let second_at = Instant::now();
    // Read the writer's progress BEFORE stopping it: the point is that the
    // second capture landed mid-write, not after the layer fell quiet.
    let commits_at_second = writer.commits();
    let second_tail = second.unwrap_or_else(|| {
        panic!(
            "last_checkpoint's tail lineage must extend a SECOND time within the drive budget \
             while the layer is committed continuously — that advance is \
             LIVE_EDIT_CHECKPOINT_INTERVAL_SECS doing its job on a layer the snapshot form can \
             never serve. It stalled at to_seq={} ({})",
            first_tail.snap_seq,
            writer.report()
        )
    });
    let (second_seq, second_ms) =
        wait_for_cursor_pair(&tv, object_id, second_tail.snap_seq, deadline).await;

    cancel_and_join(cancel, loop_handle).await;
    writer.stop().await;

    let unverified_delta =
        read_stable_cut_counter(&metrics, "unverified", REPLICA) - unverified_before;
    let captured_delta = read_tail_captures_landed(&metrics) - captured_before;
    let handles = store
        .list_layer_checkpoints(object_id)
        .await
        .expect("list_layer_checkpoints must succeed");
    let tails = of_kind(&handles, CheckpointKind::Tail);

    // Every claim below is about the CAPTURE arm alone: the cadence measured
    // from its stamps, and both tails in the newest lineage unit. A
    // snapshot-arm base would restamp the cursor and root a new unit above
    // them, so its absence is asserted first (module header, "No snapshot base
    // lands").
    assert!(
        !snapshot_base_landed(&handles, &[CAPTURED_TAIL_SEED_SEQ]),
        "the unprovable-cut seam skips every assembly of the driven layer, so no snapshot-arm \
         base may land; one here means the seam stopped firing and the cadence below is no \
         longer the capture arm's alone. Listing: {handles:?}"
    );

    // (i) Two tails, at interval-bounded spacing. The spacing is the anti-b.yut
    // half: a gate that passed on every tick would produce captures at the TICK
    // rate (50 ms), not at the interval. Slack is one poll interval on each
    // observation, so half the interval is a bound the healthy path clears by a
    // factor of four while the defect shape fails it by a factor of twenty.
    assert!(
        second_tail.snap_seq > first_tail.snap_seq,
        "the second tail must cover a strictly newer cut than the first ({} vs {})",
        second_tail.snap_seq,
        first_tail.snap_seq
    );
    let spacing = second_at.duration_since(first_at);
    assert!(
        spacing >= CADENCE_INTERVAL / 2,
        "successive captures of one layer must be interval-bounded: observed {spacing:?} between \
         the two tails at interval={CADENCE_INTERVAL:?} (allowing one {POLL_INTERVAL:?} poll of \
         observation lag at each end). A spacing near the {LOOP_TICK:?} tick means the gate is \
         passing on every tick rather than measuring time since the last checkpoint"
    );

    // (ii) + (iii) Both cursor columns advance across each capture, and the seq
    // column names the tail's own `to_seq` upper bound.
    assert!(
        second_ms > first_ms,
        "last_checkpoint_ms must advance across each capture — it is what the interval arm \
         measures from, and a capture that did not re-stamp it would make the layer eligible on \
         the very next tick ({first_ms} then {second_ms})"
    );
    assert!(
        first_seq >= first_tail.snap_seq && second_seq >= second_tail.snap_seq,
        "last_checkpoint_seq must advance to each tail's to_seq (a tail's upper, inclusive commit \
         bound IS the newest sequence now covered by a durable generation); observed \
         {first_seq} for to_seq={} and {second_seq} for to_seq={}",
        first_tail.snap_seq,
        second_tail.snap_seq
    );

    // The Case D ceiling, restated unconditionally: the cursor may never name a
    // cut the adapter does not hold. Advancing past what was written is the half
    // of b.pw6 that makes an omission permanent.
    let newest_generation_seq = handles.iter().map(|h| h.snap_seq).max().unwrap_or(0);
    assert!(
        second_seq <= newest_generation_seq,
        "last_checkpoint_seq={second_seq} is ahead of every generation the adapter holds for \
         object {object_id} (newest={newest_generation_seq}, listing={handles:?}) — the cursor \
         advanced off a cut whose artifact was never written"
    );

    // (iv) Commits still in flight ACROSS the captures.
    assert!(
        commits_at_second > commits_at_first && commits_at_first > 0,
        "writer commits must still have been in flight across the two captures (first observed at \
         {commits_at_first}, second at {commits_at_second}) — otherwise the layer went quiet and \
         the run proves nothing about a never-quiescent layer"
    );

    // (v) The healthy signature, asserted AS A PAIR (SR-8.2).
    assert!(
        unverified_delta >= 1 && captured_delta >= 2,
        "the feature's healthy signature is BOTH counters rising together: \
         stable_cut_total{{outcome=\"unverified\"}} keeps its meaning (the snapshot form still \
         could not prove a cut) while tail_total{{outcome=\"captured\"|\"captured_capped\"}} \
         records the fallback that did land. Observed unverified +{unverified_delta}, captured +{captured_delta} over \
         {} tail generation(s). A capture that SUPPRESSED the unverified tick would hide DR \
         exposure from every operator alert keyed on it",
        tails.len()
    );
    assert!(
        tails.len() >= 2,
        "both observed tails must still be on the adapter at the end of the drive — they are \
         members of the NEWEST lineage unit, which retention never prunes, and the quota is \
         parked far above this fixture's unit count anyway; listing={handles:?}"
    );
}

/// Bounded-wait for a `Tail` handle whose `to_seq` sits strictly above `above`.
/// `None` on deadline; the caller owns the panic message.
async fn wait_for_tail_above(
    store: &Arc<dyn CheckpointStore>,
    object_id: u64,
    above: u64,
    deadline: Instant,
) -> Option<CheckpointHandle> {
    // Hand-rolled rather than routed through `poll_until`: the predicate has to
    // hand the found handle back out, and a `FnMut` closure cannot let a
    // borrowed capture escape into the async block it returns.
    loop {
        let handles = store
            .list_layer_checkpoints(object_id)
            .await
            .expect("list_layer_checkpoints must succeed during a drive");
        let found = handles
            .into_iter()
            .filter(|h| h.kind == CheckpointKind::Tail && h.snap_seq > above)
            .min_by_key(|h| h.snap_seq);
        if found.is_some() {
            return found;
        }
        if Instant::now() >= deadline {
            return None;
        }
        sleep(POLL_INTERVAL).await;
    }
}

/// Bounded-wait until `last_checkpoint_seq >= at_least`, then return BOTH cursor
/// columns from one `HGETALL`.
///
/// The listing and the cursor stamp are two writes: a tail can be visible on the
/// adapter microseconds before its HSET lands, so the wait is what makes the
/// column read meaningful rather than racy.
async fn wait_for_cursor_pair(
    tv: &TestValkey,
    object_id: u64,
    at_least: u64,
    deadline: Instant,
) -> (u64, u64) {
    let reached = poll_until(deadline, || async {
        read_header_u64(tv, object_id, "last_checkpoint_seq")
            .await
            .is_some_and(|seq| seq >= at_least)
    })
    .await;
    let header = read_header(tv, object_id).await;
    assert!(
        reached,
        "last_checkpoint_seq must reach {at_least} (the sequence of the generation just observed \
         on the adapter) within the drive budget; obj:hdr for object {object_id} reads {header:?}. \
         A durable generation with no cursor means the post-capture HSET never landed"
    );
    let seq = header
        .get("last_checkpoint_seq")
        .and_then(|raw| raw.parse().ok())
        .expect("the wait above just observed a numeric last_checkpoint_seq");
    let ms = header
        .get("last_checkpoint_ms")
        .and_then(|raw| raw.parse().ok())
        .unwrap_or_else(|| {
            panic!(
                "last_checkpoint_ms must be present and numeric whenever last_checkpoint_seq is: \
                 the capture arm writes both in ONE HSET, so observing the pair half-written means \
                 a writer started stamping the cadence clock on its own (b.yut). obj:hdr reads \
                 {header:?}"
            )
        });
    (seq, ms)
}

// ────────────────────────────────────────────────────────────────────────────
// Pinned decision 1 (Task 4) — a CAPTURING tick advances BOTH
// `writes_total{skipped, layer}` and `writes_total{success, layer_tail}`.
//
// Not a re-derivation: Task 4 pinned this and this lane is the sole owner of the
// loop-driven evidence for it. The pairing is deliberate and easy to break in
// either direction:
//
//   - `{skipped, layer}` records that the SNAPSHOT form was skipped, which stays
//     true however the fallback then goes. Dropping it would make a captured
//     layer indistinguishable from one the assembler served.
//   - `{success, layer_tail}` records the artifact that DID land. The `kind`
//     dimension is the whole disambiguator — both cells live on one counter, and
//     a regression that reused `kind="layer"` for the tail write would silently
//     inflate the snapshot form's success rate.
//
// Delta-based on both cells: the family is zero-seeded, so an absolute reading
// cannot distinguish a feature that ran from one that never did.
// ────────────────────────────────────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_capturing_tick_advances_both_the_skipped_layer_and_success_layer_tail_cells() {
    let tv = TestValkey::setup_with_pool_size(8).await;
    let metrics = make_metrics();
    let store: Arc<dyn CheckpointStore> = Arc::new(InMemoryCheckpointStore::new());
    let session_id = SessionStore::new(tv.client().clone(), tv.key_schema().clone())
        .create("tail-writes-pair-session", "/stage", "tail-cadence-user")
        .await
        .expect("session creation must succeed");

    let skipped_before = read_layer_writes_counter(&metrics, "skipped");
    let tail_success_before = read_tail_writes_counter(&metrics, "success");
    let captured_before = read_tail_captures_landed(&metrics);

    let handle = drive_captured_tail(
        &tv,
        &store,
        &metrics,
        &session_id,
        "tail-writes-pair-layer",
        tail_config(true, 8),
    )
    .await;

    let skipped_delta = read_layer_writes_counter(&metrics, "skipped") - skipped_before;
    let tail_success_delta = read_tail_writes_counter(&metrics, "success") - tail_success_before;
    let captured_delta = read_tail_captures_landed(&metrics) - captured_before;

    assert_eq!(
        handle.kind,
        CheckpointKind::Tail,
        "the fixture must have produced a Tail for this claim to be about a capturing tick; got \
         {handle:?}"
    );
    assert!(
        tail_success_delta >= 1 && captured_delta >= 1,
        "a capture that landed must move BOTH the artifact-level cell \
         writes_total{{success, layer_tail}} (+{tail_success_delta}) and the outcome-level \
         capture family tail_total{{captured|captured_capped}} (+{captured_delta}). They are a pair: the outcome says what was \
         decided, the writes cell says what was written"
    );
    assert!(
        skipped_delta >= tail_success_delta,
        "every layer-tick that captured must ALSO have ticked writes_total{{skipped, layer}} — the \
         snapshot form was skipped on that tick and the fallback deliberately reuses that cell \
         rather than widening the counter's pinned label cardinality. Observed skipped \
         +{skipped_delta} against layer_tail successes +{tail_success_delta}; skipped trailing the \
         tail successes means the arm stopped recording the skipped snapshot"
    );
}

// ────────────────────────────────────────────────────────────────────────────
// Pinned decision 2 (Task 4) — a loop-driven chain-at-K tick moves its
// `tail_total` cell and NOTHING on `writes_total{*, layer_tail}`.
//
// A lineage of one base plus one tail at `max_chain = 1` is at the bound before
// the capture unit reads a single stream entry, so the FIRST tick is already at
// the bound. WHICH cell it moves changed when SR-4.1 landed: the chain-at-K
// return is now the re-base trigger, so the healthy tick collapses the lineage
// and ticks `rebased` rather than `refused_chain_full`. The
// `refused_chain_full` reading of this fixture is now the FAILURE reading — it
// would mean the re-base was attempted and did not land.
//
// The claim that matters is unchanged, and it is that the RE-BASE ITSELF writes
// nothing on the writes counter's `layer_tail` row. Neither a refusal nor a
// re-base persists a tail: a refusal persists nothing at all, and a re-base
// persists a BASE and therefore counts on `kind="layer"`. `{skipped,
// layer_tail}` in particular must stay at exactly ZERO — it is the cell an
// operator would read as "the tail path declined N times", and having either
// outcome land there as well as on `tail_total` would make the two families
// disagree about what a refusal is.
//
// # Why the claim is stated as an EQUALITY against the capture outcomes
//
// A collapse leaves the lineage zero-length again, so capture RESUMES on the new
// base: the very next driven tick is free to write a perfectly legitimate tail
// (SDD §22.20.1 — incremental tails between full bases). The drive here is a
// free-running loop at `LOOP_TICK` cancelled by a `POLL_INTERVAL` poll, so
// whether that follow-on tick fits inside the drive is decided by the runner's
// speed, not by the code under test. The earlier form of this test asserted the
// whole `layer_tail` row was UNTOUCHED, which silently assumed the drive stopped
// at the re-base tick; it held on an idle developer box and failed on CI, where
// one extra tick fitted and captured (confirmed by extending the drive locally:
// `rebased +1, captured +1, {success, layer_tail} +1`, and a second tail on the
// adapter).
//
// So the claim is pinned in the timing-independent form instead: every
// `{success, layer_tail}` tick must be MATCHED 1:1 by a `captured` /
// `captured_capped` outcome, which leaves the re-base contributing exactly zero
// to that cell — the same defect the old form was written to catch (a re-base
// filing its BASE under `kind="layer_tail"` would break the equality whether or
// not a follow-on capture ran), with none of the dependence on how many ticks
// fit. `{error, layer_tail}` and `{skipped, layer_tail}` stay at a flat zero:
// no arm of EITHER path writes them on a healthy drive.
//
// Note what this fixture deliberately does NOT constrain: the `kind="layer"`
// row. A re-base tick moves two cells there ({skipped, layer} from the enclosing
// arm plus {success, layer}), which is documented at the arm and is not this
// test's subject. It also does not constrain HOW MANY re-bases the drive gets —
// a follow-on capture re-fills the chain to K and the tick after it may collapse
// again. Every assertion below is a per-outcome accounting identity for exactly
// that reason.
// ────────────────────────────────────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_chain_at_k_tick_rebases_and_moves_no_layer_tail_writes_cell() {
    const REPLICA: &str = "tail-cadence-chain-full-replica";
    /// K. One tail above the base is already the whole quota, so the very first
    /// tick is already AT the bound and takes the re-base path.
    const MAX_CHAIN: usize = 1;
    let deadline = Instant::now() + CHAIN_AT_K_BUDGET;

    let tv = TestValkey::setup_with_pool_size(8).await;
    let metrics = make_metrics();
    let (object_id, session_id) = seed_wide_layer(&tv, "chain-full").await;
    let store: Arc<dyn CheckpointStore> = Arc::new(InMemoryCheckpointStore::new());
    let layer = layer_name("chain-full");

    // A base plus ONE tail = a full chain at K = 1. Both hand-put, so the
    // refusal is a property of the lineage the loop finds rather than of
    // anything it has to produce first.
    put_base_generation(
        &store,
        object_id,
        &session_id,
        &layer,
        CAPTURED_TAIL_SEED_SEQ,
    )
    .await;
    put_tail_generation(
        &store,
        object_id,
        &session_id,
        &layer,
        CAPTURED_TAIL_SEED_SEQ,
        CAPTURED_TAIL_SEED_SEQ,
        CAPTURED_TAIL_SEED_SEQ + 1,
    )
    .await;

    let writer = HotWriter::spawn(&tv, &session_id, "chain-full", object_id);
    wait_for_warm_up(&writer, deadline).await;

    // SR-4.1 CELL SWAP (Epic 5, t3.dfh.be.84.ec). This fixture's chain-at-K tick
    // no longer refuses — it RE-BASES, so the outcome cell moved
    // `refused_chain_full` -> `rebased`. Every other claim below is unchanged and
    // still holds: a re-base counts on `writes_total{success, kind="layer"}`
    // (it produces a BASE), so the whole `kind="layer_tail"` row stays silent,
    // and the re-base deletes nothing so the hand-put tail is still the only
    // tail on the adapter.
    let rebased_before = read_tail_counter(&metrics, "rebased");
    // The two capture spellings, read as a pair with the writes row so the
    // accounting identity below is a delta claim on both sides.
    let captured_before = read_tail_counter(&metrics, "captured");
    let capped_before = read_tail_counter(&metrics, "captured_capped");
    let writes_before = tail_writes_row(&metrics);

    let cancel = CancellationToken::new();
    let loop_handle = spawn_checkpoint_loop(
        unprovable_cut_client(&tv, object_id),
        tv.key_schema().clone(),
        Arc::clone(&store),
        // Both gates open so every tick reaches the assembler; the capture
        // fallback is downstream of the assembler LOSING, not of a gate.
        loop_config(
            REPLICA,
            Duration::ZERO,
            Duration::ZERO,
            8,
            tail_config(true, MAX_CHAIN),
        ),
        Some(Arc::clone(&metrics)),
        cancel.clone(),
    );

    let saw_rebase = poll_until(deadline, || async {
        read_tail_counter(&metrics, "rebased") > rebased_before
    })
    .await;

    cancel_and_join(cancel, loop_handle).await;
    let commits = writer.report();
    writer.stop().await;

    let rebased_delta = read_tail_counter(&metrics, "rebased") - rebased_before;
    // Read AFTER the loop has been cancelled and joined, so nothing can move
    // between the two families and the identity below is exact rather than a
    // snapshot of a running loop.
    let captured_delta = (read_tail_counter(&metrics, "captured") - captured_before)
        + (read_tail_counter(&metrics, "captured_capped") - capped_before);
    let writes_after = tail_writes_row(&metrics);
    let handles = store
        .list_layer_checkpoints(object_id)
        .await
        .expect("list_layer_checkpoints must succeed");

    assert!(
        saw_rebase && rebased_delta >= 1,
        "a lineage already carrying max_chain={MAX_CHAIN} tail(s) above its base must collapse \
         into a new base and tick `rebased` on a driven tick within {CHAIN_AT_K_BUDGET:?}; observed \
         +{rebased_delta} ({commits}). Listing: {handles:?}. Zero movement means the arm was never \
         reached — check that the unprovable-cut seam still fires on this fixture"
    );
    let success_delta = writes_after["success"] - writes_before["success"];
    assert_eq!(
        success_delta, captured_delta,
        "every writes_total{{success, kind=\"layer_tail\"}} tick must be matched 1:1 by a \
         `captured` / `captured_capped` outcome, which leaves the re-base contributing exactly \
         ZERO to that cell: a re-base produces a BASE and counts on kind=\"layer\". Observed \
         +{success_delta} against +{captured_delta} capture outcome(s) over {rebased_delta} \
         re-base(s) ({commits}). A surplus is the re-base arm filing its base under the tail \
         kind; a deficit is a capture that landed without its artifact-level cell. Listing: \
         {handles:?}"
    );
    for result in ["error", "skipped"] {
        let before = writes_before[result];
        let after = writes_after[result];
        assert_eq!(
            before, after,
            "no arm of EITHER path writes writes_total{{result={result:?}, kind=\"layer_tail\"}} \
             on a healthy drive — a re-base persists a BASE, a refusal persists nothing, and a \
             capture that landed AND stamped its cursor moves only `success`; it moved {before} \
             -> {after} over {rebased_delta} re-base(s) and {captured_delta} capture(s)"
        );
    }
    assert_eq!(
        read_tail_writes_counter(&metrics, "skipped"),
        0,
        "writes_total{{skipped, layer_tail}} must be exactly 0: no arm of the tail path writes it, \
         and a refusal landing there as well as on tail_total would make the two families disagree \
         about what a refusal is"
    );
    let tails = of_kind(&handles, CheckpointKind::Tail);
    assert!(
        tails
            .iter()
            .any(|h| h.snap_seq == CAPTURED_TAIL_SEED_SEQ + 1),
        "the hand-put tail that filled the chain must SURVIVE the collapse: the re-base itself \
         deletes NOTHING (SR-4.1; retention owns every deletion, and this drive retains 8 \
         generations). Listing: {handles:?}"
    );
    assert_eq!(
        tails.len() as u64,
        1 + captured_delta,
        "the adapter must hold the hand-put tail plus exactly one tail per capture outcome \
         observed ({captured_delta}); any other count means a tail was written by something that \
         did not tick a capture outcome, or deleted by something that is not retention. Listing: \
         {handles:?}"
    );
}

// ────────────────────────────────────────────────────────────────────────────
// Pinned decision 3 (Task 4) — a FAILING tail put moves
// `writes_total{error, layer_tail}` and leaves the whole outcome family
// untouched.
//
// The asymmetry is the design, not an oversight: `live_edit_checkpoint_tail_total`
// has no `error` cell because an error is not a DECISION. The unit decided to
// capture; the adapter then failed. Recording that as an outcome would put
// adapter health into a family operators read as "what did the capture logic
// choose", and would double-count against the `error` cell that already exists
// on the artifact-level counter for exactly this.
//
// Driven through a `DelegatingCheckpointStore` newtype that fails puts whose
// `meta.kind == Tail` — narrow on purpose, so the assembler's own base puts (if
// a tick ever escaped the seam) still succeed and cannot be confused for the
// injected failure.
// ────────────────────────────────────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_failing_tail_put_moves_error_layer_tail_and_leaves_the_outcome_family_unchanged() {
    const REPLICA: &str = "tail-cadence-put-failure-replica";
    let deadline = Instant::now() + PUT_FAILURE_BUDGET;

    let tv = TestValkey::setup_with_pool_size(8).await;
    let metrics = make_metrics();
    let (object_id, session_id) = seed_wide_layer(&tv, "put-failure").await;
    let recording = Arc::new(TailAwareCheckpointStore::failing_tail_puts());
    let store: Arc<dyn CheckpointStore> = recording.clone();

    put_base_generation(
        &store,
        object_id,
        &session_id,
        &layer_name("put-failure"),
        CAPTURED_TAIL_SEED_SEQ,
    )
    .await;

    let writer = HotWriter::spawn(&tv, &session_id, "put-failure", object_id);
    wait_for_warm_up(&writer, deadline).await;

    let error_before = read_tail_writes_counter(&metrics, "error");
    let outcomes_before = gather_tail_outcome_cells(&metrics);

    let cancel = CancellationToken::new();
    let loop_handle = spawn_checkpoint_loop(
        unprovable_cut_client(&tv, object_id),
        tv.key_schema().clone(),
        Arc::clone(&store),
        loop_config(
            REPLICA,
            Duration::ZERO,
            Duration::ZERO,
            8,
            tail_config(true, 8),
        ),
        Some(Arc::clone(&metrics)),
        cancel.clone(),
    );

    let saw_error = poll_until(deadline, || async {
        read_tail_writes_counter(&metrics, "error") > error_before
    })
    .await;

    cancel_and_join(cancel, loop_handle).await;
    let commits = writer.report();
    writer.stop().await;

    let error_delta = read_tail_writes_counter(&metrics, "error") - error_before;
    let outcomes_after = gather_tail_outcome_cells(&metrics);
    let tail_put_attempts = recording.tail_put_attempts();
    let handles = store
        .list_layer_checkpoints(object_id)
        .await
        .expect("list_layer_checkpoints must succeed");

    assert!(
        saw_error && error_delta >= 1 && tail_put_attempts >= 1,
        "a tail put failing at the adapter must tick writes_total{{error, layer_tail}}; observed \
         +{error_delta} over {tail_put_attempts} attempted tail put(s) ({commits}). Zero attempts \
         means the capture never got as far as writing — the arm was not reached at all"
    );
    assert_eq!(
        outcomes_before, outcomes_after,
        "the ENTIRE live_edit_checkpoint_tail_total family must be unchanged across a drive whose \
         every capture failed at the adapter: that family has no error cell BY DESIGN (an error is \
         not a decision), and the unit never returned one either. before={outcomes_before:?} \
         after={outcomes_after:?}"
    );
    assert!(
        of_kind(&handles, CheckpointKind::Tail).is_empty(),
        "no tail artifact may exist when every tail put was refused by the adapter; listing \
         {handles:?}"
    );
}

// ────────────────────────────────────────────────────────────────────────────
// Pinned decision 4 (Task 4) — after a capture, `obj:hdr` carries BOTH cursor
// columns and `restore_seq_bump` is ABSENT, not present-and-zero.
//
// The two-pair stamp is adjudicated on `t2.dfh.57.tj`; its consumer is Task 3's
// fossil guard, which refuses a capture whenever `restore_seq_bump != 0` and
// therefore reads the column's presence as a signal. A stamped `0` would change
// that guard's input silently — the capture arm has not EARNED the "this
// generation retired the restore correction" claim, because only an assembly
// pays the `observed_commit_span` subtraction that justifies it.
//
// The seam skips every assembly, so no snapshot-arm base may land — asserted
// first, since a tick that wrote one would legitimately stamp all THREE pairs
// — and the header claims are then unconditional.
// ────────────────────────────────────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn after_a_capture_the_header_carries_both_cursor_columns_and_no_restore_seq_bump() {
    let tv = TestValkey::setup_with_pool_size(8).await;
    let metrics = make_metrics();
    let store: Arc<dyn CheckpointStore> = Arc::new(InMemoryCheckpointStore::new());
    let session_id = SessionStore::new(tv.client().clone(), tv.key_schema().clone())
        .create("tail-cursor-stamp-session", "/stage", "tail-cadence-user")
        .await
        .expect("session creation must succeed");

    let handle = drive_captured_tail(
        &tv,
        &store,
        &metrics,
        &session_id,
        "tail-cursor-stamp-layer",
        tail_config(true, 8),
    )
    .await;

    let header = read_header(&tv, handle.object_id).await;
    let handles = store
        .list_layer_checkpoints(handle.object_id)
        .await
        .expect("list_layer_checkpoints must succeed");
    assert!(
        !snapshot_base_landed(&handles, &[CAPTURED_TAIL_SEED_SEQ]),
        "the unprovable-cut seam skips every assembly of the driven layer, so no snapshot-arm \
         base may land; one here means the seam stopped firing and this test's header claims \
         no longer isolate the capture arm. Listing: {handles:?}"
    );

    let seq: u64 = header
        .get("last_checkpoint_seq")
        .unwrap_or_else(|| {
            panic!(
                "after a capture, obj:hdr MUST carry last_checkpoint_seq — a tail's to_seq is the \
                 newest sequence covered by a durable generation, and without the column the next \
                 tick re-captures the same range forever. Header: {header:?}"
            )
        })
        .parse()
        .expect("last_checkpoint_seq must be numeric");
    let ms: u64 = header
        .get("last_checkpoint_ms")
        .unwrap_or_else(|| {
            panic!(
                "after a capture, obj:hdr MUST carry last_checkpoint_ms — it rides the SAME HSET \
                 as the seq cursor precisely so the two can never disagree about whether a \
                 generation was persisted (b.yut). Header: {header:?}"
            )
        })
        .parse()
        .expect("last_checkpoint_ms must be numeric");

    assert!(
        seq >= handle.snap_seq,
        "last_checkpoint_seq={seq} must name at least the captured tail's to_seq={} — the tail's \
         upper, inclusive commit bound IS what the cursor advances to",
        handle.snap_seq
    );
    assert!(
        ms > 0,
        "last_checkpoint_ms must be a real wall-clock stamp, not 0 (which every reader treats as \
         'never checkpointed'); got {ms}"
    );
    assert!(
        !header.contains_key("restore_seq_bump"),
        "the capture arm writes TWO pairs, not three: restore_seq_bump must be ABSENT after a \
         capture — not present-and-zero. Clearing it is the claim 'this generation retired the \
         restore correction', and only an assembly earns that by paying the observed_commit_span \
         subtraction; a tail pays nothing of the sort. Task 3's fossil guard reads this column, so \
         a stamped 0 silently changes its input. Header: {header:?}; listing (no snapshot-arm base \
         present, so nothing else could have written it): {handles:?}"
    );
}

// ────────────────────────────────────────────────────────────────────────────
// The Epic's flag-off criterion, driven — same fixture, flag off, ZERO movement.
//
// Case C (`checkpointer_metrics_test.rs`) already states this for a loop that
// never reached the arm. This one is stronger in the way that matters: it drives
// the fixture that DOES reach `SkippedNotQuiesced` (asserted, via the unverified
// counter, so the claim is not vacuous) and shows the flag gate short-circuits
// before anything at all happens.
//
// Three independent statements of "nothing", because each alone has a hole:
//   - all NINE outcome cells at exactly 0 — a refusal tick would mean the
//     capture path RAN and declined, a different and wrong story from never
//     having run;
//   - no movement on any `writes_total{*, layer_tail}` cell;
//   - no store call ever carried a tail, from a recording wrapper. This is the
//     positive form: "no tail in the listing" is also what an adapter that
//     silently dropped the put would produce.
// ────────────────────────────────────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn flag_off_drives_the_same_fixture_with_zero_tail_movement() {
    const REPLICA: &str = "tail-cadence-flag-off-replica";
    let deadline = Instant::now() + FLAG_OFF_BUDGET;

    let tv = TestValkey::setup_with_pool_size(8).await;
    let metrics = make_metrics();
    let (object_id, session_id) = seed_wide_layer(&tv, "flag-off").await;
    let recording = Arc::new(TailAwareCheckpointStore::recording());
    let store: Arc<dyn CheckpointStore> = recording.clone();

    // The lineage root IS present: with the flag on, this fixture captures. The
    // only difference between this run and a capturing one is the flag.
    put_base_generation(
        &store,
        object_id,
        &session_id,
        &layer_name("flag-off"),
        CAPTURED_TAIL_SEED_SEQ,
    )
    .await;

    let writer = HotWriter::spawn(&tv, &session_id, "flag-off", object_id);
    wait_for_warm_up(&writer, deadline).await;

    let writes_before = tail_writes_row(&metrics);

    let cancel = CancellationToken::new();
    let loop_handle = spawn_checkpoint_loop(
        unprovable_cut_client(&tv, object_id),
        tv.key_schema().clone(),
        Arc::clone(&store),
        loop_config(
            REPLICA,
            Duration::ZERO,
            Duration::ZERO,
            8,
            // The operator opt-out, pinned explicitly on top of the shipped
            // knobs. NOT a bare `TailCaptureConfig::default()`: that default
            // became `true` on 2026-09-11 (b.3n1), and this whole test is the
            // claim that the flag gate short-circuits an arm the loop DID
            // reach, so inheriting it would invert the premise.
            TailCaptureConfig {
                enabled: false,
                ..TailCaptureConfig::default()
            },
        ),
        Some(Arc::clone(&metrics)),
        cancel.clone(),
    );

    // Non-vacuity gate: the arm under test must actually have been reached.
    let reached_the_arm = poll_until(deadline, || async {
        read_stable_cut_counter(&metrics, "unverified", REPLICA) >= 1
    })
    .await;

    cancel_and_join(cancel, loop_handle).await;
    let commits = writer.report();
    writer.stop().await;

    assert!(
        reached_the_arm,
        "this test's whole claim is that the flag gate short-circuits an arm the loop DID reach, \
         so the SkippedNotQuiesced arm must have fired at least once ({commits}). Without a rising \
         stable_cut_total{{outcome=\"unverified\"}} the zero-movement assertions below would pass \
         vacuously on a fixture whose seam never fired"
    );

    let cells = gather_tail_outcome_cells(&metrics);
    for outcome in TAIL_OUTCOMES {
        assert_eq!(
            cells.get(outcome).copied(),
            Some(0.0),
            "with the tail flag off, live_edit_checkpoint_tail_total{{outcome={outcome:?}}} must \
             be PRESENT (the pre-init contract) and read exactly 0. A refusal tick here would mean \
             the capture path ran and declined — a different, and wrong, story from never having \
             run. Cells: {cells:?}"
        );
    }
    let writes_after = tail_writes_row(&metrics);
    assert_eq!(
        writes_before, writes_after,
        "no writes_total{{*, kind=\"layer_tail\"}} cell may move on a flag-off drive: with the \
         flag off the arm is exactly what it was before the feature existed — the WARN, the two \
         metric ticks it always had, and the pending-age sample, with no store call and no Valkey \
         call in between"
    );

    let put_kinds = recording.layer_put_kinds();
    assert!(
        !put_kinds.contains(&CheckpointKind::Tail),
        "no store call may carry a tail on a flag-off drive. Recorded put kinds (in call order): \
         {put_kinds:?}. Asserted positively on the CALLS rather than on the listing, because an \
         empty listing is also what an adapter silently dropping the put would produce"
    );
    let handles = store
        .list_layer_checkpoints(object_id)
        .await
        .expect("list_layer_checkpoints must succeed");
    assert!(
        of_kind(&handles, CheckpointKind::Tail).is_empty(),
        "and no tail artifact may exist afterwards; listing {handles:?}"
    );
}

// ────────────────────────────────────────────────────────────────────────────
// Vocabulary guard — exactly ten outcome cells after a DRIVEN run, and
// `rebased` still zero.
//
// Case C guards the same family from the flag-off side (all ten present, all
// zero). This is the other side: a run where the family actually MOVED, so a
// regression that materialises an out-of-band `outcome` on first occurrence —
// invisible to any per-name probe — surfaces as an extra key.
//
// `rebased` is asserted zero separately and for a different reason, and that
// reason is a property of THIS DRIVE rather than of the codebase: the fixture
// runs at K = 8 and drives exactly one capture, so its lineage never comes
// within seven tails of the chain bound and no tick can reach the re-base at
// all. `rebased` is live and does move — `a_chain_at_k_tick_rebases_and_moves_no
// _layer_tail_writes_cell` above is where it moves — so a non-zero reading here
// says the loop re-based a layer whose chain was nowhere near K, i.e. the
// trigger is firing off something other than the bound.
//
// Keep the two halves of this test together. The vocabulary guard is the
// out-of-band-outcome regression that no per-name probe can see, and it needs a
// run in which the family actually MOVED; the `rebased` probe rides that same
// run for free.
// ────────────────────────────────────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_driven_run_keeps_ten_tail_outcome_cells_and_rebased_at_zero() {
    let tv = TestValkey::setup_with_pool_size(8).await;
    let metrics = make_metrics();
    let store: Arc<dyn CheckpointStore> = Arc::new(InMemoryCheckpointStore::new());
    let session_id = SessionStore::new(tv.client().clone(), tv.key_schema().clone())
        .create("tail-vocabulary-session", "/stage", "tail-cadence-user")
        .await
        .expect("session creation must succeed");

    let handle = drive_captured_tail(
        &tv,
        &store,
        &metrics,
        &session_id,
        "tail-vocabulary-layer",
        tail_config(true, 8),
    )
    .await;

    let cells = gather_tail_outcome_cells(&metrics);
    let observed: HashSet<String> = cells.keys().cloned().collect();
    let expected: HashSet<String> = TAIL_OUTCOMES.iter().map(|o| (*o).to_string()).collect();
    assert_eq!(
        observed, expected,
        "after a driven run the live_edit_checkpoint_tail_total outcome set must STILL be exactly \
         the ten pinned values. An extra key is an out-of-band outcome materialised on its first \
         occurrence — the regression a per-name probe cannot see. Cells: {cells:?}"
    );
    assert_eq!(
        cells.get("rebased").copied(),
        Some(0.0),
        "`rebased` must stay 0 on THIS drive: it runs at K = 8 and lands one capture, so the \
         chain never approaches the bound and no tick can reach the SR-4.1 re-base. The cell is \
         live and does move elsewhere — see \
         a_chain_at_k_tick_rebases_and_moves_no_layer_tail_writes_cell — so a non-zero reading \
         here means the collapse trigger fired off something other than the chain bound. Cells: \
         {cells:?}"
    );
    assert!(
        tail_captures_landed(&cells) >= 1.0,
        "non-vacuity: the drive must actually have moved the family, or this guard is just Case \
         C's pre-init check again. The capture family, not the bare `captured` cell — which of \
         the two spellings a landed capture takes is a runner-speed artifact, not this guard's \
         claim. Captured tail: {handle:?}. Cells: {cells:?}"
    );

    // The sibling family, guarded the same way and for the same reason: a
    // capture writes through `checkpoint_writes_total` too, and a regression
    // that spelled its `kind` differently (or invented a `result`) would open a
    // NEW series alongside the pre-init zero one, leaving every dashboard keyed
    // off the wrong slice. Both vocabularies stay LITERAL here — deriving either
    // side from the production consts would make the comparison pass by
    // construction.
    let pairs = gather_writes_label_pairs(&metrics);
    let expected_pairs: HashSet<(String, String)> = WRITE_RESULTS
        .iter()
        .flat_map(|r| {
            ["layer", "session", "counters_sidecar", "layer_tail"]
                .iter()
                .map(move |k| ((*r).to_string(), (*k).to_string()))
        })
        .collect();
    assert_eq!(
        pairs, expected_pairs,
        "after a driven capture the live_edit_checkpoint_writes_total label set must still be \
         exactly {{success, error, skipped}} × {{layer, session, counters_sidecar, layer_tail}}; \
         got {pairs:?}"
    );
}

// ────────────────────────────────────────────────────────────────────────────
// A tail is retained with the lineage unit it belongs to — driven through the
// loop at `retain_generations = 1`.
//
// Task 4's unit tests cover the grouping directly; the Epic's criterion asks
// for it driven, and the driven form is what proves the prune the CAPTURE arm
// runs is the same lineage-aware one (SR-9.1 requires the fallback to inherit
// that discipline rather than grow a parallel copy — and before the tail path
// existed, a permanently-hot layer never reached the snapshot arm and so never
// pruned at all).
//
// The fixture is deliberately adversarial: at `retain = 1` a position-based
// `split_off(1)` over the mixed listing would keep the newest generation and
// delete EVERYTHING below it — including both hand-put tails and the base they
// are rooted at, i.e. total unrestorable loss of the layer's checkpoint produced
// by the one path whose job is bounding storage safely.
//
// Post-Epic-5 the tails survive for a stated reason rather than by exemption:
// they sit ABOVE the newer base in the newest-first listing, so the adjacency
// scan closes them into that base's lineage unit, and that unit is the one
// `retain = 1` keeps. The lone older base below them is its own chain-0 unit
// and is the whole of the delete set. A tail whose unit fell outside the window
// WOULD be deleted here — "a tail is never pruned" was the interim window's
// property, not the design's.
// ────────────────────────────────────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn tails_survive_the_retention_prune_at_retain_generations_one() {
    const REPLICA: &str = "tail-cadence-prune-replica";
    /// The newer base, planted above the lineage root so the prune has an older
    /// base to actually delete. Comfortably below the sequence the warm-up wait
    /// guarantees.
    const SECOND_BASE: u64 = CAPTURED_TAIL_SEED_SEQ + 10;
    /// The two hand-put tails' upper bounds. Contiguous from `SECOND_BASE`, so
    /// the lineage the loop finds is well-formed and the capture window opens
    /// just above the newer of them.
    const TAIL_ONE_TO: u64 = CAPTURED_TAIL_SEED_SEQ + 15;
    const TAIL_TWO_TO: u64 = CAPTURED_TAIL_SEED_SEQ + 20;
    /// Header sequence the writer must reach before the tails are planted: a
    /// hand-put tail claiming coverage past the layer's own newest commit would
    /// be a fixture nothing in production can produce.
    const PLANT_FLOOR: u64 = TAIL_TWO_TO + 5;
    let deadline = Instant::now() + PRUNE_BUDGET;

    let tv = TestValkey::setup_with_pool_size(8).await;
    let metrics = make_metrics();
    let (object_id, session_id) = seed_wide_layer(&tv, "prune").await;
    let store: Arc<dyn CheckpointStore> = Arc::new(InMemoryCheckpointStore::new());
    let layer = layer_name("prune");

    let root = put_base_generation(
        &store,
        object_id,
        &session_id,
        &layer,
        CAPTURED_TAIL_SEED_SEQ,
    )
    .await;

    let writer = HotWriter::spawn(&tv, &session_id, "prune", object_id);
    wait_for_warm_up(&writer, deadline).await;
    let past_the_floor = poll_until(deadline, || async {
        read_header_u64(&tv, object_id, "seq")
            .await
            .is_some_and(|seq| seq >= PLANT_FLOOR)
    })
    .await;
    assert!(
        past_the_floor,
        "the writer must carry the layer past sequence {PLANT_FLOOR} before the tails are planted \
         ({})",
        writer.report()
    );

    put_base_generation(&store, object_id, &session_id, &layer, SECOND_BASE).await;
    put_tail_generation(
        &store,
        object_id,
        &session_id,
        &layer,
        SECOND_BASE,
        SECOND_BASE,
        TAIL_ONE_TO,
    )
    .await;
    put_tail_generation(
        &store,
        object_id,
        &session_id,
        &layer,
        SECOND_BASE,
        TAIL_ONE_TO,
        TAIL_TWO_TO,
    )
    .await;

    let cancel = CancellationToken::new();
    let loop_handle = spawn_checkpoint_loop(
        unprovable_cut_client(&tv, object_id),
        tv.key_schema().clone(),
        Arc::clone(&store),
        loop_config(
            REPLICA,
            Duration::ZERO,
            Duration::ZERO,
            // The whole point: ONE lineage unit retained — which is one base
            // plus every tail grouped above it, not one artifact.
            1,
            tail_config(true, 8),
        ),
        Some(Arc::clone(&metrics)),
        cancel.clone(),
    );

    // A capture is what runs the prune, so wait for the capture FIRST and then
    // for the prune it triggers — the two are separate store round-trips and
    // listing between them is a race, not a failure.
    let captured = wait_for_tail_above(&store, object_id, TAIL_TWO_TO, deadline).await;
    let pruned = poll_until(deadline, || async {
        let handles = store
            .list_layer_checkpoints(object_id)
            .await
            .expect("list_layer_checkpoints must succeed");
        of_kind(&handles, CheckpointKind::Base).len() <= 1
    })
    .await;

    cancel_and_join(cancel, loop_handle).await;
    let commits = writer.report();
    writer.stop().await;

    let handles = store
        .list_layer_checkpoints(object_id)
        .await
        .expect("list_layer_checkpoints must succeed");
    let bases = of_kind(&handles, CheckpointKind::Base);
    let tails = of_kind(&handles, CheckpointKind::Tail);
    let tail_seqs: Vec<u64> = tails.iter().map(|h| h.snap_seq).collect();

    // At retain = 1 a snapshot-arm base landing above the tails would root the
    // newest lineage unit itself and get every tail below it pruned, so the
    // survival claim below rests on its absence: asserted first (module
    // header, "No snapshot base lands").
    assert!(
        !snapshot_base_landed(&handles, &[CAPTURED_TAIL_SEED_SEQ, SECOND_BASE]),
        "the unprovable-cut seam skips every assembly of the driven layer, so the only bases may \
         be the two planted at {CAPTURED_TAIL_SEED_SEQ} and {SECOND_BASE}; another one means the \
         seam stopped firing and the tails' survival below is no longer this test's claim. \
         Listing: {handles:?}"
    );

    let captured = captured.unwrap_or_else(|| {
        panic!(
            "the loop must capture a tail above {TAIL_TWO_TO} within {PRUNE_BUDGET:?} — the prune \
             under test runs only after a successful capture ({commits}). Listing: {handles:?}"
        )
    });
    assert!(
        pruned,
        "with retain_generations = 1 the prune that follows the capture must reduce the layer to a \
         single BASE; it left {} ({commits}). Listing: {handles:?}",
        bases.len()
    );
    assert_eq!(
        bases.len(),
        1,
        "one lineage unit is retained at retain = 1, and a unit is rooted at exactly one base, \
         so exactly one base survives. Listing: {handles:?}"
    );
    assert_eq!(
        bases[0].snap_seq, SECOND_BASE,
        "the surviving base must be the NEWEST one ({SECOND_BASE}), not the pruned root at {}. \
         Surviving: {:?}",
        root.snap_seq, bases[0]
    );
    assert!(
        tail_seqs.contains(&TAIL_ONE_TO)
            && tail_seqs.contains(&TAIL_TWO_TO)
            && tail_seqs.contains(&captured.snap_seq),
        "every TAIL must survive the prune — both hand-put ones ({TAIL_ONE_TO}, {TAIL_TWO_TO}) and \
         the captured one ({}) — even though a base OLDER than them was deleted in the same pass. \
         All three sit above the newer base, so they are MEMBERS of the lineage unit rooted at it, \
         and that unit is the one `retain = 1` keeps: a unit is retained whole or pruned whole. \
         Tails never counted toward the quota (the unit does) — but they are deletable, so this \
         assertion is now about which unit they belong to and not about an exemption. A \
         position-based split over the mixed listing at retain = 1 would have deleted all three \
         plus the base they are rooted at, which is total unrestorable loss of the layer's \
         checkpoint. Surviving tails: {tail_seqs:?}. Listing: {handles:?}",
        captured.snap_seq
    );
}
