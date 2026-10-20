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

//! The Epic's own acceptance criterion, end to end — Subtask
//! `t3.dfh.yj.u3.ue` under Task `t2.dfh.yj.u3`, Epic `t1.dfh.yj`.
//!
//! ## The claim
//!
//! A `full_valkey_wipe` under a surviving process restores a **tail-lineage**
//! layer to exact state through a production gRPC handler, and a restored
//! **never-quiescent** layer's next tail capture SUCCEEDS through the real
//! periodic loop rather than refusing — including the bootstrap-failure arm,
//! where capture refuses observably until a base lands and then converges.
//! Both SR-3.6 stream shapes are covered.
//!
//! Four tests, one per arm:
//!
//! 1. [`full_valkey_wipe_restores_a_tail_lineage_layer_to_exact_state`] — the
//!    whole-keyspace wipe, one `handle_update_layer` dispatch, the two recovery
//!    hops in series, the composed-head cursor stamp, the faithful-column
//!    oracle, the external values, and the counter re-seed.
//! 2. [`a_restored_never_quiescent_layer_captures_its_next_tail_instead_of_refusing`]
//!    — the interesting half: the SR-3.6 re-base lands, the fossil guard is
//!    retired, and the real periodic loop captures the layer's next tail.
//! 3. [`a_failed_rebase_refuses_capture_observably_then_converges_once_a_base_lands`]
//!    — the SR-3.6 re-base PUT fails, capture refuses `refused_seq_gap` on every
//!    eligible tick, and the layer converges once a verified snapshot lands.
//!    (Its name's "rebase" is SR-3.6's, per the convention below.)
//! 4. [`a_surviving_stream_restore_folds_the_fossils_and_captures_its_next_tail`]
//!    — the surviving-stream SR-3.6 shape.
//!
//! ## "Re-base" in this file always means SR-3.6's post-restore one
//!
//! The service has TWO re-bases and this file only ever exercises one of them.
//! Every unqualified "re-base" below — every "re-based cursor", every
//! `wire_restore_rebase`, every `rebase_puts` — is **SR-3.6's post-restore
//! re-base**: the fresh `Base` that `BaseBootstrap::spawn_restore_rebase` PUTs at
//! the seeded sequence after a DR restore, retiring `restore_seq_bump`.
//!
//! The other one is the **SR-4.1 chain-collapse re-base**
//! (`src/checkpoint/rebase.rs`), which the checkpoint tick runs when a hot
//! layer's tail chain reaches K and which ticks
//! `live_edit_checkpoint_tail_total{outcome="rebased"}`. It is structurally out
//! of reach here: these drives run at the stock `LIVE_EDIT_CHECKPOINT_TAIL_MAX_CHAIN`
//! (`TailCaptureConfig::from_config`) and capture at most a tail or two above a
//! fresh base, so no drive comes near the bound. Nothing in this file asserts
//! `rebased`, and nothing in it should — the chain-collapse lane owns that cell.
//!
//! The two are kept apart on purpose, and the separation is a live invariant
//! rather than a naming preference: SR-3.6's re-base is a WRITE and must not tick
//! `rebased`. If you find yourself writing "the re-base" in a new comment here,
//! it is SR-3.6's; if you mean the other one, name it in full.
//!
//! ## Why the second half is the interesting half
//!
//! Post-b.8y5 the guard band is unconditional, so every restore leaves
//! `seeded_seq > snap_seq`. Without Task 4's SR-3.6 gap re-base a restored
//! never-quiescent layer refuses capture on every subsequent skipped tick
//! (`TailCaptureRefusal::PostRestoreUnrebased`, `src/checkpoint/tail_capture.rs`
//! step 3 of `capture_inner`), i.e. it has no DR coverage beyond the generation
//! it was restored from until an assembly lands a base. Tests 2 and 3 are the
//! evidence that the loop actually recovers, and that its failure mode is
//! observable rather than silent.
//!
//! ## The drives' loop runs with the unprovable-cut seam armed
//!
//! "Never-quiescent" here means a layer whose every assembly persists nothing,
//! so its ticks fall back to tail capture. Since every read proves its cut (the
//! atomic read, and the fenced read above the atomic-read limits; Spec SDD
//! SR-4.1, SR-11.5) only a header-sequence regression does that, at any layer
//! size: without one the tick writes a base however hot the layer is, and the
//! capture path these arms are about is never reached. So [`spawn_tail_loop`]
//! runs the drives' loop on `unprovable_cut_client` (`helpers.rs`), the
//! test-only seam that stages the regression for the drives' layer, and Test
//! 3's phase 3 disarms it to let a verified base land. The seam exists only
//! under `test-checkpoint-fault-injection`, so Tests 2–4 are compiled only
//! under that feature (the CI `test:` job's feature set includes it); Test 1
//! runs in every shape.
//!
//! ## The refusal shape, pinned — `refused_seq_gap` on BOTH stream shapes
//!
//! The landed refusal on a failed or shed re-base is
//! `TailCaptureRefusal::PostRestoreUnrebased` → outcome label
//! `refused_seq_gap`, on the fresh-stream shape as much as the surviving-stream
//! one. The guard runs **before** the stream read, so the classifier's
//! `TailRefusal::Trimmed` is unreachable while `restore_seq_bump != 0` — the
//! spec child's Task-4-PM clarification item 2, restated by this Subtask's PM
//! condition C1/B1. The earlier "…or the classifier's `Trimmed` on a fresh
//! stream" wording is STRUCK; do not re-add that branch.
//!
//! ## Boundaries against the two narrower lanes
//!
//! - **`t3.dfh.yj.4f.gw` (`restore_rebase_test.rs`)** drives `capture_layer_tail`
//!   DIRECTLY as a narrow deterministic instrument on `wipe_layer_state`
//!   fixtures, and owns the re-base's cursor-column and PUT-failure stamping
//!   assertions. **This file never calls that instrument.** It drives the real
//!   periodic loop after a whole-keyspace wipe and a production-handler restore,
//!   which is the composition claim (session hop + lineage restore + re-base +
//!   next capture + counter re-seed) that no narrower test makes. The cost of
//!   that fidelity is that the refusal *variant* is not directly observable
//!   here; see [`assert_landed_refusal_label_is_seq_gap`] for how it is pinned.
//! - **`t3.dfh.yj.b7.x7` (`e2e_composed_restore_test.rs`)** owns single-restore
//!   hook wiring on a NON-wiped fixture (`wipe_layer_state`, `handle_stat_layer`,
//!   the four cursor columns of one restore). This file does not restate those
//!   assertions; it asserts what only a whole-keyspace wipe can show.
//!
//! ## Recorded mutation verification
//!
//! The Epic's headline claim is easy to assert vacuously, so it was
//! mutation-checked in BOTH directions and the results recorded here (each
//! mutation was reverted; neither is committed).
//!
//! - **Re-base removed.** Pointing Test 2 at
//!   [`setup_tail_harness_without_restore_rebase`] — i.e. clearing
//!   `CheckpointHarnessOptions::wire_restore_rebase` and nothing else — fails the
//!   test at `restore_seq_bump == 0` with `left: 1000, right: 0`: the restored
//!   layer keeps its guard band and the post-restore fossil guard would refuse on
//!   every subsequent tick. That is the DR-coverage loss the arm exists to
//!   disprove. Test 3 is the same claim from the other side, driven all the way
//!   through the loop: with the re-base's PUT failed, `refused_seq_gap` rises and
//!   `captured` stays flat.
//! - **Opt-out removed.** Pointing Test 1 at [`setup_tail_harness`] instead —
//!   i.e. restoring the default `wire_restore_rebase = true` — fails it at
//!   `last_checkpoint_seq` with `left: 1006, right: 6`: the re-based `seeded_seq`
//!   (head + the 1000 guard band) where the composed head `S` is claimed. The
//!   following `restore_seq_bump == margin` line would have failed too, the
//!   re-base having retired it. Test 1's opt-out is therefore the reason no
//!   re-base ran, not a coincidence, and its cursor-stamp claim is load-bearing.
//!   Both mutations run under the SAME SR-7.1 three-way gate assertion
//!   ([`setup_tail_harness_with_rebase_posture`]), so neither result can be
//!   explained by a gate that was closed anyway.
//!
//! Two drafting hazards are recorded here because both were live and both are
//! easy to re-introduce:
//!
//! - An earlier draft of Tests 2–4 polled "a `Tail` exists in the listing" and
//!   was green while proving nothing — every fixture here HAND-PUTS a tail as
//!   its lineage head, so that predicate holds from the first poll. The polls
//!   are metric-based for that reason; see [`poll_tail_outcome_rose`].
//! - An earlier draft then asserted on the NEWEST captured tail's `from_seq`,
//!   which flaked at roughly one run in eight: a drive that wins more than one
//!   eligible tick captures a CHAIN, and SR-2.1 makes tail *n+1* root at tail
//!   *n*'s head rather than at the cursor. See
//!   [`assert_captured_chain_roots_at`] for the claim that holds however many
//!   tails land.
//!
//! ## Task 5 non-regression
//!
//! Every harness here runs with the tail flag ON. **No assertion in this file
//! depends on a stable-cut attempt count** — the arms assert capture outcomes
//! and state only. Nothing reports a stable-cut attempt count any more — a
//! tick makes one proven read — so there is no attempt count at all.
//!
//! ## One wiring note about the shared harness
//!
//! The stock harness now ships the production hook wiring — the real
//! `ValkeyExternalValueStore` and the harness's single `BaseBootstrap`, both
//! threaded into `svc.recovery` the way `src/main.rs` does — so every test here
//! takes its hook from `setup_checkpoint_harness_with_options` and none builds
//! one by hand. Arm 1 alone uses the named opt-out,
//! `CheckpointHarnessOptions::wire_restore_rebase = false` (see
//! [`setup_tail_harness_without_restore_rebase`]), because its claim is the
//! RESTORE's own cursor stamp and a successful re-base overwrites it. That the
//! stock wiring really is production's is asserted structurally in
//! `harness_parity_test.rs`; this file does not re-argue it.
//!
//! ## Why Arm 4 is not literally under `full_valkey_wipe`
//!
//! `full_valkey_wipe` is a SCAN + UNLINK over `tv.prefix()*`, so it necessarily
//! destroys `le:{oid}:obj:stream` — the fresh-stream shape is the only one it
//! can produce. The Task says "both shapes **where feasible**", and the feasible
//! form of the surviving-stream shape is a partial loss:
//! `helpers::wipe_layer_state_preserving_stream` over the same composed lineage,
//! with real post-tail commits left in the stream as abandoned-timeline fossils.
//! Test 4 is that.
//!
//! ## Empty children lists
//!
//! An empty children list has no Valkey representation (`LayerStore::write_object`
//! skips the RPUSH/SADD on `child_ids.is_empty()`, and the `SetChildren` arm DELs
//! on the same fork), so a `ComposedLineage::state` compared directly against a
//! re-read snapshot can diverge on `{parent: []}` entries. This file builds NO
//! such oracle: every comparison here is post-write-back state (handler-driven
//! restore, then `read_object_state_for_test`) against a pre-wipe re-read, so both
//! sides went through the identical fork and the asymmetry cannot arise.
//!
//! ## Synchronization + budget
//!
//! No `tokio::time::sleep` is used as synchronization anywhere: every wait is a
//! bounded poll on observable state, under a NAMED module constant, and every
//! drain gets ONE deadline rather than one per message.
//!
//! | Wait | Bound | What a timeout means |
//! |------|-------|----------------------|
//! | `wait_for_session_checkpoint{,_quiesced}` | [`STORE_DEADLINE_SECS`] (10 s, clamped up to the helpers' 30 s load-tolerant floor) | the session-checkpoint writer never landed a blob / never drained |
//! | `wait_for_checkpoint_at_least` (the lineage-root base) | [`STORE_DEADLINE_SECS`], same clamp | the periodic loop never assembled a base at or above the fixture's committed sequence |
//! | `wait_for_writer_idle` (re-base / bootstrap drain) | [`STORE_DEADLINE_SECS`], same clamp | a `BaseBootstrap` task never completed, or was spawned onto a different tracker |
//! | loop drain after `cancel()` | [`LOOP_DRAIN_BUDGET`] (5 s) | a wedged loop; bounded so it fails promptly instead of hanging the binary |
//! | [`HotWriter::warm_up`] | [`WARM_UP_DEADLINE`] (20 s) | the writer never committed — the panic prints its own report |
//! | "a tail was captured" poll | [`CAPTURE_DEADLINE`] (25 s) | the loop never reached the capture arm; the panic prints the listing and the whole outcome-cell map |
//! | "capture refused" poll (Test 3) | [`REFUSAL_DEADLINE`] (20 s) | the loop never reached an eligible tick for this layer |
//! | "a base landed and cleared the bump" poll (Test 3) | [`CONVERGENCE_DEADLINE`] (20 s) | the healed store never took a verified snapshot |
//!
//! Every bound above is a FAILURE bound: on the happy path each poll returns in
//! one or two 1 s ticks. A single timing-out wait panics with its own
//! diagnostic, so the practical time-to-diagnostic is one bound, not their sum.
//! The largest single test (Test 3, three sequential drives) sums to ~65 s of
//! failure budget plus fixture setup — inside `.config/nextest.toml`'s 120 s
//! hard kill, which prints nothing at all when it fires. Measured: over ten
//! consecutive clean runs the four tests land in 3–8 s each, the whole file in
//! 6–8 s wall clock.
//!
//! Sandbox env note: `TestValkey::setup` reads `TEST_VALKEY_URL`; under the
//! project sandbox that variable is preset (CLAUDE.md).

// Tests 2–4 are compiled only under `test-checkpoint-fault-injection` (the
// unprovable-cut seam; see the module header), so without the feature the
// fixtures only they use read as dead. Test 1 runs in every shape.
#![cfg_attr(
    not(feature = "test-checkpoint-fault-injection"),
    allow(dead_code, unused_imports)
)]

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use bytes::Bytes;
use fred::interfaces::{HashesInterface, KeysInterface, StreamsInterface};
use tokio::task::JoinHandle;
use tokio::time::Instant;
use tokio_util::sync::CancellationToken;
use tonic::Request;

use live_edit::checkpoint::adapters::InMemoryCheckpointStore;
use live_edit::checkpoint::counters::CountersSidecar;
use live_edit::checkpoint::tail_capture::{TailCaptureDecision, TailCaptureRefusal};
use live_edit::checkpoint::{
    decode_tail_envelope, spawn_checkpoint_loop, CheckpointAdapter, CheckpointBlob,
    CheckpointError, CheckpointHandle, CheckpointKind, CheckpointLoopConfig, CheckpointMeta,
    CheckpointStore, LayerArtifact, TailCaptureConfig,
};
use live_edit::grpc::layer::{handle_create_layer, handle_stat_layer, handle_update_layer};
use live_edit::grpc::proto::{CreateSessionRequest, ListSessionsRequest, StatLayerRequest};
use live_edit::grpc::session::{handle_create_session, handle_list_sessions};
use live_edit::grpc::BearerToken;
use live_edit::observability::metrics::{SharedMetrics, CHECKPOINT_RESTORE_LINEAGE_OUTCOMES};
use live_edit::storage::diff_read::ObjectSnapshot;
use live_edit::storage::external_values::ext_content_hash;
use live_edit::storage::keys::sha256_to_hex;
use live_edit::storage::layer::LayerStore;
use live_edit::storage::test_support::read_object_state_for_test;
use live_edit::storage::test_support::TestValkey;
use oodf_ffi::{
    CommandDiffChangeSection, OwnedDelta, SetField, StructCommand, TimeSample as OodfTimeSample,
};

use super::helpers::*;
use crate::common::owned_delta::delta_to_ompf_bytes;

// ─────────────────────────────────────────────────────────────────────────────
// Named waits and fixture constants — see the module header's budget table
// ─────────────────────────────────────────────────────────────────────────────

/// Checkpoint-loop tick cadence. `1` is the value every e2e recovery file in
/// this module uses: the harness sets `interval = quiesce = 0`, so a
/// just-written layer is due on the very next tick.
const TICK_SECS: u64 = 1;

/// Deadline handed to every `helpers::wait_for_*` call. Those helpers clamp it
/// up to their own 30 s load-tolerant floor, so this is the *lower* bound on
/// how long a contended Valkey may take before the wait is called a failure.
const STORE_DEADLINE_SECS: u64 = 10;

/// Bound on the post-`cancel()` drain of a checkpoint loop's `JoinHandle`. The
/// loop returns at its next `select!` on the token, so this is never the wait
/// actually observed — it exists so a wedged loop fails the test promptly
/// instead of hanging the binary.
const LOOP_DRAIN_BUDGET: Duration = Duration::from_secs(5);

/// Ceiling on [`HotWriter::warm_up`].
const WARM_UP_DEADLINE: Duration = Duration::from_secs(20);

/// Ceiling on a "the loop captured a tail" poll.
const CAPTURE_DEADLINE: Duration = Duration::from_secs(25);

/// Ceiling on a "the loop refused a capture" poll (Test 3).
const REFUSAL_DEADLINE: Duration = Duration::from_secs(20);

/// Ceiling on the "a verified base landed and retired the bump" poll (Test 3).
const CONVERGENCE_DEADLINE: Duration = Duration::from_secs(20);

/// Poll cadence for every bounded poll in this file. Not a synchronization
/// sleep: the predicate is observable state, and the cadence only bounds how
/// hard the loop hammers Valkey while it waits.
const POLL_INTERVAL: Duration = Duration::from_millis(50);

/// Commits the hot writer must land before the periodic loop is allowed to run.
///
/// A state precondition polled off the writer's own counter, not a timing hack:
/// a loop that meets a layer with no commit above its cursor finds it not due
/// (the skip-if-no-commits gate), and the arm under test is never reached.
const WARM_UP_COMMITS: u64 = 20;

/// Sections seeded into the fixture layer.
///
/// 120 is `helpers::drive_captured_tail`'s number. Which arm a tick takes is
/// the unprovable-cut seam's ([`spawn_tail_loop`], see the module header), not
/// the width's; the width makes Test 1's exact-state oracle non-trivial.
const WIDE_SECTIONS: u64 = 120;

/// Stream retention for the hot writer's own `LayerStore`.
///
/// Deliberately far above the harness's 4096. The writer commits continuously
/// for as long as a drive runs; if `MAXLEN` trimming ever reached past the
/// capture window's floor (`{seeded_seq+1}-0`) the capture would refuse
/// `refused_trimmed` — a fixture artifact indistinguishable, at the assertion,
/// from the wiring being broken.
const WRITER_STREAM_MAXLEN: u64 = 1_000_000;

/// Retention quota for the loops this file spawns. Well above anything a test
/// plants: the landed prune counts LINEAGE UNITS (one per base, with the tails
/// above it), and eight is far past any unit count these fixtures reach — a low
/// quota would be one more variable in a fixture whose failures are already
/// hard to read, and post-Epic-5 an over-run quota can delete a tail as well as
/// a base. `drive_captured_tail` uses the same number for the same reason.
const DRIVE_RETAIN_GENERATIONS: usize = 8;

/// The section every hot-writer commit and every pre-base commit writes into.
/// Section `1` exists in every layer this file creates.
const ROOT_SECTION_ID: u64 = 1;

/// First section id minted by a tail-range commit. Each such commit creates its
/// own section, so the composed replay has structural work to do and not only
/// field writes.
const TAIL_SECTION_BASE_ID: u64 = 100_000;

/// Authoring rank stamped on every `SetField` this file's fixture builds.
/// Only needs to be stable; nothing here depends on a particular value.
const AUTHORING_RANK: u64 = 0;

/// Byte length of the external value the tail carries. Large enough to be
/// unambiguously an out-of-band OMPF part, small enough to stay free.
const EXT_BLOB_LEN: usize = 4 * 1024;

/// The one-based OMPF part index the ext-carrying `SetField` points at. Part 0
/// is the OODF delta; `delta_to_ompf_bytes(delta, &[blob])` puts `blob` at
/// Part 1.
const EXT_VALUE_INDEX: u32 = 1;

/// Commits driven inside the tail's `(S0, S]` span.
///
/// Three, not one: the composed replay must carry structural commands, inline
/// fields, a time sample AND an external value, and a single-record envelope
/// makes an off-by-one in the record/span relationship unobservable.
const TAIL_RANGE_COMMITS: u64 = 3;

/// Post-tail commits Test 4 leaves in the surviving stream as abandoned-timeline
/// fossils — real commits no durable generation covers.
const FOSSIL_COMMITS: u64 = 3;

const SESSION_BASE_PATH: &str = "/stage/full-wipe-composed";
const CLIENT_ID: &str = "full-wipe-composed-client";
const WRITER_CLIENT_ID: &str = "full-wipe-composed-hot-writer";

/// The `key_name` of the marker field Test 1's post-wipe `UpdateLayer` writes.
///
/// Excluded from the faithful-column comparison **by name**: it is the commit
/// the recovery branch re-issued, so it exists in the restored state and not in
/// the pre-wipe oracle. Its presence is itself asserted — a restore that
/// rebuilt the checkpoint but skipped the re-issue would show every oracle
/// column and no marker.
const POST_RECOVERY_MARKER_KEY: &str = "post-full-wipe-marker";

// ─────────────────────────────────────────────────────────────────────────────
// Payload builders
// ─────────────────────────────────────────────────────────────────────────────

/// CreateLayer payload: [`WIDE_SECTIONS`] sections under the root plus one
/// inline field each. See [`WIDE_SECTIONS`] for why the width is load-bearing.
fn wide_initial_delta() -> OwnedDelta {
    let mut struct_commands = Vec::with_capacity(WIDE_SECTIONS as usize);
    let mut set_fields = Vec::with_capacity(WIDE_SECTIONS as usize);
    for section_id in 1..=WIDE_SECTIONS {
        struct_commands.push(StructCommand::DiffChangeSection(CommandDiffChangeSection {
            parent_id: 0,
            section_id,
            section_name: format!("full-wipe-composed-section-{section_id}"),
            section_type: 1,
            section_order: (section_id - 1) as u32,
        }));
        set_fields.push(SetField {
            section_id,
            key_name: "initial-field".to_string(),
            value_or_ext_hash: vec![0xAA, (section_id & 0xFF) as u8, 0xCC],
            ext_value_size: 0,
            ext_value_index: 0,
            set_order: AUTHORING_RANK,
        });
    }
    OwnedDelta {
        is_diff: true,
        base_version: 0,
        source_format: "usd".to_string(),
        struct_commands,
        set_fields,
        ..Default::default()
    }
}

/// A pre-base commit: one new inline field on the root section.
///
/// `base_version: 1` (any non-zero) marks this an INCREMENTAL diff. With
/// `base_version == 0` and `is_diff == true` the domain engine takes the reset
/// branch and wipes prior state, which would empty the fixture.
fn pre_base_commit_delta(index: u64) -> OwnedDelta {
    OwnedDelta {
        is_diff: true,
        base_version: 1,
        set_fields: vec![SetField {
            section_id: ROOT_SECTION_ID,
            key_name: format!("pre-base-field-{index}"),
            value_or_ext_hash: vec![0x10 + index as u8, 0x11, 0x12],
            ext_value_size: 0,
            ext_value_index: 0,
            set_order: AUTHORING_RANK,
        }],
        ..Default::default()
    }
}

/// A tail-range commit: a NEW section under the root, one inline field inside
/// it, and one time sample.
///
/// All three record kinds in one delta on purpose. A composed restore that
/// replayed field writes but dropped structural commands (or vice versa) would
/// still satisfy a fields-only oracle; it cannot satisfy this one.
fn tail_commit_delta(index: u64) -> OwnedDelta {
    let section_id = TAIL_SECTION_BASE_ID + index;
    OwnedDelta {
        is_diff: true,
        base_version: 1,
        struct_commands: vec![StructCommand::DiffChangeSection(CommandDiffChangeSection {
            parent_id: ROOT_SECTION_ID,
            section_id,
            section_name: format!("tail-section-{index}"),
            section_type: 2,
            section_order: index as u32,
        })],
        set_fields: vec![SetField {
            section_id,
            key_name: format!("tail-field-{index}"),
            value_or_ext_hash: vec![0xC0 + index as u8, 0xDE, index as u8],
            ext_value_size: 0,
            ext_value_index: 0,
            set_order: AUTHORING_RANK,
        }],
        time_samples: vec![OodfTimeSample {
            section_id,
            time: index as f64,
            value_or_ext_hash: vec![0xF0 + index as u8],
            ext_value_size: 0,
            ext_value_index: 0,
        }],
        ..Default::default()
    }
}

/// The tail-range commit that carries an EXTERNAL VALUE.
///
/// This is what puts the value bytes into the tail record's OMPF Parts 1..N,
/// which is the only thing SR-3.4's restore-time pre-store has to work with:
/// `compose_from_artifacts` collects `(hash, Bytes)` pairs off exactly these
/// parts, and the pre-store writes that list through
/// `ExternalValueStore::store_values` before the single `write_object`.
fn tail_ext_value_delta(hash: &[u8; 32], blob_len: usize) -> OwnedDelta {
    OwnedDelta {
        is_diff: true,
        base_version: 1,
        set_fields: vec![SetField {
            section_id: ROOT_SECTION_ID,
            key_name: "tail-ext-field".to_string(),
            value_or_ext_hash: hash.to_vec(),
            ext_value_size: blob_len as u64,
            ext_value_index: EXT_VALUE_INDEX,
            set_order: AUTHORING_RANK,
        }],
        ..Default::default()
    }
}

/// Deterministic external-value payload. An RNG would make a failing run
/// unreproducible for no benefit — the bytes only have to be non-trivial and
/// stable so `ext_content_hash` of them is the fixture's own contract.
fn ext_blob() -> Vec<u8> {
    (0..EXT_BLOB_LEN).map(|i| (i % 251) as u8).collect()
}

/// The post-wipe `UpdateLayer` payload Test 1 re-issues through the recovery
/// branch. Deliberately ADDITIVE — one new field on an existing section — so
/// the pre-wipe oracle stays comparable after the commit lands.
fn post_recovery_marker_delta() -> OwnedDelta {
    OwnedDelta {
        is_diff: true,
        base_version: 1,
        set_fields: vec![SetField {
            section_id: ROOT_SECTION_ID,
            key_name: POST_RECOVERY_MARKER_KEY.to_string(),
            value_or_ext_hash: vec![0x9E, 0x9F],
            ext_value_size: 0,
            ext_value_index: 0,
            set_order: AUTHORING_RANK,
        }],
        ..Default::default()
    }
}

/// Smallest delta `CreateLayer` accepts. Test 1's counter-re-seed step is about
/// which `object_id` is minted, not about materialized content.
fn empty_delta() -> OwnedDelta {
    OwnedDelta {
        is_diff: false,
        base_version: 0,
        ..Default::default()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// The store newtype Test 3 needs
// ─────────────────────────────────────────────────────────────────────────────

/// One `put_layer_checkpoint` attempt, as the wrapper saw it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PutAttempt {
    object_id: u64,
    snap_seq: u64,
    kind: CheckpointKind,
}

/// `CheckpointStore` newtype that records every layer PUT and can FAIL the
/// SR-3.6 re-base PUT specifically.
///
/// # Why a store-level newtype and not the fault-injection seam
///
/// Task 4 landed NO `CheckpointFaultInjector` seam on the re-base (the Task's
/// 2026-09-01 PM correction, item 2), and the seam that does exist dispatches on
/// a method-name label — `put_layer_checkpoint` — which the re-base PUT and the
/// loop's own PUT share, so it structurally cannot target one without the other.
/// A [`DelegatingCheckpointStore`]-based newtype keyed on
/// `(kind == Base, armed)` can, and costs no cargo feature, which is what keeps
/// this module ungated. Same pattern as `RecordingRebaseStore` in
/// `restore_rebase_test.rs`.
///
/// # Targeting
///
/// The re-base writes `(object_id, seeded_seq, CheckpointKind::Base)` and
/// `seeded_seq` is not knowable before the restore commits, so the fail arm keys
/// on `kind == Base` and is **armed** by the test only after its lineage has been
/// planted and the periodic loop has been fenced. The first `Base` PUT the
/// wrapper sees while armed is therefore necessarily the re-base, and the test
/// asserts the recorded `snap_seq` against the header it reads back, so a
/// mis-targeted arm surfaces as a failed assertion rather than a silent pass.
///
/// # Delegation
///
/// Forwards **every** trait method, including the four with non-trivial trait
/// defaults (`read_ext_refs_sidecar`, `list_layer_artifacts`,
/// `delete_layer_temp_artifacts`, `publish_worst_case`). Omitting those would
/// not leave a call un-instrumented, it would SUBSTITUTE behaviour behind the
/// wrapper and make assertions about it vacuously green.
struct RebasePutControlStore {
    delegate: DelegatingCheckpointStore<InMemoryCheckpointStore>,
    attempts: Mutex<Vec<PutAttempt>>,
    fail_base_puts: AtomicBool,
}

impl RebasePutControlStore {
    fn new() -> Self {
        Self {
            delegate: DelegatingCheckpointStore::new(Arc::new(InMemoryCheckpointStore::new())),
            attempts: Mutex::new(Vec::new()),
            fail_base_puts: AtomicBool::new(false),
        }
    }

    fn arm(&self) {
        self.fail_base_puts.store(true, Ordering::SeqCst);
    }

    fn heal(&self) {
        self.fail_base_puts.store(false, Ordering::SeqCst);
    }

    fn attempts(&self) -> Vec<PutAttempt> {
        self.attempts
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .clone()
    }

    /// Every `Base` PUT recorded after `from_index` (a snapshot of
    /// `attempts().len()` taken before the act step).
    fn base_puts_since(&self, from_index: usize) -> Vec<PutAttempt> {
        self.attempts()
            .into_iter()
            .skip(from_index)
            .filter(|a| a.kind == CheckpointKind::Base)
            .collect()
    }
}

#[async_trait]
impl CheckpointStore for RebasePutControlStore {
    async fn put_layer_checkpoint(
        &self,
        object_id: u64,
        snap_seq: u64,
        ompf: Bytes,
        meta: CheckpointMeta,
    ) -> Result<CheckpointHandle, CheckpointError> {
        // Recorded BEFORE the failure arm: the test wants to know the call was
        // ISSUED, whatever happens to it afterwards.
        self.attempts
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .push(PutAttempt {
                object_id,
                snap_seq,
                kind: meta.kind,
            });

        if meta.kind == CheckpointKind::Base && self.fail_base_puts.load(Ordering::SeqCst) {
            return Err(CheckpointError::Transient {
                source: "synthetic transient re-base PUT failure (SR-3.6 E2E)".into(),
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
    // default returns `NotImplemented`, which would report "no such generation"
    // for a generation the delegate holds, to the restore-source selection whose
    // whole job is deciding what exists.
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

    async fn list_referenced_ext_hashes(
        &self,
    ) -> Result<std::collections::HashSet<String>, CheckpointError> {
        self.delegate.list_referenced_ext_hashes().await
    }

    async fn list_live_ext_set(
        &self,
    ) -> Result<live_edit::checkpoint::LiveExtSet, CheckpointError> {
        self.delegate.list_live_ext_set().await
    }

    // ── The four non-trivial trait defaults ──────────────────────────────
    //
    // Forwarded because omitting them SUBSTITUTES behaviour rather than merely
    // skipping instrumentation: `read_ext_refs_sidecar`'s default returns an
    // EMPTY set, which would make every sidecar-shaped assertion vacuously
    // green.
    async fn read_ext_refs_sidecar(
        &self,
        handle: &CheckpointHandle,
    ) -> Result<std::collections::HashSet<String>, CheckpointError> {
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

// ─────────────────────────────────────────────────────────────────────────────
// Hot writer + bounded polling
// ─────────────────────────────────────────────────────────────────────────────

/// A writer that commits field-set deltas as fast as the backing store will take
/// them, through its OWN `LayerStore` at [`WRITER_STREAM_MAXLEN`].
///
/// Both halves of the payload shape are load-bearing: a field-only delta rides
/// `update_layer`'s Lua fast path (a structural command takes a slower path —
/// the structural route on `lua`, or the WATCH/MULTI/EXEC slow path on the
/// rollback backends, which commits roughly an order of magnitude less often —
/// and a slower writer lets the assembler win), and an `ompf`-less update entry
/// is read by the classifier as an EMPTY commit by invariant (b.pw6) — a
/// fixture built that way records real commits as no-ops and tests a lie.
///
/// Its own `LayerStore` rather than `harness.svc.layer_store` because the
/// harness's is pinned at the production default retention of 4096; see
/// [`WRITER_STREAM_MAXLEN`]. This is `helpers::drive_captured_tail`'s and
/// `e2e_tail_bootstrap_test`'s shape.
struct HotWriter {
    handle: JoinHandle<()>,
    stop: Arc<AtomicBool>,
    commits: Arc<AtomicU64>,
    error: Arc<Mutex<Option<String>>>,
}

impl HotWriter {
    fn spawn(tv: &TestValkey, session_id: &str, layer_name: &str, object_id: u64) -> Self {
        let layer_store = Arc::new(LayerStore::new(
            tv.client().clone(),
            tv.key_schema().clone(),
            WRITER_STREAM_MAXLEN,
        ));
        let stop = Arc::new(AtomicBool::new(false));
        let commits = Arc::new(AtomicU64::new(0));
        let error: Arc<Mutex<Option<String>>> = Arc::new(Mutex::new(None));

        let handle = {
            let stop = Arc::clone(&stop);
            let commits = Arc::clone(&commits);
            let error = Arc::clone(&error);
            let session_id = session_id.to_string();
            let layer_name = layer_name.to_string();
            tokio::spawn(async move {
                let mut nonce = 500_000u64;
                while !stop.load(Ordering::Relaxed) {
                    let delta = OwnedDelta {
                        set_fields: vec![SetField {
                            section_id: ROOT_SECTION_ID,
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
                    let ompf = Bytes::from(delta_to_ompf_bytes(&delta, &[]));
                    // No pacing sleep: the writer commits as fast as the
                    // backing store will take it.
                    if let Err(e) = layer_store
                        .update_layer(
                            &session_id,
                            &layer_name,
                            object_id,
                            Some(delta),
                            Some(ompf),
                            WRITER_CLIENT_ID,
                            0,
                        )
                        .await
                    {
                        // Normal at the END of a fixture (the pool is torn down
                        // at test exit), but recorded either way: a writer that
                        // died on its second commit and one whose arm was simply
                        // not reached produce the same "arm never reached" symptom.
                        *error.lock().unwrap_or_else(|p| p.into_inner()) = Some(e.to_string());
                        return;
                    }
                    commits.fetch_add(1, Ordering::Relaxed);
                    nonce += 1;
                }
            })
        };

        Self {
            handle,
            stop,
            commits,
            error,
        }
    }

    fn report(&self) -> String {
        let n = self.commits.load(Ordering::Relaxed);
        match self.error.lock().unwrap_or_else(|p| p.into_inner()).clone() {
            Some(e) => format!("{n} writer commits, then died with: {e}"),
            None => format!("{n} writer commits, no writer error"),
        }
    }

    /// Block until the writer has landed [`WARM_UP_COMMITS`], so the loop meets
    /// an already-hot layer. A state precondition, not a timing hack.
    async fn warm_up(&self) {
        let deadline = Instant::now() + WARM_UP_DEADLINE;
        while self.commits.load(Ordering::Relaxed) < WARM_UP_COMMITS {
            assert!(
                Instant::now() < deadline,
                "hot writer never reached {WARM_UP_COMMITS} warm-up commits within \
                 {WARM_UP_DEADLINE:?} ({})",
                self.report()
            );
            tokio::time::sleep(POLL_INTERVAL).await;
        }
    }

    async fn stop_and_join(self) {
        self.stop.store(true, Ordering::Relaxed);
        let _ = self.handle.await;
    }
}

/// Bounded poll on an async predicate. Returns whether it held before the
/// deadline. No naked `sleep` is used as synchronization anywhere in this file;
/// [`POLL_INTERVAL`] is the cadence of a state poll, not a wait for a race.
async fn poll_until<F, Fut>(deadline: Duration, mut pred: F) -> bool
where
    F: FnMut() -> Fut,
    Fut: std::future::Future<Output = bool>,
{
    let start = Instant::now();
    while start.elapsed() < deadline {
        if pred().await {
            return true;
        }
        tokio::time::sleep(POLL_INTERVAL).await;
    }
    pred().await
}

/// Bounded poll until one `live_edit_checkpoint_tail_total{outcome}` cell rises
/// above the value it held in `before`.
///
/// The METRIC, deliberately, and never "a Tail exists in the listing": every
/// fixture in this file HAND-PUTS a tail as the composed lineage's head, so a
/// listing predicate is satisfied from the first poll and every drive here would
/// return instantly while asserting nothing. The rising cell is the loop's own
/// verdict for THIS drive.
async fn poll_tail_outcome_rose(
    metrics: &SharedMetrics,
    before: &HashMap<String, f64>,
    outcome: &'static str,
    deadline: Duration,
) -> bool {
    let baseline = *before.get(outcome).unwrap_or(&0.0);
    let metrics = Arc::clone(metrics);
    poll_until(deadline, || {
        let metrics = Arc::clone(&metrics);
        async move {
            gather_tail_outcome_cells(&metrics)
                .get(outcome)
                .is_some_and(|value| *value > baseline)
        }
    })
    .await
}

/// Bounded poll until **a capture lands** — i.e. until the SUM of the two
/// [`TAIL_CAPTURE_LANDED_OUTCOMES`] cells rises above what it held in `before`.
///
/// The family, never the bare `captured` cell, for the reason documented on
/// [`TAIL_CAPTURE_LANDED_OUTCOMES`]: which of the two spellings a landed
/// capture takes is SR-1.4's cap decision, driven by how many commits the
/// runner happened to fit in the window, and no claim in this file is about
/// that split. Polling `captured` alone made every test here fail on a runner
/// fast enough to outrun `max_entries` on every tick — the loop captured
/// repeatedly and the poll never saw its cell move.
async fn poll_tail_capture_landed(
    metrics: &SharedMetrics,
    before: &HashMap<String, f64>,
    deadline: Duration,
) -> bool {
    let baseline = tail_captures_landed(before);
    let metrics = Arc::clone(metrics);
    poll_until(deadline, || {
        let metrics = Arc::clone(&metrics);
        async move { tail_captures_landed(&gather_tail_outcome_cells(&metrics)) > baseline }
    })
    .await
}

/// `(captured + captured_capped)` risen across a before/after pair — the
/// delta-shaped sibling of [`poll_tail_capture_landed`], and the counterpart to
/// [`tail_delta`] for the one claim that spans two cells.
fn tail_capture_landed_delta(before: &HashMap<String, f64>, after: &HashMap<String, f64>) -> f64 {
    tail_captures_landed(after) - tail_captures_landed(before)
}

/// Assert that the tails a drive captured form an UNBROKEN chain rooted exactly
/// at `seeded_seq` — the SR-3.6 re-based cursor.
///
/// # Why the chain, and not "the newest tail's `from_seq`"
///
/// A drive that gets more than one eligible tick captures more than one tail,
/// and SR-2.1 requires `tail_{n+1}.from_seq == tail_n.to_seq`, so the NEWEST
/// tail roots at its predecessor's head rather than at the cursor. Asserting on
/// the newest one alone is a race against how many ticks the drive happened to
/// win — observed as a real flake here (`from_seq: 5105` against a cursor of
/// `1009`, on roughly one run in eight). The claim that holds however many
/// tails land is this one: the OLDEST tail above the cursor roots exactly at it,
/// and every link above it is contiguous.
///
/// Tails above `seeded_seq` only: the fixture's own hand-put lineage head sits
/// at the composed head, well below, and is not part of what this drive
/// produced.
///
/// The chain is complete by construction, and post-Epic-5 that needs its own
/// argument now that retention CAN delete a tail. Two independent reasons, each
/// alone sufficient: [`DRIVE_RETAIN_GENERATIONS`] is 8, far above the lineage-
/// unit count these fixtures form, so the prune never leaves its early return;
/// and every tail this drive captures sits above the newest base, i.e. inside
/// the newest lineage unit, which retention never prunes
/// (`checkpointer::prune_old_generations`). So a hole here is a defect, not a
/// retention artifact.
///
/// Also carries the b.pw6 guard per member: claimed range must EQUAL carried
/// content. An `ompf`-less update entry is classified as an EMPTY commit by
/// invariant, so a decodable envelope with no-op records over real content is a
/// live failure mode, and record-count-versus-span is what catches it.
async fn assert_captured_chain_roots_at(
    harness: &CheckpointHarness,
    generations: &[CheckpointHandle],
    seeded_seq: u64,
) {
    let mut tails: Vec<&CheckpointHandle> = generations
        .iter()
        .filter(|h| h.kind == CheckpointKind::Tail && h.snap_seq > seeded_seq)
        .collect();
    assert!(
        !tails.is_empty(),
        "the drive must have captured at least one Tail above the SR-3.6 re-based cursor \
         ({seeded_seq}); the only other Tail in the listing is the fixture's hand-put lineage \
         head. Listing: {generations:?}"
    );
    tails.sort_by_key(|handle| handle.snap_seq);

    let mut expected_from = seeded_seq;
    for handle in tails {
        let blob = harness
            .store
            .get_layer_checkpoint(handle)
            .await
            .expect("get_layer_checkpoint must succeed")
            .expect("a handle from the listing must address a live artifact");
        let envelope = decode_tail_envelope(&blob.bytes)
            .expect("every captured artifact must be a real tail envelope");
        assert_eq!(
            envelope.header().from_seq,
            expected_from,
            "captured tail@{} must open at {expected_from}. For the FIRST tail that is the \
             RE-BASED cursor — a lower value means the capture rooted at a pre-restore \
             generation and its records would replay over an abandoned timeline. For a later \
             one it is SR-2.1's `tail_{{n+1}}.from_seq == tail_n.to_seq` contiguity. Listing: \
             {generations:?}",
            handle.snap_seq
        );
        assert_eq!(
            envelope.header().to_seq,
            handle.snap_seq,
            "a tail's handle sequence IS its envelope to_seq"
        );
        assert_eq!(
            envelope.records().len() as u64,
            envelope.header().to_seq - envelope.header().from_seq,
            "claimed range must EQUAL carried content (b.pw6): {} record(s) for the span \
             ({}, {}]",
            envelope.records().len(),
            envelope.header().from_seq,
            envelope.header().to_seq
        );
        expected_from = envelope.header().to_seq;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Harness wiring
// ─────────────────────────────────────────────────────────────────────────────

/// A harness with `tick_secs = TICK_SECS`, the tail flag **ON** (so the SR-7.1
/// creation bootstrap and the capture path are live; which arm a tick takes is
/// the unprovable-cut seam's, see the module header) and the stock production
/// hook wiring, re-base included.
///
/// `store = None` roots the harness's own LocalDisk store; `Some(..)` installs a
/// caller-supplied one everywhere the default would have gone.
///
/// Tests 2–4 use this. Test 1 uses
/// [`setup_tail_harness_without_restore_rebase`].
async fn setup_tail_harness(store: Option<Arc<dyn CheckpointStore>>) -> CheckpointHarness {
    setup_tail_harness_with_rebase_posture(store, true).await
}

/// [`setup_tail_harness`] with the SR-3.6 re-base wiring OPTED OUT —
/// `CheckpointHarnessOptions::wire_restore_rebase = false`, the documented
/// `None` posture on the recovery hook's `BaseBootstrap` slot.
///
/// Nothing else differs, and in particular the SR-7.1 three-way gate is still
/// asserted to resolve TRUE (see [`setup_tail_harness_with_rebase_posture`]), so
/// the opt-out is provably the reason no re-base runs rather than a disabled
/// gate.
///
/// Test 1's only consumer, because its claim is `last_checkpoint_seq == S` —
/// what the RESTORE stamped — and a successful re-base overwrites that with
/// `seeded_seq`. Tests 2–4 need the opposite and take the stock wiring.
async fn setup_tail_harness_without_restore_rebase() -> CheckpointHarness {
    setup_tail_harness_with_rebase_posture(None, false).await
}

/// The one body both entry points above are built from.
async fn setup_tail_harness_with_rebase_posture(
    store: Option<Arc<dyn CheckpointStore>>,
    wire_restore_rebase: bool,
) -> CheckpointHarness {
    let mut opts = CheckpointHarnessOptions::new(TICK_SECS);
    opts.tail.enabled = true;
    opts.store = store;
    opts.wire_restore_rebase = wire_restore_rebase;
    // No `ext_value_store` / `ext_value_store_factory` override: the harness
    // default IS a real `ValkeyExternalValueStore` over the harness's own
    // `TestValkey`, which is what this file's `le:ext:*` wipe-and-restore
    // assertions need. Deliberately NOT arming `.with_checkpoint_store(..)`
    // through the factory either — Test 1's subject is the SR-3.4 restore-time
    // pre-store, and the fetch-on-miss fallback would let a MISSING pre-store
    // self-heal out of the checkpoint adapter and read green.
    let harness = setup_checkpoint_harness_with_options(opts).await;
    assert!(
        harness.config.checkpoint_tail_enabled,
        "fixture precondition: the harness Config must carry the tail flag ON — the SR-7.1 \
         bootstrap gate and the loop's TailCaptureConfig both read it, and with it off every \
         arm in this file is unreachable"
    );
    // Derived, never hardcoded: a literal `true` here would keep passing if
    // production stopped gating. Same three-way conjunction `main.rs` spells at
    // its `BaseBootstrap::new` call site.
    //
    // Asserted on BOTH postures on purpose. Under `wire_restore_rebase = true`
    // it says the re-base unit can write, so Tests 2-4 are not asserting
    // against a no-op. Under the opt-out it says the gate WOULD have resolved
    // true, which is what makes Test 1's silence evidence about the opt-out
    // rather than about a gate that was closed anyway.
    let bootstrap_enabled = harness.config.checkpoint_enabled
        && !matches!(
            harness.config.checkpoint_adapter,
            CheckpointAdapter::Disabled
        )
        && harness.config.checkpoint_tail_enabled;
    assert!(
        bootstrap_enabled,
        "fixture precondition: the SR-7.1 three-way gate must resolve TRUE on this harness, \
         otherwise the re-base unit cannot write — Tests 2-4 would assert against a no-op, and \
         Test 1's opt-out would prove nothing because no re-base could have run regardless"
    );
    // The counter re-seed Test 1 asserts on is only observable through a hooked
    // `LayerStore`; assert the wiring up front so a harness regression is
    // reported as itself rather than as a confusing id mismatch downstream.
    assert_harness_carries_object_id_counter_repair(&harness);
    harness
}

/// Spawn a periodic checkpoint loop against the harness's own wiring, with the
/// tail knobs RE-DERIVED from `harness.config` — the same single source of truth
/// `main.rs` and the harness's own loop read.
///
/// Used after the fixture has fenced the harness's loop: the leader lock is an
/// acquire-or-refresh CAS keyed on `replica_id`, so a second loop under the same
/// [`SERVICE_LABEL`] re-acquires rather than waiting out the TTL.
///
/// The loop runs on `unprovable_cut_client`, armed for `object_id`, so every
/// assembly of the drives' layer persists nothing and its ticks fall back to
/// the capture (see the module header). The seam stays armed until the test
/// disarms it.
#[cfg(feature = "test-checkpoint-fault-injection")]
async fn spawn_tail_loop(
    harness: &CheckpointHarness,
    object_id: u64,
) -> (CancellationToken, JoinHandle<()>) {
    let cancel = CancellationToken::new();
    let handle = spawn_checkpoint_loop(
        unprovable_cut_client(&harness.tv, object_id),
        harness.tv.key_schema().clone(),
        Arc::clone(&harness.store),
        CheckpointLoopConfig {
            tick: Duration::from_secs(TICK_SECS),
            // Both gates open so every tick reaches the assembler; the capture
            // fallback is downstream of the assembly being SKIPPED, not of a
            // gate.
            interval: Duration::ZERO,
            quiesce: Duration::ZERO,
            retain_generations: DRIVE_RETAIN_GENERATIONS,
            counters_sync: Duration::from_secs(3600),
            reconcile_interval: Duration::from_secs(3600),
            leader: crate::common::test_leader(SERVICE_LABEL),
            tail: TailCaptureConfig::from_config(&harness.config),
        },
        Some(Arc::clone(&harness.metrics)),
        cancel.clone(),
    );
    (cancel, handle)
}

/// Cancel a spawned loop and join it under ONE deadline.
async fn drain_loop(cancel: CancellationToken, handle: JoinHandle<()>) {
    cancel.cancel();
    let _ = tokio::time::timeout(LOOP_DRAIN_BUDGET, handle).await;
}

// ─────────────────────────────────────────────────────────────────────────────
// Drive + read helpers
// ─────────────────────────────────────────────────────────────────────────────

/// A `StatLayerRequest` carrying the stub bearer token the harness's
/// `AllowAllPermissions` / `StubUserinfo` wiring expects.
fn stat_request(session_id: &str, layer_name: &str) -> Request<StatLayerRequest> {
    let mut request = Request::new(StatLayerRequest {
        session_id: session_id.to_owned(),
        layer_name: layer_name.to_owned(),
    });
    request
        .extensions_mut()
        .insert(BearerToken::new("test-token"));
    request
}

/// Issue one `UpdateLayer` through the real handler and return
/// `(sequence_num, the OMPF bytes it sent)`.
///
/// Returning the sent bytes is the whole point: they are what
/// [`put_tail_generation_from_records`] encodes into the tail envelope, so the
/// artifact carries exactly what the live commit carried — including, for the
/// ext-value commit, the blob as OMPF Part 1. That is byte-for-byte what the
/// capture path persists (`le:{oid}:obj:stream`'s `ompf` field is the writer's
/// raw inbound OMPF).
async fn drive_commit(
    harness: &CheckpointHarness,
    session_id: &str,
    layer_name: &str,
    object_id: u64,
    delta: &OwnedDelta,
    external_values: &[&[u8]],
    client_timestamp: u64,
) -> (u64, Bytes) {
    let ompf = delta_to_ompf_bytes(delta, external_values);
    let stream = streaming_update(vec![
        make_update_header(
            session_id,
            layer_name,
            object_id,
            client_timestamp,
            CLIENT_ID,
        ),
        make_update_data(ompf.clone()),
    ]);
    let sequence_num = handle_update_layer(&harness.svc, make_streaming_token_request(stream))
        .await
        .expect("UpdateLayer must succeed while building the lineage fixture")
        .into_inner()
        .sequence_num;
    (sequence_num, Bytes::from(ompf))
}

/// `HGETALL le:{object_id}:obj:hdr`.
async fn read_obj_header(harness: &CheckpointHarness, object_id: u64) -> HashMap<String, String> {
    harness
        .tv
        .client()
        .pool
        .hgetall(&harness.tv.key_schema().obj_header(&object_id.to_string()))
        .await
        .expect("HGETALL obj:hdr must succeed")
}

/// Parse a `u64` column off an `obj:hdr` map with a diagnostic on absence.
fn header_u64(header: &HashMap<String, String>, field: &str) -> u64 {
    let raw = header.get(field).unwrap_or_else(|| {
        panic!(
            "obj:hdr MUST carry `{field}`; got fields: {:?}",
            header.keys().collect::<Vec<_>>()
        )
    });
    raw.parse::<u64>()
        .unwrap_or_else(|e| panic!("obj:hdr `{field}` MUST parse as u64; got {raw:?}: {e}"))
}

/// Like [`header_u64`] but folds an absent column to `0`. Only for the three
/// restore-cursor columns, which a never-restored layer legitimately lacks.
fn header_u64_or_zero(header: &HashMap<String, String>, field: &str) -> u64 {
    header
        .get(field)
        .map(|raw| {
            raw.parse::<u64>()
                .unwrap_or_else(|e| panic!("obj:hdr `{field}` MUST parse as u64; got {raw:?}: {e}"))
        })
        .unwrap_or(0)
}

/// `XINFO`-free high-water read: the last entry id in the per-object stream, as
/// a sequence number.
async fn stream_top_seq(harness: &CheckpointHarness, object_id: u64) -> u64 {
    let key = harness.tv.key_schema().layer_stream(object_id);
    let entries: Vec<(String, Vec<(String, Bytes)>)> = harness
        .tv
        .client()
        .pool
        .xrevrange(&key, "+", "-", Some(1))
        .await
        .expect("XREVRANGE over the layer stream must succeed");
    let (id, _) = entries
        .first()
        .expect("the surviving stream must carry at least one entry");
    id.split('-')
        .next()
        .and_then(|ms| ms.parse::<u64>().ok())
        .unwrap_or_else(|| panic!("unparseable stream entry id {id:?}"))
}

/// `EXISTS` on a fully-qualified key.
async fn key_exists(harness: &CheckpointHarness, key: &str) -> bool {
    harness
        .tv
        .client()
        .pool
        .exists(key)
        .await
        .expect("EXISTS must succeed against the live test Valkey")
}

/// The newest-first generation listing for `object_id`.
async fn listing(harness: &CheckpointHarness, object_id: u64) -> Vec<CheckpointHandle> {
    harness
        .store
        .list_layer_checkpoints(object_id)
        .await
        .expect("list_layer_checkpoints must succeed")
}

/// Persist a counters sidecar through the harness's real checkpoint store,
/// round-tripping the shared `CountersSidecar` struct rather than hand-rolled
/// JSON — the repair deserializes with the same type, so this pins the actual
/// producer/consumer contract. Models the periodic counters-sync tick, which the
/// harness pushes out to an hour so it cannot race the assertions.
async fn put_counters_sidecar(harness: &CheckpointHarness, max_object_id_observed: u64) {
    let sidecar = CountersSidecar {
        max_object_id_observed,
        updated_at: "2026-09-01T12:00:00Z".to_string(),
    };
    let bytes = serde_json::to_vec(&sidecar).expect("CountersSidecar serialization must succeed");
    harness
        .store
        .put_counters_sidecar(Bytes::from(bytes))
        .await
        .expect("put_counters_sidecar must succeed against the harness adapter");
}

// ─────────────────────────────────────────────────────────────────────────────
// The fixture: a real-commit composed lineage built through the real handlers
// ─────────────────────────────────────────────────────────────────────────────

/// The live state at one instant — the two views the restored state is compared
/// against, plus the fleet-visible session view the path-index restore is read
/// through.
struct Oracle {
    snapshot: ObjectSnapshot,
    header: HashMap<String, String>,
    /// `(session_id, name)` for every session `ListSessions(base_path)` returns.
    sessions_at_base_path: Vec<(String, String)>,
}

/// Everything the four tests share.
struct ComposedLineageFixture {
    session_id: String,
    layer_name: String,
    object_id: u64,
    /// `S0` — the loop-written base's `snap_seq`, and the layer's live sequence
    /// at the instant that base was assembled.
    s0: u64,
    /// `S` — the composed head: the last commit inside the hand-put tail's span.
    head_seq: u64,
    /// The external value one tail-range commit carried, and its hex digest.
    ext_blob: Vec<u8>,
    ext_hash_hex: String,
    /// The live state at `S`: what the composed restore must reproduce.
    head_oracle: Oracle,
}

async fn capture_oracle(harness: &CheckpointHarness, object_id: u64) -> Oracle {
    let snapshot = read_object_state_for_test(
        harness.tv.client(),
        harness.tv.key_schema(),
        &object_id.to_string(),
    )
    .await
    .expect("read_object_state_for_test must succeed while the live layer is intact");
    let header = read_obj_header(harness, object_id).await;
    let sessions_at_base_path = list_sessions_at_base_path(harness).await;
    Oracle {
        snapshot,
        header,
        sessions_at_base_path,
    }
}

/// `ListSessions(SESSION_BASE_PATH)` through the real handler, as
/// `(session_id, name)` pairs. Reads `path_names_index`, so it is the
/// end-to-end witness that the path-index survived (pre-wipe) or was restored
/// (post-wipe).
async fn list_sessions_at_base_path(harness: &CheckpointHarness) -> Vec<(String, String)> {
    handle_list_sessions(
        &harness.svc,
        make_unary_token_request(ListSessionsRequest {
            path: SESSION_BASE_PATH.to_owned(),
        }),
    )
    .await
    .expect("ListSessions must succeed against an AllowAllPermissions harness")
    .into_inner()
    .sessions
    .into_iter()
    .map(|s| (s.session_id, s.name))
    .collect()
}

/// The oracle's own self-check, run at capture time: the decoded snapshot and
/// the raw header agree on the sequence, and the layer has never been restored.
///
/// The second half is what gives every post-restore `restore_seq_bump`
/// assertion in this file its meaning — the column is proved absent BEFORE the
/// restore, so its post-restore value is a change this restore made rather than
/// a residue that was already there.
fn assert_live_oracle_is_unrestored(oracle: &Oracle) {
    assert_eq!(
        header_u64(&oracle.header, "seq"),
        oracle.snapshot.sequence_num,
        "the live obj:hdr and the decoded snapshot must agree on `seq`"
    );
    assert_eq!(
        header_u64_or_zero(&oracle.header, "restore_seq_bump"),
        0,
        "a layer that has only ever been written live MUST carry no guard band. A non-zero \
         value here means a restore already ran, and every post-restore bump assertion in this \
         file would be reading a residue"
    );
}

/// CreateSession → CreateLayer (wide) → two commits → wait for the loop's base
/// → **fence the loop** → drive the tail-range commits → hand-put the tail →
/// capture the head oracle.
///
/// The fence is `cancel()` plus a bounded join on the loop's `JoinHandle`, the
/// shape `recovery_mixed_kind_selection_test` uses. After it, no tick exists to
/// land a generation, so nothing can move the fixture underneath a test: not the
/// retention prune, and not a fresh base at a sequence the hand-put tail does
/// not link onto. Tests that need the loop back spawn a fresh one via
/// [`spawn_tail_loop`].
async fn setup_composed_lineage(
    harness: &mut CheckpointHarness,
    name: &str,
) -> ComposedLineageFixture {
    // ── The session, through the real RPC entry point. Its checkpoint blob is
    // what the post-wipe session hop rebuilds from; without the wait the wipe
    // would race the fire-and-forget writer and the restore would fail at
    // session resolution rather than at the layer claim under test.
    let session_id = handle_create_session(
        &harness.svc,
        make_unary_token_request(CreateSessionRequest {
            path: SESSION_BASE_PATH.to_owned(),
            name: format!("{name}-session"),
        }),
    )
    .await
    .expect("CreateSession must succeed against an AllowAllPermissions harness")
    .into_inner()
    .session_id;
    wait_for_session_checkpoint(&harness.store, &session_id, STORE_DEADLINE_SECS).await;

    // ── The layer, through the real streaming CreateLayer handler — the path
    // that actually calls `svc.base_bootstrap.spawn_creation_base(..)`.
    let layer_name = format!("{name}-layer");
    let ompf = delta_to_ompf_bytes(&wide_initial_delta(), &[]);
    let stream = streaming_create(vec![
        make_create_header(&session_id, &layer_name),
        make_create_data(ompf),
    ]);
    let object_id = handle_create_layer(&harness.svc, make_streaming_token_request(stream))
        .await
        .expect("CreateLayer must succeed while building the lineage fixture")
        .into_inner()
        .object_id;
    assert!(
        object_id > 0,
        "CreateLayer must mint a non-zero object_id; got {object_id}"
    );
    // Drains BOTH the CreateLayer session-checkpoint write and the SR-7.1
    // creation bootstrap task (both ride the harness's shared TaskTracker), so
    // neither can land after a later wipe and resurrect partial state.
    wait_for_session_checkpoint_quiesced(harness, &session_id, STORE_DEADLINE_SECS).await;

    // ── Two ordinary commits before the base, so the base carries state no
    // single delta produced and the restored snapshot is a genuine round trip of
    // accumulated live state.
    let mut committed_seq = 0;
    for index in 0..2 {
        let (seq, _) = drive_commit(
            harness,
            &session_id,
            &layer_name,
            object_id,
            &pre_base_commit_delta(index),
            &[],
            1_000 + index,
        )
        .await;
        committed_seq = seq;
    }

    // Gate on the committed sequence, not merely "any base": with
    // `interval = quiesce = 0` the loop can fire between CreateLayer and the
    // commits and land a generation that predates them, which would leave `S0`
    // non-deterministic and the tail's link arithmetic meaningless.
    let base_handle = wait_for_checkpoint_at_least(
        &harness.store,
        object_id,
        committed_seq,
        STORE_DEADLINE_SECS,
    )
    .await;

    // ── The fence ────────────────────────────────────────────────────────
    harness.cancel.cancel();
    let loop_task = std::mem::replace(&mut harness.loop_handle, tokio::spawn(async {}));
    let _ = tokio::time::timeout(LOOP_DRAIN_BUDGET, loop_task).await;

    let s0 = base_handle.snap_seq;
    assert_eq!(
        s0, committed_seq,
        "fixture precondition: the loop's base MUST have been assembled at the layer's live \
         sequence ({committed_seq}). A higher value means a commit landed that this fixture \
         does not know about, and the hand-put tail's `from_seq = S0` link would name a \
         predecessor covering commits nothing replays"
    );

    // ── The tail range: structural + inline + time-sample commits, then one
    // carrying an EXTERNAL VALUE. The ext commit is last so the head oracle's
    // ext reference is the newest write in the replayed range.
    let blob = ext_blob();
    let ext_hash = ext_content_hash(&blob);
    let ext_hash_hex = sha256_to_hex(&ext_hash);

    let mut records: Vec<(u64, Bytes)> = Vec::new();
    let mut expected = s0 + 1;
    for index in 0..TAIL_RANGE_COMMITS {
        let is_last = index + 1 == TAIL_RANGE_COMMITS;
        let (delta, externals): (OwnedDelta, Vec<&[u8]>) = if is_last {
            (
                tail_ext_value_delta(&ext_hash, blob.len()),
                vec![blob.as_slice()],
            )
        } else {
            (tail_commit_delta(index), Vec::new())
        };
        let record = drive_commit(
            harness,
            &session_id,
            &layer_name,
            object_id,
            &delta,
            &externals,
            2_000 + index,
        )
        .await;
        assert_eq!(
            record.0, expected,
            "fixture precondition: commit sequences must be contiguous. Commit {index} \
             returned {}, expected {expected}. A gap here means another writer shares this \
             layer, and the hand-put tail would carry a span its records cannot cover",
            record.0
        );
        expected += 1;
        records.push(record);
    }
    let head_seq = records.last().expect("commits were driven").0;

    // The ext-value commit went through the service's real
    // `ValkeyExternalValueStore` — the harness default — so both
    // content-addressed keys must exist. Otherwise the post-wipe "restored ext
    // values are byte-identical" claim would be comparing against nothing.
    for key in [
        harness.tv.key_schema().ext_meta(&ext_hash_hex),
        harness.tv.key_schema().ext_data(&ext_hash_hex),
    ] {
        assert!(
            key_exists(harness, &key).await,
            "fixture precondition: `{key}` MUST exist after the ext-carrying UpdateLayer. Its \
             absence means the service is holding an in-process store (a MockExternalValueStore, \
             say) rather than the harness's real Valkey-backed one, and the ext half of this \
             file asserts nothing"
        );
    }

    let head_oracle = capture_oracle(harness, object_id).await;
    assert_eq!(
        head_oracle.snapshot.sequence_num, head_seq,
        "fixture precondition: the head oracle must be captured at the composed head sequence"
    );
    assert_live_oracle_is_unrestored(&head_oracle);

    put_tail_generation_from_records(
        &harness.store,
        object_id,
        &session_id,
        &layer_name,
        s0,
        s0,
        &records,
        HAND_PUT_WRITTEN_AT_UNIX_MS,
        0,
    )
    .await;

    ComposedLineageFixture {
        session_id,
        layer_name,
        object_id,
        s0,
        head_seq,
        ext_blob: blob,
        ext_hash_hex,
        head_oracle,
    }
}

/// Drive [`FOSSIL_COMMITS`] real commits ABOVE the hand-put tail's head — the
/// abandoned-timeline entries Test 4's surviving stream keeps. Returns the last
/// committed sequence.
async fn drive_fossil_commits(
    harness: &CheckpointHarness,
    fixture: &ComposedLineageFixture,
) -> u64 {
    let mut last = fixture.head_seq;
    for index in 0..FOSSIL_COMMITS {
        let (seq, _) = drive_commit(
            harness,
            &fixture.session_id,
            &fixture.layer_name,
            fixture.object_id,
            &pre_base_commit_delta(50 + index),
            &[],
            3_000 + index,
        )
        .await;
        last = seq;
    }
    last
}

// ─────────────────────────────────────────────────────────────────────────────
// Metric helpers — every assertion below is a BEFORE/AFTER delta
// ─────────────────────────────────────────────────────────────────────────────

/// Snapshot every cell of `live_edit_checkpoint_restore_lineage_total` for a
/// before/after delta. The whole pinned vocabulary, never one cell: a step-down
/// that ticks the WRONG reason is a different defect from one that does not tick
/// at all, and only a full sweep tells them apart.
fn lineage_cells(harness: &CheckpointHarness) -> Vec<(&'static str, u64)> {
    CHECKPOINT_RESTORE_LINEAGE_OUTCOMES
        .iter()
        .map(|outcome| {
            (
                *outcome,
                read_restore_lineage_counter(&harness.metrics, outcome),
            )
        })
        .collect()
}

/// Assert the per-outcome deltas since `before`. Outcomes absent from `expected`
/// must be flat.
fn assert_lineage_deltas(
    harness: &CheckpointHarness,
    before: &[(&'static str, u64)],
    expected: &[(&str, u64)],
    context: &str,
) {
    for (outcome, pre) in before {
        let want = expected
            .iter()
            .find(|(name, _)| name == outcome)
            .map(|(_, value)| *value)
            .unwrap_or(0);
        let got = read_restore_lineage_counter(&harness.metrics, outcome) - pre;
        assert_eq!(
            got, want,
            "{context}: lineage outcome {outcome:?} MUST tick {want} time(s), got {got}. This \
             family is NOT one-tick-per-attempt — one attempt that steps down twice before \
             resolving ticks three cells"
        );
    }
}

/// The `outcome → value` delta of `live_edit_checkpoint_tail_total` between two
/// [`gather_tail_outcome_cells`] snapshots.
///
/// The family carries `outcome` ONLY — deliberately service-free — so a matcher
/// naming a `service` label would match NOTHING and every assertion would read
/// zero. The shared readers in `helpers.rs` carry that discipline; this is only
/// their subtraction.
fn tail_delta(before: &HashMap<String, f64>, after: &HashMap<String, f64>, outcome: &str) -> f64 {
    let pre = *before.get(outcome).unwrap_or_else(|| {
        panic!(
            "live_edit_checkpoint_tail_total{{outcome={outcome:?}}} was ABSENT from the BEFORE \
             snapshot. Every member of CHECKPOINT_TAIL_OUTCOMES is zero-seeded at Metrics::new, \
             so an absent cell means the vocabulary or the label set moved"
        )
    });
    let post = *after.get(outcome).unwrap_or_else(|| {
        panic!(
            "live_edit_checkpoint_tail_total{{outcome={outcome:?}}} was ABSENT from the AFTER \
             snapshot but present in the BEFORE one — the family's label set changed mid-test"
        )
    });
    post - pre
}

/// Pin the landed refusal SHAPE in code, so the dead `Trimmed` branch cannot be
/// re-added.
///
/// This lane drives the REAL periodic loop, which reports its verdict through
/// the metric family rather than by returning a `TailCaptureDecision`, so the
/// enum variant is not directly observable here — the narrow instrument that
/// asserts it directly is `t3.dfh.yj.4f.gw`'s (`restore_rebase_test.rs`), and
/// this file deliberately does not call it. What IS pinned here, statically, is
/// the mapping the metric assertion then rides on:
/// `TailCaptureRefusal::PostRestoreUnrebased` → `refused_seq_gap`, on BOTH
/// stream shapes.
///
/// `Trimmed` is unreachable while `restore_seq_bump != 0` because the
/// post-restore fossil guard runs BEFORE the stream read
/// (`tail_capture::capture_inner` step 3) — the spec child's Task-4-PM
/// clarification item 2, and this Subtask's PM condition C1/B1, which STRUCK the
/// earlier "or the classifier's `Trimmed` on a fresh stream" wording. Callers
/// pair this with a proof that the guard's precondition held at the moment of
/// the refusal (a non-zero `restore_seq_bump` read off `obj:hdr`), which is what
/// makes `PostRestoreUnrebased` the only producer that could have ticked the
/// cell.
///
/// # The third producer, and why it cannot be the one that ticked
///
/// `refused_seq_gap` has three producers, not two: the classifier's observed
/// ascending jump, the capture unit's pre-read fossil guard (this one), and — as
/// of SR-4.1 — the checkpoint loop's chain-at-K re-base admission gate, which
/// declines to start a chain-collapse re-base on the same header condition and
/// ticks the same cell. The third is out of reach here for a structural reason,
/// stated so a later reader does not have to re-derive it: it is only reachable
/// on a tick where the capture unit already returned `ChainFull`, and these
/// drives run at the stock chain bound with a fresh base and at most a tail or
/// two above it. If a fixture in this file is ever rewritten to approach K, this
/// helper's "only producer" claim needs re-earning.
fn assert_landed_refusal_label_is_seq_gap(observed_bump: u64) {
    assert!(
        observed_bump != 0,
        "the post-restore fossil guard fires on `restore_seq_bump != 0`; with the column at \
         zero the refusal under assertion could not be PostRestoreUnrebased and this pin would \
         be describing a different arm"
    );
    let landed = TailCaptureDecision::Refused(TailCaptureRefusal::PostRestoreUnrebased {
        restore_seq_bump: observed_bump,
    });
    assert_eq!(
        landed.outcome_label(),
        "refused_seq_gap",
        "PostRestoreUnrebased MUST report `refused_seq_gap` on BOTH stream shapes. `Trimmed` is \
         unreachable while restore_seq_bump != 0 (the guard runs before the stream read) — see \
         the spec child's Task-4-PM clarification item 2; do not re-add that branch"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// Arm 1 — the whole-keyspace wipe, restored to exact state
// ═════════════════════════════════════════════════════════════════════════════

/// Epic AC: a `full_valkey_wipe` followed by ONE `UpdateLayer` against the
/// original `(session_id, layer_name)` restores a TAIL-LINEAGE layer to exact
/// state, composing every hop in the chain.
///
/// What only this test asserts, in one dispatch:
///
/// - both recovery hops fire (`restored{scope="session"} +1` AND
///   `restored{scope="layer"} +1`, with `error` / `not_recoverable` flat);
/// - the lineage was COMPOSED, not stepped down (`gathered_composed +1`, every
///   `stepped_down_*` and `exhausted` flat);
/// - `last_checkpoint_seq == S` — the composed head, not the base's `snap_seq`.
///   The base's value is the regression this line exists to exclude: it
///   under-states what the restore materialized and seeds the writer cursor
///   below the layer's real pre-wipe sequence;
/// - the restored state equals the pre-wipe oracle on the faithful basis, and
///   the replayed range's per-entry versions are the live commits' own;
/// - the external value the tail carried is reachable BYTE-IDENTICALLY after the
///   wipe destroyed the whole `le:ext:*` namespace;
/// - the path index is back (`ListSessions` returns the original session);
/// - and the object-id counter re-seeded from the sidecar rather than
///   re-establishing at zero.
///
/// The recovery hook here is wired WITHOUT the SR-3.6 re-base, through the named
/// opt-out [`setup_tail_harness_without_restore_rebase`]: a successful re-base
/// stamps `last_checkpoint_seq = seeded_seq` over the restore's own stamp, which
/// is the very column this test pins. The opt-out is the ONLY difference from
/// Tests 2–4's harness — the SR-7.1 gate is still asserted true — so the absence
/// of a re-base here is the opt-out's doing and nothing else. The re-base is
/// Tests 2–4's subject.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn full_valkey_wipe_restores_a_tail_lineage_layer_to_exact_state() {
    let mut harness = setup_tail_harness_without_restore_rebase().await;

    let fixture = setup_composed_lineage(&mut harness, "exact-state").await;

    // The periodic counters-sync tick, modelled by hand: the harness pushes that
    // cadence out to an hour so a timer cannot race the assertions, and the
    // post-wipe allocation needs a sidecar to re-seed from.
    put_counters_sidecar(&harness, fixture.object_id).await;
    wait_for_writer_idle(&harness.session_checkpoint_tracker, STORE_DEADLINE_SECS).await;

    // ── The keyspace loss. Namespaced SCAN + UNLINK — never FLUSHALL; the
    // backing instance is shared with concurrent agents and CI jobs.
    full_valkey_wipe(&harness.tv).await;

    // Sanity on three specific keys before the assertion phase: a failed wipe
    // would let the next UpdateLayer succeed without exercising the recovery
    // branch at all, silently passing everything below.
    let session_meta_key = harness.tv.key_schema().session_meta(&fixture.session_id);
    let mapping_key = harness
        .tv
        .key_schema()
        .layer_mapping(&fixture.session_id, &fixture.layer_name);
    let obj_header_key = harness
        .tv
        .key_schema()
        .obj_header(&fixture.object_id.to_string());
    for key in [&session_meta_key, &mapping_key, &obj_header_key] {
        assert!(
            !key_exists(&harness, key).await,
            "full_valkey_wipe must remove `{key}`; a surviving key here means the SCAN missed \
             it and the corresponding recovery hop would silently not fire"
        );
    }
    assert!(
        !key_exists(&harness, &harness.tv.key_schema().counter_object_id()).await,
        "full_valkey_wipe must remove le:counter:object_id — that absence is the whole signal \
         the b.b2j repair path keys off"
    );
    assert!(
        !key_exists(
            &harness,
            &harness.tv.key_schema().ext_data(&fixture.ext_hash_hex)
        )
        .await,
        "full_valkey_wipe must remove the ext-value namespace, so the post-restore byte \
         comparison is evidence the restore RE-STORED the blob rather than that it survived"
    );

    // The pre-wipe calls populated the in-process base-path cache, whose positive
    // TTL outlasts this test. In production the `session:deleted` pub/sub fanout
    // drives cross-replica invalidation; under E2E there is no peer replica
    // writing the wipe, so without this the handler would HIT the cache and
    // session recovery would silently not fire.
    harness
        .svc
        .session_base_path_cache
        .invalidate(&fixture.session_id);

    // ── Baselines ────────────────────────────────────────────────────────
    let session_restored_pre = session_recovery_metric(&harness, "restored");
    let session_error_pre = session_recovery_metric(&harness, "error");
    let session_not_recoverable_pre = session_recovery_metric(&harness, "not_recoverable");
    let layer_restored_pre = layer_recovery_metric(&harness, "restored");
    let layer_error_pre = layer_recovery_metric(&harness, "error");
    let layer_not_recoverable_pre = layer_recovery_metric(&harness, "not_recoverable");
    let lineage_pre = lineage_cells(&harness);

    // ── The single production dispatch. Two hops in series: the cache funnel's
    // `get_base_path` sees NotFound and fires session recovery; `update_layer_once`
    // then resolves the restored mapping, HGETALLs an empty `obj:hdr`, and the
    // layer-recovery branch composes base@S0 + tail(S0, S] before re-issuing the
    // commit.
    let (post_recovery_seq, _) = drive_commit(
        &harness,
        &fixture.session_id,
        &fixture.layer_name,
        fixture.object_id,
        &post_recovery_marker_delta(),
        &[],
        9_000,
    )
    .await;
    assert!(
        post_recovery_seq > fixture.head_seq,
        "Epic AC: the post-recovery sequence_num ({post_recovery_seq}) MUST be strictly greater \
         than the pre-wipe head ({}). A smaller value re-issues sequence numbers the replayed \
         tail already used",
        fixture.head_seq
    );

    // ── Both recovery hops, exactly once each ────────────────────────────
    wait_for_recovery_metric(
        &harness,
        "session",
        "restored",
        session_restored_pre + 1,
        STORE_DEADLINE_SECS,
    )
    .await;
    wait_for_recovery_metric(
        &harness,
        "layer",
        "restored",
        layer_restored_pre + 1,
        STORE_DEADLINE_SECS,
    )
    .await;
    assert_eq!(
        session_recovery_metric(&harness, "error"),
        session_error_pre,
        "`error{{scope=\"session\"}}` MUST be flat — no backend failure was injected"
    );
    assert_eq!(
        session_recovery_metric(&harness, "not_recoverable"),
        session_not_recoverable_pre,
        "`not_recoverable{{scope=\"session\"}}` MUST be flat — the session-checkpoint blob was \
         readable, so the apply branch ran"
    );
    assert_eq!(
        layer_recovery_metric(&harness, "error"),
        layer_error_pre,
        "`error{{scope=\"layer\"}}` MUST be flat — a tick here means the composed restore \
         errored rather than composing"
    );
    assert_eq!(
        layer_recovery_metric(&harness, "not_recoverable"),
        layer_not_recoverable_pre,
        "`not_recoverable{{scope=\"layer\"}}` MUST be flat — the lineage verified"
    );

    // ── The lineage was COMPOSED ─────────────────────────────────────────
    assert_lineage_deltas(
        &harness,
        &lineage_pre,
        &[("gathered_composed", 1)],
        "full-wipe composed restore",
    );

    // ── The cursor stamp names the composed head ─────────────────────────
    let header = read_obj_header(&harness, fixture.object_id).await;
    assert_eq!(
        header_u64(&header, "last_checkpoint_seq"),
        fixture.head_seq,
        "last_checkpoint_seq MUST be the COMPOSED HEAD ({}), not the base's snap_seq ({}). The \
         base's value is the regression this line exists to exclude: it under-states what the \
         restore materialized and seeds the writer cursor {} commits too low, which re-issues \
         sequence numbers the replayed tail already used",
        fixture.head_seq,
        fixture.s0,
        fixture.head_seq - fixture.s0
    );
    assert_eq!(
        header_u64_or_zero(&header, "restore_seq_bump"),
        harness.config.checkpoint_seq_safety_margin,
        "with no measured commit span on the hand-put meta the guard band folds to the \
         configured margin, and the restore stamps it so the next assembly can tell the \
         artificial bump apart from real commit traffic"
    );

    // ── Exact state, on the faithful basis ───────────────────────────────
    let restored = read_object_state_for_test(
        harness.tv.client(),
        harness.tv.key_schema(),
        &fixture.object_id.to_string(),
    )
    .await
    .expect("read_object_state_for_test must succeed against the restored layer");
    assert_eq!(
        restored.object_id, fixture.object_id,
        "the restore must rebuild state under the SAME object_id rather than re-minting one"
    );
    // The marker key is ignored by NAME: the recovery branch re-issued that
    // commit after the oracle was captured, so it exists on the restored side
    // only. Its presence is asserted separately just below — the ignore list
    // must not be allowed to hide a missing re-issue.
    assert_faithful_columns(
        &restored,
        &fixture.head_oracle.snapshot,
        FaithfulColumnsOptions::ignoring(&[POST_RECOVERY_MARKER_KEY]),
        "full-wipe composed restore",
    );
    // b.2ti: the restore stamps every entry version a diff builder FILTERS on at
    // its own `seeded_seq` — one write of the whole object — so those persisted
    // columns are a single value rather than the oracle's per-commit spread.
    // `Section::first_version` is the exempt column and comes back preserved;
    // this restore resolved the COMPOSED lineage (`gathered_composed` above), so
    // the split is at the lineage root `fixture.s0`. The floor is the composed
    // head (the full wipe took the stream, so the high-water probe folds to
    // `None`) and the band is the configured margin, as the `restore_seq_bump`
    // assertion just above pins. The marker key is excluded on both halves for
    // the same reason `assert_faithful_columns` excludes it: it is a POST-restore
    // commit, so it legitimately sits one sequence ABOVE the seed. It is a
    // `set_fields`-only delta, so it creates no section and the exempt column is
    // untouched by it.
    // `fixture.s0` is passed twice — once as the presence-range floor, once as
    // the lineage root the exempt column's base-era/tail-era split keys on. One
    // lineage generation here, so they coincide (see the helper's rustdoc).
    let seeded_seq = fixture.head_seq + harness.config.checkpoint_seq_safety_margin;
    assert_restored_entry_versions_flattened(
        &restored,
        &fixture.head_oracle.snapshot,
        RestoredVersionRanges {
            replayed_range_floor: fixture.s0,
            lineage_root: fixture.s0,
        },
        seeded_seq,
        RestoredFirstVersions::ComposedLineage,
        FaithfulColumnsOptions::ignoring(&[POST_RECOVERY_MARKER_KEY]),
        "full-wipe composed restore",
    );
    assert_eq!(
        post_recovery_seq,
        seeded_seq + 1,
        "and the re-issued commit MUST be the very next sequence after the seed — the arithmetic \
         the line above depends on, stated rather than assumed"
    );
    assert!(
        restored
            .fields
            .iter()
            .any(|f| f.key_name == POST_RECOVERY_MARKER_KEY),
        "the re-issued commit MUST have landed: a restore that rebuilt the checkpoint and \
         skipped the re-issue would show every oracle column and no `{POST_RECOVERY_MARKER_KEY}`"
    );

    // ── The external value is reachable byte-identically ─────────────────
    for key in [
        harness.tv.key_schema().ext_meta(&fixture.ext_hash_hex),
        harness.tv.key_schema().ext_data(&fixture.ext_hash_hex),
    ] {
        assert!(
            key_exists(&harness, &key).await,
            "`{key}` MUST be back after the composed restore's SR-3.4 pre-store — the wipe \
             removed the whole ext namespace, so its presence is evidence the restore wrote it"
        );
    }
    let restored_blob: Bytes = harness
        .tv
        .client()
        .pool
        .get(&harness.tv.key_schema().ext_data(&fixture.ext_hash_hex))
        .await
        .expect("GET ext_data must succeed after the restore");
    assert_eq!(
        restored_blob.as_ref(),
        fixture.ext_blob.as_slice(),
        "the restored external value MUST be byte-identical to what the tail-range commit sent"
    );

    // ── The path index is back ───────────────────────────────────────────
    let sessions = list_sessions_at_base_path(&harness).await;
    assert_eq!(
        sessions, fixture.head_oracle.sessions_at_base_path,
        "post-recovery ListSessions({SESSION_BASE_PATH}) MUST return exactly what it returned \
         pre-wipe. ListSessions reads `path_names_index`; without the path-index restore it \
         returns an empty list"
    );

    // ── The object-id counter re-seeded ──────────────────────────────────
    //
    // Only `allocate_object_id` consults the b.b2j repair hook, so this needs a
    // CreateLayer — `handle_update_layer` above does not allocate. Driven here,
    // after the two-hop restore, so the whole post-wipe composition is in one
    // test rather than split across two fixtures.
    let margin = harness.config.checkpoint_counter_safety_margin;
    let ompf = delta_to_ompf_bytes(&empty_delta(), &[]);
    let stream = streaming_create(vec![
        make_create_header(&fixture.session_id, "post-wipe-counter-probe-layer"),
        make_create_data(ompf),
    ]);
    let oid_post = handle_create_layer(&harness.svc, make_streaming_token_request(stream))
        .await
        .expect("post-wipe CreateLayer must succeed")
        .into_inner()
        .object_id;
    assert_eq!(
        oid_post,
        fixture.object_id + margin + 1,
        "the post-wipe allocation must resume above the sidecar-derived floor \
         (max_object_id_observed={} + safety_margin={margin}); got {oid_post}. A value of 1 is \
         the b.b2j bug verbatim — a counter re-established at zero, re-minting ids the \
         surviving checkpoint blobs still claim",
        fixture.object_id
    );
    assert_eq!(
        object_id_counter_repair_metric(&harness, "unhooked_initialized"),
        0,
        "the HOOKED path must be what ran; this cell increments only on `allocate_object_id`'s \
         no-hook arm"
    );
    assert_eq!(
        object_id_counter_repair_metric(&harness, "repaired_seeded"),
        1,
        "exactly one repair must have re-seeded from the counters sidecar — every increment of \
         this cell is evidence the Valkey keyspace was lost under a running pod"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// Arm 2 — the restored never-quiescent layer captures its next tail
// ═════════════════════════════════════════════════════════════════════════════

/// The Epic's second half: after a whole-keyspace wipe and a production-handler
/// restore, a layer that is never quiescent again CAPTURES its next tail through
/// the real periodic loop rather than refusing.
///
/// The chain this composes, none of which any narrower test runs end to end:
/// session hop → composed lineage restore → SR-3.6 gap re-base (a fresh `Base`
/// at `seeded_seq`, and `restore_seq_bump` RETIRED) → the loop's assembly
/// persisting nothing (the unprovable-cut seam) → the capture fallback → a tail
/// whose `from_seq` is exactly the SR-3.6 re-based cursor.
///
/// Without the re-base, `refused_seq_gap` ticks on every skipped tick and
/// `captured` never does — which is precisely Test 3's arm.
///
/// Compiled only under `test-checkpoint-fault-injection` (the unprovable-cut
/// seam, see the module header); without the feature it is compiled out, not
/// skipped.
#[cfg(feature = "test-checkpoint-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_restored_never_quiescent_layer_captures_its_next_tail_instead_of_refusing() {
    let mut harness = setup_tail_harness(None).await;

    let fixture = setup_composed_lineage(&mut harness, "next-capture").await;
    wait_for_writer_idle(&harness.session_checkpoint_tracker, STORE_DEADLINE_SECS).await;

    full_valkey_wipe(&harness.tv).await;
    harness
        .svc
        .session_base_path_cache
        .invalidate(&fixture.session_id);

    // `handle_stat_layer` rather than `handle_update_layer`: StatLayer is
    // READ-ONLY, so `obj:hdr seq` is exactly the restore's `seeded_seq` when it
    // returns and the re-base's target is directly readable. Test 1 owns the
    // commit-re-issue path.
    handle_stat_layer(
        &harness.svc,
        stat_request(&fixture.session_id, &fixture.layer_name),
    )
    .await
    .unwrap_or_else(|status| {
        panic!(
            "StatLayer after the whole-keyspace wipe MUST run both recovery hops and return \
             Ok, got {:?}: {}",
            status.code(),
            status.message()
        )
    });

    // Drain the fire-and-forget re-base task before reading anything it stamps.
    wait_for_writer_idle(&harness.session_checkpoint_tracker, STORE_DEADLINE_SECS).await;

    let header = read_obj_header(&harness, fixture.object_id).await;
    let seeded_seq = header_u64(&header, "seq");
    assert!(
        seeded_seq > fixture.head_seq,
        "post-b.8y5 the guard band is unconditional, so the restore must seed the writer cursor \
         strictly above the composed head ({seeded_seq} vs {})",
        fixture.head_seq
    );
    assert_eq!(
        header_u64_or_zero(&header, "restore_seq_bump"),
        0,
        "the SR-3.6 re-base must RETIRE the fossil guard's input. While this column is non-zero \
         `tail_capture` refuses PostRestoreUnrebased BEFORE the stream read, on every tick, \
         forever — the DR-coverage loss this arm exists to disprove"
    );
    assert_eq!(
        header_u64(&header, "last_checkpoint_seq"),
        seeded_seq,
        "a successful re-base advances the cursor to the Base it just wrote"
    );
    let generations = listing(&harness, fixture.object_id).await;
    assert!(
        generations
            .iter()
            .any(|h| h.kind == CheckpointKind::Base && h.snap_seq == seeded_seq),
        "the re-base must have landed a Base at the seeded cursor ({seeded_seq}) — that is what \
         puts S0 above every fossil so the next capture's window opens on live commits. \
         Listing: {generations:?}"
    );

    // ── The drive: a never-quiescent layer met by the real periodic loop ──
    let before = gather_tail_outcome_cells(&harness.metrics);
    let writer = HotWriter::spawn(
        &harness.tv,
        &fixture.session_id,
        &fixture.layer_name,
        fixture.object_id,
    );
    writer.warm_up().await;
    let (cancel, loop_handle) = spawn_tail_loop(&harness, fixture.object_id).await;

    let captured = poll_tail_capture_landed(&harness.metrics, &before, CAPTURE_DEADLINE).await;

    let writer_report = writer.report();
    writer.stop_and_join().await;
    drain_loop(cancel, loop_handle).await;

    let after = gather_tail_outcome_cells(&harness.metrics);
    let generations = listing(&harness, fixture.object_id).await;
    assert!(
        captured,
        "the restored never-quiescent layer's next tail capture MUST SUCCEED within \
         {CAPTURE_DEADLINE:?}. {writer_report}. Listing: {generations:?}. Outcome cells before: \
         {before:?}, after: {after:?}. A rising `refused_seq_gap` means the re-base did not \
         retire `restore_seq_bump`; a rising `refused_no_base` means it did not land a Base"
    );
    assert!(
        tail_capture_landed_delta(&before, &after) >= 1.0,
        "`live_edit_checkpoint_tail_total{{outcome=\"captured\"|\"captured_capped\"}}` MUST \
         have ticked at least once over the drive; before {before:?}, after {after:?}"
    );
    assert_eq!(
        tail_delta(&before, &after, "refused_seq_gap"),
        0.0,
        "`refused_seq_gap` MUST be flat over the drive — the re-base retired the bump before \
         the loop was spawned, so the post-restore fossil guard has nothing to fire on. \
         before {before:?}, after {after:?}"
    );
    assert_eq!(
        tail_delta(&before, &after, "refused_trimmed"),
        0.0,
        "`refused_trimmed` MUST be flat — the hot writer's own LayerStore retains \
         {WRITER_STREAM_MAXLEN} entries, so trimming cannot reach the capture window's floor. \
         before {before:?}, after {after:?}"
    );

    // ── The captured chain roots at the SR-3.6 re-based cursor ────────────
    assert_captured_chain_roots_at(&harness, &generations, seeded_seq).await;
}

// ═════════════════════════════════════════════════════════════════════════════
// Arm 3 — a failed re-base refuses observably, then converges
// ═════════════════════════════════════════════════════════════════════════════

/// The bootstrap-failure arm the Task names: "capture refuses until a base
/// lands, then converges".
///
/// Three phases, each with its own before/after metric window:
///
/// 1. The re-base PUT fails. The RESTORE still succeeds — a failed re-base
///    stamps nothing and is not allowed to fail the DR path — but
///    `restore_seq_bump` is never retired.
/// 2. The layer, still never quiescent, is met by the real periodic loop. Every
///    eligible tick refuses **observably**: `refused_seq_gap` rises and
///    `captured` stays flat. The refusal SHAPE is pinned by
///    [`assert_landed_refusal_label_is_seq_gap`] against the `restore_seq_bump`
///    read at that moment.
/// 3. The fault is cleared and the seam disarmed (the regression is gone), so
///    the next tick's proven read lands a verified base — which clears
///    `restore_seq_bump` in the same HSET that advances
///    `last_checkpoint_seq`. With the seam re-armed and the layer hot again it
///    CONVERGES: a tail is captured and `refused_seq_gap` stops advancing.
///
/// Compiled only under `test-checkpoint-fault-injection` (the unprovable-cut
/// seam, see the module header); without the feature it is compiled out, not
/// skipped.
#[cfg(feature = "test-checkpoint-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_failed_rebase_refuses_capture_observably_then_converges_once_a_base_lands() {
    let control = Arc::new(RebasePutControlStore::new());
    let store: Arc<dyn CheckpointStore> = Arc::clone(&control) as Arc<dyn CheckpointStore>;
    let mut harness = setup_tail_harness(Some(store)).await;

    let fixture = setup_composed_lineage(&mut harness, "rebase-failure").await;
    wait_for_writer_idle(&harness.session_checkpoint_tracker, STORE_DEADLINE_SECS).await;

    // ── Phase 1: arm the fault, then restore ─────────────────────────────
    //
    // Armed only now — the lineage is planted and the periodic loop is fenced
    // (see `setup_composed_lineage`), so the first `Base` PUT the wrapper sees
    // while armed is necessarily the re-base.
    let attempts_before = control.attempts().len();
    control.arm();

    full_valkey_wipe(&harness.tv).await;
    harness
        .svc
        .session_base_path_cache
        .invalidate(&fixture.session_id);

    handle_stat_layer(
        &harness.svc,
        stat_request(&fixture.session_id, &fixture.layer_name),
    )
    .await
    .unwrap_or_else(|status| {
        panic!(
            "the restore MUST still succeed when the re-base PUT fails — a failed re-base \
             stamps nothing and must not fail the DR path. Got {:?}: {}",
            status.code(),
            status.message()
        )
    });
    wait_for_writer_idle(&harness.session_checkpoint_tracker, STORE_DEADLINE_SECS).await;

    let header = read_obj_header(&harness, fixture.object_id).await;
    let seeded_seq = header_u64(&header, "seq");
    let outstanding_bump = header_u64_or_zero(&header, "restore_seq_bump");
    assert_eq!(
        outstanding_bump, harness.config.checkpoint_seq_safety_margin,
        "a FAILED re-base must leave the restore's own stamp UNCHANGED — the bump is simply \
         never retired. A zero here would mean the re-base stamped despite its PUT failing"
    );
    assert_eq!(
        header_u64(&header, "last_checkpoint_seq"),
        fixture.head_seq,
        "and `last_checkpoint_seq` must still name the composed head the restore materialized, \
         not the Base the re-base failed to write"
    );

    let rebase_puts = control.base_puts_since(attempts_before);
    assert_eq!(
        rebase_puts.len(),
        1,
        "exactly one Base PUT must have been issued while armed — the re-base's. Got \
         {rebase_puts:?}"
    );
    assert_eq!(
        rebase_puts[0],
        PutAttempt {
            object_id: fixture.object_id,
            snap_seq: seeded_seq,
            kind: CheckpointKind::Base,
        },
        "the failed PUT must be the re-base's, addressed at the restore's seeded cursor"
    );
    let generations = listing(&harness, fixture.object_id).await;
    assert!(
        !generations
            .iter()
            .any(|h| h.kind == CheckpointKind::Base && h.snap_seq == seeded_seq),
        "the failed PUT must have persisted NOTHING at {seeded_seq}. Listing: {generations:?}"
    );

    // ── Phase 2: every eligible tick refuses, observably ─────────────────
    let before_refusal = gather_tail_outcome_cells(&harness.metrics);
    let writer = HotWriter::spawn(
        &harness.tv,
        &fixture.session_id,
        &fixture.layer_name,
        fixture.object_id,
    );
    writer.warm_up().await;
    let (cancel, loop_handle) = spawn_tail_loop(&harness, fixture.object_id).await;

    let refused = poll_tail_outcome_rose(
        &harness.metrics,
        &before_refusal,
        "refused_seq_gap",
        REFUSAL_DEADLINE,
    )
    .await;

    // Read the guard's input at the moment of the refusal, BEFORE anything is
    // healed: it is what makes `PostRestoreUnrebased` the only variant that
    // could have ticked the cell (the guard returns before the stream read, so
    // the classifier's `Trimmed` is unreachable).
    let bump_at_refusal = header_u64_or_zero(
        &read_obj_header(&harness, fixture.object_id).await,
        "restore_seq_bump",
    );
    let after_refusal = gather_tail_outcome_cells(&harness.metrics);
    let writer_report = writer.report();
    writer.stop_and_join().await;

    assert!(
        refused,
        "with the re-base failed, every eligible tick MUST refuse observably within \
         {REFUSAL_DEADLINE:?}. {writer_report}. Outcome cells before: {before_refusal:?}, \
         after: {after_refusal:?}"
    );
    assert!(
        tail_delta(&before_refusal, &after_refusal, "refused_seq_gap") >= 1.0,
        "`refused_seq_gap` MUST have ticked; before {before_refusal:?}, after {after_refusal:?}"
    );
    assert_eq!(
        tail_capture_landed_delta(&before_refusal, &after_refusal),
        0.0,
        "and NOTHING may have been captured while the fossil guard is outstanding — a capture \
         here would have swept abandoned-timeline commits into a generation claiming to extend \
         the live lineage. BOTH capture spellings are read: a cap-bounded capture persists the \
         same fossil-bearing artifact an uncapped one would, so `captured_capped` is the identical \
         violation. before {before_refusal:?}, after {after_refusal:?}"
    );
    assert_landed_refusal_label_is_seq_gap(bump_at_refusal);

    // ── Phase 3: heal, let a base land, and converge ─────────────────────
    //
    // The seam is disarmed, so the next tick's proven read is not judged
    // regressed and the loop writes a verified base, which clears
    // `restore_seq_bump` in the same HSET that advances `last_checkpoint_seq`
    // (`CheckpointCursorStamp`'s documented secondary clearer, for exactly the
    // case where the re-base failed or was shed).
    control.heal();
    harness
        .tv
        .client()
        .disarm_unprovable_checkpoint_cut_for_test(fixture.object_id);

    let harness_ref = &harness;
    let object_id = fixture.object_id;
    let converged_base = poll_until(CONVERGENCE_DEADLINE, || async move {
        let header = read_obj_header(harness_ref, object_id).await;
        header_u64_or_zero(&header, "restore_seq_bump") == 0
            && header_u64_or_zero(&header, "last_checkpoint_seq") > 0
    })
    .await;
    let post_heal_header = read_obj_header(&harness, fixture.object_id).await;
    let post_heal_listing = listing(&harness, fixture.object_id).await;
    assert!(
        converged_base,
        "once the store heals and the seam is disarmed, the periodic loop MUST land a verified \
         base and retire the outstanding bump within {CONVERGENCE_DEADLINE:?}. Header: \
         {post_heal_header:?}. Listing: {post_heal_listing:?}"
    );
    let healed_cursor = header_u64(&post_heal_header, "last_checkpoint_seq");
    assert!(
        post_heal_listing
            .iter()
            .any(|h| h.kind == CheckpointKind::Base && h.snap_seq == healed_cursor),
        "the retired bump must be explained by a real Base at the advanced cursor \
         ({healed_cursor}). Listing: {post_heal_listing:?}"
    );

    // Now the layer's assemblies are skipped again and it goes hot: it must
    // CAPTURE, and stop refusing.
    harness
        .tv
        .client()
        .arm_unprovable_checkpoint_cut_for_test(fixture.object_id);
    let before_convergence = gather_tail_outcome_cells(&harness.metrics);
    let writer = HotWriter::spawn(
        &harness.tv,
        &fixture.session_id,
        &fixture.layer_name,
        fixture.object_id,
    );
    writer.warm_up().await;

    let captured =
        poll_tail_capture_landed(&harness.metrics, &before_convergence, CAPTURE_DEADLINE).await;

    let writer_report = writer.report();
    writer.stop_and_join().await;
    drain_loop(cancel, loop_handle).await;

    let after_convergence = gather_tail_outcome_cells(&harness.metrics);
    let final_listing = listing(&harness, fixture.object_id).await;
    assert!(
        captured,
        "after a base landed, the layer MUST converge and capture its next tail within \
         {CAPTURE_DEADLINE:?}. {writer_report}. Listing: {final_listing:?}. Outcome cells \
         before: {before_convergence:?}, after: {after_convergence:?}"
    );
    assert!(
        tail_capture_landed_delta(&before_convergence, &after_convergence) >= 1.0,
        "a capture (`captured` or `captured_capped`) MUST tick over the convergence drive; \
         before {before_convergence:?}, after {after_convergence:?}"
    );
    assert_eq!(
        tail_delta(&before_convergence, &after_convergence, "refused_seq_gap"),
        0.0,
        "and `refused_seq_gap` MUST STOP advancing once the bump is retired — a still-rising \
         cell means the layer never actually recovered its DR coverage. before \
         {before_convergence:?}, after {after_convergence:?}"
    );
}

// ═════════════════════════════════════════════════════════════════════════════
// Arm 4 — the surviving-stream shape
// ═════════════════════════════════════════════════════════════════════════════

/// The SR-3.6 surviving-stream shape: a PARTIAL state loss leaves
/// `le:{oid}:obj:stream` intact, so the restore's plausibility anchor must fold
/// the surviving fossils and seed strictly above every one of them — after which
/// the re-base puts S0 there and the next capture's window opens on live
/// commits only.
///
/// Not literally under `full_valkey_wipe`, and could not be: that helper's SCAN
/// + UNLINK over `tv.prefix()*` necessarily takes the stream, so it can only
/// produce the fresh-stream shape. The Task says "both shapes where feasible";
/// this is the feasible form. See the module header.
///
/// The refusal shape is unchanged by the stream's survival — the post-restore
/// guard runs before the stream read — which is why this test asserts the same
/// `refused_seq_gap == 0` window as Test 2 rather than a `Trimmed` variant.
///
/// Compiled only under `test-checkpoint-fault-injection` (the unprovable-cut
/// seam, see the module header); without the feature it is compiled out, not
/// skipped.
#[cfg(feature = "test-checkpoint-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_surviving_stream_restore_folds_the_fossils_and_captures_its_next_tail() {
    let mut harness = setup_tail_harness(None).await;

    let fixture = setup_composed_lineage(&mut harness, "surviving-stream").await;

    // Real commits ABOVE the hand-put tail's head that NO durable generation
    // covers: the abandoned-timeline entries the surviving stream keeps. Without
    // them the stream's high-water would equal the composed head and this shape
    // would degenerate into Test 2's.
    let fossil_seq = drive_fossil_commits(&harness, &fixture).await;
    wait_for_writer_idle(&harness.session_checkpoint_tracker, STORE_DEADLINE_SECS).await;

    let stream_key = wipe_layer_state_preserving_stream(&harness.tv, fixture.object_id).await;
    let fossil_top = stream_top_seq(&harness, fixture.object_id).await;
    assert_eq!(
        fossil_top, fossil_seq,
        "fixture precondition: the surviving stream `{stream_key}` must still top out at the \
         last pre-loss commit ({fossil_seq})"
    );
    assert!(
        fossil_top > fixture.head_seq,
        "fixture precondition: the fossils must sit ABOVE the composed head ({fossil_top} vs \
         {}), otherwise the high-water probe contributes nothing and this shape is \
         indistinguishable from the fresh-stream one",
        fixture.head_seq
    );

    handle_stat_layer(
        &harness.svc,
        stat_request(&fixture.session_id, &fixture.layer_name),
    )
    .await
    .unwrap_or_else(|status| {
        panic!(
            "StatLayer after the partial loss MUST restore the composed lineage and return Ok, \
             got {:?}: {}",
            status.code(),
            status.message()
        )
    });
    wait_for_writer_idle(&harness.session_checkpoint_tracker, STORE_DEADLINE_SECS).await;

    let header = read_obj_header(&harness, fixture.object_id).await;
    let seeded_seq = header_u64(&header, "seq");
    assert!(
        seeded_seq > fossil_top,
        "the restore's plausibility anchor MUST fold the surviving fossils: the seeded cursor \
         ({seeded_seq}) has to sit STRICTLY ABOVE the stream's last-generated id \
         ({fossil_top}). At or below it, both commit paths' XADD at `{{new_seq}}-0` wedges on \
         `ERR The ID specified in XADD is equal or smaller than the target stream top item`"
    );
    assert!(
        seeded_seq > fixture.head_seq + harness.config.checkpoint_seq_safety_margin,
        "and it must exceed `head + margin` ({}), which is where a folded-to-None probe would \
         have left it — a non-degeneracy check that this case really is about the shape it \
         names",
        fixture.head_seq + harness.config.checkpoint_seq_safety_margin
    );
    assert_eq!(
        header_u64_or_zero(&header, "restore_seq_bump"),
        0,
        "the SR-3.6 re-base must retire the fossil guard on this shape too — the guard runs \
         before the stream read, so a surviving stream does not change the refusal shape"
    );
    let generations = listing(&harness, fixture.object_id).await;
    assert!(
        generations
            .iter()
            .any(|h| h.kind == CheckpointKind::Base && h.snap_seq == seeded_seq),
        "the re-base must have landed a Base at {seeded_seq}, i.e. above every fossil. \
         Listing: {generations:?}"
    );

    // ── The next capture, through the real periodic loop ──────────────────
    let before = gather_tail_outcome_cells(&harness.metrics);
    let writer = HotWriter::spawn(
        &harness.tv,
        &fixture.session_id,
        &fixture.layer_name,
        fixture.object_id,
    );
    writer.warm_up().await;
    let (cancel, loop_handle) = spawn_tail_loop(&harness, fixture.object_id).await;

    let captured = poll_tail_capture_landed(&harness.metrics, &before, CAPTURE_DEADLINE).await;

    let writer_report = writer.report();
    writer.stop_and_join().await;
    drain_loop(cancel, loop_handle).await;

    let after = gather_tail_outcome_cells(&harness.metrics);
    let generations = listing(&harness, fixture.object_id).await;
    assert!(
        captured,
        "the surviving-stream restore's next capture MUST SUCCEED within {CAPTURE_DEADLINE:?}. \
         {writer_report}. Listing: {generations:?}. Outcome cells before: {before:?}, after: \
         {after:?}"
    );
    assert_eq!(
        tail_delta(&before, &after, "refused_seq_gap"),
        0.0,
        "`refused_seq_gap` MUST be flat — the re-base retired the bump before the loop ran. \
         before {before:?}, after {after:?}"
    );
    assert_eq!(
        tail_delta(&before, &after, "refused_trimmed"),
        0.0,
        "`refused_trimmed` MUST be flat: S0 sits above every fossil, so the read window opens \
         on live entries the writer's own {WRITER_STREAM_MAXLEN}-entry retention keeps. before \
         {before:?}, after {after:?}"
    );

    // Above the SR-3.6 re-based cursor, so this can only be the drive's own Tail and
    // never the fixture's hand-put lineage head.
    // The whole point of SR-3.6 on this shape: the chain the drive captured
    // opens at the SR-3.6 re-based cursor, i.e. ABOVE every surviving fossil.
    assert_captured_chain_roots_at(&harness, &generations, seeded_seq).await;
}
