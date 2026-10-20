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

//! PRD AC-5 end-to-end — the SR-7.1 creation-time first-base bootstrap driven
//! through the REAL `CreateLayer` commit path and the REAL periodic checkpoint
//! loop. Subtask `t3.dfh.57.1u.e6`, Part 1.
//!
//! ## What this file adds over `first_base_bootstrap_test.rs`
//!
//! That module drives `BaseBootstrap::run_once` / `::spawn_creation_base`
//! directly and explicitly fences off the loop ("No checkpoint loop", "No
//! `refused_no_base`, no tail capture"). Everything here is on the other side of
//! that fence: the bootstrap is reached only via `handle_create_layer` on the
//! harness's wired `LiveEditService`, and every consequence is observed through
//! the periodic loop — the lineage root a captured tail actually roots at, the
//! `refused_no_base` outcome cell a failed bootstrap leaves behind, and the
//! flag-flip transition signature the rollout recipe describes.
//!
//! Three arms, one per AC-5 obligation:
//!
//! 1. **Success** — a layer that is hot from creation acquires its base at
//!    `snap_seq = 1`, with BOTH header cursor columns stamped in one HSET, and
//!    then, on ticks whose assembly persists nothing, captures a tail whose
//!    `from_seq` is that bootstrap base.
//! 2. **Failure** — the bootstrap PUT fails, so NOTHING is stamped; the loop's
//!    capture on a skipped tick surfaces `refused_no_base`; and the first
//!    eligible tick after the store heals checkpoints the layer normally.
//! 3. **Rollout caveat** — a layer created with the tail flag OFF, met by a loop
//!    started with the flag ON, surfaces `refused_no_base` on a skipped tick.
//!    That is the expected transition signature of enabling the feature on an
//!    existing fleet, not a fault.
//!
//! ## Why this module is feature-gated
//!
//! Every arm reaches the loop's tail-capture fallback, and since every
//! checkpoint read proves its cut (the atomic read, and the fenced read above
//! the atomic-read limits; Spec SDD SR-4.1, SR-11.5) only a header-sequence
//! regression gets a tick there — a hot writer no longer does. The test-only
//! unprovable-cut seam (`ValkeyClient::arm_unprovable_checkpoint_cut_for_test`)
//! is what stages one, and it exists only under
//! `test-checkpoint-fault-injection`, so the whole module is gated on that
//! feature. A gated module compiles to nothing without it
//! (`docs-internal/test-writing-guide.md:75`): run it with the feature (the
//! CI `test:` job's feature set includes it). Each arm arms the seam for its
//! layer once the layer exists, on the harness client, whose clones — the
//! harness loop's included — share it.
//!
//! The failure arm does not use the checkpoint fault injector, though:
//! `CheckpointFaultInjector::maybe_inject` dispatches on a method-name label
//! (`src/checkpoint/adapters/mod.rs:243-255`), and the bootstrap PUT and the
//! loop's own PUT are both `"put_layer_checkpoint"`, so it structurally cannot
//! target one without the other. A [`DelegatingCheckpointStore`]-based newtype
//! keyed on `(snap_seq == 1, kind == Base)` can.
//!
//! [`DelegatingCheckpointStore`]: super::helpers::DelegatingCheckpointStore
//!
//! ## The single-HSET proof, and why it needs the PUT gate
//!
//! `BaseBootstrap` writes `last_checkpoint_seq` and `last_checkpoint_ms` in
//! ONE `HSET` (`src/checkpoint/base_bootstrap.rs`, step 6), so the header can never
//! be observed with the cadence clock set and the sequence cursor absent. The
//! success arm proves that the way Case D proves its negative
//! (`checkpointer_metrics_test.rs:1178-1234`): a concurrent sampler counting
//! observations of that impossible half-written state. One `HGETALL` reads the
//! pair atomically server-side, so a false positive is structurally impossible.
//!
//! The sampler needs the header key, which needs the `object_id`, which is only
//! known once `CreateLayer` returns — by which time the bootstrap task may
//! already have stamped. [`BootstrapPutControlStore`]'s gate closes that: the
//! bootstrap PUT parks until the test releases it, so the sampler is
//! provably running (and provably observing an unstamped header) before the
//! stamp can happen. Stimulus, not synchronisation-by-sleep.
//!
//! ## Bounded waits
//!
//! Every wait in this file has an explicit deadline; none is a bare `sleep` used
//! for synchronisation.
//!
//! | Wait | Bound | What a timeout means |
//! |------|-------|----------------------|
//! | [`BootstrapPutControlStore`] gate | [`GATE_WAIT`] (10 s) | the test never released the gate; the PUT proceeds so the arm fails on an assertion rather than hanging |
//! | `wait_for_writer_idle` (bootstrap task drain) | 10 s, clamped to the helpers' 30 s load-tolerant floor | the bootstrap task never completed |
//! | [`HotWriter::warm_up`] | [`DRIVE_DEADLINE`] | the writer never committed — see its `report()` |
//! | tail / refusal polls ([`poll_until`]) | [`DRIVE_DEADLINE`] (45 s) | the loop never reached the arm; the panic prints the listing and the whole outcome-cell map |
//! | `wait_for_checkpoint_at_least` (post-heal) | 20 s, clamped to the 30 s floor | the healed store never took a generation |
//! | loop drain on teardown | 5 s | a wedged loop fails promptly instead of hanging the binary |
//!
//! [`DRIVE_DEADLINE`] sits well inside `.config/nextest.toml`'s 120 s hard kill,
//! which prints nothing at all when it fires.
//!
//! Sandbox env note: `TestValkey::setup` reads `TEST_VALKEY_URL`; under the
//! project sandbox that variable is preset (CLAUDE.md).

#![cfg(feature = "test-checkpoint-fault-injection")]

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use bytes::Bytes;
use fred::interfaces::{HashesInterface, KeysInterface};
use tokio::sync::oneshot;
use tokio::task::JoinHandle;
use tokio::time::Instant;

use live_edit::checkpoint::adapters::LocalDiskCheckpointStore;
use live_edit::checkpoint::{
    decode_tail_envelope, CheckpointBlob, CheckpointError, CheckpointHandle, CheckpointKind,
    CheckpointMeta, CheckpointStore,
};
use live_edit::grpc::layer::handle_create_layer;
use live_edit::storage::layer::LayerStore;
use live_edit::storage::session::SessionStore;
use live_edit::storage::test_support::TestValkey;
use oodf_ffi::{CommandCreateSection, OwnedDelta, SetField, StructCommand};

use super::helpers::*;
use crate::common::owned_delta::delta_to_ompf_bytes;

// ── Fixture constants ───────────────────────────────────────────────────────

/// The `snap_seq` SR-7.1 roots every lineage at, restated here so the
/// assertions read against a name rather than a bare `1`.
const BOOTSTRAP_SNAP_SEQ: u64 = 1;

/// Sections seeded into every layer this file creates.
///
/// 120 is `drive_captured_tail`'s number and Case D's before it: content every
/// tail record can address. Which arm a tick takes is the unprovable-cut
/// seam's, not the width's — every read proves its cut however hot and wide
/// the layer is (see the module header).
const WIDE_SECTIONS: u64 = 120;

/// Stream retention for the hot writer's own `LayerStore`.
///
/// Deliberately far above the harness's 4096. The writer commits continuously
/// for as long as a drive runs; if `MAXLEN` trimming reached past the capture
/// window's floor (`{S0+1}-0`, i.e. `2-0` for a bootstrap lineage) the capture
/// would refuse `refused_trimmed` — a fixture artifact indistinguishable, at the
/// assertion, from the wiring being broken.
const WRITER_STREAM_MAXLEN: u64 = 1_000_000;

/// Commits the writer must land before the loop is allowed to run.
///
/// A state precondition polled off the writer's own counter, not a timing hack:
/// a loop that meets a layer with no commit above its lineage root has no tail
/// to capture.
const WARM_UP_COMMITS: u64 = 20;

/// Wall-clock ceiling on any single drive in this file.
const DRIVE_DEADLINE: Duration = Duration::from_secs(45);

/// Ceiling on how long a gated bootstrap PUT parks before giving up and
/// proceeding. Only reached when the test forgot to release it.
const GATE_WAIT: Duration = Duration::from_secs(10);

/// Foreign holder of `le:checkpoint:leader` while a test seeds.
///
/// Parking the lock is how the seeding phase is fenced from the harness's own
/// loop (the precedent is `e2e_object_id_preservation_test.rs`): the loop's
/// `try_acquire_leader_lock` CAS leaves a lock it does not own untouched and
/// skips the tick, so "the bootstrap is the only possible writer" is a fact
/// rather than a race against `tokio::time::interval`'s immediate first tick.
const PARKED_LEADER_ID: &str = "e2e-tail-bootstrap-parked-leader";

/// TTL on the parked lock — a backstop; every test releases it explicitly.
const PARKED_LEADER_TTL_SECS: i64 = 120;

const CREATED_BY: &str = "e2e-tail-bootstrap-user";
const WRITER_CLIENT_ID: &str = "e2e-tail-bootstrap-writer";

// ── The store wrapper both instrumented arms share ──────────────────────────

/// One `put_layer_checkpoint` attempt, as the wrapper saw it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PutAttempt {
    object_id: u64,
    snap_seq: u64,
    kind: CheckpointKind,
}

/// `CheckpointStore` newtype that records every layer PUT and can GATE or FAIL
/// the bootstrap PUT specifically — `(snap_seq == 1, kind == Base)`, the one
/// address `BaseBootstrap` writes.
///
/// Composed atop [`DelegatingCheckpointStore`](super::helpers::DelegatingCheckpointStore)
/// so every non-target method forwards verbatim, including the four with
/// non-trivial trait defaults (`read_ext_refs_sidecar`, `list_layer_artifacts`,
/// `delete_layer_temp_artifacts`, `publish_worst_case`). Omitting those would
/// not leave a call un-instrumented, it would SUBSTITUTE behaviour behind the
/// wrapper and make assertions about it vacuously green — the trap that helper's
/// doc records.
///
/// Targeting by `(snap_seq, kind)` rather than by the `test-checkpoint-fault-injection`
/// seam is what keeps this module ungated; see the module header.
struct BootstrapPutControlStore {
    delegate: super::helpers::DelegatingCheckpointStore<LocalDiskCheckpointStore>,
    attempts: Mutex<Vec<PutAttempt>>,
    /// When set, the bootstrap PUT returns `Transient` instead of delegating.
    fail_bootstrap_put: AtomicBool,
    /// Taken by the first bootstrap PUT, which then parks on it (bounded by
    /// [`GATE_WAIT`]) before doing anything else.
    gate: Mutex<Option<oneshot::Receiver<()>>>,
}

impl BootstrapPutControlStore {
    fn new(inner: Arc<LocalDiskCheckpointStore>) -> Self {
        Self {
            delegate: super::helpers::DelegatingCheckpointStore::new(inner),
            attempts: Mutex::new(Vec::new()),
            fail_bootstrap_put: AtomicBool::new(false),
            gate: Mutex::new(None),
        }
    }

    /// Park the next bootstrap PUT until the returned sender fires.
    fn arm_gate(&self) -> oneshot::Sender<()> {
        let (tx, rx) = oneshot::channel();
        *self.gate.lock().unwrap_or_else(|p| p.into_inner()) = Some(rx);
        tx
    }

    fn fail_bootstrap_puts(&self) {
        self.fail_bootstrap_put.store(true, Ordering::SeqCst);
    }

    fn heal(&self) {
        self.fail_bootstrap_put.store(false, Ordering::SeqCst);
    }

    fn attempts(&self) -> Vec<PutAttempt> {
        self.attempts
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .clone()
    }

    fn is_bootstrap_put(snap_seq: u64, meta: &CheckpointMeta) -> bool {
        snap_seq == BOOTSTRAP_SNAP_SEQ && meta.kind == CheckpointKind::Base
    }
}

#[async_trait]
impl CheckpointStore for BootstrapPutControlStore {
    async fn put_layer_checkpoint(
        &self,
        object_id: u64,
        snap_seq: u64,
        ompf: Bytes,
        meta: CheckpointMeta,
    ) -> Result<CheckpointHandle, CheckpointError> {
        // Recorded BEFORE any gate or failure arm: the test wants to know the
        // call was ISSUED, whatever happens to it afterwards.
        self.attempts
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .push(PutAttempt {
                object_id,
                snap_seq,
                kind: meta.kind,
            });

        if Self::is_bootstrap_put(snap_seq, &meta) {
            let gated = self.gate.lock().unwrap_or_else(|p| p.into_inner()).take();
            if let Some(rx) = gated {
                // Bounded: a test that forgot to release must fail on an
                // assertion, not hang the binary until nextest's hard kill.
                let _ = tokio::time::timeout(GATE_WAIT, rx).await;
            }
            if self.fail_bootstrap_put.load(Ordering::SeqCst) {
                return Err(CheckpointError::Transient {
                    source: "test-injected transient failure on the SR-7.1 bootstrap PUT"
                        .to_string()
                        .into(),
                });
            }
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
    // for a generation the delegate holds.
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

    // The four below carry non-trivial trait defaults — see the type's doc.
    async fn read_ext_refs_sidecar(
        &self,
        handle: &CheckpointHandle,
    ) -> Result<std::collections::HashSet<String>, CheckpointError> {
        self.delegate.read_ext_refs_sidecar(handle).await
    }

    async fn list_layer_artifacts(
        &self,
    ) -> Result<Vec<live_edit::checkpoint::LayerArtifact>, CheckpointError> {
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

// ── Leader-lock parking ─────────────────────────────────────────────────────

/// Claim `le:checkpoint:leader` under a foreign id so no tick can land a
/// generation until [`release_leader_lock`] runs.
async fn park_leader_lock(tv: &TestValkey) {
    let key = tv.key_schema().checkpoint_leader_lock();
    let _: () = tv
        .client()
        .pool
        .set(
            &key,
            PARKED_LEADER_ID,
            Some(fred::types::Expiration::EX(PARKED_LEADER_TTL_SECS)),
            None,
            false,
        )
        .await
        .unwrap_or_else(|e| panic!("parking the checkpoint leader lock must succeed: {e}"));
}

/// Release the parked lock, letting the next tick's election run for real.
async fn release_leader_lock(tv: &TestValkey) {
    let key = tv.key_schema().checkpoint_leader_lock();
    let _: i64 = tv
        .client()
        .pool
        .del(&key)
        .await
        .unwrap_or_else(|e| panic!("releasing the checkpoint leader lock must succeed: {e}"));
}

// ── Layer creation through the real handler ─────────────────────────────────

/// Create a session plus a WIDE layer through `handle_create_layer`, i.e. the
/// path that actually calls `svc.base_bootstrap.spawn_creation_base(object_id, state)`.
///
/// Width gives every tail record content to address (see [`WIDE_SECTIONS`]);
/// which arm a tick takes is the unprovable-cut seam's, not the width's.
/// Driving it through the handler rather than `LayerStore::create_layer` is
/// what makes the bootstrap fire at all.
async fn create_wide_layer(
    harness: &CheckpointHarness,
    session_name: &str,
    layer_name: &str,
) -> (String, u64) {
    let session_id =
        SessionStore::new(harness.tv.client().clone(), harness.tv.key_schema().clone())
            .create(session_name, "/stage", CREATED_BY)
            .await
            .expect("session creation must succeed");

    let mut delta = OwnedDelta {
        source_format: "usda".to_string(),
        ..Default::default()
    };
    for section_id in 1..=WIDE_SECTIONS {
        delta
            .struct_commands
            .push(StructCommand::CreateSection(CommandCreateSection {
                parent_id: 0,
                section_id,
                section_name: format!("bootstrap-seed-{section_id}"),
                section_type: 1,
            }));
        delta.set_fields.push(SetField {
            section_id,
            key_name: "seed-field".to_string(),
            value_or_ext_hash: vec![0x5E, 0xED],
            ext_value_size: 0,
            ext_value_index: 0,
            set_order: section_id,
        });
    }

    let stream = streaming_create(vec![
        make_create_header(&session_id, layer_name),
        make_create_data(delta_to_ompf_bytes(&delta, &[])),
    ]);
    let object_id = handle_create_layer(&harness.svc, make_streaming_token_request(stream))
        .await
        .expect("CreateLayer must succeed")
        .into_inner()
        .object_id;

    (session_id, object_id)
}

// ── The hot writer ──────────────────────────────────────────────────────────

/// A writer committing field-set deltas as fast as the store will take them.
///
/// Both halves of the delta shape are load-bearing, and both are
/// `drive_captured_tail`'s: a field-only delta rides `update_layer`'s Lua fast
/// path (a structural command takes a slower path — the structural route on
/// `lua`, or the WATCH/MULTI/EXEC slow path on the rollback backends, which
/// commits roughly an order of magnitude less often), and every commit carries
/// a REAL `relay_ompf` because an `ompf`-less update entry is read by the tail
/// classifier as an EMPTY commit by invariant.
///
/// Its `LayerStore` is its own, at [`WRITER_STREAM_MAXLEN`] — the harness's is
/// bounded at 4096 and would trim the capture window's floor out from under a
/// long drive.
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
                let mut nonce = 10_000u64;
                while !stop.load(Ordering::Relaxed) {
                    let delta = OwnedDelta {
                        set_fields: vec![SetField {
                            // Section 1 exists in every layer this file creates.
                            section_id: 1,
                            key_name: "hot-field".to_string(),
                            value_or_ext_hash: nonce.to_le_bytes().to_vec(),
                            ext_value_size: 0,
                            ext_value_index: 0,
                            set_order: nonce,
                        }],
                        ..Default::default()
                    };
                    let ompf = Bytes::from(delta_to_ompf_bytes(&delta, &[]));
                    // No pacing sleep: the writer must commit as fast as the
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
                        // not reached produce the same "arm never reached"
                        // symptom.
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
    /// an already-hot layer.
    async fn warm_up(&self) {
        let deadline = Instant::now() + DRIVE_DEADLINE;
        while self.commits.load(Ordering::Relaxed) < WARM_UP_COMMITS {
            assert!(
                Instant::now() < deadline,
                "hot writer never reached {WARM_UP_COMMITS} warm-up commits within \
                 {DRIVE_DEADLINE:?} ({})",
                self.report()
            );
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
    }

    async fn stop_and_join(self) {
        self.stop.store(true, Ordering::Relaxed);
        let _ = self.handle.await;
    }
}

// ── Small shared probes ─────────────────────────────────────────────────────

/// Bounded poll — no naked `sleep` used as synchronisation.
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
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
    pred().await
}

/// One `HGETALL` of `le:{object_id}:obj:hdr`.
async fn read_header(tv: &TestValkey, object_id: u64) -> HashMap<String, String> {
    tv.client()
        .pool
        .hgetall::<HashMap<String, String>, _>(tv.key_schema().obj_header(&object_id.to_string()))
        .await
        .expect("HGETALL obj:hdr must succeed")
}

/// Cancel the harness's periodic loop and await its exit. Mirrors the
/// `mem::replace` shape the harness self-guards use, so the harness's own drop
/// stays safe.
async fn drain_harness_loop(harness: &mut CheckpointHarness) {
    harness.cancel.cancel();
    let loop_task = std::mem::replace(&mut harness.loop_handle, tokio::spawn(async {}));
    let _ = tokio::time::timeout(Duration::from_secs(5), loop_task).await;
}

/// Build a harness whose store is a [`BootstrapPutControlStore`] over a real
/// LocalDisk adapter, with the tail feature as given. The harness loop runs on
/// a clone of the harness client, so an arm that needs its fallback arms the
/// unprovable-cut seam on `harness.tv.client()` once its layer exists.
///
/// Returns the wrapper alongside the harness (the harness holds it only as
/// `Arc<dyn CheckpointStore>`) and the `TempDir`, which the caller must keep
/// alive: the LocalDisk root lives under it.
async fn harness_with_controlled_store(
    tail_enabled: bool,
) -> (
    CheckpointHarness,
    Arc<BootstrapPutControlStore>,
    tempfile::TempDir,
) {
    let tempdir = tempfile::tempdir().expect("tempdir creation must succeed");
    let inner = Arc::new(
        LocalDiskCheckpointStore::new(tempdir.path().to_path_buf())
            .expect("LocalDiskCheckpointStore::new must succeed on a fresh root"),
    );
    let control = Arc::new(BootstrapPutControlStore::new(inner));

    let mut opts = CheckpointHarnessOptions::new(1);
    opts.tail.enabled = tail_enabled;
    opts.store = Some(Arc::clone(&control) as Arc<dyn CheckpointStore>);
    let harness = setup_checkpoint_harness_with_options(opts).await;

    (harness, control, tempdir)
}

// ── Arm 1: success ──────────────────────────────────────────────────────────

/// AC-5, success arm: a layer that is hot from creation acquires its lineage
/// root at `snap_seq = 1` through the real `CreateLayer`, stamps BOTH cursor
/// columns in one HSET, and then captures a tail rooted at that base through the
/// real periodic loop.
///
/// The `from_seq == 1` assertion on the decoded envelope is the load-bearing
/// one: it is the only observation that ties the two halves of AC-5 together —
/// the generation the bootstrap wrote IS the lineage head the capture window
/// opened above. A tail rooted anywhere else would mean some other generation
/// (an assembler base, say) supplied S0 and the bootstrap was decorative.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn hot_from_creation_layer_bootstraps_its_base_then_captures_a_tail_rooted_at_it() {
    let (mut harness, control, _tempdir) = harness_with_controlled_store(true).await;
    // Fence the seeding phase: with the lock parked, the loop's immediate first
    // tick cannot land a `snap_seq = 1` base of its own and steal the arm.
    park_leader_lock(&harness.tv).await;

    // Park the bootstrap PUT so the sampler is provably running, and the header
    // provably unstamped, before the single HSET can happen.
    let release_put = control.arm_gate();

    let (session_id, object_id) =
        create_wide_layer(&harness, "e2e-tail-bootstrap-success", "success-layer").await;

    let header_key = harness.tv.key_schema().obj_header(&object_id.to_string());
    let sampler_stop = Arc::new(AtomicBool::new(false));
    let half_written = Arc::new(AtomicUsize::new(0));
    let samples = Arc::new(AtomicUsize::new(0));
    let sampler = {
        let client = harness.tv.client().clone();
        let header_key = header_key.clone();
        let stop = Arc::clone(&sampler_stop);
        let half_written = Arc::clone(&half_written);
        let samples = Arc::clone(&samples);
        tokio::spawn(async move {
            while !stop.load(Ordering::Relaxed) {
                // ONE HGETALL: the server reads the pair atomically, so a false
                // positive from two round-trips is structurally impossible.
                let Ok(header) = client
                    .pool
                    .hgetall::<HashMap<String, String>, _>(&header_key)
                    .await
                else {
                    // The pool is torn down at test exit; stop quietly.
                    return;
                };
                samples.fetch_add(1, Ordering::Relaxed);
                if header.contains_key("last_checkpoint_ms")
                    && !header.contains_key("last_checkpoint_seq")
                {
                    half_written.fetch_add(1, Ordering::Relaxed);
                }
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        })
    };

    // The sampler is live and the stamp has demonstrably not happened yet.
    assert!(
        poll_until(DRIVE_DEADLINE, || async {
            samples.load(Ordering::Relaxed) > 0
        })
        .await,
        "the half-write sampler never took a sample"
    );
    let pre_stamp = read_header(&harness.tv, object_id).await;
    assert!(
        !pre_stamp.contains_key("last_checkpoint_seq")
            && !pre_stamp.contains_key("last_checkpoint_ms"),
        "with the bootstrap PUT still gated and the loop's leader lock parked, neither cursor \
         column can exist yet — a column here means something other than the bootstrap stamps \
         them. Header: {pre_stamp:?}"
    );

    let _ = release_put.send(());
    wait_for_writer_idle(&harness.session_checkpoint_tracker, 10).await;

    sampler_stop.store(true, Ordering::Relaxed);
    let _ = sampler.await;
    assert_eq!(
        half_written.load(Ordering::Relaxed),
        0,
        "observed the impossible half-written cursor pair ({} of {} HGETALL samples had \
         last_checkpoint_ms set with last_checkpoint_seq absent). The bootstrap writes both \
         columns in ONE HSET (base_bootstrap.rs step 6); a single observation means they were split \
         across two writes",
        half_written.load(Ordering::Relaxed),
        samples.load(Ordering::Relaxed)
    );

    // The bootstrap generation itself.
    let generations = harness
        .store
        .list_layer_checkpoints(object_id)
        .await
        .expect("list_layer_checkpoints must succeed");
    assert_eq!(
        generations.len(),
        1,
        "the parked loop cannot write, so the bootstrap must be the ONLY generation. Got \
         {generations:?}"
    );
    assert_eq!(generations[0].kind, CheckpointKind::Base);
    assert_eq!(generations[0].snap_seq, BOOTSTRAP_SNAP_SEQ);
    assert_eq!(generations[0].object_id, object_id);
    assert_eq!(
        control.attempts(),
        vec![PutAttempt {
            object_id,
            snap_seq: BOOTSTRAP_SNAP_SEQ,
            kind: CheckpointKind::Base,
        }],
        "exactly one layer PUT — the bootstrap's — may have been issued during the seeding phase"
    );

    // Both cursor columns, at the bootstrap's sequence.
    let stamped = read_header(&harness.tv, object_id).await;
    assert_eq!(
        stamped.get("last_checkpoint_seq").map(String::as_str),
        Some("1"),
        "the bootstrap must stamp last_checkpoint_seq = 1. Header: {stamped:?}"
    );
    assert!(
        stamped
            .get("last_checkpoint_ms")
            .and_then(|v| v.parse::<u64>().ok())
            .is_some_and(|ms| ms > 0),
        "the bootstrap must stamp a non-zero last_checkpoint_ms cadence clock in the same HSET. \
         Header: {stamped:?}"
    );

    // ── Now the loop half: hot layer, tail path on, assembly skipped ────────
    harness
        .tv
        .client()
        .arm_unprovable_checkpoint_cut_for_test(object_id);
    let writer = HotWriter::spawn(&harness.tv, &session_id, "success-layer", object_id);
    writer.warm_up().await;

    let captured_before = read_tail_captures_landed(&harness.metrics);
    release_leader_lock(&harness.tv).await;

    let store = Arc::clone(&harness.store);
    poll_until(DRIVE_DEADLINE, || {
        let store = Arc::clone(&store);
        async move {
            store
                .list_layer_checkpoints(object_id)
                .await
                .expect("list_layer_checkpoints must succeed during the drive")
                .iter()
                .any(|h| h.kind == CheckpointKind::Tail)
        }
    })
    .await;
    // `poll_until` yields a bool; re-read once for the handle itself so the
    // diagnostic below can print the whole listing either way.
    let listing = store
        .list_layer_checkpoints(object_id)
        .await
        .expect("list_layer_checkpoints must succeed");
    let tail = listing
        .iter()
        .find(|h| h.kind == CheckpointKind::Tail)
        .cloned()
        .unwrap_or_else(|| {
            panic!(
                "no Tail generation landed for object_id={object_id} within {DRIVE_DEADLINE:?} \
             ({}). Listing: {listing:?}. Outcome cells: {:?}. The two usual causes: the seam \
             did not fire (the assembler wrote an ordinary base — look for a second Base above) \
             or the capture refused (a refusal writes nothing; the outcome cells say which arm)",
                writer.report(),
                gather_tail_outcome_cells(&harness.metrics)
            )
        });

    writer.stop_and_join().await;
    drain_harness_loop(&mut harness).await;

    assert!(
        read_tail_captures_landed(&harness.metrics) > captured_before,
        "a Tail landed, so the tick must have ticked one of the two capture outcomes (the \
         capped/uncapped split is the unit's, and either is a success here). Cells: {:?}",
        gather_tail_outcome_cells(&harness.metrics)
    );

    let blob = harness
        .store
        .get_layer_checkpoint(&tail)
        .await
        .expect("get_layer_checkpoint must succeed")
        .expect("the captured tail handle must address a live artifact");
    let envelope =
        decode_tail_envelope(&blob.bytes).expect("the captured artifact must be a tail envelope");
    assert_eq!(
        envelope.header().from_seq,
        BOOTSTRAP_SNAP_SEQ,
        "AC-5's whole claim: the capture window opened above the BOOTSTRAP base, so the tail's \
         exclusive lower bound is snap_seq = 1. A different from_seq means some other generation \
         supplied S0 and the bootstrap was not the lineage root — look for a second Base in the \
         listing, which is what a tick the seam did not skip would leave: {listing:?}"
    );
    assert_eq!(
        blob.meta.base_snap_seq,
        Some(BOOTSTRAP_SNAP_SEQ),
        "the tail meta's lineage root must name the bootstrap base"
    );
    assert_eq!(
        envelope.records().len() as u64,
        envelope.header().to_seq - envelope.header().from_seq,
        "claimed range must EQUAL carried content (b.pw6): {} record(s) for the span ({}, {}]",
        envelope.records().len(),
        envelope.header().from_seq,
        envelope.header().to_seq
    );
}

// ── Arm 2: failure ──────────────────────────────────────────────────────────

/// AC-5, failure arm (SR-7.2): the bootstrap PUT fails, so the layer has no
/// lineage root. Three obligations, in the order an operator meets them:
///
/// 1. Nothing is stamped — the failure arm returns before the HSET, so both
///    cursor columns stay absent. Stamping a cursor for a generation that does
///    not exist would suppress the next tick's re-attempt.
/// 2. The refusal is OBSERVABLE: the loop's next eligible tick whose assembly
///    persists nothing (the unprovable-cut seam) surfaces `refused_no_base` on
///    `live_edit_checkpoint_tail_total`, read as a before/after delta because
///    the cell is zero-seeded.
/// 3. It is not a wedge: once the store heals and the seam is disarmed, the
///    first eligible tick checkpoints it normally.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn bootstrap_put_failure_stamps_nothing_and_the_loop_refuses_no_base_until_the_store_heals() {
    let (mut harness, control, _tempdir) = harness_with_controlled_store(true).await;
    park_leader_lock(&harness.tv).await;
    control.fail_bootstrap_puts();

    let errors_before = read_layer_writes_counter(&harness.metrics, "error");
    let (session_id, object_id) =
        create_wide_layer(&harness, "e2e-tail-bootstrap-failure", "failure-layer").await;
    wait_for_writer_idle(&harness.session_checkpoint_tracker, 10).await;

    // (1) The PUT was issued at the bootstrap address, and it stamped nothing.
    assert_eq!(
        control.attempts(),
        vec![PutAttempt {
            object_id,
            snap_seq: BOOTSTRAP_SNAP_SEQ,
            kind: CheckpointKind::Base,
        }],
        "the bootstrap must have attempted exactly one PUT, at (snap_seq = 1, Base)"
    );
    let generations = harness
        .store
        .list_layer_checkpoints(object_id)
        .await
        .expect("list_layer_checkpoints must succeed");
    assert!(
        generations.is_empty(),
        "the bootstrap PUT failed and the loop is parked, so this object must hold no durable \
         generation. Got {generations:?}"
    );
    let header = read_header(&harness.tv, object_id).await;
    for column in ["last_checkpoint_seq", "last_checkpoint_ms"] {
        let value = header.get(column);
        assert!(
            value.is_none_or(|v| v == "0"),
            "a failed bootstrap must stamp NOTHING — {column} reads {value:?}. Stamping a cursor \
             for a generation that does not exist would fire the next tick's skip-if-no-commits \
             gate on a blob nobody wrote. Header: {header:?}"
        );
    }
    assert!(
        read_layer_writes_counter(&harness.metrics, "error") > errors_before,
        "the failed bootstrap PUT must be visible on checkpoint_writes_total{{error, layer}} — \
         that is what makes a fleet-wide bootstrap failure observable immediately rather than \
         only later as refused_no_base"
    );

    // (2) The loop's capture refuses `no_base` on the layer's own behalf, on a
    //     tick whose assembly is skipped.
    harness
        .tv
        .client()
        .arm_unprovable_checkpoint_cut_for_test(object_id);
    let writer = HotWriter::spawn(&harness.tv, &session_id, "failure-layer", object_id);
    writer.warm_up().await;
    let refused_before = read_tail_counter(&harness.metrics, "refused_no_base");
    release_leader_lock(&harness.tv).await;

    let refused = poll_until(DRIVE_DEADLINE, || async {
        read_tail_counter(&harness.metrics, "refused_no_base") > refused_before
    })
    .await;
    assert!(
        refused,
        "with no base on the store the loop's tail capture must refuse `no_base` (SR-7.2) — the \
         defined-refusal arm of AC-5. It never ticked within {DRIVE_DEADLINE:?} ({}). Outcome \
         cells: {:?}",
        writer.report(),
        gather_tail_outcome_cells(&harness.metrics)
    );

    // (3) Not a wedge: heal the store, disarm the seam (and quiesce the
    //     layer), and the next eligible tick checkpoints it normally.
    control.heal();
    writer.stop_and_join().await;
    harness
        .tv
        .client()
        .disarm_unprovable_checkpoint_cut_for_test(object_id);
    let handle = wait_for_checkpoint_at_least(&harness.store, object_id, 2, 20).await;
    drain_harness_loop(&mut harness).await;
    assert_eq!(
        handle.kind,
        CheckpointKind::Base,
        "the post-heal generation must be an ordinary assembler base"
    );
    assert!(
        handle.snap_seq > BOOTSTRAP_SNAP_SEQ,
        "the post-heal base must cover the writer's commits, not re-attempt the bootstrap's \
         snap_seq = 1; got {handle:?}"
    );
}

// ── Arm 3: the rollout caveat ───────────────────────────────────────────────

/// The flag-flip transition signature the operator recipe describes: a layer
/// created while `LIVE_EDIT_CHECKPOINT_TAIL_ENABLED` was OFF has no bootstrap
/// base, so a loop later started with the flag ON finds no lineage root for it
/// and refuses `refused_no_base` on any tick whose assembly persists nothing.
///
/// Such a tick is a header-sequence regression's — staged here by the
/// unprovable-cut seam — since every read proves its cut (Spec SDD SR-4.1,
/// SR-11.5): the layer's first ordinary eligible tick lands the missing base
/// however hot the layer is, so it does not linger beyond that.
///
/// Pinned as behaviour rather than left to documentation because the observable
/// is identical to the failure arm's — the same cell, ticking for a reason that
/// is NOT a fault. An operator reading `refused_no_base` after enabling the
/// feature needs both readings to exist.
///
/// The harness's own loop is drained rather than re-configured: its tail knobs
/// are derived from the harness `Config` (the same round-trip `main.rs` does),
/// so "created with the flag off" and "loop running with the flag on" cannot be
/// expressed by one harness. The second loop is spawned with the flag on, over
/// the same store and Valkey.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn layer_created_with_the_flag_off_refuses_no_base_on_a_skipped_tick_under_a_flag_on_loop() {
    use live_edit::checkpoint::{spawn_checkpoint_loop, CheckpointLoopConfig, TailCaptureConfig};
    use tokio_util::sync::CancellationToken;

    // This arm's premise is a flag-OFF deployment, so the flag is PINNED rather
    // than inherited: the binary (and therefore harness) default is `true` since
    // b.3n1, and an inherited value would have CreateLayer plant the very
    // bootstrap base this test needs absent.
    let mut opts = CheckpointHarnessOptions::new(1);
    opts.tail = TailCaptureConfig {
        enabled: false,
        ..TailCaptureConfig::default()
    };
    assert!(
        !opts.tail.enabled,
        "fixture precondition: this arm's premise is a flag-OFF deployment; it is vacuous \
         otherwise"
    );
    let mut harness = setup_checkpoint_harness_with_options(opts).await;
    park_leader_lock(&harness.tv).await;

    let (session_id, object_id) =
        create_wide_layer(&harness, "e2e-tail-bootstrap-rollout", "rollout-layer").await;
    wait_for_writer_idle(&harness.session_checkpoint_tracker, 10).await;

    let generations = harness
        .store
        .list_layer_checkpoints(object_id)
        .await
        .expect("list_layer_checkpoints must succeed");
    assert!(
        generations.is_empty(),
        "with the tail flag off the SR-7.1 gate is false, so CreateLayer must leave this object \
         with no generation at all. Got {generations:?}"
    );

    // Replace the flag-off harness loop with a flag-ON one over the same store.
    drain_harness_loop(&mut harness).await;
    let writer = HotWriter::spawn(&harness.tv, &session_id, "rollout-layer", object_id);
    writer.warm_up().await;

    let refused_before = read_tail_counter(&harness.metrics, "refused_no_base");
    let cancel = CancellationToken::new();
    let loop_handle = spawn_checkpoint_loop(
        unprovable_cut_client(&harness.tv, object_id),
        harness.tv.key_schema().clone(),
        Arc::clone(&harness.store),
        CheckpointLoopConfig {
            tick: Duration::from_secs(1),
            interval: Duration::ZERO,
            quiesce: Duration::ZERO,
            retain_generations: 2,
            counters_sync: Duration::from_secs(3600),
            reconcile_interval: Duration::from_secs(3600),
            leader: crate::common::test_leader(SERVICE_LABEL),
            tail: TailCaptureConfig {
                enabled: true,
                ..TailCaptureConfig::default()
            },
        },
        Some(Arc::clone(&harness.metrics)),
        cancel.clone(),
    );
    release_leader_lock(&harness.tv).await;

    let refused = poll_until(DRIVE_DEADLINE, || async {
        read_tail_counter(&harness.metrics, "refused_no_base") > refused_before
    })
    .await;

    // Read the listing at the moment of the observation, before teardown lets
    // any further tick run: the claim is that a skipped tick of an unrooted
    // layer REFUSES, so a tail must not have been captured to reach this point.
    let listing = harness
        .store
        .list_layer_checkpoints(object_id)
        .await
        .expect("list_layer_checkpoints must succeed");

    writer.stop_and_join().await;
    cancel.cancel();
    let _ = tokio::time::timeout(Duration::from_secs(5), loop_handle).await;

    assert!(
        refused,
        "a layer created under the flag-off gate has no lineage root, so a flag-on loop must \
         surface refused_no_base for it. It never ticked within {DRIVE_DEADLINE:?}. Cells: {:?}",
        gather_tail_outcome_cells(&harness.metrics)
    );
    assert!(
        !listing.iter().any(|h| h.kind == CheckpointKind::Tail),
        "the transition signature is a REFUSAL, not a capture: a tail cannot exist for a layer \
         that has no base to root it at. Listing: {listing:?}"
    );
}
