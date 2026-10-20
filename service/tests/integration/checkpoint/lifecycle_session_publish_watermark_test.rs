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

//! b.443 regression coverage — the session-checkpoint publish watermark.
//!
//! # The bug
//!
//! `put_session_checkpoint` is last-writer-wins to ONE fixed path per
//! session, and before b.443 each lifecycle event spawned an independent,
//! unordered dispatch task. So a build that started earlier, stalled, and
//! PUT later durably overwrote a fresher payload — typically one naming a
//! layer the stale payload predates. Nothing re-triggered a repair, so if no
//! further lifecycle event ever landed on that session the loss was
//! permanent and silent.
//!
//! That is not a cosmetic staleness. The session checkpoint is the **root of
//! reachability** for disaster recovery: `apply_checkpoint`'s SR-2.4 guard
//! refuses to restore a layer with no live `layer_name → object_id` mapping,
//! and the session applier's SET loop is the only thing that re-creates
//! those mappings after state loss. A durable session blob missing one
//! mapping makes that layer's own checkpoint permanently unrestorable —
//! with the layer blob sitting right there, intact and unreachable.
//!
//! # Why in-process ordering is not the fix
//!
//! There is no session affinity: concurrent lifecycle RPCs for one session
//! routinely land on DIFFERENT replicas. The per-session serialized dispatch
//! in `session_writer.rs` orders operations *within* a replica; it cannot
//! order them across replicas, and the adapter cannot arbitrate either
//! (`CheckpointMeta` is discarded by LocalDisk and ignored by StorageApi,
//! SR-1.5 forbids parsing the body, and the object surface has no CAS).
//! Valkey is the one thing every replica shares, so the ordering is anchored
//! there: every build mints a monotone stamp atomically with the state read
//! it stamps, and after a successful adapter call the writer publishes that
//! stamp. A refused publish means this operation clobbered a strictly
//! fresher one, and the writer repairs it.
//!
//! **Every test in this file therefore drives TWO writers over ONE shared
//! store and ONE shared Valkey** — that is the deployment shape the bug
//! lives in, and a single-writer test cannot reach it.
//!
//! # The two retry phases
//!
//! One coalesced operation runs in two phases with separate budgets, and the
//! boundary is a correctness boundary. The **adapter phase** (build/mint plus
//! the `put` / `delete`) has nothing durable behind it, so a transient failure
//! retries the whole operation and re-mints. The **publish phase** starts the
//! instant the adapter call LANDS: from there the operation is never re-run,
//! and only `publish_stamp` is retried, carrying the SAME stamp. Several
//! tests below exist purely to pin that boundary, because crossing it is what
//! makes the resurrection bug reachable.
//!
//! # Coverage
//!
//! Repair protocol:
//! - `stale_write_landing_after_fresher_write_is_repaired` — the headline
//!   case. A stale PUT clobbers a fresher one; the durable blob converges
//!   back to the fresh content.
//! - `stale_write_landing_after_delete_is_repaired_by_reissuing_delete` —
//!   the DELETE arm. A stale PUT resurrects a deleted session's checkpoint;
//!   the repair re-issues the DELETE.
//! - `publish_repair_budget_exhausted_ticks_exhausted_and_terminates` — the
//!   give-up arm, including that the repair loop terminates rather than
//!   spinning.
//! - `repair_budget_is_restored_when_a_chain_settles` /
//!   `repair_budget_is_restored_when_a_chain_exhausts` — the budget is per
//!   CHAIN, not per task, on BOTH ways a chain can end.
//!
//! Keyspace loss under the watermark (b.hka (a)):
//! - `stale_build_publishing_after_the_ckpt_key_was_deleted_is_repaired`
//! - `stale_build_publishing_after_the_ckpt_key_expired_is_repaired`
//!
//!   The watermark orders operations only while `le:{sid}:ckpt` SURVIVES.
//!   Lose the key and `seq` restarts at 1, so a build minted before the loss
//!   is left holding a stamp drawn from a counter that no longer exists — and
//!   that stamp outranks the fresh watermark. These two pin the incarnation
//!   token that closes it, once per way the key can vanish: an out-of-band
//!   `DEL` (a wipe, a failover onto a lagging replica) and the 24 h TTL
//!   expiring on an idle session that then sees one more lifecycle event.
//!
//! Dispatcher contract:
//! - `write_for_an_absent_session_touches_no_adapter_and_ticks_no_cell`.
//!
//! Publish-phase failures (`mod publish_faults`, needs
//! `test-checkpoint-fault-injection`):
//! - `a_transient_publish_failure_retries_the_publish_not_the_operation`
//! - `publish_exhaustion_ticks_error_once_and_leaves_the_watermark_unadvanced`
//! - `publish_exhaustion_warns_that_the_blob_is_durable_but_unordered`
//! - `stale_write_resurrecting_a_deleted_session_repairs_through_a_publish_blip`
//!   — the regression that motivated splitting the phases.
//!
//! Sandbox env note: these tests use `TestValkey::setup`, which reads
//! `TEST_VALKEY_URL`; see CLAUDE.md.

use std::collections::HashSet;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use bytes::Bytes;
use fred::interfaces::{HashesInterface, KeysInterface};
use tokio::sync::{mpsc, oneshot};
use tokio_util::task::TaskTracker;

use live_edit::checkpoint::adapters::InMemoryCheckpointStore;
use live_edit::checkpoint::{
    CheckpointBlob, CheckpointError, CheckpointHandle, CheckpointMeta, CheckpointStore,
    SessionCheckpoint, SessionCheckpointWriter,
};
use live_edit::domain::types::ObjectState;
use live_edit::observability::metrics::SharedMetrics;
use live_edit::storage::external_values::MockExternalValueStore;
use live_edit::storage::layer::LayerStore;
use live_edit::storage::session::SessionStore;
use live_edit::storage::test_support::TestValkey;

const SERVICE_LABEL: &str = "b443-watermark-test";
const BASE_PATH: &str = "/b443/watermark";

/// Mirror of `MAX_PUBLISH_REPAIRS` in `src/checkpoint/session_writer.rs` —
/// how many supersession repairs one dispatch task performs before it gives
/// up on the chain and ticks `result="exhausted"`.
///
/// Mirrored rather than imported for the same reason as
/// `MAX_ADAPTER_ATTEMPTS` in `lifecycle_session_writes_error_test.rs`: the
/// production constant is a private module item and exporting it purely for
/// a test would widen the dispatcher's surface. The exhausted-arm test below
/// fails loudly with observed-vs-expected counts if the two drift.
const MAX_PUBLISH_REPAIRS: u64 = 3;

// =============================================================================
// A checkpoint store whose Nth `put_session_checkpoint` can be held open.
// =============================================================================

/// Wraps an in-memory store and lets the test hold ONE chosen
/// `put_session_checkpoint` call open indefinitely, which is how these tests
/// manufacture the "an older build's PUT lands last" interleaving
/// deterministically rather than by racing sleeps.
///
/// The gated call signals `reached` on entry and then awaits `release`, so
/// the test can (a) know exactly when the stale writer is parked inside the
/// adapter, (b) drive the fresher operation to completion meanwhile, and
/// (c) choose the instant the stale write lands. Every other call — before
/// the gated one and after it, including the repair PUT the stale writer
/// issues once it discovers it was superseded — passes straight through.
///
/// `oneshot` rather than `Notify`: a `Notify` permit can be lost if the
/// notify runs before the wait, which would deadlock the test on a timing
/// change. A `oneshot` carries its value regardless of ordering.
struct GatedSessionPutStore {
    delegate: super::helpers::DelegatingCheckpointStore<InMemoryCheckpointStore>,
    /// 1-based index of the `put_session_checkpoint` call to hold open.
    gate_on_call: usize,
    /// When set, EVERY `put_session_checkpoint` announces itself here and
    /// then waits for the test to release it — see [`PutStep`].
    steps: Option<mpsc::UnboundedSender<PutStep>>,
    put_calls: AtomicUsize,
    delete_calls: AtomicUsize,
    reached_tx: Mutex<Option<oneshot::Sender<()>>>,
    release_rx: Mutex<Option<oneshot::Receiver<()>>>,
}

/// Handles the test keeps for the gated call: await `reached` to learn the
/// stale writer is parked inside the adapter, then send on `release` to let
/// it land.
struct PutGate {
    reached: oneshot::Receiver<()>,
    release: oneshot::Sender<()>,
}

/// One `put_session_checkpoint` call, handed to the test body for
/// step-by-step control.
///
/// A single gate is enough to manufacture "an older build's PUT lands last";
/// the repair-budget tests need MORE control points than that — they have to
/// change the Valkey watermark, and queue further work, at several specific
/// points in ONE dispatch task's lifetime. A rendezvous on every PUT gives
/// exactly that, without a sleep anywhere.
///
/// The PUT is the right control point because the stamp is minted during the
/// build, i.e. strictly before it: whatever the test writes to `le:{sid}:ckpt`
/// while a PUT is parked is what that operation's publish will be judged
/// against.
struct PutStep {
    /// 1-based call index, asserted by [`next_put`] so a drifted interleaving
    /// fails with the step it actually reached instead of hanging.
    call: usize,
    release: oneshot::Sender<()>,
}

impl GatedSessionPutStore {
    fn new(gate_on_call: usize) -> (Arc<Self>, PutGate) {
        let (reached_tx, reached_rx) = oneshot::channel();
        let (release_tx, release_rx) = oneshot::channel();
        let store = Arc::new(Self {
            delegate: super::helpers::DelegatingCheckpointStore::new(Arc::new(
                InMemoryCheckpointStore::new(),
            )),
            gate_on_call,
            steps: None,
            put_calls: AtomicUsize::new(0),
            delete_calls: AtomicUsize::new(0),
            reached_tx: Mutex::new(Some(reached_tx)),
            release_rx: Mutex::new(Some(release_rx)),
        });
        (
            store,
            PutGate {
                reached: reached_rx,
                release: release_tx,
            },
        )
    }

    /// A store with no gating at all — every call passes through.
    fn ungated() -> Arc<Self> {
        // Gate on a call index no test reaches.
        Self::new(usize::MAX).0
    }

    /// A store that parks on EVERY `put_session_checkpoint` until the test
    /// releases it. Drop the receiver to let all further calls run free.
    fn stepping() -> (Arc<Self>, mpsc::UnboundedReceiver<PutStep>) {
        let (steps_tx, steps_rx) = mpsc::unbounded_channel();
        let store = Arc::new(Self {
            delegate: super::helpers::DelegatingCheckpointStore::new(Arc::new(
                InMemoryCheckpointStore::new(),
            )),
            gate_on_call: usize::MAX,
            steps: Some(steps_tx),
            put_calls: AtomicUsize::new(0),
            delete_calls: AtomicUsize::new(0),
            reached_tx: Mutex::new(None),
            release_rx: Mutex::new(None),
        });
        (store, steps_rx)
    }

    fn put_calls(&self) -> usize {
        self.put_calls.load(Ordering::Acquire)
    }

    fn delete_calls(&self) -> usize {
        self.delete_calls.load(Ordering::Acquire)
    }
}

#[async_trait]
impl CheckpointStore for GatedSessionPutStore {
    async fn put_layer_checkpoint(
        &self,
        object_id: u64,
        snap_seq: u64,
        ompf: Bytes,
        meta: CheckpointMeta,
    ) -> Result<CheckpointHandle, CheckpointError> {
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
        let call = self.put_calls.fetch_add(1, Ordering::AcqRel) + 1;
        if call == self.gate_on_call {
            // Announce arrival, then park until the test releases us. The
            // BODY is captured now and written after the release, which is
            // exactly the stale-payload-lands-late shape.
            if let Some(tx) = self
                .reached_tx
                .lock()
                .expect("reached_tx mutex poisoned")
                .take()
            {
                let _ = tx.send(());
            }
            let rx = self
                .release_rx
                .lock()
                .expect("release_rx mutex poisoned")
                .take();
            if let Some(rx) = rx {
                let _ = rx.await;
            }
        }
        if let Some(steps) = &self.steps {
            let (release, release_rx) = oneshot::channel();
            // A closed receiver means the test has stopped driving; fall
            // through rather than deadlocking the dispatch task.
            if steps.send(PutStep { call, release }).is_ok() {
                let _ = release_rx.await;
            }
        }
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
        self.delete_calls.fetch_add(1, Ordering::AcqRel);
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
}

// =============================================================================
// Shared helpers.
// =============================================================================

/// Build a `SessionCheckpointWriter` over `store` on its own `TaskTracker`,
/// standing in for one replica. Two of these over one store + one Valkey is
/// the no-session-affinity shape the watermark exists for.
fn spawn_replica(
    store: &Arc<dyn CheckpointStore>,
    tv: &TestValkey,
    metrics: &SharedMetrics,
    label: &str,
) -> (Arc<SessionCheckpointWriter>, TaskTracker) {
    let tracker = TaskTracker::new();
    let writer = Arc::new(SessionCheckpointWriter::new(
        Arc::clone(store),
        tv.client().clone(),
        tv.key_schema().clone(),
        tracker.clone(),
        Arc::clone(metrics),
        label.to_string(),
        true,
    ));
    (writer, tracker)
}

/// `close()` + `wait()` under a timeout. A hang here is a dispatcher bug —
/// most plausibly a dispatch task that re-spawns instead of looping, or a
/// repair chain with no budget — not a test-only flake, so it must fail
/// fast and loud rather than stalling the suite.
async fn drain(tracker: &TaskTracker, who: &str) {
    tracker.close();
    tokio::time::timeout(Duration::from_secs(20), tracker.wait())
        .await
        .unwrap_or_else(|_| {
            panic!(
                "{who} tracker.wait() must drain inside 20s — a hang points at a \
                 SessionCheckpointWriter dispatch/repair-loop bug"
            )
        });
}

/// Decode the durable session-checkpoint blob, or `None` if the adapter
/// holds no blob for `session_id`.
async fn read_durable(
    store: &Arc<dyn CheckpointStore>,
    session_id: &str,
) -> Option<SessionCheckpoint> {
    let blob = store
        .get_session_checkpoint(session_id)
        .await
        .expect("get_session_checkpoint against the in-memory store must not fail")?;
    Some(
        SessionCheckpoint::from_bytes(&blob.bytes)
            .expect("the dispatcher only ever PUTs bytes produced by SessionCheckpoint::to_bytes"),
    )
}

/// Read one `live_edit_checkpoint_session_publish_superseded_total{result}`
/// cell.
fn superseded(metrics: &SharedMetrics, result: &str) -> u64 {
    live_edit::observability::metrics::test_util::sample_value(
        &live_edit::observability::metrics::test_util::render(metrics),
        "live_edit_checkpoint_session_publish_superseded_total",
        &[("result", result)],
    )
    .unwrap_or(0.0) as u64
}

/// Read one `live_edit_checkpoint_writes_total{result, kind="session"}` cell.
fn session_writes(metrics: &SharedMetrics, result: &str) -> u64 {
    live_edit::observability::metrics::test_util::sample_value(
        &live_edit::observability::metrics::test_util::render(metrics),
        "live_edit_checkpoint_writes_total",
        &[("result", result), ("kind", "session")],
    )
    .unwrap_or(0.0) as u64
}

/// Read one field off `le:{sid}:ckpt`.
async fn ckpt_field(tv: &TestValkey, session_id: &str, field: &str) -> Option<String> {
    tv.client()
        .pool
        .hget(tv.key_schema().session_checkpoint_state(session_id), field)
        .await
        .unwrap_or_else(|e| panic!("HGET {field} on le:{{{session_id}}}:ckpt must succeed: {e}"))
}

/// Stamp value pinned into `pub` to force every publish to be refused.
///
/// Far above any stamp a test can mint (`seq` starts at 1 and increments per
/// build), and a refused publish leaves `pub` untouched, so ONE pin holds for
/// the whole chain until a test explicitly lifts it.
const UNREACHABLE_STAMP: &str = "1000000000";

/// Pin `pub` out of reach, so the next publish — and every publish after it,
/// until [`unpin_watermark`] — is refused.
async fn pin_watermark(tv: &TestValkey, session_id: &str) {
    let _: () = tv
        .client()
        .pool
        .hset(
            tv.key_schema().session_checkpoint_state(session_id),
            vec![
                ("pub", UNREACHABLE_STAMP.to_string()),
                ("pub_kind", "write".to_string()),
            ],
        )
        .await
        .expect("HSET pin must succeed");
}

/// Lift the pin so the next publish is accepted. `seq` is left alone, so the
/// in-flight operation's already-minted stamp still outranks the (now zero)
/// watermark.
async fn unpin_watermark(tv: &TestValkey, session_id: &str) {
    let _: () = tv
        .client()
        .pool
        .hdel(
            tv.key_schema().session_checkpoint_state(session_id),
            vec!["pub", "pub_kind"],
        )
        .await
        .expect("HDEL pin must succeed");
}

/// Receive the next parked `put_session_checkpoint`, asserting it is call
/// `expected`, and hand back its release channel.
///
/// The timeout is the real assertion in the failure case: a dispatcher that
/// stops issuing PUTs earlier than the test expects (the shape a missing
/// repair-budget reset produces) hangs here rather than producing a confusing
/// count mismatch, so the message names the step that never arrived.
async fn next_put(
    steps: &mut mpsc::UnboundedReceiver<PutStep>,
    expected: usize,
) -> oneshot::Sender<()> {
    let step = tokio::time::timeout(Duration::from_secs(20), steps.recv())
        .await
        .unwrap_or_else(|_| {
            panic!(
                "put_session_checkpoint call #{expected} never arrived within 20s — the dispatch \
                 task issued fewer adapter calls than this interleaving requires (most likely a \
                 repair that was never enqueued)"
            )
        })
        .unwrap_or_else(|| {
            panic!("the stepping store was dropped before put_session_checkpoint call #{expected}")
        });
    assert_eq!(
        step.call, expected,
        "the dispatch task's adapter calls arrived out of the expected order"
    );
    step.release
}

/// Create a session with one layer and no checkpoint writer wired.
async fn seed_session(tv: &TestValkey, name: &str) -> String {
    let session_id = SessionStore::new(tv.client().clone(), tv.key_schema().clone())
        .create(name, BASE_PATH, "b443-test-user")
        .await
        .expect("SessionStore::create must succeed");
    create_layer_without_dispatch(tv, &session_id, "L1").await;
    session_id
}

/// Create a layer on `session_id` through a `LayerStore` with NO checkpoint
/// writer wired, so the mutation lands in Valkey without enqueueing a
/// session-checkpoint operation of its own. These tests drive the dispatcher
/// explicitly; an implicit extra enqueue would blur which operation each
/// adapter call belongs to.
async fn create_layer_without_dispatch(tv: &TestValkey, session_id: &str, layer_name: &str) {
    let layer_store = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096);
    let initial = ObjectState {
        sequence_num: 1,
        first_version: 1,
        source_format: "usd".to_string(),
        ..Default::default()
    };
    layer_store
        .create_layer(
            session_id,
            layer_name,
            "b443-test-user",
            &initial,
            Arc::new(MockExternalValueStore::new()),
        )
        .await
        .expect("LayerStore::create_layer must succeed against a healthy Valkey");
}

// =============================================================================
// Test 1 — the headline case: stale write vs fresher write.
// =============================================================================

/// **A stale PUT that clobbers a fresher one is detected and repaired, and
/// the durable blob converges to the fresh content.**
///
/// Interleaving, made deterministic by gating adapter PUT #1:
///
/// 1. Session exists with layer `L1`.
/// 2. Replica A enqueues a write. Its build reads `{L1}` and mints stamp
///    `SA`. Its PUT parks inside the gated adapter.
/// 3. Layer `L2` is created (no dispatch of its own).
/// 4. Replica B enqueues a write. Its build reads `{L1, L2}` and mints
///    `SB > SA` — strictly higher because A minted before it parked. B's PUT
///    lands and its publish is accepted.
/// 5. A is released. Its stale `{L1}` payload lands **on top of** B's, which
///    is the durable loss. Its publish of `SA` is refused, naming `write` as
///    the kind of the fresher operation it clobbered.
/// 6. A repairs: rebuild (minting `SC > SB`), PUT, publish accepted.
///
/// The assertion that matters is step 6's outcome — `L2` is back in the
/// durable blob. Without the watermark the run ends at step 5 with a blob
/// naming only `L1`, and `L2`'s own layer checkpoint is unrestorable
/// forever.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn stale_write_landing_after_fresher_write_is_repaired() {
    let tv = TestValkey::setup().await;
    let metrics: SharedMetrics = Arc::new(crate::common::metrics_scrape_only(SERVICE_LABEL));

    let (gated, gate) = GatedSessionPutStore::new(1);
    let store: Arc<dyn CheckpointStore> = Arc::clone(&gated) as Arc<dyn CheckpointStore>;

    // Session + L1, with no writer wired: the tests below drive the
    // dispatcher explicitly.
    let session_store = SessionStore::new(tv.client().clone(), tv.key_schema().clone());
    let session_id = session_store
        .create("b443-stale-write-session", BASE_PATH, "b443-test-user")
        .await
        .expect("SessionStore::create must succeed");
    create_layer_without_dispatch(&tv, &session_id, "L1").await;

    let (writer_a, tracker_a) = spawn_replica(&store, &tv, &metrics, "replica-a");
    let (writer_b, tracker_b) = spawn_replica(&store, &tv, &metrics, "replica-b");

    // Step 2 — A builds `{L1}`, mints SA, parks in the adapter.
    writer_a.enqueue_session_checkpoint_write(session_id.clone());
    tokio::time::timeout(Duration::from_secs(20), gate.reached)
        .await
        .expect("replica A's put_session_checkpoint must be reached inside 20s")
        .expect("the gated put must signal arrival before parking");

    // Step 3 + 4 — L2 appears, and B writes the fresher payload.
    create_layer_without_dispatch(&tv, &session_id, "L2").await;
    writer_b.enqueue_session_checkpoint_write(session_id.clone());
    drain(&tracker_b, "replica B").await;

    // Sanity: the fresh payload really is durable before A clobbers it.
    // Without this the final assertion could pass for the wrong reason (A's
    // repair rebuilding a payload B never actually managed to write).
    let fresh = read_durable(&store, &session_id)
        .await
        .expect("replica B's write must be durable before replica A is released");
    assert!(
        fresh.layer_mappings.contains_key("L2"),
        "precondition: replica B's payload must name L2; got {:?}",
        fresh.layer_mappings.keys().collect::<Vec<_>>()
    );

    // Step 5 + 6 — the stale write lands, is detected, and repairs itself.
    gate.release
        .send(())
        .expect("the gated put must still be parked and awaiting release");
    drain(&tracker_a, "replica A").await;

    let final_blob = read_durable(&store, &session_id)
        .await
        .expect("a session checkpoint must still be durable after the repair");
    let names: Vec<&String> = final_blob.layer_mappings.keys().collect();
    assert!(
        final_blob.layer_mappings.contains_key("L1")
            && final_blob.layer_mappings.contains_key("L2"),
        "b.443: the durable session checkpoint MUST converge to the FRESH content naming both \
         L1 and L2. Seeing only L1 means the stale build's PUT durably clobbered the fresher \
         one and no repair followed — the exact silent, permanent loss b.443 fixes, which \
         renders L2's own layer checkpoint unrestorable. Got {names:?}"
    );
    // The repair is observable, not just inferable from the final content.
    assert_eq!(
        superseded(&metrics, "write"),
        1,
        "live_edit_checkpoint_session_publish_superseded_total{{result=\"write\"}} MUST tick \
         exactly once — replica A clobbered exactly one fresher WRITE. A zero here with a \
         correct final blob would mean the convergence was luck (e.g. an extra rebuild from \
         some other path) rather than the watermark doing its job"
    );
    assert_eq!(
        superseded(&metrics, "delete"),
        0,
        "no DELETE was involved in this interleaving; a tick on the delete arm means the \
         publish script's kind reporting is wrong"
    );
    assert_eq!(
        superseded(&metrics, "exhausted"),
        0,
        "one repair is well inside the budget; an `exhausted` tick means the repair chain did \
         not settle"
    );

    // Three PUTs: A's stale one, B's fresh one, A's repair.
    let puts = gated.put_calls();
    assert_eq!(
        puts, 3,
        "expected exactly 3 put_session_checkpoint calls (A stale, B fresh, A repair); got \
         {puts}. A count of 2 means no repair PUT was issued"
    );
}

// =============================================================================
// Test 1b — keyspace loss under the watermark (b.hka (a)).
// =============================================================================

/// How `le:{sid}:ckpt` disappears in the two cases below.
///
/// Both are real: the `DEL` stands in for a wipe, an out-of-band delete, or a
/// failover onto a replica that never received the key; the TTL expiry is the
/// 24 h reaper firing on a session that went idle and then saw one more
/// lifecycle event. They are worth separating because the second one is the
/// case `SESSION_CHECKPOINT_STATE_TTL_MS`'s own doc-comment makes a safety
/// claim about — the key is deliberately left out of `SessionStore::delete`'s
/// teardown and reaped by TTL alone, so nothing but this test enforces that the
/// reaper cannot cost a session its newest checkpoint.
#[derive(Clone, Copy)]
enum CkptKeyLoss {
    /// `DEL le:{sid}:ckpt`. Scoped to this fixture's own key — never a
    /// `FLUSHALL` / `FLUSHDB`, which would destroy the fixtures of every other
    /// agent and CI job sharing the backing instance (CLAUDE.md).
    Deleted,
    /// `PEXPIRE le:{sid}:ckpt 1`, then wait for the key to actually go.
    Expired,
}

/// How long [`CkptKeyLoss::Expired`] waits for a 1 ms `PEXPIRE` to take effect.
///
/// A deadline, not a sleep: the healthy path returns on the first or second
/// probe. Sized to be unreachable under ordinary CI-runner load while staying
/// far below nextest's 120 s hard kill, so a genuine hang fails here with a
/// readable message rather than being SIGKILLed with no output at all.
const KEY_EXPIRY_BUDGET: Duration = Duration::from_secs(10);

/// The `seq` value seeded before writer A builds, so A's stamp is 5.
///
/// The point of a *high* pre-loss stamp is that it beats the post-loss
/// watermark outright: after the loss the counter restarts, so writer B mints
/// 1, and 5 > 1 is accepted by any comparison that trusts the stamp alone.
const PRE_LOSS_SEQ: &str = "4";
const PRE_LOSS_STAMP: &str = "5";

/// Drive `le:{sid}:ckpt` out of existence and confirm it is gone.
async fn lose_ckpt_key(tv: &TestValkey, session_id: &str, how: CkptKeyLoss) {
    let key = tv.key_schema().session_checkpoint_state(session_id);
    match how {
        CkptKeyLoss::Deleted => {
            let _: () = tv
                .client()
                .pool
                .del(&key)
                .await
                .expect("DEL of the fixture's own ckpt key must succeed");
        }
        CkptKeyLoss::Expired => {
            let _: () = tv
                .client()
                .pool
                .pexpire(&key, 1, None)
                .await
                .expect("PEXPIRE must succeed");
        }
    }

    let deadline = tokio::time::Instant::now() + KEY_EXPIRY_BUDGET;
    loop {
        let exists: i64 = tv
            .client()
            .pool
            .exists(&key)
            .await
            .expect("EXISTS must succeed");
        if exists == 0 {
            return;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "{key} was still present {KEY_EXPIRY_BUDGET:?} after the loss was requested; the \
             fixture never reached the state the test is about"
        );
        tokio::time::sleep(Duration::from_millis(10)).await;
    }
}

/// **A build minted before `le:{sid}:ckpt` was lost cannot durably clobber a
/// build minted after it.**
///
/// The shared body of the two cases below; `how` is the only difference.
///
/// Interleaving, made deterministic by gating adapter PUT #1 — the same
/// two-writers-one-store shape as the headline test, with a keyspace loss
/// inserted between the two builds:
///
/// 1. Session exists with layer `L1`, and `seq` is seeded to
///    [`PRE_LOSS_SEQ`] so the next mint is [`PRE_LOSS_STAMP`].
/// 2. Replica A enqueues a write. Its build reads `{L1}`, mints stamp 5 under
///    incarnation `IA`, and parks inside the gated adapter.
/// 3. Layer `L2` is created, and `le:{sid}:ckpt` is LOST.
/// 4. Replica B enqueues a write. Its build re-creates the key under a fresh
///    incarnation `IB`, mints stamp **1** — the counter restarted — PUTs
///    `{L1, L2}`, and publishes 1, which is accepted.
/// 5. A is released. Its stale `{L1}` payload lands on top of B's. Its publish
///    carries stamp 5 under `IA`, which is no longer the key's incarnation, so
///    the script refuses it with `stale_incarnation` **having written
///    nothing**.
/// 6. A repairs as a WRITE — the only arm that cannot destroy information,
///    since a stale incarnation says nothing about what else landed — minting
///    stamp 2 under `IB`, PUTting, and publishing successfully.
///
/// **This is the test that fails without the b.hka (a) fix.** Without the
/// incarnation the run ends at step 5 with the comparison `5 > 1`: the stale
/// publish is *accepted*, the durable blob is A's pre-loss `{L1}` payload,
/// `L2`'s own layer checkpoint is permanently unrestorable, and
/// `checkpoint_session_publish_superseded_total` is untouched — no repair is
/// scheduled because nothing was detected. Worse than the b.443 case it
/// resembles: the watermark is left at 5, so the next four legitimate builds
/// are swallowed too.
async fn stale_build_after_ckpt_key_loss_is_repaired(how: CkptKeyLoss, session_name: &str) {
    let tv = TestValkey::setup().await;
    let metrics: SharedMetrics = Arc::new(crate::common::metrics_scrape_only(SERVICE_LABEL));

    let (gated, gate) = GatedSessionPutStore::new(1);
    let store: Arc<dyn CheckpointStore> = Arc::clone(&gated) as Arc<dyn CheckpointStore>;

    let session_store = SessionStore::new(tv.client().clone(), tv.key_schema().clone());
    let session_id = session_store
        .create(session_name, BASE_PATH, "b443-test-user")
        .await
        .expect("SessionStore::create must succeed");
    create_layer_without_dispatch(&tv, &session_id, "L1").await;

    // Step 1 — seed `seq` so the pre-loss build's stamp is comfortably above
    // anything the post-loss counter can reach. `HSET seq` and nothing else:
    // the incarnation is left for replica A's own build to claim, exactly as a
    // key that has only ever been minted against would look.
    let _: () = tv
        .client()
        .pool
        .hset(
            tv.key_schema().session_checkpoint_state(&session_id),
            vec![("seq", PRE_LOSS_SEQ.to_string())],
        )
        .await
        .expect("HSET seq must succeed");

    let (writer_a, tracker_a) = spawn_replica(&store, &tv, &metrics, "replica-a");
    let (writer_b, tracker_b) = spawn_replica(&store, &tv, &metrics, "replica-b");

    // Step 2 — A builds `{L1}`, mints stamp 5 under the key's incarnation, and
    // parks in the adapter.
    writer_a.enqueue_session_checkpoint_write(session_id.clone());
    tokio::time::timeout(Duration::from_secs(20), gate.reached)
        .await
        .expect("replica A's put_session_checkpoint must be reached inside 20s")
        .expect("the gated put must signal arrival before parking");
    assert_eq!(
        ckpt_field(&tv, &session_id, "seq").await.as_deref(),
        Some(PRE_LOSS_STAMP),
        "precondition: replica A must be holding stamp {PRE_LOSS_STAMP}, or the post-loss \
         counter would not be strictly lower and the test would not reach the failure"
    );

    // Step 3 — the fresher content appears, and the coordination key vanishes.
    create_layer_without_dispatch(&tv, &session_id, "L2").await;
    lose_ckpt_key(&tv, &session_id, how).await;

    // Step 4 — B re-creates the key under a NEW incarnation, mints 1, lands.
    writer_b.enqueue_session_checkpoint_write(session_id.clone());
    drain(&tracker_b, "replica B").await;
    assert_eq!(
        ckpt_field(&tv, &session_id, "seq").await.as_deref(),
        Some("1"),
        "precondition: the counter MUST have restarted at 1 — that restart is the whole hazard, \
         and if the key had survived the loss this test would be pinning nothing"
    );
    let fresh = read_durable(&store, &session_id)
        .await
        .expect("replica B's write must be durable before replica A is released");
    assert!(
        fresh.layer_mappings.contains_key("L2"),
        "precondition: replica B's payload must name L2; got {:?}",
        fresh.layer_mappings.keys().collect::<Vec<_>>()
    );

    // Steps 5 + 6 — the pre-loss write lands, is refused as stale, and repairs.
    gate.release
        .send(())
        .expect("the gated put must still be parked and awaiting release");
    drain(&tracker_a, "replica A").await;

    let final_blob = read_durable(&store, &session_id)
        .await
        .expect("a session checkpoint must still be durable after the repair");
    let names: Vec<&String> = final_blob.layer_mappings.keys().collect();
    assert!(
        final_blob.layer_mappings.contains_key("L1")
            && final_blob.layer_mappings.contains_key("L2"),
        "b.hka (a): the durable session checkpoint MUST converge back to the POST-loss content \
         naming both L1 and L2. Seeing only L1 means the pre-loss build's stamp of \
         {PRE_LOSS_STAMP} outranked the restarted watermark of 1 and its payload was accepted as \
         the newest — the silent, permanent loss the incarnation token exists to prevent, which \
         also renders L2's own layer checkpoint unrestorable. Got {names:?}"
    );
    assert_eq!(
        superseded(&metrics, "stale_incarnation"),
        1,
        "the refusal MUST be booked as `stale_incarnation` exactly once. Zero means the publish \
         was accepted and nothing was detected at all (the pre-fix behaviour); a correct blob \
         with a zero counter would mean the convergence was luck rather than the gate doing its \
         job"
    );
    assert_eq!(
        superseded(&metrics, "write"),
        0,
        "and it MUST NOT be booked as an ordinary supersession: `write` means the ordering \
         protocol caught a race between two live builds, which an operator reads as healthy. \
         This is a keyspace-integrity event and needs its own cell to alert on"
    );
    assert_eq!(
        superseded(&metrics, "delete"),
        0,
        "no DELETE was involved in this interleaving"
    );
    assert_eq!(
        superseded(&metrics, "exhausted"),
        0,
        "one repair is well inside the budget; an `exhausted` tick means the repair chain did \
         not settle"
    );

    // Three PUTs: A's stale one, B's fresh one, A's repair.
    let puts = gated.put_calls();
    assert_eq!(
        puts, 3,
        "expected exactly 3 put_session_checkpoint calls (A stale, B fresh, A repair); got \
         {puts}. A count of 2 means no repair PUT was issued"
    );
}

/// The `DEL` arm — a wipe, an out-of-band delete, or a failover onto a replica
/// that never received `le:{sid}:ckpt`.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn stale_build_publishing_after_the_ckpt_key_was_deleted_is_repaired() {
    stale_build_after_ckpt_key_loss_is_repaired(CkptKeyLoss::Deleted, "b.hka-ckpt-key-deleted")
        .await;
}

/// The TTL arm — the same hazard reached through the reaper the key is
/// deliberately left to.
///
/// Worth its own case rather than folding into the `DEL` one: `le:{sid}:ckpt`
/// is excluded from `SessionStore::delete`'s teardown *on purpose* (a repair
/// reads the watermark after a DELETE has landed), so its TTL is the only thing
/// that ever removes it. `SESSION_CHECKPOINT_STATE_TTL_MS`'s doc-comment makes
/// a safety claim about exactly that expiry, and nothing else in the suite
/// enforces it.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn stale_build_publishing_after_the_ckpt_key_expired_is_repaired() {
    stale_build_after_ckpt_key_loss_is_repaired(CkptKeyLoss::Expired, "b.hka-ckpt-key-expired")
        .await;
}

// =============================================================================
// Test 2 — the DELETE arm.
// =============================================================================

/// **A stale PUT that lands after the session was deleted is repaired by
/// re-issuing the DELETE.**
///
/// This is the arm where getting it wrong is worst: the stale PUT
/// *resurrects* a deleted session's checkpoint, and recovery would happily
/// restore a session the operator deleted. Repairing as a rebuild would not
/// help either — `build_session_checkpoint` returns `Ok(None)` for an absent
/// session and exits silently, leaving the resurrected blob in place. The
/// repair therefore has to be driven by the *kind* the watermark reports
/// (`delete`), not by re-running the local operation.
///
/// The DELETE path mints its stamp before its adapter call, exactly as the
/// write path does — otherwise a delete could not be ordered against writes
/// whose builds are already in flight on another replica, and this test
/// would deadlock into the wrong outcome.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn stale_write_landing_after_delete_is_repaired_by_reissuing_delete() {
    let tv = TestValkey::setup().await;
    let metrics: SharedMetrics = Arc::new(crate::common::metrics_scrape_only(SERVICE_LABEL));

    let (gated, gate) = GatedSessionPutStore::new(1);
    let store: Arc<dyn CheckpointStore> = Arc::clone(&gated) as Arc<dyn CheckpointStore>;

    let plain_session_store = SessionStore::new(tv.client().clone(), tv.key_schema().clone());
    let session_id = plain_session_store
        .create("b443-stale-vs-delete-session", BASE_PATH, "b443-test-user")
        .await
        .expect("SessionStore::create must succeed");
    create_layer_without_dispatch(&tv, &session_id, "L1").await;

    let (writer_a, tracker_a) = spawn_replica(&store, &tv, &metrics, "replica-a");
    let (writer_b, tracker_b) = spawn_replica(&store, &tv, &metrics, "replica-b");

    // Replica A builds `{L1}`, mints SA, and parks in the adapter.
    writer_a.enqueue_session_checkpoint_write(session_id.clone());
    tokio::time::timeout(Duration::from_secs(20), gate.reached)
        .await
        .expect("replica A's put_session_checkpoint must be reached inside 20s")
        .expect("the gated put must signal arrival before parking");

    // Replica B deletes the session for real — through `SessionStore::delete`
    // with the writer wired, so the delete flows through the same production
    // hook site a `DeleteSession` RPC uses.
    let deleting_session_store = SessionStore::new(tv.client().clone(), tv.key_schema().clone())
        .with_checkpoint_writer(Arc::clone(&writer_b));
    deleting_session_store
        .delete(&session_id)
        .await
        .expect("SessionStore::delete must succeed for a freshly-created session");
    drain(&tracker_b, "replica B").await;

    // Release the stale write. It resurrects the checkpoint, discovers the
    // supersession, and re-issues the DELETE.
    gate.release
        .send(())
        .expect("the gated put must still be parked and awaiting release");
    drain(&tracker_a, "replica A").await;

    let residual = read_durable(&store, &session_id).await;
    assert!(
        residual.is_none(),
        "b.443: no session checkpoint may survive for a DELETED session. A blob here is the \
         stale in-flight write resurrecting it — recovery would then restore a session the \
         operator deleted. Got {:?}",
        residual.map(|c| c.layer_mappings.keys().cloned().collect::<Vec<_>>())
    );

    assert_eq!(
        superseded(&metrics, "delete"),
        1,
        "live_edit_checkpoint_session_publish_superseded_total{{result=\"delete\"}} MUST tick \
         exactly once — the stale write clobbered exactly one fresher DELETE"
    );
    assert_eq!(
        superseded(&metrics, "write"),
        0,
        "the fresher operation was a DELETE, so reporting `write` here would send the repair \
         down the rebuild arm — which exits silently on an absent session and would leave the \
         resurrected blob in place"
    );

    // Two deletes: replica B's original, replica A's repair.
    let deletes = gated.delete_calls();
    assert_eq!(
        deletes, 2,
        "expected exactly 2 delete_session_checkpoint calls (B's original, A's repair); got \
         {deletes}. A count of 1 means the repair never re-issued the DELETE"
    );
}

// =============================================================================
// Test 3 — the give-up arm.
// =============================================================================

/// **A repair chain that never settles gives up, ticks `exhausted`, and
/// terminates.**
///
/// Driven by pinning the watermark out of reach: `pub` is set far above any
/// stamp the dispatcher can mint, so every publish is refused no matter how
/// many times the task rebuilds. That stands in for the pathological case
/// the budget exists for — an unbounded write storm on one session — without
/// having to actually generate one.
///
/// Two things are being pinned, and the second matters more than the first:
///
/// 1. The budget is `MAX_PUBLISH_REPAIRS` repairs, then exactly one
///    `exhausted` tick. `exhausted` is the alertable cell — `write` /
///    `delete` are the protocol working, `exhausted` means a repair loop
///    gave up and the durable blob is left stale.
/// 2. **The task terminates.** Each repair re-enqueues work on the same
///    session slot, so a missing or mis-reset budget is not a slow path but
///    an infinite loop inside a `TaskTracker` — which would hang graceful
///    shutdown forever. `drain`'s timeout is that assertion.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn publish_repair_budget_exhausted_ticks_exhausted_and_terminates() {
    let tv = TestValkey::setup().await;
    let metrics: SharedMetrics = Arc::new(crate::common::metrics_scrape_only(SERVICE_LABEL));

    let gated = GatedSessionPutStore::ungated();
    let store: Arc<dyn CheckpointStore> = Arc::clone(&gated) as Arc<dyn CheckpointStore>;

    let session_store = SessionStore::new(tv.client().clone(), tv.key_schema().clone());
    let session_id = session_store
        .create("b443-exhausted-session", BASE_PATH, "b443-test-user")
        .await
        .expect("SessionStore::create must succeed");
    create_layer_without_dispatch(&tv, &session_id, "L1").await;

    // Pin the watermark above every reachable stamp. `seq` is left untouched
    // so the read script's `HINCRBY` still starts from 1.
    let ckpt_key = tv.key_schema().session_checkpoint_state(&session_id);
    let _: () = tv
        .client()
        .pool
        .hset(
            &ckpt_key,
            vec![
                ("pub", "1000000000".to_string()),
                ("pub_kind", "write".to_string()),
            ],
        )
        .await
        .expect("HSET against the test Valkey must succeed");

    let (writer, tracker) = spawn_replica(&store, &tv, &metrics, "replica-a");
    writer.enqueue_session_checkpoint_write(session_id.clone());
    drain(&tracker, "replica A").await;

    assert_eq!(
        superseded(&metrics, "write"),
        MAX_PUBLISH_REPAIRS,
        "the repair budget is {MAX_PUBLISH_REPAIRS} attempts, each of which ticks the `write` \
         arm before retrying"
    );
    assert_eq!(
        superseded(&metrics, "exhausted"),
        1,
        "live_edit_checkpoint_session_publish_superseded_total{{result=\"exhausted\"}} MUST tick \
         exactly once when the budget runs out. This is the alertable cell — a steady trickle \
         of `write`/`delete` is the protocol working, `exhausted` means the durable blob is \
         left stale until the next lifecycle event on this session"
    );

    // One initial operation plus one PUT per repair.
    let puts = gated.put_calls() as u64;
    assert_eq!(
        puts,
        MAX_PUBLISH_REPAIRS + 1,
        "expected {} put_session_checkpoint calls (the original plus one per repair); got \
         {puts}",
        MAX_PUBLISH_REPAIRS + 1
    );

    // The refused publishes must not have advanced the watermark — a repair
    // that moved `pub` down to its own stamp would let the NEXT stale write
    // through undetected.
    let pinned: Option<String> = tv
        .client()
        .pool
        .hget(&ckpt_key, "pub")
        .await
        .expect("HGET against the test Valkey must succeed");
    assert_eq!(
        pinned.as_deref(),
        Some(UNREACHABLE_STAMP),
        "a refused publish MUST leave `pub` untouched; lowering it would re-open the window \
         for the next stale write to land undetected"
    );
}

// =============================================================================
// Tests 4 + 5 — the repair budget is per CHAIN, and BOTH ways a chain can end
// restore it.
// =============================================================================

/// **A chain that SETTLES restores the repair budget for the next chain on
/// the same task.**
///
/// `repairs_remaining` is a local of the dispatch task, and a task on a hot
/// session drains many operations before it retires. If the budget were
/// per-task rather than per-chain, one three-deep supersession chain would
/// permanently bar that task from ever repairing again: every subsequent
/// supersession would tick `exhausted` instead of repairing, so a
/// self-healing condition would present as a continuously-firing alert AND
/// the durable blob would stop converging.
///
/// Driven through ONE dispatch task, which is the whole point — draining
/// between the two chains would hand the second chain a fresh task with a
/// fresh budget and the test would pass without the reset existing. The
/// round-2 write is therefore enqueued while chain 1's last operation is
/// parked inside the adapter, so the running task picks it up.
///
/// Each chain is exactly `MAX_PUBLISH_REPAIRS` refusals followed by an
/// accept, so the reset is the ONLY thing standing between this and an
/// `exhausted` tick.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn repair_budget_is_restored_when_a_chain_settles() {
    let tv = TestValkey::setup().await;
    let metrics: SharedMetrics = Arc::new(crate::common::metrics_scrape_only(SERVICE_LABEL));

    let (stepping, mut steps) = GatedSessionPutStore::stepping();
    let store: Arc<dyn CheckpointStore> = Arc::clone(&stepping) as Arc<dyn CheckpointStore>;
    let session_id = seed_session(&tv, "b443-budget-settle-session").await;

    let (writer, tracker) = spawn_replica(&store, &tv, &metrics, "replica-a");
    writer.enqueue_session_checkpoint_write(session_id.clone());

    let repairs = MAX_PUBLISH_REPAIRS as usize;
    // Chain 1: `repairs` refusals. The pin survives a refused publish, so it
    // is set once and holds for the whole chain.
    for call in 1..=repairs {
        let release = next_put(&mut steps, call).await;
        if call == 1 {
            pin_watermark(&tv, &session_id).await;
        }
        release
            .send(())
            .expect("the parked put must still be waiting");
    }

    // Chain 1's last repair. Lift the pin so its publish is ACCEPTED — that
    // is the `Settled` outcome whose budget reset is under test — and queue
    // chain 2's write while this task still owns the session.
    let release = next_put(&mut steps, repairs + 1).await;
    unpin_watermark(&tv, &session_id).await;
    writer.enqueue_session_checkpoint_write(session_id.clone());
    release
        .send(())
        .expect("the parked put must still be waiting");

    // Chain 2, same task: another full budget's worth of refusals...
    for call in (repairs + 2)..=(2 * repairs + 1) {
        let release = next_put(&mut steps, call).await;
        if call == repairs + 2 {
            pin_watermark(&tv, &session_id).await;
        }
        release
            .send(())
            .expect("the parked put must still be waiting");
    }
    // ...then an accept, so the task terminates instead of chaining forever.
    let release = next_put(&mut steps, 2 * repairs + 2).await;
    unpin_watermark(&tv, &session_id).await;
    release
        .send(())
        .expect("the parked put must still be waiting");

    drain(&tracker, "replica A").await;

    assert_eq!(
        superseded(&metrics, "write"),
        2 * MAX_PUBLISH_REPAIRS,
        "both chains must have repaired their full budget — {MAX_PUBLISH_REPAIRS} ticks each. A \
         value of {MAX_PUBLISH_REPAIRS} means chain 2 never repaired at all: the budget was not \
         restored when chain 1 settled, so the first supersession of chain 2 went straight to \
         `exhausted` and the durable blob stopped converging"
    );
    assert_eq!(
        superseded(&metrics, "exhausted"),
        0,
        "neither chain exceeded its budget, so the alertable `exhausted` cell MUST stay at zero"
    );
    assert_eq!(
        stepping.put_calls(),
        2 * repairs + 2,
        "one PUT per operation: {repairs} refused + 1 accepted, twice over"
    );
}

/// **A chain that EXHAUSTS its budget also restores it for the next chain on
/// the same task.**
///
/// The `exhausted` arm ends the chain exactly as settling does — it abandons
/// the repair rather than re-enqueueing it — so it has to restore the budget
/// too. Leaving it at zero is the more insidious variant of the bug the test
/// above pins: the task that most needs its repairs back (it is on the
/// session that just produced a three-deep supersession chain) is the one
/// left permanently unable to repair, so every later supersession on that
/// session ticks `exhausted`. The alert then fires continuously for a
/// condition that healed the moment the write storm subsided.
///
/// Isolated from the settle-reset above on purpose: no operation is allowed
/// to settle between the exhaustion and the next supersession, so the ONLY
/// thing that can restore the budget here is the `exhausted` arm itself.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn repair_budget_is_restored_when_a_chain_exhausts() {
    let tv = TestValkey::setup().await;
    let metrics: SharedMetrics = Arc::new(crate::common::metrics_scrape_only(SERVICE_LABEL));

    let (stepping, mut steps) = GatedSessionPutStore::stepping();
    let store: Arc<dyn CheckpointStore> = Arc::clone(&stepping) as Arc<dyn CheckpointStore>;
    let session_id = seed_session(&tv, "b443-budget-exhaust-session").await;

    let (writer, tracker) = spawn_replica(&store, &tv, &metrics, "replica-a");
    writer.enqueue_session_checkpoint_write(session_id.clone());

    let repairs = MAX_PUBLISH_REPAIRS as usize;
    // Chain 1: `repairs` refusals, each repaired.
    for call in 1..=repairs {
        let release = next_put(&mut steps, call).await;
        if call == 1 {
            pin_watermark(&tv, &session_id).await;
        }
        release
            .send(())
            .expect("the parked put must still be waiting");
    }

    // The last repair's publish is ALSO refused, with the budget now at zero
    // — the `exhausted` arm. Queue another write while it is parked, so the
    // task has work to pick up after the chain ends; without that the task
    // retires and the next operation would get a fresh task (and a fresh
    // budget) for free, testing nothing.
    let release = next_put(&mut steps, repairs + 1).await;
    writer.enqueue_session_checkpoint_write(session_id.clone());
    release
        .send(())
        .expect("the parked put must still be waiting");

    // The post-exhaustion operation, on the SAME task, still pinned. If the
    // `exhausted` arm restored the budget it repairs (ticking `write`); if it
    // did not, it ticks `exhausted` a second time, abandons, finds nothing
    // pending, and the run stops at PUT #{repairs + 2} — which `next_put`
    // below reports as a missing call rather than a silent count mismatch.
    let release = next_put(&mut steps, repairs + 2).await;
    release
        .send(())
        .expect("the parked put must still be waiting");

    // The repair that proves the budget came back. Let it settle so the task
    // terminates.
    let release = next_put(&mut steps, repairs + 3).await;
    unpin_watermark(&tv, &session_id).await;
    release
        .send(())
        .expect("the parked put must still be waiting");

    drain(&tracker, "replica A").await;

    assert_eq!(
        superseded(&metrics, "exhausted"),
        1,
        "exactly one chain exceeded its budget, so `exhausted` MUST tick exactly once. A second \
         tick means the post-exhaustion operation found the budget still at zero"
    );
    assert_eq!(
        superseded(&metrics, "write"),
        MAX_PUBLISH_REPAIRS + 1,
        "{MAX_PUBLISH_REPAIRS} repairs in the exhausting chain, plus one AFTER the exhaustion — \
         that last tick is the whole assertion: it can only happen if the `exhausted` arm \
         restored the repair budget before continuing"
    );
    assert_eq!(
        stepping.put_calls(),
        repairs + 3,
        "one PUT per operation: {repairs} in the exhausting chain, the exhausting operation \
         itself, then the post-exhaustion write and its repair"
    );
}

// =============================================================================
// Test 6 — the absent-session dispatcher contract.
// =============================================================================

/// **A write enqueued for a session that does not exist reaches no adapter
/// and reports nothing.**
///
/// `build_session_checkpoint` answers `Ok(None)` for an absent session, and
/// the dispatcher must treat that as "nothing to do": no PUT, no DELETE, no
/// publish, no metric tick on ANY cell.
///
/// Each half matters for a different reason. Not escalating to a DELETE is a
/// correctness requirement — an empty `le:{sid}:sess` is also what selective
/// key loss looks like, and tombstoning on it would destroy the very artifact
/// recovery needs. Not ticking any cell is an observability requirement: the
/// absence is not a write attempt, so counting it as `success` would inflate
/// the durable-write rate with operations that wrote nothing, and counting it
/// as `error` would make routine post-delete lifecycle races look like
/// adapter failures.
///
/// The stamp IS minted (the read script mints before it short-circuits), and
/// that is correct: the mint is what orders a concurrent DELETE on another
/// replica. What must NOT appear is a published watermark — nothing landed.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn write_for_an_absent_session_touches_no_adapter_and_ticks_no_cell() {
    let tv = TestValkey::setup().await;
    let metrics: SharedMetrics = Arc::new(crate::common::metrics_scrape_only(SERVICE_LABEL));

    let counting = GatedSessionPutStore::ungated();
    let store: Arc<dyn CheckpointStore> = Arc::clone(&counting) as Arc<dyn CheckpointStore>;

    let (writer, tracker) = spawn_replica(&store, &tv, &metrics, "replica-a");
    // Never created through `SessionStore` — `le:{sid}:sess` does not exist.
    let session_id = "b443-never-created-session".to_string();
    writer.enqueue_session_checkpoint_write(session_id.clone());
    drain(&tracker, "replica A").await;

    assert_eq!(
        counting.put_calls(),
        0,
        "an absent session MUST NOT reach the adapter at all — `build_session_checkpoint` \
         answered Ok(None) and there is nothing to write"
    );
    assert_eq!(
        counting.delete_calls(),
        0,
        "and it MUST NOT be escalated to a DELETE: an empty `le:{{sid}}:sess` is also what \
         selective key loss looks like, so tombstoning on it would destroy the very artifact \
         recovery needs"
    );

    for result in ["success", "error", "skipped"] {
        let cell = session_writes(&metrics, result);
        assert_eq!(
            cell, 0,
            "live_edit_checkpoint_writes_total{{result={result:?}, kind=\"session\"}} MUST stay at \
             zero for an absent session — the absence is not a write attempt, and the metric \
             should read as zero traffic for that session; got {cell}"
        );
    }
    for result in ["write", "delete", "exhausted"] {
        assert_eq!(
            superseded(&metrics, result),
            0,
            "nothing landed at the adapter, so nothing may publish and no supersession verdict \
             can exist"
        );
    }

    assert_eq!(
        ckpt_field(&tv, &session_id, "seq").await.as_deref(),
        Some("1"),
        "the build MUST still mint its stamp — the mint is what orders this build against a \
         concurrent DELETE on another replica, and the read script mints before it \
         short-circuits on the absent session"
    );
    assert_eq!(
        ckpt_field(&tv, &session_id, "pub").await,
        None,
        "nothing landed at the adapter, so the publish watermark MUST NOT advance. A `pub` here \
         would let a genuinely-landed operation on another replica be told it was superseded by \
         an operation that wrote nothing at all"
    );
    assert_eq!(
        ckpt_field(&tv, &session_id, "pub_kind").await,
        None,
        "`pub` and `pub_kind` are written by one HSET; neither may appear"
    );
}

// =============================================================================
// Tests 7-10 — the publish-phase failure contract, via the `PublishFault` seam.
// =============================================================================

/// The publish-phase tests need the process-global `PublishFault` seam, which
/// exists only under `test-checkpoint-fault-injection`.
///
/// **Run these under `cargo nextest`** (canonical per CLAUDE.md
/// `## Build Commands`). The seam is a process-global queue popped by every
/// `publish_stamp` call in the process; `lock_publish_fault_injector_serial`
/// serializes the opt-in consumers against each other, but only nextest's
/// process-per-test isolation stops an unrelated concurrent test in this
/// binary from popping someone else's fault.
#[cfg(feature = "test-checkpoint-fault-injection")]
mod publish_faults {
    // Every test here holds `lock_publish_fault_injector_serial`'s
    // `std::sync::MutexGuard` across awaits, deliberately: the guard's job is
    // to serialize whole test bodies against each other, and each body owns it
    // start to finish with no contended-async-await pattern. Same allow, for
    // the same reason, as the `lock_fault_injector_serial` consumers in
    // `e2e_session_recovery_atomicity_test.rs`.
    #![allow(clippy::await_holding_lock)]

    use super::*;
    use std::collections::VecDeque;

    use live_edit::checkpoint::session_writer::{
        clear_publish_fault_injector_for_test, set_publish_fault_injector_for_test, PublishFault,
    };

    /// Mirror of `MAX_PUBLISH_ATTEMPTS` in
    /// `src/checkpoint/session_writer.rs` — how many times a LANDED
    /// operation's publish is retried before the writer gives up on
    /// establishing its ordering.
    ///
    /// Mirrored rather than imported, for the same reason as
    /// `MAX_PUBLISH_REPAIRS` above. Deliberately a separate constant from
    /// `MAX_ADAPTER_ATTEMPTS` in the production code as well: the two phases
    /// fail for different reasons and have different stakes, so a future
    /// change that collapses them into one number should have to confront
    /// this test.
    const MAX_PUBLISH_ATTEMPTS: usize = 3;

    fn transient(what: &str) -> CheckpointError {
        CheckpointError::Transient {
            source: format!("synthetic publish fault for b.443 test: {what}").into(),
        }
    }

    /// **A transient publish failure retries only the PUBLISH — the landed
    /// operation is never re-run.**
    ///
    /// This is the correctness boundary between the two retry phases. Before
    /// the split, a publish failure retried the WHOLE operation, which is
    /// what made the resurrection bug reachable (see the regression test
    /// below) and also meant one flaky `EVALSHA` cost a redundant multi-MB
    /// adapter PUT.
    ///
    /// Three observations pin it, and the third is the one that cannot be
    /// faked: `seq` stays at 1. A re-run would rebuild, and every build mints
    /// a stamp — so a second mint is the fingerprint of a re-run even if the
    /// adapter-call count were somehow preserved. And `pub` equal to that
    /// same stamp is the "carried forward, not re-minted" half.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_transient_publish_failure_retries_the_publish_not_the_operation() {
        let _serial = super::super::helpers::lock_publish_fault_injector_serial();
        clear_publish_fault_injector_for_test();

        let tv = TestValkey::setup().await;
        let metrics: SharedMetrics = Arc::new(crate::common::metrics_scrape_only(SERVICE_LABEL));

        let gated = GatedSessionPutStore::ungated();
        let store: Arc<dyn CheckpointStore> = Arc::clone(&gated) as Arc<dyn CheckpointStore>;
        let session_id = seed_session(&tv, "b443-publish-transient-session").await;

        set_publish_fault_injector_for_test(VecDeque::from([PublishFault::Fail(transient(
            "first publish attempt",
        ))]));

        let (writer, tracker) = spawn_replica(&store, &tv, &metrics, "replica-a");
        writer.enqueue_session_checkpoint_write(session_id.clone());
        drain(&tracker, "replica A").await;
        clear_publish_fault_injector_for_test();

        assert_eq!(
            gated.put_calls(),
            1,
            "the adapter call MUST NOT be re-run after it landed — only the publish is retried. A \
             count of 2 means the whole operation was retried, which re-opens the resurrection \
             path AND doubles the bytes written for one flaky EVALSHA"
        );
        assert_eq!(
            ckpt_field(&tv, &session_id, "seq").await.as_deref(),
            Some("1"),
            "exactly one build, so exactly one mint. A `seq` of 2 is the fingerprint of a re-run \
             even where the adapter-call count happens to survive"
        );
        assert_eq!(
            ckpt_field(&tv, &session_id, "pub").await.as_deref(),
            Some("1"),
            "the retry MUST carry the SAME stamp the landed bytes were built under, so the \
             watermark ends up at that stamp"
        );
        assert_eq!(
            ckpt_field(&tv, &session_id, "pub_kind").await.as_deref(),
            Some("write")
        );

        assert_eq!(
            session_writes(&metrics, "success"),
            1,
            "the operation landed and its publish resolved, so it ticks `success` exactly once"
        );
        assert_eq!(
            session_writes(&metrics, "error"),
            0,
            "an INTERMEDIATE publish failure MUST NOT tick the counter — the operation was still \
             in progress, and ticking per attempt would report several outcomes for one landed \
             write"
        );
    }

    /// **A landed operation whose publish never resolves ticks `error` once,
    /// never `success`, and leaves the watermark unadvanced.**
    ///
    /// The `error` cell here is a deliberate choice between two half-truths:
    /// the bytes ARE durable, but the operation's ordering was never
    /// established and any clobber it caused will go unrepaired. "Did not
    /// complete" is what this cell has always meant, and reporting `success`
    /// would hide the one outcome where a durable clobber goes unrepaired —
    /// which is the entire failure mode b.443 exists to close.
    ///
    /// The watermark staying absent is the other half: publishing a stamp the
    /// writer could not confirm would tell the NEXT operation it had been
    /// superseded by one whose own ordering is unknown.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn publish_exhaustion_ticks_error_once_and_leaves_the_watermark_unadvanced() {
        let _serial = super::super::helpers::lock_publish_fault_injector_serial();
        clear_publish_fault_injector_for_test();

        let tv = TestValkey::setup().await;
        let metrics: SharedMetrics = Arc::new(crate::common::metrics_scrape_only(SERVICE_LABEL));

        let gated = GatedSessionPutStore::ungated();
        let store: Arc<dyn CheckpointStore> = Arc::clone(&gated) as Arc<dyn CheckpointStore>;
        let session_id = seed_session(&tv, "b443-publish-exhausted-session").await;

        let faults: VecDeque<PublishFault> = (0..MAX_PUBLISH_ATTEMPTS)
            .map(|attempt| PublishFault::Fail(transient(&format!("attempt {}", attempt + 1))))
            .collect();
        set_publish_fault_injector_for_test(faults);

        let (writer, tracker) = spawn_replica(&store, &tv, &metrics, "replica-a");
        writer.enqueue_session_checkpoint_write(session_id.clone());
        drain(&tracker, "replica A").await;
        clear_publish_fault_injector_for_test();

        assert_eq!(
            session_writes(&metrics, "error"),
            1,
            "a landed operation whose publish could not be confirmed ticks `error` EXACTLY once — \
             the unit is the operation, not the publish attempt. A value of \
             {MAX_PUBLISH_ATTEMPTS} means the intermediate attempts are ticking too, which \
             reports several outcomes for one landed write"
        );
        assert_eq!(
            session_writes(&metrics, "success"),
            0,
            "an operation whose ordering was never established MUST NOT report success — that is \
             precisely the state where a durable clobber goes unrepaired"
        );
        assert_eq!(
            gated.put_calls(),
            1,
            "publish-phase retries MUST NOT re-run the adapter call; the bytes landed once"
        );
        assert_eq!(
            ckpt_field(&tv, &session_id, "pub").await,
            None,
            "the publish never succeeded, so the watermark MUST NOT advance. A `pub` here would \
             tell the next operation it was superseded by one whose own ordering is unknown"
        );
        for result in ["write", "delete", "exhausted"] {
            assert_eq!(
                superseded(&metrics, result),
                0,
                "an unconfirmed publish is not a supersession verdict — no repair is scheduled, \
                 deliberately: a blind rebuild could resurrect a session deleted meanwhile, and a \
                 blind delete could destroy a live checkpoint"
            );
        }
    }

    /// **THE resurrection regression (b.443), with the publish-transient
    /// twist that was the actual hole.**
    ///
    /// Interleaving:
    ///
    /// 1. Session exists with layer `L1`. Replica A enqueues a write; its
    ///    build mints `SA` and its PUT parks inside the adapter.
    /// 2. Replica B runs a real `DeleteSession` — tombstone at the adapter,
    ///    and a DELETE published under `SB > SA`.
    /// 3. A is released. Its PUT lands ON TOP of the tombstone: the deleted
    ///    session's checkpoint is now durable again.
    /// 4. **A's first publish attempt fails transiently.** This is the twist.
    /// 5. A's second attempt carries the SAME stamp `SA`, is refused, and is
    ///    told the fresher operation was a `delete` — so A repairs by
    ///    re-issuing the DELETE.
    ///
    /// Before the two-phase ladder, step 4 retried the WHOLE operation:
    /// attempt 2 rebuilt, `build_session_checkpoint` returned `Ok(None)` for
    /// the (now deleted) session, and the dispatcher settled SILENTLY. The
    /// resurrected blob was never detected, never repaired, and
    /// `SessionRecoveryRunner` re-materialised the deleted session on the
    /// next cold lookup — an operator's delete undone with no log and no
    /// metric anywhere.
    ///
    /// The fault is installed only AFTER replica B has drained, so B's own
    /// publish cannot consume it.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn stale_write_resurrecting_a_deleted_session_repairs_through_a_publish_blip() {
        let _serial = super::super::helpers::lock_publish_fault_injector_serial();
        clear_publish_fault_injector_for_test();

        let tv = TestValkey::setup().await;
        let metrics: SharedMetrics = Arc::new(crate::common::metrics_scrape_only(SERVICE_LABEL));

        let (gated, gate) = GatedSessionPutStore::new(1);
        let store: Arc<dyn CheckpointStore> = Arc::clone(&gated) as Arc<dyn CheckpointStore>;
        let session_id = seed_session(&tv, "b443-resurrect-blip-session").await;

        let (writer_a, tracker_a) = spawn_replica(&store, &tv, &metrics, "replica-a");
        let (writer_b, tracker_b) = spawn_replica(&store, &tv, &metrics, "replica-b");

        // Step 1 — A builds `{L1}`, mints SA, parks in the adapter.
        writer_a.enqueue_session_checkpoint_write(session_id.clone());
        tokio::time::timeout(Duration::from_secs(20), gate.reached)
            .await
            .expect("replica A's put_session_checkpoint must be reached inside 20s")
            .expect("the gated put must signal arrival before parking");

        // Step 2 — B deletes for real, through the production hook site.
        let deleting_session_store =
            SessionStore::new(tv.client().clone(), tv.key_schema().clone())
                .with_checkpoint_writer(Arc::clone(&writer_b));
        deleting_session_store
            .delete(&session_id)
            .await
            .expect("SessionStore::delete must succeed for a freshly-created session");
        drain(&tracker_b, "replica B").await;
        assert!(
            read_durable(&store, &session_id).await.is_none(),
            "precondition: B's DELETE must have tombstoned the checkpoint before A resurrects it"
        );

        // Steps 3-5 — one transient publish blip, then the refusal + repair.
        set_publish_fault_injector_for_test(VecDeque::from([PublishFault::Fail(transient(
            "replica A's first publish attempt",
        ))]));
        gate.release
            .send(())
            .expect("the gated put must still be parked and awaiting release");
        drain(&tracker_a, "replica A").await;
        clear_publish_fault_injector_for_test();

        let residual = read_durable(&store, &session_id).await;
        assert!(
            residual.is_none(),
            "b.443: the resurrected blob MUST NOT survive a transient publish failure. Pre-fix, \
             the failure re-ran the whole operation, attempt 2's build saw the deleted session, \
             returned Ok(None) and settled SILENTLY — so the blob written on top of the tombstone \
             was never detected and recovery re-materialised a session the operator deleted. Got \
             {:?}",
            residual.map(|c| c.layer_mappings.keys().cloned().collect::<Vec<_>>())
        );
        assert_eq!(
            gated.put_calls(),
            1,
            "the stale write MUST NOT be re-run by the publish retry — a re-run is exactly the \
             path that hides the resurrection"
        );
        assert_eq!(
            gated.delete_calls(),
            2,
            "two DELETEs: replica B's original and replica A's repair. A count of 1 means the \
             repair never fired and the resurrected blob is still durable"
        );
        assert_eq!(
            superseded(&metrics, "delete"),
            1,
            "the retry carried the SAME stamp, so the second attempt was correctly told the \
             fresher operation was a DELETE. A `write` verdict here would send the repair down \
             the rebuild arm, which exits silently on the absent session and leaves the \
             resurrection in place"
        );
        assert_eq!(
            session_writes(&metrics, "error"),
            0,
            "the publish resolved on attempt 2, so the intermediate failure must not have ticked"
        );
    }

    /// **The publish-exhaustion WARN says the bytes are durable but
    /// unordered.**
    ///
    /// The `error` metric cell folds this outcome together with "nothing was
    /// written at all", so the log is the only place an operator can tell the
    /// two apart — and they call for opposite responses (a lost write
    /// self-heals on the next lifecycle event; an unordered landed write may
    /// have clobbered a fresher payload that nothing will repair). Pinning
    /// the structured fields, not the prose, keeps this from being an
    /// exact-log-text assertion.
    ///
    /// **`flavor = "current_thread"` is load-bearing**:
    /// `tracing::subscriber::set_default` installs the dispatcher in a
    /// thread-local, and `tokio::spawn` captures the current dispatcher at
    /// spawn time. On a multi-thread runtime the dispatch task would run on a
    /// worker thread and its WARN would never reach the capture buffer. Same
    /// pattern as `supervisor_transition_log_names_service_token_in_dependency_set`
    /// in `tests/integration/health_supervisor_service_token_test.rs`.
    #[tokio::test]
    async fn publish_exhaustion_warns_that_the_blob_is_durable_but_unordered() {
        let _serial = super::super::helpers::lock_publish_fault_injector_serial();
        clear_publish_fault_injector_for_test();

        let tv = TestValkey::setup().await;
        let metrics: SharedMetrics = Arc::new(crate::common::metrics_scrape_only(SERVICE_LABEL));

        let store: Arc<dyn CheckpointStore> =
            Arc::clone(&GatedSessionPutStore::ungated()) as Arc<dyn CheckpointStore>;
        let session_id = seed_session(&tv, "b443-publish-warn-session").await;

        let faults: VecDeque<PublishFault> = (0..MAX_PUBLISH_ATTEMPTS)
            .map(|_| PublishFault::Fail(transient("warn-path attempt")))
            .collect();
        set_publish_fault_injector_for_test(faults);

        let (capture, captured) = CapturingWriter::new();
        let subscriber = tracing_subscriber::fmt()
            .with_writer(capture)
            .with_max_level(tracing::Level::WARN)
            .with_ansi(false)
            .finish();
        let dispatch_guard = tracing::subscriber::set_default(subscriber);

        let (writer, tracker) = spawn_replica(&store, &tv, &metrics, "replica-a");
        writer.enqueue_session_checkpoint_write(session_id.clone());
        drain(&tracker, "replica A").await;

        drop(dispatch_guard);
        clear_publish_fault_injector_for_test();

        let buf = captured.lock().expect("capture buffer mutex poisoned");
        let output = String::from_utf8_lossy(&buf);

        assert!(
            output.contains("WARN"),
            "publish exhaustion MUST emit a WARN; captured: {output}"
        );
        for field in ["session_id", "kind=\"write\"", "stamp=1", "attempts=3"] {
            assert!(
                output.contains(field),
                "the publish-exhaustion WARN MUST carry the structured field {field:?} — the \
                 metric cell cannot distinguish `durable but unordered` from `never written`, so \
                 these fields are the operator's only triage signal; captured: {output}"
            );
        }
        assert!(
            output.contains("ordering is unestablished"),
            "the WARN MUST say the bytes are durable but their ordering is unestablished, not \
             merely that a write failed; captured: {output}"
        );
    }

    /// A `MakeWriter` that captures everything written to it. Mirrors the
    /// pattern in `tests/integration/health_supervisor_service_token_test.rs`.
    #[derive(Clone)]
    struct CapturingWriter(Arc<Mutex<Vec<u8>>>);

    impl CapturingWriter {
        fn new() -> (Self, Arc<Mutex<Vec<u8>>>) {
            let buf = Arc::new(Mutex::new(Vec::new()));
            (Self(Arc::clone(&buf)), buf)
        }
    }

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
                .expect("capture buffer mutex poisoned")
                .extend_from_slice(buf);
            Ok(buf.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
}
