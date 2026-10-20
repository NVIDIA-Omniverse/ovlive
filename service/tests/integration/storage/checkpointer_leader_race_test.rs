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

// Task t2.fgj.ya.9j Subtask .nt — leader-elected loop two-replica race.
//
// Verifies SDD SR-2.1: when two replicas spawn `spawn_checkpoint_loop`
// against the same Valkey + KeySchema, only ONE wins the
// `le:checkpoint:leader` lock (30 s TTL) per window, and only that replica
// runs the per-layer assembler + adapter writes. Pre-contract, every
// replica's tick would call `put_layer_checkpoint` independently, doubling
// the durable-write fan-out on every tick.
//
// Test strategy:
//
// 1. `TestValkey::setup_with_pool_size(8)` — generous pool so the two
//    concurrent replicas have room for their maintenance probes + the
//    per-tick SCAN traffic without contention masking the race.
// 2. Seed two real layers via `LayerStore::create_layer` so the
//    assembler's `load_consistent_snapshot` call has a complete obj:hdr +
//    sections + fields shape to walk; `sequence_num = 1` keeps the
//    skip-if-no-commits gate from firing.
// 3. Run the loop with `tick = 100 ms`, `interval = 0`, `quiesce = 0`
//    so every tick reaches the assembler — the test wants to force the
//    leader-only branch to do real work, not to verify the cadence gates
//    (which `checkpointer_skip_gates_test.rs` owns).
// 4. Wrap a single shared `InMemoryCheckpointStore` in a counting adapter
//    so we can capture WHICH replica issued each put. The CheckpointMeta
//    carries `creator_replica_id`, threaded through from the loop's
//    `config.leader.service_label` via the assembler's `service_label`
//    argument —
//    that's the load-bearing signal that pins "exactly one replica did
//    the per-layer work".
// 5. Bounded-poll `GET le:checkpoint:leader` until it holds one of the two
//    replicas' lock values (`LeaderIdentity::lock_value()`,
//    `label:instance`); bounded-poll the counting store until at least one
//    layer has been checkpointed; cancel + join both spawn handles. The
//    lock value and `creator_replica_id` are deliberately different strings:
//    the writer is matched to the lock holder by the holder's
//    `service_label`, which is what `creator_replica_id` carries.
//
// All polling is bounded with `tokio::time::timeout` against an explicit
// deadline — no naked `sleep` for coordination, per CLAUDE.md.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use bytes::Bytes;
use fred::interfaces::KeysInterface;
use tokio::sync::Mutex;
use tokio::time::Instant;
use tokio_util::sync::CancellationToken;

use live_edit::checkpoint::adapters::InMemoryCheckpointStore;
use live_edit::checkpoint::{
    spawn_checkpoint_loop, CheckpointBlob, CheckpointError, CheckpointHandle, CheckpointLoopConfig,
    CheckpointMeta, CheckpointStore, TailCaptureConfig,
};
use live_edit::domain::types::ObjectState;
use live_edit::storage::external_values::MockExternalValueStore;
use live_edit::storage::layer::LayerStore;
use live_edit::storage::session::SessionStore;
use live_edit::storage::test_support::TestValkey;

const REPLICA_A: &str = "replica-a";
const REPLICA_B: &str = "replica-b";

/// Counting wrapper composed atop `crate::checkpoint::helpers::DelegatingCheckpointStore`
/// (b.dzd Phase 2). Overrides only `put_layer_checkpoint` to record the
/// `creator_replica_id` of each call; every other trait method delegates
/// verbatim through the embedded `DelegatingCheckpointStore`. The recorded
/// vector is the test's "which replica did the per-tick work" signal.
struct CountingCheckpointStore {
    delegate: crate::checkpoint::helpers::DelegatingCheckpointStore<InMemoryCheckpointStore>,
    put_replica_ids: Mutex<Vec<String>>,
}

impl CountingCheckpointStore {
    fn new() -> Self {
        Self {
            delegate: crate::checkpoint::helpers::DelegatingCheckpointStore::new(Arc::new(
                InMemoryCheckpointStore::new(),
            )),
            put_replica_ids: Mutex::new(Vec::new()),
        }
    }

    /// Snapshot the put-replica-id vector for assertion. Returns the
    /// recorded ids in put-order so the test can also reason about ordering
    /// if it wants to (currently only checks the set of unique ids).
    async fn put_replica_ids_snapshot(&self) -> Vec<String> {
        self.put_replica_ids.lock().await.clone()
    }
}

#[async_trait]
impl CheckpointStore for CountingCheckpointStore {
    async fn put_layer_checkpoint(
        &self,
        object_id: u64,
        snap_seq: u64,
        ompf: Bytes,
        meta: CheckpointMeta,
    ) -> Result<CheckpointHandle, CheckpointError> {
        // Record BEFORE delegating: even on a future adapter error the test
        // wants to know which replica initiated the put. The
        // `InMemoryCheckpointStore` cannot fail in practice (it's
        // mutex-guarded in-memory state), so put-order matches record-order.
        self.put_replica_ids
            .lock()
            .await
            .push(meta.creator_replica_id.clone());
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
}

/// Seed a minimal-but-valid layer pair the loop's assembler can walk: one
/// session plus two real layers with `sequence_num = 1` so the
/// skip-if-no-commits gate (`seq == last_checkpoint_seq == 0`) does NOT
/// fire. Returns the two object ids.
async fn seed_two_layers(tv: &TestValkey) -> [u64; 2] {
    let session_store = SessionStore::new(tv.client().clone(), tv.key_schema().clone());
    let session_id = session_store
        .create("leader-race-session", "/stage", "leader-race-user")
        .await
        .expect("session creation must succeed");

    let layer_store = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096);
    let mock_ext = Arc::new(MockExternalValueStore::new());

    let state = ObjectState {
        sequence_num: 1,
        first_version: 1,
        source_format: "usda".to_string(),
        ..Default::default()
    };

    let oid_a = layer_store
        .create_layer(
            &session_id,
            "layer-a",
            "leader-race-user",
            &state,
            mock_ext.clone(),
        )
        .await
        .expect("create_layer A must succeed");
    let oid_b = layer_store
        .create_layer(&session_id, "layer-b", "leader-race-user", &state, mock_ext)
        .await
        .expect("create_layer B must succeed");

    [oid_a, oid_b]
}

/// Bounded-poll a future-returning closure until it yields `true` or the
/// deadline expires. Mirrors the `wait_for_*` helpers in
/// `presence_recovery_test.rs` — no naked `sleep` loops, every retry sits
/// behind a deadline so a hung condition surfaces as a deterministic
/// timeout rather than a 60 s test stall.
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
        // 25 ms between probes keeps the poll cost modest against the
        // 100 ms loop tick — we sample ~4× per tick, fast enough to catch
        // a single-tick lock acquisition.
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
    pred().await
}

/// SR-2.1: two replicas race for `le:checkpoint:leader`; only one wins per
/// TTL window and only that replica's tick body runs the per-layer
/// assembler + adapter put. The counting store records the
/// `creator_replica_id` of every put it sees — the test asserts that the
/// recorded set is a SINGLETON containing one of the two replica ids
/// (never both, never empty, never some third value).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn two_replicas_only_leader_writes_per_tick() {
    let tv = TestValkey::setup_with_pool_size(8).await;
    let keys = tv.key_schema().clone();

    // Pre-populate two real layers so the loop has work to do every tick.
    let oids = seed_two_layers(&tv).await;

    // One shared counting store between both spawn-loops — modelling the
    // production wiring where both replicas write into the same durable
    // adapter (same LocalDisk volume or same Azure container).
    let counting_store: Arc<CountingCheckpointStore> = Arc::new(CountingCheckpointStore::new());
    let store_a: Arc<dyn CheckpointStore> = counting_store.clone();
    let store_b: Arc<dyn CheckpointStore> = counting_store.clone();

    // `tick = 100 ms` so we get ~5 ticks in our 500 ms observation window;
    // `interval = 0`, `quiesce = 0` so every tick reaches the assembler —
    // this isolates the leader-only invariant from the cadence gates.
    let leader_a = crate::common::test_leader(REPLICA_A);
    let leader_b = crate::common::test_leader(REPLICA_B);
    let loop_config_a = CheckpointLoopConfig {
        tick: Duration::from_millis(100),
        interval: Duration::ZERO,
        quiesce: Duration::ZERO,
        retain_generations: 2,
        // Epic 6 counters-sync cadence; pushed well above the test runtime
        // so the counters-sidecar timer does not fire during this test.
        counters_sync: Duration::from_secs(3600),
        reconcile_interval: Duration::from_secs(3600),
        leader: leader_a.clone(),
        // Tails pinned explicitly OFF. b.3n1 promoted the binary default
        // to `true`; this suite predates the tail feature and was written
        // against an inert tail path, so the flag is part of the fixture
        // rather than something to inherit.
        tail: TailCaptureConfig {
            enabled: false,
            ..TailCaptureConfig::default()
        },
    };
    let loop_config_b = CheckpointLoopConfig {
        leader: leader_b.clone(),
        ..loop_config_a.clone()
    };

    let cancel = CancellationToken::new();
    let handle_a = spawn_checkpoint_loop(
        tv.client().clone(),
        keys.clone(),
        store_a,
        loop_config_a,
        None,
        cancel.clone(),
    );
    let handle_b = spawn_checkpoint_loop(
        tv.client().clone(),
        keys.clone(),
        store_b,
        loop_config_b,
        None,
        cancel.clone(),
    );

    // ── Wait for the leader lock to be acquired ─────────────────────────
    // The lock value is the winning replica's `LeaderIdentity::lock_value()`
    // (the shared LEADER_ACQUIRE_OR_REFRESH_SCRIPT used by GC + presence
    // sweepers stores whatever lock value the acquiring process passes).
    let lock_a = leader_a.lock_value();
    let lock_b = leader_b.lock_value();
    let lock_key = keys.checkpoint_leader_lock();
    let pool = tv.client().pool.clone();
    let lock_key_for_poll = lock_key.clone();
    let lock_acquired = poll_until(Duration::from_secs(5), || {
        let pool = pool.clone();
        let key = lock_key_for_poll.clone();
        let (lock_a, lock_b) = (lock_a.clone(), lock_b.clone());
        async move {
            let value: Option<String> = pool.get(&key).await.unwrap_or(None);
            value.is_some_and(|v| v == lock_a || v == lock_b)
        }
    })
    .await;
    assert!(
        lock_acquired,
        "le:checkpoint:leader must be acquired by exactly one of the two replicas \
         within 5 s; the leader-lock probe is the gating step before any per-layer \
         work happens",
    );

    // Capture WHO won — every subsequent put MUST carry the winner's
    // `service_label` as its `creator_replica_id`.
    let leader_lock_value: String = pool
        .get::<Option<String>, _>(&lock_key)
        .await
        .expect("GET checkpoint leader lock must succeed")
        .expect("leader lock value must be present after the bounded-poll wait");
    let leader_label: &str = if leader_lock_value == lock_a {
        REPLICA_A
    } else if leader_lock_value == lock_b {
        REPLICA_B
    } else {
        panic!(
            "leader lock value must be exactly one of the two replicas' lock values \
             ({lock_a:?} / {lock_b:?}), got {leader_lock_value:?}"
        )
    };

    // ── Wait for the leader to actually write ───────────────────────────
    // The assertion target is "exactly one replica did the per-layer
    // work", which we observe through the counting store. We need at
    // least one recorded put before we can reason about set membership;
    // a 5 s budget gives the loop's 100 ms tick ~50 attempts.
    let store_for_poll = Arc::clone(&counting_store);
    let saw_put = poll_until(Duration::from_secs(5), || {
        let store = Arc::clone(&store_for_poll);
        async move { !store.put_replica_ids_snapshot().await.is_empty() }
    })
    .await;
    assert!(
        saw_put,
        "leader must issue at least one put_layer_checkpoint within 5 s \
         (interval = quiesce = 0 force every tick to reach the assembler)",
    );

    // ── Soak: let the loop run for ~500 ms across many ticks ─────────────
    // If the leader-only invariant ever flipped (lock not honoured, lock
    // races, etc.), the soak window would surface puts from BOTH replica
    // ids. The lock TTL is 30 s so a winning replica keeps winning for
    // the whole soak — exactly one id is the only legal outcome.
    tokio::time::sleep(Duration::from_millis(500)).await;

    // ── Assert: only one replica wrote ──────────────────────────────────
    let recorded = counting_store.put_replica_ids_snapshot().await;
    assert!(
        !recorded.is_empty(),
        "post-soak the counting store must hold at least one put",
    );
    let unique_replicas: std::collections::HashSet<&str> =
        recorded.iter().map(String::as_str).collect();
    assert_eq!(
        unique_replicas.len(),
        1,
        "exactly one replica must have issued put_layer_checkpoint calls; \
         pre-SR-2.1 every replica's tick would call the assembler and the \
         set would have 2 members. Recorded ids: {recorded:?}",
    );
    let recorded_writer = unique_replicas
        .iter()
        .next()
        .copied()
        .expect("unique set is non-empty per the assertion above");
    // `creator_replica_id` is the writer's `service_label`, so it is
    // compared with the lock holder's label, not with the lock value.
    assert_eq!(
        recorded_writer, leader_label,
        "the only replica that wrote must equal the one holding the \
         leader lock; mismatch implies a non-leader bypassed the lock probe. \
         leader={leader_lock_value} (label {leader_label}), writer={recorded_writer}",
    );

    // Sanity: every recorded id must be one of the two configured replicas.
    // Guards against a future refactor that synthesises some other id into
    // `CheckpointMeta::creator_replica_id`.
    for id in &recorded {
        assert!(
            id == REPLICA_A || id == REPLICA_B,
            "recorded creator_replica_id must be one of the two configured \
             replicas, got {id:?} (full list: {recorded:?})",
        );
    }

    // Sanity: the leader actually checkpointed BOTH seeded layers across the
    // soak window. Without this, a regression that silently dropped per-layer
    // iteration (e.g. SCAN broken on cluster) would still pass the
    // single-replica check above. Each layer gets at least one put per tick
    // in our config; we just need >=1 across both.
    //
    // Group recorded puts by (object_id, snap_seq) via the inner adapter
    // state — the counting wrapper itself records only the replica id.
    let mut object_ids_seen: HashMap<u64, usize> = HashMap::new();
    for oid in oids {
        let handles = counting_store
            .list_layer_checkpoints(oid)
            .await
            .expect("list_layer_checkpoints must succeed against the in-memory adapter");
        object_ids_seen.insert(oid, handles.len());
    }
    for (oid, count) in &object_ids_seen {
        assert!(
            *count >= 1,
            "leader must have checkpointed object_id={oid} at least once \
             during the soak (full counts: {object_ids_seen:?})",
        );
    }

    // ── Teardown: cancel + bounded-join both handles ────────────────────
    cancel.cancel();
    for h in [handle_a, handle_b] {
        let join_outcome = tokio::time::timeout(Duration::from_secs(5), h).await;
        assert!(
            join_outcome.is_ok(),
            "checkpoint loop join must complete within 5 s of cancellation; \
             a stuck shutdown would have wedged main.rs's drain budget too",
        );
        join_outcome
            .expect("timeout already checked")
            .expect("checkpoint loop task must not panic on shutdown");
    }
}
