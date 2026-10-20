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

// Fault-injection integration test for the retention prune's relationship to
// `CheckpointStore::list_referenced_ext_hashes`.
//
// ## Re-targeted at b.9qd
//
// Originally (Task t2.fgj.zc.56 Subtask `.1n`, AC #5) this file asserted that
// the **inline cascade** was recoverable from a transient
// `list_referenced_ext_hashes` failure: when the union read failed on one
// tick, the cascade was skipped, the hashes of that tick's pruned generation
// were permanently leaked ("under-counting is safe"), and a later prune cycle
// caught up on the NEXT generation's hashes.
//
// b.9qd deleted that premise root and branch. The retention prune no longer
// reads the live set at all — it records the pruned generations' hashes on
// `le:ckpt:{recon}:tomb` and deletes the generations, and the live-set read
// moved to the reconciliation pass's collect phase, where a failure defers
// rather than leaks (the tombstones are durable). So "the cascade is skipped
// on a live-set failure" is no longer a statement about this code path.
//
// What replaces it here is the stronger claim the change makes possible, and
// the one an integration test is uniquely placed to make: **the retention
// prune makes full progress while the live-set read is failing, because it
// never calls it.** The failure arms of the collect phase — abort-only,
// tombstones survive, next pass recovers — are pinned in-crate in
// `checkpointer::tests::collect_live_set_read_failure_aborts_only_the_collect`
// and `checkpointer::tests::collect_retries_a_failed_delete_ext_blob_on_the_next_pass`,
// where `run_reconcile_tick` can be driven directly at an explicit grace.
//
// The file stays feature-gated because it still uses the
// `CheckpointFaultInjector` seam, which only exists under
// `test-checkpoint-fault-injection`.
//
// Mental model:
//
//   - `CountingAlwaysFailListRefsInjector` counts every
//     `list_referenced_ext_hashes` call and fails ALL of them. A count of
//     zero after a full run of prune cycles is the assertion: the prune is
//     structurally decoupled from the live set.
//
//   - `RecordingCheckpointStore` wraps the injector-equipped
//     `InMemoryCheckpointStore` and records every `delete_ext_blob(hash)`.
//     Post-b.9qd an empty log is the expected outcome, not a symptom.

#![cfg(feature = "test-checkpoint-fault-injection")]

use std::collections::HashSet;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use bytes::Bytes;
use fred::interfaces::SortedSetsInterface;
use tokio::time::{sleep, timeout};
use tokio_util::sync::CancellationToken;

use live_edit::checkpoint::adapters::{CheckpointFaultInjector, InMemoryCheckpointStore};
use live_edit::checkpoint::{
    spawn_checkpoint_loop, CheckpointBlob, CheckpointError, CheckpointHandle, CheckpointLoopConfig,
    CheckpointMeta, CheckpointStore, LayerArtifact, TailCaptureConfig,
};
use live_edit::domain::types::ObjectState;
use live_edit::storage::external_values::MockExternalValueStore;
use live_edit::storage::layer::LayerStore;
use live_edit::storage::session::SessionStore;
use live_edit::storage::test_support::relay::RelayedUpdates;
use live_edit::storage::test_support::TestValkey;
use oodf_ffi::{OwnedDelta, SetField};

// ────────────────────────────────────────────────────────────────────────
// `CountingAlwaysFailListRefsInjector` — every `list_referenced_ext_hashes`
// call is counted and answered with `Transient`; every other op passes
// through untouched.
//
// The counter is the load-bearing half. "The prune still worked" is a weak
// claim on its own — it would also hold if the prune called the live set and
// merely tolerated the error. `call_count() == 0` is what proves the call is
// gone from the path entirely, which is the b.9qd change and the reason the
// old permanent-leak failure mode cannot recur.
// ────────────────────────────────────────────────────────────────────────

struct CountingAlwaysFailListRefsInjector {
    call_count: AtomicUsize,
}

impl CountingAlwaysFailListRefsInjector {
    fn new() -> Self {
        Self {
            call_count: AtomicUsize::new(0),
        }
    }

    fn call_count(&self) -> usize {
        self.call_count.load(Ordering::SeqCst)
    }
}

impl CheckpointFaultInjector for CountingAlwaysFailListRefsInjector {
    fn maybe_inject(&self, op: &str) -> Result<(), CheckpointError> {
        if op != "list_referenced_ext_hashes" {
            return Ok(());
        }
        self.call_count.fetch_add(1, Ordering::SeqCst);
        Err(CheckpointError::Transient {
            source: Box::new(std::io::Error::other(
                "synthetic list_referenced_ext_hashes failure",
            )),
        })
    }
}

// ────────────────────────────────────────────────────────────────────────
// `RecordingCheckpointStore` — observation wrapper over an
// `InMemoryCheckpointStore` that records every `delete_ext_blob(hash)` call.
// Same shape as the sibling bundled cascade-test file's wrapper; the only
// difference is that the already-fault-injector-equipped adapter is passed in.
//
// Every non-recording trait method forwards to the inner store — including the
// b.9qd reconciliation trio (`list_layer_artifacts`,
// `delete_layer_temp_artifacts`, `publish_worst_case`) — so the wrapper is
// purely additive observation and never a behaviour change. An un-forwarded
// method silently substitutes the trait default, which is the quietest possible
// way to make a test pass vacuously.
// ────────────────────────────────────────────────────────────────────────

struct RecordingCheckpointStore {
    inner: InMemoryCheckpointStore,
    delete_log: Arc<Mutex<Vec<String>>>,
}

impl RecordingCheckpointStore {
    fn new_with_injector(injector: Arc<dyn CheckpointFaultInjector>) -> Self {
        Self {
            inner: InMemoryCheckpointStore::new().with_fault_injector(injector),
            delete_log: Arc::new(Mutex::new(Vec::new())),
        }
    }

    fn delete_log(&self) -> Arc<Mutex<Vec<String>>> {
        Arc::clone(&self.delete_log)
    }
}

#[async_trait]
impl CheckpointStore for RecordingCheckpointStore {
    async fn put_layer_checkpoint(
        &self,
        object_id: u64,
        snap_seq: u64,
        ompf: Bytes,
        meta: CheckpointMeta,
    ) -> Result<CheckpointHandle, CheckpointError> {
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

    // Forwarded per `CheckpointStore::get_layer_checkpoint`'s MUST — its trait
    // default returns `NotImplemented`, which would mask a generation the
    // inner store holds rather than merely leaving the call un-recorded.
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
        self.delete_log
            .lock()
            .expect("delete_log mutex poisoned")
            .push(hash_hex.to_string());
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

    async fn read_ext_refs_sidecar(
        &self,
        handle: &CheckpointHandle,
    ) -> Result<HashSet<String>, CheckpointError> {
        self.inner.read_ext_refs_sidecar(handle).await
    }

    // ── b.9qd reconciliation surface ────────────────────────────────────
    //
    // These three MUST forward even though nothing in this file records
    // them. Omitting them silently inherits the trait defaults —
    // `Ok(Vec::new())`, `Ok(0)`, and the local publish bound — which does
    // not fail any test, it makes the reconciliation arm of the loop under
    // test *inert*: the pass enumerates nothing, so it registers nothing,
    // reaps nothing, and every assertion about it passes vacuously against a
    // wrapper that hid the adapter's real behaviour.
    async fn list_layer_artifacts(&self) -> Result<Vec<LayerArtifact>, CheckpointError> {
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

// ────────────────────────────────────────────────────────────────────────
// Fixture helpers (same shape as the sibling bundled cascade-test file).
// ────────────────────────────────────────────────────────────────────────

fn hash_bytes(byte: u8) -> Vec<u8> {
    vec![byte; 32]
}

fn hash_hex(byte: u8) -> String {
    let mut out = String::with_capacity(64);
    for _ in 0..32 {
        out.push_str(&format!("{byte:02x}"));
    }
    out
}

fn mock_ext_store() -> Arc<MockExternalValueStore> {
    Arc::new(MockExternalValueStore::new())
}

fn layer_store(tv: &TestValkey) -> LayerStore {
    LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096)
}

fn session_store(tv: &TestValkey) -> SessionStore {
    SessionStore::new(tv.client().clone(), tv.key_schema().clone())
}

async fn create_test_session(tv: &TestValkey, name: &str) -> String {
    session_store(tv)
        .create(name, "/stage", "cascade-fi-test-user")
        .await
        .expect("session creation must succeed")
}

async fn tombstones(tv: &TestValkey) -> HashSet<String> {
    let members: Vec<String> = tv
        .client()
        .pool
        .zrange(
            tv.key_schema().checkpoint_reconcile_tombstones(),
            0,
            -1,
            None,
            false,
            None,
            false,
        )
        .await
        .expect("ZRANGE over the tombstone ZSET must succeed");
    members.into_iter().collect()
}

fn ext_set_field_delta(section_id: u64, key: &str, hash_byte: u8, set_order: u64) -> OwnedDelta {
    OwnedDelta {
        is_diff: true,
        base_version: 1,
        set_fields: vec![SetField {
            section_id,
            key_name: key.to_string(),
            value_or_ext_hash: hash_bytes(hash_byte),
            ext_value_size: 1,
            ext_value_index: 0,
            set_order,
        }],
        ..Default::default()
    }
}

async fn wait_for_at_least(
    store: &Arc<dyn CheckpointStore>,
    object_id: u64,
    at_least: usize,
) -> usize {
    let deadline = Duration::from_secs(10);
    timeout(deadline, async {
        loop {
            let len = store
                .list_layer_checkpoints(object_id)
                .await
                .expect("list_layer_checkpoints must succeed")
                .len();
            if len >= at_least {
                return len;
            }
            sleep(Duration::from_millis(25)).await;
        }
    })
    .await
    .unwrap_or_else(|_| {
        panic!(
            "timed out waiting for list_layer_checkpoints(oid={object_id}).len() >= {at_least} \
             within {deadline:?}"
        )
    })
}

async fn wait_for_head_snap_seq_above(
    store: &Arc<dyn CheckpointStore>,
    object_id: u64,
    prev_snap_seq: u64,
) -> u64 {
    let deadline = Duration::from_secs(10);
    timeout(deadline, async {
        loop {
            let handles = store
                .list_layer_checkpoints(object_id)
                .await
                .expect("list must succeed");
            if let Some(head) = handles.first() {
                if head.snap_seq > prev_snap_seq {
                    return head.snap_seq;
                }
            }
            sleep(Duration::from_millis(25)).await;
        }
    })
    .await
    .unwrap_or_else(|_| {
        panic!(
            "timed out waiting for head snap_seq > {prev_snap_seq} on object {object_id} \
             within {deadline:?}"
        )
    })
}

async fn wait_for_tombstone(tv: &TestValkey, hash: &str) -> bool {
    timeout(Duration::from_secs(10), async {
        loop {
            if tombstones(tv).await.contains(hash) {
                return true;
            }
            sleep(Duration::from_millis(25)).await;
        }
    })
    .await
    .unwrap_or(false)
}

fn loop_config(replica_id: &str, retain_generations: usize) -> CheckpointLoopConfig {
    CheckpointLoopConfig {
        tick: Duration::from_millis(50),
        interval: Duration::ZERO,
        quiesce: Duration::ZERO,
        retain_generations,
        counters_sync: Duration::from_secs(3600),
        // `tokio::time::interval` fires immediately, so ONE reconciliation
        // pass runs at t=0 against an empty store. Its collect phase returns
        // before the live-set read (nothing is tombstoned yet), which is why
        // the injector's call count can be asserted at exactly zero below.
        reconcile_interval: Duration::from_secs(3600),
        leader: crate::common::test_leader(replica_id),
        // Tails pinned explicitly OFF. b.3n1 promoted the binary default
        // to `true`; this suite predates the tail feature and was written
        // against an inert tail path, so the flag is part of the fixture
        // rather than something to inherit.
        tail: TailCaptureConfig {
            enabled: false,
            ..TailCaptureConfig::default()
        },
    }
}

async fn cancel_and_join(cancel: CancellationToken, handle: tokio::task::JoinHandle<()>) {
    cancel.cancel();
    timeout(Duration::from_secs(5), handle)
        .await
        .expect("checkpoint loop must exit within 5 s of cancellation")
        .expect("checkpoint loop task must not panic on shutdown");
}

fn snapshot_delete_log(log: &Arc<Mutex<Vec<String>>>) -> Vec<String> {
    log.lock().expect("delete_log mutex poisoned").clone()
}

// ────────────────────────────────────────────────────────────────────────
// The retention prune completes — deletes generations, records tombstones —
// while EVERY `list_referenced_ext_hashes` call is failing, because it makes
// none.
//
// Layout (retain=2, fast tick, gates disabled so every commit is due):
//
//   - CreateLayer     → initial generation, empty sidecar
//   - Commit 1: a=H1  → gen 1 sidecar {H1}; count = 2, no prune
//   - Commit 2: a=H2  → gen 2 sidecar {H2}; count = 3 → prune the initial
//   - Commit 3: a=H3  → gen 3 sidecar {H3}; count = 3 → prune gen 1 ({H1})
//
// Pre-b.9qd, commit 3's tick would call `list_referenced_ext_hashes`, take the
// injected `Transient`, skip the cascade, and leak H1 **permanently** — the
// sidecar naming it had already been deleted in the same function, so no later
// tick could recover it. That permanent leak is the failure this file's
// original test encoded as expected behaviour.
//
// Post-b.9qd the prune records H1 on `le:ckpt:{recon}:tomb` before deleting
// gen 1, so the deferred work is durable and the reconciliation pass collects
// it later. The three assertions below are exactly that story: the injector
// never fires, the tombstone is recorded, and nothing is deleted inline.
// ────────────────────────────────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn retention_prune_records_tombstones_without_ever_reading_the_live_set() {
    let tv = TestValkey::setup_with_pool_size(8).await;
    let ls = layer_store(&tv);
    let keys = tv.key_schema().clone();
    let client = tv.client().clone();
    let session_id = create_test_session(&tv, "cascade-fi-session").await;

    let initial_state = ObjectState {
        sequence_num: 1,
        first_version: 1,
        source_format: "usd".to_string(),
        ..Default::default()
    };
    let oid = ls
        .create_layer(
            &session_id,
            "cascade-layer-fi",
            "cascade-fi-test-user",
            &initial_state,
            mock_ext_store(),
        )
        .await
        .expect("create_layer must succeed");

    // Construct the injector concretely so the test can read its call count;
    // the trait-object handle is what the adapter takes. Both `Arc`s point at
    // the same underlying struct.
    let injector_concrete = Arc::new(CountingAlwaysFailListRefsInjector::new());
    let injector_for_adapter: Arc<dyn CheckpointFaultInjector> =
        Arc::clone(&injector_concrete) as Arc<dyn CheckpointFaultInjector>;
    let recording = Arc::new(RecordingCheckpointStore::new_with_injector(
        injector_for_adapter,
    ));
    let delete_log = recording.delete_log();
    let store: Arc<dyn CheckpointStore> = recording;
    let cancel = CancellationToken::new();
    let handle = spawn_checkpoint_loop(
        client,
        keys,
        Arc::clone(&store),
        loop_config("cascade-fi-replica", 2),
        None,
        cancel.clone(),
    );

    wait_for_at_least(&store, oid, 1).await;
    let mut last_head = store
        .list_layer_checkpoints(oid)
        .await
        .expect("list must succeed")
        .first()
        .expect("head present")
        .snap_seq;

    for (i, hash_byte) in [0x01_u8, 0x02_u8, 0x03_u8].iter().enumerate() {
        ls.update_layer_relayed(
            &session_id,
            "cascade-layer-fi",
            oid,
            Some(ext_set_field_delta(1, "a", *hash_byte, 10 + i as u64 * 10)),
            "cascade-fi-client",
            5_000 + i as u64,
        )
        .await
        .expect("update_layer must succeed");
        last_head = wait_for_head_snap_seq_above(&store, oid, last_head).await;
    }
    let _ = last_head;

    // Commit 3's tick prunes gen 1 (sidecar {H1}) — the prune cycle that,
    // pre-b.9qd, was the one the injected failure destroyed.
    let h1 = hash_hex(0x01);
    assert!(
        wait_for_tombstone(&tv, &h1).await,
        "the prune must record H1 even though every live-set read is failing; tombstones = {:?}",
        tombstones(&tv).await
    );

    let prune_deadline = Duration::from_secs(10);
    let final_len = timeout(prune_deadline, async {
        loop {
            let len = store
                .list_layer_checkpoints(oid)
                .await
                .expect("list must succeed")
                .len();
            if len <= 2 {
                return len;
            }
            sleep(Duration::from_millis(25)).await;
        }
    })
    .await
    .unwrap_or_else(|_| panic!("retain=2 cap never re-enforced within {prune_deadline:?}"));

    // Tear the loop down FIRST so no late tick can slip work between the
    // reads below. The 5 s join budget is load-bearing in its own right: a
    // regression that propagated the injected error up instead of never
    // making the call would surface as a hung loop here.
    cancel_and_join(cancel, handle).await;

    assert_eq!(
        final_len, 2,
        "the retention prune must complete and enforce retain=2 with the live-set read broken"
    );
    assert_eq!(
        injector_concrete.call_count(),
        0,
        "THE assertion: the retention prune must never call `list_referenced_ext_hashes`. \
         Pre-b.9qd it did, and a failure there meant the tick's pruned hashes were leaked \
         PERMANENTLY — the sidecar naming them was deleted in the same function, so no later \
         tick could recover them. The call moved to the reconciliation pass's collect phase, \
         where a failure defers instead, because the tombstones outlive it"
    );
    assert!(
        tombstones(&tv).await.contains(&h1),
        "and the deferred work is durable: H1 sits on le:ckpt:{{recon}}:tomb waiting for a \
         collect, rather than being lost with its sidecar"
    );
    let deleted = snapshot_delete_log(&delete_log);
    assert!(
        deleted.is_empty(),
        "no ext blob may be deleted inline by the prune, with or without a working live set. \
         Got {deleted:?}"
    );
}
