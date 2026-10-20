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

// Integration coverage for the retention prune's ext-blob bookkeeping under
// the *real* leader-elected checkpoint loop.
//
// ## Re-targeted at b.9qd
//
// Originally (Task t2.fgj.zc.56 Subtask `.eg`) these tests asserted the
// **inline cascade**: the loop's retention prune computed `pruned − live` and
// issued `delete_ext_blob` for the difference, and the four ACs were about
// which hashes appeared in a recorder wrapped around that call.
//
// b.9qd replaced the inline cascade with a claim-then-collect protocol. The
// prune now records the pruned generations' hashes on `le:ckpt:{recon}:tomb`
// **before** deleting the generations and deletes no blob at all; the delete
// happens in the reconciliation pass a grace window later, after ONE live-set
// read for the whole pass. So the loop-level observable moved: it is the
// tombstone ZSET, plus the *absence* of any `delete_ext_blob` call.
//
// The four tests below were re-aimed at that observable rather than deleted,
// because what they uniquely cover is still uncovered elsewhere: the wiring
// from a REAL commit, through the assembler's own emitted ext-refs sidecar,
// through the spawned loop's retention prune, into the ZSET. The unit tests in
// `src/checkpoint/checkpointer.rs` drive `prune_old_generations` and
// `run_reconcile_tick` directly against hand-written sidecars; they cannot
// prove the loop feeds them the right hashes, and this file does not
// re-litigate the protocol they already pin.
//
// ## Why the collect phase is NOT driven here
//
// `run_reconcile_tick` is private, and the grace window it acts on is derived
// from the adapter's `publish_worst_case` with a 60 s floor — so an
// integration binary can neither call it nor wait it out per assertion. The
// collect-phase contracts (delete vs spare, live-set failure, retry) live
// in-crate in `checkpointer::tests` and `checkpointer::reconcile_tests`, where
// the grace is an explicit parameter. See the module docs there.
//
// Mental model (unchanged):
//
//   - Drive ext-blob references via the real `LayerStore::update_layer` path
//     with `SetField { ext_value_size: 1, value_or_ext_hash: <32 bytes>, .. }`
//     so the assembler walks the snapshot, calls `collect_external_value_hashes`,
//     and emits a real ext-refs sidecar attached to `CheckpointMeta::sidecar_bytes`.
//     No hand-crafted sidecar bytes — that would only re-test what the unit
//     tests already pin.
//
//   - `RecordingCheckpointStore` wraps an inner `InMemoryCheckpointStore` and
//     records every `delete_ext_blob(hash)` call. Post-b.9qd its job is
//     inverted: it is the witness that the prune deletes NOTHING inline.
//
//   - Each test cancels the checkpoint loop FIRST (via `CancellationToken`)
//     before reading the recorder and the ZSET, so a late-firing tick cannot
//     land work between the read and the assert.

use std::collections::HashSet;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use async_trait::async_trait;
use bytes::Bytes;
use fred::interfaces::SortedSetsInterface;
use tokio::time::{sleep, timeout};
use tokio_util::sync::CancellationToken;

use live_edit::checkpoint::adapters::InMemoryCheckpointStore;
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
// `RecordingCheckpointStore` — wraps `InMemoryCheckpointStore`, records
// every `delete_ext_blob(hash)` call.
//
// Every non-recording trait method forwards to the inner store — including the
// b.9qd reconciliation trio (`list_layer_artifacts`,
// `delete_layer_temp_artifacts`, `publish_worst_case`) — so the wrapper is
// purely additive observation and never a fault-injection vector. An
// un-forwarded method silently substitutes the trait default, which is the
// quietest possible way to make a test pass vacuously.
// ────────────────────────────────────────────────────────────────────────

struct RecordingCheckpointStore {
    inner: InMemoryCheckpointStore,
    delete_log: Arc<Mutex<Vec<String>>>,
}

impl RecordingCheckpointStore {
    fn new() -> Self {
        Self {
            inner: InMemoryCheckpointStore::new(),
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
        // Record BEFORE delegating so a delegate failure still surfaces the
        // intent. Post-b.9qd every entry in this log during a prune-only run
        // is a regression: the prune must not delete ext blobs.
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
    // These three MUST forward even though nothing here records them.
    // Omitting them silently inherits the trait defaults —
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
// Hash construction helpers.
//
// Each `hash_for(byte)` produces a 32-byte all-`byte` hash whose hex form is
// trivially predictable, so the tests can assert on the ZSET's members
// without computing SHA-256s of real ext-value content.
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

// ────────────────────────────────────────────────────────────────────────
// Test-fixture wiring helpers.
// ────────────────────────────────────────────────────────────────────────

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
        .create(name, "/stage", "cascade-test-user")
        .await
        .expect("session creation must succeed")
}

/// Members of `le:ckpt:{recon}:tomb` — the hashes the retention prune has
/// recorded as candidates for a later collect.
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

/// The Valkey-side first-observation score of one tombstone, or `None`.
async fn tombstone_score(tv: &TestValkey, hash: &str) -> Option<f64> {
    tv.client()
        .pool
        .zscore(tv.key_schema().checkpoint_reconcile_tombstones(), hash)
        .await
        .expect("ZSCORE must succeed")
}

/// Build a fast-path-eligible incremental delta that sets one `(section, key)`
/// pair to an external-value reference. `is_diff=true, base_version=1` keeps
/// the Lua fast path eligible (a `base_version=0` diff would route to the
/// reset branch).
fn ext_set_field_delta(section_id: u64, key: &str, hash_byte: u8, set_order: u64) -> OwnedDelta {
    OwnedDelta {
        is_diff: true,
        base_version: 1,
        set_fields: vec![SetField {
            section_id,
            key_name: key.to_string(),
            // The protocol requires `value_or_ext_hash` to be exactly 32 bytes
            // when `ext_value_size > 0`; `field_value_from_oodf` rejects
            // anything else as `InvalidExternalHashLength`.
            value_or_ext_hash: hash_bytes(hash_byte),
            ext_value_size: 1,
            ext_value_index: 0,
            set_order,
        }],
        ..Default::default()
    }
}

/// Same shape as [`ext_set_field_delta`] but writes two fields in one delta
/// so a single commit can place two distinct hashes into the post-commit
/// snapshot (and therefore into the sidecar the assembler emits on the next
/// tick).
fn two_field_ext_delta(
    section_id: u64,
    key_a: &str,
    hash_a: u8,
    key_b: &str,
    hash_b: u8,
    base_set_order: u64,
) -> OwnedDelta {
    OwnedDelta {
        is_diff: true,
        base_version: 1,
        set_fields: vec![
            SetField {
                section_id,
                key_name: key_a.to_string(),
                value_or_ext_hash: hash_bytes(hash_a),
                ext_value_size: 1,
                ext_value_index: 0,
                set_order: base_set_order,
            },
            SetField {
                section_id,
                key_name: key_b.to_string(),
                value_or_ext_hash: hash_bytes(hash_b),
                ext_value_size: 1,
                ext_value_index: 0,
                set_order: base_set_order + 1,
            },
        ],
        ..Default::default()
    }
}

/// Poll `list_layer_checkpoints(oid).len()` until at least `at_least`. Keeps
/// coordination off `tokio::time::sleep` and behind a deadline so a regression
/// that wedges the loop surfaces as a deterministic timeout.
async fn wait_for_at_least(
    store: &Arc<dyn CheckpointStore>,
    object_id: u64,
    at_least: usize,
) -> usize {
    let deadline = Duration::from_secs(10);
    let result = timeout(deadline, async {
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
    .await;
    result.unwrap_or_else(|_| {
        panic!(
            "timed out waiting for list_layer_checkpoints(oid={object_id}).len() >= {at_least} \
             within {deadline:?}"
        )
    })
}

/// Wait for the head of `list_layer_checkpoints(object_id)` to advance past
/// `prev_snap_seq` — i.e. for the loop to have written a *new* generation
/// since `prev_snap_seq`.
async fn wait_for_head_snap_seq_above(
    store: &Arc<dyn CheckpointStore>,
    object_id: u64,
    prev_snap_seq: u64,
) -> u64 {
    let deadline = Duration::from_secs(10);
    let result = timeout(deadline, async {
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
    .await;
    result.unwrap_or_else(|_| {
        panic!(
            "timed out waiting for head snap_seq > {prev_snap_seq} on object {object_id} \
             within {deadline:?}"
        )
    })
}

/// Wait for the per-object generation count to settle back at the retain cap.
async fn wait_for_retain_cap(
    store: &Arc<dyn CheckpointStore>,
    object_id: u64,
    cap: usize,
) -> usize {
    let deadline = Duration::from_secs(10);
    timeout(deadline, async {
        loop {
            let len = store
                .list_layer_checkpoints(object_id)
                .await
                .expect("list must succeed")
                .len();
            if len <= cap {
                return len;
            }
            sleep(Duration::from_millis(25)).await;
        }
    })
    .await
    .unwrap_or_else(|_| {
        panic!("retain={cap} cap never re-enforced on object {object_id} within {deadline:?}")
    })
}

/// Wait for `hash` to appear on the tombstone ZSET, bounded. The retention
/// prune records tombstones inside the same tick body that deletes the
/// generations, so the cap-poll can exit fractionally ahead of the ZADD.
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
    // tick=50ms gives ~20 ticks/sec; interval=0 + quiesce=0 disables the
    // skip-if-idle gates so every commit becomes immediately due.
    CheckpointLoopConfig {
        tick: Duration::from_millis(50),
        interval: Duration::ZERO,
        quiesce: Duration::ZERO,
        retain_generations,
        // Epic 6 counters-sync cadence; pushed well above the test runtime
        // so the counters-sidecar timer does not fire during this test.
        counters_sync: Duration::from_secs(3600),
        // b.9qd reconciliation cadence, likewise pushed past the test runtime.
        // `tokio::time::interval` fires immediately on its first tick, so one
        // reconciliation pass DOES run at t=0 against an empty store (no
        // candidates, no tombstones, no-op); no further pass fires.
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
// AC #1 (re-targeted) — the prune records the pruned generation's WHOLE hash
// set on the tombstone ZSET, and deletes no ext blob itself.
//
// Three generations referencing distinct ext-blob sets:
//
//   - gen 1 snapshot has fields {a=H1, b=H2}    -> sidecar {H1, H2}
//   - gen 2 snapshot has fields {a=H2, b=H3}    -> sidecar {H2, H3}
//   - gen 3 snapshot has fields {a=H3, b=H3}    -> sidecar {H3} (dedup)
//
// retain=2 prunes the initial (empty-sidecar) generation and gen 1. Pre-b.9qd
// the loop computed `{H1, H2} − {H2, H3} = {H1}` and deleted H1 inline. It now
// records `{H1, H2}` — including H2, which is still live — and deletes
// nothing. That difference is the point: **recording is not classifying**. The
// verdict moves to the collect phase, which re-reads the live set a grace
// window later and spares H2 (pinned in-crate by
// `checkpointer::tests::prune_tombstones_then_collect_deletes_only_the_unique_hash`).
// ────────────────────────────────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn prune_tombstones_the_pruned_generations_hashes_and_deletes_no_blob_inline() {
    let tv = TestValkey::setup_with_pool_size(8).await;
    let ls = layer_store(&tv);
    let keys = tv.key_schema().clone();
    let client = tv.client().clone();
    let session_id = create_test_session(&tv, "cascade-ac1-session").await;

    let initial_state = ObjectState {
        sequence_num: 1,
        first_version: 1,
        source_format: "usd".to_string(),
        ..Default::default()
    };
    let oid = ls
        .create_layer(
            &session_id,
            "cascade-layer-ac1",
            "cascade-test-user",
            &initial_state,
            mock_ext_store(),
        )
        .await
        .expect("create_layer must succeed");

    let recording = Arc::new(RecordingCheckpointStore::new());
    let delete_log = recording.delete_log();
    let store: Arc<dyn CheckpointStore> = recording;
    let cancel = CancellationToken::new();
    let handle = spawn_checkpoint_loop(
        client,
        keys,
        Arc::clone(&store),
        loop_config("cascade-ac1-replica", 2),
        None,
        cancel.clone(),
    );

    // Wait for the initial CreateLayer tick to land its first generation so
    // we have a known head snap_seq to walk past on each subsequent commit.
    assert!(
        wait_for_at_least(&store, oid, 1).await >= 1,
        "initial generation must land"
    );
    let mut last_head = store
        .list_layer_checkpoints(oid)
        .await
        .expect("list must succeed")
        .first()
        .expect("head present")
        .snap_seq;

    for (i, (a_hash, b_hash, base_order)) in [
        (0x01_u8, 0x02_u8, 10_u64),
        (0x02_u8, 0x03_u8, 20_u64),
        (0x03_u8, 0x03_u8, 30_u64),
    ]
    .iter()
    .enumerate()
    {
        ls.update_layer_relayed(
            &session_id,
            "cascade-layer-ac1",
            oid,
            Some(two_field_ext_delta(
                1,
                "a",
                *a_hash,
                "b",
                *b_hash,
                *base_order,
            )),
            "cascade-client",
            1_000 + i as u64,
        )
        .await
        .expect("update_layer must succeed");
        last_head = wait_for_head_snap_seq_above(&store, oid, last_head).await;
    }
    let _ = last_head;

    let final_len = wait_for_retain_cap(&store, oid, 2).await;
    assert_eq!(
        final_len, 2,
        "retain=2 must cap the store at exactly 2 generations after the prune tick"
    );

    let h1 = hash_hex(0x01);
    let h2 = hash_hex(0x02);
    let h3 = hash_hex(0x03);
    assert!(
        wait_for_tombstone(&tv, &h1).await,
        "the prune must record the pruned generation's hashes on le:ckpt:{{recon}}:tomb; \
         tombstones = {:?}",
        tombstones(&tv).await
    );

    cancel_and_join(cancel, handle).await;

    let recorded_tombstones = tombstones(&tv).await;
    assert!(
        recorded_tombstones.contains(&h1),
        "H1 was unique to the pruned generation and must be recorded for collection; \
         got {recorded_tombstones:?}"
    );
    assert!(
        recorded_tombstones.contains(&h2),
        "H2 must ALSO be recorded even though a retained generation still references it. The \
         prune records the pruned generation's whole hash set and does not classify — \
         `pruned − live` was exactly the racy step b.9qd moved behind a grace window, and \
         tombstoning a live hash is harmless because the collect re-reads the live set and \
         spares it. Got {recorded_tombstones:?}"
    );
    assert!(
        !recorded_tombstones.contains(&h3),
        "H3 belongs only to retained generations — nothing pruned it, so nothing may record \
         it. Got {recorded_tombstones:?}"
    );

    let deleted = snapshot_delete_log(&delete_log);
    assert!(
        deleted.is_empty(),
        "the retention prune must delete NO ext blob inline. The inline cascade is what b.9qd \
         replaced: it could classify a concurrently-publishing generation's hashes as prunable \
         and delete bytes a durable checkpoint points at. Got {deleted:?}"
    );
}

// ────────────────────────────────────────────────────────────────────────
// AC #2 (re-targeted) — a tombstone is a claim, not a verdict.
//
// Three commits all sticking H1 onto the same `(section, key)` pair, so every
// generation's sidecar is `{H1}`. The pruned generation's hash IS recorded —
// even though every retained generation still references it — and no blob is
// deleted. Pre-b.9qd this test asserted an empty recorder, which post-b.9qd
// would be satisfied by a prune that recorded nothing at all; asserting the
// ZSET content is what keeps it non-vacuous.
// ────────────────────────────────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn prune_tombstones_a_still_live_hash_without_deleting_it() {
    let tv = TestValkey::setup_with_pool_size(8).await;
    let ls = layer_store(&tv);
    let keys = tv.key_schema().clone();
    let client = tv.client().clone();
    let session_id = create_test_session(&tv, "cascade-ac2-session").await;

    let initial_state = ObjectState {
        sequence_num: 1,
        first_version: 1,
        source_format: "usd".to_string(),
        ..Default::default()
    };
    let oid = ls
        .create_layer(
            &session_id,
            "cascade-layer-ac2",
            "cascade-test-user",
            &initial_state,
            mock_ext_store(),
        )
        .await
        .expect("create_layer must succeed");

    let recording = Arc::new(RecordingCheckpointStore::new());
    let delete_log = recording.delete_log();
    let store: Arc<dyn CheckpointStore> = recording;
    let cancel = CancellationToken::new();
    let handle = spawn_checkpoint_loop(
        client,
        keys,
        Arc::clone(&store),
        loop_config("cascade-ac2-replica", 2),
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

    for (i, set_order) in [10_u64, 20_u64, 30_u64].iter().enumerate() {
        ls.update_layer_relayed(
            &session_id,
            "cascade-layer-ac2",
            oid,
            Some(ext_set_field_delta(1, "a", 0x01, *set_order)),
            "cascade-client",
            2_000 + i as u64,
        )
        .await
        .expect("update_layer must succeed");
        last_head = wait_for_head_snap_seq_above(&store, oid, last_head).await;
    }
    let _ = last_head;

    wait_for_retain_cap(&store, oid, 2).await;

    let h1 = hash_hex(0x01);
    assert!(
        wait_for_tombstone(&tv, &h1).await,
        "tombstones = {:?}",
        tombstones(&tv).await
    );

    cancel_and_join(cancel, handle).await;

    assert_eq!(
        tombstones(&tv).await,
        [h1.clone()].into_iter().collect::<HashSet<String>>(),
        "exactly the pruned generation's single hash is recorded"
    );
    let deleted = snapshot_delete_log(&delete_log);
    assert!(
        deleted.is_empty(),
        "and nothing is deleted: a tombstone is a claim the collect phase adjudicates against \
         the live set a grace window later, not a deletion. H1 is in fact still referenced by \
         both retained generations, so the collect would spare it — but the prune does not \
         know that and must not act as though it did. Got {deleted:?}"
    );
}

// ────────────────────────────────────────────────────────────────────────
// AC #3 (re-targeted) — cross-object: a hash another object still references
// is tombstoned by the prune and deleted by nobody.
//
// Two distinct object_ids both referencing H1. Pruning O1's oldest generation
// records H1 — the prune has no cross-object knowledge and is not supposed to
// — while the blob stays intact.
//
// The "live set is global, not per-object" invariant itself moved to the
// collect phase, where a single `list_referenced_ext_hashes` read adjudicates
// it; it is pinned in-crate by
// `checkpointer::tests::prune_does_not_delete_cross_referenced_hash` and
// `checkpointer::reconcile_tests::collect_deletes_the_unreferenced_hash_and_spares_the_still_live_one`.
// What THIS test adds is that the loop, driving two real objects through the
// assembler, never reaches for a delete on the way there.
// ────────────────────────────────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn cross_object_hash_is_tombstoned_by_the_prune_and_deleted_by_nobody() {
    let tv = TestValkey::setup_with_pool_size(8).await;
    let ls = layer_store(&tv);
    let keys = tv.key_schema().clone();
    let client = tv.client().clone();
    let session_id = create_test_session(&tv, "cascade-ac3-session").await;

    let initial_state = ObjectState {
        sequence_num: 1,
        first_version: 1,
        source_format: "usd".to_string(),
        ..Default::default()
    };
    let oid_a = ls
        .create_layer(
            &session_id,
            "cascade-layer-ac3-a",
            "cascade-test-user",
            &initial_state,
            mock_ext_store(),
        )
        .await
        .expect("create_layer A must succeed");
    let oid_b = ls
        .create_layer(
            &session_id,
            "cascade-layer-ac3-b",
            "cascade-test-user",
            &initial_state,
            mock_ext_store(),
        )
        .await
        .expect("create_layer B must succeed");

    let recording = Arc::new(RecordingCheckpointStore::new());
    let delete_log = recording.delete_log();
    let store: Arc<dyn CheckpointStore> = recording;
    let cancel = CancellationToken::new();
    let handle = spawn_checkpoint_loop(
        client,
        keys,
        Arc::clone(&store),
        loop_config("cascade-ac3-replica", 2),
        None,
        cancel.clone(),
    );

    wait_for_at_least(&store, oid_a, 1).await;
    wait_for_at_least(&store, oid_b, 1).await;
    let mut last_head_a = store
        .list_layer_checkpoints(oid_a)
        .await
        .expect("list A must succeed")
        .first()
        .expect("head A present")
        .snap_seq;
    let last_head_b = store
        .list_layer_checkpoints(oid_b)
        .await
        .expect("list B must succeed")
        .first()
        .expect("head B present")
        .snap_seq;

    // O1: three commits, each referencing H1 (shared) plus a per-commit
    // marker hash so each generation's sidecar is distinct.
    for (i, marker_hash) in [0x04_u8, 0x05_u8, 0x06_u8].iter().enumerate() {
        ls.update_layer_relayed(
            &session_id,
            "cascade-layer-ac3-a",
            oid_a,
            Some(two_field_ext_delta(
                1,
                "shared",
                0x01,
                "marker",
                *marker_hash,
                10 + i as u64 * 10,
            )),
            "cascade-client",
            3_000 + i as u64,
        )
        .await
        .expect("update_layer on A must succeed");
        last_head_a = wait_for_head_snap_seq_above(&store, oid_a, last_head_a).await;
    }
    let _ = last_head_a;

    // O2: one commit referencing H1 only — the cross-object live reference.
    ls.update_layer_relayed(
        &session_id,
        "cascade-layer-ac3-b",
        oid_b,
        Some(ext_set_field_delta(1, "shared", 0x01, 10)),
        "cascade-client",
        3_100,
    )
    .await
    .expect("update_layer on B must succeed");
    let _ = wait_for_head_snap_seq_above(&store, oid_b, last_head_b).await;

    wait_for_retain_cap(&store, oid_a, 2).await;

    let h1 = hash_hex(0x01);
    assert!(
        wait_for_tombstone(&tv, &h1).await,
        "tombstones = {:?}",
        tombstones(&tv).await
    );

    cancel_and_join(cancel, handle).await;

    let recorded = tombstones(&tv).await;
    assert!(
        recorded.contains(&h1),
        "H1 rode on O1's pruned generation, so the prune records it — deliberately without \
         consulting whether object B still references it. Got {recorded:?}"
    );
    let deleted = snapshot_delete_log(&delete_log);
    assert!(
        deleted.is_empty(),
        "no ext blob may be deleted by the loop's retention prune, on any object — and \
         object B's live generation still references H1, so a prune that DID delete would be \
         destroying bytes a durable checkpoint points at. Got {deleted:?}"
    );
}

// ────────────────────────────────────────────────────────────────────────
// AC #4 (re-targeted) — quiescent re-ticking neither re-records nor deletes.
//
// After the first prune records its tombstones, the loop keeps ticking with no
// new commits. What must hold is that a quiescent tick is genuinely inert:
//
//   - no new tombstone members;
//   - no existing member's score moves; and
//   - no `delete_ext_blob` call.
//
// Note what the score assertion is and is not. A quiescent tick issues no ZADD
// at all — there is nothing to prune, so the sink is never called — which means
// this is an "the ZSET was not touched" observation, NOT a test of the `ZADD NX`
// pin. (Read the other way round it would be vacuous: a member nobody wrote to
// cannot have its score refreshed.) The genuine NX pins, where a *second* write
// against an already-registered member is driven and the score is asserted
// unchanged, live in-crate:
// `checkpointer::reconcile_tests::orphan_reaping_defers_on_the_first_sighting_and_acts_on_the_second`
// for the orphan ZSET and the `record`-twice path for the tombstone ZSET. What
// this file uniquely adds is that the REAL loop, ticking with nothing to do,
// leaves the durable state alone.
// ────────────────────────────────────────────────────────────────────────

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn quiescent_ticks_neither_re_record_nor_delete() {
    let tv = TestValkey::setup_with_pool_size(8).await;
    let ls = layer_store(&tv);
    let keys = tv.key_schema().clone();
    let client = tv.client().clone();
    let session_id = create_test_session(&tv, "cascade-ac4-session").await;

    let initial_state = ObjectState {
        sequence_num: 1,
        first_version: 1,
        source_format: "usd".to_string(),
        ..Default::default()
    };
    let oid = ls
        .create_layer(
            &session_id,
            "cascade-layer-ac4",
            "cascade-test-user",
            &initial_state,
            mock_ext_store(),
        )
        .await
        .expect("create_layer must succeed");

    let recording = Arc::new(RecordingCheckpointStore::new());
    let delete_log = recording.delete_log();
    let store: Arc<dyn CheckpointStore> = recording;
    let cancel = CancellationToken::new();
    let handle = spawn_checkpoint_loop(
        client,
        keys,
        Arc::clone(&store),
        loop_config("cascade-ac4-replica", 2),
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

    for (i, (a_hash, b_hash, base_order)) in [
        (0x01_u8, 0x02_u8, 10_u64),
        (0x02_u8, 0x03_u8, 20_u64),
        (0x03_u8, 0x03_u8, 30_u64),
    ]
    .iter()
    .enumerate()
    {
        ls.update_layer_relayed(
            &session_id,
            "cascade-layer-ac4",
            oid,
            Some(two_field_ext_delta(
                1,
                "a",
                *a_hash,
                "b",
                *b_hash,
                *base_order,
            )),
            "cascade-client",
            4_000 + i as u64,
        )
        .await
        .expect("update_layer must succeed");
        last_head = wait_for_head_snap_seq_above(&store, oid, last_head).await;
    }
    let _ = last_head;

    wait_for_retain_cap(&store, oid, 2).await;
    let h1 = hash_hex(0x01);
    assert!(
        wait_for_tombstone(&tv, &h1).await,
        "the first prune must record its tombstones; got {:?}",
        tombstones(&tv).await
    );

    let first_tombstones = tombstones(&tv).await;
    let first_score = tombstone_score(&tv, &h1)
        .await
        .expect("the recorded tombstone must carry a score");

    // Drive the loop forward with no new commits: ~12 ticks at 50 ms. The
    // assertion would fail loud if anything re-fired, so an over-long
    // quiescence allowance is the safe side.
    sleep(Duration::from_millis(600)).await;

    cancel_and_join(cancel, handle).await;

    assert_eq!(
        tombstones(&tv).await,
        first_tombstones,
        "quiescent ticks must not add tombstone members — there is nothing left to prune"
    );
    assert_eq!(
        tombstone_score(&tv, &h1).await,
        Some(first_score),
        "and an existing member's score must be exactly where the first prune left it. A \
         quiescent tick writes nothing, so this is the 'the loop did not touch the ZSET' \
         observation — the aging clock a tombstone is judged against is the one stamped when \
         it was first recorded, and a loop that re-stamped it on an idle tick would keep every \
         tombstone permanently inside its grace window. (The ZADD-NX behaviour on a genuine \
         re-write is pinned in-crate; see the block comment above.)"
    );
    let deleted = snapshot_delete_log(&delete_log);
    assert!(
        deleted.is_empty(),
        "and no quiescent tick may delete an ext blob. Got {deleted:?}"
    );
}
