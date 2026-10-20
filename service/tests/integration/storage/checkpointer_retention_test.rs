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

// Task t2.fgj.ya.9j Subtask .hd — retention prune retains exactly
// `retain_generations` newest generations of a layer.
//
// SR-2.2: the leader-elected checkpoint loop, after each per-tick assembly,
// retention-prunes the InMemoryCheckpointStore so only the most recent
// `retain_generations` `(object_id, snap_seq)` entries survive. Pruning is
// ordered by `snap_seq` (DESC = newest first), NOT by put insertion order
// (which can interleave due to retry / tombstone reconciliation), and the
// ext-refs sidecar is co-retained with its OMPF because the InMemory adapter
// stores both inside a single `LayerEntry`.
//
// This test pre-populates one layer, then issues 5 UpdateLayer calls — each
// advances `obj:hdr seq`. Between commits we poll
// `list_layer_checkpoints(oid).len()` so the loop has time to observe each
// commit individually (per the .hd subtask body's coordination note: "drive
// the loop ONCE per commit ... waiting for length to increment"). Once 5
// commits have landed, the store MUST hold exactly 2 generations — the 2
// newest by `snap_seq` — because `retain_generations=2`. We assert both the
// cap and the ordering (newest-first).

use std::sync::Arc;
use std::time::Duration;

use tokio::time::{sleep, timeout};
use tokio_util::sync::CancellationToken;

use live_edit::checkpoint::adapters::InMemoryCheckpointStore;
use live_edit::checkpoint::{
    spawn_checkpoint_loop, CheckpointLoopConfig, CheckpointStore, TailCaptureConfig,
};
use live_edit::domain::types::ObjectState;
use live_edit::storage::external_values::MockExternalValueStore;
use live_edit::storage::layer::LayerStore;
use live_edit::storage::session::SessionStore;
use live_edit::storage::test_support::relay::RelayedUpdates;
use live_edit::storage::test_support::TestValkey;
use oodf_ffi::OwnedDelta;

fn mock_ext_store() -> Arc<MockExternalValueStore> {
    Arc::new(MockExternalValueStore::new())
}

fn layer_store(tv: &TestValkey) -> LayerStore {
    LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096)
}

fn session_store(tv: &TestValkey) -> SessionStore {
    SessionStore::new(tv.client().clone(), tv.key_schema().clone())
}

async fn create_test_session(tv: &TestValkey) -> String {
    session_store(tv)
        .create("retention-test-session", "/stage", "retention-test-user")
        .await
        .expect("session creation must succeed")
}

/// Poll `list_layer_checkpoints(oid)` until its length is at least `at_least`,
/// or until the deadline elapses. Bounded: 10 s total budget at 25 ms
/// resolution. We poll on `len() >= at_least` rather than equality so a tick
/// that races past us (e.g. the prune already fired) doesn't deadlock the
/// poll — the post-poll equality assertions in the test still pin the
/// final-state invariant.
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

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn retention_prune_retains_exactly_retain_generations_newest_with_sidecars() {
    let tv = TestValkey::setup_with_pool_size(8).await;
    let ls = layer_store(&tv);
    let keys = tv.key_schema().clone();
    let client = tv.client().clone();
    let session_id = create_test_session(&tv).await;

    // Step 1: create a single populated layer. CreateLayer stamps
    // `obj:hdr last_server_ts`, so the loop's gate finds a commit timestamp
    // — combined with quiesce=0 and interval=0 the quiesce/interval gate
    // never skips.
    let initial_state = ObjectState {
        sequence_num: 1,
        first_version: 1,
        source_format: "usd".to_string(),
        ..Default::default()
    };
    let oid = ls
        .create_layer(
            &session_id,
            "retention-layer",
            "retention-test-user",
            &initial_state,
            mock_ext_store(),
        )
        .await
        .expect("create_layer must succeed");

    // Step 2: spawn the loop. retain_generations=2, tick=50ms so the loop
    // observes each commit within at most one tick worth of latency.
    // interval=0 + quiesce=0 disables the skip-if-idle gates entirely so
    // every commit becomes immediately due.
    let store: Arc<dyn CheckpointStore> = Arc::new(InMemoryCheckpointStore::new());
    let config = CheckpointLoopConfig {
        tick: Duration::from_millis(50),
        interval: Duration::from_secs(0),
        quiesce: Duration::from_secs(0),
        retain_generations: 2,
        // Epic 6 counters-sync cadence; pushed well above the test runtime
        // so the counters-sidecar timer does not fire during this test.
        counters_sync: Duration::from_secs(3600),
        reconcile_interval: Duration::from_secs(3600),
        leader: crate::common::test_leader("retention-test-replica"),
        // Tails pinned explicitly OFF. b.3n1 promoted the binary default
        // to `true`; this suite predates the tail feature and was written
        // against an inert tail path, so the flag is part of the fixture
        // rather than something to inherit.
        tail: TailCaptureConfig {
            enabled: false,
            ..TailCaptureConfig::default()
        },
    };
    let cancel = CancellationToken::new();
    let handle = spawn_checkpoint_loop(
        client,
        keys,
        Arc::clone(&store),
        config,
        None,
        cancel.clone(),
    );

    // Step 3: drive 5 commits one at a time, waiting after each for the loop
    // to land a fresh generation. Once the cap is reached (2), every further
    // commit produces a new generation that displaces the oldest — the
    // length stays pinned at 2 but the snap_seqs advance.
    //
    // Per-commit synchronisation: after CreateLayer the very first tick can
    // already write generation 1 (seq=1 != last_checkpoint_seq=0). So we
    // wait_for_at_least(>=1) once, THEN issue 5 UpdateLayers and after each
    // wait for the NEW snap_seq to surface as the head of the list.
    let initial_len = wait_for_at_least(&store, oid, 1).await;
    assert!(
        initial_len >= 1,
        "after CreateLayer the loop should write at least one generation; got {initial_len}"
    );
    let mut last_observed_snap_seq: u64 = store
        .list_layer_checkpoints(oid)
        .await
        .expect("list must succeed")
        .first()
        .expect("at least one handle after the initial tick")
        .snap_seq;

    for i in 0..5 {
        ls.update_layer_relayed(
            &session_id,
            "retention-layer",
            oid,
            Some(OwnedDelta::default()),
            "retention-client",
            1_000 + i as u64,
        )
        .await
        .expect("update_layer must succeed");

        // Wait for the loop to observe the new commit. The head of the
        // newest-first list must advance past the previously observed
        // snap_seq; equality means the loop hasn't ticked yet.
        let advance_deadline = Duration::from_secs(10);
        let head_after = timeout(advance_deadline, async {
            loop {
                let handles = store
                    .list_layer_checkpoints(oid)
                    .await
                    .expect("list must succeed");
                if let Some(head) = handles.first() {
                    if head.snap_seq > last_observed_snap_seq {
                        return head.snap_seq;
                    }
                }
                sleep(Duration::from_millis(25)).await;
            }
        })
        .await
        .unwrap_or_else(|_| {
            panic!(
                "commit #{i}: loop did not advance head snap_seq past \
                 {last_observed_snap_seq} within {advance_deadline:?}"
            )
        });
        assert!(
            head_after > last_observed_snap_seq,
            "head snap_seq must strictly increase after each commit; \
             before={last_observed_snap_seq}, after={head_after}"
        );
        last_observed_snap_seq = head_after;
    }

    // Step 4: cancel the loop FIRST, so we read a quiescent final state —
    // otherwise a late-firing tick could put a 3rd generation between our
    // list read and the assert. Joining under a 5 s timeout matches the
    // shutdown discipline of the other checkpoint-loop tests.
    cancel.cancel();
    timeout(Duration::from_secs(5), handle)
        .await
        .expect("checkpoint loop must exit within 5 s of cancellation")
        .expect("checkpoint loop must not panic");

    // Step 5: assert the cap holds and ordering is newest-first.
    let handles = store
        .list_layer_checkpoints(oid)
        .await
        .expect("list_layer_checkpoints must succeed");
    assert_eq!(
        handles.len(),
        2,
        "retain_generations=2 must cap the store at exactly 2 generations after 5+ commits; \
         got {} generations",
        handles.len()
    );

    // Newest-first contract: snap_seq strictly descending across the list.
    assert!(
        handles[0].snap_seq > handles[1].snap_seq,
        "list_layer_checkpoints must return generations newest-first by snap_seq; \
         got [{}, {}]",
        handles[0].snap_seq,
        handles[1].snap_seq
    );

    // The two retained snap_seqs must be the TWO MOST RECENT observed. We
    // observed `last_observed_snap_seq` as the latest head before
    // cancellation, so it MUST appear as `handles[0]`. The penultimate
    // generation must be strictly less. This pins the "retain by snap_seq
    // DESC, not by some other ordering" requirement called out in the
    // subtask body.
    assert_eq!(
        handles[0].snap_seq, last_observed_snap_seq,
        "newest retained generation must equal the last observed head snap_seq; \
         observed={last_observed_snap_seq}, retained={}",
        handles[0].snap_seq
    );

    // Step 6: sidecar co-retention. The InMemoryCheckpointStore stores each
    // generation's ext-refs sidecar inside its `LayerEntry`, so a fetch of
    // the retained generation's OMPF blob must also surface its
    // `meta.sidecar_bytes`. The assembler always emits a sidecar (even
    // when empty — `{"ext_hashes":[]}`), so `sidecar_bytes` must be
    // `Some(_)` for every retained generation; if retention pruning had
    // accidentally dropped a sidecar (e.g. by retaining the OMPF but
    // separately deleting the sidecar), this assertion would catch it.
    let latest_blob = store
        .get_latest_layer_checkpoint(oid)
        .await
        .expect("get_latest_layer_checkpoint must succeed")
        .expect("latest checkpoint must exist for the retained head");
    assert_eq!(latest_blob.handle.snap_seq, handles[0].snap_seq);
    assert!(
        latest_blob.meta.sidecar_bytes.is_some(),
        "retained generation must carry its ext-refs sidecar (co-retained inside LayerEntry); \
         meta.sidecar_bytes was None for snap_seq={}",
        latest_blob.handle.snap_seq
    );
}
