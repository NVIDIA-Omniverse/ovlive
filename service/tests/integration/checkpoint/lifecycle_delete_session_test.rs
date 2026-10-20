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

//! Subtask `t3.fgj.7v.p9.c3` — `DeleteSession` lifecycle hook MUST route
//! through the `delete_session_checkpoint` dispatch path, NOT the
//! `put_session_checkpoint` path used by the other three lifecycle events
//! (CreateSession / CreateLayer / DeleteLayer).
//!
//! Per the Epic 7 cross-Task contract with Task 6 (`t2.fgj.7v.fp`),
//! `SessionStore::delete` is the one asymmetric hook: it tombstones the
//! session-checkpoint so the next request's recovery hook does NOT silently
//! resurrect the just-deleted session. Task 6 relies on the invariant that
//! `get_session_checkpoint(session_id)` returns `Ok(None)` after
//! `SessionStore::delete` completes.
//!
//! Inline `#[cfg(test)]` tests on the dispatch primitive cannot verify the
//! call-site wiring picked the `delete_*` branch (versus the `put_*` branch
//! the other three lifecycle events use): the call-site lives in
//! `SessionStore::delete`'s success arm, downstream of the EXEC of the
//! meta+layers-index MULTI. This integration test pins that wiring against
//! a real `TestValkey` + a counting wrapper that exposes both
//! `put_session_checkpoint_calls` and `delete_session_checkpoint_calls`.
//!
//! The dual-counter approach is load-bearing: a regression that wired
//! `SessionStore::delete` to `enqueue_session_checkpoint_write` instead of
//! `enqueue_session_checkpoint_delete` would still produce a non-zero
//! checkpoint-store interaction (one `put_session_checkpoint` call) and the
//! Task-6 recovery hook would see the just-deleted session's body, not
//! `Ok(None)`. We therefore assert BOTH that the delete counter incremented
//! by exactly 1 AND that the put counter did NOT tick on the delete-path.
//!
//! Sandbox env note: this test uses `TestValkey::setup`, which reads
//! `TEST_VALKEY_URL`. Under the project sandbox the URL must include the
//! password and `host.docker.internal` host; see CLAUDE.md.

use std::collections::HashSet;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use bytes::Bytes;
use tokio_util::task::TaskTracker;

use live_edit::checkpoint::adapters::InMemoryCheckpointStore;
use live_edit::checkpoint::{
    CheckpointBlob, CheckpointError, CheckpointHandle, CheckpointMeta, CheckpointStore,
    SessionCheckpointWriter,
};
use live_edit::observability::metrics::SharedMetrics;
use live_edit::storage::session::SessionStore;
use live_edit::storage::test_support::TestValkey;

const SERVICE_LABEL: &str = "lifecycle-delete-session-test-replica";

// =============================================================================
// Counting session-checkpoint store wrapper.
// =============================================================================

/// Wraps an [`InMemoryCheckpointStore`] and counts every
/// `put_session_checkpoint` and `delete_session_checkpoint` call separately
/// via `AtomicUsize`. Mirrors the `CountingCheckpointStore` pattern in
/// `tests/integration/checkpoint/recovery_admission_test.rs` — composes
/// `super::helpers::DelegatingCheckpointStore<InMemoryCheckpointStore>` for
/// the verbatim 14-method delegation and overrides only the two methods that
/// actually count. The remaining method bodies forward to `self.delegate` so
/// the per-file boilerplate centralises around one source of truth for the
/// delegation shape.
///
/// Counting both the put and the delete (rather than just the delete) is the
/// load-bearing instrumentation for the asymmetry assertion: a regression
/// that wired `SessionStore::delete` to `enqueue_session_checkpoint_write`
/// would tick the put counter on the delete-path AND leave the delete
/// counter at zero — exactly the inverse of the contract under test.
struct CountingSessionCheckpointStore {
    delegate: super::helpers::DelegatingCheckpointStore<InMemoryCheckpointStore>,
    put_session_checkpoint_calls: AtomicUsize,
    delete_session_checkpoint_calls: AtomicUsize,
}

impl CountingSessionCheckpointStore {
    fn new(inner: Arc<InMemoryCheckpointStore>) -> Self {
        Self {
            delegate: super::helpers::DelegatingCheckpointStore::new(inner),
            put_session_checkpoint_calls: AtomicUsize::new(0),
            delete_session_checkpoint_calls: AtomicUsize::new(0),
        }
    }

    fn put_count(&self) -> usize {
        self.put_session_checkpoint_calls.load(Ordering::Acquire)
    }

    fn delete_count(&self) -> usize {
        self.delete_session_checkpoint_calls.load(Ordering::Acquire)
    }
}

#[async_trait]
impl CheckpointStore for CountingSessionCheckpointStore {
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
        // Increment BEFORE the inner call so concurrent callers cannot mask
        // the counter from observers. The CreateSession path drives this on
        // the success arm; the assertion below snapshots the value AFTER
        // `tracker.wait()` returns, by which point every spawned task has
        // either ticked here or short-circuited silently.
        self.put_session_checkpoint_calls
            .fetch_add(1, Ordering::AcqRel);
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
        // Same ordering rationale as `put_session_checkpoint` above. The
        // DeleteSession path is the one lifecycle event that MUST hit this
        // arm; the test below asserts this counter incremented by exactly 1
        // and the put counter did NOT tick on the same delete-path.
        self.delete_session_checkpoint_calls
            .fetch_add(1, Ordering::AcqRel);
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
// Test case.
// =============================================================================

/// `SessionStore::delete` MUST enqueue a `delete_session_checkpoint` call
/// (NOT a `put_session_checkpoint` call). After the delete-enqueue drains,
/// the underlying store's `get_session_checkpoint(session_id)` MUST return
/// `Ok(None)` — the Task 6 cross-Task contract.
///
/// Drive shape:
/// 1. Wire `SessionStore` + `SessionCheckpointWriter` against a
///    [`CountingSessionCheckpointStore`] over a shared [`InMemoryCheckpointStore`].
/// 2. `SessionStore::create` → drain tracker → snapshot put/delete counts.
/// 3. `SessionStore::delete` → drain tracker → assert delete += 1, put += 0.
/// 4. Assert `get_session_checkpoint` returns `Ok(None)` (Task 6 contract).
///
/// Note on tracker isolation: a `TaskTracker::wait()` future resolves only
/// once the tracker is **closed** AND every spawned task has drained.
/// Calling `wait()` without first `close()`-ing the tracker hangs
/// indefinitely (the in-flight count can transition to zero and back as
/// new tasks spawn). And a `close()` is terminal — once closed, future
/// `tracker.spawn(...)` calls drop the task on the floor. We therefore
/// build **two** `SessionCheckpointWriter`s — one for create, one for
/// delete — each with its own `TaskTracker` so we can `close()` the
/// create-tracker after Phase 1 without leaking the close into Phase 2's
/// delete-enqueue. Both writers share the same underlying
/// [`CountingSessionCheckpointStore`] over a shared
/// [`InMemoryCheckpointStore`], so the put-vs-delete counter snapshot is
/// observable end-to-end. A single `SessionStore` instance cannot be
/// re-wired with a different writer mid-lifetime — `with_checkpoint_writer`
/// is a consume-self builder — so each phase uses its own `SessionStore`
/// over the same `TestValkey` (the Valkey state IS the shared lifecycle
/// surface; only the dispatch-side writer needs to be per-phase).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn delete_session_enqueues_delete_session_checkpoint() {
    let tv = TestValkey::setup().await;

    let metrics: SharedMetrics = Arc::new(crate::common::metrics_scrape_only(SERVICE_LABEL));

    // Build the counting wrapper over a shared in-memory store. We hold the
    // counting wrapper through a typed `Arc<CountingSessionCheckpointStore>`
    // so the test can read the counters after each phase; the writers hold
    // it through `Arc<dyn CheckpointStore>` (the trait-object shape the
    // dispatcher accepts).
    let inner = Arc::new(InMemoryCheckpointStore::new());
    let counting = Arc::new(CountingSessionCheckpointStore::new(Arc::clone(&inner)));
    let store: Arc<dyn CheckpointStore> = Arc::clone(&counting) as Arc<dyn CheckpointStore>;

    // -------------------------------------------------------------------------
    // Phase 1 — create the session, drain its put-enqueue, snapshot counters.
    // Each phase gets its own writer + tracker so we can `close()+wait()` at
    // the end of the phase without poisoning the next phase's enqueue path.
    // -------------------------------------------------------------------------
    let create_tracker = TaskTracker::new();
    let create_writer = Arc::new(SessionCheckpointWriter::new(
        Arc::clone(&store),
        tv.client().clone(),
        tv.key_schema().clone(),
        create_tracker.clone(),
        Arc::clone(&metrics),
        SERVICE_LABEL.to_string(),
        true,
    ));
    let create_session_store = SessionStore::new(tv.client().clone(), tv.key_schema().clone())
        .with_checkpoint_writer(Arc::clone(&create_writer));

    let session_id = create_session_store
        .create(
            "p9-c3-delete-routing-session",
            "/p9-c3/delete-routing",
            "p9-c3-test-user",
        )
        .await
        .expect("SessionStore::create must succeed against a healthy Valkey");
    assert!(
        !session_id.is_empty(),
        "session_id must be a non-empty UUID4 — empty value indicates an unexpected early-return path"
    );

    // Drain in-flight create-side puts. Closing the tracker is what makes
    // `wait()` resolve once the in-flight count hits zero. The
    // create_tracker is dead after this point; the Phase-2 delete-enqueue
    // uses a separate tracker.
    create_tracker.close();
    tokio::time::timeout(Duration::from_secs(10), create_tracker.wait())
        .await
        .expect(
            "create_tracker.wait() must drain inside 10s — a hang here points at a \
             SessionCheckpointWriter bug, not a test-only flake",
        );

    let put_count_after_create = counting.put_count();
    let delete_count_after_create = counting.delete_count();
    assert_eq!(
        put_count_after_create, 1,
        "SessionStore::create must enqueue exactly one put_session_checkpoint call; \
         got put_count={put_count_after_create}. The create-phase counter snapshot is \
         the baseline against which the delete-phase asymmetry assertion is made."
    );
    assert_eq!(
        delete_count_after_create, 0,
        "SessionStore::create must NOT enqueue a delete_session_checkpoint call; \
         got delete_count={delete_count_after_create}. The create-path must route \
         through the put-branch, not the delete-branch."
    );

    // -------------------------------------------------------------------------
    // Phase 2 — delete the session, drain its delete-enqueue, assert
    //           branch routing (delete += 1, put += 0).
    // -------------------------------------------------------------------------
    let delete_tracker = TaskTracker::new();
    let delete_writer = Arc::new(SessionCheckpointWriter::new(
        Arc::clone(&store),
        tv.client().clone(),
        tv.key_schema().clone(),
        delete_tracker.clone(),
        Arc::clone(&metrics),
        SERVICE_LABEL.to_string(),
        true,
    ));
    let delete_session_store = SessionStore::new(tv.client().clone(), tv.key_schema().clone())
        .with_checkpoint_writer(Arc::clone(&delete_writer));

    delete_session_store
        .delete(&session_id)
        .await
        .expect("SessionStore::delete must succeed for a freshly-created session");

    delete_tracker.close();
    tokio::time::timeout(Duration::from_secs(10), delete_tracker.wait())
        .await
        .expect(
            "delete_tracker.wait() must drain inside 10s — a hang here points at a \
             SessionCheckpointWriter bug, not a test-only flake",
        );

    let put_count_after_delete = counting.put_count();
    let delete_count_after_delete = counting.delete_count();

    // The load-bearing branch-routing assertions: delete += 1, put += 0.
    // A regression that wired `SessionStore::delete` to
    // `enqueue_session_checkpoint_write` would invert these — put_count
    // would be 2 and delete_count would be 0.
    assert_eq!(
        delete_count_after_delete - delete_count_after_create,
        1,
        "SessionStore::delete must increment delete_session_checkpoint_calls by exactly 1; \
         got delta={} (before={delete_count_after_create}, after={delete_count_after_delete}). \
         A regression that routed DeleteSession to the put-branch would leave this delta at 0.",
        delete_count_after_delete - delete_count_after_create
    );
    assert_eq!(
        put_count_after_delete, put_count_after_create,
        "SessionStore::delete must NOT increment put_session_checkpoint_calls; \
         got put_count_after_delete={put_count_after_delete}, \
         put_count_after_create={put_count_after_create}. The delete-path must NOT \
         fall through to the put-path — Task 6's recovery hook relies on the absence \
         of a put on the delete-path so a just-deleted session is not silently resurrected."
    );

    // -------------------------------------------------------------------------
    // Phase 3 — Task 6 cross-Task contract: post-delete the checkpoint
    //           store reports `Ok(None)` for the session.
    // -------------------------------------------------------------------------
    let post_delete_blob = inner
        .get_session_checkpoint(&session_id)
        .await
        .expect("get_session_checkpoint must succeed against the in-memory adapter");
    assert!(
        post_delete_blob.is_none(),
        "get_session_checkpoint(session_id) MUST return Ok(None) after SessionStore::delete \
         drains (Task 6 cross-Task contract — recovery hook must not see the deleted session). \
         Got: {:?}",
        post_delete_blob.map(|b| (b.handle.adapter_key, b.bytes.len()))
    );
}
