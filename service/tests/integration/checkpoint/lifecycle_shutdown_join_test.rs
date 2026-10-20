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

//! Subtask `t3.fgj.7v.p9.sy` — graceful shutdown MUST join in-flight
//! session-checkpoint writes via the shared `TaskTracker`.
//!
//! The Task body's Success Criteria require: "Graceful shutdown joins
//! in-flight session-checkpoint writes via the existing `TaskTracker`
//! (no orphan tasks after shutdown)." Per SDD §5.6, the shutdown sequence
//! joins background TaskTracker handles under the
//! `LIVE_EDIT_SHUTDOWN_TIMEOUT_SECS` budget; a session-checkpoint write
//! that lands AFTER `TaskTracker::close` was called violates the
//! bounded-shutdown invariant.
//!
//! Inline `#[cfg(test)]` tests on the dispatch primitive cannot prove the
//! production TaskTracker is the one actually being joined — they only
//! verify the primitive's own spawn shape. This integration test drives
//! a slow-adapter case through the dispatcher's TaskTracker so the
//! `tracker.close()` → `tracker.wait().await` sequence the production
//! shutdown path performs is exercised end-to-end against a real
//! `SessionStore` / `LayerStore` write.
//!
//! ## Shape of each test
//!
//! Each test:
//!
//! 1. Wires a `SessionStore` / `LayerStore` against a [`SlowCheckpointStore`]
//!    wrapper that sleeps `SLOW_DELAY` inside the targeted method before
//!    delegating to an `InMemoryCheckpointStore` and incrementing a
//!    completion counter (the counter increments only AFTER the delegated
//!    call returns, so it captures actual completion rather than enqueue).
//! 2. Calls the lifecycle hook (`SessionStore::create`,
//!    `SessionStore::delete`, or `LayerStore::create_layer`) which enqueues
//!    a fire-and-forget write into the shared `TaskTracker`.
//! 3. WITHOUT polling for completion, calls `tracker.close()` followed by
//!    `tracker.wait().await` — this is the exact sequence the production
//!    shutdown path executes (SDD §5.6).
//! 4. Asserts the completion counter equals 1 — proving the wait actually
//!    drained the in-flight task rather than abandoning it.
//! 5. Asserts the relevant `live_edit_checkpoint_writes_total` cell ticked
//!    by exactly 1 — the metric increments only on the real outcome
//!    (Ok/Err) at the bottom of the spawned task, so a tick is independent
//!    proof that the spawned task ran to completion through the dispatcher.
//!
//! ## Why three test cases
//!
//! Each `enqueue_*` call site lives on a different `*Store`:
//!
//! - `SessionStore::create` → `enqueue_session_checkpoint_write`.
//! - `LayerStore::create_layer` → `enqueue_session_checkpoint_write`
//!   (lifecycle-distinct call site; a regression that drops the writer
//!   from `LayerStore`'s wire-up surfaces here, not in the SessionStore
//!   tests).
//! - `SessionStore::delete` → `enqueue_session_checkpoint_delete`.
//!
//! All three share one dispatcher TaskTracker, so the load-bearing
//! shutdown-join invariant is a per-call-site property: each writer hand-off
//! must spawn into the shared tracker so the single `tracker.close()`+
//! `wait()` sequence drains every in-flight task.
//!
//! Sandbox env note: these tests use `TestValkey::setup`, which reads
//! `TEST_VALKEY_URL`. Under the project sandbox the URL must include the
//! password and `host.docker.internal` host; see CLAUDE.md.

#![cfg(feature = "test-checkpoint-fault-injection")]

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
use live_edit::domain::types::ObjectState;
use live_edit::observability::metrics::SharedMetrics;
use live_edit::storage::external_values::MockExternalValueStore;
use live_edit::storage::layer::LayerStore;
use live_edit::storage::session::SessionStore;
use live_edit::storage::test_support::TestValkey;

const SERVICE_LABEL: &str = "lifecycle-shutdown-join-test-replica";

/// Synthetic delay the slow adapter inserts inside its session-checkpoint
/// put/delete methods before delegating. Chosen to be:
///
/// - long enough that a `tracker.close()` issued IMMEDIATELY after the
///   lifecycle hook returns is guaranteed to land while the spawned task
///   is still suspended in the sleep (so a `tracker.wait()` that did NOT
///   drain in-flight tasks would observe a zero completion counter), AND
/// - short enough to keep each test well under the 1 s per-case runtime
///   budget called out in the Subtask body's acceptance criteria.
const SLOW_DELAY: Duration = Duration::from_millis(200);

/// Bound on how long `tracker.wait()` is allowed to take. A hang here points
/// at a `SessionCheckpointWriter` bug (e.g. the spawn never landed in this
/// tracker), not a test-only flake.
const WAIT_TIMEOUT: Duration = Duration::from_secs(10);

// =============================================================================
// Slow session-checkpoint store wrapper.
// =============================================================================

/// Wraps an [`InMemoryCheckpointStore`] and inserts `SLOW_DELAY` inside
/// `put_session_checkpoint` and `delete_session_checkpoint` BEFORE delegating
/// to the inner adapter. Increments `put_completed_calls` /
/// `delete_completed_calls` AFTER the delegated call returns so the counters
/// capture *completion*, not enqueue.
///
/// The counter-after-delegate ordering is load-bearing: a regression that
/// abandoned in-flight writes at `tracker.close()` would leave the slow
/// adapter's sleep cancelled mid-`tokio::time::sleep`, the increment line
/// would never execute, and the assertion below would observe a zero
/// counter even though the spawn was issued. That is exactly the orphan-
/// task signal we want this test to catch.
///
/// Every other trait method delegates verbatim — only the session-write and
/// session-delete paths are slowed. Layer-checkpoint paths run at the inner
/// adapter's native speed because none of the lifecycle hooks under test
/// here go through them.
struct SlowCheckpointStore {
    inner: Arc<InMemoryCheckpointStore>,
    put_completed_calls: AtomicUsize,
    delete_completed_calls: AtomicUsize,
}

impl SlowCheckpointStore {
    fn new() -> Self {
        Self {
            inner: Arc::new(InMemoryCheckpointStore::new()),
            put_completed_calls: AtomicUsize::new(0),
            delete_completed_calls: AtomicUsize::new(0),
        }
    }

    fn put_completed(&self) -> usize {
        self.put_completed_calls.load(Ordering::Acquire)
    }

    fn delete_completed(&self) -> usize {
        self.delete_completed_calls.load(Ordering::Acquire)
    }
}

#[async_trait]
impl CheckpointStore for SlowCheckpointStore {
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
    // inner store holds rather than merely leaving the call un-instrumented.
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
        // Sleep BEFORE delegation so the in-flight task is observably
        // suspended at the moment `tracker.close()` is called. If shutdown
        // were to abandon the in-flight task, this sleep would be cancelled
        // and the increment below would never run.
        tokio::time::sleep(SLOW_DELAY).await;
        let result = self
            .inner
            .put_session_checkpoint(session_id, body, meta)
            .await;
        // Increment AFTER delegation so the counter only reflects writes
        // that actually completed end-to-end through the inner adapter.
        self.put_completed_calls.fetch_add(1, Ordering::AcqRel);
        result
    }

    async fn get_session_checkpoint(
        &self,
        session_id: &str,
    ) -> Result<Option<CheckpointBlob>, CheckpointError> {
        self.inner.get_session_checkpoint(session_id).await
    }

    async fn delete_session_checkpoint(&self, session_id: &str) -> Result<(), CheckpointError> {
        // Same delay-before-delegate / increment-after-delegate ordering as
        // `put_session_checkpoint` above. The delete-path assertion below
        // observes this counter, so any regression that abandons the
        // in-flight delete at shutdown surfaces here.
        tokio::time::sleep(SLOW_DELAY).await;
        let result = self.inner.delete_session_checkpoint(session_id).await;
        self.delete_completed_calls.fetch_add(1, Ordering::AcqRel);
        result
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
}

// =============================================================================
// Test cases.
// =============================================================================

/// `SessionStore::create` enqueues a `put_session_checkpoint` into the
/// dispatcher's `TaskTracker`. Immediately after the lifecycle hook returns
/// (without polling for completion), the test issues
/// `tracker.close()` + `tracker.wait().await` — the exact sequence the
/// production shutdown path runs (SDD §5.6).
///
/// Both assertions are load-bearing:
///
/// - `put_completed == 1` proves the wait actually drained the in-flight
///   task. A shutdown that abandoned the task would leave the slow
///   adapter's sleep cancelled and the completion counter at zero.
/// - `result="success", kind="session"` cell == 1 proves the spawned task
///   reached its post-`put_session_checkpoint` arm and called
///   `handle_put_outcome(Ok(()))`. Because the metric tick happens
///   AFTER the inner adapter returns, it is independent corroboration
///   that the wait drained completion, not enqueue.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn shutdown_joins_in_flight_session_checkpoint_writes() {
    let tv = TestValkey::setup().await;

    let metrics: SharedMetrics = Arc::new(crate::common::metrics_scrape_only(SERVICE_LABEL));

    let slow = Arc::new(SlowCheckpointStore::new());
    let store: Arc<dyn CheckpointStore> = Arc::clone(&slow) as Arc<dyn CheckpointStore>;

    let tracker = TaskTracker::new();
    let writer = Arc::new(SessionCheckpointWriter::new(
        Arc::clone(&store),
        tv.client().clone(),
        tv.key_schema().clone(),
        tracker.clone(),
        Arc::clone(&metrics),
        SERVICE_LABEL.to_string(),
        true,
    ));
    let session_store = SessionStore::new(tv.client().clone(), tv.key_schema().clone())
        .with_checkpoint_writer(Arc::clone(&writer));

    let session_id = session_store
        .create(
            "p9-sy-shutdown-join-put-session",
            "/p9-sy/shutdown-join-put",
            "p9-sy-test-user",
        )
        .await
        .expect("SessionStore::create must succeed against a healthy Valkey");
    assert!(
        !session_id.is_empty(),
        "session_id must be a non-empty UUID4 — empty value indicates an unexpected early-return path"
    );

    // Close + wait IMMEDIATELY — do NOT poll for completion. The slow
    // adapter is still suspended in its `SLOW_DELAY` sleep at this point;
    // a shutdown that did not drain the tracker would see
    // `put_completed == 0` below.
    tracker.close();
    tokio::time::timeout(WAIT_TIMEOUT, tracker.wait())
        .await
        .expect(
            "tracker.wait() must drain inside WAIT_TIMEOUT — a hang here points at a \
             SessionCheckpointWriter bug (e.g. the spawn did not land in this tracker), \
             not a test-only flake",
        );

    let put_completed = slow.put_completed();
    assert_eq!(
        put_completed, 1,
        "put_completed_calls must equal 1 after tracker.wait() resolves — \
         the wait MUST drain the in-flight session-checkpoint write rather than \
         abandon it. Got put_completed={put_completed}. A zero value indicates \
         the shutdown sequence skipped the in-flight task, violating SDD §5.6's \
         bounded-shutdown invariant."
    );

    let success_cell = live_edit::observability::metrics::test_util::sample_value(
        &live_edit::observability::metrics::test_util::render(&metrics),
        "live_edit_checkpoint_writes_total",
        &[("result", "success"), ("kind", "session")],
    )
    .unwrap_or(0.0);
    assert_eq!(
        success_cell, 1.0,
        "live_edit_checkpoint_writes_total{{kind=\"session\", result=\"success\"}} \
         MUST increment by exactly 1 after the wait resolves; got {success_cell}. \
         The metric tick happens AFTER the adapter's put returns, so a non-1 value \
         here is independent corroboration that the tracker.wait() did not drain \
         the in-flight task to completion."
    );

    let error_cell = live_edit::observability::metrics::test_util::sample_value(
        &live_edit::observability::metrics::test_util::render(&metrics),
        "live_edit_checkpoint_writes_total",
        &[("result", "error"), ("kind", "session")],
    )
    .unwrap_or(0.0);
    assert_eq!(
        error_cell, 0.0,
        "live_edit_checkpoint_writes_total{{kind=\"session\", result=\"error\"}} \
         MUST stay at 0 for a successful write; got {error_cell}"
    );
}

/// Symmetric test for the delete dispatch path:
/// `SessionStore::delete` enqueues a `delete_session_checkpoint` into the
/// dispatcher's `TaskTracker`; shutdown's `tracker.close()` +
/// `tracker.wait().await` must drain it before returning.
///
/// Drive shape:
///
/// 1. Phase 1: create the session (which enqueues a put), drain that
///    put-tracker so the delete-tracker assertions are not polluted.
/// 2. Phase 2: build a fresh `TaskTracker` + `SessionCheckpointWriter`
///    sharing the SAME `SlowCheckpointStore`, then `SessionStore::delete`
///    → immediate `tracker.close()` + `wait()` and assert the delete
///    counter ticked.
///
/// Two trackers are required because `TaskTracker::close()` is terminal —
/// once closed, future `tracker.spawn(...)` calls drop the task on the
/// floor (mirrors the `lifecycle_delete_session_test` two-phase pattern in
/// this directory). A single `SessionStore` cannot be re-wired with a new
/// writer mid-lifetime (`with_checkpoint_writer` is consume-self), so each
/// phase uses its own `SessionStore` over the same `TestValkey` — the
/// Valkey state is the shared lifecycle surface; only the dispatch-side
/// writer needs to be per-phase.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn shutdown_joins_in_flight_session_checkpoint_delete() {
    let tv = TestValkey::setup().await;

    let metrics: SharedMetrics = Arc::new(crate::common::metrics_scrape_only(SERVICE_LABEL));

    let slow = Arc::new(SlowCheckpointStore::new());
    let store: Arc<dyn CheckpointStore> = Arc::clone(&slow) as Arc<dyn CheckpointStore>;

    // -------------------------------------------------------------------------
    // Phase 1 — create the session, drain its put-enqueue.
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
            "p9-sy-shutdown-join-delete-session",
            "/p9-sy/shutdown-join-delete",
            "p9-sy-test-user",
        )
        .await
        .expect("SessionStore::create must succeed against a healthy Valkey");

    create_tracker.close();
    tokio::time::timeout(WAIT_TIMEOUT, create_tracker.wait())
        .await
        .expect(
            "create_tracker.wait() must drain inside WAIT_TIMEOUT — a hang here points \
             at a SessionCheckpointWriter bug, not a test-only flake",
        );
    assert_eq!(
        slow.put_completed(),
        1,
        "Phase-1 baseline: create enqueued one put; got put_completed={}",
        slow.put_completed()
    );

    // -------------------------------------------------------------------------
    // Phase 2 — delete the session, immediately close+wait the
    //           delete-tracker, assert the delete drained.
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

    // Close + wait IMMEDIATELY — same shape as the put-side test above.
    delete_tracker.close();
    tokio::time::timeout(WAIT_TIMEOUT, delete_tracker.wait())
        .await
        .expect(
            "delete_tracker.wait() must drain inside WAIT_TIMEOUT — a hang here points at \
             a SessionCheckpointWriter bug (e.g. the spawn did not land in this tracker), \
             not a test-only flake",
        );

    let delete_completed = slow.delete_completed();
    assert_eq!(
        delete_completed, 1,
        "delete_completed_calls must equal 1 after delete_tracker.wait() resolves — \
         the wait MUST drain the in-flight delete rather than abandon it. \
         Got delete_completed={delete_completed}. A zero value indicates \
         the shutdown sequence skipped the in-flight task, violating SDD §5.6's \
         bounded-shutdown invariant."
    );

    // The success cell now reflects BOTH the Phase-1 put AND the Phase-2
    // delete (both successful — the delete dispatches through the same
    // `handle_put_outcome` Ok-arm and ticks the same cell). A tick of 2
    // here is independent corroboration that the delete-side adapter call
    // completed end-to-end before the wait returned.
    let success_cell = live_edit::observability::metrics::test_util::sample_value(
        &live_edit::observability::metrics::test_util::render(&metrics),
        "live_edit_checkpoint_writes_total",
        &[("result", "success"), ("kind", "session")],
    )
    .unwrap_or(0.0);
    assert_eq!(
        success_cell, 2.0,
        "live_edit_checkpoint_writes_total{{kind=\"session\", result=\"success\"}} \
         MUST equal 2 after both create + delete drain (one tick per dispatch); \
         got {success_cell}"
    );
}

/// LayerStore-hosted side of the writer: `LayerStore::create_layer` enqueues a
/// `put_session_checkpoint` into the same dispatcher `TaskTracker`. This test
/// pins the second `enqueue_session_checkpoint_write` call site lives on the
/// same shared tracker — a regression that wired `LayerStore`'s writer to a
/// different tracker would surface here as a zero completion counter even
/// though the SessionStore tests above pass.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn shutdown_joins_in_flight_layer_store_session_checkpoint_writes() {
    let tv = TestValkey::setup().await;

    let metrics: SharedMetrics = Arc::new(crate::common::metrics_scrape_only(SERVICE_LABEL));

    let slow = Arc::new(SlowCheckpointStore::new());
    let store: Arc<dyn CheckpointStore> = Arc::clone(&slow) as Arc<dyn CheckpointStore>;

    // -------------------------------------------------------------------------
    // Phase 1 — seed a session so the layer has somewhere to live, drain
    //           the session-create put before observing the layer-create
    //           put under the second tracker.
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
    let session_store = SessionStore::new(tv.client().clone(), tv.key_schema().clone())
        .with_checkpoint_writer(Arc::clone(&create_writer));
    let session_id = session_store
        .create(
            "p9-sy-shutdown-join-layer-session",
            "/p9-sy/shutdown-join-layer",
            "p9-sy-test-user",
        )
        .await
        .expect("SessionStore::create must succeed against a healthy Valkey");
    create_tracker.close();
    tokio::time::timeout(WAIT_TIMEOUT, create_tracker.wait())
        .await
        .expect(
            "create_tracker.wait() must drain inside WAIT_TIMEOUT — a hang here points at \
             a SessionCheckpointWriter bug, not a test-only flake",
        );
    assert_eq!(
        slow.put_completed(),
        1,
        "Phase-1 baseline: session create enqueued one put; got put_completed={}",
        slow.put_completed()
    );

    // -------------------------------------------------------------------------
    // Phase 2 — wire a fresh tracker + writer, attach it to a `LayerStore`,
    //           call `create_layer` → immediate `close()`+`wait()`, assert
    //           the in-flight write drained.
    // -------------------------------------------------------------------------
    let layer_tracker = TaskTracker::new();
    let layer_writer = Arc::new(SessionCheckpointWriter::new(
        Arc::clone(&store),
        tv.client().clone(),
        tv.key_schema().clone(),
        layer_tracker.clone(),
        Arc::clone(&metrics),
        SERVICE_LABEL.to_string(),
        true,
    ));
    let layer_store = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096)
        .with_checkpoint_writer(Arc::clone(&layer_writer));

    let initial = ObjectState {
        sequence_num: 1,
        first_version: 1,
        source_format: "usd".to_string(),
        ..Default::default()
    };
    let _oid = layer_store
        .create_layer(
            &session_id,
            "p9-sy-shutdown-join-layer",
            "p9-sy-test-user",
            &initial,
            Arc::new(MockExternalValueStore::new()),
        )
        .await
        .expect("LayerStore::create_layer must succeed against a healthy Valkey");

    // Close + wait IMMEDIATELY — same shape as the SessionStore tests.
    layer_tracker.close();
    tokio::time::timeout(WAIT_TIMEOUT, layer_tracker.wait())
        .await
        .expect(
            "layer_tracker.wait() must drain inside WAIT_TIMEOUT — a hang here points at a \
             SessionCheckpointWriter bug (e.g. LayerStore's writer wired to a different \
             tracker than this one), not a test-only flake",
        );

    let put_completed = slow.put_completed();
    assert_eq!(
        put_completed, 2,
        "put_completed_calls must equal 2 after layer_tracker.wait() resolves \
         (one tick from Phase-1 session-create + one tick from Phase-2 layer-create); \
         got put_completed={put_completed}. A value of 1 indicates the LayerStore-side \
         writer spawned into a different tracker than the one shutdown joined."
    );

    let success_cell = live_edit::observability::metrics::test_util::sample_value(
        &live_edit::observability::metrics::test_util::render(&metrics),
        "live_edit_checkpoint_writes_total",
        &[("result", "success"), ("kind", "session")],
    )
    .unwrap_or(0.0);
    assert_eq!(
        success_cell, 2.0,
        "live_edit_checkpoint_writes_total{{kind=\"session\", result=\"success\"}} \
         MUST equal 2 after both session-create + layer-create drain; got {success_cell}"
    );
}
