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

//! Subtask `t3.fgj.7v.p9.2c` — Disabled-adapter silent-swallow contract
//! against the production wire-up.
//!
//! The Task's Success Criteria pin that "Disabled adapter is a no-op — the
//! hot path with checkpoints off matches pre-feature behavior." Inline
//! `#[cfg(test)]` tests in `src/checkpoint/session_writer.rs` already cover
//! the dispatcher primitive's silent-on-`NotImplemented` branch
//! (`handle_put_outcome_silent_on_not_implemented`), but they cannot verify
//! that the production hook sites (`SessionStore::create`,
//! `LayerStore::create_layer`, `LayerStore::delete_layer`,
//! `SessionStore::delete`) skip the OUTCOME-metric tick when wired against
//! the production `NoopCheckpointStore` shim that
//! `build_checkpoint_store(Disabled)` hands out.
//!
//! Pinned end-to-end shape, as of b.443:
//!
//! - The dispatcher is constructed with `enabled = !matches!(adapter,
//!   Disabled)`, exactly as `main.rs` computes it. Under `Disabled` that is
//!   `false`, and **both enqueue entry points return before doing anything**:
//!   no session-state read against Valkey, no build stamp minted on
//!   `le:{sid}:ckpt`, no adapter call, no slot bookkeeping.
//! - Consequently: ZERO adapter calls, no `le:{sid}:ckpt` key, and no
//!   `live_edit_checkpoint_writes_total` tick on ANY `result` label.
//!
//! **This inverts the pre-b.443 shape, deliberately.** The test used to
//! assert the adapter trait WAS reached (3 puts + 1 delete) and that the
//! dispatcher then swallowed the resulting `NotImplemented`. That was a
//! faithful description of the old code and a bad deal: a `Disabled`
//! deployment paid a full multi-round-trip session read on every lifecycle
//! event purely to throw the payload away one call later, and post-b.443 the
//! read also mints a stamp, so the discarded work would now write to Valkey
//! too. The Success Criterion has always been "Disabled adapter is a no-op —
//! the hot path with checkpoints off matches pre-feature behavior"; only now
//! is that literally true, and this file asserts the literal version.
//!
//! The dispatcher's silent-on-`NotImplemented` rule is still a live
//! invariant — a non-`Disabled` adapter may return `NotImplemented` from an
//! unimplemented method — and is covered by
//! `handle_put_outcome_silent_on_not_implemented` in
//! `src/checkpoint/session_writer.rs`. What is no longer reachable is
//! `Disabled` as its *producer*, which is the point of the gate.
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

use fred::interfaces::KeysInterface;
use live_edit::checkpoint::{
    CheckpointAdapter, CheckpointBlob, CheckpointError, CheckpointHandle, CheckpointMeta,
    CheckpointStore, SessionCheckpointWriter,
};
use live_edit::config::{Config, PodMemorySource};
use live_edit::domain::types::ObjectState;
use live_edit::observability::metrics::SharedMetrics;
use live_edit::storage::external_values::MockExternalValueStore;
use live_edit::storage::layer::LayerStore;
use live_edit::storage::session::SessionStore;
use live_edit::storage::test_support::TestValkey;

const SERVICE_LABEL: &str = "lifecycle-disabled-adapter-test-replica";

// =============================================================================
// Counting wrapper over the production `NoopCheckpointStore`.
// =============================================================================

/// Wraps the production `Disabled`-adapter shim handed out by
/// `build_checkpoint_store(Disabled)` and counts every method invocation on
/// the session-checkpoint surface (`put_session_checkpoint`,
/// `delete_session_checkpoint`, `get_session_checkpoint`).
///
/// Counter increments happen BEFORE the delegated call so a concurrent caller
/// cannot mask the counter from observers; the assertion below snapshots the
/// values AFTER `tracker.wait()` returns, by which point every spawned task
/// has either ticked here or short-circuited silently. As of b.443 the
/// expected reading is zero on every counter — the dispatcher's `enabled`
/// gate means no task is spawned at all — so these counters have become an
/// "it stayed silent" witness rather than a call-shape assertion.
///
/// We construct the inner `Arc<dyn CheckpointStore>` via
/// `build_checkpoint_store(Disabled)` rather than naming `NoopCheckpointStore`
/// directly — the production shim is private to `src/checkpoint/builder.rs`
/// (see the rationale comment there: "intentionally NOT a public adapter").
/// Going through the factory means this test exercises the same trait-object
/// surface a production caller hits, and a future refactor that switches the
/// Disabled arm to a different no-op type surfaces here as a behaviour
/// change (one of the assertions below would fail) rather than a silent
/// drift.
///
/// Composes `super::helpers::DelegatingCheckpointStore<dyn CheckpointStore>`
/// for the verbatim 14-method delegation and overrides only the three methods
/// that actually count. The `?Sized` bound on `DelegatingCheckpointStore<T>`
/// lets us thread the trait-object `dyn CheckpointStore` through the same
/// delegation skeleton the sized-inner-store call sites use.
struct CountingNoopCheckpointStore {
    delegate: super::helpers::DelegatingCheckpointStore<dyn CheckpointStore>,
    put_session_checkpoint_calls: AtomicUsize,
    delete_session_checkpoint_calls: AtomicUsize,
    get_session_checkpoint_calls: AtomicUsize,
}

impl CountingNoopCheckpointStore {
    fn new(inner: Arc<dyn CheckpointStore>) -> Self {
        Self {
            delegate: super::helpers::DelegatingCheckpointStore::new(inner),
            put_session_checkpoint_calls: AtomicUsize::new(0),
            delete_session_checkpoint_calls: AtomicUsize::new(0),
            get_session_checkpoint_calls: AtomicUsize::new(0),
        }
    }

    fn put_count(&self) -> usize {
        self.put_session_checkpoint_calls.load(Ordering::Acquire)
    }

    fn delete_count(&self) -> usize {
        self.delete_session_checkpoint_calls.load(Ordering::Acquire)
    }

    fn get_count(&self) -> usize {
        self.get_session_checkpoint_calls.load(Ordering::Acquire)
    }
}

#[async_trait]
impl CheckpointStore for CountingNoopCheckpointStore {
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
        // Increment BEFORE delegating so a concurrent observer that races
        // the dispatcher's spawned task cannot read a counter that has not
        // yet captured this call. The Disabled shim's `put_session_checkpoint`
        // returns `Err(NotImplemented { .. })` unconditionally; the
        // dispatcher's `handle_put_outcome` recognises that classifier and
        // silently swallows, so the OUTCOME-metric assertion below MUST
        // hold zero on every `result` label.
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
        self.get_session_checkpoint_calls
            .fetch_add(1, Ordering::AcqRel);
        self.delegate.get_session_checkpoint(session_id).await
    }

    async fn delete_session_checkpoint(&self, session_id: &str) -> Result<(), CheckpointError> {
        // Same ordering rationale as `put_session_checkpoint` above.
        // `SessionStore::delete` is the one lifecycle event that routes
        // through this method (the other three route through the put
        // branch); the assertion below pins exactly one tick after the
        // full lifecycle sequence drains.
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

/// End-to-end Disabled-adapter contract: drive the full lifecycle sequence
/// (CreateSession → CreateLayer → DeleteLayer → DeleteSession) against the
/// production `NoopCheckpointStore` shim and pin
///
/// 1. The adapter trait is NEVER reached — every counter on the wrapping
///    `CountingNoopCheckpointStore` stays at zero across all four lifecycle
///    events, because `enqueue_session_checkpoint_{write,delete}` return
///    before building anything (b.443).
///
/// 2. No `le:{sid}:ckpt` key is created. The build-stamp mint lives inside
///    the session read, so this is what proves the gate sits at the enqueue
///    rather than somewhere after the build.
///
/// 3. NO tick on `live_edit_checkpoint_writes_total{kind="session"}` for ANY
///    `result` label (`success`, `error`, `skipped`) — the load-bearing
///    operator signal that distinguishes "operator turned checkpointing off"
///    from "adapter broken, page someone."
///
/// The wrapping counting store is still built around the production
/// `build_checkpoint_store(Disabled)` shim rather than a bare stub: the
/// zero-call assertion is only meaningful if the store it counts is the one
/// a Disabled deployment actually gets.
///
/// Tracker isolation note (mirrors the pattern in
/// `lifecycle_delete_session_test.rs`): each phase gets its own
/// `TaskTracker` so we can `close()+wait()` at the end of the phase without
/// poisoning the next phase's enqueue path. A `TaskTracker::wait()` future
/// resolves only once the tracker is **closed** AND every spawned task has
/// drained; `close()` is terminal, so re-using one tracker across phases
/// would drop later phases' tasks on the floor.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn disabled_adapter_is_a_full_no_op_on_all_lifecycle_events() {
    let tv = TestValkey::setup().await;

    let metrics: SharedMetrics = Arc::new(crate::common::metrics_scrape_only(SERVICE_LABEL));

    // Build the production Disabled-adapter shim via the factory. Going
    // through `build_checkpoint_store` (rather than naming the private
    // `NoopCheckpointStore` type directly, which is intentionally
    // crate-private per `src/checkpoint/builder.rs`) means this test
    // exercises the same trait-object surface a production caller hits.
    let cfg = build_disabled_config();
    let inner_disabled_store = live_edit::checkpoint::build_checkpoint_store(&cfg, None, None)
        .await
        .expect("build_checkpoint_store(Disabled) must succeed — it's just an Arc allocation");

    // Derive the dispatcher's `enabled` flag with the SAME expression
    // `main.rs` uses, from the SAME `Config` that produced the store above.
    // Hard-coding `false` here would let the test keep passing if production
    // ever stopped gating on `Disabled`; deriving it means a change to the
    // gate's shape shows up as a compile error or as this assertion firing.
    let dispatch_enabled = !matches!(cfg.checkpoint_adapter, CheckpointAdapter::Disabled);
    assert!(
        !dispatch_enabled,
        "LIVE_EDIT_CHECKPOINT_ADAPTER=Disabled MUST yield a disabled dispatcher — if this fires, \
         `build_disabled_config` no longer produces a Disabled adapter and the rest of this test \
         is asserting the wrong deployment shape"
    );

    // Wrap the production shim in the counting wrapper. The counting
    // wrapper holds the inner store through `Arc<dyn CheckpointStore>` so
    // every method on `CountingNoopCheckpointStore` delegates to the
    // production shim's `NotImplemented`-returning implementation.
    let counting = Arc::new(CountingNoopCheckpointStore::new(inner_disabled_store));
    let store: Arc<dyn CheckpointStore> = Arc::clone(&counting) as Arc<dyn CheckpointStore>;

    // -------------------------------------------------------------------------
    // Phase 1 — CreateSession. Enqueues one put_session_checkpoint call.
    // -------------------------------------------------------------------------
    let create_session_tracker = TaskTracker::new();
    let create_session_writer = Arc::new(SessionCheckpointWriter::new(
        Arc::clone(&store),
        tv.client().clone(),
        tv.key_schema().clone(),
        create_session_tracker.clone(),
        Arc::clone(&metrics),
        SERVICE_LABEL.to_string(),
        dispatch_enabled,
    ));
    let create_session_store = SessionStore::new(tv.client().clone(), tv.key_schema().clone())
        .with_checkpoint_writer(Arc::clone(&create_session_writer));

    let session_id = create_session_store
        .create(
            "p9-2c-disabled-adapter-session",
            "/p9-2c/disabled-adapter",
            "p9-2c-test-user",
        )
        .await
        .expect("SessionStore::create must succeed against a healthy Valkey — the off-hot-path checkpoint dispatch failure is swallowed");
    assert!(
        !session_id.is_empty(),
        "session_id must be a non-empty UUID4 — empty value indicates an unexpected early-return path"
    );
    create_session_tracker.close();
    tokio::time::timeout(Duration::from_secs(10), create_session_tracker.wait())
        .await
        .expect(
            "create_session_tracker.wait() must drain inside 10s — a hang here points at a \
             SessionCheckpointWriter bug, not a test-only flake",
        );

    // -------------------------------------------------------------------------
    // Phase 2 — CreateLayer. Enqueues one put_session_checkpoint call.
    // -------------------------------------------------------------------------
    let create_layer_tracker = TaskTracker::new();
    let create_layer_writer = Arc::new(SessionCheckpointWriter::new(
        Arc::clone(&store),
        tv.client().clone(),
        tv.key_schema().clone(),
        create_layer_tracker.clone(),
        Arc::clone(&metrics),
        SERVICE_LABEL.to_string(),
        dispatch_enabled,
    ));
    let create_layer_store = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096)
        .with_checkpoint_writer(Arc::clone(&create_layer_writer));

    let initial = ObjectState {
        sequence_num: 1,
        first_version: 1,
        source_format: "usd".to_string(),
        ..Default::default()
    };
    let _oid = create_layer_store
        .create_layer(
            &session_id,
            "p9-2c-disabled-adapter-layer",
            "p9-2c-test-user",
            &initial,
            Arc::new(MockExternalValueStore::new()),
        )
        .await
        .expect("LayerStore::create_layer must succeed against a healthy Valkey");
    create_layer_tracker.close();
    tokio::time::timeout(Duration::from_secs(10), create_layer_tracker.wait())
        .await
        .expect(
            "create_layer_tracker.wait() must drain inside 10s — a hang here points at a \
             SessionCheckpointWriter bug, not a test-only flake",
        );

    // -------------------------------------------------------------------------
    // Phase 3 — DeleteLayer. Enqueues one put_session_checkpoint call
    //           (NOT a delete — per SR-2.7 / Task t2.fgj.7v.p9, DeleteLayer
    //           removes one layer from a session that still exists, so the
    //           session-checkpoint is rewritten to reflect the smaller
    //           layer set rather than tombstoned).
    // -------------------------------------------------------------------------
    let delete_layer_tracker = TaskTracker::new();
    let delete_layer_writer = Arc::new(SessionCheckpointWriter::new(
        Arc::clone(&store),
        tv.client().clone(),
        tv.key_schema().clone(),
        delete_layer_tracker.clone(),
        Arc::clone(&metrics),
        SERVICE_LABEL.to_string(),
        dispatch_enabled,
    ));
    let delete_layer_store = LayerStore::new(tv.client().clone(), tv.key_schema().clone(), 4096)
        .with_checkpoint_writer(Arc::clone(&delete_layer_writer));

    delete_layer_store
        .delete_layer(
            &session_id,
            "p9-2c-disabled-adapter-layer",
            Arc::new(MockExternalValueStore::new()),
        )
        .await
        .expect("LayerStore::delete_layer must succeed for a freshly-created layer");
    delete_layer_tracker.close();
    tokio::time::timeout(Duration::from_secs(10), delete_layer_tracker.wait())
        .await
        .expect(
            "delete_layer_tracker.wait() must drain inside 10s — a hang here points at a \
             SessionCheckpointWriter bug, not a test-only flake",
        );

    // -------------------------------------------------------------------------
    // Phase 4 — DeleteSession. Enqueues one delete_session_checkpoint call
    //           (the one asymmetric hook — see lifecycle_delete_session_test
    //           for the put-vs-delete branch-routing assertion).
    // -------------------------------------------------------------------------
    let delete_session_tracker = TaskTracker::new();
    let delete_session_writer = Arc::new(SessionCheckpointWriter::new(
        Arc::clone(&store),
        tv.client().clone(),
        tv.key_schema().clone(),
        delete_session_tracker.clone(),
        Arc::clone(&metrics),
        SERVICE_LABEL.to_string(),
        dispatch_enabled,
    ));
    let delete_session_store = SessionStore::new(tv.client().clone(), tv.key_schema().clone())
        .with_checkpoint_writer(Arc::clone(&delete_session_writer));

    delete_session_store
        .delete(&session_id)
        .await
        .expect("SessionStore::delete must succeed for a freshly-created session");
    delete_session_tracker.close();
    tokio::time::timeout(Duration::from_secs(10), delete_session_tracker.wait())
        .await
        .expect(
            "delete_session_tracker.wait() must drain inside 10s — a hang here points at a \
             SessionCheckpointWriter bug, not a test-only flake",
        );

    // -------------------------------------------------------------------------
    // Assertion 1 — the adapter is NEVER reached (b.443).
    //
    // The enqueue gate short-circuits ahead of the build, so no lifecycle
    // event produces an adapter call at all. Pre-b.443 these read 3 / 1 / 0:
    // the dispatcher called through to the Disabled shim and threw the
    // `NotImplemented` away one frame later, having already paid for the
    // session read that produced the payload.
    //
    // `put_count` is the one that would move first under a regression that
    // dropped the gate — CreateSession, CreateLayer and DeleteLayer all
    // route through `enqueue_session_checkpoint_write`.
    // -------------------------------------------------------------------------
    let put_count = counting.put_count();
    let delete_count = counting.delete_count();
    let get_count = counting.get_count();
    assert_eq!(
        put_count, 0,
        "put_session_checkpoint MUST NOT be called at all under a Disabled adapter — the \
         dispatcher's `enabled` gate returns before the build (b.443); got \
         put_count={put_count}. A count of 3 is the pre-b.443 behaviour: every lifecycle event \
         paid for a full session read whose payload was discarded at the adapter."
    );
    assert_eq!(
        delete_count, 0,
        "delete_session_checkpoint MUST NOT be called at all under a Disabled adapter; got \
         delete_count={delete_count}"
    );
    // get_session_checkpoint is the recovery-hook surface, not the
    // lifecycle-write surface. None of the four lifecycle methods drive
    // recovery, so this counter stays at zero for a second, independent
    // reason — pinning it here catches a future regression that mistakenly
    // wires a recovery probe into a write-path hook site.
    assert_eq!(
        get_count, 0,
        "get_session_checkpoint MUST NOT be called by any of the four lifecycle write \
         methods — it's the recovery-hook surface, not the write surface; got \
         get_count={get_count}. A non-zero value indicates a write-path hook site is \
         mistakenly probing the checkpoint store before writing."
    );

    // -------------------------------------------------------------------------
    // Assertion 2 — no session-checkpoint coordination key was created.
    //
    // This is the assertion that distinguishes "the gate is in the right
    // place" from "the gate is somewhere after the build". Post-b.443 the
    // session read is also the mint of the build stamp — it `HINCRBY`s
    // `le:{sid}:ckpt` and applies a 24-hour TTL. A gate that sat between the
    // build and the adapter call would leave the counter assertions above at
    // zero while still writing one such key per lifecycle event, i.e. still
    // charging a Disabled deployment for Valkey traffic and resident memory
    // it can never use. Only a gate at the enqueue leaves no key.
    // -------------------------------------------------------------------------
    let ckpt_state_key = tv.key_schema().session_checkpoint_state(&session_id);
    let ckpt_state_exists: bool = tv
        .client()
        .pool
        .exists(&ckpt_state_key)
        .await
        .expect("EXISTS against the test Valkey must succeed");
    assert!(
        !ckpt_state_exists,
        "{ckpt_state_key} MUST NOT exist after a full lifecycle sequence under a Disabled \
         adapter — the build-stamp mint lives inside the session read, so a key here means the \
         dispatcher did the read anyway and the `enabled` gate is downstream of the build \
         instead of at the enqueue"
    );

    // -------------------------------------------------------------------------
    // Assertion 3 — outcome metric stays zero on EVERY `result` label.
    //
    // This is the load-bearing operator-signal contract, and it survives the
    // b.443 inversion unchanged: whether the dispatcher never calls the
    // adapter (now) or calls it and swallows the `NotImplemented` (before),
    // a Disabled deployment must never tick this counter. A regression that
    // ticked `result="error"` here would page operators on every
    // CreateSession when checkpointing is intentionally off.
    // -------------------------------------------------------------------------
    for result in ["success", "error", "skipped"] {
        let cell = live_edit::observability::metrics::test_util::sample_value(
            &live_edit::observability::metrics::test_util::render(&metrics),
            "live_edit_checkpoint_writes_total",
            &[("result", result), ("kind", "session")],
        )
        .unwrap_or(0.0);
        assert_eq!(
            cell, 0.0,
            "live_edit_checkpoint_writes_total{{kind=\"session\", result={result:?}}} \
             MUST read zero after the full Disabled-adapter lifecycle sequence — \
             the dispatcher's silent-on-NotImplemented rule means no metric tick on \
             ANY result label; got {cell}. A non-zero `result=\"error\"` cell would \
             indicate a regression that classified NotImplemented as a generic error \
             and would page operators whenever checkpointing is intentionally off."
        );
    }
}

/// Build a `Config` with `LIVE_EDIT_CHECKPOINT_ADAPTER=Disabled` so
/// `build_checkpoint_store` hands back the production `NoopCheckpointStore`
/// shim. Drives `Config::from_env()` under `temp_env::with_vars` so the
/// parse-side gates (e.g. `LIVE_EDIT_STORAGE_SERVICE_ENDPOINT` scheme
/// validation) are exercised exactly as `main.rs` would. The closure scope
/// ends before `TestValkey::setup` so the temp env mutation does not
/// interleave with TestValkey's read of `TEST_VALKEY_URL`.
fn build_disabled_config() -> Config {
    temp_env::with_vars(
        [
            (
                "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                Some("http://auth.example.com"),
            ),
            (
                "LIVE_EDIT_USERINFO_ENDPOINT",
                Some("http://userinfo.example.com"),
            ),
            ("LIVE_EDIT_CHECKPOINT_ADAPTER", Some("Disabled")),
        ],
        || {
            Config::from_env_with(PodMemorySource::Reference)
                .expect("Config::from_env with Disabled adapter must succeed")
        },
    )
}
