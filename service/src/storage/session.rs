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

// Session CRUD operations against Valkey.
//
// Implements SDD §7.10 (CreateSession), §7.13 (ListSessions), §7.8 (DeleteSession stub).
// Name validation follows PRD §3.1.1 and SDD §23.1.

use std::collections::HashMap;
use std::sync::Arc;

use fred::interfaces::{HashesInterface, KeysInterface, SetsInterface, TransactionInterface};

use crate::checkpoint::SessionCheckpointWriter;
use crate::domain::time::now_micros;
use crate::error::ServiceError;
use crate::observability::metrics::{
    WATCH_VOIDED_ABORTED_PRE_EXEC, WATCH_VOIDED_COMMITTED_UNWATCHED, WATCH_VOIDED_COMPENSATED,
    WATCH_VOIDED_FINGERPRINT_UNAVAILABLE, WATCH_VOIDED_PATH_CREATE_SESSION,
    WATCH_VOIDED_WITNESS_UNAVAILABLE,
};
use crate::storage::client::{
    map_fred_err_for_grpc, map_transient_fred_err, FredErrOrigin, ValkeyClient, POOL_LABEL_DATA,
    POOL_LABEL_TXN,
};
use crate::storage::keys::KeySchema;
use crate::storage::retry::{retry_aborted_with_jitter, RetryConfig};
use crate::storage::watch_guard::WatchGuard;
use crate::storage::watched_txn::{DispatchVerdict, WatchFingerprint, WatchedExec, WatchedTxn};

/// Metadata for a session stored in Valkey.
#[derive(Debug, Clone)]
pub struct SessionMeta {
    pub session_id: String,
    pub name: String,
    pub base_path: String,
    pub created_by: String,
    pub created_at: u64,
}

/// A path-index member whose `HGETALL le:{sid}:sess` returned empty during
/// [`SessionStore::list`] — i.e. the path-index lists the session_id but the
/// session metadata hash does not exist. Before session-recovery (SDD SR-2.7,
/// Epic `t1.fgj.7v`) this was unambiguously a "true orphan" and `list` itself
/// SREM'd the entry from both indices. With recovery in play, the absence is
/// instead the discriminator the caller uses to decide whether to invoke
/// `try_recover_session`: a checkpoint may exist and the member is a
/// recovery candidate, not a deletable ghost.
///
/// `list` surfaces these to the caller as `GhostMember` records; the caller
/// (today: [`crate::grpc::session::handle_list_sessions`]) orchestrates the
/// recovery+SREM loop. Keeping the SREM out of `storage::session` preserves
/// the architectural constraint that `storage` does not depend on
/// `checkpoint` for new behavior.
#[derive(Debug, Clone)]
pub struct GhostMember {
    /// Session name parsed from the `path_names_index` composite member.
    pub name: String,
    /// Session ID parsed from the `path_names_index` composite member.
    pub sid: String,
    /// Pre-derived `name\0session_id` composite — the exact member string to
    /// SREM from `path_names_index` when the caller decides this is a true
    /// orphan. Cached here so the caller does not need to re-derive via
    /// [`KeySchema::path_name_member`].
    pub composite_member: String,
}

/// Outcome of [`SessionStore::list`] under the post-recovery contract: the
/// `present` sessions whose `HGETALL` returned non-empty, plus the `ghosts`
/// whose `HGETALL` returned empty. The caller decides per-ghost whether to
/// invoke session recovery (and preserve the member on `Recovered`) or to
/// SREM the member as a true orphan (today's behaviour, preserved on the
/// `NoCheckpoint` / disabled-hook path).
#[derive(Debug, Clone)]
pub struct ListResult {
    pub present: Vec<SessionMeta>,
    pub ghosts: Vec<GhostMember>,
}

/// Test-only outcome to inject into [`SessionStore::delete`] at the EXEC
/// boundary (b.91j C-3 regression coverage). Lets an integration test drive
/// the two retriable arms (`Value::Null` and `is_canceled()`) without
/// actually severing the Valkey connection.
///
/// Gated behind `#[cfg(any(test, feature = "test-fault-injection"))]` so the
/// enum (and its consumer branch) is absent from release library builds.
#[cfg(any(test, feature = "test-fault-injection"))]
#[derive(Debug, Clone, Copy)]
pub enum DeleteExecFault {
    /// Substitute the `Ok(Value::Null)` arm — fred observed a null EXEC reply
    /// (connection drop pattern). Expected to produce
    /// `Aborted("transaction reply null (connection drop), retry_on_valkey_drop")`.
    ReturnNull,
    /// Substitute the `Err(e) if e.is_canceled()` arm — fred observed a
    /// canceled transaction (channel close / reconnect exhaustion). Expected
    /// to produce
    /// `Aborted("transaction canceled (connection drop), retry_on_valkey_drop")`.
    ReturnCanceled,
}

/// Storage layer for session CRUD against Valkey.
#[derive(Clone)]
pub struct SessionStore {
    client: ValkeyClient,
    keys: KeySchema,
    retry_max_attempts: u32,
    retry_initial_delay_ms: u64,
    /// Test-only fault-injection queue for `SessionStore::delete`, consumed
    /// at the top of `delete` BEFORE any Valkey round-trip. When present
    /// and non-empty, each call pops one entry; a popped entry substitutes
    /// the corresponding `exec_txn::<fred::types::Value>(...)` match arm
    /// without issuing the EXEC, letting the b.91j C-3 regression test
    /// drive the `Aborted` branches without severing the real connection.
    ///
    /// An empty queue (or no seam installed) lets `delete` run normally.
    ///
    /// **Test-only.** Production code never populates this — it is exposed
    /// only through the `with_delete_exec_fault_injector_for_test` builder
    /// and gated behind `#[cfg(any(test, feature = "test-fault-injection"))]`
    /// so release builds never carry the field, builder, or consumer.
    #[cfg(any(test, feature = "test-fault-injection"))]
    delete_exec_fault_injector_for_test:
        Option<std::sync::Arc<std::sync::Mutex<std::collections::VecDeque<DeleteExecFault>>>>,
    /// Test-only fault-injection queue for `SessionStore::create_once`,
    /// consumed at the top of each attempt BEFORE any Valkey round-trip.
    /// When present and non-empty, each attempt pops one entry; a popped
    /// `Some(err)` short-circuits that attempt with `err`, an empty queue
    /// lets the attempt run normally.
    ///
    /// Used by b.91j H-stor-3 coverage to drive `create`'s retry loop to
    /// exhaustion without a real WATCH-contention race, so the final
    /// `Aborted("create_session retriable failures, N attempts exhausted ...")`
    /// can be asserted deterministically at a caller-chosen `N`.
    ///
    /// **Test-only.** Production code never populates this.
    #[cfg(any(test, feature = "test-fault-injection"))]
    create_fault_injector_for_test:
        Option<std::sync::Arc<std::sync::Mutex<std::collections::VecDeque<ServiceError>>>>,
    /// Test-only fault-injection queue for [`Self::get_base_path`],
    /// consumed at the top of each call BEFORE the Valkey HGET round-trip.
    /// A popped `Some(err)` short-circuits the call and returns `err`; an
    /// empty queue lets the call run normally.
    ///
    /// Used to drive `acquire_session_cancel`'s "transient Internal error"
    /// branch deterministically. That branch must surface the error and leave
    /// the `session_tokens` entry — and the token another caller may already
    /// share — alone: only a `NotFound` removes and cancels it. Tests inject
    /// `Internal` here to verify it.
    ///
    /// **Test-only.** Production code never populates this.
    #[cfg(any(test, feature = "test-fault-injection"))]
    get_base_path_fault_injector_for_test:
        Option<std::sync::Arc<std::sync::Mutex<std::collections::VecDeque<ServiceError>>>>,
    /// Optional async-off-hot-path session-checkpoint dispatcher (SR-2.7,
    /// Task `t2.fgj.7v.p9`). When set, `create_once` and `delete` enqueue a
    /// fire-and-forget write or delete after their respective success arms
    /// so the parent Task's `SessionCheckpointWriter` can persist a fresh
    /// session-checkpoint off the lifecycle hot path. `None` (the default
    /// for tests and integration tests that construct `SessionStore`
    /// directly) makes the hook a no-op, so existing call sites keep their
    /// pre-feature behaviour unchanged.
    checkpoint_writer: Option<Arc<SessionCheckpointWriter>>,
    /// Test-only hook awaited once inside `create_once`'s WATCH window,
    /// immediately after `mark_watched` (b.bhb). See
    /// [`Self::with_watch_window_hook_for_test`].
    #[cfg(any(test, feature = "test-fault-injection"))]
    watch_window_hook_for_test: Option<crate::storage::watched_txn::WatchWindowHook>,
    /// Test-only hook awaited once in `create_once` after the pre-dispatch
    /// connection-identity check passes and before the commit pipeline is sent
    /// (b.bhb). See [`Self::with_pre_dispatch_hook_for_test`].
    #[cfg(any(test, feature = "test-fault-injection"))]
    pre_dispatch_hook_for_test: Option<crate::storage::watched_txn::WatchWindowHook>,
    /// Test-only queue of substituted `EXEC` verdicts for the watched commit
    /// (b.bhb). See [`Self::with_watched_exec_override_for_test`].
    #[cfg(any(test, feature = "test-fault-injection"))]
    watched_exec_override_for_test: Option<crate::storage::watched_txn::WatchedExecOverrideQueue>,
}

/// Default upper bound on the full-jitter backoff window used by
/// [`SessionStore::create`]. Matches `update_layer`'s cap so the two retry
/// loops are tuned identically on the same Valkey (b.91j H-stor-3).
pub const SESSION_RETRY_CAP_DELAY_MS: u64 = 50;

/// Default attempts when `SessionStore::new` is used without a builder call.
/// Matches `LIVE_EDIT_SESSION_RETRY_MAX_ATTEMPTS` default.
pub const DEFAULT_SESSION_RETRY_MAX_ATTEMPTS: u32 = 32;

/// Default initial jitter window when `SessionStore::new` is used without a
/// builder call. Matches `LIVE_EDIT_SESSION_RETRY_INITIAL_DELAY_MS` default.
pub const DEFAULT_SESSION_RETRY_INITIAL_DELAY_MS: u64 = 1;

impl SessionStore {
    pub fn new(client: ValkeyClient, keys: KeySchema) -> Self {
        Self {
            client,
            keys,
            retry_max_attempts: DEFAULT_SESSION_RETRY_MAX_ATTEMPTS,
            retry_initial_delay_ms: DEFAULT_SESSION_RETRY_INITIAL_DELAY_MS,
            #[cfg(any(test, feature = "test-fault-injection"))]
            delete_exec_fault_injector_for_test: None,
            #[cfg(any(test, feature = "test-fault-injection"))]
            create_fault_injector_for_test: None,
            #[cfg(any(test, feature = "test-fault-injection"))]
            get_base_path_fault_injector_for_test: None,
            checkpoint_writer: None,
            #[cfg(any(test, feature = "test-fault-injection"))]
            watch_window_hook_for_test: None,
            #[cfg(any(test, feature = "test-fault-injection"))]
            pre_dispatch_hook_for_test: None,
            #[cfg(any(test, feature = "test-fault-injection"))]
            watched_exec_override_for_test: None,
        }
    }

    /// Install a hook awaited once inside `create_once`'s WATCH window,
    /// immediately after the `WATCH` is acknowledged and the guard is armed
    /// (b.bhb). The hook receives the *guarded* fred `Client` — the connection
    /// actually carrying the `WATCH`.
    ///
    /// **Test-only.** The seam for driving the real mechanism b.bhb defends
    /// against: read the connection's `CLIENT ID`, `CLIENT KILL ID` it from a
    /// second connection, create a session with the same name, and observe that
    /// the commit is refused or compensated rather than silently landing a
    /// duplicate. The pre-existing injector seams cannot express a window
    /// *between* two commands on one connection.
    ///
    /// Compiled only under `#[cfg(any(test, feature = "test-fault-injection"))]`.
    #[cfg(any(test, feature = "test-fault-injection"))]
    pub fn with_watch_window_hook_for_test(
        mut self,
        hook: crate::storage::watched_txn::WatchWindowHook,
    ) -> Self {
        self.watch_window_hook_for_test = Some(hook);
        self
    }

    /// Install a hook awaited once in `create_once` AFTER the pre-dispatch
    /// connection-identity check passes and BEFORE the commit pipeline is sent
    /// (b.bhb).
    ///
    /// **Test-only.** The check cannot see a reconnect that completes after it
    /// returns; this hook places one there deterministically, so the
    /// in-transaction `CLIENT ID` witness and its compensation path are
    /// exercised without racing a real failover.
    ///
    /// Compiled only under `#[cfg(any(test, feature = "test-fault-injection"))]`.
    #[cfg(any(test, feature = "test-fault-injection"))]
    pub fn with_pre_dispatch_hook_for_test(
        mut self,
        hook: crate::storage::watched_txn::WatchWindowHook,
    ) -> Self {
        self.pre_dispatch_hook_for_test = Some(hook);
        self
    }

    /// Install a FIFO queue of substituted `EXEC` verdicts for `create_once`'s
    /// watched commit (b.bhb). One entry is consumed per attempt that reaches
    /// dispatch; an empty queue runs the real pipeline and decodes the real
    /// verdict.
    ///
    /// **Test-only.** A real `WatchedExec::InDoubt` comes from a truncated reply
    /// set — the socket dying with the pipeline in flight, or the `EXEC` reply
    /// outliving the command timeout. A connection stall can stage only the
    /// second, and only as "the pipeline reached the server and will run once
    /// released". This override stages "no answer" with the keyspace state
    /// chosen — our name member landed, or nothing was written — which are
    /// exactly the two outcomes [`Self::reconcile_in_doubt_create`] has to tell
    /// apart, so both must be stageable. See
    /// [`crate::storage::watched_txn::WatchedExecOverride`].
    ///
    /// Compiled only under `#[cfg(any(test, feature = "test-fault-injection"))]`.
    #[cfg(any(test, feature = "test-fault-injection"))]
    pub fn with_watched_exec_override_for_test(
        mut self,
        queue: crate::storage::watched_txn::WatchedExecOverrideQueue,
    ) -> Self {
        self.watched_exec_override_for_test = Some(queue);
        self
    }

    /// Builder: override the max-attempts budget for `create`'s retry loop
    /// (b.91j H-stor-3). Values below 1 are clamped to 1 — zero would disable
    /// the first attempt.
    pub fn with_retry_max_attempts(mut self, max_attempts: u32) -> Self {
        self.retry_max_attempts = max_attempts.max(1);
        self
    }

    /// Builder: override the full-jitter initial delay (ms) for `create`'s
    /// retry loop (b.91j H-stor-3).
    pub fn with_retry_initial_delay_ms(mut self, initial_delay_ms: u64) -> Self {
        self.retry_initial_delay_ms = initial_delay_ms.max(1);
        self
    }

    /// Builder: install the async-off-hot-path session-checkpoint dispatcher
    /// (SR-2.7, Task `t2.fgj.7v.p9`). Once installed, `create` enqueues a
    /// fire-and-forget checkpoint write after its success arm (committed
    /// session, post-HSET) and `delete` enqueues a fire-and-forget
    /// checkpoint delete after its success arm. Early-return paths
    /// (`AlreadyExists`, `Aborted`, `NotFound`, `Unavailable`, `Internal`) do
    /// not enqueue, matching the "no committed state change → no checkpoint
    /// dispatch" contract.
    pub fn with_checkpoint_writer(mut self, writer: Arc<SessionCheckpointWriter>) -> Self {
        self.checkpoint_writer = Some(writer);
        self
    }

    /// Install a queue of synthetic [`DeleteExecFault`] values consumed at
    /// the top of [`Self::delete`], substituting for the real
    /// `exec_txn::<fred::types::Value>(...)` round-trip. Each call pops
    /// one entry: a popped entry drives the matching match arm without
    /// issuing EXEC; an empty queue lets `delete` run normally.
    ///
    /// **Test-only.** Production call sites never invoke this — it exists
    /// so that b.91j C-3 regression tests can drive the `Aborted` branches
    /// (`Value::Null` / `is_canceled()`) deterministically without a real
    /// connection drop. Compiled only under
    /// `#[cfg(any(test, feature = "test-fault-injection"))]`.
    #[cfg(any(test, feature = "test-fault-injection"))]
    pub fn with_delete_exec_fault_injector_for_test(
        mut self,
        queue: std::sync::Arc<std::sync::Mutex<std::collections::VecDeque<DeleteExecFault>>>,
    ) -> Self {
        self.delete_exec_fault_injector_for_test = Some(queue);
        self
    }

    /// Install a queue of synthetic `ServiceError`s consumed at the top of
    /// [`Self::create_once`] on every attempt of the retry loop. Each
    /// attempt pops one entry; a popped `Some(err)` short-circuits that
    /// attempt and returns `err`, an empty queue lets the attempt run
    /// normally.
    ///
    /// **Test-only.** Production call sites never invoke this — it exists
    /// so that b.91j H-stor-3 coverage can drive the retry loop to
    /// exhaustion deterministically. Compiled only under
    /// `#[cfg(any(test, feature = "test-fault-injection"))]`.
    #[cfg(any(test, feature = "test-fault-injection"))]
    pub fn with_create_fault_injector_for_test(
        mut self,
        queue: std::sync::Arc<std::sync::Mutex<std::collections::VecDeque<ServiceError>>>,
    ) -> Self {
        self.create_fault_injector_for_test = Some(queue);
        self
    }

    /// Install a queue of synthetic `ServiceError`s consumed at the top of
    /// [`Self::get_base_path`]. Each call pops one entry; a popped
    /// `Some(err)` short-circuits the call with `err` without issuing the
    /// HGET, an empty queue lets the call run normally.
    ///
    /// **Test-only.** Production call sites never invoke this — it exists
    /// so that regression coverage can drive the "transient Internal" branch
    /// of `acquire_session_cancel` deterministically.
    /// Compiled only under `#[cfg(any(test, feature = "test-fault-injection"))]`.
    #[cfg(any(test, feature = "test-fault-injection"))]
    pub fn with_get_base_path_fault_injector_for_test(
        mut self,
        queue: std::sync::Arc<std::sync::Mutex<std::collections::VecDeque<ServiceError>>>,
    ) -> Self {
        self.get_base_path_fault_injector_for_test = Some(queue);
        self
    }

    /// Create a new session. Returns the generated session_id.
    ///
    /// Validates the session name and stores session metadata with
    /// WATCH/MULTI/EXEC to prevent duplicate-name races (SDD §7.10).
    ///
    /// Each attempt of the retry loop runs under its own fresh UUID4
    /// `session_id`, and the id returned is the successful attempt's; the
    /// duplicate-name scans skip names members carrying any id this call
    /// created. See [`Self::undo_create_attempt`] (and SDD §7.10) for why.
    ///
    /// ## Names-set uniqueness (b.91j C-7)
    ///
    /// The duplicate-name check and the commit use a **single** same-slot set,
    /// `le:path:{hex(base_path)}:names` (`path_names_index`). Members are the
    /// composite `name\0session_id` (see [`KeySchema::path_name_member`]).
    /// Because it shares the `{hex(base_path)}` hash tag with
    /// [`KeySchema::path_index`], the WATCH-SMEMBERS-SADD sequence is
    /// slot-safe on Valkey Cluster and fully atomic — the pre-fix race
    /// window, which depended on `HSET session_meta` happening on a
    /// different slot AFTER EXEC, is closed because the same-slot names set
    /// is the fence the duplicate check reads.
    ///
    /// `HSET session_meta` still runs after EXEC and supplies the timestamps /
    /// created_by metadata. It is not the fence, but when it is sent still
    /// matters for uniqueness — see [`Self::undo_create_attempt`].
    ///
    /// A crash between EXEC and HSET leaves an entry in the names set whose
    /// `session_meta` is missing — `list` surfaces it as a ghost member
    /// (which the ListSessions handler removes once the recovery probe finds
    /// no checkpoint), and a subsequent `DeleteSession` on the same
    /// session_id returns `NotFound` from the empty-HGETALL short-circuit.
    ///
    /// A failed `HSET`, or a failed read while reconciling an `EXEC` whose
    /// reply was lost, first undoes the attempt; see
    /// [`Self::undo_create_attempt`] for the verdict and what can be left
    /// behind.
    pub async fn create(
        &self,
        name: &str,
        base_path: &str,
        created_by: &str,
    ) -> Result<String, ServiceError> {
        validate_session_name(name)?;

        let created_at = now_micros();

        let path_index_key = self.keys.path_index(base_path);
        let path_names_key = self.keys.path_names_index(base_path);

        let retry_cfg = RetryConfig {
            max_attempts: self.retry_max_attempts,
            initial_delay_ms: self.retry_initial_delay_ms,
            cap_delay_ms: SESSION_RETRY_CAP_DELAY_MS,
        };

        // Every `session_id` this call has created, one per attempt; the
        // duplicate scans skip them.
        let mut own_session_ids: Vec<String> = Vec::new();

        retry_aborted_with_jitter("create_session", retry_cfg, |_attempt| {
            // A fresh id per attempt (see `undo_create_attempt` for why).
            let session_id = uuid::Uuid::new_v4().to_string();
            own_session_ids.push(session_id.clone());
            let own_session_ids = own_session_ids.clone();
            let session_meta_key = self.keys.session_meta(&session_id);
            let name_member = KeySchema::path_name_member(name, &session_id);
            let path_index_key = path_index_key.clone();
            let path_names_key = path_names_key.clone();
            async move {
                self.create_once(
                    name,
                    base_path,
                    created_by,
                    &session_id,
                    created_at,
                    &path_index_key,
                    &path_names_key,
                    &session_meta_key,
                    &name_member,
                    &own_session_ids,
                )
                .await
            }
        })
        .await
    }

    /// Single WATCH/MULTI/EXEC attempt of [`Self::create`]. Factored out so
    /// the retry loop can drive it via [`retry_aborted_with_jitter`] (b.91j
    /// H-stor-3). Each attempt runs under its own `session_id`;
    /// `own_session_ids` lists every id the call has created so far, this
    /// attempt's included. Returns `Aborted` on retriable failures (WATCH
    /// conflict, EXEC connection drop, a transient store error, an undone
    /// unwatched commit, or a failure after the commit was sent that
    /// [`Self::undo_create_attempt`] rules retriable); all other errors are
    /// terminal.
    #[allow(clippy::too_many_arguments)]
    async fn create_once(
        &self,
        name: &str,
        base_path: &str,
        created_by: &str,
        session_id: &str,
        created_at: u64,
        path_index_key: &str,
        path_names_key: &str,
        session_meta_key: &str,
        name_member: &str,
        own_session_ids: &[String],
    ) -> Result<String, ServiceError> {
        // Test-only: if a fault injector is installed and has a pending
        // error, pop it and short-circuit before any Valkey round-trip. Lets
        // the b.91j H-stor-3 test drive the retry loop to exhaustion without
        // a real WATCH-contention race. Gated behind `test-fault-injection`
        // so release builds never see this branch.
        #[cfg(any(test, feature = "test-fault-injection"))]
        if let Some(queue) = self.create_fault_injector_for_test.as_ref() {
            if let Some(err) = queue.lock().unwrap().pop_front() {
                return Err(err);
            }
        }

        // Acquire an EXCLUSIVE connection for the WATCH/MULTI/EXEC window.
        // See the pre-b.91j notes on the `WatchGuard`/`ExclusivePool` pairing
        // — WATCH is per-connection, so a multiplexed pool would lose the
        // watch set when other callers interleave commands (SDD §7.10, b.6e7).
        // b.7td: pinned to the `{hex(base_path)}` slot the WATCH below lands
        // on, so the checkout hygiene probe reaches the node this transaction
        // will actually run against.
        let checkout = self.client.acquire_txn_for(path_names_key).await;
        let mut client = WatchGuard::new(checkout, self.client.watch_unwatch_tracker())
            .with_metrics(self.client.metrics_snapshot());

        // WATCH the authoritative names index (b.91j C-7). A concurrent
        // `create` or `delete` that touches the same `(base_path)` mutates
        // this set inside a MULTI on the same Valkey Cluster slot, so our
        // EXEC will return nil and we retry — this replaces the pre-fix
        // WATCH(path_index) scheme, which raced against the `HSET session_meta`
        // running on a different slot after EXEC.
        //
        // b.bhb: classified through the shared transient classifier, not wrapped
        // as `Internal` directly. The txn pool writes each command exactly once
        // (`max_command_attempts = 1`), so a routine reconnect landing on this
        // command is now visible here instead of being absorbed by fred — and
        // the caller's safe move is unchanged: `Aborted` re-enters
        // `retry_aborted_with_jitter`, which takes a fresh checkout and issues a
        // fresh `WATCH`. Permanent kinds still surface as `Internal`.
        //
        // The `WATCH` rides in a pipeline that ALSO calibrates the L3 witness
        // baseline — `MULTI; CLIENT ID; CLIENT ID; EXEC` ahead of it, on the
        // same connection, in one router turn and one round trip — so the
        // witness read from inside the commit is compared against an id sampled
        // in the same context rather than against fred's handshake-time id. On
        // a proxied managed store the two can come from different id spaces,
        // and comparing them would make every ordinary `CreateSession`
        // compensate and retry to exhaustion. See `storage::watched_txn`.
        let watch_hash = fred::types::ClusterHash::from(path_names_key);
        let witness_baseline = crate::storage::watched_txn::watch_with_baseline(
            &client,
            &watch_hash,
            &[path_names_key],
        )
        .await
        .map_err(|e| {
            map_transient_fred_err(
                FredErrOrigin::client(&self.client, POOL_LABEL_TXN),
                "WATCH path names index",
                e,
            )
        })?;
        // Record the slot the WATCH landed on so teardown is routed back to
        // the same cluster node (b.ji8). Before this, the keyless UNWATCH was
        // routed to an arbitrary node and this `{hex(base_path)}` WATCH leaked
        // onto the shared txn connection, surfacing as a CROSSSLOT on whatever
        // unrelated `{object_id}` command the next borrower issued.
        client.mark_watched(path_names_key, &[]);

        // The store would not give a stable in-`MULTI` `CLIENT ID`, so L3 is
        // disarmed for this attempt: the commit queues no witness (queueing one
        // on such a store would `EXECABORT` the whole transaction) and nothing
        // is ever accused. Not a failure — record it and continue on L1+L2.
        if let Some(reason) = witness_baseline.unavailable_reason() {
            if let Some(metrics) = self.client.metrics_snapshot() {
                metrics.inc_storage_watch_voided(
                    WATCH_VOIDED_PATH_CREATE_SESSION,
                    WATCH_VOIDED_WITNESS_UNAVAILABLE,
                );
            }
            crate::storage::watched_txn::warn_witness_unavailable(
                WATCH_VOIDED_PATH_CREATE_SESSION,
                reason,
            );
        }

        // The identity of the connection now carrying the WATCH, for the L2
        // pre-dispatch check below. A `WATCH` dies with its connection, so a
        // replacement between here and the dispatch voids the watch set.
        // Reading it is a local map lookup — no round trip, so it does not
        // widen the window it measures.
        let watch_fingerprint = WatchFingerprint::take(&client, &watch_hash);

        // Test-only: the WATCH is now live on a known connection. See
        // `LayerStore::with_watch_window_hook_for_test` for what this seam
        // exists to drive.
        #[cfg(any(test, feature = "test-fault-injection"))]
        if let Some(hook) = self.watch_window_hook_for_test.clone() {
            hook(&client).await;
        }

        // Duplicate-name check: iterate the names set, splitting each member
        // on NUL to recover `(existing_name, existing_sid)`. `validate_session_name`
        // rejects NUL in names, so a literal NUL in a member unambiguously
        // separates the two halves.
        //
        // b.bhb: same classification as the `WATCH` above — this read runs on
        // the same one-attempt txn connection, and a reconnect that interrupts
        // it is a retriable condition, not an `Internal`. Retrying is also the
        // only *correct* answer: this read is the fence the transaction below
        // relies on, so a read that never completed must not be treated as
        // "no duplicate found".
        let existing_members: Vec<String> = client.smembers(path_names_key).await.map_err(|e| {
            map_transient_fred_err(
                FredErrOrigin::client(&self.client, POOL_LABEL_TXN),
                "SMEMBERS path names index",
                e,
            )
        })?;

        // Members carrying an id this call created are skipped. See
        // `undo_create_attempt` (and SDD §7.10) for why that cannot admit a
        // second session with this name.
        for member in &existing_members {
            if Self::names_a_competing_holder(member, name, own_session_ids) {
                // UNWATCH before returning the error (WatchGuard would
                // also clean up on drop, but synchronous UNWATCH avoids
                // one background round-trip on the common duplicate path).
                // The funnel disarms the guard itself on success and leaves
                // it armed on failure, so a failed teardown is retried by
                // Drop under the pool mutex — nothing to do with the
                // result here (b.ji8).
                let _ = client.unwatch_pinned().await;
                return Err(ServiceError::AlreadyExists(format!(
                    "Session '{name}' already exists"
                )));
            }
        }

        // Queue SADDs inside MULTI/EXEC. Both the canonical path_index
        // (session-id set) and the new names index live on the same slot
        // (both tagged `{hex(base_path)}`), so the transaction is slot-safe
        // even on Valkey Cluster.
        // b.bhb: a fred `Pipeline` of regular commands, NOT a fred
        // `Transaction`. `Transaction`'s router replays the whole `MULTI … EXEC`
        // on a fresh connection after a reconnect and on a fresh node after a
        // `MOVED` (`fred-10.1.0/src/router/transactions.rs` `send`), and a
        // replayed `EXEC` runs unwatched — Valkey commits it unconditionally.
        // Neither `SADD` carries a server-side fence of its own, so on this path
        // a void WATCH is a *silent* loss of the duplicate-name check above: two
        // sessions with one name at one base path, which is the uniqueness
        // guarantee `create` exists to provide. See `storage::watched_txn`.
        let txn = match WatchedTxn::open(&client, &watch_hash, &witness_baseline).await {
            Ok(txn) => txn,
            Err(e) => {
                let _ = client.unwatch_pinned().await;
                return Err(map_transient_fred_err(
                    FredErrOrigin::client(&self.client, POOL_LABEL_TXN),
                    "MULTI queue",
                    e,
                ));
            }
        };
        let trx = txn.commands();
        let _: () = trx
            .sadd(path_index_key, session_id)
            .await
            .map_err(|e| ServiceError::Internal(format!("SADD path_index queue failed: {e}")))?;
        let _: () = trx
            .sadd(path_names_key, name_member)
            .await
            .map_err(|e| ServiceError::Internal(format!("SADD path_names queue failed: {e}")))?;

        // Pre-dispatch watch-validity check (b.bhb). Nothing has reached the
        // wire yet — queueing into the pipeline is a local buffer push — so this
        // is the last free moment to notice that the connection carrying the
        // WATCH has been replaced, or is being replaced right now, which would
        // make the EXEC below commit unconditionally. Abort into the existing
        // retry loop, which re-runs the duplicate scan under a fresh WATCH.
        //
        // Narrowing, not a fence. The residual is any reconnect fred *processes*
        // between this check and the socket write — not only one that completes
        // in that gap: a peer-side close enqueues a `Reconnect` on the same
        // router channel this pipeline is queued on, and if it lands first the
        // router reconnects and writes the pipeline onto a connection that never
        // held the WATCH, with no attempt charged. The `CLIENT ID` witness
        // inside the transaction detects it and `compensate_unwatched_create`
        // undoes it, which is what makes that safe (or, when the compensating
        // SREMs fail, visible on the alertable metric cell).
        match crate::storage::watched_txn::dispatch_verdict(
            &client,
            &watch_fingerprint,
            &watch_hash,
        ) {
            DispatchVerdict::Dispatch => {}
            // fred is connected but holds no id for this connection, so L2 has
            // nothing to compare. Dispatch anyway — refusing here would abort
            // EVERY CreateSession on a store that does not answer `CLIENT ID`
            // at handshake — and record that the layer is off.
            DispatchVerdict::DispatchUncomparable => {
                if let Some(metrics) = self.client.metrics_snapshot() {
                    metrics.inc_storage_watch_voided(
                        WATCH_VOIDED_PATH_CREATE_SESSION,
                        WATCH_VOIDED_FINGERPRINT_UNAVAILABLE,
                    );
                }
                crate::storage::watched_txn::warn_fingerprint_unavailable(
                    WATCH_VOIDED_PATH_CREATE_SESSION,
                );
            }
            DispatchVerdict::Refuse => {
                let _ = client.unwatch_pinned().await;
                if let Some(metrics) = self.client.metrics_snapshot() {
                    metrics.inc_storage_watch_voided(
                        WATCH_VOIDED_PATH_CREATE_SESSION,
                        WATCH_VOIDED_ABORTED_PRE_EXEC,
                    );
                }
                return Err(ServiceError::Aborted(
                    "create_session WATCH connection replaced before EXEC, retry".to_owned(),
                ));
            }
        }

        // Test-only: the residual window the check above cannot cover. See
        // `LayerStore::with_pre_dispatch_hook_for_test`.
        #[cfg(any(test, feature = "test-fault-injection"))]
        if let Some(hook) = self.pre_dispatch_hook_for_test.clone() {
            hook(&client).await;
        }

        // The reply is decoded per-frame (`WatchedTxn::exec` → `WatchedExec`)
        // rather than as a single value. Decoding it as `()` or `Vec<T>` would
        // be the b.8q1 defect: fred's `FromValue` accepts `Value::Null` for
        // both, so a WATCH-canceled reply would read as success and the caller
        // would be handed a session_id for a session that was never written.
        //
        // Test-only: a queued override substitutes the verdict here, at the one
        // point where the site's whole post-exec handling — the `mark_unwatched`
        // bookkeeping, the guard's armed/disarmed state, the reconciliation —
        // still runs exactly as it does in production. See
        // `with_watched_exec_override_for_test`.
        let txn_origin = FredErrOrigin::client(&self.client, POOL_LABEL_TXN);
        #[cfg(any(test, feature = "test-fault-injection"))]
        let exec_outcome = match self
            .watched_exec_override_for_test
            .as_ref()
            .and_then(|q| q.lock().unwrap().pop_front())
        {
            Some(over) => txn.exec_overridden(over, txn_origin).await,
            None => txn.exec(txn_origin).await,
        };
        #[cfg(not(any(test, feature = "test-fault-injection")))]
        let exec_outcome = txn.exec(txn_origin).await;

        match exec_outcome {
            WatchedExec::Conflict => {
                // EXEC canceled — watched key modified. Return Aborted so
                // the retry helper sleeps with jitter and tries again.
                client.mark_unwatched();
                Err(ServiceError::Aborted(
                    "create_session concurrent write conflict, retry".to_owned(),
                ))
            }
            WatchedExec::Committed { witness } => {
                client.mark_unwatched();
                if witness_baseline.contradicted_by(witness) {
                    // b.bhb: the transaction's own `CLIENT ID` disagrees with
                    // the baseline sampled on the WATCH's own connection — a
                    // PROVEN unwatched commit, so the duplicate-name scan above
                    // did not fence this write. Both members are uniquely ours
                    // (the session_id is a UUID minted for this attempt alone,
                    // and `session_meta` has not been written yet), so the
                    // commit can be undone and the call retried.
                    //
                    // The re-taken L2 fingerprint is diagnostic decoration on
                    // the compensation log, never an input: a completed
                    // reconnect is the one thing that rewrites fred's
                    // connection-id map, so whether it moved says whether both
                    // observations agree on the cause.
                    let fingerprint_moved = watch_fingerprint.moved_since(&client, &watch_hash);
                    return self
                        .compensate_unwatched_create(
                            name,
                            session_id,
                            path_index_key,
                            path_names_key,
                            name_member,
                            fingerprint_moved,
                        )
                        .await;
                }
                // Uniqueness has committed. The metadata write supplies the
                // auxiliary fields (created_by/created_at). A crash here leaves
                // the names entry + path_index entry pointing at a session_id
                // whose meta is empty; list/delete handle that orphan case
                // already (see delete's empty-HGETALL short-circuit and list's
                // ghost members). A failed write is undone inside
                // `finish_create`, and a transient one is then retried.
                self.finish_create(
                    name,
                    base_path,
                    created_by,
                    session_id,
                    created_at,
                    path_index_key,
                    path_names_key,
                    session_meta_key,
                    name_member,
                )
                .await
            }
            // A discarded transaction (`EXECABORT`, or the cluster check's
            // bare `-TRYAGAIN` to the `EXEC`): the server ran NO queued
            // command, so it is classified from its `cause` — the queued
            // command's own refusal, the `EXEC`'s `EXECABORT … because of:
            // <refusal>`, or the `-TRYAGAIN` itself — never from a bare
            // `EXECABORT` reply, whose "previous errors" text names no cause
            // (b.xd1). Through the mapper: a store write refusal (`-OOM` on a
            // full store, …) is a plain `Unavailable` the retry loop does not
            // retry, a transient cause (`-LOADING`, `-BUSY`, a slot
            // migration's `-TRYAGAIN`) is `Aborted` and retried under a fresh
            // session id — nothing ran, so nothing is left behind — and
            // anything else is `Internal`. No tear handling and no
            // reconciliation: nothing was written. The guard is left armed for
            // the reason the `ExecError` arm below gives; Valkey's
            // `discardTransaction` has cleared the watch set, so the Drop-time
            // `UNWATCH` is redundant, but it is the safe direction.
            WatchedExec::Discarded { exec: _, cause } => Err(map_transient_fred_err(
                FredErrOrigin::client(&self.client, POOL_LABEL_TXN),
                "EXEC create_session",
                cause,
            )),
            WatchedExec::ExecError(e) => {
                // An error frame came back from the commit. The variant covers
                // more than a failed `SADD`: an errored first element, a
                // trailing `EXEC` that itself errored without running the queued
                // frames, and an unexpected reply shape all land here (an
                // `EXECABORT` and the cluster check's `-TRYAGAIN` to the
                // `EXEC` are the `Discarded` arm above). So the
                // classification is delegated rather than asserted — the shared
                // transient-vs-terminal classifier keeps `WRONGTYPE` (a
                // keyspace anomaly, not a concurrency outcome) on the pre-b.bhb
                // non-retriable `Internal`, while a transient server frame
                // becomes a retriable `Aborted` that re-enters the caller's
                // retry — matching the structurally identical arm in
                // `update_layer_once_watch`. (A `-CLUSTERDOWN` frame never
                // reaches this arm: fred drops the connection on it, so the
                // pipeline answers one `Err`, which is `InDoubt`.)
                //
                // Deliberately NOT counted as a void watch: unlike
                // `update_layer`'s torn `XADD`, nothing here makes an error
                // frame evidence that the WATCH was gone.
                //
                // The guard is left ARMED on purpose (b.bhb round 4). Two of the
                // shapes above never prove the `EXEC` executed — an errored
                // `results[0]` whose trailing `EXEC` also failed without running,
                // and the unexpected-reply path — so the watch set may still be
                // live, and disarming here would skip the Drop-time slot-pinned
                // `UNWATCH` and re-open the b.jov / b.ji8 leaked-WATCH class.
                // The cost is one redundant slot-pinned `UNWATCH` on the shapes
                // where the server did clear the set, which is the same trade
                // `WatchedExecOverride::ReplaceOutcomeAfterExec` already
                // documents.
                Err(map_transient_fred_err(
                    FredErrOrigin::client(&self.client, POOL_LABEL_TXN),
                    "EXEC create_session",
                    e,
                ))
            }
            WatchedExec::InDoubt(e) => {
                // A timed-out EXEC was already counted by `WatchedTxn::exec`;
                // the reconciler never maps `e` itself.
                //
                // Connection drop / lost reply mid-EXEC. Whether the SADDs ran
                // is UNKNOWN. Pre-b.bhb this was an unconditional `Aborted`,
                // which is wrong in the direction that matters: if the commit
                // DID land, a blind retry throws away the session the caller
                // just successfully created (then under the same id, its own
                // member answered `AlreadyExists`; under a fresh id per attempt
                // it would be left behind as a ghost). Ask the keyspace instead.
                self.reconcile_in_doubt_create(
                    name,
                    base_path,
                    created_by,
                    session_id,
                    created_at,
                    path_index_key,
                    path_names_key,
                    session_meta_key,
                    name_member,
                    own_session_ids,
                    e,
                )
                .await
            }
        }
    }

    /// Post-EXEC tail shared by the committed and in-doubt-but-landed paths:
    /// write the auxiliary metadata and enqueue the session checkpoint.
    ///
    /// A failed `HSET session_meta` is answered by
    /// [`Self::undo_create_attempt`], which undoes the attempt and decides the
    /// returned error.
    #[allow(clippy::too_many_arguments)]
    async fn finish_create(
        &self,
        name: &str,
        base_path: &str,
        created_by: &str,
        session_id: &str,
        created_at: u64,
        path_index_key: &str,
        path_names_key: &str,
        session_meta_key: &str,
        name_member: &str,
    ) -> Result<String, ServiceError> {
        let hset_result = self
            .client
            .pool
            .hset::<(), _, _>(
                session_meta_key,
                [
                    ("name", name.to_owned()),
                    ("base_path", base_path.to_owned()),
                    ("created_by", created_by.to_owned()),
                    ("created_at", created_at.to_string()),
                ],
            )
            .await;
        if let Err(e) = hset_result {
            return Err(self
                .undo_create_attempt(
                    name,
                    base_path,
                    session_id,
                    path_index_key,
                    path_names_key,
                    session_meta_key,
                    name_member,
                    "HSET session meta (create)",
                    e,
                )
                .await);
        }

        tracing::info!(
            session_id = %session_id,
            session_name = %name,
            base_path = %base_path,
            "session created"
        );
        // Enqueue the async-off-hot-path session-checkpoint write.
        // Reached only after the EXEC committed and the post-EXEC
        // HSET landed, so the dispatcher's read-side assembler sees
        // populated `session_meta`. `None` (the default for tests
        // that construct `SessionStore` directly via `new`) makes
        // this a no-op so existing call sites compile unchanged.
        if let Some(writer) = &self.checkpoint_writer {
            writer.enqueue_session_checkpoint_write(session_id.to_owned());
        }
        Ok(session_id.to_owned())
    }

    /// Whether the `path_names_index` member `member` names a holder of `name`
    /// other than this call: its name is `name` and its `session_id` is not one
    /// of `own_session_ids` (every id this [`Self::create`] call has created).
    /// A member without a NUL separator names no holder.
    ///
    /// Shared by both duplicate-name scans — `create_once`'s scan under the
    /// `WATCH` and the in-doubt re-scan in `reconcile_in_doubt_create` — so the
    /// two cannot drift. See [`Self::undo_create_attempt`] (and SDD §7.10) for
    /// why skipping the call's own ids cannot admit a second session with the
    /// same name.
    fn names_a_competing_holder(member: &str, name: &str, own_session_ids: &[String]) -> bool {
        KeySchema::split_path_name_member(member).is_some_and(|(existing_name, sid)| {
            existing_name == name && !own_session_ids.iter().any(|own| own == sid)
        })
    }

    /// Undo a `create_once` commit that the in-transaction `CLIENT ID` witness
    /// proved ran without its `WATCH` (b.bhb).
    ///
    /// Both members are uniquely ours — the `session_id` is a UUID minted for
    /// this attempt, and `session_meta` has not been written yet — so
    /// `SREM`ing them restores the pre-attempt keyspace exactly and the
    /// caller's retry, under a new `session_id`, re-decides the duplicate scan
    /// under a live `WATCH`. Runs on the data pool: the txn slot is being
    /// released, and its connection is the one whose identity is in question.
    ///
    /// A failed compensation is the one residual where a duplicate name can
    /// survive, so it is loud on both channels — `error!` plus the alertable
    /// `committed_unwatched` cell — and reported as `Internal` rather than
    /// `Aborted`, since retrying would leave the unfenced members in place —
    /// and the retry's scan skips this attempt's id, so they would not even
    /// stop it.
    ///
    /// `fingerprint_moved` is diagnostic decoration only (see
    /// [`crate::storage::watched_txn::WatchFingerprint::moved_since`]); it never
    /// participates in the verdict.
    #[allow(clippy::too_many_arguments)]
    async fn compensate_unwatched_create(
        &self,
        name: &str,
        session_id: &str,
        path_index_key: &str,
        path_names_key: &str,
        name_member: &str,
        fingerprint_moved: Option<bool>,
    ) -> Result<String, ServiceError> {
        let index_srem = self
            .client
            .pool
            .srem::<i64, _, _>(path_index_key, session_id)
            .await;
        let names_srem = self
            .client
            .pool
            .srem::<i64, _, _>(path_names_key, name_member)
            .await;

        let origin = FredErrOrigin::client(&self.client, POOL_LABEL_DATA);
        for e in [index_srem.as_ref().err(), names_srem.as_ref().err()]
            .into_iter()
            .flatten()
        {
            origin.observe_error(e);
        }
        let metrics = self.client.metrics_snapshot();
        match (index_srem, names_srem) {
            (Ok(_), Ok(_)) => {
                if let Some(metrics) = metrics {
                    metrics.inc_storage_watch_voided(
                        WATCH_VOIDED_PATH_CREATE_SESSION,
                        WATCH_VOIDED_COMPENSATED,
                    );
                }
                tracing::warn!(
                    session_id = %session_id,
                    session_name = %name,
                    fingerprint_moved = ?fingerprint_moved,
                    "create_session committed on a connection that did not hold its WATCH; both \
                     members were uniquely ours and have been removed — retrying the attempt \
                     under a fresh WATCH"
                );
                Err(ServiceError::Aborted(
                    "create_session committed unwatched, compensated, retry".to_owned(),
                ))
            }
            (index_result, names_result) => {
                if let Some(metrics) = metrics {
                    metrics.inc_storage_watch_voided(
                        WATCH_VOIDED_PATH_CREATE_SESSION,
                        WATCH_VOIDED_COMMITTED_UNWATCHED,
                    );
                }
                tracing::error!(
                    session_id = %session_id,
                    session_name = %name,
                    fingerprint_moved = ?fingerprint_moved,
                    path_index_error = ?index_result.err(),
                    path_names_error = ?names_result.err(),
                    "create_session committed on a connection that did not hold its WATCH and \
                     the compensating SREM failed: the duplicate-name check did not run and its \
                     members are still in the index. A second session may now share this name at \
                     this base path; the leaked member must be removed by hand"
                );
                Err(ServiceError::Internal(
                    "create_session committed without its WATCH and could not be undone".to_owned(),
                ))
            }
        }
    }

    /// Undo a `create_once` attempt whose commit landed, or may have landed,
    /// and whose next round trip failed; then decide the error the attempt
    /// returns (b.62m). This rustdoc is the source home of the CreateSession
    /// retry invariants; other sites point here (full rationale: SDD §7.10).
    ///
    /// Callers: the `HSET session_meta` in [`Self::finish_create`], and in
    /// [`Self::reconcile_in_doubt_create`] the `SISMEMBER` (the commit may
    /// not have landed) and the re-scan after a landed commit. `cause` is
    /// classified and its timeout counted here; callers must not count it.
    ///
    /// **One `session_id` per attempt.** A command that timed out on our side
    /// can still run later — fred never writes one after its timeout fired,
    /// but does not recall one it had already written — so no attempt reuses
    /// an id earlier commands
    /// targeted: every late write lands on the abandoned id's keys, never on
    /// the session the call returns. That is also why the undo is safe
    /// whether or not the commit landed.
    ///
    /// **The scans skip the call's own ids** (via
    /// [`Self::names_a_competing_holder`]). This cannot admit a second session
    /// with the same name, because at most one id per call is ever completed
    /// by an `HSET`: it is sent only after an `EXEC` known to have run exactly
    /// once (a Committed reply, or `SISMEMBER` reading present — the txn pool
    /// writes each command once and the commit is a `Pipeline`, never a
    /// replayed `Transaction`), and no retry follows an `HSET` failure whose
    /// names `SREM` was not confirmed. A later attempt can meet an own-id
    /// member only from an earlier attempt's late `EXEC`: a ghost.
    ///
    /// **The undo:** three concurrent data-pool commands, so a stalled store
    /// costs one command timeout, not three — `SREM` the names member, `SREM`
    /// the `path_index` member, and `DEL session_meta` (which matters only to
    /// the `HSET` caller). Each one's timeout is counted on the data pool.
    ///
    /// **The verdict turns on the names `SREM` alone** — the set the scans
    /// read:
    /// - **Released** → `map_transient_fred_err(cause)`: transient (the
    ///   store's `-LOADING` / `-BUSY` included) is `Aborted` and the call
    ///   retries under a new id; a store write refusal (an `HSET session_meta`
    ///   refused `-OOM`, …) is a plain `Unavailable` and is not retried
    ///   (b.xd1); anything else is `Internal`. A failed `path_index` `SREM` or
    ///   `DEL` only logs a WARN; it leaves unlisted debris under the abandoned
    ///   id.
    /// - **Not released** → `map_fred_err_for_grpc(cause)`: `Unavailable`
    ///   (transient or a store write refusal) / `Internal`, never `Aborted`,
    ///   plus a WARN. For the `HSET` caller a
    ///   retry would break uniqueness: that id's `HSET` may still land and
    ///   complete it beside the retry's session. For the in-doubt callers no
    ///   `HSET` was sent, so stopping leaves only a self-healing ghost, while a
    ///   retry would go back into the failing pool and risk a complete session
    ///   nobody was told about. If the commit landed, the name stays claimed;
    ///   with no metadata, until the next ListSessions of the base path gets
    ///   `NoCheckpoint` and removes the entry. If it never landed (possible
    ///   only on the `SISMEMBER` path), nothing holds the name.
    ///
    /// **Late writes** (no undo covers a write that lands after it; all are
    /// keyed by the abandoned id):
    /// - the original `HSET`, landing after the `DEL` or with no `DEL` run:
    ///   an unlisted, inert orphan `session_meta` if the name was released; if
    ///   the names `SREM` also failed, a complete session with no checkpoint
    ///   yet, which ListSessions lists and never removes (accepted: the usual
    ///   ambiguity of a timed-out write);
    /// - the undo's own `SREM path_index` / `DEL`, landing late or never: an
    ///   inert `path_index` member or orphan metadata;
    /// - an in-flight `EXEC` (after the `SISMEMBER` read "absent", a released
    ///   undo, or a transient `EXEC` error frame): a ghost that the call's own
    ///   retry ignores and ListSessions removes.
    ///
    /// No metric, unlike [`Self::compensate_unwatched_create`]: no leftover is
    /// ever a second session sharing the name.
    #[allow(clippy::too_many_arguments)]
    async fn undo_create_attempt(
        &self,
        name: &str,
        base_path: &str,
        session_id: &str,
        path_index_key: &str,
        path_names_key: &str,
        session_meta_key: &str,
        name_member: &str,
        op: &str,
        cause: fred::error::Error,
    ) -> ServiceError {
        let (names_srem, index_srem, meta_del) = tokio::join!(
            self.client
                .pool
                .srem::<i64, _, _>(path_names_key, name_member),
            self.client
                .pool
                .srem::<i64, _, _>(path_index_key, session_id),
            self.client.pool.del::<i64, _>(session_meta_key),
        );

        let origin = FredErrOrigin::client(&self.client, POOL_LABEL_DATA);
        for e in [
            names_srem.as_ref().err(),
            index_srem.as_ref().err(),
            meta_del.as_ref().err(),
        ]
        .into_iter()
        .flatten()
        {
            origin.observe_error(e);
        }

        if names_srem.is_ok() {
            if index_srem.is_err() || meta_del.is_err() {
                tracing::warn!(
                    session_id = %session_id,
                    session_name = %name,
                    base_path = %base_path,
                    op = %op,
                    cause = %cause,
                    path_index_error = ?index_srem.err(),
                    session_meta_error = ?meta_del.err(),
                    "create_session failed after its commit was sent; the name was released but \
                     part of the undo failed. A transient failure retries under a new session_id; \
                     whatever the failed commands leave behind belongs to this abandoned \
                     session_id and stays unlisted"
                );
            }
            return map_transient_fred_err(origin, op, cause);
        }

        tracing::warn!(
            session_id = %session_id,
            session_name = %name,
            base_path = %base_path,
            op = %op,
            cause = %cause,
            path_names_error = ?names_srem.err(),
            path_index_error = ?index_srem.err(),
            session_meta_error = ?meta_del.err(),
            "create_session failed after its commit was sent and the undo could not release \
             the name. If the commit landed, the name stays claimed: the next ListSessions of \
             this base path removes the incomplete entry if it has no metadata, but if a \
             timed-out metadata write still reached the store, the session is complete and \
             stays"
        );
        map_fred_err_for_grpc(origin, op, cause)
    }

    /// Resolve a `create_once` whose `EXEC` reply never arrived (b.bhb).
    ///
    /// The names-set member is unique to this attempt (`name\0{fresh uuid}`), so
    /// one `SISMEMBER` answers "has my transaction run?" without guessing. Runs
    /// on the data pool — the txn connection is the one that just failed.
    ///
    /// When that `SISMEMBER`, or the re-scan after a landed commit, fails, the
    /// attempt is undone through [`Self::undo_create_attempt`] before anything
    /// is retried. The `SISMEMBER` answer holds only for the moment it was
    /// read; see [`Self::undo_create_attempt`] for a late-landing `EXEC`.
    #[allow(clippy::too_many_arguments)]
    async fn reconcile_in_doubt_create(
        &self,
        name: &str,
        base_path: &str,
        created_by: &str,
        session_id: &str,
        created_at: u64,
        path_index_key: &str,
        path_names_key: &str,
        session_meta_key: &str,
        name_member: &str,
        own_session_ids: &[String],
        cause: fred::error::Error,
    ) -> Result<String, ServiceError> {
        let landed: bool = match self
            .client
            .pool
            .sismember(path_names_key, name_member)
            .await
        {
            Ok(present) => present,
            Err(e) => {
                // Cannot tell whether the commit landed. A bare `Aborted` would
                // leave a landed commit behind as a ghost holding the name for
                // other calls, so undo the attempt first; see
                // `undo_create_attempt` for the verdict.
                tracing::warn!(
                    session_id = %session_id,
                    cause = %cause,
                    error = %e,
                    "create_session EXEC reply lost and the reconciling SISMEMBER also failed; \
                     undoing the attempt"
                );
                return Err(self
                    .undo_create_attempt(
                        name,
                        base_path,
                        session_id,
                        path_index_key,
                        path_names_key,
                        session_meta_key,
                        name_member,
                        "SISMEMBER path names index (in-doubt reconcile)",
                        e,
                    )
                    .await);
            }
        };

        if !landed {
            // The transaction had not run when the `SISMEMBER` read. Same
            // disposition as pre-b.bhb. A late `EXEC` of this attempt leaves
            // only a ghost that the retry ignores (see `undo_create_attempt`).
            return Err(ServiceError::Aborted(
                "create_session transaction canceled (connection drop), retry".to_owned(),
            ));
        }

        // It committed — but possibly unwatched, in which case the duplicate
        // scan above was not fenced. Re-scan now that the write is visible:
        // anything else claiming this name — any holder other than an id this
        // call created — is a uniqueness violation this attempt must not paper
        // over.
        //
        // A failed re-scan undoes the landed attempt first rather than leave it
        // as a ghost; see `undo_create_attempt` for the verdict.
        let members: Vec<String> = match self.client.pool.smembers(path_names_key).await {
            Ok(members) => members,
            Err(e) => {
                return Err(self
                    .undo_create_attempt(
                        name,
                        base_path,
                        session_id,
                        path_index_key,
                        path_names_key,
                        session_meta_key,
                        name_member,
                        "SMEMBERS path names index (in-doubt re-scan)",
                        e,
                    )
                    .await);
            }
        };
        // `own_session_ids` includes this attempt's `session_id`, so our own
        // landed member is skipped here too.
        let duplicate = members
            .iter()
            .any(|member| Self::names_a_competing_holder(member, name, own_session_ids));
        if duplicate {
            return match self
                .compensate_unwatched_create(
                    name,
                    session_id,
                    path_index_key,
                    path_names_key,
                    name_member,
                    // No fingerprint comparison is available on this path: the
                    // txn connection whose identity would be re-read was
                    // released when the reply was lost, and the conclusion here
                    // came from the keyspace, not from the L3 witness.
                    None,
                )
                .await
            {
                // Compensation removed our members, leaving the pre-existing
                // holder of the name: this attempt lost the race.
                Err(ServiceError::Aborted(_)) => Err(ServiceError::AlreadyExists(format!(
                    "Session '{name}' already exists"
                ))),
                other => other,
            };
        }

        tracing::warn!(
            session_id = %session_id,
            session_name = %name,
            cause = %cause,
            "create_session EXEC reply was lost but the commit landed and the name is still \
             unique; completing the create"
        );
        self.finish_create(
            name,
            base_path,
            created_by,
            session_id,
            created_at,
            path_index_key,
            path_names_key,
            session_meta_key,
            name_member,
        )
        .await
    }

    /// Delete a session's metadata, path-index entry, and layers index
    /// (SDD §7.8 step 3). Full cascade (layer cleanup, stream termination) is
    /// handled at the gRPC handler level.
    ///
    /// The three keys touched here span two Cluster slots — the path-index key
    /// (`le:path:{hex(base_path)}`) is hashed on `base_path`, while the
    /// session-meta and layers-index keys share the `{session_id}` hash tag.
    /// A single MULTI/EXEC across all three would fail with `CROSSSLOT` on
    /// Valkey Cluster, so we split the operation (mirroring `create`, b.6e7):
    ///
    /// 1. `HGETALL` session_meta to recover `base_path`.
    /// 2. `SREM` the path-index entry, then the names-index member, FIRST (on
    ///    the shared pool). `list()` and the `CreateSession(name=...)`
    ///    duplicate-name check both read the names index, so the session
    ///    disappears from both only once the second `SREM` has run.
    /// 3. MULTI { DEL session_meta, DEL layers_index, SET session_deleted_marker
    ///    EX [`SESSION_DELETED_MARKER_TTL_SECS`] } EXEC on the `{session_id}`
    ///    slot via `txn_pool.acquire()`. The marker keeps session recovery from
    ///    restoring this session from its checkpoint until the checkpoint
    ///    tombstone enqueued after the EXEC has landed
    ///    ([`KeySchema::session_deleted_marker`]).
    ///
    /// [`SESSION_DELETED_MARKER_TTL_SECS`]: crate::storage::keys::SESSION_DELETED_MARKER_TTL_SECS
    ///
    /// Atomicity contract: we preserve the invariant **"every member of
    /// `path_index(base_path)` and of `path_names_index(base_path)` points to
    /// readable `session_meta({sid})`"**. A crash between the two `SREM`s
    /// leaves the names member and its metadata in place, so `list()` still
    /// shows the session as normal and a `DeleteSession` retry completes it. A
    /// crash between (2) and (3) leaves orphan session_meta + layers_index on
    /// a session_id no longer in either index — those orphans are not
    /// surfaced by `list()` (which enumerates the names index), and a
    /// `DeleteSession` retry with the same `session_id` still reads the
    /// metadata, finds both `SREM`s already done, and deletes the orphans.
    /// The pre-b.yis ordering had the inverse failure mode (an index entry
    /// pointing to missing metadata, which `list` surfaces); the current
    /// ordering eliminates that class of inconsistency (b.yis sub-item 3).
    pub async fn delete(&self, session_id: &str) -> Result<(), ServiceError> {
        // Step 1: Read session metadata to recover base_path for the path-index SREM.
        let meta: HashMap<String, String> = self
            .client
            .pool
            .hgetall(self.keys.session_meta(session_id))
            .await
            .map_err(|e| {
                map_fred_err_for_grpc(
                    FredErrOrigin::client(&self.client, POOL_LABEL_DATA),
                    "HGETALL session meta (delete)",
                    e,
                )
            })?;

        if meta.is_empty() {
            return Err(ServiceError::NotFound(format!(
                "session {session_id} not found"
            )));
        }

        let base_path = meta.get("base_path").ok_or_else(|| {
            ServiceError::Internal("session metadata missing base_path".to_owned())
        })?;
        // `name` must be present on any session created post-fix; its absence
        // indicates an orphan written before `HSET session_meta` completed.
        // Fall back to an empty name — the names-set SREM will then be a
        // no-op (there was no matching composite member), which is the
        // correct outcome for an orphan.
        let name = meta.get("name").cloned().unwrap_or_default();

        // Step 2: SREM both path_index (session-id set) and path_names_index
        // (composite-name set) FIRST, outside the meta/layers-index MULTI.
        // Both keys share the `{hex(base_path)}` hash tag and live on a
        // different Cluster slot from session_meta / layers_index (tagged
        // `{session_id}`), so they cannot participate in the same MULTI/EXEC
        // as the DELs. Running them first ensures any partial-failure window
        // leaves an orphan session_meta (invisible to `list`) rather than an
        // index entry pointing at missing metadata (a names-index entry is
        // what `list` enumerates).
        //
        // The two SREMs share a Cluster slot with each other, so they could
        // run in a single MULTI; but keeping them as two independent calls
        // costs one extra round-trip in the success path while keeping the
        // failure analysis simple — a crash between the two leaves the names
        // member and its metadata in place, so `list` still shows the session
        // as normal and a `DeleteSession` retry finishes the removal.
        let path_index_key = self.keys.path_index(base_path);
        let path_names_key = self.keys.path_names_index(base_path);
        let name_member = KeySchema::path_name_member(&name, session_id);

        self.client
            .pool
            .srem::<(), _, _>(&path_index_key, session_id)
            .await
            .map_err(|e| {
                map_fred_err_for_grpc(
                    FredErrOrigin::client(&self.client, POOL_LABEL_DATA),
                    "SREM path index (delete)",
                    e,
                )
            })?;
        self.client
            .pool
            .srem::<(), _, _>(&path_names_key, &name_member)
            .await
            .map_err(|e| {
                map_fred_err_for_grpc(
                    FredErrOrigin::client(&self.client, POOL_LABEL_DATA),
                    "SREM path names index (delete)",
                    e,
                )
            })?;

        // Step 3: Atomic DEL of session_meta + layers_index, plus the deletion
        // marker, on the `{session_id}` slot via an exclusive txn-pool
        // connection (b.yis sub-item 3). All three keys share the
        // `{session_id}` hash tag per `KeySchema`, so the MULTI/EXEC is
        // slot-safe. Acquiring the
        // exclusive connection (instead of `pool.next().multi()`) matches
        // the `create` path's guarantee that WATCH/MULTI semantics are
        // isolated to one Valkey session, even though we don't WATCH here —
        // it also avoids interleaving two logical MULTI blocks on a shared
        // multiplexed connection if fred ever regresses that behavior.
        // The checkout carries the `valkey_pool_in_use{pool="txn"}` gauge
        // increment and the checkout permit inside itself, so both track the
        // actual slot-held interval by construction rather than by this scope
        // being written correctly (b.3z7 H-obs-3, b.f4f).
        // b.7td: every command below shares the `{session_id}` hash tag, so the
        // session-meta key names the one slot this transaction touches and the
        // checkout probe is pinned to it.
        let session_meta_key = self.keys.session_meta(session_id);
        let client = self.client.acquire_txn_for(&session_meta_key).await;
        let trx = client.multi();
        let _: () = trx
            .del(&session_meta_key)
            .await
            .map_err(|e| ServiceError::Internal(format!("DEL session meta queue failed: {e}")))?;
        let _: () = trx
            .del(self.keys.layers_index(session_id))
            .await
            .map_err(|e| ServiceError::Internal(format!("DEL layers index queue failed: {e}")))?;
        // The deletion marker, in the same transaction and on the same
        // `{session_id}` slot, so it exists from the instant the metadata is
        // gone. Session recovery refuses a marked session, which keeps a lookup
        // made before the checkpoint tombstone below lands from restoring the
        // session this call deleted (`KeySchema::session_deleted_marker`).
        let _: () = trx
            .set(
                self.keys.session_deleted_marker(session_id),
                "1",
                Some(fred::types::Expiration::EX(
                    crate::storage::keys::SESSION_DELETED_MARKER_TTL_SECS,
                )),
                None,
                false,
            )
            .await
            .map_err(|e| {
                ServiceError::Internal(format!("SET session deleted marker queue failed: {e}"))
            })?;

        // Test-only fault-injection for the `exec_txn` match below. Lets
        // the b.91j C-3 regression tests drive the `Value::Null` / canceled
        // arms deterministically without severing the real connection. A
        // queued fault overrides the real EXEC; absence falls through to
        // the production path. Gated behind `test-fault-injection` so
        // release builds never see this branch.
        #[cfg(any(test, feature = "test-fault-injection"))]
        let injected_exec_fault: Option<DeleteExecFault> = self
            .delete_exec_fault_injector_for_test
            .as_ref()
            .and_then(|q| q.lock().unwrap().pop_front());

        // Decode EXEC as `fred::types::Value` (mirroring the contract in
        // `create` at session.rs:123-131) so that a canceled transaction
        // surfaces as `Aborted` rather than being silently coerced to
        // `Ok(())` by `FromValue for ()`. DEL is naturally idempotent, so a
        // retry by the client is safe: re-issuing `delete(session_id)` after
        // a connection drop either completes the same logical cleanup or
        // returns `NotFound` from the HGETALL short-circuit if the earlier
        // attempt actually committed. Without this, a mid-EXEC drop would
        // be surfaced as `Internal` (UNKNOWN/INTERNAL gRPC status) and
        // clients would not retry, leaving orphan session_meta and/or
        // layers_index behind (b.91j C-3).
        let exec_result: Result<fred::types::Value, fred::error::Error> = {
            #[cfg(any(test, feature = "test-fault-injection"))]
            {
                match injected_exec_fault {
                    Some(DeleteExecFault::ReturnNull) => {
                        // Drop the queued transaction by dispatching a
                        // harmless DISCARD-style abandon: the builder
                        // happens to release its queued commands on drop,
                        // and we skip the real EXEC entirely.
                        drop(trx);
                        Ok(fred::types::Value::Null)
                    }
                    Some(DeleteExecFault::ReturnCanceled) => {
                        drop(trx);
                        Err(fred::error::Error::new(
                            fred::error::ErrorKind::Canceled,
                            "injected: transaction canceled (connection drop)",
                        ))
                    }
                    None => {
                        self.client
                            .exec_txn::<fred::types::Value>(
                                &client,
                                trx,
                                &session_meta_key,
                                "DeleteSession",
                                true,
                            )
                            .await
                    }
                }
            }
            #[cfg(not(any(test, feature = "test-fault-injection")))]
            {
                self.client
                    .exec_txn::<fred::types::Value>(
                        &client,
                        trx,
                        &session_meta_key,
                        "DeleteSession",
                        true,
                    )
                    .await
            }
        };

        match exec_result {
            Ok(v) if v.is_null() => {
                // `delete` does not WATCH any keys, so a Valkey-returned
                // null on EXEC is unexpected and most plausibly indicates
                // the underlying connection was dropped and fred surfaced
                // that as a null reply rather than a canceled error. Treat
                // it as retriable — the session has no WATCH set to have
                // raced against.
                tracing::warn!(
                    session_id = %session_id,
                    "DeleteSession EXEC returned null, treating as retriable connection drop"
                );
                return Err(ServiceError::Aborted(
                    "transaction reply null (connection drop), retry_on_valkey_drop".to_owned(),
                ));
            }
            Ok(_) => {}
            Err(e) if e.is_canceled() => {
                // fred raises `is_canceled()` on connection drop, oneshot-
                // channel close, or reconnect exhaustion — never on a
                // server-returned nil, which flows through the null arm
                // above. Surface as `Aborted`; no retry loop wraps `delete`,
                // so it escapes the handler and reaches the client as gRPC
                // `UNAVAILABLE` (`src/error.rs`), which clients retry (DEL is
                // idempotent).
                tracing::warn!(
                    session_id = %session_id,
                    "DeleteSession transaction canceled (connection or channel drop)"
                );
                return Err(ServiceError::Aborted(
                    "transaction canceled (connection drop), retry_on_valkey_drop".to_owned(),
                ));
            }
            Err(e) => {
                // b.bhb: classified rather than wrapped `Internal` directly.
                // `is_canceled()` above only covers ONE of the shapes a dropped
                // connection produces; a write that fails on the socket is
                // reported by `check_final_write_attempt` as `ErrorKind::IO`,
                // and a command that arrives while the connection is being
                // replaced fails routing as `Routing`/`Unknown` "Failed to route
                // command". With the txn pool at one attempt those are no longer
                // absorbed by fred, so they must land on the same disposition as
                // the canceled arm — retriable `Aborted` (DEL is idempotent).
                // `exec(true)` returns a queued command's own refusal as the
                // error (fred's `process_responses`), so the store's
                // not-serving `-LOADING` / `-BUSY` is `Aborted` the same way,
                // and a store write refusal (`-OOM` on the marker `SET`,
                // `-MISCONF`, …) is a plain `Unavailable` (b.xd1).
                //
                // A slot that moved or is moving reaches here as fred's
                // `Transaction` router giving up, "Max redirections exceeded"
                // (the txn pool allows no redirect): on Valkey 9.x the `EXEC`
                // itself is redirected, on ≤ 8.x and Redis 7.x a queued
                // command — the absent deletion marker's `SET` in a migrating
                // slot. No `EXEC` ran in that attempt, but fred's one replay
                // may follow a first send that landed, so the classifier
                // grades it transient (`Aborted`, which DEL's idempotence
                // makes safe to retry), never "nothing ran"
                // (`is_transaction_router_exhausted_error`, b.cyu). fred does
                // not re-sync this connection's routing on that redirect;
                // `exec_txn` above already did, before the error reached this
                // arm. Anything else is `Internal`.
                return Err(map_transient_fred_err(
                    FredErrOrigin::client(&self.client, POOL_LABEL_TXN),
                    "EXEC session DEL meta+layers",
                    e,
                ));
            }
        }

        tracing::info!(session_id = %session_id, "session deleted");

        // Enqueue the async-off-hot-path session-checkpoint delete. Reached
        // only on the success arm — `NotFound`, `Aborted`, `Unavailable`, and
        // `Internal` early-return paths do not enqueue. Ordering: we delete from live
        // Valkey first and only then schedule the checkpoint delete, so a
        // crash between the two leaves a stale checkpoint that the next
        // `CreateSession` on the same `session_id` would overwrite (low
        // risk: session_ids are UUID4). The alternative — deleting the
        // checkpoint first — would lose recoverability if the Valkey delete
        // failed partway through. Until this delete lands at the adapter, the
        // deletion marker the EXEC wrote is what keeps session recovery from
        // restoring the session from its still-present checkpoint.
        if let Some(writer) = &self.checkpoint_writer {
            writer.enqueue_session_checkpoint_delete(session_id.to_owned());
        }

        Ok(())
    }

    /// List all sessions under a base path (SDD §7.13).
    ///
    /// Iterates the `path_names_index` set (authoritative for session names
    /// and uniqueness, b.91j C-7) and pipelines `HGETALL session_meta` for
    /// each member. Returns a [`ListResult`] that surfaces both the live
    /// sessions and the **ghost members** (path-index entries whose
    /// `session_meta` HGETALL came back empty).
    ///
    /// **Recovery-aware contract (Epic `t1.fgj.7v`, SDD SR-2.7).** This call
    /// no longer opportunistically `SREM`s ghosts. With session-recovery
    /// (SR-2.7) in play, an empty HGETALL is ambiguous: it may indicate a
    /// truly-orphaned member (today's "selective key loss survivor" being
    /// the *cleanup* end of that contract) OR a recovery candidate whose
    /// session-checkpoint would restore the missing keys. The discrimination
    /// primitive between those two cases is
    /// [`crate::checkpoint::SessionRecoveryHook::try_recover_session`], which
    /// `storage::session` MUST NOT call directly (architectural rule: no new
    /// `storage → checkpoint` dependency). The caller — today
    /// [`crate::grpc::session::handle_list_sessions`] — orchestrates the
    /// recovery+SREM loop and decides whether each ghost is preserved
    /// (recovered) or SREM'd (NoCheckpoint / disabled hook).
    pub async fn list(&self, base_path: &str) -> Result<ListResult, ServiceError> {
        let path_names_key = self.keys.path_names_index(base_path);

        // Read all composite `name\0session_id` members from the names set.
        let members: Vec<String> =
            self.client
                .pool
                .smembers(&path_names_key)
                .await
                .map_err(|e| {
                    map_fred_err_for_grpc(
                        FredErrOrigin::client(&self.client, POOL_LABEL_DATA),
                        "SMEMBERS path names index (list)",
                        e,
                    )
                })?;

        if members.is_empty() {
            return Ok(ListResult {
                present: Vec::new(),
                ghosts: Vec::new(),
            });
        }

        // Split each member back into `(name, session_id)`. A malformed
        // member (no NUL separator) indicates data corruption or a foreign
        // writer — drop it and warn rather than failing the whole list.
        let mut pairs: Vec<(String, String)> = Vec::with_capacity(members.len());
        for member in &members {
            match KeySchema::split_path_name_member(member) {
                Some((name, sid)) => pairs.push((name.to_owned(), sid.to_owned())),
                None => {
                    tracing::warn!(
                        base_path = %base_path,
                        member = %member,
                        "path_names_index member missing NUL separator; skipping"
                    );
                }
            }
        }

        // Pipeline HGETALL for each session ID (SDD §7.6).
        let mut meta_futures = Vec::with_capacity(pairs.len());
        let origin = FredErrOrigin::client(&self.client, POOL_LABEL_DATA);
        for (name, sid) in &pairs {
            let pool = self.client.pool.clone();
            let key = self.keys.session_meta(sid);
            let name_clone = name.clone();
            let sid_clone = sid.clone();
            meta_futures.push(async move {
                let meta: HashMap<String, String> = pool
                    .hgetall(&key)
                    .await
                    .map_err(|e| map_fred_err_for_grpc(origin, "HGETALL session meta (list)", e))?;
                Ok::<_, ServiceError>((name_clone, sid_clone, meta))
            });
        }

        let results = futures::future::try_join_all(meta_futures).await?;

        let mut present = Vec::with_capacity(pairs.len());
        let mut ghosts: Vec<GhostMember> = Vec::new();

        for (name, sid, meta) in results {
            if meta.is_empty() {
                tracing::debug!(
                    session_id = %sid,
                    base_path = %base_path,
                    "path_names_index member with empty session_meta; surfacing as ghost for caller-side recovery probe"
                );
                let composite_member = KeySchema::path_name_member(&name, &sid);
                ghosts.push(GhostMember {
                    name,
                    sid,
                    composite_member,
                });
                continue;
            }

            // Prefer the canonical name from session_meta (an eventually-
            // consistent source of truth), falling back to the names-set
            // composite if meta is mid-write.
            let meta_name = meta.get("name").cloned().unwrap_or_else(|| name.clone());
            let bp = meta.get("base_path").cloned().unwrap_or_default();
            let created_by = meta.get("created_by").cloned().unwrap_or_default();
            let created_at = meta
                .get("created_at")
                .and_then(|s| s.parse::<u64>().ok())
                .unwrap_or(0);

            present.push(SessionMeta {
                session_id: sid,
                name: meta_name,
                base_path: bp,
                created_by,
                created_at,
            });
        }

        Ok(ListResult { present, ghosts })
    }

    /// Fetch a single session's metadata by `session_id`.
    ///
    /// Returns `Ok(Some(meta))` if `HGETALL le:{session_id}:sess` returned
    /// non-empty, `Ok(None)` if it returned empty (the session does not
    /// currently exist in Valkey), or `Err` on a Valkey backend failure.
    ///
    /// Extracted from the previously-inlined HGETALL inside [`Self::list`]
    /// so callers (notably
    /// [`crate::grpc::session::handle_list_sessions`]) have a clean retry
    /// primitive to re-issue after
    /// [`crate::checkpoint::SessionRecoveryHook::try_recover_session`]
    /// reports `Recovered` (Epic `t1.fgj.7v`, SDD SR-2.7).
    pub async fn get_session_meta(
        &self,
        session_id: &str,
    ) -> Result<Option<SessionMeta>, ServiceError> {
        let key = self.keys.session_meta(session_id);
        let meta: HashMap<String, String> = self.client.pool.hgetall(&key).await.map_err(|e| {
            map_fred_err_for_grpc(
                FredErrOrigin::client(&self.client, POOL_LABEL_DATA),
                "HGETALL session meta",
                e,
            )
        })?;

        if meta.is_empty() {
            return Ok(None);
        }

        let name = meta.get("name").cloned().unwrap_or_default();
        let base_path = meta.get("base_path").cloned().unwrap_or_default();
        let created_by = meta.get("created_by").cloned().unwrap_or_default();
        let created_at = meta
            .get("created_at")
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(0);

        Ok(Some(SessionMeta {
            session_id: session_id.to_owned(),
            name,
            base_path,
            created_by,
            created_at,
        }))
    }

    /// Look up a session's base_path by session_id.
    ///
    /// Used by gRPC handlers that receive only a session_id and need the
    /// base_path for authorization checks.
    pub async fn get_base_path(&self, session_id: &str) -> Result<String, ServiceError> {
        // Test-only: short-circuit with a queued error before the HGET.
        // Lets a test drive `acquire_session_cancel`'s branch for transient
        // (non-`NotFound`) storage errors.
        // Gated behind `test-fault-injection` so release builds never see
        // this branch.
        #[cfg(any(test, feature = "test-fault-injection"))]
        if let Some(queue) = self.get_base_path_fault_injector_for_test.as_ref() {
            if let Some(err) = queue.lock().unwrap().pop_front() {
                return Err(err);
            }
        }

        // b.fd9: HGET failures route through `map_fred_err_for_grpc` so a
        // transient Valkey blip surfaces as `ServiceError::Unavailable`
        // (Code::Unavailable, retriable) rather than `Internal` (non-
        // retriable). Every hot-path RPC that runs an auth check calls
        // through here via the session-base-path cache; a Valkey blip used
        // to flap every in-flight request to `Code::Internal`.
        let base_path: Option<String> = self
            .client
            .pool
            .hget(self.keys.session_meta(session_id), "base_path")
            .await
            .map_err(|e| {
                map_fred_err_for_grpc(
                    FredErrOrigin::client(&self.client, POOL_LABEL_DATA),
                    "HGET session base_path",
                    e,
                )
            })?;

        base_path.ok_or_else(|| ServiceError::NotFound(format!("session {session_id} not found")))
    }

    /// Whether `session_id` carries the deletion marker
    /// ([`KeySchema::session_deleted_marker`]) that [`Self::delete`] writes in
    /// the transaction that removes the session's metadata.
    ///
    /// Tells a session a `DeleteSession` removed apart from one merely missing
    /// from the live store (a keyspace loss). The two callers that cancel a
    /// session's token on a store `NotFound` — `acquire_session_cancel` and the
    /// `session_tokens` sweeper — read it to pick the cancel's cause, which
    /// decides whether the streams end as a deletion or retryably. A read
    /// failure is surfaced (`Unavailable` for a transient blip, by the uniform
    /// classifier); those callers fall back to the retryable cause on it.
    pub async fn deleted_marker_exists(&self, session_id: &str) -> Result<bool, ServiceError> {
        session_deleted_marker_exists(&self.client, &self.keys, session_id)
            .await
            .map_err(|e| {
                map_fred_err_for_grpc(
                    FredErrOrigin::client(&self.client, POOL_LABEL_DATA),
                    "EXISTS session deleted marker",
                    e,
                )
            })
    }
}

/// `EXISTS` on the session deletion marker
/// ([`KeySchema::session_deleted_marker`]), on the data pool: the one read
/// behind [`SessionStore::deleted_marker_exists`] and the session recovery
/// hook's deleted-session refusal
/// (`crate::checkpoint::SessionRecoveryHook::try_recover_session`, steps 4a
/// and 6a). Returns the raw fred error so each caller classifies and counts it
/// on its own error type; this function counts nothing.
pub(crate) async fn session_deleted_marker_exists(
    client: &ValkeyClient,
    keys: &KeySchema,
    session_id: &str,
) -> Result<bool, fred::error::Error> {
    client
        .pool
        .exists(keys.session_deleted_marker(session_id))
        .await
}

/// Validate a session name per PRD §3.1.1 and SDD §23.1.
///
/// Delegates to the shared validation logic in `domain::validation`.
pub fn validate_session_name(name: &str) -> Result<(), ServiceError> {
    crate::domain::validation::validate_session_name(name)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Rejection tests ─────────────────────────────────────────────

    #[test]
    fn validate_session_name_empty_string_returns_error() {
        let err = validate_session_name("").unwrap_err();
        assert!(matches!(err, ServiceError::InvalidArgument(_)));
    }

    #[test]
    fn validate_session_name_whitespace_only_returns_error() {
        let err = validate_session_name("   ").unwrap_err();
        assert!(matches!(err, ServiceError::InvalidArgument(_)));
    }

    #[test]
    fn validate_session_name_exceeds_255_chars_returns_error() {
        let long_name = "a".repeat(256);
        let err = validate_session_name(&long_name).unwrap_err();
        assert!(matches!(err, ServiceError::InvalidArgument(_)));
    }

    #[test]
    fn validate_session_name_null_byte_returns_error() {
        let err = validate_session_name("hello\0world").unwrap_err();
        assert!(matches!(err, ServiceError::InvalidArgument(_)));
    }

    #[test]
    fn validate_session_name_newline_returns_error() {
        let err = validate_session_name("hello\nworld").unwrap_err();
        assert!(matches!(err, ServiceError::InvalidArgument(_)));
    }

    #[test]
    fn validate_session_name_carriage_return_returns_error() {
        let err = validate_session_name("hello\rworld").unwrap_err();
        assert!(matches!(err, ServiceError::InvalidArgument(_)));
    }

    #[test]
    fn validate_session_name_tab_returns_error() {
        let err = validate_session_name("hello\tworld").unwrap_err();
        assert!(matches!(err, ServiceError::InvalidArgument(_)));
    }

    #[test]
    fn validate_session_name_bell_control_char_returns_error() {
        let err = validate_session_name("hello\x07world").unwrap_err();
        assert!(matches!(err, ServiceError::InvalidArgument(_)));
    }

    #[test]
    fn validate_session_name_double_dot_returns_error() {
        let err = validate_session_name("foo..bar").unwrap_err();
        assert!(matches!(err, ServiceError::InvalidArgument(_)));
    }

    #[test]
    fn validate_session_name_backslash_prefix_returns_error() {
        let err = validate_session_name("\\\\server\\share").unwrap_err();
        assert!(matches!(err, ServiceError::InvalidArgument(_)));
    }

    #[test]
    fn validate_session_name_double_slash_prefix_returns_error() {
        let err = validate_session_name("//server/share").unwrap_err();
        assert!(matches!(err, ServiceError::InvalidArgument(_)));
    }

    #[test]
    fn validate_session_name_windows_drive_uppercase_returns_error() {
        let err = validate_session_name("C:\\folder").unwrap_err();
        assert!(matches!(err, ServiceError::InvalidArgument(_)));
    }

    #[test]
    fn validate_session_name_windows_drive_lowercase_returns_error() {
        let err = validate_session_name("c:\\folder").unwrap_err();
        assert!(matches!(err, ServiceError::InvalidArgument(_)));
    }

    // ── Acceptance tests ────────────────────────────────────────────

    #[test]
    fn validate_session_name_normal_ascii_succeeds() {
        assert!(validate_session_name("my-session").is_ok());
    }

    #[test]
    fn validate_session_name_utf8_multibyte_succeeds() {
        assert!(validate_session_name("セッション").is_ok());
    }

    #[test]
    fn validate_session_name_exactly_255_chars_succeeds() {
        let name = "a".repeat(255);
        assert!(validate_session_name(&name).is_ok());
    }

    #[test]
    fn validate_session_name_single_slash_succeeds() {
        assert!(validate_session_name("foo/bar").is_ok());
    }

    #[test]
    fn validate_session_name_single_backslash_in_middle_succeeds() {
        assert!(validate_session_name("foo\\bar").is_ok());
    }

    #[test]
    fn validate_session_name_single_dot_succeeds() {
        assert!(validate_session_name("foo.bar").is_ok());
    }

    // ── names_a_competing_holder: which names members the duplicate scans
    //    count (b.62m) ────────────────────────────────────────────────────

    /// Ids this call "created", for the `names_a_competing_holder` cases.
    fn own_ids() -> Vec<String> {
        vec!["own-attempt-1".to_owned(), "own-attempt-2".to_owned()]
    }

    #[test]
    fn names_a_competing_holder_member_with_an_own_id_is_not_a_competitor() {
        let member = KeySchema::path_name_member("sess", "own-attempt-1");
        assert!(
            !SessionStore::names_a_competing_holder(&member, "sess", &own_ids()),
            "a member carrying an id this call created is at worst its own abandoned \
             attempt's ghost, never a competing holder"
        );
    }

    #[test]
    fn names_a_competing_holder_same_name_under_another_id_is_a_competitor() {
        let member = KeySchema::path_name_member("sess", "someone-else");
        assert!(
            SessionStore::names_a_competing_holder(&member, "sess", &own_ids()),
            "the same name under an id this call did not create is a duplicate"
        );
    }

    #[test]
    fn names_a_competing_holder_different_name_is_not_a_competitor() {
        let member = KeySchema::path_name_member("other-sess", "someone-else");
        assert!(
            !SessionStore::names_a_competing_holder(&member, "sess", &own_ids()),
            "a member holding a different name does not compete for this one"
        );
    }

    #[test]
    fn names_a_competing_holder_member_without_a_separator_is_not_a_competitor() {
        assert!(
            !SessionStore::names_a_competing_holder("sess", "sess", &own_ids()),
            "a malformed member with no NUL separator names no holder"
        );
    }

    // ─── Recovery-aware `list` + new `get_session_meta` helper ─────────────
    //
    // Subtask `t3.fgj.7v.fp.ka` (Epic `t1.fgj.7v`, SDD SR-2.7).
    //
    // These tests pin the new contract:
    //   - `list` returns `ListResult { present, ghosts }` and does NOT
    //     auto-SREM on empty HGETALL. The caller (today
    //     `crate::grpc::session::handle_list_sessions`) orchestrates the
    //     recovery-aware cleanup. The pre-recovery integration test
    //     `list_sessions_orphan_surfaced_as_ghost_not_removed`
    //     (`tests/integration/storage/session_test.rs`) covers the
    //     full end-to-end shape — these inline tests are scoped to the
    //     `GhostMember` payload's `composite_member` invariant and the
    //     new `get_session_meta` helper's Ok(Some)/Ok(None) split.

    use crate::storage::test_support::TestValkey;
    use fred::interfaces::{HashesInterface, SetsInterface};

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn list_returns_present_and_ghosts_with_pre_derived_composite_member() {
        let tv = TestValkey::setup().await;
        let ss = SessionStore::new(tv.client().clone(), tv.key_schema().clone());

        // Seed one live session and one synthetic ghost (path-indices
        // populated but no `le:{sid}:sess`).
        let sid_live = ss.create("live", "/stage", "alice").await.unwrap();
        let sid_ghost = format!("ghost-{}", uuid::Uuid::new_v4());

        let keys = tv.key_schema();
        let _: () = tv
            .client()
            .pool
            .sadd(keys.path_index("/stage"), sid_ghost.clone())
            .await
            .expect("SADD path_index");
        let composite = KeySchema::path_name_member("ghost-name", &sid_ghost);
        let _: () = tv
            .client()
            .pool
            .sadd(keys.path_names_index("/stage"), composite.clone())
            .await
            .expect("SADD path_names_index");

        let result = ss.list("/stage").await.expect("list");
        assert_eq!(result.present.len(), 1, "live session in `present`");
        assert_eq!(result.present[0].session_id, sid_live);

        assert_eq!(result.ghosts.len(), 1, "ghost surfaced");
        assert_eq!(result.ghosts[0].sid, sid_ghost);
        assert_eq!(result.ghosts[0].name, "ghost-name");
        assert_eq!(
            result.ghosts[0].composite_member, composite,
            "composite_member must be pre-derived so the caller does not re-derive"
        );

        // `list` MUST NOT have SREM'd either index — the caller owns
        // that decision (recovery-aware).
        let path_index_members: Vec<String> = tv
            .client()
            .pool
            .smembers(keys.path_index("/stage"))
            .await
            .expect("SMEMBERS");
        assert!(
            path_index_members.contains(&sid_ghost),
            "ghost MUST remain in path_index after list — caller owns SREM"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn get_session_meta_returns_some_for_live_session() {
        let tv = TestValkey::setup().await;
        let ss = SessionStore::new(tv.client().clone(), tv.key_schema().clone());

        let sid = ss.create("live", "/stage", "alice").await.unwrap();

        let meta = ss
            .get_session_meta(&sid)
            .await
            .expect("get_session_meta")
            .expect("session_meta present");
        assert_eq!(meta.session_id, sid);
        assert_eq!(meta.name, "live");
        assert_eq!(meta.base_path, "/stage");
        assert_eq!(meta.created_by, "alice");
        assert!(meta.created_at > 0);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn get_session_meta_returns_none_for_absent_session() {
        let tv = TestValkey::setup().await;
        let ss = SessionStore::new(tv.client().clone(), tv.key_schema().clone());

        let sid = format!("ghost-{}", uuid::Uuid::new_v4());
        let result = ss
            .get_session_meta(&sid)
            .await
            .expect("get_session_meta must succeed on absent session");
        assert!(
            result.is_none(),
            "get_session_meta MUST return Ok(None) on empty HGETALL (the post-recovery race signal)"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn get_session_meta_returns_some_after_external_hset() {
        // Pins the post-recovery re-read pattern: even when the session
        // was created by an external writer (the recovery applier),
        // `get_session_meta` resolves the hash fields verbatim.
        let tv = TestValkey::setup().await;
        let ss = SessionStore::new(tv.client().clone(), tv.key_schema().clone());

        let sid = format!("ext-{}", uuid::Uuid::new_v4());
        let key = tv.key_schema().session_meta(&sid);
        let _: () = tv
            .client()
            .pool
            .hset(
                &key,
                [
                    ("name", "ext-name"),
                    ("base_path", "/ext-stage"),
                    ("created_by", "ext-user"),
                    ("created_at", "123456"),
                ],
            )
            .await
            .expect("HSET");

        let meta = ss
            .get_session_meta(&sid)
            .await
            .expect("get_session_meta")
            .expect("session_meta present");
        assert_eq!(meta.session_id, sid);
        assert_eq!(meta.name, "ext-name");
        assert_eq!(meta.base_path, "/ext-stage");
        assert_eq!(meta.created_by, "ext-user");
        assert_eq!(meta.created_at, 123_456);
    }
}
