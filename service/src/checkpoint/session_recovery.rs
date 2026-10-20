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

//! Lazy on-demand session-state recovery (SDD SR-2.7 / Epic 2).
//!
//! [`SessionRecoveryHook::try_recover_session`] is the recovery primitive the
//! gRPC handler call sites in sibling Subtasks will use when a session lookup
//! (`EXISTS le:{sid}:sess`) returns `false`. The shape mirrors
//! [`crate::checkpoint::recovery::RecoveryHook::try_restore_layer`] (the
//! Epic-1 canonical reference): the two hooks live side-by-side and share an
//! admission-discipline shape, but key different scopes and call different
//! appliers:
//!
//! - [`crate::checkpoint::recovery::RecoveryHook`] — layer-state recovery,
//!   keyed by `u64 object_id`, calls
//!   [`crate::checkpoint::apply::apply_checkpoint`].
//! - [`SessionRecoveryHook`] (this module) — session-state recovery, keyed
//!   by `String session_id`, calls
//!   [`crate::checkpoint::session::apply_session_checkpoint`].
//!
//! ## Admission discipline (mirrored from SR-2.6)
//!
//! Per-`session_id` admission is enforced via a **K=1** `tokio::sync::Mutex`
//! kept in a `DashMap<String, Arc<AdmissionEntry>>`, parallel to (not shared
//! with) the layer-state admission map at
//! [`crate::checkpoint::recovery::RecoveryHook::admission_len`]. The two maps
//! protect different scopes and key types; sharing them would either weaken
//! the per-session K=1 guarantee or cause cross-scope contention. Map entries
//! are retired by a reference-counted RAII guard ([`AdmissionMapGuard`]) on
//! the last waiter's drop — see the Epic-1 module-level docstring in
//! `src/checkpoint/recovery.rs` for the full rationale (refcount semantics,
//! `remove_if` race guard against concurrent `get_or_create` increments).
//!
//! The [`AdmissionEntry`] and [`AdmissionMapGuard`] types here are
//! intentionally **duplicated** from `src/checkpoint/recovery.rs`. Extraction
//! into a generic helper module was considered but rejected for this Subtask:
//! the load-bearing layer-recovery path is already at ~1800 lines and is
//! covered by a large existing test surface; refactoring it to consume a
//! generic helper widens the change blast-radius beyond this Subtask's
//! "add session recovery" scope. The duplication is ~40 lines and is
//! cross-referenced via this docstring so future refactors can lift both
//! copies into one place.
//!
//! ## Outcome vocabulary harmonised with Epic 1
//!
//! [`SessionRecoveryOutcome`] uses `restored` / `not_recoverable` / `error`
//! for the same semantics as [`crate::checkpoint::RecoveryOutcome`]. The
//! `AlreadyPresent` outcome is a new value distinct to session recovery: it
//! ticks the `already_present` cell on the recovery metric and signals to
//! the caller that the underlying state was found on the pre-acquire (or
//! post-acquire) `EXISTS` check, with no apply attempt issued. Callers
//! handle `AlreadyPresent` and `Recovered` the same way — retry the
//! original lookup — but the metric distinction lets operators reason
//! about the proportion of admission-contended attempts vs. genuine
//! restores.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use dashmap::DashMap;
use fred::interfaces::KeysInterface;
use tokio::sync::Mutex;

use crate::checkpoint::session::apply_session_checkpoint;
use crate::checkpoint::session_payload::SessionCheckpoint;
use crate::checkpoint::store::{CheckpointError, CheckpointStore};
use crate::config::Config;
use crate::observability::metrics::SharedMetrics;
use crate::storage::client::{FredErrOrigin, ValkeyClient, POOL_LABEL_DATA};
use crate::storage::keys::KeySchema;

/// Outcome of a single [`SessionRecoveryHook::try_recover_session`] attempt.
///
/// `Recovered` — a checkpoint existed and the session was materialized. The
/// caller SHOULD retry its original `le:{sid}:sess` lookup.
///
/// `NoCheckpoint` — recovery cannot proceed because no checkpoint exists
/// for this `session_id`, or because the session carries a deletion marker
/// (it was deleted and its checkpoint may still await its tombstone), or
/// because checkpointing is disabled. The caller SHOULD surface `NotFound` to its
/// client (matches the Epic-1 layer hook's `NotRecoverable` arm under the
/// `Ok(None)` branch from the adapter).
///
/// `AlreadyPresent` — the pre-acquire (or post-acquire) `EXISTS
/// le:{sid}:sess` check returned `true`. Either a peer replica landed the
/// recovery first, or the previous admission-mutex holder on this replica
/// did. The caller SHOULD retry its original lookup; no apply pipeline ran.
///
/// `Error(_)` — a transient / permanent backend failure occurred. The
/// caller MAY retry later (transient) or SHOULD surface an internal error
/// to its client (permanent). The variant carries the classified
/// [`CheckpointError`] verbatim so the caller can branch on its kind.
#[derive(Debug)]
pub enum SessionRecoveryOutcome {
    Recovered,
    NoCheckpoint,
    AlreadyPresent,
    Error(CheckpointError),
}

/// Per-`session_id` admission entry: the K=1 mutex itself plus a refcount
/// tracking how many concurrent restorers currently hold a clone of the
/// `Arc`. The refcount is consulted by [`AdmissionMapGuard`] to defer map-
/// entry retirement until the last waiter drops out. Duplicated from the
/// `AdmissionEntry` in `src/checkpoint/recovery.rs` per the module-level
/// docstring's "duplication-with-cross-reference" decision.
struct AdmissionEntry {
    mutex: Mutex<()>,
    refs: AtomicUsize,
}

impl AdmissionEntry {
    fn new() -> Self {
        Self {
            mutex: Mutex::new(()),
            refs: AtomicUsize::new(0),
        }
    }
}

/// RAII guard that retires a per-`session_id` admission map entry when the
/// last in-flight restorer for that session drops out. Mirrors
/// `AdmissionMapGuard` in `src/checkpoint/recovery.rs` with `String` keying
/// rather than `u64`. The reference-counted retirement is load-bearing:
/// removing the entry on first drop would let a later attempt for the same
/// `session_id` mint a fresh `AdmissionEntry` that does NOT serialize
/// against a still-queued original waiter, defeating the K=1 invariant.
struct AdmissionMapGuard<'a> {
    map: &'a DashMap<String, Arc<AdmissionEntry>>,
    entry: Arc<AdmissionEntry>,
    session_id: String,
}

impl<'a> AdmissionMapGuard<'a> {
    fn new(
        map: &'a DashMap<String, Arc<AdmissionEntry>>,
        entry: Arc<AdmissionEntry>,
        session_id: String,
    ) -> Self {
        entry.refs.fetch_add(1, Ordering::AcqRel);
        Self {
            map,
            entry,
            session_id,
        }
    }
}

impl Drop for AdmissionMapGuard<'_> {
    fn drop(&mut self) {
        // Same drop semantics as `recovery.rs::AdmissionMapGuard::drop`: the
        // `remove_if` closure re-checks the refcount under the shard lock
        // so a concurrent `get_or_create` that just incremented `refs` does
        // not race the removal — preventing the leak failure mode where a
        // removed entry leaves a still-queued waiter unable to serialize
        // against the next attempt.
        let prev = self.entry.refs.fetch_sub(1, Ordering::AcqRel);
        if prev == 1 {
            self.map.remove_if(&self.session_id, |_k, v| {
                v.refs.load(Ordering::Acquire) == 0
            });
        }
    }
}

/// Session-state recovery hook. Constructed once at startup in `main.rs`
/// (wiring lands in the sibling cache + ListSessions Subtasks), shared as
/// `Arc<SessionRecoveryHook>` on the gRPC service. The per-`session_id`
/// admission map (`admission`) is K=1 and parallel to the K=1 layer
/// admission map at
/// [`crate::checkpoint::recovery::RecoveryHook::admission_len`].
pub struct SessionRecoveryHook {
    store: Arc<dyn CheckpointStore>,
    client: ValkeyClient,
    keys: KeySchema,
    /// Read on the recovery path for
    /// [`Config::checkpoint_recovery_fetch_budget`](crate::config::Config::checkpoint_recovery_fetch_budget)
    /// (b.gg7), which bounds the step-5 store read. Held as the whole `Config`
    /// rather than the resolved `Duration` so the budget is re-derived from the
    /// single expression on every call and cannot be snapshotted out of sync
    /// with the layer hook's copy.
    config: Arc<Config>,
    metrics: SharedMetrics,
    service_label: String,
    enabled: bool,
    /// Per-`session_id` admission map (K=1). Entries are created on first
    /// attempt and retired by [`AdmissionMapGuard`] on the last waiter's
    /// drop. Parallel to (NOT shared with) the layer-recovery admission
    /// map.
    admission: Arc<DashMap<String, Arc<AdmissionEntry>>>,
}

impl SessionRecoveryHook {
    /// Construct a new session-recovery hook.
    ///
    /// `enabled` is read from
    /// [`Config::checkpoint_enabled`](crate::config::Config::checkpoint_enabled);
    /// passing `false` makes every [`Self::try_recover_session`] call
    /// return [`SessionRecoveryOutcome::NoCheckpoint`] without touching the
    /// admission map, the store, or the metric (matching the Epic-1
    /// disabled-short-circuit shape).
    pub fn new(
        store: Arc<dyn CheckpointStore>,
        client: ValkeyClient,
        keys: KeySchema,
        config: Arc<Config>,
        metrics: SharedMetrics,
    ) -> Self {
        let service_label = config.service_label.clone();
        let enabled = config.checkpoint_enabled;
        Self {
            store,
            client,
            keys,
            config,
            metrics,
            service_label,
            enabled,
            admission: Arc::new(DashMap::new()),
        }
    }

    /// Current size of the admission map.
    ///
    /// Exposed for tests that assert map-entry retirement on the last
    /// waiter's drop. Also a defensible operator-introspection point — a
    /// future `/metrics` exporter could surface this as a gauge to flag a
    /// leak without coupling the metrics scraper to private fields.
    pub fn admission_len(&self) -> usize {
        self.admission.len()
    }

    /// Lazy on-demand session-state recovery for `session_id` (SR-2.7).
    ///
    /// Algorithm:
    ///
    /// 1. Disabled short-circuit when `!self.enabled` — return
    ///    [`SessionRecoveryOutcome::NoCheckpoint`] without side effects (no
    ///    admission churn, no store call, no metric tick).
    /// 2. Pre-acquire `EXISTS le:{sid}:sess`. `Ok(true)` →
    ///    [`SessionRecoveryOutcome::AlreadyPresent`] (no admission entry
    ///    created — load-bearing for the no-regression guarantee that
    ///    sessions that exist on cold lookup pay zero recovery overhead).
    /// 3. Acquire per-`session_id` admission: get-or-insert
    ///    `Arc<AdmissionEntry>`, increment refcount via
    ///    [`AdmissionMapGuard::new`], lock its K=1 mutex.
    /// 4. Post-acquire `EXISTS le:{sid}:sess` re-check. `Ok(true)` →
    ///    [`SessionRecoveryOutcome::AlreadyPresent`] (a peer replica or
    ///    the previous mutex holder on this replica landed the recovery
    ///    while we were queued).
    ///
    ///    Step 4a: `EXISTS le:{sid}:sess:deleted`
    ///    ([`KeySchema::session_deleted_marker`]). Present →
    ///    [`SessionRecoveryOutcome::NoCheckpoint`]: `SessionStore::delete`
    ///    writes the marker atomically with the session's metadata DEL and
    ///    enqueues the checkpoint tombstone only afterwards, off the hot path,
    ///    so without this check a lookup in between restores the session that
    ///    was just deleted. A failed read → [`SessionRecoveryOutcome::Error`].
    /// 5. Fetch
    ///    [`CheckpointStore::get_session_checkpoint`](crate::checkpoint::CheckpointStore::get_session_checkpoint),
    ///    under a
    ///    [`Config::checkpoint_recovery_fetch_budget`](crate::config::Config::checkpoint_recovery_fetch_budget)
    ///    timeout (b.gg7 — `JoinSession` is deadline-exempt, so nothing else on
    ///    this path bounds the adapter, and the admission mutex is held across
    ///    the read). `Ok(None)` → [`SessionRecoveryOutcome::NoCheckpoint`].
    ///    `Err(_)` (including an elapse, as `Transient`) →
    ///    [`SessionRecoveryOutcome::Error`].
    /// 6. Deserialize via [`SessionCheckpoint::from_bytes`]. Parse failure
    ///    surfaces as [`SessionRecoveryOutcome::Error`] wrapping a
    ///    [`CheckpointError::Permanent`]. The error message cites only the
    ///    `session_id` and a generic "deserialize failed" descriptor —
    ///    the raw blob bytes are never embedded into the error string
    ///    (mirrors CLAUDE.md's auth-redaction discipline; a malformed
    ///    blob may carry sensitive content that operator logs should
    ///    not retain).
    ///
    ///    Step 6a: re-check the deletion marker, as in step 4a. The admission
    ///    mutex is per replica, so while this replica fetched, another could
    ///    restore the session and a `DeleteSession` delete (and mark) it
    ///    again. The re-check narrows that race to the apply below; it does
    ///    not close it.
    /// 7. Apply via [`apply_session_checkpoint`]. `Ok(())` →
    ///    [`SessionRecoveryOutcome::Recovered`]. `Err(_)` →
    ///    [`SessionRecoveryOutcome::Error`].
    /// 8. Admission entry retired by [`AdmissionMapGuard::drop`] when the
    ///    last in-flight attempt for this `session_id` exits.
    pub async fn try_recover_session(&self, session_id: &str) -> SessionRecoveryOutcome {
        // Step 1: disabled short-circuit. No admission, no store call, no
        // metric — an operator who disabled checkpointing observes zero
        // recovery activity. Returning `NoCheckpoint` (vs. a dedicated
        // `Disabled` variant) keeps the caller branch identical to the
        // genuinely-never-checkpointed path.
        if !self.enabled {
            return SessionRecoveryOutcome::NoCheckpoint;
        }

        // Step 2: pre-acquire EXISTS le:{sid}:sess. Load-bearing
        // optimisation: if the session is already present (the common
        // case after a peer replica recovered it, or simply the first
        // post-startup access landing on an unaffected session), short-
        // circuit BEFORE creating an admission entry. The Subtask body
        // explicitly requires this path to not churn the admission map.
        match self.is_session_present(session_id).await {
            Ok(true) => {
                self.metrics.inc_checkpoint_recovery_attempt(
                    "already_present",
                    "session",
                    &self.service_label,
                );
                return SessionRecoveryOutcome::AlreadyPresent;
            }
            Ok(false) => {
                // Fall through to admission acquire.
            }
            Err(err) => {
                self.metrics.inc_checkpoint_recovery_attempt(
                    "error",
                    "session",
                    &self.service_label,
                );
                return SessionRecoveryOutcome::Error(err);
            }
        }

        // Step 3: per-`session_id` admission. `entry()` on the DashMap is
        // get-or-insert; we clone the `Arc<AdmissionEntry>` out of the
        // entry guard so we drop the dashmap shard lock before acquiring
        // the mutex (holding both would deadlock against a concurrent
        // `remove_if` from another thread's `AdmissionMapGuard::drop`).
        let entry = self
            .admission
            .entry(session_id.to_string())
            .or_insert_with(|| Arc::new(AdmissionEntry::new()))
            .clone();
        let _map_guard =
            AdmissionMapGuard::new(&self.admission, entry.clone(), session_id.to_string());
        let _mutex_guard = entry.mutex.lock().await;

        // Step 4: post-acquire EXISTS re-check. While we were queued at
        // the mutex, a previous holder on this replica may have
        // completed the recovery for this session. Short-circuit with
        // `AlreadyPresent` rather than re-running the full pipeline.
        match self.is_session_present(session_id).await {
            Ok(true) => {
                self.metrics.inc_checkpoint_recovery_attempt(
                    "already_present",
                    "session",
                    &self.service_label,
                );
                return SessionRecoveryOutcome::AlreadyPresent;
            }
            Ok(false) => {
                // Fall through to the deletion-marker check.
            }
            Err(err) => {
                self.metrics.inc_checkpoint_recovery_attempt(
                    "error",
                    "session",
                    &self.service_label,
                );
                return SessionRecoveryOutcome::Error(err);
            }
        }

        // Step 4a: refuse a deleted session. `SessionStore::delete` writes the
        // marker in the transaction that deletes `le:{sid}:sess`, and only then
        // enqueues the checkpoint tombstone, off the hot path. So a session
        // step 4 found absent because it was deleted carries the marker, while
        // its checkpoint may still be at the adapter: fetching and applying it
        // would restore the session the client just deleted. Checked before the
        // fetch so a deleted session costs no adapter read.
        if let Some(outcome) = self.refuse_if_deleted(session_id).await {
            return outcome;
        }

        // Step 5: fetch the session-checkpoint blob, bounded at the caller's
        // own tolerance (b.gg7).
        //
        // The layer twin of this bound lives at the equivalent step of
        // `RecoveryHook::try_restore_layer`, and the argument is the same one
        // twice: `JoinSession` is the *other* deadline-exempt RPC
        // (`EXEMPT_FROM_DEADLINE` in `src/grpc/metrics_layer.rs` lists exactly
        // `JoinSession` and `SubscribeReadLayer`), it reaches this hook lazily
        // through `SessionBasePathCache::get_base_path`, and the tower deadline
        // layer that bounds every unary handler deliberately does not apply. So
        // on this path nothing else bounds the adapter — whose own enforced
        // ceiling is 180 s per call and 540 s per enumeration — and the K=1
        // admission mutex acquired above means the cost is not paid by one
        // joiner but by every joiner of the same session, in series.
        //
        // Same budget expression as the layer hook
        // (`Config::checkpoint_recovery_fetch_budget`) even though this is a
        // single GET rather than a three-request sequence. The budget's
        // `2 × request_timeout` term is sized for the layer hook's multi-MB
        // OMPF GET, so it is generous here — deliberately: it is a *ceiling* on
        // pathology, not a latency target, and two hooks bounding the same
        // hazard with two different numbers is how one of them silently rots.
        //
        // Preemptive rather than cooperative for the same reason: a pure read
        // (one GET, no PUT, no DELETE) has no partial write to roll back, so
        // dropping the future cancels an HTTP request and nothing else. An
        // elapse lands on the existing `Error` arm, so the `error` cell of
        // `checkpoint_recovery_attempt` and every caller's handling of a failed
        // recovery are unchanged.
        //
        // NOT a restore-SELECTION site, and structurally cannot become one:
        // the session-checkpoint surface is keyed by `session_id` alone, with
        // no generations and no `CheckpointKind`, so SR-2.5's newest-base-wins
        // rule has nothing to select over here. The single production
        // restore-selection site is `RecoveryHook::try_restore_layer`'s
        // `list_layer_checkpoints` → `get_layer_checkpoint` step in
        // `checkpoint/recovery.rs`. Recorded in place so "did we catch every
        // restore path?" is answerable without re-deriving it.
        let fetch_budget = self.config.checkpoint_recovery_fetch_budget();
        let fetch =
            match tokio::time::timeout(fetch_budget, self.store.get_session_checkpoint(session_id))
                .await
            {
                Ok(result) => result,
                Err(_elapsed) => {
                    tracing::warn!(
                        session_id,
                        budget_ms = fetch_budget.as_millis() as u64,
                        "session recovery abandoned its checkpoint fetch at the recovery fetch \
                     budget; the adapter's own ceiling is far longer than any caller waits, and \
                     holding the per-session admission mutex meanwhile would queue every other \
                     joiner of this session behind it"
                    );
                    Err(CheckpointError::Transient {
                        source: format!(
                            "get_session_checkpoint exceeded the {}ms recovery fetch budget",
                            fetch_budget.as_millis()
                        )
                        .into(),
                    })
                }
            };
        let blob = match fetch {
            Ok(Some(blob)) => blob,
            Ok(None) => {
                self.metrics.inc_checkpoint_recovery_attempt(
                    "not_recoverable",
                    "session",
                    &self.service_label,
                );
                return SessionRecoveryOutcome::NoCheckpoint;
            }
            Err(err) => {
                self.metrics.inc_checkpoint_recovery_attempt(
                    "error",
                    "session",
                    &self.service_label,
                );
                return SessionRecoveryOutcome::Error(err);
            }
        };

        // Step 6: deserialize. Permanent failure on parse — the blob
        // will not heal on retry; surface to the operator. CRITICALLY,
        // the error message MUST NOT echo blob bytes: a malformed blob
        // may carry sensitive payload, and operator-log redaction is a
        // load-bearing CLAUDE.md auth-discipline invariant. We cite the
        // `session_id` and a generic descriptor only.
        let checkpoint = match SessionCheckpoint::from_bytes(&blob.bytes) {
            Ok(c) => c,
            Err(_) => {
                self.metrics.inc_checkpoint_recovery_attempt(
                    "error",
                    "session",
                    &self.service_label,
                );
                return SessionRecoveryOutcome::Error(CheckpointError::Permanent {
                    source: format!(
                        "session-checkpoint deserialize failed for session_id={session_id}"
                    )
                    .into(),
                });
            }
        };

        // Step 6a: re-check the deletion marker. The admission mutex serializes
        // recoveries on this replica only: during the fetch above, another
        // replica can restore the session and a `DeleteSession` can then delete
        // it again, marking it. Applying now would restore it a second time.
        // This narrows that race to the apply itself; it does not close it (the
        // marker is not read inside the apply's transactions).
        if let Some(outcome) = self.refuse_if_deleted(session_id).await {
            return outcome;
        }

        // Step 7: apply. `apply_session_checkpoint` is the Task 5 applier
        // — it splits into the two-step path-index-first / session-scoped-
        // second protocol per SR-2.7 and classifies all backend failures
        // as `CheckpointError::Transient`.
        match apply_session_checkpoint(&self.client, &self.keys, session_id, &checkpoint).await {
            Ok(()) => {
                self.metrics.inc_checkpoint_recovery_attempt(
                    "restored",
                    "session",
                    &self.service_label,
                );
                SessionRecoveryOutcome::Recovered
            }
            Err(err) => {
                self.metrics.inc_checkpoint_recovery_attempt(
                    "error",
                    "session",
                    &self.service_label,
                );
                SessionRecoveryOutcome::Error(err)
            }
        }

        // Step 8: admission entry retired implicitly by the
        // `AdmissionMapGuard::drop` when this function's `_map_guard`
        // local goes out of scope.
    }

    /// Probe whether `le:{session_id}:sess` is currently present.
    ///
    /// Returns `Ok(true)` if the session hash exists, `Ok(false)`
    /// otherwise. Surfaces transient Valkey errors as
    /// [`CheckpointError::Transient`] — same classification as the
    /// Epic-1 hook's `is_layer_present` helper.
    ///
    /// **Deliberately still a bare `EXISTS`** (b.f4t). Its layer-side sibling
    /// moved to a structural predicate because `obj:hdr` has one: two live read
    /// paths refuse a header missing `seq` / `created_by` and route that refusal
    /// straight back into the recovery hook, so a presence check weaker than its
    /// consumers reported "already restored" for a restore that never ran.
    /// `le:{sid}:sess` has no analogous contract — no read path names a required
    /// field of it — so there is nothing for a structural predicate here to be
    /// at least as strict as, and inventing a field list would refuse sessions
    /// the service is perfectly willing to serve.
    async fn is_session_present(&self, session_id: &str) -> Result<bool, CheckpointError> {
        let session_meta_key = self.keys.session_meta(session_id);
        let exists: bool = self
            .client
            .pool
            .exists(&session_meta_key)
            .await
            .inspect_err(|e| FredErrOrigin::client(&self.client, POOL_LABEL_DATA).observe_error(e))
            .map_err(|e| CheckpointError::Transient {
                source: format!(
                    "EXISTS sess failed during session recovery for session_id={session_id}: {e}"
                )
                .into(),
            })?;
        Ok(exists)
    }

    /// The deletion-marker check behind steps 4a and 6a of
    /// [`Self::try_recover_session`].
    ///
    /// `Some(NoCheckpoint)` when `le:{sid}:sess:deleted`
    /// ([`KeySchema::session_deleted_marker`]) is present: the session was
    /// deleted, and its checkpoint, if still at the adapter, is awaiting its
    /// tombstone. Counted in the `not_recoverable` cell, the one a missing
    /// checkpoint counts in, because to the caller the two are the same answer.
    /// `Some(Error)` on a failed read, counted as `error`: a marker that could
    /// not be read is no licence to restore. `None` when the marker is absent.
    async fn refuse_if_deleted(&self, session_id: &str) -> Option<SessionRecoveryOutcome> {
        let marked: Result<bool, CheckpointError> =
            crate::storage::session::session_deleted_marker_exists(
                &self.client,
                &self.keys,
                session_id,
            )
            .await
            .inspect_err(|e| FredErrOrigin::client(&self.client, POOL_LABEL_DATA).observe_error(e))
            .map_err(|e| CheckpointError::Transient {
                source: format!(
                    "EXISTS session deletion marker failed during session recovery for \
                     session_id={session_id}: {e}"
                )
                .into(),
            });
        match marked {
            Ok(false) => None,
            Ok(true) => {
                tracing::debug!(
                    session_id,
                    "session recovery refused: the session was deleted (deletion marker present)"
                );
                self.metrics.inc_checkpoint_recovery_attempt(
                    "not_recoverable",
                    "session",
                    &self.service_label,
                );
                Some(SessionRecoveryOutcome::NoCheckpoint)
            }
            Err(err) => {
                self.metrics.inc_checkpoint_recovery_attempt(
                    "error",
                    "session",
                    &self.service_label,
                );
                Some(SessionRecoveryOutcome::Error(err))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::checkpoint::adapters::in_memory::InMemoryCheckpointStore;
    use crate::checkpoint::store::{CheckpointBlob, CheckpointHandle, CheckpointMeta};
    use crate::config::Config;
    use crate::observability::metrics::metrics_scrape_only;
    use crate::storage::keys::hex_encode;
    use crate::storage::test_support::TestValkey;
    use async_trait::async_trait;
    use bytes::Bytes;
    use fred::interfaces::{HashesInterface, SetsInterface};
    use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
    use std::sync::Mutex as StdMutex;
    use std::time::Duration;

    const TEST_SERVICE_LABEL: &str = "test-replica";

    fn test_config(enabled: bool) -> Config {
        Config {
            catchup_snapshot_enabled: false,
            port: 0,
            metrics_port: 0,
            valkey_url: String::new(),
            valkey_pool_size: 1,
            subscriber_pool_size: 1,
            subscriber_admission_wait_ms: 2_000,
            subscriber_max_subscriptions: 4096,
            subscriber_max_concurrent_catchups: 64,
            subscriber_admission_max_waiters: None,
            maintenance_pool_size: 1,
            catchup_snapshot_read_concurrency:
                crate::config::CATCHUP_SNAPSHOT_READ_CONCURRENCY_DEFAULT,
            catchup_fenced_read_concurrency: crate::config::CATCHUP_FENCED_READ_CONCURRENCY_DEFAULT,
            catchup_build_memory_budget_mib: crate::config::CATCHUP_BUILD_MEMORY_BUDGET_MIB_DEFAULT,
            snapshot_pool_size: 1,
            snapshot_atomic_read_max_items: crate::config::SNAPSHOT_ATOMIC_READ_MAX_ITEMS_DEFAULT,
            snapshot_atomic_read_max_bytes: crate::config::SNAPSHOT_ATOMIC_READ_MAX_BYTES_DEFAULT,
            structural_cache_max_mib: crate::config::STRUCTURAL_CACHE_MAX_MIB_DEFAULT,
            structural_cache_audit_rate: crate::config::STRUCTURAL_CACHE_AUDIT_RATE_DEFAULT,
            structural_cache_audit_above_ceiling: false,
            valkey_connect_timeout_secs: 5,
            valkey_command_timeout_ms: 5000,
            valkey_reconnect_max_delay_ms: 30_000,
            valkey_auth_mode: crate::config::ValkeyAuthMode::Static,
            valkey_token_file: None,
            valkey_token_username: None,
            valkey_token_refresh_lead_secs: 180,
            valkey_iam_username: None,
            valkey_iam_region: None,
            valkey_iam_cluster_host: None,
            valkey_iam_service: crate::config::ValkeyIamService::MemoryDb,
            storage_service_endpoint: None,
            auth_timeout_ms: 1000,
            auth_cache_ttl_secs: 60,
            auth_cache_negative_ttl_secs: 30,
            auth_userinfo_negative_ttl_secs: 5,
            auth_cache_max_entries: 1_000,
            userinfo_endpoint: None,
            max_layer_size: 1_048_576,
            upload_memory_budget_mib:
                crate::grpc::upload_admission::UPLOAD_MEMORY_BUDGET_MIB_DEFAULT,
            upload_admission_wait_ms:
                crate::grpc::upload_admission::UPLOAD_ADMISSION_WAIT_MS_DEFAULT,
            upload_admission_max_waiters:
                crate::grpc::upload_admission::UPLOAD_ADMISSION_MAX_WAITERS_DEFAULT,
            layer_stream_maxlen: 4096,
            layer_stream_max_bytes: crate::config::LAYER_STREAM_MAX_BYTES_DEFAULT,
            presence_broadcast_capacity: 4096,
            presence_lease_ttl_secs: crate::presence::DEFAULT_PRESENCE_LEASE_TTL_SECS,
            presence_refresh_interval_secs: crate::presence::DEFAULT_PRESENCE_REFRESH_INTERVAL_SECS,
            presence_keepalive_interval_ms: crate::presence::DEFAULT_PRESENCE_KEEPALIVE_INTERVAL_MS,
            shutdown_timeout_secs: 1,
            shutdown_overall_budget_secs: 5,
            shutdown_stream_end_rate: 25,
            shutdown_stream_end_max_window_secs: 20,
            log_level: "warn".to_string(),
            log_format: crate::config::LogFormat::Json,
            trace_filter: crate::config::DEFAULT_TRACE_FILTER.to_string(),
            sweeper_interval_secs: 15,
            session_list_timeout_ms: 1000,
            rpc_deadline_ms: 5000,
            service_label: TEST_SERVICE_LABEL.to_string(),
            valkey_memory_bytes: None,
            ext_release_max_bytes: None,
            ext_reconcile_interval_secs: 3600,
            auth_max_attempts: 3,
            auth_retry_initial_delay_ms: 100,
            authn_enabled: true,
            authz_enabled: false,
            userinfo_service_enabled: true,
            test_only_dangerously_allow_unvalidated_tokens: false,
            health_probe_interval_ms: 5_000,
            health_probe_timeout_ms: 2_000,
            auth_protocol: crate::config::AuthProtocol::Rest,
            update_layer_retry_max_attempts: 128,
            update_layer_per_object_concurrency: 16,
            update_layer_backend: crate::config::UpdateLayerBackend::Lua,
            ext_value_verify: crate::config::ExtValueVerifyMode::Enforce,
            session_retry_max_attempts: 32,
            session_retry_initial_delay_ms: 1,
            pod_memory: crate::config::PodMemoryPlan::reference(),
            session_tokens_sweep_interval_secs:
                crate::grpc::session::SESSION_TOKENS_SWEEP_INTERVAL_SECS,
            checkpoint_enabled: enabled,
            checkpoint_adapter: crate::checkpoint::CheckpointAdapter::InMemory,
            checkpoint_interval_secs: 60,
            checkpoint_quiesce_secs: 30,
            checkpoint_tick_secs: 10,
            checkpoint_retain_generations: 2,
            checkpoint_seq_safety_margin: 1000,
            checkpoint_counters_sync_secs: 60,
            checkpoint_reconcile_interval_secs: 3600,
            checkpoint_counter_safety_margin: 1000,
            checkpoint_tail_enabled: false,
            checkpoint_tail_max_chain: 8,
            checkpoint_tail_max_entries: 4096,
            checkpoint_tail_max_bytes: 67_108_864,
            checkpoint_local_disk_path: None,
            checkpoint_storage_api_endpoint: None,
            checkpoint_base_path: None,
            checkpoint_connect_timeout_ms: 5_000,
            checkpoint_request_timeout_ms: 30_000,
            checkpoint_ext_mirror_workers: 2,
            checkpoint_ext_mirror_queue_capacity: 1_000,
            service_token_file: None,
            service_token_poll_interval_secs: 15,
            service_token_client_credentials_enabled: false,
        }
    }

    fn make_metrics() -> SharedMetrics {
        Arc::new(metrics_scrape_only(TEST_SERVICE_LABEL))
    }

    /// Probe the live label-cell counter value for a given outcome at
    /// `scope="session"` (this module's hook), read from the rendered scrape
    /// exposition. The `result × scope × service` grid is pre-initialised in
    /// `Metrics::new`, so every cell is present from the first render.
    fn metric_value(metrics: &SharedMetrics, result: &str) -> u64 {
        use crate::observability::metrics::test_util::{render, sample_value};
        sample_value(
            &render(metrics),
            "live_edit_checkpoint_recovery_attempts_total",
            &[
                ("result", result),
                ("scope", "session"),
                ("service", TEST_SERVICE_LABEL),
            ],
        )
        .unwrap_or(0.0) as u64
    }

    /// Build a representative `SessionCheckpoint` for a given `session_id`.
    /// Mirrors `sample_apply_checkpoint` from `session.rs` tests, with a
    /// deterministic shape so apply-side assertions land predictably.
    fn sample_checkpoint(session_id: &str) -> SessionCheckpoint {
        let base_path = "/stage/recovery".to_string();
        let mut sc = SessionCheckpoint {
            schema_version: crate::checkpoint::session_payload::SESSION_CHECKPOINT_SCHEMA_VERSION,
            session_id: session_id.to_string(),
            name: "recover-stage".to_string(),
            base_path,
            created_by: "recover-user".to_string(),
            created_at: 1_700_000_000_000_000,
            layer_mappings: BTreeMap::from([("root.usda".to_string(), 9001_u64)]),
            layers_index: BTreeSet::from(["root.usda".to_string()]),
            path_name_members: BTreeSet::new(),
            path_index_session_id_member: session_id.to_string(),
        };
        sc.add_path_name_member("recover-stage", session_id);
        sc
    }

    /// Counting wrapper around `InMemoryCheckpointStore`. Records every
    /// invocation so the disabled / pre-acquire short-circuit tests can
    /// prove ZERO adapter calls happened on those paths.
    struct CountingCheckpointStore {
        inner: InMemoryCheckpointStore,
        calls: StdMutex<Vec<&'static str>>,
    }

    impl CountingCheckpointStore {
        fn new() -> Self {
            Self {
                inner: InMemoryCheckpointStore::new(),
                calls: StdMutex::new(Vec::new()),
            }
        }

        fn record(&self, method: &'static str) {
            self.calls.lock().unwrap().push(method);
        }

        fn call_count(&self) -> usize {
            self.calls.lock().unwrap().len()
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
            self.record("put_layer_checkpoint");
            self.inner
                .put_layer_checkpoint(object_id, snap_seq, ompf, meta)
                .await
        }

        async fn get_latest_layer_checkpoint(
            &self,
            object_id: u64,
        ) -> Result<Option<CheckpointBlob>, CheckpointError> {
            self.record("get_latest_layer_checkpoint");
            self.inner.get_latest_layer_checkpoint(object_id).await
        }

        // Forwarded per `CheckpointStore::get_layer_checkpoint`'s MUST — its
        // trait default returns `NotImplemented`, which would both mask a
        // generation the inner store holds and leave the call invisible to
        // `record()`, silently weakening the "which methods were called"
        // assertions.
        async fn get_layer_checkpoint(
            &self,
            handle: &CheckpointHandle,
        ) -> Result<Option<CheckpointBlob>, CheckpointError> {
            self.record("get_layer_checkpoint");
            self.inner.get_layer_checkpoint(handle).await
        }

        async fn list_layer_checkpoints(
            &self,
            object_id: u64,
        ) -> Result<Vec<CheckpointHandle>, CheckpointError> {
            self.record("list_layer_checkpoints");
            self.inner.list_layer_checkpoints(object_id).await
        }

        async fn delete_layer_checkpoint(
            &self,
            handle: CheckpointHandle,
        ) -> Result<(), CheckpointError> {
            self.record("delete_layer_checkpoint");
            self.inner.delete_layer_checkpoint(handle).await
        }

        async fn put_session_checkpoint(
            &self,
            session_id: &str,
            body: Bytes,
            meta: CheckpointMeta,
        ) -> Result<CheckpointHandle, CheckpointError> {
            self.record("put_session_checkpoint");
            self.inner
                .put_session_checkpoint(session_id, body, meta)
                .await
        }

        async fn get_session_checkpoint(
            &self,
            session_id: &str,
        ) -> Result<Option<CheckpointBlob>, CheckpointError> {
            self.record("get_session_checkpoint");
            self.inner.get_session_checkpoint(session_id).await
        }

        async fn delete_session_checkpoint(&self, session_id: &str) -> Result<(), CheckpointError> {
            self.record("delete_session_checkpoint");
            self.inner.delete_session_checkpoint(session_id).await
        }

        async fn list_session_checkpoint_ids(&self) -> Result<Vec<String>, CheckpointError> {
            self.record("list_session_checkpoint_ids");
            self.inner.list_session_checkpoint_ids().await
        }

        async fn put_ext_blob(&self, hash_hex: &str, bytes: Bytes) -> Result<(), CheckpointError> {
            self.record("put_ext_blob");
            self.inner.put_ext_blob(hash_hex, bytes).await
        }

        async fn get_ext_blob(&self, hash_hex: &str) -> Result<Option<Bytes>, CheckpointError> {
            self.record("get_ext_blob");
            self.inner.get_ext_blob(hash_hex).await
        }

        async fn ext_blob_exists(&self, hash_hex: &str) -> Result<bool, CheckpointError> {
            self.record("ext_blob_exists");
            self.inner.ext_blob_exists(hash_hex).await
        }

        async fn delete_ext_blob(&self, hash_hex: &str) -> Result<(), CheckpointError> {
            self.record("delete_ext_blob");
            self.inner.delete_ext_blob(hash_hex).await
        }

        async fn put_counters_sidecar(&self, body: Bytes) -> Result<(), CheckpointError> {
            self.record("put_counters_sidecar");
            self.inner.put_counters_sidecar(body).await
        }

        async fn get_counters_sidecar(&self) -> Result<Option<Bytes>, CheckpointError> {
            self.record("get_counters_sidecar");
            self.inner.get_counters_sidecar().await
        }

        async fn list_all_object_ids(&self) -> Result<Vec<u64>, CheckpointError> {
            self.record("list_all_object_ids");
            self.inner.list_all_object_ids().await
        }

        async fn list_referenced_ext_hashes(&self) -> Result<HashSet<String>, CheckpointError> {
            self.record("list_referenced_ext_hashes");
            self.inner.list_referenced_ext_hashes().await
        }

        async fn list_live_ext_set(
            &self,
        ) -> Result<crate::checkpoint::LiveExtSet, CheckpointError> {
            self.record("list_live_ext_set");
            self.inner.list_live_ext_set().await
        }
    }

    /// Test-only `CheckpointStore` that always returns `Transient` from
    /// `get_session_checkpoint`. Exercises the `Error(Transient)` arm
    /// without depending on a live-Valkey fault.
    struct AlwaysTransientSessionStore;

    #[async_trait]
    impl CheckpointStore for AlwaysTransientSessionStore {
        async fn put_layer_checkpoint(
            &self,
            _object_id: u64,
            _snap_seq: u64,
            _ompf: Bytes,
            _meta: CheckpointMeta,
        ) -> Result<CheckpointHandle, CheckpointError> {
            unreachable!("test fixture: only get_session_checkpoint is exercised")
        }

        async fn get_latest_layer_checkpoint(
            &self,
            _object_id: u64,
        ) -> Result<Option<CheckpointBlob>, CheckpointError> {
            Ok(None)
        }

        // `get_layer_checkpoint` deliberately inherits the loud
        // `NotImplemented` trait default: this is a standalone fake with no
        // inner store, so there is nothing to forward to.

        async fn list_layer_checkpoints(
            &self,
            _object_id: u64,
        ) -> Result<Vec<CheckpointHandle>, CheckpointError> {
            Ok(vec![])
        }

        async fn delete_layer_checkpoint(
            &self,
            _handle: CheckpointHandle,
        ) -> Result<(), CheckpointError> {
            Ok(())
        }

        async fn put_session_checkpoint(
            &self,
            _session_id: &str,
            _body: Bytes,
            _meta: CheckpointMeta,
        ) -> Result<CheckpointHandle, CheckpointError> {
            unreachable!("test fixture: only get_session_checkpoint is exercised")
        }

        async fn get_session_checkpoint(
            &self,
            _session_id: &str,
        ) -> Result<Option<CheckpointBlob>, CheckpointError> {
            Err(CheckpointError::Transient {
                source: "synthetic transient failure from AlwaysTransientSessionStore".into(),
            })
        }

        async fn delete_session_checkpoint(
            &self,
            _session_id: &str,
        ) -> Result<(), CheckpointError> {
            Ok(())
        }

        async fn put_ext_blob(
            &self,
            _hash_hex: &str,
            _bytes: Bytes,
        ) -> Result<(), CheckpointError> {
            Ok(())
        }

        async fn get_ext_blob(&self, _hash_hex: &str) -> Result<Option<Bytes>, CheckpointError> {
            Ok(None)
        }

        async fn delete_ext_blob(&self, _hash_hex: &str) -> Result<(), CheckpointError> {
            Ok(())
        }

        async fn put_counters_sidecar(&self, _body: Bytes) -> Result<(), CheckpointError> {
            Ok(())
        }

        async fn get_counters_sidecar(&self) -> Result<Option<Bytes>, CheckpointError> {
            Ok(None)
        }

        async fn list_all_object_ids(&self) -> Result<Vec<u64>, CheckpointError> {
            Ok(vec![])
        }

        async fn list_referenced_ext_hashes(&self) -> Result<HashSet<String>, CheckpointError> {
            Ok(HashSet::new())
        }
    }

    /// A store whose `get_session_checkpoint` never answers within any
    /// deadline a caller would tolerate — the session twin of
    /// `recovery::tests::StallingStore` (b.gg7).
    ///
    /// `STALL` is long enough that a test which timed out on it instead of on
    /// the recovery fetch budget would blow its own nextest budget rather than
    /// passing by accident. The `Ok(None)` at the end is deliberately the
    /// *benign* answer: if the timeout ever stopped applying, the test below
    /// would see `NoCheckpoint`, not an error, so a regression cannot disguise
    /// itself as the outcome under test.
    ///
    /// Deliberately **not** shared with `recovery::tests::FakeCheckpointStore`,
    /// which collapsed that module's five doubles into one. The layer double's
    /// defining property is that its three session methods *panic* — that is
    /// how the Epic-1 boundary is pinned on every layer-recovery test — and
    /// this fixture's whole job is to stall inside one of them. Sharing would
    /// mean re-introducing the behaviour axis the collapse removed, on the one
    /// method whose contract differs between the two hooks, to save one
    /// fixture in one file.
    struct StallingSessionStore;

    impl StallingSessionStore {
        const STALL: Duration = Duration::from_secs(30);
    }

    #[async_trait]
    impl CheckpointStore for StallingSessionStore {
        async fn put_layer_checkpoint(
            &self,
            _object_id: u64,
            _snap_seq: u64,
            _ompf: Bytes,
            _meta: CheckpointMeta,
        ) -> Result<CheckpointHandle, CheckpointError> {
            unreachable!("test fixture: only get_session_checkpoint is exercised")
        }

        async fn get_latest_layer_checkpoint(
            &self,
            _object_id: u64,
        ) -> Result<Option<CheckpointBlob>, CheckpointError> {
            Ok(None)
        }

        // `get_layer_checkpoint` deliberately inherits the loud
        // `NotImplemented` trait default: this is a standalone fake with no
        // inner store, so there is nothing to forward to.

        async fn list_layer_checkpoints(
            &self,
            _object_id: u64,
        ) -> Result<Vec<CheckpointHandle>, CheckpointError> {
            Ok(vec![])
        }

        async fn delete_layer_checkpoint(
            &self,
            _handle: CheckpointHandle,
        ) -> Result<(), CheckpointError> {
            Ok(())
        }

        async fn put_session_checkpoint(
            &self,
            _session_id: &str,
            _body: Bytes,
            _meta: CheckpointMeta,
        ) -> Result<CheckpointHandle, CheckpointError> {
            unreachable!("test fixture: only get_session_checkpoint is exercised")
        }

        async fn get_session_checkpoint(
            &self,
            _session_id: &str,
        ) -> Result<Option<CheckpointBlob>, CheckpointError> {
            tokio::time::sleep(Self::STALL).await;
            Ok(None)
        }

        async fn delete_session_checkpoint(
            &self,
            _session_id: &str,
        ) -> Result<(), CheckpointError> {
            Ok(())
        }

        async fn put_ext_blob(
            &self,
            _hash_hex: &str,
            _bytes: Bytes,
        ) -> Result<(), CheckpointError> {
            Ok(())
        }

        async fn get_ext_blob(&self, _hash_hex: &str) -> Result<Option<Bytes>, CheckpointError> {
            Ok(None)
        }

        async fn delete_ext_blob(&self, _hash_hex: &str) -> Result<(), CheckpointError> {
            Ok(())
        }

        async fn put_counters_sidecar(&self, _body: Bytes) -> Result<(), CheckpointError> {
            Ok(())
        }

        async fn get_counters_sidecar(&self) -> Result<Option<Bytes>, CheckpointError> {
            Ok(None)
        }

        async fn list_all_object_ids(&self) -> Result<Vec<u64>, CheckpointError> {
            Ok(vec![])
        }

        async fn list_referenced_ext_hashes(&self) -> Result<HashSet<String>, CheckpointError> {
            Ok(HashSet::new())
        }
    }

    /// Wrapper around `InMemoryCheckpointStore` that overrides
    /// `get_session_checkpoint` to return raw garbage bytes — used to
    /// drive the `Permanent` parse-failure arm without relying on a
    /// production caller writing a malformed blob through `put_session_checkpoint`
    /// (which serialises through `to_bytes`, so it cannot produce
    /// garbage in normal operation).
    struct GarbagePayloadStore {
        /// Distinctive bytes the parse failure must NOT echo. The test
        /// asserts the wrapped error message does not contain this
        /// substring, pinning the redaction discipline.
        garbage: Bytes,
    }

    #[async_trait]
    impl CheckpointStore for GarbagePayloadStore {
        async fn put_layer_checkpoint(
            &self,
            _object_id: u64,
            _snap_seq: u64,
            _ompf: Bytes,
            _meta: CheckpointMeta,
        ) -> Result<CheckpointHandle, CheckpointError> {
            unreachable!()
        }

        async fn get_latest_layer_checkpoint(
            &self,
            _object_id: u64,
        ) -> Result<Option<CheckpointBlob>, CheckpointError> {
            Ok(None)
        }

        // `get_layer_checkpoint` deliberately inherits the loud
        // `NotImplemented` trait default: this is a standalone fake with no
        // inner store, so there is nothing to forward to.

        async fn list_layer_checkpoints(
            &self,
            _object_id: u64,
        ) -> Result<Vec<CheckpointHandle>, CheckpointError> {
            Ok(vec![])
        }

        async fn delete_layer_checkpoint(
            &self,
            _handle: CheckpointHandle,
        ) -> Result<(), CheckpointError> {
            Ok(())
        }

        async fn put_session_checkpoint(
            &self,
            _session_id: &str,
            _body: Bytes,
            _meta: CheckpointMeta,
        ) -> Result<CheckpointHandle, CheckpointError> {
            unreachable!("test fixture: only get_session_checkpoint is exercised")
        }

        async fn get_session_checkpoint(
            &self,
            _session_id: &str,
        ) -> Result<Option<CheckpointBlob>, CheckpointError> {
            Ok(Some(CheckpointBlob {
                bytes: self.garbage.clone(),
                meta: CheckpointMeta::default(),
                handle: CheckpointHandle::base(0, 0, String::new()),
            }))
        }

        async fn delete_session_checkpoint(
            &self,
            _session_id: &str,
        ) -> Result<(), CheckpointError> {
            Ok(())
        }

        async fn put_ext_blob(
            &self,
            _hash_hex: &str,
            _bytes: Bytes,
        ) -> Result<(), CheckpointError> {
            Ok(())
        }

        async fn get_ext_blob(&self, _hash_hex: &str) -> Result<Option<Bytes>, CheckpointError> {
            Ok(None)
        }

        async fn delete_ext_blob(&self, _hash_hex: &str) -> Result<(), CheckpointError> {
            Ok(())
        }

        async fn put_counters_sidecar(&self, _body: Bytes) -> Result<(), CheckpointError> {
            Ok(())
        }

        async fn get_counters_sidecar(&self) -> Result<Option<Bytes>, CheckpointError> {
            Ok(None)
        }

        async fn list_all_object_ids(&self) -> Result<Vec<u64>, CheckpointError> {
            Ok(vec![])
        }

        async fn list_referenced_ext_hashes(&self) -> Result<HashSet<String>, CheckpointError> {
            Ok(HashSet::new())
        }
    }

    // ───────────────────────────────────────────────────────────────────────
    // Test 1 — Disabled short-circuit (no admission, store, or metric churn).
    // ───────────────────────────────────────────────────────────────────────
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn try_recover_session_disabled_short_circuits_without_side_effects() {
        let tv = TestValkey::setup().await;
        let counting = Arc::new(CountingCheckpointStore::new());
        let store: Arc<dyn CheckpointStore> = counting.clone();
        let metrics = make_metrics();
        let hook = SessionRecoveryHook::new(
            store,
            tv.client().clone(),
            tv.key_schema().clone(),
            Arc::new(test_config(false)),
            metrics.clone(),
        );

        let outcome = hook.try_recover_session("any-session").await;
        assert!(
            matches!(outcome, SessionRecoveryOutcome::NoCheckpoint),
            "disabled hook MUST return NoCheckpoint, got {outcome:?}"
        );

        assert_eq!(
            counting.call_count(),
            0,
            "disabled hook MUST NOT invoke any CheckpointStore method"
        );
        assert_eq!(
            hook.admission_len(),
            0,
            "disabled hook MUST NOT touch the admission map"
        );
        assert_eq!(metric_value(&metrics, "restored"), 0);
        assert_eq!(metric_value(&metrics, "not_recoverable"), 0);
        assert_eq!(metric_value(&metrics, "error"), 0);
        assert_eq!(metric_value(&metrics, "already_present"), 0);
    }

    // ───────────────────────────────────────────────────────────────────────
    // Test 2 — Pre-acquire AlreadyPresent. The session is already in Valkey;
    // the hook MUST short-circuit at step 2 with NO admission entry and NO
    // adapter call.
    // ───────────────────────────────────────────────────────────────────────
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn try_recover_session_pre_acquire_already_present_no_admission_or_store() {
        let tv = TestValkey::setup().await;
        let session_id = format!("pre-present-{}", uuid::Uuid::new_v4());

        // Seed the session-meta hash directly so the pre-acquire EXISTS
        // returns true.
        let session_meta_key = tv.key_schema().session_meta(&session_id);
        let _: () = tv
            .client()
            .pool
            .hset(&session_meta_key, ("name", "seeded"))
            .await
            .expect("HSET seed session_meta");

        let counting = Arc::new(CountingCheckpointStore::new());
        let store: Arc<dyn CheckpointStore> = counting.clone();
        let metrics = make_metrics();
        let hook = SessionRecoveryHook::new(
            store,
            tv.client().clone(),
            tv.key_schema().clone(),
            Arc::new(test_config(true)),
            metrics.clone(),
        );

        let outcome = hook.try_recover_session(&session_id).await;
        assert!(
            matches!(outcome, SessionRecoveryOutcome::AlreadyPresent),
            "pre-acquire AlreadyPresent MUST return AlreadyPresent, got {outcome:?}"
        );

        assert_eq!(
            counting.call_count(),
            0,
            "pre-acquire AlreadyPresent MUST NOT invoke any CheckpointStore method (no admission entry created either)"
        );
        assert_eq!(
            hook.admission_len(),
            0,
            "pre-acquire AlreadyPresent MUST NOT create an admission entry"
        );
        assert_eq!(metric_value(&metrics, "already_present"), 1);
        assert_eq!(metric_value(&metrics, "restored"), 0);
        assert_eq!(metric_value(&metrics, "not_recoverable"), 0);
        assert_eq!(metric_value(&metrics, "error"), 0);
    }

    // ───────────────────────────────────────────────────────────────────────
    // Test 3 — NoCheckpoint arm. `get_session_checkpoint` returns Ok(None);
    // hook reports NoCheckpoint and ticks `not_recoverable` (Epic-1
    // harmonised vocabulary).
    // ───────────────────────────────────────────────────────────────────────
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn try_recover_session_no_checkpoint_returns_no_checkpoint() {
        let tv = TestValkey::setup().await;
        let session_id = format!("no-cp-{}", uuid::Uuid::new_v4());
        let store: Arc<dyn CheckpointStore> = Arc::new(InMemoryCheckpointStore::new());
        let metrics = make_metrics();
        let hook = SessionRecoveryHook::new(
            store,
            tv.client().clone(),
            tv.key_schema().clone(),
            Arc::new(test_config(true)),
            metrics.clone(),
        );

        let outcome = hook.try_recover_session(&session_id).await;
        assert!(
            matches!(outcome, SessionRecoveryOutcome::NoCheckpoint),
            "NoCheckpoint arm MUST return NoCheckpoint, got {outcome:?}"
        );
        assert_eq!(metric_value(&metrics, "not_recoverable"), 1);
        assert_eq!(metric_value(&metrics, "restored"), 0);
        assert_eq!(metric_value(&metrics, "error"), 0);
        assert_eq!(metric_value(&metrics, "already_present"), 0);
        assert_eq!(hook.admission_len(), 0);
    }

    // ───────────────────────────────────────────────────────────────────────
    // Test 4 — Recovered arm. Seeds a checkpoint, drives the hook through
    // the full apply pipeline, asserts session-meta hash + path-index
    // members are materialised, and metric ticks `restored`.
    // ───────────────────────────────────────────────────────────────────────
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn try_recover_session_recovered_materialises_session_state() {
        let tv = TestValkey::setup().await;
        let session_id = format!("rec-{}", uuid::Uuid::new_v4());

        // Seed the checkpoint store with a session-checkpoint blob.
        let sc = sample_checkpoint(&session_id);
        let blob_bytes = sc.to_bytes().expect("to_bytes");
        let in_mem = Arc::new(InMemoryCheckpointStore::new());
        let meta = CheckpointMeta {
            creator_replica_id: TEST_SERVICE_LABEL.to_string(),
            written_at_unix_ms: 1_700_000_000_000,
            snap_seq: None,
            session_id: session_id.clone(),
            layer_name: String::new(),
            created_by: sc.created_by.clone(),
            server_ts: 123,
            ..CheckpointMeta::default()
        };
        in_mem
            .put_session_checkpoint(&session_id, blob_bytes, meta)
            .await
            .expect("put_session_checkpoint");

        let store: Arc<dyn CheckpointStore> = in_mem;
        let metrics = make_metrics();
        let hook = SessionRecoveryHook::new(
            store,
            tv.client().clone(),
            tv.key_schema().clone(),
            Arc::new(test_config(true)),
            metrics.clone(),
        );

        let outcome = hook.try_recover_session(&session_id).await;
        assert!(
            matches!(outcome, SessionRecoveryOutcome::Recovered),
            "Recovered arm MUST return Recovered, got {outcome:?}"
        );

        // Session-meta hash is now present with the four sanctioned fields.
        let session_meta_key = tv.key_schema().session_meta(&session_id);
        let meta_after: HashMap<String, String> = tv
            .client()
            .pool
            .hgetall(&session_meta_key)
            .await
            .expect("HGETALL session_meta");
        assert_eq!(meta_after.get("name"), Some(&sc.name));
        assert_eq!(meta_after.get("base_path"), Some(&sc.base_path));
        assert_eq!(meta_after.get("created_by"), Some(&sc.created_by));
        assert_eq!(
            meta_after.get("created_at"),
            Some(&sc.created_at.to_string())
        );

        // Path-index legacy + composite sets carry the seeded members.
        let path_index_key = tv.key_schema().path_index(&sc.base_path);
        let path_index_members: Vec<String> = tv
            .client()
            .pool
            .smembers(&path_index_key)
            .await
            .expect("SMEMBERS path_index");
        assert!(
            path_index_members.contains(&session_id),
            "path_index must carry session_id after Recovered; got {path_index_members:?}"
        );

        let path_names_key = tv.key_schema().path_names_index(&sc.base_path);
        let path_names_members: Vec<String> = tv
            .client()
            .pool
            .smembers(&path_names_key)
            .await
            .expect("SMEMBERS path_names");
        for expected in &sc.path_name_members {
            assert!(
                path_names_members.contains(expected),
                "path_names must carry composite {expected:?}; got {path_names_members:?}"
            );
        }

        // Layers-index carries the hex-encoded layer name.
        let layers_index_key = tv.key_schema().layers_index(&session_id);
        let layers_members: Vec<String> = tv
            .client()
            .pool
            .smembers(&layers_index_key)
            .await
            .expect("SMEMBERS layers_index");
        assert!(
            layers_members.contains(&hex_encode("root.usda")),
            "layers_index must carry hex(root.usda); got {layers_members:?}"
        );

        assert_eq!(metric_value(&metrics, "restored"), 1);
        assert_eq!(metric_value(&metrics, "not_recoverable"), 0);
        assert_eq!(metric_value(&metrics, "error"), 0);
        assert_eq!(metric_value(&metrics, "already_present"), 0);
        assert_eq!(
            hook.admission_len(),
            0,
            "admission entry MUST be retired on the last waiter's drop"
        );
    }

    // ───────────────────────────────────────────────────────────────────────
    // Test 5 — Post-acquire AlreadyPresent. Race two concurrent
    // `try_recover_session` calls on the same session_id; ONE applier
    // call lands, the other observes AlreadyPresent. Pins the parent
    // Task's K=1 admission Success Criteria.
    // ───────────────────────────────────────────────────────────────────────
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn concurrent_recoveries_serialize_and_only_one_applies() {
        let tv = TestValkey::setup().await;
        let session_id = format!("race-{}", uuid::Uuid::new_v4());

        let sc = sample_checkpoint(&session_id);
        let blob_bytes = sc.to_bytes().expect("to_bytes");
        let in_mem = Arc::new(InMemoryCheckpointStore::new());
        let meta = CheckpointMeta {
            creator_replica_id: TEST_SERVICE_LABEL.to_string(),
            written_at_unix_ms: 1_700_000_000_000,
            session_id: session_id.clone(),
            ..CheckpointMeta::default()
        };
        in_mem
            .put_session_checkpoint(&session_id, blob_bytes, meta)
            .await
            .expect("put_session_checkpoint");

        let store: Arc<dyn CheckpointStore> = in_mem;
        let metrics = make_metrics();
        let hook = Arc::new(SessionRecoveryHook::new(
            store,
            tv.client().clone(),
            tv.key_schema().clone(),
            Arc::new(test_config(true)),
            metrics.clone(),
        ));

        // Fire 6 concurrent recoverers. One should observe `Recovered`;
        // the others observe `AlreadyPresent` via the post-acquire EXISTS
        // re-check after queuing on the K=1 mutex.
        let mut handles = Vec::new();
        for _ in 0..6 {
            let hook = Arc::clone(&hook);
            let sid = session_id.clone();
            handles.push(tokio::spawn(
                async move { hook.try_recover_session(&sid).await },
            ));
        }
        let mut recovered = 0;
        let mut already = 0;
        for h in handles {
            match h.await.expect("join") {
                SessionRecoveryOutcome::Recovered => recovered += 1,
                SessionRecoveryOutcome::AlreadyPresent => already += 1,
                other => panic!("unexpected outcome under contention: {other:?}"),
            }
        }
        assert_eq!(
            recovered, 1,
            "exactly one applier call MUST land under K=1 admission; got recovered={recovered} already={already}"
        );
        assert_eq!(
            already, 5,
            "the other 5 callers MUST observe AlreadyPresent via post-acquire EXISTS"
        );

        // Metric cells match the outcome distribution.
        assert_eq!(metric_value(&metrics, "restored"), 1);
        assert_eq!(metric_value(&metrics, "already_present"), 5);

        // Admission entry retired on the last waiter's drop.
        assert_eq!(
            hook.admission_len(),
            0,
            "admission map entry MUST be retired after all 6 attempts drop out"
        );
    }

    // ───────────────────────────────────────────────────────────────────────
    // Test 6 — Error(Permanent) on parse failure. The blob is garbage; the
    // hook returns Error(Permanent). The error message MUST NOT contain
    // the blob bytes (auth-redaction discipline).
    // ───────────────────────────────────────────────────────────────────────
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn try_recover_session_parse_failure_returns_permanent_without_blob_bytes() {
        let tv = TestValkey::setup().await;
        let session_id = format!("garbage-{}", uuid::Uuid::new_v4());

        // Distinctive garbage bytes the error message must not echo.
        let garbage = Bytes::from_static(
            b"SECRET_TOKEN_MUST_NOT_LEAK_INTO_LOGS not even close to valid JSON",
        );
        let store: Arc<dyn CheckpointStore> = Arc::new(GarbagePayloadStore {
            garbage: garbage.clone(),
        });
        let metrics = make_metrics();
        let hook = SessionRecoveryHook::new(
            store,
            tv.client().clone(),
            tv.key_schema().clone(),
            Arc::new(test_config(true)),
            metrics.clone(),
        );

        let outcome = hook.try_recover_session(&session_id).await;
        match outcome {
            SessionRecoveryOutcome::Error(CheckpointError::Permanent { source }) => {
                let rendered = source.to_string();
                assert!(
                    !rendered.contains("SECRET_TOKEN_MUST_NOT_LEAK_INTO_LOGS"),
                    "error message MUST NOT echo garbage blob bytes; rendered={rendered:?}"
                );
                assert!(
                    rendered.contains(&session_id),
                    "error message SHOULD cite session_id for operator triage; rendered={rendered:?}"
                );
            }
            other => panic!("expected Error(Permanent) on parse failure; got {other:?}"),
        }
        assert_eq!(metric_value(&metrics, "error"), 1);
        assert_eq!(metric_value(&metrics, "restored"), 0);
        assert_eq!(metric_value(&metrics, "not_recoverable"), 0);
        assert_eq!(hook.admission_len(), 0);
    }

    // ───────────────────────────────────────────────────────────────────────
    // Test 7 — Error(Transient) from store. `get_session_checkpoint` returns
    // a synthetic transient error; hook surfaces Error(Transient).
    // ───────────────────────────────────────────────────────────────────────
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn try_recover_session_transient_store_failure_returns_error() {
        let tv = TestValkey::setup().await;
        let session_id = format!("transient-{}", uuid::Uuid::new_v4());

        let store: Arc<dyn CheckpointStore> = Arc::new(AlwaysTransientSessionStore);
        let metrics = make_metrics();
        let hook = SessionRecoveryHook::new(
            store,
            tv.client().clone(),
            tv.key_schema().clone(),
            Arc::new(test_config(true)),
            metrics.clone(),
        );

        let outcome = hook.try_recover_session(&session_id).await;
        match outcome {
            SessionRecoveryOutcome::Error(CheckpointError::Transient { .. }) => {}
            other => panic!("expected Error(Transient); got {other:?}"),
        }
        assert_eq!(metric_value(&metrics, "error"), 1);
        assert_eq!(metric_value(&metrics, "restored"), 0);
        assert_eq!(metric_value(&metrics, "not_recoverable"), 0);
        assert_eq!(metric_value(&metrics, "already_present"), 0);
        // Lifecycle invariant — Error(_) arm still retires the admission entry.
        assert_eq!(hook.admission_len(), 0);
    }

    // ───────────────────────────────────────────────────────────────────────
    // Test 8 — Admission map entry retirement after a sequence of attempts
    // on the same session_id across different outcomes. Pins the
    // RAII-guard discipline across the full surface of return paths.
    // ───────────────────────────────────────────────────────────────────────
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn admission_map_retires_entries_across_outcome_classes() {
        let tv = TestValkey::setup().await;
        let metrics = make_metrics();

        // Arm 1: NoCheckpoint — empty store on a fresh session_id.
        let sid1 = format!("retire-1-{}", uuid::Uuid::new_v4());
        let store1: Arc<dyn CheckpointStore> = Arc::new(InMemoryCheckpointStore::new());
        let hook1 = SessionRecoveryHook::new(
            store1,
            tv.client().clone(),
            tv.key_schema().clone(),
            Arc::new(test_config(true)),
            metrics.clone(),
        );
        let _ = hook1.try_recover_session(&sid1).await;
        assert_eq!(
            hook1.admission_len(),
            0,
            "NoCheckpoint arm MUST retire admission entry; got len={}",
            hook1.admission_len()
        );

        // Arm 2: Error(Transient) — synthetic transient store.
        let sid2 = format!("retire-2-{}", uuid::Uuid::new_v4());
        let store2: Arc<dyn CheckpointStore> = Arc::new(AlwaysTransientSessionStore);
        let hook2 = SessionRecoveryHook::new(
            store2,
            tv.client().clone(),
            tv.key_schema().clone(),
            Arc::new(test_config(true)),
            metrics.clone(),
        );
        let _ = hook2.try_recover_session(&sid2).await;
        assert_eq!(
            hook2.admission_len(),
            0,
            "Error(Transient) arm MUST retire admission entry; got len={}",
            hook2.admission_len()
        );

        // Arm 3: Recovered — seed and apply.
        let sid3 = format!("retire-3-{}", uuid::Uuid::new_v4());
        let sc = sample_checkpoint(&sid3);
        let in_mem = Arc::new(InMemoryCheckpointStore::new());
        let meta = CheckpointMeta {
            creator_replica_id: TEST_SERVICE_LABEL.to_string(),
            written_at_unix_ms: 1_700_000_000_000,
            session_id: sid3.clone(),
            ..CheckpointMeta::default()
        };
        in_mem
            .put_session_checkpoint(&sid3, sc.to_bytes().expect("to_bytes"), meta)
            .await
            .expect("put_session_checkpoint");
        let store3: Arc<dyn CheckpointStore> = in_mem;
        let hook3 = SessionRecoveryHook::new(
            store3,
            tv.client().clone(),
            tv.key_schema().clone(),
            Arc::new(test_config(true)),
            metrics.clone(),
        );
        let outcome3 = hook3.try_recover_session(&sid3).await;
        assert!(
            matches!(outcome3, SessionRecoveryOutcome::Recovered),
            "Arm 3 MUST yield Recovered, got {outcome3:?}"
        );
        assert_eq!(
            hook3.admission_len(),
            0,
            "Recovered arm MUST retire admission entry; got len={}",
            hook3.admission_len()
        );
    }

    // ───────────────────────────────────────────────────────────────────────
    // Test 9 — Cross-`session_id` isolation. Two concurrent attempts on
    // DIFFERENT session_ids do NOT block each other — K=1 admission is
    // per-session, not global.
    // ───────────────────────────────────────────────────────────────────────
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn concurrent_recoveries_on_different_sessions_do_not_serialize() {
        let tv = TestValkey::setup().await;
        let sid_a = format!("iso-a-{}", uuid::Uuid::new_v4());
        let sid_b = format!("iso-b-{}", uuid::Uuid::new_v4());

        let sc_a = sample_checkpoint(&sid_a);
        let sc_b = sample_checkpoint(&sid_b);
        let in_mem = Arc::new(InMemoryCheckpointStore::new());
        for (sid, sc) in [(&sid_a, &sc_a), (&sid_b, &sc_b)] {
            let meta = CheckpointMeta {
                creator_replica_id: TEST_SERVICE_LABEL.to_string(),
                written_at_unix_ms: 1_700_000_000_000,
                session_id: sid.clone(),
                ..CheckpointMeta::default()
            };
            in_mem
                .put_session_checkpoint(sid, sc.to_bytes().expect("to_bytes"), meta)
                .await
                .expect("put_session_checkpoint");
        }

        let store: Arc<dyn CheckpointStore> = in_mem;
        let metrics = make_metrics();
        let hook = Arc::new(SessionRecoveryHook::new(
            store,
            tv.client().clone(),
            tv.key_schema().clone(),
            Arc::new(test_config(true)),
            metrics.clone(),
        ));

        // Drive both recoveries concurrently. They MUST both progress past
        // admission (different keys → different mutex) and both land
        // Recovered. We don't try to assert timing here (CI noise) — only
        // that both observe Recovered, which proves the admission map is
        // keyed per-session rather than global.
        let hook_a = Arc::clone(&hook);
        let hook_b = Arc::clone(&hook);
        let sid_a_clone = sid_a.clone();
        let sid_b_clone = sid_b.clone();
        let (out_a, out_b) = tokio::join!(
            tokio::spawn(async move { hook_a.try_recover_session(&sid_a_clone).await }),
            tokio::spawn(async move { hook_b.try_recover_session(&sid_b_clone).await }),
        );
        let out_a = out_a.expect("join a");
        let out_b = out_b.expect("join b");
        assert!(
            matches!(out_a, SessionRecoveryOutcome::Recovered),
            "session A MUST recover independently of B; got {out_a:?}"
        );
        assert!(
            matches!(out_b, SessionRecoveryOutcome::Recovered),
            "session B MUST recover independently of A; got {out_b:?}"
        );

        // Both restored cells ticked.
        assert_eq!(metric_value(&metrics, "restored"), 2);

        // No leftover admission entries (both retired on drop).
        assert_eq!(hook.admission_len(), 0);
    }

    // ── b.gg7: the step-5 fetch deadline (session twin) ──────────────────
    //
    // Near-verbatim port of
    // `recovery::tests::stalled_checkpoint_fetch_releases_the_admission_mutex_
    // for_the_next_caller`, because the hazard is the same one twice:
    // `JoinSession` is the other deadline-exempt RPC, it reaches this hook
    // lazily through `SessionBasePathCache::get_base_path`, and the hook holds
    // a K=1 per-`session_id` mutex across `get_session_checkpoint`.

    /// A `Config` whose recovery fetch budget is small enough to assert
    /// against.
    ///
    /// The budget is `max(rpc_deadline_ms, 2 × checkpoint_request_timeout_ms)`
    /// (see `Config::checkpoint_recovery_fetch_budget`), so BOTH knobs have to
    /// be scaled down — the shared fixture's 30 s request timeout alone would
    /// pin the budget at 60 s no matter what the deadline said.
    fn stall_test_config(budget_ms: u64) -> Config {
        Config {
            rpc_deadline_ms: budget_ms,
            checkpoint_request_timeout_ms: budget_ms / 2,
            ..test_config(true)
        }
    }

    /// A stalled adapter surfaces on the existing `Error(_)` arm, and does so
    /// without pinning the next joiner of the same session behind it.
    ///
    /// Two concurrent callers on the SAME `session_id` is the smallest
    /// configuration that can tell "bounded per holder" from "bounded per
    /// caller": the second one queues on the admission mutex the first holds
    /// across the fetch, so with the deadline the pair costs ~2 × budget and
    /// without it ~2 × the adapter's stall. The assertion is on the pair's wall
    /// clock because that is the observable the head-of-line problem is stated
    /// in; `STALL / 2` is a deliberately loose bound — the point is the order
    /// of magnitude, not the constant.
    ///
    /// The classification half matters just as much: the elapse must land
    /// inside the `Error` arm the hook already had, because that arm is what
    /// defines the `error` cell of `checkpoint_recovery_attempt` and every
    /// caller's handling of a failed recovery.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn stalled_session_checkpoint_fetch_releases_the_admission_mutex() {
        const BUDGET_MS: u64 = 200;

        let tv = TestValkey::setup().await;
        let session_id = format!("stall-{}", uuid::Uuid::new_v4());
        let store: Arc<dyn CheckpointStore> = Arc::new(StallingSessionStore);
        let metrics = make_metrics();
        let hook = Arc::new(SessionRecoveryHook::new(
            store,
            tv.client().clone(),
            tv.key_schema().clone(),
            Arc::new(stall_test_config(BUDGET_MS)),
            metrics.clone(),
        ));

        let started = std::time::Instant::now();
        let mut handles = Vec::new();
        for _ in 0..2 {
            let hook = Arc::clone(&hook);
            let sid = session_id.clone();
            handles.push(tokio::spawn(
                async move { hook.try_recover_session(&sid).await },
            ));
        }
        let mut outcomes = Vec::new();
        for handle in handles {
            outcomes.push(handle.await.expect("join"));
        }
        let elapsed = started.elapsed();

        for outcome in &outcomes {
            assert!(
                matches!(
                    outcome,
                    SessionRecoveryOutcome::Error(CheckpointError::Transient { .. })
                ),
                "a fetch that outran the recovery budget must land on the existing \
                 Error(Transient) arm — that arm defines the `error` metric and the caller's \
                 handling of a failed recovery, and any other variant silently redefines both. \
                 Got {outcome:?}"
            );
        }
        assert!(
            elapsed < StallingSessionStore::STALL / 2,
            "two serialized callers must cost ~2 × {BUDGET_MS}ms, not 2 × the adapter's stall — \
             an unreleased admission mutex turns one slow fetch into head-of-line blocking for \
             every joiner of the session. Took {elapsed:?}"
        );
        assert_eq!(
            metric_value(&metrics, "error"),
            2,
            "each caller records its own attempt"
        );
        assert_eq!(metric_value(&metrics, "not_recoverable"), 0);
        assert_eq!(metric_value(&metrics, "restored"), 0);
        assert_eq!(
            hook.admission_len(),
            0,
            "and the entry retires after the last waiter drops, same as every other Error arm"
        );
    }

    // ── The deletion marker (steps 4a and 6a) ─────────────────────────────
    //
    // `SessionStore::delete` writes `le:{sid}:sess:deleted` in the transaction
    // that deletes the session's metadata and enqueues the checkpoint
    // tombstone only afterwards. Until the tombstone lands, the checkpoint is
    // still at the adapter, and these tests keep it there: the store below
    // always holds a valid checkpoint for the session, so every refusal they
    // pin is the marker's doing, not a missing checkpoint's.

    /// What [`OnFetchStore::get_session_checkpoint`] does before it returns
    /// the checkpoint.
    // The size gap is the `ValkeyClient` the marker variant carries (its
    // test-build fault-seam handles included). One of these is built per test,
    // so boxing it would buy nothing.
    #[allow(clippy::large_enum_variant)]
    enum OnFetch {
        /// Nothing: a plain fetch.
        Nothing,
        /// Write the deletion marker, as a `DeleteSession` on another replica
        /// would while this replica's fetch is in flight.
        SetMarker {
            client: crate::storage::client::ValkeyClient,
            key: String,
        },
        /// Stall every data-pool connection, so the hook's next Valkey
        /// command (step 6a's marker read) fails at its command timeout.
        StallDataPool {
            pool: fred::clients::Pool,
            observer: fred::clients::Client,
            key_prefix: String,
        },
    }

    /// An in-memory checkpoint store that counts session fetches and runs an
    /// [`OnFetch`] action inside each one.
    struct OnFetchStore {
        inner: InMemoryCheckpointStore,
        on_fetch: OnFetch,
        fetches: AtomicUsize,
        stalls: StdMutex<Vec<crate::storage::test_support::ConnStall>>,
    }

    impl OnFetchStore {
        /// A store holding [`sample_checkpoint`] for `session_id`.
        async fn holding_checkpoint(session_id: &str, on_fetch: OnFetch) -> Arc<Self> {
            let store = Arc::new(Self {
                inner: InMemoryCheckpointStore::new(),
                on_fetch,
                fetches: AtomicUsize::new(0),
                stalls: StdMutex::new(Vec::new()),
            });
            store
                .inner
                .put_session_checkpoint(
                    session_id,
                    sample_checkpoint(session_id).to_bytes().expect("to_bytes"),
                    CheckpointMeta {
                        session_id: session_id.to_string(),
                        ..CheckpointMeta::default()
                    },
                )
                .await
                .expect("put_session_checkpoint");
            store
        }

        fn fetches(&self) -> usize {
            self.fetches.load(Ordering::Acquire)
        }

        /// Release every stall the fetch parked.
        async fn release_stalls(&self) {
            let stalls = std::mem::take(&mut *self.stalls.lock().unwrap());
            crate::storage::test_support::release_all(stalls).await;
        }
    }

    #[async_trait]
    impl CheckpointStore for OnFetchStore {
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
        // Forwarded per `CheckpointStore::get_layer_checkpoint`'s MUST.
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
            self.fetches.fetch_add(1, Ordering::AcqRel);
            match &self.on_fetch {
                OnFetch::Nothing => {}
                OnFetch::SetMarker { client, key } => {
                    let _: () = client
                        .pool
                        .set(key, "1", None, None, false)
                        .await
                        .expect("SET deletion marker during the fetch");
                }
                OnFetch::StallDataPool {
                    pool,
                    observer,
                    key_prefix,
                } => {
                    let stalls =
                        crate::storage::test_support::stall_pool(pool, observer, key_prefix).await;
                    self.stalls.lock().unwrap().extend(stalls);
                }
            }
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
        ) -> Result<crate::checkpoint::LiveExtSet, CheckpointError> {
            self.inner.list_live_ext_set().await
        }
    }

    fn hook_over(
        store: Arc<OnFetchStore>,
        tv: &TestValkey,
        metrics: &SharedMetrics,
    ) -> SessionRecoveryHook {
        SessionRecoveryHook::new(
            store,
            tv.client().clone(),
            tv.key_schema().clone(),
            Arc::new(test_config(true)),
            metrics.clone(),
        )
    }

    async fn session_meta_exists(tv: &TestValkey, session_id: &str) -> bool {
        tv.client()
            .pool
            .exists(tv.key_schema().session_meta(session_id))
            .await
            .expect("EXISTS session meta")
    }

    /// Step 4a: a marked session is refused `NoCheckpoint` before the fetch —
    /// the adapter is never read — although its checkpoint is still there.
    /// Counted under `not_recoverable`, the answer a missing checkpoint gets,
    /// because to the caller the two are the same answer.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn try_recover_session_marked_session_is_refused_without_fetching() {
        let tv = TestValkey::setup().await;
        let session_id = format!("marked-{}", uuid::Uuid::new_v4());
        let _: () = tv
            .client()
            .pool
            .set(
                tv.key_schema().session_deleted_marker(&session_id),
                "1",
                None,
                None,
                false,
            )
            .await
            .expect("SET deletion marker");
        let store = OnFetchStore::holding_checkpoint(&session_id, OnFetch::Nothing).await;
        let metrics = make_metrics();
        let hook = hook_over(Arc::clone(&store), &tv, &metrics);

        let outcome = hook.try_recover_session(&session_id).await;

        assert!(
            matches!(outcome, SessionRecoveryOutcome::NoCheckpoint),
            "a session carrying the deletion marker must be refused, not restored from its \
             not-yet-tombstoned checkpoint; got {outcome:?}"
        );
        assert_eq!(
            store.fetches(),
            0,
            "the marker is checked before the fetch, so a deleted session costs no adapter read"
        );
        assert!(
            !session_meta_exists(&tv, &session_id).await,
            "nothing may be applied for a deleted session"
        );
        assert_eq!(metric_value(&metrics, "not_recoverable"), 1);
        assert_eq!(metric_value(&metrics, "restored"), 0);
        assert_eq!(metric_value(&metrics, "error"), 0);
        assert_eq!(hook.admission_len(), 0);
    }

    /// Step 6a: the marker appears while the fetch is in flight — another
    /// replica restored the session and a `DeleteSession` deleted it again —
    /// and the hook re-checks it before the apply, refusing `NoCheckpoint`
    /// with nothing applied.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn try_recover_session_marker_written_during_the_fetch_is_refused_before_the_apply() {
        let tv = TestValkey::setup().await;
        let session_id = format!("marked-mid-fetch-{}", uuid::Uuid::new_v4());
        let store = OnFetchStore::holding_checkpoint(
            &session_id,
            OnFetch::SetMarker {
                client: tv.client().clone(),
                key: tv.key_schema().session_deleted_marker(&session_id),
            },
        )
        .await;
        let metrics = make_metrics();
        let hook = hook_over(Arc::clone(&store), &tv, &metrics);

        let outcome = hook.try_recover_session(&session_id).await;

        assert_eq!(
            store.fetches(),
            1,
            "fixture self-check: step 4a found no marker, so the fetch ran"
        );
        assert!(
            matches!(outcome, SessionRecoveryOutcome::NoCheckpoint),
            "a marker that appeared during the fetch must stop the apply; got {outcome:?}"
        );
        assert!(
            !session_meta_exists(&tv, &session_id).await,
            "the fetched checkpoint must not be applied over a deletion"
        );
        assert_eq!(metric_value(&metrics, "not_recoverable"), 1);
        assert_eq!(metric_value(&metrics, "restored"), 0);
    }

    /// A marker read that fails is no licence to restore: `Error`, counted
    /// under `error`, and nothing applied.
    ///
    /// The failing read is step 6a's (the fetch stalls the data pool before it
    /// returns, so the next command — the re-check — fails at the fixture's
    /// command timeout). Steps 4a and 6a share one read, so this pins the
    /// failure arm for both.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn try_recover_session_failed_marker_read_returns_error_without_applying() {
        let tv = TestValkey::setup().await;
        let session_id = format!("marker-unreadable-{}", uuid::Uuid::new_v4());
        let store = OnFetchStore::holding_checkpoint(
            &session_id,
            OnFetch::StallDataPool {
                pool: tv.client().pool.clone(),
                observer: tv.client().maintenance_pool.next().clone(),
                key_prefix: format!("{}conn-stall:data:", tv.prefix()),
            },
        )
        .await;
        let metrics = make_metrics();
        let hook = hook_over(Arc::clone(&store), &tv, &metrics);

        let outcome = tokio::time::timeout(
            Duration::from_secs(20),
            hook.try_recover_session(&session_id),
        )
        .await
        .expect("the marker read must fail at its command timeout, not hang");
        store.release_stalls().await;

        assert_eq!(store.fetches(), 1, "fixture self-check: the fetch ran");
        assert!(
            matches!(
                outcome,
                SessionRecoveryOutcome::Error(CheckpointError::Transient { .. })
            ),
            "an unreadable marker must fail the recovery, retryably; got {outcome:?}"
        );
        assert!(
            !session_meta_exists(&tv, &session_id).await,
            "nothing may be applied when the marker could not be read"
        );
        assert_eq!(metric_value(&metrics, "error"), 1);
        assert_eq!(metric_value(&metrics, "restored"), 0);
        assert_eq!(hook.admission_len(), 0);
    }

    /// The control for the three above: the same store, the same checkpoint,
    /// no marker — the session is restored. Without it a refusal above could
    /// be a checkpoint the fixture failed to make restorable.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn try_recover_session_unmarked_session_with_the_same_checkpoint_is_restored() {
        let tv = TestValkey::setup().await;
        let session_id = format!("unmarked-{}", uuid::Uuid::new_v4());
        let store = OnFetchStore::holding_checkpoint(&session_id, OnFetch::Nothing).await;
        let metrics = make_metrics();
        let hook = hook_over(Arc::clone(&store), &tv, &metrics);

        let outcome = hook.try_recover_session(&session_id).await;

        assert!(
            matches!(outcome, SessionRecoveryOutcome::Recovered),
            "got {outcome:?}"
        );
        assert_eq!(store.fetches(), 1);
        assert!(session_meta_exists(&tv, &session_id).await);
        assert_eq!(metric_value(&metrics, "restored"), 1);
    }
}
