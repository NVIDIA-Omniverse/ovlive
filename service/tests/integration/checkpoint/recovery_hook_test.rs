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

//! Handler-level recovery-hook integration tests for Subtask
//! `t3.fgj.ya.qm.v7` (under Task `t2.fgj.ya.qm`, Epic `t1.fgj.ya`).
//!
//! Sibling Task `t2.fgj.ya.wg` covers the end-to-end "restore-then-succeed"
//! happy path against every wired gRPC handler. The two contracts this file
//! pins are unique to Task `t2.fgj.ya.qm`:
//!
//! 1. **Retry-once contract.** When the engineer-wired recovery branch in
//!    `handle_update_layer` / `handle_stat_layer` / `handle_subscribe_read_layer`
//!    fires `RecoveryHook::try_restore_layer` and the hook returns
//!    `Restored`, the handler MUST retry the failing sub-operation exactly
//!    ONCE. A second `NotFound` on the retry MUST surface to the client as
//!    gRPC `Code::NotFound` (no second recovery attempt) — and the
//!    `live_edit_checkpoint_recovery_attempts_total{result="restored"}`
//!    counter MUST have incremented exactly once (the single restore that
//!    landed before the post-restore re-wipe).
//!
//! 2. **Disabled-mode gating.** When
//!    `LIVE_EDIT_CHECKPOINT_ENABLED=false`, the recovery hook MUST
//!    short-circuit `NotRecoverable` without touching the underlying
//!    `CheckpointStore`, the per-`object_id` admission map, or the
//!    recovery-attempts metric. The handlers MUST therefore surface the
//!    original `NotFound` directly without any recovery-path side effects.
//!
//! Bonus per the PM-review patch on the Subtask body, as amended by b.33e:
//! the `RecoveryOutcome::Error(err)` arm on every wired handler must
//! CLASSIFY the carried `CheckpointError` rather than mapping every kind
//! to one status. Per SDD §18.1, `Code::Unavailable` (14) is reserved for
//! retriable conditions, so only `CheckpointError::Transient` earns it;
//! `Permanent` / `NotFound` / `Conflict` / `NotImplemented` surface as
//! `Code::Internal` (13), whose wire message is redacted to
//! `"internal error"` while the operator log keeps the detail. Synthetic
//! errors from the store's layer-read entry points are the cleanest way to
//! exercise both halves without depending on a real backend outage:
//! [`AlwaysTransientStore`] drives the retriable half, [`SelectionErrorStore`]
//! the non-retriable half for each of the four remaining kinds.
//!
//! ## Wiring strategy per RPC
//!
//! The three handlers expose the NotFound surface differently, which
//! drives a per-RPC choice of how to provoke the recovery branch:
//!
//! - **UpdateLayer** — the handler carries TWO recovery arms, and both fire
//!   on the same condition from different places: `get_layer_mapping`
//!   returning `None` in the step-4 mapping-resolve loop
//!   (`src/grpc/layer.rs`), and `Err(ServiceError::NotFound(_))` from
//!   the commit itself (`src/grpc/layer.rs`). The **retry-once and
//!   disabled-mode** tests drive only the first: they wipe
//!   `le:{sid}:layer:{hex(name)}` and leave `obj:*` intact, so
//!   `get_layer_mapping` returns `None`, recovery runs, the post-acquire
//!   structural-header re-check (recovery.rs step 3) sees the
//!   still-present header and reports `Restored` cheaply, the handler
//!   retries `get_layer_mapping`, sees `None` again (apply does NOT
//!   restore the mapping — see `write_object`), and surfaces `NotFound`
//!   via the `recovered_once` latch.
//!
//!   The later b.33e **error-classification** tests cover both UpdateLayer
//!   arms in the table below, and both without a fault-injection seam:
//!   `update_layer_mapping_resolve_permanent_error_returns_internal` wipes
//!   the mapping, and `update_layer_empty_header_permanent_error_returns_internal`
//!   inverts the wipe (wipe `obj:*`, leave the mapping intact) so the
//!   commit's own `NotFound` is what reaches the arm. Before b.f1v the
//!   second of those needed a seam — the storage layer answered a wiped
//!   `obj:hdr` with `FailedPrecondition` — and a seam-gated near-duplicate
//!   existed for it; b.f4t deleted that duplicate (see the surviving
//!   test's rustdoc).
//!
//! - **StatLayer / SubscribeReadLayer** — both handlers' wired recovery
//!   branches fire on `read_object_header` / `stat_layer_for_object_id`
//!   returning `NotFound`, which is the NATURAL symptom of a wiped
//!   `obj:hdr` — and, since b.f4t, of a header that is present but missing
//!   `seq` / `created_by`, which those two read paths refuse identically.
//!   Tests wipe `obj:*` to trigger the first call's NotFound;
//!   the engineer-wired `RecoveryHook::with_force_post_restore_notfound_for_test`
//!   one-shot latch (b.sxc) re-`UNLINK`s `obj:hdr` synchronously on the
//!   recovery hook's own connection immediately before returning
//!   `Restored`, so the handler's post-restore retry deterministically
//!   re-observes `NotFound` and trips the `recovered_once` latch. The
//!   prior 16-wiper race-based pattern flaked at ~2-3 % on slow CI hosts
//!   and is replaced by this deterministic seam.
//!
//! For the **disabled-mode tests** we use a [`CallCountingStore`]
//! wrapper around `InMemoryCheckpointStore` to prove ZERO store-side
//! method invocations when `checkpoint_enabled=false`, in addition to
//! the metric / admission-map assertions the engineer's
//! `build_disabled_recovery_hook` already covers semantically. Its
//! counted set includes the two restore-source-selection entry points
//! (`list_layer_checkpoints`, `get_layer_checkpoint`), so the
//! zero-calls assertion covers the selection step too.
//!
//! For the **adapter-transient-error tests** we use an
//! [`AlwaysTransientStore`] fixture that synthesizes a
//! `CheckpointError::Transient(_)` from every layer-read entry point,
//! driving the handler's `Error(_)` arm into `ServiceError::Unavailable`
//! → gRPC `Code::Unavailable`. The injection point that actually fires
//! is `list_layer_checkpoints`: `RecoveryHook::select_restore_source`
//! (SR-2.5) picks the newest `Base` by listing first, so the list is the
//! first store call any recovery attempt makes. For UpdateLayer's
//! transient test we wipe BOTH the layer mapping AND `obj:*` so step 3's
//! structural-header re-check falls through to step 5's selection.
//!
//! [`SelectionErrorStore`] covers the same selection call for every
//! NON-retriable kind, pinning `Code::Internal` at the handler boundary
//! (b.33e). Coverage is two-dimensional, because the classification lives
//! in one shared helper (`ServiceError::from_recovery_error`) consumed at
//! **four** distinct `RecoveryOutcome::Error(err)` arms — three in
//! `src/grpc/layer.rs`, one in `src/grpc/subscribe.rs`:
//!
//! | # | Arm | Non-retriable test |
//! |---|-----|--------------------|
//! | 1 | `handle_update_layer` mapping-resolve loop (`src/grpc/layer.rs`) | `update_layer_mapping_resolve_permanent_error_returns_internal` |
//! | 2 | `handle_update_layer` merged post-commit `NotFound` (`src/grpc/layer.rs`) | `update_layer_empty_header_permanent_error_returns_internal` |
//! | 3 | `handle_stat_layer` (`src/grpc/layer.rs`) | `stat_layer_selection_permanent_error_returns_internal` |
//! | 4 | `try_recover_after_notfound` (`src/grpc/subscribe.rs`) | `subscribe_read_layer_permanent_error_returns_internal` |
//!
//! **The count has moved twice and both moves are recorded here.** It was
//! written as four, corrected to five by the b.33e PM review (the
//! mapping-resolve loop, arm 1, had been missed entirely), and is four again
//! since b.f1v: that change deleted `handle_update_layer`'s third arm — the
//! one matching the empty-header sentinel *by message equality* — so the
//! empty-header origin and the post-commit `NotFound` origin now share ONE
//! arm keyed on the variant. Arm 1 remains genuinely distinct (a different
//! `match`, on a different value, at a different point in the handler) and
//! keeps its own test.
//!
//! The other three non-retriable KINDS (`NotFound` / `Conflict` /
//! `NotImplemented`) are pinned once each on the StatLayer arm — the
//! classification is per-kind and arm-independent, so re-running all four
//! kinds × all four arms would be sixteen tests proving one `match`. Each
//! kind test also asserts the `error` metric cell, which is what catches a
//! regression that folded the kind into the `NotRecoverable` verdict
//! instead (that would surface `NOT_FOUND` and tick `not_recoverable`).
//!
//! Sandbox env note: these tests use `TestValkey::setup`, which reads
//! `TEST_VALKEY_URL`. Under the project sandbox the URL must include the
//! password and `host.docker.internal` host; see CLAUDE.md.

#![cfg(feature = "test-checkpoint-fault-injection")]

use std::collections::HashSet;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

use async_trait::async_trait;
use bytes::Bytes;
use fred::interfaces::KeysInterface;
use futures::StreamExt;
use tonic::Request;

use live_edit::checkpoint::adapters::InMemoryCheckpointStore;
use live_edit::checkpoint::{
    CheckpointBlob, CheckpointError, CheckpointHandle, CheckpointMeta, CheckpointStore,
    RecoveryHook,
};
use live_edit::grpc::layer::{handle_stat_layer, handle_update_layer};
use live_edit::grpc::proto::{StatLayerRequest, SubscribeReadLayerRequest};
use live_edit::grpc::service::LiveEditService;
use live_edit::grpc::subscribe::handle_subscribe_read_layer;
use live_edit::grpc::BearerToken;
use live_edit::storage::test_support::TestValkey;

/// The `live_edit_checkpoint_restore_lineage_total` reader, shared with the
/// checkpoint e2e lane. Read here for the b.hxk arm alone: that arm's whole
/// subject is WHICH exhaustion the selection reported, and the two exhaustion
/// cells are the only place that distinction is observable at the handler.
use super::helpers::read_restore_lineage_counter;
/// Promoted out of this file by b.f4t sub-task 5; `replace_recovery` and its
/// latch variant stayed local because their signatures diverge.
use crate::common::{
    recovery_metric, replace_metrics, seed_layer_checkpoint, update_request, wipe_object_keys,
};
use crate::grpc::helpers::{
    build_service, create_session_and_layer, AllowAllPermissions, DenyAllPermissions,
};

/// Pinned to `common::RECOVERY_SERVICE_LABEL` by construction: the shared
/// [`recovery_metric`] reads that `{service=...}` cell, so a divergence here
/// would silently zero every metric assertion in this file.
const SERVICE_LABEL: &str = crate::common::RECOVERY_SERVICE_LABEL;

/// The `(timestamp, client_id)` this file's `UpdateLayer` headers carry —
/// unchanged from the pre-promotion local `update_request`, named so the
/// five-argument shared helper reads as one call rather than five literals.
const CLIENT_TS: u64 = 1000;
const CLIENT_ID: &str = "recovery-hook-client";

// =============================================================================
// Test-only `CheckpointStore` wrappers + fixtures.
// =============================================================================

/// Counting wrapper around any `CheckpointStore`. Records every trait-method
/// invocation count so the disabled-mode tests can prove the hook performs
/// ZERO store calls when `checkpoint_enabled=false`.
///
/// Every arm is tick-then-**forward**: the counter must not change what the
/// inner store answers, or the disabled-mode tests would be proving something
/// about this wrapper rather than about the hook. That applies with particular
/// force to `get_layer_checkpoint`, whose trait default is
/// `CheckpointError::NotImplemented` — a decorator that ticked without
/// forwarding (or omitted the arm entirely and inherited the default) would
/// turn a healthy inner store into a hard failure on the restore path. The
/// trait's own "# Default impl" block states the rule: any wrapper
/// `CheckpointStore` MUST forward this method wherever it forwards the other
/// layer methods.
///
/// `list_layer_checkpoints` + `get_layer_checkpoint` are the two entry points
/// `RecoveryHook::select_restore_source` calls (SR-2.5, list-then-get); both
/// are counted here so the zero-store-calls assertion in the disabled-mode
/// tests covers the selection step and not only the older
/// `get_latest_layer_checkpoint` surface.
struct CallCountingStore {
    inner: Arc<dyn CheckpointStore>,
    calls: AtomicUsize,
}

impl CallCountingStore {
    fn new(inner: Arc<dyn CheckpointStore>) -> Self {
        Self {
            inner,
            calls: AtomicUsize::new(0),
        }
    }

    fn call_count(&self) -> usize {
        self.calls.load(Ordering::Acquire)
    }

    fn tick(&self) {
        self.calls.fetch_add(1, Ordering::AcqRel);
    }
}

#[async_trait]
impl CheckpointStore for CallCountingStore {
    async fn put_layer_checkpoint(
        &self,
        object_id: u64,
        snap_seq: u64,
        ompf: Bytes,
        meta: CheckpointMeta,
    ) -> Result<CheckpointHandle, CheckpointError> {
        self.tick();
        self.inner
            .put_layer_checkpoint(object_id, snap_seq, ompf, meta)
            .await
    }

    async fn get_latest_layer_checkpoint(
        &self,
        object_id: u64,
    ) -> Result<Option<CheckpointBlob>, CheckpointError> {
        self.tick();
        self.inner.get_latest_layer_checkpoint(object_id).await
    }

    async fn list_layer_checkpoints(
        &self,
        object_id: u64,
    ) -> Result<Vec<CheckpointHandle>, CheckpointError> {
        self.tick();
        self.inner.list_layer_checkpoints(object_id).await
    }

    async fn get_layer_checkpoint(
        &self,
        handle: &CheckpointHandle,
    ) -> Result<Option<CheckpointBlob>, CheckpointError> {
        self.tick();
        self.inner.get_layer_checkpoint(handle).await
    }

    async fn delete_layer_checkpoint(
        &self,
        handle: CheckpointHandle,
    ) -> Result<(), CheckpointError> {
        self.tick();
        self.inner.delete_layer_checkpoint(handle).await
    }

    async fn put_session_checkpoint(
        &self,
        session_id: &str,
        body: Bytes,
        meta: CheckpointMeta,
    ) -> Result<CheckpointHandle, CheckpointError> {
        self.tick();
        self.inner
            .put_session_checkpoint(session_id, body, meta)
            .await
    }

    async fn get_session_checkpoint(
        &self,
        session_id: &str,
    ) -> Result<Option<CheckpointBlob>, CheckpointError> {
        self.tick();
        self.inner.get_session_checkpoint(session_id).await
    }

    async fn delete_session_checkpoint(&self, session_id: &str) -> Result<(), CheckpointError> {
        self.tick();
        self.inner.delete_session_checkpoint(session_id).await
    }

    async fn list_session_checkpoint_ids(&self) -> Result<Vec<String>, CheckpointError> {
        self.tick();
        self.inner.list_session_checkpoint_ids().await
    }

    async fn put_ext_blob(&self, hash_hex: &str, bytes: Bytes) -> Result<(), CheckpointError> {
        self.tick();
        self.inner.put_ext_blob(hash_hex, bytes).await
    }

    async fn get_ext_blob(&self, hash_hex: &str) -> Result<Option<Bytes>, CheckpointError> {
        self.tick();
        self.inner.get_ext_blob(hash_hex).await
    }

    async fn ext_blob_exists(&self, hash_hex: &str) -> Result<bool, CheckpointError> {
        self.tick();
        self.inner.ext_blob_exists(hash_hex).await
    }

    async fn delete_ext_blob(&self, hash_hex: &str) -> Result<(), CheckpointError> {
        self.tick();
        self.inner.delete_ext_blob(hash_hex).await
    }

    async fn put_counters_sidecar(&self, body: Bytes) -> Result<(), CheckpointError> {
        self.tick();
        self.inner.put_counters_sidecar(body).await
    }

    async fn get_counters_sidecar(&self) -> Result<Option<Bytes>, CheckpointError> {
        self.tick();
        self.inner.get_counters_sidecar().await
    }

    async fn list_all_object_ids(&self) -> Result<Vec<u64>, CheckpointError> {
        self.tick();
        self.inner.list_all_object_ids().await
    }

    async fn list_referenced_ext_hashes(&self) -> Result<HashSet<String>, CheckpointError> {
        self.tick();
        self.inner.list_referenced_ext_hashes().await
    }

    async fn list_live_ext_set(
        &self,
    ) -> Result<live_edit::checkpoint::LiveExtSet, CheckpointError> {
        self.tick();
        self.inner.list_live_ext_set().await
    }
}

/// `CheckpointStore` fixture whose **every layer-read entry point** returns
/// `CheckpointError::Transient`. Drives the handler's `Error(_)` arm into
/// `ServiceError::Unavailable` → gRPC `Code::Unavailable`.
///
/// Mirrors the `AlwaysTransientStore` fixture inside
/// `src/checkpoint/recovery.rs::tests` (different file because that one is
/// `#[cfg(test)]` and not reachable from integration tests).
///
/// All three of `list_layer_checkpoints`, `get_layer_checkpoint`, and
/// `get_latest_layer_checkpoint` fail, deliberately. `RecoveryHook`'s restore
/// source is selected by `select_restore_source` (SR-2.5) as a **list-then-get
/// pair**, so the list is the first store call any recovery attempt makes and
/// the fixture must fail there to reach the `Error(_)` arm at all — an earlier
/// revision returned `Ok(vec![])` from the list and the hook short-circuited to
/// `NotRecoverable`, turning all three transient tests into `NOT_FOUND`.
/// `get_layer_checkpoint` fails for the same reason one step later: it keeps
/// the fixture's contract "this backend is transiently broken" true regardless
/// of *which* selection entry point a future refactor reaches first, so the
/// transient-error proof (SDD §18.1) cannot be silently un-wired again by
/// reordering the selection. `get_latest_layer_checkpoint` is retained because
/// the trait still exposes it and a caller reaching it must observe the same
/// backend state.
struct AlwaysTransientStore;

#[async_trait]
impl CheckpointStore for AlwaysTransientStore {
    async fn put_layer_checkpoint(
        &self,
        _object_id: u64,
        _snap_seq: u64,
        _ompf: Bytes,
        _meta: CheckpointMeta,
    ) -> Result<CheckpointHandle, CheckpointError> {
        unreachable!("test fixture: only the layer-read entry points are exercised")
    }

    async fn get_latest_layer_checkpoint(
        &self,
        _object_id: u64,
    ) -> Result<Option<CheckpointBlob>, CheckpointError> {
        Err(CheckpointError::Transient {
            source: "synthetic transient failure from AlwaysTransientStore".into(),
        })
    }

    async fn list_layer_checkpoints(
        &self,
        _object_id: u64,
    ) -> Result<Vec<CheckpointHandle>, CheckpointError> {
        Err(CheckpointError::Transient {
            source: "synthetic transient failure from AlwaysTransientStore".into(),
        })
    }

    async fn get_layer_checkpoint(
        &self,
        _handle: &CheckpointHandle,
    ) -> Result<Option<CheckpointBlob>, CheckpointError> {
        Err(CheckpointError::Transient {
            source: "synthetic transient failure from AlwaysTransientStore".into(),
        })
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
        Err(CheckpointError::NotImplemented {
            method: "put_session_checkpoint",
        })
    }

    async fn get_session_checkpoint(
        &self,
        _session_id: &str,
    ) -> Result<Option<CheckpointBlob>, CheckpointError> {
        Ok(None)
    }

    async fn delete_session_checkpoint(&self, _session_id: &str) -> Result<(), CheckpointError> {
        Ok(())
    }

    async fn put_ext_blob(&self, _hash_hex: &str, _bytes: Bytes) -> Result<(), CheckpointError> {
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

/// `CheckpointStore` fixture whose **restore-source selection** fails with a
/// caller-chosen [`CheckpointError`] — the listing call `select_restore_source`
/// (SR-2.5) makes first, plus its paired get for the same
/// reorder-proofing reason [`AlwaysTransientStore`] documents.
///
/// The selection listing is a store call the recovery hot path did not use to
/// make, and `CheckpointStore::list_layer_checkpoints` is contractually allowed
/// to answer `Permanent` (a malformed persisted meta, for one — the trait's own
/// error contract). Without a handler-boundary test pinning what a `Permanent`
/// from *that* call surfaces as, a regression that folded it into
/// `Ok(vec![])` / `NotRecoverable` would silently downgrade a corrupt-store
/// signal to "this layer was never checkpointed", and nothing would notice.
///
/// Parameterised over the error rather than hardcoding `Permanent` (b.33e):
/// the classification under test is per-`CheckpointError`-kind, and all four
/// non-retriable kinds must reach the handler boundary as `Code::Internal`.
/// One stub with a `fn() -> CheckpointError` constructor keeps the four kind
/// tests from being four copy-pasted 90-line trait impls — the exact
/// "copy-paste tests with one-line differences" shape the test-review guide
/// flags. Build one via [`SelectionErrorStore::permanent`] and friends.
///
/// The non-layer methods are inert stubs, identical to
/// [`AlwaysTransientStore`]'s: the fixture is about one classification on one
/// call path, and a recovery attempt touches nothing else.
struct SelectionErrorStore {
    /// Invoked afresh per call: `CheckpointError` is not `Clone`, and every
    /// layer-read entry point must be able to answer the same kind.
    make_error: fn() -> CheckpointError,
}

impl SelectionErrorStore {
    fn new(make_error: fn() -> CheckpointError) -> Self {
        Self { make_error }
    }

    /// The b.33e headline kind: a corrupt blob / malformed persisted meta.
    fn permanent() -> Self {
        Self::new(|| CheckpointError::Permanent {
            source: "synthetic permanent failure from SelectionErrorStore".into(),
        })
    }

    /// Plain absence is `Ok(None)`, so a `NotFound` *error* from a layer-read
    /// entry point means an inconsistent store — not a retriable condition.
    fn not_found() -> Self {
        Self::new(|| CheckpointError::NotFound)
    }

    fn conflict() -> Self {
        Self::new(|| CheckpointError::Conflict)
    }

    /// Defensive: every wired adapter implements the layer-read methods, so
    /// this kind reaching a handler indicates adapter-surface drift.
    fn not_implemented() -> Self {
        Self::new(|| CheckpointError::NotImplemented {
            method: "list_layer_checkpoints",
        })
    }

    fn error(&self) -> CheckpointError {
        (self.make_error)()
    }
}

#[async_trait]
impl CheckpointStore for SelectionErrorStore {
    async fn put_layer_checkpoint(
        &self,
        _object_id: u64,
        _snap_seq: u64,
        _ompf: Bytes,
        _meta: CheckpointMeta,
    ) -> Result<CheckpointHandle, CheckpointError> {
        unreachable!("test fixture: only the layer-read entry points are exercised")
    }

    async fn get_latest_layer_checkpoint(
        &self,
        _object_id: u64,
    ) -> Result<Option<CheckpointBlob>, CheckpointError> {
        Err(self.error())
    }

    async fn list_layer_checkpoints(
        &self,
        _object_id: u64,
    ) -> Result<Vec<CheckpointHandle>, CheckpointError> {
        Err(self.error())
    }

    async fn get_layer_checkpoint(
        &self,
        _handle: &CheckpointHandle,
    ) -> Result<Option<CheckpointBlob>, CheckpointError> {
        Err(self.error())
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
        Err(CheckpointError::NotImplemented {
            method: "put_session_checkpoint",
        })
    }

    async fn get_session_checkpoint(
        &self,
        _session_id: &str,
    ) -> Result<Option<CheckpointBlob>, CheckpointError> {
        Ok(None)
    }

    async fn delete_session_checkpoint(&self, _session_id: &str) -> Result<(), CheckpointError> {
        Ok(())
    }

    async fn put_ext_blob(&self, _hash_hex: &str, _bytes: Bytes) -> Result<(), CheckpointError> {
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

/// `CheckpointStore` fixture whose listing advertises ONE full (`Base`)
/// generation and whose fetch of it **never returns** — the store shape that
/// makes a restore-source selection exhaust *without ever reading the one
/// artifact that could have restored the layer* (b.hxk).
///
/// # Why this is not a variant of [`SelectionErrorStore`]
///
/// Every other fixture here fails a store call. This one fails NOTHING: the
/// listing succeeds, the generation exists, and the layer really is restorable
/// — the store is simply too slow to say so inside the budget. That is the
/// whole distinction the fix turns on. `Exhaustion::AllBroken` (folded onto
/// `exhausted`, answering `Ok(None)`) is a claim that the selection *read* every
/// full generation and none is usable; here it read none of them, so the honest
/// verdict is `Exhaustion::BudgetInconclusive` and a retriable status.
///
/// The wedge is ended by the gather's own per-fetch cap — `timeout_at` on a
/// bound of at most one `checkpoint_request_timeout_ms` — so the test spends one
/// (deliberately tiny) reserve and the pending future is dropped by that
/// timeout, leaking nothing. This is the same instrument
/// `restore_fault_injection_matrix_test.rs`'s `GetFault::Wedge` uses; it is
/// injected latency, not a synchronisation `sleep`.
struct WedgedBaseListingStore;

impl WedgedBaseListingStore {
    /// `snap_seq` of the single advertised generation. Any value: nothing ever
    /// reads the artifact, which is the point.
    const SNAP_SEQ: u64 = 1;
}

#[async_trait]
impl CheckpointStore for WedgedBaseListingStore {
    async fn put_layer_checkpoint(
        &self,
        _object_id: u64,
        _snap_seq: u64,
        _ompf: Bytes,
        _meta: CheckpointMeta,
    ) -> Result<CheckpointHandle, CheckpointError> {
        unreachable!("test fixture: only the restore-source selection is exercised")
    }

    async fn get_latest_layer_checkpoint(
        &self,
        _object_id: u64,
    ) -> Result<Option<CheckpointBlob>, CheckpointError> {
        unreachable!("test fixture: the composed path selects via list + get, never this")
    }

    async fn list_layer_checkpoints(
        &self,
        object_id: u64,
    ) -> Result<Vec<CheckpointHandle>, CheckpointError> {
        // Succeeds, and advertises a Base — so the selection has something it
        // is obliged to adjudicate before it may claim the layer is
        // unrestorable.
        Ok(vec![CheckpointHandle::base(
            object_id,
            Self::SNAP_SEQ,
            format!("{object_id}.{}", Self::SNAP_SEQ),
        )])
    }

    async fn get_layer_checkpoint(
        &self,
        _handle: &CheckpointHandle,
    ) -> Result<Option<CheckpointBlob>, CheckpointError> {
        std::future::pending::<()>().await;
        unreachable!("a pending future never resolves")
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
        Err(CheckpointError::NotImplemented {
            method: "put_session_checkpoint",
        })
    }

    async fn get_session_checkpoint(
        &self,
        _session_id: &str,
    ) -> Result<Option<CheckpointBlob>, CheckpointError> {
        Ok(None)
    }

    async fn delete_session_checkpoint(&self, _session_id: &str) -> Result<(), CheckpointError> {
        Ok(())
    }

    async fn put_ext_blob(&self, _hash_hex: &str, _bytes: Bytes) -> Result<(), CheckpointError> {
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

// =============================================================================
// Test-only helpers.
// =============================================================================

/// Replace the service's recovery hook with a freshly-constructed one
/// backed by `store`, sharing the service's existing `layer_store` and
/// `metrics`. `checkpoint_enabled` controls the hook's short-circuit gate.
///
/// Built fresh per-test to avoid leaking the disabled-mode hook from
/// `build_service` into the retry-once tests; the helper keeps the rest
/// of the service wiring (auth, valkey, key schema) intact.
fn replace_recovery(
    svc: &mut LiveEditService,
    store: Arc<dyn CheckpointStore>,
    checkpoint_enabled: bool,
) {
    let mut cfg = crate::common::test_config();
    cfg.checkpoint_enabled = checkpoint_enabled;
    cfg.service_label = SERVICE_LABEL.to_string();
    // Hand-built on purpose, NOT a surviving `CheckpointHarness` workaround: this
    // file drives a plain `build_service` service with no external-value store and
    // no base bootstrap behind it, so the bare `RecoveryHook::new` under a
    // caller-chosen `checkpoint_enabled` IS the subject.
    svc.recovery = Arc::new(RecoveryHook::new(
        store,
        Arc::clone(&svc.layer_store),
        Arc::new(cfg),
        Arc::clone(&svc.metrics),
    ));
}

/// Variant of [`replace_recovery`] that additionally dials the two knobs
/// `Config::checkpoint_recovery_fetch_budget()` is composed from, so a test can
/// make a wedged fetch cost milliseconds instead of the 30 s
/// `common::test_config()` inherits from production.
///
/// Both knobs are set to `reserve`, which makes the budget exactly
/// `max(reserve, 2 x reserve) = 2 x reserve` — production's own ratio, spelled
/// with one number so a later edit cannot perturb the ratio by moving one term.
fn replace_recovery_with_reserve(
    svc: &mut LiveEditService,
    store: Arc<dyn CheckpointStore>,
    checkpoint_enabled: bool,
    reserve: std::time::Duration,
) {
    let mut cfg = crate::common::test_config();
    cfg.checkpoint_enabled = checkpoint_enabled;
    cfg.service_label = SERVICE_LABEL.to_string();
    cfg.checkpoint_request_timeout_ms = reserve.as_millis() as u64;
    cfg.rpc_deadline_ms = reserve.as_millis() as u64;
    assert_eq!(
        cfg.checkpoint_recovery_fetch_budget(),
        reserve * 2,
        "the caller's precondition: budget == 2 x reserve, production's ratio"
    );
    svc.recovery = Arc::new(RecoveryHook::new(
        store,
        Arc::clone(&svc.layer_store),
        Arc::new(cfg),
        Arc::clone(&svc.metrics),
    ));
}

/// Variant of [`replace_recovery`] that installs the b.sxc one-shot
/// post-restore-NotFound latch on the freshly-built `RecoveryHook`. When
/// the caller subsequently provokes a recovery attempt, the seam fires
/// exactly once at the post-Restored boundary: the hook synchronously
/// `UNLINK`s the just-restored `obj:hdr` before returning
/// `RecoveryOutcome::Restored`, guaranteeing the handler's post-restore
/// retry observes `NotFound`. Replaces the prior `WipeAfterRestoreStore`
/// race that flaked at ~2-3 % on slow CI hosts.
fn replace_recovery_with_post_restore_notfound_latch(
    svc: &mut LiveEditService,
    store: Arc<dyn CheckpointStore>,
    checkpoint_enabled: bool,
    latch: Arc<AtomicBool>,
) {
    let mut cfg = crate::common::test_config();
    cfg.checkpoint_enabled = checkpoint_enabled;
    cfg.service_label = SERVICE_LABEL.to_string();
    // Hand-built for the same reason as [`replace_recovery`] above — no harness
    // behind this service — with the b.sxc test-only latch chained on.
    svc.recovery = Arc::new(
        RecoveryHook::new(
            store,
            Arc::clone(&svc.layer_store),
            Arc::new(cfg),
            Arc::clone(&svc.metrics),
        )
        .with_force_post_restore_notfound_for_test(latch),
    );
}

/// `UNLINK` the `le:{sid}:layer:{hex(name)}` mapping key for a layer.
///
/// Used by the UpdateLayer tests to drive the handler's FIRST recovery
/// surface — the step-4 mapping-resolve loop, where `get_layer_mapping`
/// answers `None` (`src/grpc/layer.rs`). Wiping just the mapping (NOT
/// `obj:*`) is what selects that surface over the commit's own `NotFound`
/// arm: with `obj:*` intact the recovery hook's post-acquire
/// structural-header re-check (recovery.rs step 3) observes the
/// already-present, structurally-complete `obj:hdr` and reports `Restored`
/// cheaply — without running apply at all. The retry then re-issues
/// `get_layer_mapping`, observes `None` a second time, hits the
/// `recovered_once` latch and surfaces `NotFound`.
///
/// b.f1v note: a wiped `obj:hdr` now surfaces as `ServiceError::NotFound`
/// from both commit backends, not `FailedPrecondition` — so the commit is a
/// second *natural* route into the merged arm, and no fault-injection seam is
/// needed to reach it. This helper's job is to pick the OTHER route
/// deterministically.
async fn wipe_layer_mapping(tv: &TestValkey, session_id: &str, layer_name: &str) {
    let mapping_key = tv.key_schema().layer_mapping(session_id, layer_name);
    let _: () = tv
        .client()
        .pool
        .unlink(&mapping_key)
        .await
        .expect("UNLINK layer mapping must succeed");
}

fn stat_request(session_id: &str, layer_name: &str) -> Request<StatLayerRequest> {
    let mut request = Request::new(StatLayerRequest {
        session_id: session_id.to_owned(),
        layer_name: layer_name.to_owned(),
    });
    request
        .extensions_mut()
        .insert(BearerToken::new("test-token"));
    request
}

fn subscribe_request(
    session_id: &str,
    layer_name: &str,
    object_id: u64,
    sequence_num: u64,
) -> Request<SubscribeReadLayerRequest> {
    let mut request = Request::new(SubscribeReadLayerRequest {
        session_id: session_id.to_owned(),
        layer_name: layer_name.to_owned(),
        object_id,
        sequence_num,
        client_id: "recovery-hook-subscriber".to_owned(),
    });
    request
        .extensions_mut()
        .insert(BearerToken::new("test-token"));
    request
}

// =============================================================================
// Disabled-mode tests — hook bypassed, NotFound surfaces, store untouched.
// =============================================================================

/// UpdateLayer disabled-mode: the layer mapping is wiped, leaving
/// `get_layer_mapping` returning `None`. The handler enters the recovery
/// branch (first surface). With `checkpoint_enabled=false`, the hook
/// short-circuits to `NotRecoverable` without touching the store, the
/// admission map, or the metric; the handler MUST surface `NotFound`
/// directly.
///
/// The counting store wrapper proves the ZERO-store-calls property: a
/// regression that accidentally consulted the store anyway (e.g.,
/// short-circuiting AFTER fetching the latest checkpoint) would tick the
/// counter and fail the assertion.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_recovery_disabled_returns_not_found_directly() {
    let tv = TestValkey::setup().await;
    let mut svc = build_service(&tv, Arc::new(AllowAllPermissions));
    let metrics = replace_metrics(&mut svc);

    let (session_id, object_id) = create_session_and_layer(
        &tv,
        &svc,
        "disabled-update-session",
        "disabled-update-layer",
    )
    .await;

    // Seed a checkpoint so a regression that DID call the store would
    // surface a real blob (and thus a successful restore) — not a vacuous
    // `Ok(None)` from an empty store. The counting wrapper records every
    // store-side method call.
    let in_mem = Arc::new(InMemoryCheckpointStore::new());
    seed_layer_checkpoint(&in_mem, object_id, &session_id, "disabled-update-layer", 7).await;
    let counting = Arc::new(CallCountingStore::new(in_mem as Arc<dyn CheckpointStore>));
    replace_recovery(
        &mut svc,
        counting.clone() as Arc<dyn CheckpointStore>,
        false,
    );

    // Wipe just the layer mapping (NOT obj:*) so the handler's
    // get_layer_mapping returns None — triggering the engineer's first
    // recovery surface (recovery.rs / handle_update_layer step 4).
    wipe_layer_mapping(&tv, &session_id, "disabled-update-layer").await;

    let status = handle_update_layer(
        &svc,
        update_request(
            &session_id,
            "disabled-update-layer",
            object_id,
            CLIENT_TS,
            CLIENT_ID,
        ),
    )
    .await
    .expect_err("disabled-mode + missing mapping must surface gRPC error");

    assert_eq!(
        status.code(),
        tonic::Code::NotFound,
        "disabled-mode handler MUST surface NotFound directly, got: {}",
        status.code()
    );

    assert_eq!(
        counting.call_count(),
        0,
        "disabled-mode hook MUST NOT invoke any CheckpointStore method (got {} calls)",
        counting.call_count()
    );
    assert_eq!(
        recovery_metric(&metrics, "restored"),
        0,
        "disabled-mode hook MUST NOT increment the `restored` recovery cell"
    );
    assert_eq!(recovery_metric(&metrics, "not_recoverable"), 0);
    assert_eq!(recovery_metric(&metrics, "error"), 0);
    assert_eq!(
        svc.recovery.admission_len(),
        0,
        "disabled-mode hook MUST NOT touch the per-object_id admission map"
    );
}

/// StatLayer disabled-mode: same shape as the UpdateLayer test above —
/// header missing, hook disabled, NotFound surfaces directly, store
/// untouched, metric untouched.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn stat_layer_recovery_disabled_returns_not_found_directly() {
    let tv = TestValkey::setup().await;
    let mut svc = build_service(&tv, Arc::new(AllowAllPermissions));
    let metrics = replace_metrics(&mut svc);

    let (session_id, object_id) =
        create_session_and_layer(&tv, &svc, "disabled-stat-session", "disabled-stat-layer").await;

    let in_mem = Arc::new(InMemoryCheckpointStore::new());
    seed_layer_checkpoint(&in_mem, object_id, &session_id, "disabled-stat-layer", 3).await;
    let counting = Arc::new(CallCountingStore::new(in_mem as Arc<dyn CheckpointStore>));
    replace_recovery(
        &mut svc,
        counting.clone() as Arc<dyn CheckpointStore>,
        false,
    );

    wipe_object_keys(&tv, object_id).await;

    let status = handle_stat_layer(&svc, stat_request(&session_id, "disabled-stat-layer"))
        .await
        .expect_err("disabled-mode + wiped obj must surface gRPC error");

    assert_eq!(
        status.code(),
        tonic::Code::NotFound,
        "disabled-mode StatLayer MUST surface NotFound directly, got: {}",
        status.code()
    );

    assert_eq!(
        counting.call_count(),
        0,
        "disabled-mode hook MUST NOT invoke any CheckpointStore method"
    );
    assert_eq!(recovery_metric(&metrics, "restored"), 0);
    assert_eq!(recovery_metric(&metrics, "not_recoverable"), 0);
    assert_eq!(recovery_metric(&metrics, "error"), 0);
    assert_eq!(svc.recovery.admission_len(), 0);
}

/// SubscribeReadLayer disabled-mode: header missing, hook disabled.
/// The catch-up path's NotFound on `read_object_header` falls through to
/// the `handle_deleted_and_rebootstrap` path (the same path a real
/// CreateLayer-overwrite-during-state-loss takes) under
/// `RebootstrapCause::StateGone`. With no live mapping pointing at a
/// different object_id and no truly-deleted layer, the rebootstrap falls
/// back to NotFound on the stream.
///
/// SubscribeReadLayer delivers NotFound as the FIRST stream item (the
/// handler returns Ok at the top level and the spawned task pushes the
/// status into the stream) — assertion shape matches
/// `subscribe_nonexistent_layer_returns_not_found` in
/// `subscribe_test.rs`.
///
/// # The MESSAGE is asserted too (b.hxk)
///
/// Disabled mode is the cheapest driver for the `StateGone` terminal arm —
/// the hook short-circuits without touching the store, so nothing else can
/// influence what the subscriber is told — and what it is told used to be
/// `layer '<name>' was deleted`. That is a false claim about durable data
/// on every one of the eleven call sites that reach this arm: nothing on
/// this path observed a deletion, the mapping still names the object, and
/// under `LIVE_EDIT_CHECKPOINT_ENABLED=false` the service has formed no
/// opinion about the layer at all. Only the affirmative `deleted` sentinel
/// read off the object's stream licenses that wording, and
/// `subscribe_deleted_sentinel_terminal_status_says_the_layer_was_deleted_*`
/// (one arm per in-window catch-up builder) in
/// `tests/integration/grpc/subscribe_test.rs` are the arms that own it.
///
/// The status CODE is unchanged and deliberately so: there is no state to
/// serve and no other object to bootstrap onto, so NOT_FOUND is right. Only
/// the claim moves.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn subscribe_read_layer_recovery_disabled_returns_not_found_directly() {
    let tv = TestValkey::setup().await;
    let mut svc = build_service(&tv, Arc::new(AllowAllPermissions));
    let metrics = replace_metrics(&mut svc);

    let (session_id, object_id) = create_session_and_layer(
        &tv,
        &svc,
        "disabled-subscribe-session",
        "disabled-subscribe-layer",
    )
    .await;

    let in_mem = Arc::new(InMemoryCheckpointStore::new());
    seed_layer_checkpoint(
        &in_mem,
        object_id,
        &session_id,
        "disabled-subscribe-layer",
        5,
    )
    .await;
    let counting = Arc::new(CallCountingStore::new(in_mem as Arc<dyn CheckpointStore>));
    replace_recovery(
        &mut svc,
        counting.clone() as Arc<dyn CheckpointStore>,
        false,
    );

    wipe_object_keys(&tv, object_id).await;

    // Subscriber sends have_object_id+have_version matching the live
    // mapping so catch-up enters the same-object path and hits the
    // header-read NotFound branch we want to exercise. A `0/0`
    // subscription would go through the brand-new (Regime 1) path that
    // also reads the header but via `catch_up_brand_new`.
    let response = handle_subscribe_read_layer(
        &svc,
        subscribe_request(&session_id, "disabled-subscribe-layer", object_id, 1),
    )
    .await
    .expect("handler must return Ok; NotFound is delivered on the stream");

    let mut stream = response.into_inner();
    let first = stream
        .next()
        .await
        .expect("stream must yield the NotFound status");
    let status = first.expect_err("first stream item must be an error");
    assert_eq!(
        status.code(),
        tonic::Code::NotFound,
        "disabled-mode SubscribeReadLayer MUST surface NotFound directly, got: {}",
        status.code()
    );
    assert_eq!(
        status.message(),
        "layer 'disabled-subscribe-layer' state was lost and could not be recovered",
        "b.hxk: the mapping still names this object and nothing observed a deletion, so the \
         terminal status MUST NOT claim the layer was deleted. `was deleted` here is a false \
         claim about durable data — and under disabled mode the service has not even consulted \
         a backup, so it is a claim it cannot make in any configuration"
    );

    assert_eq!(
        counting.call_count(),
        0,
        "disabled-mode hook MUST NOT invoke any CheckpointStore method"
    );
    assert_eq!(recovery_metric(&metrics, "restored"), 0);
    assert_eq!(recovery_metric(&metrics, "not_recoverable"), 0);
    assert_eq!(recovery_metric(&metrics, "error"), 0);
    assert_eq!(svc.recovery.admission_len(), 0);
}

/// The sibling of the test above on the OTHER `StateGone` terminal arm: the one
/// reached when the mapping re-read inside `handle_deleted_and_rebootstrap`
/// answers `None`.
///
/// # Why this arm needs its own test
///
/// `RebootstrapCause` crosses `handle_deleted_and_rebootstrap`'s two terminal
/// arms independently — mapping-absent and same-`object_id` — and each arm
/// spells its own message. The test above pins the same-`object_id` arm (mapping
/// still names the object → `state was lost and could not be recovered`); this
/// one pins the mapping-absent arm, whose wording widens the claim to cover the
/// mapping as well (`state and its session mapping are gone and could not be
/// recovered`). Both are produced by `state_gone_not_found`, and the reason it
/// is ONE function is that the two spellings must not drift; a test on only one
/// of them lets the other drift freely, which is exactly the gap b.hxk was
/// filed about in the first place.
///
/// The claim being pinned is a negative one, and it is the whole point of the
/// arm: an absent mapping is NOT independent evidence of a deletion on this
/// cause. The keyspace loss that emptied `obj:*` can equally have taken
/// `le:{sid}:layer:{hex(name)}` with it (the DR scenario), and a `DeleteLayer`
/// racing the state read produces an indistinguishable shape. Neither licenses
/// `was deleted` — only a `deleted` sentinel actually read off the stream does,
/// which is what the `DeletedSentinel` half of this same arm asserts (see
/// `subscribe_deleted_sentinel_terminal_status_says_the_layer_was_deleted_*`).
///
/// # How the shape is reached deterministically
///
/// `run_catch_up_inner` reads the mapping FIRST and returns
/// `layer '<name>' not found in session '<sid>'` when it is already absent — so
/// wiping the mapping up front never reaches this arm at all. The arm requires
/// the mapping to be PRESENT at the top-of-catch-up read and ABSENT at the
/// re-read, i.e. a wipe landing between the two. `RunCatchUpInnerSeam` (b.xmk /
/// b.msb) parks the subscriber in exactly that window — after the mapping read
/// and the overwrite-detect branch, before `read_object_header` — so the wipe is
/// a rendezvous rather than a race. With the object keys wiped alongside it, the
/// released subscriber's header read answers `NotFound`, disabled-mode recovery
/// short-circuits `NotRecoverable`, and the `StateGone` re-read finds no
/// mapping.
///
/// Disabled mode is used for the same reason the test above uses it: the hook
/// short-circuits without touching the store, so nothing but the arm under test
/// can influence what the subscriber is told.
#[cfg(feature = "test-fault-injection")]
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn subscribe_read_layer_state_and_mapping_gone_returns_not_found_without_claiming_deletion() {
    let tv = TestValkey::setup().await;
    let mut svc = build_service(&tv, Arc::new(AllowAllPermissions));
    let metrics = replace_metrics(&mut svc);

    let seam = Arc::new(live_edit::grpc::subscribe::RunCatchUpInnerSeam::new());
    svc.run_catch_up_inner_seam_for_test = Some(Arc::clone(&seam));

    let (session_id, object_id) = create_session_and_layer(
        &tv,
        &svc,
        "mapping-gone-subscribe-session",
        "mapping-gone-subscribe-layer",
    )
    .await;

    // Seeded exactly as the sibling test seeds it, and for the same reason: a
    // regression that removed the disabled-mode short-circuit would otherwise
    // meet an empty store and report `NotRecoverable` anyway, hiding itself.
    // With a checkpoint present, such a regression restores the layer and this
    // test fails loudly instead.
    let in_mem = Arc::new(InMemoryCheckpointStore::new());
    seed_layer_checkpoint(
        &in_mem,
        object_id,
        &session_id,
        "mapping-gone-subscribe-layer",
        5,
    )
    .await;
    let counting = Arc::new(CallCountingStore::new(in_mem as Arc<dyn CheckpointStore>));
    replace_recovery(
        &mut svc,
        counting.clone() as Arc<dyn CheckpointStore>,
        false,
    );

    // `have_object_id`/`have_version` match the live mapping, which is what
    // selects the same-object path the seam sits on — a `0/0` subscription
    // would take `catch_up_brand_new` and never park.
    let response = handle_subscribe_read_layer(
        &svc,
        subscribe_request(&session_id, "mapping-gone-subscribe-layer", object_id, 1),
    )
    .await
    .expect("handler must return Ok; NotFound is delivered on the stream");

    // Rendezvous 1: the subscriber has read `mapping == object_id` and is
    // parked just before its header read. Only now is the wipe safe to fire.
    tokio::time::timeout(std::time::Duration::from_secs(10), seam.wait_until_parked())
        .await
        .expect(
            "the subscriber must reach the Site-B park within 10 s — a timeout here means it \
             never took the same-object path, so the wipe below would race a subscriber that is \
             somewhere else entirely",
        );

    wipe_object_keys(&tv, object_id).await;
    wipe_layer_mapping(&tv, &session_id, "mapping-gone-subscribe-layer").await;

    // Rendezvous 2: release into a keyspace where both the state and the
    // mapping are gone.
    seam.release().await;

    let mut stream = response.into_inner();
    let first = tokio::time::timeout(std::time::Duration::from_secs(10), stream.next())
        .await
        .expect("the subscription must terminate once the header read answers NotFound")
        .expect("stream must yield the NotFound status");
    let status = first.expect_err("first stream item must be an error");
    assert_eq!(
        status.code(),
        tonic::Code::NotFound,
        "state and mapping both gone leaves nothing to serve and no other object to bootstrap \
         onto, so NOT_FOUND is right, got: {}",
        status.code()
    );
    assert_eq!(
        status.message(),
        "layer 'mapping-gone-subscribe-layer' state and its session mapping are gone and could \
         not be recovered",
        "b.hxk: nothing observed a deletion here — an absent mapping under `StateGone` is as \
         consistent with keyspace loss that took the mapping too as it is with a DeleteLayer \
         that raced the state read, and this arm cannot tell them apart. Claiming `was deleted` \
         asserts a fact about durable data the service does not have; the mapping-PRESENT \
         wording (`state was lost and could not be recovered`) would under-report by naming only \
         the state. The two spellings share one producer so they cannot drift"
    );

    assert_eq!(
        counting.call_count(),
        0,
        "disabled-mode hook MUST NOT invoke any CheckpointStore method — the terminal wording is \
         the subject, and a store consulted here would mean a different arm ran"
    );
    assert_eq!(recovery_metric(&metrics, "restored"), 0);
    assert_eq!(recovery_metric(&metrics, "not_recoverable"), 0);
    assert_eq!(recovery_metric(&metrics, "error"), 0);
    assert_eq!(svc.recovery.admission_len(), 0);
}

// =============================================================================
// Retry-once tests — Restored on first attempt, NotFound on retry.
// =============================================================================

/// UpdateLayer retry-once: the layer mapping is wiped (obj:* kept
/// intact). `get_layer_mapping` returns None → recovery branch fires.
/// Recovery's post-acquire structural-header re-check (step 3 of
/// `try_restore_layer`) sees `obj:hdr` still present and returns
/// `Restored` cheaply — bumping the `restored` metric exactly once. The
/// handler's loop then re-issues `get_layer_mapping`, observes `None`
/// again (mapping is still wiped — apply_checkpoint never restores the
/// mapping, only obj:* state), trips the `recovered_once` latch, and
/// surfaces `NotFound` to the client.
///
/// The `restored` recovery-metric cell MUST equal exactly 1 — proving
/// that exactly ONE recovery attempt landed (not zero, which would mean
/// the recovery branch was never taken; not two, which would mean the
/// retry triggered a second recovery in violation of the at-most-once
/// contract).
///
/// We deliberately seed a checkpoint even though the post-acquire
/// structural-header short-circuit means apply never runs: a regression that
/// removed the step-3 short-circuit would fall through to step 4
/// (`get_latest_layer_checkpoint`), and an empty store would surface as
/// `NotRecoverable` — quietly hiding the regression. With a seeded
/// checkpoint such a regression would manifest as the full apply
/// pipeline running, the `obj:hdr` getting re-written, and the test
/// assertion on `restored = 1` continuing to hold while the retry
/// would succeed (no NotFound) — falsifying the test.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_retry_once_after_restore_second_not_found_surfaces_not_found() {
    let tv = TestValkey::setup().await;
    let mut svc = build_service(&tv, Arc::new(AllowAllPermissions));
    let metrics = replace_metrics(&mut svc);

    let (session_id, object_id) =
        create_session_and_layer(&tv, &svc, "retry-update-session", "retry-update-layer").await;

    let in_mem = Arc::new(InMemoryCheckpointStore::new());
    seed_layer_checkpoint(&in_mem, object_id, &session_id, "retry-update-layer", 11).await;
    replace_recovery(&mut svc, in_mem as Arc<dyn CheckpointStore>, true);

    // Wipe just the layer mapping. obj:* stays intact so the recovery
    // hook's post-acquire structural-header re-check observes the live header and
    // reports Restored without running apply.
    wipe_layer_mapping(&tv, &session_id, "retry-update-layer").await;

    let status = handle_update_layer(
        &svc,
        update_request(
            &session_id,
            "retry-update-layer",
            object_id,
            CLIENT_TS,
            CLIENT_ID,
        ),
    )
    .await
    .expect_err("missing mapping after restore-short-circuit must surface NotFound");

    assert_eq!(
        status.code(),
        tonic::Code::NotFound,
        "UpdateLayer retry-once with second NotFound MUST surface NotFound, got: {}",
        status.code()
    );
    assert_eq!(
        recovery_metric(&metrics, "restored"),
        1,
        "exactly ONE recovery attempt must have reported `restored` \
         (the at-most-once contract pins the handler to one recovery per RPC)"
    );
    assert_eq!(recovery_metric(&metrics, "not_recoverable"), 0);
    assert_eq!(recovery_metric(&metrics, "error"), 0);
}

/// StatLayer retry-once: handler's header-read NotFound triggers recovery
/// once; the engineer-wired b.sxc one-shot latch on `RecoveryHook`
/// synchronously `UNLINK`s `obj:hdr` at the post-Restored boundary,
/// guaranteeing the post-restore retry observes `NotFound` and the
/// `recovered_once` latch surfaces `NotFound` to the client. Replaces
/// the prior `WipeAfterRestoreStore` 16-wiper race that flaked at
/// ~2-3 % on slow CI hosts.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn stat_layer_retry_once_after_restore_second_not_found_surfaces_not_found() {
    let tv = TestValkey::setup().await;
    let mut svc = build_service(&tv, Arc::new(AllowAllPermissions));
    let metrics = replace_metrics(&mut svc);

    let (session_id, object_id) =
        create_session_and_layer(&tv, &svc, "retry-stat-session", "retry-stat-layer").await;

    let in_mem = Arc::new(InMemoryCheckpointStore::new());
    seed_layer_checkpoint(&in_mem, object_id, &session_id, "retry-stat-layer", 9).await;
    // Arm the one-shot post-restore-NotFound latch BEFORE installing
    // the recovery hook on the service. The latch self-disarms via
    // `swap(false, AcqRel)` on the single firing, so only the first
    // restore attempt re-wipes `obj:hdr` — any subsequent (regression)
    // recoveries would land cleanly and the test would falsify.
    let latch = Arc::new(AtomicBool::new(true));
    replace_recovery_with_post_restore_notfound_latch(
        &mut svc,
        in_mem as Arc<dyn CheckpointStore>,
        true,
        Arc::clone(&latch),
    );

    wipe_object_keys(&tv, object_id).await;

    let status = handle_stat_layer(&svc, stat_request(&session_id, "retry-stat-layer"))
        .await
        .expect_err("post-restore latch wipe must drive StatLayer to NotFound");

    assert_eq!(
        status.code(),
        tonic::Code::NotFound,
        "StatLayer retry-once with second NotFound MUST surface NotFound, got: {}",
        status.code()
    );
    assert_eq!(
        recovery_metric(&metrics, "restored"),
        1,
        "exactly ONE recovery attempt must have reported `restored`"
    );
    assert_eq!(recovery_metric(&metrics, "not_recoverable"), 0);
    assert_eq!(recovery_metric(&metrics, "error"), 0);
    // The b.sxc seam self-disarms via `swap(false, AcqRel)` after firing
    // once. If the latch is still armed (`true`), the seam never ran —
    // either the recovery branch did not fire at all, or a regression
    // skipped the firing site. Either case falsifies the test premise.
    assert!(
        !latch.load(Ordering::Acquire),
        "post-restore-NotFound latch MUST have fired (self-disarmed) — \
         armed=true means the seam never ran"
    );
}

/// SubscribeReadLayer retry-once: the retry path differs from
/// StatLayer's in two ways. (1) The handler's catch-up site retries the
/// `read_object_header` call inside `run_catch_up_inner` (and
/// `catch_up_out_of_window`). (2) After a second NotFound on the retry,
/// the catch-up path falls through to `handle_deleted_and_rebootstrap`
/// which re-reads the layer mapping — since the mapping still points at
/// the same (now state-less) `object_id` and no overwrite minted a new
/// one, the handler surfaces `NotFound` to the subscriber via the
/// stream.
///
/// Caller note: `read_object_header` on the catch-up's same-object_id
/// branch returns NotFound when `header.is_empty()` — exactly the
/// post-wipe shape. The post-restore retry then trips the same condition
/// because the b.sxc one-shot latch on `RecoveryHook` has already
/// synchronously `UNLINK`-ed `obj:hdr` at the post-Restored boundary.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn subscribe_read_layer_retry_once_after_restore_second_not_found_surfaces_not_found() {
    let tv = TestValkey::setup().await;
    let mut svc = build_service(&tv, Arc::new(AllowAllPermissions));
    let metrics = replace_metrics(&mut svc);

    let (session_id, object_id) = create_session_and_layer(
        &tv,
        &svc,
        "retry-subscribe-session",
        "retry-subscribe-layer",
    )
    .await;

    let in_mem = Arc::new(InMemoryCheckpointStore::new());
    seed_layer_checkpoint(&in_mem, object_id, &session_id, "retry-subscribe-layer", 13).await;
    // Arm the one-shot post-restore-NotFound latch (b.sxc) on the
    // freshly-installed `RecoveryHook`. The latch self-disarms via
    // `swap(false, AcqRel)` after firing once.
    let latch = Arc::new(AtomicBool::new(true));
    replace_recovery_with_post_restore_notfound_latch(
        &mut svc,
        in_mem as Arc<dyn CheckpointStore>,
        true,
        Arc::clone(&latch),
    );

    wipe_object_keys(&tv, object_id).await;

    // have_object_id=object_id forces the same-object path through
    // `run_catch_up_inner`'s `read_object_header`, where the recovery
    // branch is wired.
    let response = handle_subscribe_read_layer(
        &svc,
        subscribe_request(&session_id, "retry-subscribe-layer", object_id, 1),
    )
    .await
    .expect("handler must return Ok; NotFound is delivered on the stream");

    let mut stream = response.into_inner();
    let first = stream
        .next()
        .await
        .expect("stream must yield the NotFound status");
    let status = first.expect_err("first stream item must be an error");
    assert_eq!(
        status.code(),
        tonic::Code::NotFound,
        "SubscribeReadLayer retry-once with second NotFound MUST surface NotFound, got: {}",
        status.code()
    );
    assert_eq!(
        recovery_metric(&metrics, "restored"),
        1,
        "exactly ONE recovery attempt must have reported `restored`"
    );
    assert_eq!(recovery_metric(&metrics, "not_recoverable"), 0);
    assert_eq!(recovery_metric(&metrics, "error"), 0);
    // The b.sxc seam self-disarms via `swap(false, AcqRel)` after firing
    // once. Armed=true after the RPC means the seam never ran.
    assert!(
        !latch.load(Ordering::Acquire),
        "post-restore-NotFound latch MUST have fired (self-disarmed) — \
         armed=true means the seam never ran"
    );
}

// =============================================================================
// Transient-error tests — checkpoint backend transient failure → UNAVAILABLE.
// =============================================================================

/// UpdateLayer adapter transient: layer mapping AND obj:* both wiped
/// so the recovery hook falls THROUGH the post-acquire structural-header re-check
/// (step 3 sees no header) and reaches step 4
/// (`get_latest_layer_checkpoint`). The `AlwaysTransientStore` returns
/// `CheckpointError::Transient(_)`, the hook returns
/// `RecoveryOutcome::Error(_)`, and the handler maps to gRPC
/// `Code::Unavailable` (the standard transient-error contract) — carrying
/// `UPSTREAM_UNAVAILABLE`, since the restore precedes the commit — while
/// incrementing the `error` recovery-metric cell.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_adapter_transient_error_returns_unavailable() {
    let tv = TestValkey::setup().await;
    let mut svc = build_service(&tv, Arc::new(AllowAllPermissions));
    let metrics = replace_metrics(&mut svc);

    let (session_id, object_id) = create_session_and_layer(
        &tv,
        &svc,
        "transient-update-session",
        "transient-update-layer",
    )
    .await;

    replace_recovery(
        &mut svc,
        Arc::new(AlwaysTransientStore) as Arc<dyn CheckpointStore>,
        true,
    );

    // Wipe BOTH the layer mapping and obj:* keys so the recovery hook's
    // post-acquire structural-header re-check (step 3) falls through to step 4 —
    // the `get_latest_layer_checkpoint` call that hits the Transient
    // store and exercises the `Error(_)` arm we are pinning.
    wipe_layer_mapping(&tv, &session_id, "transient-update-layer").await;
    wipe_object_keys(&tv, object_id).await;

    let status = handle_update_layer(
        &svc,
        update_request(
            &session_id,
            "transient-update-layer",
            object_id,
            CLIENT_TS,
            CLIENT_ID,
        ),
    )
    .await
    .expect_err("transient checkpoint backend must surface gRPC error");

    assert_eq!(
        status.code(),
        tonic::Code::Unavailable,
        "transient checkpoint backend MUST map to Code::Unavailable, got: {}",
        status.code()
    );
    // The lazy restore runs at the mapping resolve, before the commit marker,
    // so the refusal carries the b.xd1 pre-commit hint: this call applied
    // nothing.
    crate::common::refusal_detail::assert_upstream_unavailable(&status);
    assert_eq!(
        recovery_metric(&metrics, "error"),
        1,
        "transient error MUST increment the `error` recovery cell exactly once"
    );
    assert_eq!(recovery_metric(&metrics, "restored"), 0);
    assert_eq!(recovery_metric(&metrics, "not_recoverable"), 0);
}

/// StatLayer adapter transient: analogous to the UpdateLayer transient
/// test — the recovery hook surfaces `CheckpointError::Transient(_)` and
/// the handler maps to `Code::Unavailable`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn stat_layer_adapter_transient_error_returns_unavailable() {
    let tv = TestValkey::setup().await;
    let mut svc = build_service(&tv, Arc::new(AllowAllPermissions));
    let metrics = replace_metrics(&mut svc);

    let (session_id, object_id) =
        create_session_and_layer(&tv, &svc, "transient-stat-session", "transient-stat-layer").await;

    replace_recovery(
        &mut svc,
        Arc::new(AlwaysTransientStore) as Arc<dyn CheckpointStore>,
        true,
    );

    wipe_object_keys(&tv, object_id).await;

    let status = handle_stat_layer(&svc, stat_request(&session_id, "transient-stat-layer"))
        .await
        .expect_err("transient checkpoint backend must surface gRPC error");

    assert_eq!(
        status.code(),
        tonic::Code::Unavailable,
        "transient checkpoint backend on StatLayer MUST map to Code::Unavailable, got: {}",
        status.code()
    );
    assert_eq!(
        recovery_metric(&metrics, "error"),
        1,
        "transient error MUST increment the `error` recovery cell exactly once"
    );
    assert_eq!(recovery_metric(&metrics, "restored"), 0);
    assert_eq!(recovery_metric(&metrics, "not_recoverable"), 0);
}

/// SubscribeReadLayer adapter transient (PM-review patch): the catch-up
/// site's recovery branch surfaces `RecoveryOutcome::Error(_)` as gRPC
/// `Code::Unavailable` on the stream — symmetric with the UpdateLayer
/// and StatLayer transient tests. Adds the third RPC's transient coverage
/// the PM patch explicitly called for.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn subscribe_read_layer_adapter_transient_error_returns_unavailable() {
    let tv = TestValkey::setup().await;
    let mut svc = build_service(&tv, Arc::new(AllowAllPermissions));
    let metrics = replace_metrics(&mut svc);

    let (session_id, object_id) = create_session_and_layer(
        &tv,
        &svc,
        "transient-subscribe-session",
        "transient-subscribe-layer",
    )
    .await;

    replace_recovery(
        &mut svc,
        Arc::new(AlwaysTransientStore) as Arc<dyn CheckpointStore>,
        true,
    );

    wipe_object_keys(&tv, object_id).await;

    let response = handle_subscribe_read_layer(
        &svc,
        subscribe_request(&session_id, "transient-subscribe-layer", object_id, 1),
    )
    .await
    .expect("handler must return Ok; Unavailable is delivered on the stream");

    let mut stream = response.into_inner();
    let first = stream
        .next()
        .await
        .expect("stream must yield the Unavailable status");
    let status = first.expect_err("first stream item must be an error");
    assert_eq!(
        status.code(),
        tonic::Code::Unavailable,
        "transient checkpoint backend on SubscribeReadLayer MUST map to Code::Unavailable, got: {}",
        status.code()
    );
    assert_eq!(
        recovery_metric(&metrics, "error"),
        1,
        "transient error MUST increment the `error` recovery cell exactly once"
    );
    assert_eq!(recovery_metric(&metrics, "restored"), 0);
    assert_eq!(recovery_metric(&metrics, "not_recoverable"), 0);
}

/// SubscribeReadLayer, budget-inconclusive selection (b.hxk): a store that
/// **fails nothing** but is too slow to read the one full generation it
/// advertises must end the stream `UNAVAILABLE`, not `NOT_FOUND`.
///
/// # Why this RPC, next to its transient sibling above
///
/// The sibling drives a store that ERRORS, so its `RecoveryOutcome::Error` is
/// carried in from outside. This one drives a store that answers every call it
/// completes correctly: the listing succeeds and advertises a healthy `Base`,
/// and the only fault is latency. The `Error` therefore has to be MINTED by the
/// selection itself, from the fact that it never adjudicated that base — which
/// is the whole of the fix. Pre-b.hxk this trace produced
/// `Exhaustion::AllBroken` → `Ok(None)` → `RecoveryOutcome::NotRecoverable`,
/// and the subscriber's stream ended `NOT_FOUND`.
///
/// SubscribeReadLayer is where that matters most in production, and the reason
/// is structural rather than a matter of taste: it is the one deadline-exempt
/// RPC (CLAUDE.md, "Streaming Chunk Pattern"), so it is the RPC most likely to
/// still be running when a checkpoint store degrades — and a `NOT_FOUND` on a
/// stream is not merely a wrong word. The standard client retry policy does not
/// retry it, so a subscriber told its layer is gone stays gone, while an
/// `UNAVAILABLE` invites the reconnect that succeeds once the store recovers.
///
/// # The fixture's arithmetic
///
/// [`SUBSCRIBE_INCONCLUSIVE_RESERVE`] is both knobs, so the budget is two
/// reserves. The listing holds exactly one generation, so the walk has exactly
/// one candidate: its un-gated first fetch is allowanced `min(reserve,
/// remaining)`, wedges for one reserve, caches `LinkFact::Wedged`, and the
/// selection drains with that base unadjudicated. One reserve of wall clock,
/// well inside the outer preemptive budget of two — so the verdict asserted here
/// is the COOPERATIVE one, not a plain timeout.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn subscribe_read_layer_budget_inconclusive_selection_returns_unavailable_not_not_found() {
    /// Dialled far below the 30 s production reserve because it is the arm's
    /// wall-clock cost: one wedged fetch. Must stay comfortably above the
    /// latency of a healthy fetch — which, since this fixture's fetch never
    /// completes at all, only has to hold for the surrounding Valkey calls.
    ///
    /// # Why 2 s and not the 500 ms this shipped at
    ///
    /// The value is a **scheduler-jitter margin**, and this arm runs at
    /// production's ratio (`budget == 2 x reserve`, asserted by
    /// [`replace_recovery_with_reserve`]) so it has no headroom by construction
    /// — the same reasoning that sizes `SHIPPED_RATIO_RESERVE` at 3 s in
    /// `restore_fault_injection_matrix_test.rs`, and the constant to keep this
    /// one in step with.
    ///
    /// The two timers race deliberately: the COOPERATIVE verdict is ready at one
    /// reserve (the wedged fetch's `timeout_at` bound), and the outer PREEMPTIVE
    /// wrapper in `RecoveryHook::try_restore_layer` fires at two. The assertions
    /// below are only meaningful if the cooperative one wins, so the margin the
    /// arm actually has is the one reserve between them. At 500 ms that margin
    /// was 500 ms of real wall clock on a shared, contended runner — 6x tighter
    /// than the matrix arm's, for a claim that is if anything more delicate,
    /// because this one crosses a real Valkey round trip and a tonic stream on
    /// the way.
    ///
    /// The failure that buys is confusing rather than loud: if the wrapper wins,
    /// the status is still `Unavailable` (a preemptive elapse routes through
    /// `Transient` too) and `error` still ticks, so the first three assertions
    /// stay green and only `exhausted_budget == 1` goes red — the selection
    /// never reached its own exhaustion, so no lineage cell moved at all. A
    /// reader triaging that sees a metric-wiring failure, not a load flake.
    /// 2 s quadruples the margin; the arm's runtime is this constant, so the
    /// cost is ~1.5 s against a 120 s hang-kill budget.
    const SUBSCRIBE_INCONCLUSIVE_RESERVE: std::time::Duration =
        std::time::Duration::from_millis(2_000);

    let tv = TestValkey::setup().await;
    let mut svc = build_service(&tv, Arc::new(AllowAllPermissions));
    let metrics = replace_metrics(&mut svc);

    let (session_id, object_id) = create_session_and_layer(
        &tv,
        &svc,
        "inconclusive-subscribe-session",
        "inconclusive-subscribe-layer",
    )
    .await;

    replace_recovery_with_reserve(
        &mut svc,
        Arc::new(WedgedBaseListingStore) as Arc<dyn CheckpointStore>,
        true,
        SUBSCRIBE_INCONCLUSIVE_RESERVE,
    );

    wipe_object_keys(&tv, object_id).await;

    let response = handle_subscribe_read_layer(
        &svc,
        subscribe_request(&session_id, "inconclusive-subscribe-layer", object_id, 1),
    )
    .await
    .expect("handler must return Ok; the terminal status is delivered on the stream");

    let mut stream = response.into_inner();
    // `WedgedBaseListingStore` parks its base listing on `pending()` forever, so
    // this read only completes because the per-fetch `timeout_at` clamp and the
    // outer preemptive wrapper bound it. Cap it rather than letting a regression
    // in BOTH ride to nextest's 120 s SIGKILL, which prints none of the
    // diagnostics below.
    let first = tokio::time::timeout(std::time::Duration::from_secs(10), stream.next())
        .await
        .expect(
            "the subscription must reach a terminal status within 10 s — a timeout here means the \
             wedged base listing was never bounded, so neither the per-fetch clamp nor the \
             preemptive wrapper is holding",
        )
        .expect("stream must yield a terminal status");
    let status = first.expect_err("first stream item must be an error");

    assert_eq!(
        status.code(),
        tonic::Code::Unavailable,
        "THE CLAIM (b.hxk). The selection never READ the layer's only full generation, so it \
         holds no evidence that the layer is unrestorable and must not answer as though it did. \
         A `NotFound` here is the defect: it tells a subscriber on the one deadline-exempt RPC \
         that its durable data is gone, and the standard client retry policy does not retry it. \
         Got {:?}: {}",
        status.code(),
        status.message()
    );

    assert_eq!(
        recovery_metric(&metrics, "error"),
        1,
        "an inconclusive exhaustion routes through `CheckpointError::Transient` -> \
         `RecoveryOutcome::Error`, which is what carries the retriable status"
    );
    assert_eq!(
        recovery_metric(&metrics, "not_recoverable"),
        0,
        "and `not_recoverable` MUST stay flat. That cell is the affirmative claim `nothing here \
         can be restored`, which this selection cannot make about a generation it never read"
    );
    assert_eq!(recovery_metric(&metrics, "restored"), 0);
    assert_eq!(
        read_restore_lineage_counter(&metrics, "exhausted_budget"),
        1,
        "the inconclusive exhaustion gets its OWN outcome cell — an operator triaging a restore \
         wave must be able to separate `these layers have no usable backup` from `this store was \
         too slow to tell`"
    );
    assert_eq!(
        read_restore_lineage_counter(&metrics, "exhausted"),
        0,
        "and the CONCLUSIVE cell must stay flat: a rise in it does indicate checkpoint loss, \
         which is exactly what did not happen here"
    );
}

// =============================================================================
// Non-retriable-error tests (b.33e) — Permanent / NotFound / Conflict /
// NotImplemented → INTERNAL, at each of the four `RecoveryOutcome::Error` arms.
// (Four since b.f1v merged `handle_update_layer`'s message-equality arm into
// the variant-keyed one; the module header carries the arm → test table.)
// =============================================================================

/// StatLayer restore-source-selection `Permanent`: the store's
/// `list_layer_checkpoints` — the FIRST call `select_restore_source`
/// (SR-2.5) makes, and a call the recovery path did not use to make at
/// all — fails `CheckpointError::Permanent`. Pins two things at once.
///
/// **First**, that the error propagates out of selection to the handler
/// boundary as an error at all, rather than being folded into the
/// `Ok(None)` → `NotRecoverable` → `NOT_FOUND` verdict that an empty
/// listing takes. That is what the `error`-cell metric assertion below
/// guards, and it is unchanged from this test's original form.
///
/// **Second — the b.33e contract — that the pinned code is
/// `Code::Internal`, not `Code::Unavailable`.** SDD §18.1 reserves
/// `Code::Unavailable` (14) for retriable conditions, and a `Permanent`
/// means "MUST NOT retry": a corrupt blob or malformed persisted meta
/// cannot be healed by re-issuing the RPC, so surfacing it as retriable
/// invited the client's standard retry policy to hammer a failure that
/// cannot succeed. All four arms now route through
/// `ServiceError::from_recovery_error` (`src/error.rs`), which classifies
/// `Transient` → `Unavailable` and every other kind → `Internal`,
/// matching what the session-recovery sibling in
/// `src/grpc/session_cache.rs` already did.
///
/// The wire message is asserted too: `ServiceError::Internal` →
/// `Status::internal` deliberately redacts to `"internal error"` so the
/// backend's error string never reaches the client, while the operator's
/// ERROR log keeps it verbatim. A regression that surfaced the raw
/// `"checkpoint recovery failed for object …"` text on the wire would be
/// a (minor) information leak and is caught here.
///
/// # History
///
/// This test was `stat_layer_selection_permanent_error_returns_unavailable`
/// and asserted `Code::Unavailable`, with a rustdoc that recorded the
/// uniform-`Unavailable` mapping as *observed, not endorsed* and named
/// itself as the test to update when the mapping was fixed. b.33e is that
/// fix; this is that update. The old rustdoc also enumerated **four**
/// variant-blind arms — an undercount corrected to **five** by the b.33e
/// PM review (the `handle_update_layer` mapping-resolve loop was missing),
/// and back to **four** since b.f1v deleted the message-equality arm. The
/// module header above carries the full arm → test table.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn stat_layer_selection_permanent_error_returns_internal() {
    let tv = TestValkey::setup().await;
    let mut svc = build_service(&tv, Arc::new(AllowAllPermissions));
    let metrics = replace_metrics(&mut svc);

    let (session_id, object_id) = create_session_and_layer(
        &tv,
        &svc,
        "selection-permanent-stat-session",
        "selection-permanent-stat-layer",
    )
    .await;

    replace_recovery(
        &mut svc,
        Arc::new(SelectionErrorStore::permanent()) as Arc<dyn CheckpointStore>,
        true,
    );

    // Wipe obj:* so `stat_layer_for_object_id` returns NotFound, the
    // recovery branch fires, and the hook's post-acquire structural-header re-check
    // (step 3) falls through to selection — where the Permanent is.
    wipe_object_keys(&tv, object_id).await;

    let status = handle_stat_layer(
        &svc,
        stat_request(&session_id, "selection-permanent-stat-layer"),
    )
    .await
    .expect_err("permanent selection failure must surface a gRPC error");

    assert_eq!(
        status.code(),
        tonic::Code::Internal,
        "a Permanent from the selection listing MUST surface as Code::Internal \
         (non-retriable, SDD §18.1) — Unavailable here means the handler arm went \
         back to being variant-blind (b.33e), and NotFound would mean selection \
         swallowed the error into NotRecoverable. Got: {}",
        status.code()
    );
    assert_eq!(
        status.message(),
        "internal error",
        "ServiceError::Internal → Status::internal redacts the body; the backend's \
         error string belongs in the operator log, not on the wire. Got: {:?}",
        status.message()
    );
    assert_eq!(
        recovery_metric(&metrics, "error"),
        1,
        "a permanent selection failure MUST end the attempt as `error`, exactly once"
    );
    assert_eq!(
        recovery_metric(&metrics, "not_recoverable"),
        0,
        "a store error is NOT the `not_recoverable` verdict an empty listing earns"
    );
    assert_eq!(recovery_metric(&metrics, "restored"), 0);
}

/// UpdateLayer **arm 1** — the layer-name → `object_id` mapping-resolve loop
/// (`src/grpc/layer.rs`), the arm the pre-b.33e enumeration missed entirely.
///
/// Same recipe as `update_layer_adapter_transient_error_returns_unavailable`
/// (wipe BOTH the mapping and `obj:*`, so `get_layer_mapping` answers `None`
/// and the hook's post-acquire structural-header re-check falls through to selection),
/// with the store swapped for a `Permanent` one. That pairing is the point:
/// the two tests differ ONLY in the injected `CheckpointError` kind and must
/// therefore differ ONLY in the resulting status code. A fix that classified
/// the other three arms but left this one variant-blind — the exact failure
/// mode the PM-review undercount would have produced — passes every other
/// test in this file and fails here.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_mapping_resolve_permanent_error_returns_internal() {
    let tv = TestValkey::setup().await;
    let mut svc = build_service(&tv, Arc::new(AllowAllPermissions));
    let metrics = replace_metrics(&mut svc);

    let (session_id, object_id) = create_session_and_layer(
        &tv,
        &svc,
        "permanent-map-update-session",
        "permanent-map-update-layer",
    )
    .await;

    replace_recovery(
        &mut svc,
        Arc::new(SelectionErrorStore::permanent()) as Arc<dyn CheckpointStore>,
        true,
    );

    wipe_layer_mapping(&tv, &session_id, "permanent-map-update-layer").await;
    wipe_object_keys(&tv, object_id).await;

    let status = handle_update_layer(
        &svc,
        update_request(
            &session_id,
            "permanent-map-update-layer",
            object_id,
            CLIENT_TS,
            CLIENT_ID,
        ),
    )
    .await
    .expect_err("permanent checkpoint backend must surface gRPC error");

    assert_eq!(
        status.code(),
        tonic::Code::Internal,
        "a Permanent on the UpdateLayer mapping-resolve arm MUST surface as \
         Code::Internal (b.33e arm 1 — the arm the original four-arm enumeration \
         missed); Unavailable means this arm is still variant-blind. Got: {}",
        status.code()
    );
    assert_eq!(status.message(), "internal error");
    assert_eq!(
        recovery_metric(&metrics, "error"),
        1,
        "the recovery attempt MUST actually have run and ended as `error`, exactly \
         once — a zero here would make the status assertion above vacuous"
    );
    assert_eq!(recovery_metric(&metrics, "restored"), 0);
    assert_eq!(recovery_metric(&metrics, "not_recoverable"), 0);
}

/// UpdateLayer **arm 2** — the merged post-commit `NotFound` arm
/// (`src/grpc/layer.rs`), reached from its EMPTY-HEADER origin.
///
/// Wipes `obj:*` **only**, leaving the layer mapping intact. That is the
/// natural symptom of a Valkey state wipe and the discriminator against arm 1:
/// the mapping resolves normally, so the handler proceeds to the commit, where
/// both the Lua and txn paths surface an empty `obj:hdr` HGETALL as
/// `ServiceError::NotFound(EMPTY_HEADER_SENTINEL)` — landing on arm 2 rather
/// than arm 1. The hook's post-acquire structural-header re-check then sees no
/// header and falls through to selection, where the `Permanent` is.
///
/// The `not_recoverable == 0` assertion is what keeps this honest: if a
/// refactor moved the empty-header surface off the recovery branch entirely,
/// the RPC would surface its storage error with every metric cell at zero and
/// the status assertion alone would not say why.
///
/// # What this test absorbed (b.f4t sub-task 4)
///
/// It used to be "arm 3", one of two near-identical tests: the other,
/// `update_layer_notfound_arm_permanent_error_returns_internal`, drove "arm 2"
/// by queueing a synthetic `ServiceError::NotFound` through the
/// `test-fault-injection` `LayerStore` seam. Its stated justification was that
/// "`update_layer` never naturally emits `ServiceError::NotFound` (a wiped
/// `obj:hdr` surfaces as `FailedPrecondition`)" — falsified first by b.ph9's
/// `mapping_deleted` verdict and then decisively by b.f1v, which flipped the
/// empty-header surface to `NotFound` and deleted the sentinel arm outright.
/// The two tests then reached the SAME `match` arm on the SAME variant, one
/// through a seam and one without, so the seam-gated copy was deleted: a
/// feature-gated near-duplicate is strictly worse than the seam-free original,
/// because a missing feature compiles it out silently and the run reports green
/// having covered nothing.
///
/// The arm's OTHER origin — the mapping-resolve loop returning `None`
/// (`src/grpc/layer.rs`, a genuinely distinct arm) — keeps its own test,
/// [`update_layer_mapping_resolve_permanent_error_returns_internal`].
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_empty_header_permanent_error_returns_internal() {
    let tv = TestValkey::setup().await;
    let mut svc = build_service(&tv, Arc::new(AllowAllPermissions));
    let metrics = replace_metrics(&mut svc);

    let (session_id, object_id) = create_session_and_layer(
        &tv,
        &svc,
        "permanent-hdr-update-session",
        "permanent-hdr-update-layer",
    )
    .await;

    replace_recovery(
        &mut svc,
        Arc::new(SelectionErrorStore::permanent()) as Arc<dyn CheckpointStore>,
        true,
    );

    // Mapping deliberately NOT wiped — that is what routes this to the
    // commit's own NotFound arm instead of the mapping-resolve loop.
    wipe_object_keys(&tv, object_id).await;

    let status = handle_update_layer(
        &svc,
        update_request(
            &session_id,
            "permanent-hdr-update-layer",
            object_id,
            CLIENT_TS,
            CLIENT_ID,
        ),
    )
    .await
    .expect_err("permanent checkpoint backend must surface gRPC error");

    assert_eq!(
        status.code(),
        tonic::Code::Internal,
        "a Permanent on the UpdateLayer merged NotFound arm, reached from its empty-header \
         origin, MUST surface as Code::Internal (b.33e); Unavailable means the arm is still \
         variant-blind, and FailedPrecondition means the empty-header surface stopped reaching \
         the recovery branch at all. Got: {}",
        status.code()
    );
    assert_eq!(status.message(), "internal error");
    assert_eq!(
        recovery_metric(&metrics, "error"),
        1,
        "the recovery attempt MUST actually have run and ended as `error`, exactly once"
    );
    assert_eq!(recovery_metric(&metrics, "restored"), 0);
    assert_eq!(recovery_metric(&metrics, "not_recoverable"), 0);
}

/// SubscribeReadLayer **arm 4** — `try_recover_after_notfound`
/// (`src/grpc/subscribe.rs`), whose message prefix is `"layer-state recovery
/// failed"` rather than the layer path's `"checkpoint recovery failed"`.
///
/// Mirrors `subscribe_read_layer_adapter_transient_error_returns_unavailable`
/// exactly, store kind aside. The status arrives ON THE STREAM, not from the
/// handler call — `handle_subscribe_read_layer` returns `Ok` and the first
/// stream item is the error — which is why the handler result is `expect`ed
/// rather than `expect_err`.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn subscribe_read_layer_permanent_error_returns_internal() {
    let tv = TestValkey::setup().await;
    let mut svc = build_service(&tv, Arc::new(AllowAllPermissions));
    let metrics = replace_metrics(&mut svc);

    let (session_id, object_id) = create_session_and_layer(
        &tv,
        &svc,
        "permanent-subscribe-session",
        "permanent-subscribe-layer",
    )
    .await;

    replace_recovery(
        &mut svc,
        Arc::new(SelectionErrorStore::permanent()) as Arc<dyn CheckpointStore>,
        true,
    );

    wipe_object_keys(&tv, object_id).await;

    let response = handle_subscribe_read_layer(
        &svc,
        subscribe_request(&session_id, "permanent-subscribe-layer", object_id, 1),
    )
    .await
    .expect("handler must return Ok; the status is delivered on the stream");

    let mut stream = response.into_inner();
    let first = stream
        .next()
        .await
        .expect("stream must yield the Internal status");
    let status = first.expect_err("first stream item must be an error");

    assert_eq!(
        status.code(),
        tonic::Code::Internal,
        "a Permanent on the SubscribeReadLayer recovery arm MUST surface as \
         Code::Internal (b.33e arm 4); Unavailable means the arm is still \
         variant-blind. Got: {}",
        status.code()
    );
    assert_eq!(
        status.message(),
        "internal error",
        "the `layer-state recovery failed for object …` detail belongs in the \
         operator log, not on the wire. Got: {:?}",
        status.message()
    );
    assert_eq!(
        recovery_metric(&metrics, "error"),
        1,
        "the recovery attempt MUST actually have run and ended as `error`, exactly once"
    );
    assert_eq!(recovery_metric(&metrics, "restored"), 0);
    assert_eq!(recovery_metric(&metrics, "not_recoverable"), 0);
}

/// Shared body for the remaining non-retriable **kinds** on the StatLayer arm.
///
/// The three callers below differ only in the `CheckpointError` kind the store
/// answers, so the arrange/act/assert body is written once here rather than
/// copy-pasted three times (test-review guide, "copy-paste tests with one-line
/// differences"). `kind_label` names the kind in every panic message and also
/// seeds the session / layer names, keeping the callers' Valkey keys disjoint.
///
/// `error == 1` is the load-bearing assertion alongside the status code: the
/// regression this guards against is a `match` that folded the kind into the
/// `NotRecoverable` verdict, which surfaces `NOT_FOUND` and ticks
/// `not_recoverable` instead — indistinguishable from "this layer was never
/// checkpointed" without the metric.
async fn assert_stat_layer_kind_returns_internal(store: SelectionErrorStore, kind_label: &str) {
    let tv = TestValkey::setup().await;
    let mut svc = build_service(&tv, Arc::new(AllowAllPermissions));
    let metrics = replace_metrics(&mut svc);

    let session_name = format!("selection-{kind_label}-stat-session");
    let layer_name = format!("selection-{kind_label}-stat-layer");
    let (session_id, object_id) =
        create_session_and_layer(&tv, &svc, &session_name, &layer_name).await;

    replace_recovery(&mut svc, Arc::new(store) as Arc<dyn CheckpointStore>, true);

    wipe_object_keys(&tv, object_id).await;

    // `.err()` + `unwrap_or_else` rather than `expect_err`: the panic message
    // interpolates `{kind_label}`, and `expect_err` takes a `&str` it cannot
    // format. (`expect_err` itself would compile fine here — sibling tests in
    // this file use it on the identical result type.)
    let status = handle_stat_layer(&svc, stat_request(&session_id, &layer_name))
        .await
        .err()
        .unwrap_or_else(|| {
            panic!("a CheckpointError::{kind_label} selection failure must surface a gRPC error")
        });

    assert_eq!(
        status.code(),
        tonic::Code::Internal,
        "a CheckpointError::{kind_label} MUST surface as Code::Internal — it is \
         non-retriable per SDD §18.1, so Unavailable would invite a client retry \
         that cannot succeed, and NotFound would mean the kind was folded into \
         the NotRecoverable verdict. Got: {}",
        status.code()
    );
    assert_eq!(
        status.message(),
        "internal error",
        "Internal redacts the backend detail from the wire. Got: {:?}",
        status.message()
    );
    assert_eq!(
        recovery_metric(&metrics, "error"),
        1,
        "a {kind_label} selection failure MUST end the attempt as `error`, exactly \
         once — a zero here (with `not_recoverable` at 1) is the regression that \
         folds a malfunctioning store into 'never checkpointed'"
    );
    assert_eq!(
        recovery_metric(&metrics, "not_recoverable"),
        0,
        "a store error is NOT the `not_recoverable` verdict an empty listing earns"
    );
    assert_eq!(recovery_metric(&metrics, "restored"), 0);
}

/// StatLayer selection `NotFound`. Plain absence is `Ok(None)` and is routed
/// to `NotRecoverable` long before the `Error` arm, so a `NotFound` *error*
/// here means an inconsistent store — not something a retry heals.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn stat_layer_selection_not_found_error_returns_internal() {
    assert_stat_layer_kind_returns_internal(SelectionErrorStore::not_found(), "not-found").await;
}

/// StatLayer selection `Conflict`. A concurrent-write conflict on a *read*
/// path is a store-consistency fault, not a retriable backend blip.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn stat_layer_selection_conflict_error_returns_internal() {
    assert_stat_layer_kind_returns_internal(SelectionErrorStore::conflict(), "conflict").await;
}

/// StatLayer selection `NotImplemented`. Defensive: every wired adapter
/// implements the layer-read methods, so this kind reaching a handler is
/// adapter-surface drift — a permanent condition, and retrying it forever is
/// the worst possible response.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn stat_layer_selection_not_implemented_error_returns_internal() {
    assert_stat_layer_kind_returns_internal(
        SelectionErrorStore::not_implemented(),
        "not-implemented",
    )
    .await;
}

// =============================================================================
// Recovery-branch scoping tests — non-NotFound failures do NOT invoke recovery.
// =============================================================================
//
// The wired recovery branches in `handle_update_layer`, `handle_stat_layer`,
// and `handle_subscribe_read_layer` fire EXCLUSIVELY on the
// `ServiceError::NotFound` VARIANT (and, for UpdateLayer, on
// `get_layer_mapping` returning `None`). Every other `ServiceError` variant —
// `InvalidArgument`, `PermissionDenied`, `FailedPrecondition`, `Unavailable`,
// `Aborted`, `Internal` — MUST propagate to the client UNCHANGED, without
// invoking the recovery hook.
//
// **No carve-out, and that is the strengthened form of this invariant.** It
// used to read "the one carve-out is `handle_update_layer`'s arm 3, which
// matches the specific `FailedPrecondition(EMPTY_HEADER_SENTINEL)` message" —
// true until b.f1v deleted that arm outright, flipping the storage surface to
// `NotFound` so the empty-header case is caught by the variant like everything
// else. Recovery now dispatches on nothing but the variant: no message match,
// no carve-out. That is what keeps the branch armed — a message-equality arm
// is one cosmetic reword away from silently disarming recovery, which is why
// the storage-side sentinel constants are module-PRIVATE and "no handler
// matches this string" is compiler-enforced. "FailedPrecondition never invokes
// recovery" is therefore now unconditional, which is exactly what the
// UpdateLayer scoping test below pins via its object_id-mismatch shape.
//
// A regression that broadened the recovery match to a catch-all (e.g.
// `Err(e)` instead of `Err(ServiceError::NotFound(_))`) would silently
// retry-once on every failed RPC: a malformed `InvalidArgument` would burn a
// `get_latest_layer_checkpoint` round-trip for nothing, and worse — a
// `PermissionDenied` would invoke recovery on an `object_id` the caller has
// no read permission for, weakening the auth boundary.
//
// We pin the boundary on UpdateLayer (FailedPrecondition arm — the
// natural shape after the recovery branch in `handle_update_layer`) and
// StatLayer (PermissionDenied via DenyAllPermissions — exercises the
// upstream auth check). The existing disabled-mode SubscribeReadLayer
// test covers the third RPC's scoping: a disabled hook IS the no-op
// proxy for "recovery never fired", and the production recovery code
// path for non-NotFound errors mirrors disabled-mode (both bypass the
// hook), so a regression broadening the match would still tick the
// store wrapper in the disabled-mode test.

/// UpdateLayer recovery scoping: a `FailedPrecondition` (object_id
/// mismatch — the client claims an `object_id` the live mapping does
/// NOT resolve to) MUST NOT invoke the recovery hook. `FailedPrecondition`
/// is the natural shape that surfaces from `handle_update_layer` AFTER
/// the mapping-resolve recovery branch has succeeded (the mapping
/// lookup returned a real `object_id`, just not the one the client
/// claimed). A regression that caught a non-`NotFound` arm into the
/// recovery branch would surface here as either a spurious
/// `restored`/`not_recoverable`/`error` metric increment or a non-zero
/// `CallCountingStore` call count.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn update_layer_failed_precondition_does_not_invoke_recovery() {
    let tv = TestValkey::setup().await;
    let mut svc = build_service(&tv, Arc::new(AllowAllPermissions));
    let metrics = replace_metrics(&mut svc);

    let (session_id, real_object_id) =
        create_session_and_layer(&tv, &svc, "scoping-update-session", "scoping-update-layer").await;

    // Counting store: any recovery-hook → store invocation would tick this.
    let in_mem = Arc::new(InMemoryCheckpointStore::new());
    seed_layer_checkpoint(
        &in_mem,
        real_object_id,
        &session_id,
        "scoping-update-layer",
        17,
    )
    .await;
    let counting = Arc::new(CallCountingStore::new(in_mem as Arc<dyn CheckpointStore>));
    replace_recovery(&mut svc, counting.clone() as Arc<dyn CheckpointStore>, true);

    // Send UpdateLayer with a deliberately-wrong object_id. The mapping
    // resolves to `real_object_id`; the FailedPrecondition surface fires
    // before any NotFound surface could be reached.
    let wrong_object_id = real_object_id.wrapping_add(99_999);
    let status = handle_update_layer(
        &svc,
        update_request(
            &session_id,
            "scoping-update-layer",
            wrong_object_id,
            CLIENT_TS,
            CLIENT_ID,
        ),
    )
    .await
    .expect_err("object_id mismatch must surface gRPC error");

    assert_eq!(
        status.code(),
        tonic::Code::FailedPrecondition,
        "UpdateLayer object_id mismatch MUST surface FailedPrecondition, got: {}",
        status.code()
    );

    // Load-bearing scoping assertion: the recovery hook MUST NOT have
    // been consulted. Three independent observations of that contract:
    // (1) the store wrapper recorded zero method calls (no
    //     `get_latest_layer_checkpoint` round-trip);
    // (2) every recovery-metric cell stayed at zero;
    // (3) the per-object_id admission map stayed empty.
    // A regression broadening the recovery match would tick at least
    // one of the three.
    assert_eq!(
        counting.call_count(),
        0,
        "FailedPrecondition path MUST NOT invoke any CheckpointStore method (got {} calls)",
        counting.call_count()
    );
    assert_eq!(
        recovery_metric(&metrics, "restored"),
        0,
        "FailedPrecondition path MUST NOT bump the `restored` recovery cell"
    );
    assert_eq!(recovery_metric(&metrics, "not_recoverable"), 0);
    assert_eq!(recovery_metric(&metrics, "error"), 0);
    assert_eq!(
        svc.recovery.admission_len(),
        0,
        "FailedPrecondition path MUST NOT populate the admission map"
    );
}

/// StatLayer recovery scoping: a `PermissionDenied` (auth backend
/// rejects the read permission check) MUST NOT invoke the recovery
/// hook. The auth check fires AFTER session-id / layer-name validation
/// but BEFORE the mapping resolve + header read. We pin the contract
/// end-to-end: zero store calls, zero metric increments, even when the
/// wired RPC produces a non-`NotFound` error post-auth.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn stat_layer_permission_denied_does_not_invoke_recovery() {
    let tv = TestValkey::setup().await;
    // First: create the session + layer with AllowAll so the test can
    // set up a valid layer to read against.
    let mut svc = build_service(&tv, Arc::new(AllowAllPermissions));
    let (session_id, object_id) =
        create_session_and_layer(&tv, &svc, "scoping-stat-session", "scoping-stat-layer").await;

    // Now swap the permissions client to DenyAll so the stat read
    // hits `PermissionDenied`. Built-in DenyAllPermissions returns
    // `AuthResult::Denied` for every check (b.ohw H-grpc-1 contract).
    svc.permissions_client = Arc::new(DenyAllPermissions);

    let metrics = replace_metrics(&mut svc);

    let in_mem = Arc::new(InMemoryCheckpointStore::new());
    seed_layer_checkpoint(&in_mem, object_id, &session_id, "scoping-stat-layer", 19).await;
    let counting = Arc::new(CallCountingStore::new(in_mem as Arc<dyn CheckpointStore>));
    replace_recovery(&mut svc, counting.clone() as Arc<dyn CheckpointStore>, true);

    // Wipe obj:* so that IF the recovery branch fired (it must not), it
    // would have a real NotFound symptom to chew on. Without the wipe,
    // a regression broadening the match might still surface
    // PermissionDenied first; the wipe ensures the regression would
    // actually be observable.
    wipe_object_keys(&tv, object_id).await;

    let status = handle_stat_layer(&svc, stat_request(&session_id, "scoping-stat-layer"))
        .await
        .expect_err("DenyAllPermissions must surface gRPC error");

    assert_eq!(
        status.code(),
        tonic::Code::PermissionDenied,
        "StatLayer PermissionDenied MUST surface PermissionDenied, got: {}",
        status.code()
    );

    // Recovery hook scoping invariant — same three observations as the
    // UpdateLayer scoping test.
    assert_eq!(
        counting.call_count(),
        0,
        "PermissionDenied path MUST NOT invoke any CheckpointStore method (got {} calls)",
        counting.call_count()
    );
    assert_eq!(recovery_metric(&metrics, "restored"), 0);
    assert_eq!(recovery_metric(&metrics, "not_recoverable"), 0);
    assert_eq!(recovery_metric(&metrics, "error"), 0);
    assert_eq!(svc.recovery.admission_len(), 0);
}
