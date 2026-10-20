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

//! Lazy on-demand layer-state recovery (SDD SR-2.4 / SR-2.6).
//!
//! [`RecoveryHook::try_restore_layer`] is the recovery primitive the three
//! gRPC handler subtasks call when an `obj:hdr` lookup returns `NotFound`:
//! it selects a *restorable* layer-state lineage via
//! `RecoveryHook::select_restore_source`, delegates to
//! [`apply_restore`](super::apply::apply_restore) to materialize the
//! recovered state, and surfaces a classified [`RecoveryOutcome`] so the
//! caller can decide whether to retry the lookup or surface `NotFound` to
//! the client.
//!
//! ## "Re-base" in this module means the SR-3.6 post-restore one
//!
//! The service has **two** distinct re-bases and they must not be conflated:
//!
//! - **SR-3.6 post-restore re-base** — this module's. After a DR restore
//!   materializes recovered state, `BaseBootstrap::spawn_restore_rebase` PUTs a
//!   fresh base at the seeded sequence and clears `restore_seq_bump`, re-rooting
//!   the layer above the abandoned timeline's fossils. It is a *write* on the
//!   recovery path; it ticks `checkpoint_writes_total{…, layer}` and
//!   deliberately **no** `live_edit_checkpoint_tail_total` cell.
//! - **SR-4.1 chain-collapse re-base** — [`crate::checkpoint::rebase`]'s. On the
//!   checkpoint tick, a layer whose tail chain has reached K (tails are captured
//!   only on ticks a header-sequence regression skipped) has its whole lineage
//!   composed into one new base, ticking
//!   `live_edit_checkpoint_tail_total{outcome="rebased"}`.
//!
//! Unqualified "re-base" anywhere in this module is the first. The second is
//! always named in full. The invariant that keeps them apart — SR-3.6's re-base
//! never ticks `rebased` — is live and pinned by
//! `tests/integration/checkpoint/restore_rebase_test.rs`.
//!
//! ## Restore-source selection — newest *restorable lineage* wins (SR-3.1)
//!
//! A lineage interleaves full-state [`CheckpointKind::Base`] snapshots with
//! incremental [`CheckpointKind::Tail`] envelopes under one `object_id`, in
//! one strict descending order over the sequence — so the *newest*
//! generation is routinely a tail. Selection therefore does **not** ask "which
//! generation", it asks "which **lineage**": a base plus the contiguous run of
//! tails that link onto it, replayed to the head sequence `S`, **stepping down
//! to the next older restorable generation when a lineage is broken**.
//!
//! Two properties make that safe rather than merely richer:
//!
//! - **The floor.** A lineage whose newest verified member is a base alone is
//!   byte-for-byte the pre-Epic-4 answer, and it is what the walk falls back to.
//!   Composed restore can only ever *add* coverage; it can never resolve to
//!   something worse than "newest base".
//! - **A broken lineage is a verdict, never an error.** A missing link, a header
//!   that contradicts its link, an unreadable envelope, an exhausted budget, or
//!   a composition the engine rejected all *step down* — they never strand a
//!   perfectly restorable base sitting one generation behind, which is the
//!   failure class FR-6 / SR-2.5 / SR-3.1 forbid.
//!
//! A **base-only** lineage is deliberately NOT routed through the composition
//! engine: a corrupt base keeps surfacing exactly as it does today (a decode
//! failure inside apply → `Error`), not as a step-down. That is a deliberate
//! divergence from a literal reading of SR-3.5, and it is what makes "existing
//! base-only restore behaviour unchanged" trivially provable.
//!
//! The invariant is **independent of any feature flag** — nothing in
//! `RecoveryHook::select_restore_source` or in `checkpoint::lineage` reads
//! [`Config`] for a policy decision. Tail *capture*
//! ships behind its own flag, but a store can already hold tails with the
//! flag off (written by a peer replica, written before the flag was turned
//! back off, or restored from an older backup), and those layers must still
//! recover. Read side and write side are decoupled on purpose.
//!
//! [`CheckpointStore::get_latest_layer_checkpoint`]'s contract is
//! deliberately **NOT** touched by this: it remains "the blob whose handle
//! is `list_layer_checkpoints(..).first()`". Redefining it adapter-side as
//! "newest base" — or as "newest restorable lineage" — is the tempting shortcut
//! and it is refused twice over:
//! it would invalidate the SR-1.3 conformance assertion tying an adapter's
//! two enumerations to each other, and it would push a **restore-path
//! policy** decision into three adapters, where it can only be tested three
//! times. The policy lives here, once. Full argument in the `# Ordering`
//! block on [`CheckpointStore`].
//!
//! ## Where a step-down is observable (b.8j's breadcrumb, re-sited)
//!
//! On `live_edit_checkpoint_restore_lineage_total{outcome, service}`
//! (`observability::metrics`, vocabulary pinned by
//! `CHECKPOINT_RESTORE_LINEAGE_OUTCOMES`) — a family **orthogonal** to
//! `checkpoint_recovery_attempts_total`, deliberately rather than as a fifth
//! `result` value on it. `recovery_attempts_total` stays strictly
//! one-tick-per-attempt so `sum by (result) == attempts` holds; one attempt that
//! steps down twice before resolving increments the lineage family three times.
//! Labels are minted by `checkpoint::lineage`'s decision types
//! ([`GatheredLineage::outcome_label`],
//! [`LineageBreak::reason_label`](crate::checkpoint::lineage::LineageBreak::reason_label),
//! [`Exhaustion::outcome_label`]) and incremented **only** by this hook —
//! `checkpoint::lineage` holds no metrics handle at all.
//!
//! ## Admission discipline (SR-2.6)
//!
//! Per-`object_id` admission is enforced via a **K=1** `tokio::sync::Mutex`
//! kept in a `DashMap<u64, Arc<Mutex<()>>>` that is **parallel to (NOT
//! shared with)** the K=16 `update_layer` admission map at
//! [`LayerStore::semaphores`](crate::storage::layer::LayerStore). The two
//! admission maps protect different invariants:
//!
//! - The K=16 update-layer admission caps concurrent writers per object —
//!   the operation is correct under concurrency but the per-object commit
//!   pipeline benefits from being serialized lightly.
//! - The K=1 recovery admission ensures **applying a checkpoint twice in
//!   parallel is forbidden** — `apply_checkpoint` is idempotent under its
//!   structural-header short-circuit (an `HGETALL obj:hdr`), but running it
//!   twice in parallel
//!   wastes a `write_restored_object` MULTI/EXEC and an ext-blob GC schedule on the
//!   second attempt. The mutex is the cheap way to coalesce concurrent
//!   restorers for the same `object_id`.
//!
//! Sharing the maps would mean either weakening the update-layer cap to 1
//! (a regression on per-object write throughput) or weakening the recovery
//! cap to 16 (a regression on the duplicate-work property the K=1 here
//! prevents). The two maps stay separate.
//!
//! ## Cross-replica race guard
//!
//! The K=1 mutex serializes recovery attempts within a single replica;
//! peer replicas may race their own recovery attempts concurrently. The
//! cross-replica guards run *inside* [`apply_restore`], and there are **three**
//! of them, ordered outermost first:
//!
//! - **Step 1 — the structural-header short-circuit.** One
//!   `HGETALL le:{oid}:obj:hdr` round-trip whose predicate is
//!   `header_is_structurally_complete` ("the key is there AND carries `seq`
//!   and `created_by`"), not a bare `EXISTS`; it short-circuits with
//!   `ApplyOutcome::AlreadyPresent` when a peer landed the recovery first.
//!   Being a point-in-time read, it closes the race only up to the instant it
//!   answers.
//! - **Step 3b — the `WATCH`-fenced residue wipe.** The conditional pre-write
//!   wipe in `crate::checkpoint::residue` takes its own slot-pinned `txn_pool`
//!   connection, `WATCH`es `obj:hdr`, and issues its `UNLINK` batch under that
//!   fence — so a peer replica landing its structurally complete header `HSET`
//!   (the 12 canonical fields; a real peer's is a superset) in the gap
//!   trips the `EXEC` and the wipe deletes nothing. Only reachable on a restore
//!   that found surviving residue.
//! - **Step 5 — the `WATCH`-fenced materialization** (b.h4m).
//!   `LayerStore::write_restored_object` `WATCH`es the same `obj:hdr` and
//!   requires it to be ABSENT, then queues the whole object through a
//!   `WatchedTxn`. Valkey serializes the two replicas' `EXEC`s on the
//!   `{object_id}` shard, so the second one returns nil and writes nothing.
//!   This is the innermost of the two `WATCH`-held fences and the only *fence*
//!   a **total-loss** restore can reach — the common DR shape, where there is
//!   no residue for step 3b to notice. Step 1 is not ruled out there: it is a
//!   point-in-time probe rather than a fence, and every restore runs it.
//!   Before it existed, two concurrent rebuilds were *merged*: the
//!   loser's members were added to the winner's rather than replacing them,
//!   leaving the layer a mixture of two backup generations with no error and no
//!   counter moving.
//!
//! The last two *hold* across a write rather than sampling before one; step 1
//! only samples, and the admission mutex is per-process. All three report on
//! `live_edit_checkpoint_restore_fenced_total{stage}`, whose three `stage`
//! values are these three positions.
//!
//! We also re-check the step-1 condition here after acquiring the admission
//! mutex (step 3) so a task that queued behind a winner on the same replica
//! observes the restored state and reports `Restored` without re-issuing the
//! apply pipeline.
//!
//! ## Map-entry lifecycle
//!
//! Admission map entries are created lazily on first attempt per
//! `object_id` and retired by a reference-counted RAII guard
//! ([`AdmissionMapGuard`]) on the last drop — even when an admission entry
//! sees multiple concurrent waiters, the entry stays alive until every
//! waiter has dropped its `Arc<Mutex<()>>` clone. Mirrors the
//! `SemaphoreMapGuard` pattern in `src/storage/layer.rs` (b.xet).

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, LazyLock};
use std::time::Duration;

use bytes::Bytes;
use dashmap::DashMap;
use fred::interfaces::KeysInterface;
use fred::types::scripts::Script;
use tokio::sync::Mutex;
use tokio::time::Instant;

use crate::checkpoint::apply::{
    apply_restore, is_over_entry_limit_base, live_mapping_resolves_blob_identity,
    AppliedCheckpoint, ApplyOutcome, ExtPrestore, RECOVERY_SEAM_STALE_STREAM_UNLINK,
    RECOVERY_SEAM_STREAM_HIGH_WATER_PROBE, RECOVERY_SEAM_STREAM_KEY_TYPE,
};
// Seam-only now that the peer-presence re-read logs under its own op label
// (`RECOVERY_STREAM_PEER_PRESENCE_REREAD_OP`), so the sole remaining consumer is
// the `inject_recovery_seam` call inside `peer_presence_reread_once`.
#[cfg(feature = "test-checkpoint-fault-injection")]
use crate::checkpoint::apply::RESTORE_SEAM_BEFORE_STALE_STREAM_UNLINK;
use crate::checkpoint::assembler::service_err_to_checkpoint_err;
use crate::checkpoint::base_bootstrap::{BaseBootstrap, BootstrapTarget};
use crate::checkpoint::chunked_base::is_chunked_base;
use crate::checkpoint::compose::{compose_from_artifacts, ComposeError};
use crate::checkpoint::lineage::{
    gather_restore_lineage, Exhaustion, GatheredLineage, LineageSelection,
};
use crate::checkpoint::store::{
    CheckpointBlob, CheckpointError, CheckpointKind, CheckpointMeta, CheckpointStore,
};
use crate::checkpoint::RestorePayload;
use crate::config::Config;
use crate::domain::limits::MAX_DELTA_ENTRIES;
use crate::error::ServiceError;
use crate::observability::metrics::SharedMetrics;
use crate::storage::client::{map_snapshot_load_err, FredErrOrigin, POOL_LABEL_DATA};
use crate::storage::external_values::ExternalValueStore;
use crate::storage::layer::LayerStore;
use crate::storage::retry::{retry_aborted_with_jitter, RetryConfig};

/// Compiled Lua script backing
/// [`RecoveryHook::probe_stream_high_water`] (b.8y5).
///
/// Projects `XINFO STREAM ... FULL COUNT 1`'s `last-generated-id`
/// server-side, returning that ID string alone (or `nil` for a missing or
/// wrong-type key), so the probe's reply is O(1) in layer size rather than
/// carrying a full stream entry. The plausibility gate that interprets the
/// ID lives on the Rust side — see the script's own header, and the
/// "Why `XINFO STREAM`, and why server-side" section on the probe, for the
/// cost argument and the server-side-work caveat.
///
/// The script lives alongside its siblings in `src/storage/scripts/` (repo
/// convention: that directory is the home for every Lua script, regardless
/// of which module owns the call site), while the `Script` handle is
/// declared here next to its only caller — mirroring how
/// `src/storage/lua.rs` and `src/storage/external_values.rs` each declare
/// the handles they use.
///
/// `Script::from_lua` is infallible — it just hashes the source. The first
/// `evalsha_with_reload` call sends `SCRIPT LOAD` on `NOSCRIPT` before
/// retrying.
static STREAM_HIGH_WATER_PROBE_SCRIPT: LazyLock<Script> = LazyLock::new(|| {
    Script::from_lua(include_str!(
        "../storage/scripts/stream_high_water_probe.lua"
    ))
});

/// Outcome of a single [`RecoveryHook::try_restore_layer`] attempt.
///
/// `Restored` — a checkpoint existed and the layer was materialized (or a
/// peer recovery for the same object_id already landed it). The caller
/// SHOULD retry its original `obj:hdr` lookup.
///
/// `NotRecoverable` — recovery cannot proceed: either no checkpoint exists
/// for the object, or [`apply_restore`] rejected the apply because no
/// live layer mapping resolves to this `object_id` anymore (SR-2.4
/// DeleteLayer / CreateLayer-overwrite races). The caller SHOULD surface
/// `NotFound` to its client.
///
/// `Error(_)` — a transient / permanent backend failure occurred. The
/// caller MAY retry later (transient) or SHOULD surface an internal error
/// to its client (permanent). The variant carries the classified
/// [`CheckpointError`] verbatim so the caller can branch on its kind.
///
/// # Composed restore adds NO new verdict — do not widen this
///
/// Epic 4's lineage-aware restore (a base plus its contiguous tails, stepping
/// down when a lineage is broken) is entirely a *selection* concern. It reports
/// itself on `live_edit_checkpoint_restore_lineage_total{outcome}`, orthogonal
/// to this enum and to `checkpoint_recovery_attempts_total`. A step-down that
/// still restores is `Restored`; a selection that exhausts **conclusively** is
/// `NotRecoverable`; one that exhausts with a listed `Base` left unread
/// ([`Exhaustion::BudgetInconclusive`]) is `Error(Transient)`, because that
/// verdict is about the selection rather than about the layer and the client
/// should retry; a store failure is `Error`.
///
/// That is deliberate, not an omission: the gRPC handlers' retry-once dispatch
/// branches on exactly these three variants and must not learn about lineages.
/// A fourth variant would have to be handled at every `RecoveryOutcome` match
/// arm in the handlers — `update_layer_inner`'s two (mapping-absent resolve, and
/// the post-commit "object gone" retry) plus `stat_layer_inner`'s in
/// `src/grpc/layer.rs`, and `try_recover_after_notfound`'s in
/// `src/grpc/subscribe.rs` — whose correct behaviour is identical to one of the
/// three that already exist. (`src/grpc/session.rs` and
/// `src/grpc/session_cache.rs` match the separate `SessionRecoveryOutcome`, not
/// this enum.)
#[derive(Debug)]
pub enum RecoveryOutcome {
    Restored,
    NotRecoverable,
    Error(CheckpointError),
}

/// What [`RecoveryHook::select_restore_source`] resolved to: the newest
/// *restorable* lineage, already reduced to what the apply core needs.
///
/// Post-C1 shape: the external-value set rides **inside**
/// [`RestorePayload::Composed`] rather than as a sibling field, so the apply
/// core can see both "this restore needs values pre-stored" and "no store was
/// wired" and fail closed on the mismatch.
struct SelectedRestore {
    /// [`RestorePayload::Ompf`] carrying the base's bytes on a base-only
    /// lineage; [`RestorePayload::Composed`] carrying the replayed head state
    /// plus its deduplicated external values otherwise.
    payload: RestorePayload,
    /// **S** — the sequence this restore lands on: the head tail's `to_seq`, or
    /// the base's own sequence on a base-only lineage.
    head_seq: u64,
    /// The meta the write is stamped from. Identity and header columns are
    /// always **base-sourced**; on a composed lineage exactly two non-identity
    /// columns are head-sourced. See `select_restore_source`'s
    /// `# The composed meta` block.
    meta: CheckpointMeta,
    /// Tails replayed onto the base — `0` on a base-only lineage. Diagnostics
    /// only.
    chain_len: usize,
    /// This selection's `outcome` label on
    /// `live_edit_checkpoint_restore_lineage_total`, minted by
    /// [`GatheredLineage::outcome_label`] — never spelled as a literal here.
    outcome: &'static str,
}

/// The on-access refusal of a base over the per-delta entry limit
/// (SR-14.7). It is logged here, once per refusal, and returns a
/// [`CheckpointError::Permanent`] for the hook's `Error` arm.
///
/// Called only from [`RecoveryHook::adjudicate_over_limit_base_on_access`],
/// and only once the SDD §22.6 step-2 identity verdict has found the layer
/// still mapped to this `object_id`. A layer deleted or replaced during the
/// state loss is answered not-found there instead, and never reaches this
/// refusal or its WARN.
///
/// This release writes a base over [`MAX_DELTA_ENTRIES`] entries as a
/// chunked base (`checkpoint::chunked_base`), which decodes correctly. An
/// earlier release wrote it as one plain OMPF delta, which no decoder accepts.
/// Restoring a chunked base on access would put the whole layer into
/// `write_restored_object`'s single transaction under its flat timeout.
///
/// Neither arm of [`RecoveryHook::try_restore_layer`] takes a step-down for
/// such a base, and neither reaches the residue wipe, the ext pre-store or the
/// write:
/// - **Base-only arm.** The hook body reaches this for a chunked base. A plain
///   over-limit base fails `Permanent` at `apply_restore`'s decode, before
///   those steps, without calling this; `apply_restore`'s own identity guard
///   runs ahead of that decode, so a gone layer is not-found there too.
/// - **Composed arm.** [`RecoveryHook::select_restore_source`] reaches this
///   for a chunked base before composing. It also reaches it for a plain
///   over-limit base when the composition's base decode fails with that typed
///   error (`apply::is_over_entry_limit_base`).
///
/// Without a step-down, the selection usually loses nothing: the next
/// candidate would normally be the same base, offered base-only. In the
/// jumped-base shape of SDD §22.20.3, though, it would be a different, older
/// base. The refusal gives that generation up even when it is restorable, a
/// cost the SR-14.7 ruling accepts.
///
/// `head_seq` is the selected lineage's head sequence, logged for diagnostics.
fn refuse_over_limit_base_on_access(object_id: u64, head_seq: u64) -> CheckpointError {
    tracing::warn!(
        object_id,
        head_seq,
        max_delta_entries = MAX_DELTA_ENTRIES,
        "checkpoint recovery refused the backup it selected: a backup over the per-delta entry \
         limit cannot be restored on access"
    );
    CheckpointError::Permanent {
        source: format!(
            "the backup selected to restore object {object_id} is over the per-delta entry \
             limit of {MAX_DELTA_ENTRIES} entries; a backup over the per-delta entry limit \
             cannot be restored on access"
        )
        .into(),
    }
}

/// Build the [`CheckpointMeta`] that stands for a **composed lineage**: the
/// base's meta with the two columns a composition legitimately re-sources
/// overridden.
///
/// # Why this is one function and not two copies
///
/// There are exactly two producers of a composed-lineage meta and they must not
/// drift:
///
/// - [`RecoveryHook::select_restore_source`], which stamps the restored
///   `obj:hdr`;
/// - [`crate::checkpoint::rebase::rebase_layer_lineage`] (SR-4.1), which stamps
///   the **new base** it PUTs at the collapsed head.
///
/// The rule they share is the load-bearing one, and it is a MUST:
/// **identity is base-sourced.** `session_id` / `layer_name` / `created_by` /
/// `server_ts` come from `base_meta` and from nothing else, because
/// `apply::live_mapping_resolves_blob_identity` refuses a blob with an empty
/// `session_id` or `layer_name` outright (see [`GatheredLineage`]'s "Identity
/// is base-sourced — MUST", which also records that the adapters' identity-empty
/// meta-absent tail placeholder — the rule's original motivation — was replaced
/// by an adapter-side refusal in b.hka (d); the MUST stands on its own).
/// Every other column of `base_meta` rides
/// through the struct update unchanged, so a caller that needs to re-source one
/// (kind, `snap_seq`, `sidecar_bytes`, `creator_replica_id`) does so *visibly*,
/// at its own call site.
///
/// [`GatheredLineage`]: crate::checkpoint::lineage::GatheredLineage
///
/// # The two arguments, and the one place the callers genuinely differ
///
/// - `observed_commit_span` — **deliberately a parameter, because the two
///   adjudications differ here and the difference is correct.** Restore passes
///   the HEAD tail's envelope span (`header.to_seq - header.from_seq`): the DR
///   guard band must cover commits that landed after the newest generation and
///   died with the crash, so the honest input is this layer's most recent
///   measured interval, and the whole-lineage span would inflate the band by K
///   intervals at once. Re-base passes the COLLAPSED lineage's total span
///   (`S - base_snap_seq`): the base it writes genuinely covers every commit
///   from the old root to the head, so "commits this generation covers that the
///   previous did not" *is* the whole span — a head-tail-sized value there would
///   understate a hot layer's band, the b.8y5 smaller-band hazard direction.
///   Passing it in is what keeps that divergence a visible argument at two call
///   sites rather than two silently forked struct literals.
/// - `written_at_unix_ms` — the newest generation's own write time (SR-3.3).
///   Restore passes the head tail's `meta.written_at_unix_ms`; re-base passes
///   `now_unix_ms()`, because on that path the re-base *is* the newest
///   generation's write. `0` is the "no write time recorded" sentinel — its
///   original producer was the adapters' meta-absent placeholder, which
///   b.hka (d) replaced with a refusal, so what remains is a meta that genuinely
///   carries 0 — and it falls back to the base's value here (which then falls
///   through apply's existing `0 -> now_unix_ms()` arm); on the re-base path the
///   fallback is inert but harmless, since a wall clock does not read zero.
pub(crate) fn adjudicate_composed_meta(
    base_meta: CheckpointMeta,
    observed_commit_span: u64,
    written_at_unix_ms: u64,
) -> CheckpointMeta {
    let written_at_unix_ms = if written_at_unix_ms == 0 {
        base_meta.written_at_unix_ms
    } else {
        written_at_unix_ms
    };
    CheckpointMeta {
        observed_commit_span,
        written_at_unix_ms,
        ..base_meta
    }
}

/// Recovery hook. Constructed once at startup in `main.rs`, shared as
/// `Arc<RecoveryHook>` on [`LiveEditService`](crate::grpc::service::LiveEditService).
///
/// The per-`object_id` admission map (`admission`) is K=1 and parallel to
/// the K=16 update-layer admission map at
/// [`LayerStore`](crate::storage::layer::LayerStore). See the module-level
/// docstring for why the two are not shared.
pub struct RecoveryHook {
    store: Arc<dyn CheckpointStore>,
    layer_store: Arc<LayerStore>,
    config: Arc<Config>,
    metrics: SharedMetrics,
    service_label: String,
    enabled: bool,
    /// Per-`object_id` admission map (K=1). Entries are created on first
    /// attempt and retired by [`AdmissionMapGuard`] on the last waiter's
    /// drop. Parallel to (NOT shared with) `LayerStore::semaphores`.
    admission: Arc<DashMap<u64, Arc<AdmissionEntry>>>,
    /// The external-value store the SR-3.4 composed-restore pre-store writes
    /// through. See [`Self::with_ext_value_store`] for why it is optional.
    ext_value_store: Option<Arc<dyn ExternalValueStore>>,
    /// The SR-3.6 post-restore gap re-base vehicle. See
    /// [`Self::with_base_bootstrap`] for why it is optional and what `None`
    /// means.
    base_bootstrap: Option<Arc<BaseBootstrap>>,
    /// Test-only restore fault injector (SR-3.4 / AC-8). See
    /// [`Self::with_restore_fault_injector_for_test`].
    #[cfg(feature = "test-checkpoint-fault-injection")]
    restore_fault_injector: Option<Arc<dyn crate::checkpoint::adapters::CheckpointFaultInjector>>,
    /// Test-only one-shot latch (b.sxc): when `Some(latch)` and
    /// `latch.swap(false, AcqRel)` returns `true`, the production path
    /// SYNCHRONOUSLY `UNLINK`s the just-restored object's `obj:hdr`
    /// immediately before returning `RecoveryOutcome::Restored`. This is
    /// the deterministic replacement for the 16-wiper race-based pattern
    /// in the retry-once StatLayer + SubscribeReadLayer integration tests
    /// (see `tests/integration/checkpoint/recovery_hook_test.rs`): the
    /// test arms the latch via [`Self::with_force_post_restore_notfound_for_test`]
    /// before issuing the RPC; the recovery hook fires the UNLINK exactly
    /// once at the post-Restored boundary and `.await`s its round-trip
    /// completion before returning. Determinism comes from two
    /// properties, NOT from connection identity (the UNLINK runs on the
    /// round-robin multiplexed data pool, not on the `txn_pool`
    /// connection that committed the apply — that connection is already
    /// released back to the `txn_pool` by the time the seam fires): (1)
    /// the explicit `.await` on the UNLINK orders it before the seam
    /// returns, and (2) Valkey's single-threaded command loop processes
    /// commands in arrival order, so the originating handler's
    /// subsequent retry HGETALL — on any pool slot — observes the
    /// UNLINK's effect. Gated under `test-checkpoint-fault-injection` so
    /// it compiles out of production builds entirely (analogous to the
    /// `test-fault-injection` seam on `LayerStore`).
    #[cfg(any(test, feature = "test-checkpoint-fault-injection"))]
    force_post_restore_notfound_for_test: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
}

/// A single per-`object_id` admission entry: the K=1 mutex itself plus a
/// reference count tracking how many concurrent restorers currently hold a
/// clone of the `Arc`. The count is used by [`AdmissionMapGuard`] to defer
/// map-entry removal until the last waiter drops out — early removal would
/// leak the entry forever if a still-queued waiter's `Arc` kept the
/// `AdmissionEntry` alive past the removal point (b.xet rationale, ported
/// from `src/storage/layer.rs:SemaphoreMapGuard`).
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

/// RAII guard that retires a per-`object_id` admission map entry when the
/// last in-flight restorer for that object drops out.
///
/// Mirrors `src/storage/layer.rs::SemaphoreMapGuard` (b.xet) but with
/// reference-count semantics rather than unconditional removal on drop:
/// recovery attempts queue behind the K=1 mutex (only one at a time
/// progresses past `lock().await`), but each queued waiter holds an
/// `Arc<AdmissionEntry>` clone. If we removed the map entry on first
/// drop, a later attempt for the same `object_id` would lazily mint a
/// *fresh* `AdmissionEntry` that does NOT serialize against the still-
/// queued original waiter — defeating the K=1 invariant. The
/// reference-counted guard defers removal to the last drop, after which a
/// fresh attempt is safe to mint a new entry.
struct AdmissionMapGuard<'a> {
    map: &'a DashMap<u64, Arc<AdmissionEntry>>,
    entry: Arc<AdmissionEntry>,
    object_id: u64,
}

impl<'a> AdmissionMapGuard<'a> {
    fn new(
        map: &'a DashMap<u64, Arc<AdmissionEntry>>,
        entry: Arc<AdmissionEntry>,
        object_id: u64,
    ) -> Self {
        entry.refs.fetch_add(1, Ordering::AcqRel);
        Self {
            map,
            entry,
            object_id,
        }
    }
}

impl Drop for AdmissionMapGuard<'_> {
    fn drop(&mut self) {
        // Decrement refs; if we were the last waiter, remove the map
        // entry. The `remove_if` closure re-checks ref count under the
        // shard lock so a concurrent `get_or_create` that just incremented
        // refs does not race the removal — preventing a removed entry
        // from leaving a still-queued waiter unable to serialize against
        // the next attempt.
        let prev = self.entry.refs.fetch_sub(1, Ordering::AcqRel);
        if prev == 1 {
            self.map
                .remove_if(&self.object_id, |_k, v| v.refs.load(Ordering::Acquire) == 0);
        }
    }
}

impl RecoveryHook {
    /// Construct a new recovery hook.
    ///
    /// `enabled` is read from
    /// [`Config::checkpoint_enabled`](crate::config::Config::checkpoint_enabled);
    /// passing `false` makes every [`Self::try_restore_layer`] call return
    /// [`RecoveryOutcome::NotRecoverable`] without touching the
    /// admission map, the store, or the metric (SR-2.4 disabled
    /// short-circuit).
    pub fn new(
        store: Arc<dyn CheckpointStore>,
        layer_store: Arc<LayerStore>,
        config: Arc<Config>,
        metrics: SharedMetrics,
    ) -> Self {
        let service_label = config.service_label.clone();
        let enabled = config.checkpoint_enabled;
        Self {
            store,
            layer_store,
            config,
            metrics,
            service_label,
            enabled,
            admission: Arc::new(DashMap::new()),
            ext_value_store: None,
            base_bootstrap: None,
            #[cfg(feature = "test-checkpoint-fault-injection")]
            restore_fault_injector: None,
            #[cfg(any(test, feature = "test-checkpoint-fault-injection"))]
            force_post_restore_notfound_for_test: None,
        }
    }

    /// Attach the [`ExternalValueStore`] the SR-3.4 pre-store writes
    /// tail-carried external values through, **strictly before** the single
    /// `write_restored_object` MULTI/EXEC.
    ///
    /// # Why a builder rather than a 5th `new` parameter
    ///
    /// [`Self::new`] has ~20 in-file call sites plus integration-binary ones,
    /// and those fixtures are the base-only regression fence for this Epic.
    /// Churning every one of them to thread a handle none of them exercises is
    /// pure risk, so this follows the shape the type already uses for its other
    /// optional collaborator ([`Self::with_force_post_restore_notfound_for_test`]).
    ///
    /// # Why the field is `Option`, and what happens when it is `None`
    ///
    /// Purely a test affordance. Those in-file fixtures restore **base-only**
    /// lineages, which carry no inline external values by construction — the
    /// assembler serializes base OMPFs with **zero** external parts
    /// (`OmpfContainer::serialize(&delta_bytes, &[])`,
    /// `src/checkpoint/assembler.rs`), so a base-only restore has nothing to
    /// pre-store and `ext: None` is the honest argument.
    ///
    /// **Production always wires `Some`** — `main.rs` calls this
    /// unconditionally, matching the hook's own "constructed unconditionally,
    /// the gate lives inside `try_restore_layer`" posture.
    ///
    /// A **composed** lineage with a non-empty ext set and `None` here does not
    /// silently skip the pre-store: it fails closed at the apply core's
    /// `Permanent` arm (`apply::apply_restore`, step 4b), which is observable in
    /// the `error` recovery-attempt cell and logged at ERROR. Never silent.
    pub fn with_ext_value_store(mut self, ext_value_store: Arc<dyn ExternalValueStore>) -> Self {
        self.ext_value_store = Some(ext_value_store);
        self
    }

    /// Attach the SR-3.6 post-restore gap re-base vehicle — the SAME
    /// [`BaseBootstrap`] instance `CreateLayer` uses.
    ///
    /// # One instance, threaded to both call sites
    ///
    /// `main.rs` constructs the unit once and hands the same `Arc` here and to
    /// `LiveEditService`. That is a wiring **invariant**, not a saving: the
    /// unit's in-flight admission bound
    /// ([`BOOTSTRAP_MAX_INFLIGHT`](crate::checkpoint::BOOTSTRAP_MAX_INFLIGHT))
    /// is a field on the unit, so a second instance constructed for this hook
    /// would be a second, independent bound and the "one bound across both
    /// sites" property would be silently untrue.
    ///
    /// # Why a builder, and what `None` means
    ///
    /// Same reasoning as [`Self::with_ext_value_store`]: [`Self::new`] has ~20
    /// in-file call sites plus integration-binary ones, and those fixtures are
    /// this Epic's restore regression fence. `None` means no re-base is spawned
    /// — the pre-SR-3.6 behaviour exactly, which is what every fixture that does
    /// not opt in continues to observe.
    ///
    /// `None` is NOT how a tail-flag-off deployment is expressed. Production
    /// wires `Some` unconditionally and the unit's own three-way gate
    /// (`checkpoint_enabled && !Disabled && checkpoint_tail_enabled`) decides
    /// whether the spawn does anything — the same "gate lives inside the unit"
    /// posture the hook itself uses, so neither this call site nor the handler
    /// branches on configuration.
    pub fn with_base_bootstrap(mut self, base_bootstrap: Arc<BaseBootstrap>) -> Self {
        self.base_bootstrap = Some(base_bootstrap);
        self
    }

    /// Test-only seam (SR-3.4): install the fault injector the composed restore
    /// path hands to `ExtPrestore::with_fault_injector`, so Task 6's AC-8 matrix
    /// can drive the window **between** the external-value pre-store and the
    /// single `write_restored_object` MULTI/EXEC.
    ///
    /// The op label is
    /// [`RESTORE_SEAM_AFTER_EXT_PRESTORE`](crate::checkpoint::RESTORE_SEAM_AFTER_EXT_PRESTORE),
    /// spelled once in `apply.rs` so no consumer can misspell it.
    ///
    /// Gated on `test-checkpoint-fault-injection` **exactly** — matching
    /// `CheckpointFaultInjector`'s own cfg rather than
    /// `any(test, feature = ...)`, which is what the b.sxc latch above uses —
    /// so the field, this builder, and the pass-through site all compile out of
    /// production builds entirely.
    #[cfg(feature = "test-checkpoint-fault-injection")]
    pub fn with_restore_fault_injector_for_test(
        mut self,
        injector: Arc<dyn crate::checkpoint::adapters::CheckpointFaultInjector>,
    ) -> Self {
        self.restore_fault_injector = Some(injector);
        self
    }

    /// Test-only seam (b.sxc): install a one-shot latch that, when armed
    /// (`AtomicBool::store(true, _)`), causes the next successful
    /// [`Self::try_restore_layer`] call to SYNCHRONOUSLY `UNLINK` the
    /// just-restored object's `obj:hdr` immediately before returning
    /// `RecoveryOutcome::Restored`. The latch self-disarms via
    /// `swap(false, AcqRel)` so subsequent restores on the same hook are
    /// unaffected.
    ///
    /// This is the deterministic replacement for the 16-wiper race-based
    /// pattern used by the retry-once StatLayer + SubscribeReadLayer
    /// integration tests to verify "after a successful restore, the
    /// originating RPC retries ONCE; if the retry also gets `NotFound`,
    /// the RPC surfaces `NotFound`". Determinism comes from the explicit
    /// `.await` on the UNLINK ordering it before the seam returns,
    /// combined with Valkey's single-threaded command loop processing
    /// commands in arrival order — NOT from any transactional-connection
    /// identity (the UNLINK targets the round-robin multiplexed data
    /// pool, and the apply's `txn_pool` connection is already released
    /// by the time the seam fires). Together those two properties
    /// guarantee the post-restore retry HGETALL — on any pool slot —
    /// observes `NotFound` without any cross-task timing dependency.
    ///
    /// Gated under `test-checkpoint-fault-injection` so the field, the
    /// builder, and the gate-check site all compile out of production
    /// library builds (analogous to the `test-fault-injection` seam on
    /// `LayerStore`).
    #[cfg(any(test, feature = "test-checkpoint-fault-injection"))]
    pub fn with_force_post_restore_notfound_for_test(
        mut self,
        latch: std::sync::Arc<std::sync::atomic::AtomicBool>,
    ) -> Self {
        self.force_post_restore_notfound_for_test = Some(latch);
        self
    }

    /// Current size of the admission map.
    ///
    /// Exposed for tests that assert map-entry retirement on the last
    /// waiter's drop (the K=1 admission discipline per SR-2.6). Also a
    /// defensible operator-introspection point — a future `/metrics`
    /// exporter could surface this as a gauge to flag a leak (entry not
    /// retired after all waiters drop) without coupling the metrics
    /// scraper to private fields.
    pub fn admission_len(&self) -> usize {
        self.admission.len()
    }

    /// Lazy on-demand recovery for `object_id` (SR-2.4 / SR-2.6).
    ///
    /// Algorithm:
    ///
    /// Step numbering matches SDD §22.5's hook flow, and the `// Step N:`
    /// markers in the body below.
    ///
    /// 1. Disabled short-circuit when `!self.enabled` — return
    ///    [`RecoveryOutcome::NotRecoverable`] without side effects.
    /// 2. Admission acquire: get-or-insert a `Arc<AdmissionEntry>` from
    ///    [`Self::admission`], increment its refcount via
    ///    [`AdmissionMapGuard::new`], and lock its K=1 mutex.
    /// 3. Map-entry retirement: the entry is released by
    ///    [`AdmissionMapGuard::drop`] only when the last in-flight attempt
    ///    for this `object_id` exits, so a queued waiter can never mint a
    ///    fresh entry that fails to serialize against the original.
    /// 4. Post-acquire re-check: if `obj:hdr` already exists for
    ///    `object_id` (a peer replica or the previous mutex holder landed
    ///    the recovery), return [`RecoveryOutcome::Restored`]
    ///    immediately.
    /// 5. Select the restore source via [`Self::select_restore_source`] —
    ///    the newest *restorable lineage* (a base plus its contiguous tails,
    ///    stepping down when one is broken; SR-3.1), with no feature
    ///    flag consulted — under a
    ///    [`crate::config::Config::checkpoint_recovery_fetch_budget`] timeout
    ///    (b.gg7 — the adapter's own ceiling is far longer than any caller
    ///    waits, and the deadline-exempt `SubscribeReadLayer` path has no tower
    ///    layer to bound it). `Ok(None)` → `NotRecoverable`; an elapse →
    ///    `Error`.
    ///
    ///    A selected lineage whose base is over the per-delta entry limit ends
    ///    `Error(Permanent)` (SR-14.7). There is no step-down, and it never
    ///    reaches the residue wipe, the ext pre-store or the write. The SDD
    ///    §22.6 step-2 identity verdict runs first, on the selected meta: when
    ///    no live mapping resolves to this `object_id` (the layer was deleted
    ///    or replaced during the state loss) the attempt ends
    ///    `NotRecoverable` exactly as step 7's `NotFound` arm does, a failed
    ///    mapping read ends `Error(Transient)`, and a wrong-typed mapping key
    ///    ends `Error(Permanent)`. Where that happens depends on the arm and
    ///    the base's shape:
    ///    - **Chunked (LEBC) base.** Refused before step 6: on the base-only
    ///      arm right after selection, on the composed arm inside the
    ///      selection before composing.
    ///    - **Plain OMPF base from an earlier release, composed arm.** Refused
    ///      inside the selection when the composition's base decode fails.
    ///    - **Plain OMPF base from an earlier release, base-only arm.** Fails
    ///      at `apply_restore`'s decode in step 7, after that function's own
    ///      identity guard.
    ///
    ///    See [`Self::adjudicate_over_limit_base_on_access`].
    ///
    ///    [`CheckpointStore::get_latest_layer_checkpoint`] is deliberately no
    ///    longer the selection call, and its trait contract is deliberately
    ///    left alone rather than redefined to mean "newest base" — see the
    ///    module header and `select_restore_source`'s own doc for why that
    ///    redefinition is refused.
    /// 6. High-water probe via [`Self::probe_stream_high_water`], supplying
    ///    `apply_checkpoint`'s `max_seq_observed`. **Not best-effort** (b.4et):
    ///    `Ok(None)` still means "no surviving stream, floor back to
    ///    `snap_seq`", but it is now only reachable on *positive* evidence that
    ///    the stream key is absent (or holds a foreign, non-stream value). An
    ///    `Err` — the stream is there and its high-water could neither be read
    ///    nor cleared — propagates straight to
    ///    [`RecoveryOutcome::Error`] WITHOUT calling `apply_restore`, so no
    ///    header is ever seeded underneath a stream top we could not prove we
    ///    cleared. See the probe's own "Fail-closed contract".
    /// 7. Apply via
    ///    [`apply_restore`](crate::checkpoint::apply::apply_restore), the
    ///    applied-outcome core.
    ///    `Ok(ApplyOutcome::Applied(..))` and `Ok(ApplyOutcome::AlreadyPresent)`
    ///    both → `Restored` (identical to pre-Epic-4 behaviour, where the EXISTS
    ///    short-circuit also returned `Ok(())`).
    ///    `Err(NotFound)` (apply's live-mapping guard rejected) →
    ///    `NotRecoverable`. Other errors → `Error(_)`. Every outcome arm
    ///    increments the recovery-attempt metric.
    pub async fn try_restore_layer(&self, object_id: u64) -> RecoveryOutcome {
        // Step 1: disabled short-circuit. The Subtask body explicitly
        // requires this path to NOT touch the admission map, store, or
        // metric — operators who disabled checkpointing should observe
        // zero recovery activity.
        if !self.enabled {
            return RecoveryOutcome::NotRecoverable;
        }

        // Step 2: per-`object_id` admission. `entry()` on the DashMap is
        // get-or-insert; we then clone the `Arc<AdmissionEntry>` out of
        // the entry guard so we can drop the dashmap shard lock before
        // acquiring the mutex (holding both would deadlock a concurrent
        // `remove_if` from another thread's `AdmissionMapGuard::drop`).
        let entry = self
            .admission
            .entry(object_id)
            .or_insert_with(|| Arc::new(AdmissionEntry::new()))
            .clone();
        let _map_guard = AdmissionMapGuard::new(&self.admission, entry.clone(), object_id);
        let _mutex_guard = entry.mutex.lock().await;

        // Step 4: post-acquire re-check. (Step 3, map-entry retirement, is
        // the `AdmissionMapGuard` above — it has no statement of its own.)
        // While we were queued at the
        // mutex, a previous holder may have completed the recovery for
        // this object. `apply_checkpoint` runs its own structural-header
        // short-circuit (step 1 of its algorithm), so we COULD let it run
        // through — but that wastes the restore-source selection's
        // round-trips, which on a misconfigured LocalDisk root with high
        // latency can be expensive. The cheaper short-circuit here is
        // strictly an optimization; the load-bearing correctness guard
        // lives inside `apply_checkpoint`.
        //
        // b.f4t: "present" here means structurally complete, not merely
        // `EXISTS` — a header missing `seq` / `created_by` is what its readers
        // already call `NotFound`, so it falls through to the restore rather
        // than short-circuiting as `Restored`.
        match self.is_layer_present(object_id).await {
            Ok(true) => {
                self.metrics.inc_checkpoint_recovery_attempt(
                    "restored",
                    "layer",
                    &self.service_label,
                );
                self.maybe_force_post_restore_notfound_for_test(object_id)
                    .await;
                return RecoveryOutcome::Restored;
            }
            Ok(false) => {
                // Fall through to checkpoint fetch + apply.
            }
            Err(err) => {
                self.metrics
                    .inc_checkpoint_recovery_attempt("error", "layer", &self.service_label);
                return RecoveryOutcome::Error(err);
            }
        }

        // Step 5: select the restore-source lineage, bounded at the caller's own
        // tolerance (b.gg7).
        //
        // ── What the bounded work actually is, post-gather ────────────────
        //
        // On the StorageApi adapter this is **one paginated LIST plus, per
        // lineage member reached, up to one fetch pair** (the multi-MB OMPF /
        // envelope GET and its meta GET). It is no longer the fixed
        // three-request sequence of the newest-base era: a lineage holds up to
        // `1 base + K tails`, and a selection that steps down walks more than one
        // candidate. Selection-wide fact caching keeps the *total* number of
        // generation probes bounded by the listing length rather than by
        // candidates × members — ~18 artifacts at stock knobs
        // (`retain_generations = 2`, `max_chain = 8`) — but the honest ceiling on
        // mutex-held work is still an order of magnitude above the pre-Epic-4
        // 1 LIST + 1-2 GETs.
        //
        // ── Why that makes the bound MORE load-bearing, not less (b.gg7) ──
        //
        // Each of those calls is bounded by the adapter's per-call ceiling and
        // the whole walk by its enumeration ceiling: knowable, but 540 s at the
        // defaults, which is two orders of magnitude past what an RPC waiter will
        // sit through — and the per-holder work profile above now compounds it.
        // The unary handlers are already protected: the tower deadline layer
        // bounds them at `LIVE_EDIT_RPC_DEADLINE_MS` regardless of what this hook
        // does. `SubscribeReadLayer` is deliberately exempt from that layer (it
        // is a streaming RPC), so on the lazy-recovery read path this call is the
        // one place a subscriber could block for the adapter's full ceiling — and
        // not alone: the per-`object_id` admission mutex acquired above means
        // every other waiter for the same layer is queued behind it. The bound
        // extends the unary posture to the streaming path and caps the
        // head-of-line cost at one budget per holder. That argument, and the 60 s
        // number it produces, are unchanged by gather; only the derivation of the
        // work it covers is.
        //
        // The budget is `Config::checkpoint_recovery_fetch_budget` — NOT
        // `rpc_deadline_ms` on its own, which would be the one number this
        // sequence provably cannot fit inside (SR-4.2 sizes the per-request
        // timeout so the OMPF GET alone may consume all of it, so a one-deadline
        // budget elapses deterministically on exactly the large layers DR exists
        // for). See that method for the full derivation.
        //
        // ── Two halves: cooperative inside, preemptive outside ────────────
        //
        // The SAME budget is spent twice over, deliberately:
        //
        //  - **Cooperatively**, as the `deadline` handed to
        //    `gather_restore_lineage` (with a one-fetch `reserve` of
        //    `checkpoint_request_timeout_ms`). A candidate that cannot fund its
        //    next member fetch inside the remaining budget is *abandoned and
        //    stepped down*, not errored — so running out of time can still reach
        //    the floor answer instead of stranding a restorable base.
        //  - **Preemptively**, as the `tokio::time::timeout` wrapper below, kept
        //    verbatim as the hard backstop. It is what covers the case the
        //    cooperative half structurally cannot see: the StorageApi adapter's
        //    *enforced* per-call ceiling is 180 s, so a single wedged call can
        //    outlive the whole budget before any cooperative check runs.
        //
        // The deadline is NOT inset below the wrapper's budget. Both are anchored
        // at the same instant on the same duration: the cooperative half is meant
        // to reach its verdict first in the ordinary case, and shaving it would
        // buy nothing but a narrower window in which a step-down beats the
        // backstop.
        //
        // The tie-break turns on where the gather anchors its per-fetch timer.
        // It matters because the gather's base fetch is allowanced at
        // `min(reserve, remaining)`: a base fetch — and equally the un-gated
        // first fetch, clamped to the same remainder — may legitimately run
        // right up to the cooperative deadline, where before b.bu8 it was
        // declined a whole reserve short of it. Landing AT the deadline must
        // resolve as a step-down (or a successful BASE-ONLY restore), never as
        // the preemptive `Transient` -> `Error` stranding.
        //
        // WHAT THE b.bu8 ACCOUNTING BUYS, exactly: a per-fetch timer can no
        // longer elapse AFTER the cooperative deadline. The gather spends each
        // fetch through `tokio::time::timeout_at(bound, ..)` on a bound derived
        // from the cooperative `deadline` itself (`lineage.rs`, `fetch_member`;
        // property (i) of `gather_restore_lineage`'s doc is the canonical
        // statement of that and this is a pointer at it). The duration form it
        // replaced could not say that: a `timeout(allowance, ..)` re-anchors
        // where the timer is CONSTRUCTED, some statements after `remaining` was
        // sampled off the deadline, so a fetch clamped to the remainder elapsed
        // at `deadline + delta` — possibly past the wrapper's own deadline. That
        // was the defect, and it is fixed.
        //
        // WHAT IT DOES NOT BUY: a strict ordering between the two deadlines.
        // Both are `tokio::time::Instant::now()` plus the same `fetch_budget` —
        // `deadline` below, and `tokio::time::timeout`'s own
        // `Instant::now().checked_add(duration)` two statements later (tokio
        // 1.53.1, `src/time/timeout.rs`). Under `tokio::time::pause()` — every
        // `start_paused = true` fence in `lineage.rs` — that clock is frozen, so
        // the two deadlines are EXACTLY EQUAL; on a live clock they differ by
        // the nanoseconds between two statements. And the bound covers when the
        // inner timer ELAPSES, not when the synchronous step-down after it
        // finishes, which is real work taking longer than that gap. So the tie —
        // inner elapse at the cooperative deadline, which is also the wrapper's —
        // is not a corner case but the normal case under a paused clock.
        //
        // AT THE TIE the cooperative verdict still wins, for a reason this crate
        // does not own. `Timeout::poll` polls its INNER future first and
        // consults its own delay only if that poll returned `Pending` (tokio
        // 1.53.1, `src/time/timeout.rs`). The walk's post-elapse work is
        // entirely synchronous — record the `Wedged` fact and the budget
        // step-down, decline every remaining candidate at a zero allowance (no
        // store call, so nothing can yield), `exhaust()`, return — so it
        // completes inside that same poll and the wrapper returns
        // `Poll::Ready(Ok(None))` without ever reaching its delay.
        //
        // UPGRADE OBLIGATION, scoped to that tie window: a tokio bump must
        // re-verify that `Timeout::poll` still polls its inner future before
        // consulting its own delay, and any edit to the walk must keep its
        // post-elapse path yield-free. The version above is the one locked in
        // `Cargo.lock`, NOT a pin: `Cargo.toml` declares `tokio = { version =
        // "1", .. }`, a caret range, so the lockfile and not the manifest is
        // what fixes what this was read against. Nothing in the build fails on
        // either reorder — the claim lives in tokio's source and in this walk's
        // shape, not in a type this crate names — and no test catches it either
        // (see the fence note below).
        //
        // One scoping caveat survives the change, for a different reason: on a
        // COMPOSED lineage `select_restore_source` awaits
        // `tokio::task::spawn_blocking(..)` for `compose_from_artifacts`
        // immediately after the base's fetch returns. The arithmetic above bounds
        // when the FETCH ends, not when the composition after it does, so a
        // composed restore can still run past the wrapper's deadline and elapse
        // it — the `Transient` -> `Error` the paragraph above rules out for the
        // `Ok(None)` step-down and the base-only restore. Left as is because the
        // direction is the benign one: `Transient` maps to `UNAVAILABLE`, which
        // the standard client retry policy retries, so a composed restore raced
        // at the deadline costs a retry rather than stranding a restorable base
        // at a terminal status.
        //
        // FENCE NOTE, and read it before citing a test here. The two b.bu8
        // Group C cases
        // `a_slow_listing_that_eats_the_budget_exhausts_honestly_rather_than_overrunning`
        // and
        // `the_rebase_budget_boundary_funds_the_chains_own_base_at_exactly_one_reserve`
        // (`lineage.rs`) drive a paused clock to exact `elapsed` equality, but
        // both call `gather_restore_lineage` DIRECTLY and never construct the
        // wrapper. What they pin is the ALLOWANCE ARITHMETIC — no fetch is
        // granted a bound past the cooperative deadline, so the selection stays
        // inside its own budget — not the tie-break. The tie-break itself has NO
        // fence and cannot be given one: a paused clock makes the tie exact and
        // makes the same-task step-down deterministic, so it cannot be made to
        // lose in a test.
        //
        // Preemption is safe here in the sense that matters: this is a pure read
        // — LIST + GETs, no PUT, no DELETE — so there is no partial write to
        // roll back and no durable state to leave half-formed. An elapse maps
        // into the existing `Error` arm, so the caller's retry-once semantics
        // and the `error` recovery-attempt metric are unchanged.
        //
        // It is NOT free, and the pre-Epic-4 claim that dropping the future
        // "cancels an in-flight request and nothing else" no longer holds. The
        // selection composes on `spawn_blocking`, and a `JoinHandle` dropped
        // mid-flight does not cancel the blocking task: it detaches. So an
        // elapse during composition ORPHANS a `compose_from_artifacts` call that
        // runs to completion on the blocking pool with its artifact `Bytes` (the
        // base plus every tail) pinned until it does. The bound on that:
        //
        //  - **Correctness — none.** `compose_from_artifacts` is pure. It writes
        //    nothing, touches no store, and its `ComposedLineage` is dropped
        //    unread when the detached task ends.
        //  - **Cost — one blocking-pool slot plus up to `TAIL_MAX_BYTES x
        //    TAIL_MAX_CHAIN` = 512 MiB of pinned artifact bytes per orphan**,
        //    for the remaining duration of one composition.
        //  - **Orphans can stack on one hot layer.** The timeout releases the
        //    admission mutex (that is the point of the budget — see the
        //    `stalled_checkpoint_fetch_releases_the_admission_mutex_for_the_next_caller`
        //    fence), so the next waiter starts its own selection while the
        //    previous composition is still running. Repeated elapses on the same
        //    object therefore accumulate concurrent orphans rather than
        //    serializing behind one, which is the direction that argues for the
        //    budget being generous enough that an elapse is genuinely
        //    exceptional, not a routine outcome on large layers.
        //
        // Accepted as-is: making it cancellable would need a cooperative check
        // threaded through the replay loop, which buys nothing on the
        // correctness side and costs a yield per record on the path DR exists
        // for.
        //
        // Note also what this budget does NOT cover (SR-3.4): the composed
        // restore's external-value pre-store at step 7 — up to 512 MiB worth of
        // `store_values` transactions — runs AFTER the fetch budget is spent and
        // still under the admission mutex acquired above. The holder-side cost
        // of a composed restore is therefore `checkpoint_recovery_fetch_budget`
        // *plus* an unbounded-by-this-budget write phase. Accepted: each
        // individual command in that phase is bounded by
        // `LIVE_EDIT_VALKEY_COMMAND_TIMEOUT_MS`, and the work is proportional to
        // the bytes the restore must write to be correct — capping it would mean
        // abandoning a restore mid-way, which the single `write_restored_object`
        // MULTI/EXEC exists precisely to avoid.
        let fetch_budget = self.config.checkpoint_recovery_fetch_budget();
        let deadline = Instant::now() + fetch_budget;
        let reserve = Duration::from_millis(self.config.checkpoint_request_timeout_ms);
        // The whole of `select_restore_source` sits INSIDE the timeout — the
        // listing, every member fetch, every composition and every step-down.
        // Hoisting any of it above the `tokio::time::timeout` would leave that
        // work unbounded while the per-`object_id` admission mutex acquired above
        // is held — reintroducing precisely the head-of-line stall b.gg7 added
        // this budget to cap.
        let fetch = tokio::time::timeout(
            fetch_budget,
            self.select_restore_source(object_id, deadline, reserve),
        )
        .await;
        let fetch = match fetch {
            Ok(result) => result,
            Err(_elapsed) => {
                tracing::warn!(
                    object_id,
                    budget_ms = fetch_budget.as_millis() as u64,
                    "checkpoint recovery abandoned its restore-source selection at the recovery \
                     fetch budget; the adapter's own ceiling is far longer than any caller \
                     waits, and holding the per-object admission mutex meanwhile would queue \
                     every other reader of this layer behind it"
                );
                Err(CheckpointError::Transient {
                    source: format!(
                        "restore-source selection exceeded the {}ms recovery fetch budget",
                        fetch_budget.as_millis()
                    )
                    .into(),
                })
            }
        };
        let selected = match fetch {
            Ok(Some(selected)) => selected,
            Ok(None) => {
                // No *restorable* lineage for this object, and — the part that
                // licenses a terminal answer — the selection KNOWS that rather
                // than having run out of time: the layer was never checkpointed
                // (pre-first-tick window after CreateLayer), was already pruned
                // by retention, holds only non-base generations, or every
                // candidate lineage broke with every listed full generation
                // adjudicated (a missing link, a header that contradicts its
                // link, an unreadable envelope, a budget break above bases the
                // walk did read, or a composition the engine rejected). Every
                // one of those is `Ok(None)` from `select_restore_source` and
                // lands here identically: recovery is not possible; surface
                // NotFound to the caller. The remaining distinction among them
                // is *diagnostic* and lives on the `exhausted` lineage-metric
                // cell and the step-down WARNs, not on this arm.
                //
                // What does NOT reach here is an exhaustion that left a listed
                // base unread. That is `Exhaustion::BudgetInconclusive`, which
                // `select_restore_source` returns as `CheckpointError::Transient`
                // so it lands on the `Err` arm below and the client retries
                // (b.hxk). Budget exhaustion is therefore no longer a single
                // policy ending at `NotFound`: it splits on whether the walk had
                // evidence about the artifacts that can root a lineage.
                //
                // One more `Ok(None)` is not an exhaustion: a composed lineage
                // whose base is over the per-delta entry limit, whose layer the
                // identity verdict found gone (no live mapping resolves to this
                // `object_id`). It lands here deliberately, because this arm's
                // answer is the one the apply's `NotFound` arm below gives a
                // gone layer; `select_restore_source` ticks no lineage cell for
                // it.
                self.metrics.inc_checkpoint_recovery_attempt(
                    "not_recoverable",
                    "layer",
                    &self.service_label,
                );
                return RecoveryOutcome::NotRecoverable;
            }
            Err(err) => {
                self.metrics
                    .inc_checkpoint_recovery_attempt("error", "layer", &self.service_label);
                return RecoveryOutcome::Error(err);
            }
        };

        let SelectedRestore {
            payload,
            head_seq,
            meta,
            chain_len,
            outcome: lineage_outcome,
        } = selected;

        // SR-14.7, base-only arm: a chunked (LEBC) base is refused here, before
        // the high-water probe and `apply_restore`, so its residue wipe, ext
        // pre-store and single `write_restored_object` transaction are never
        // reached. The composed arm is refused inside `select_restore_source`,
        // before its composition. An earlier release's plain OMPF base over the
        // limit is not caught here. It fails `Permanent` at `apply_restore`'s
        // decode, which also runs before those writes (and after that
        // function's own identity guard), though after the high-water probe
        // below.
        //
        // The refusal is preceded by the SDD §22.6 step-2 identity verdict, so
        // a layer deleted or replaced during the state loss answers
        // `NotRecoverable` — the same verdict and metric cell as the
        // `Err(CheckpointError::NotFound)` arm at the bottom of this function —
        // rather than the refusal's `Error(Permanent)`. See
        // `adjudicate_over_limit_base_on_access` for each of its answers.
        //
        // The detection is a synchronous check on bytes already in hand, and
        // the verdict's round trip runs only inside this branch, so no `.await`
        // is added to the success path. Boxed for the same debug-build stack
        // reason as the `apply_restore` call below: this future is inlined into
        // `SubscribeReadLayer`'s catch-up chain.
        if matches!(&payload, RestorePayload::Ompf(bytes) if is_chunked_base(bytes)) {
            return match Box::pin(
                self.adjudicate_over_limit_base_on_access(object_id, head_seq, &meta),
            )
            .await
            {
                CheckpointError::NotFound => {
                    self.metrics.inc_checkpoint_recovery_attempt(
                        "not_recoverable",
                        "layer",
                        &self.service_label,
                    );
                    RecoveryOutcome::NotRecoverable
                }
                err => {
                    self.metrics.inc_checkpoint_recovery_attempt(
                        "error",
                        "layer",
                        &self.service_label,
                    );
                    RecoveryOutcome::Error(err)
                }
            };
        }

        // `snap_seq` is **S**, the lineage head — NOT the base's sequence.
        //
        // Pre-Epic-4 this was `blob.meta.snap_seq.unwrap_or(blob.handle.snap_seq)`,
        // which for a composed lineage would report the BASE's number and
        // silently under-state what the restore actually materialized: apply
        // would then seed the writer cursor from a floor `S - base_snap_seq`
        // commits too low and re-issue sequence numbers the replayed tails
        // already used. On a base-only lineage the two are provably the same
        // value — `select_restore_source` asserts the agreement at the source
        // and prefers the old expression on a mismatch — so this line is
        // behaviour-identical there.
        let snap_seq = head_seq;

        // The SR-3.6 re-base's second required input, captured HERE as a plain
        // `u64` rather than read off `meta` at the applied arm. `meta` is still
        // live down there and the read would compile either way, but pinning the
        // value at the selection site removes a cross-lane lifetime coupling:
        // nothing about how the apply call borrows or consumes `meta` can change
        // what the re-base is told. It is deliberately NOT recoverable from
        // `AppliedCheckpoint` — `band = max(configured margin,
        // observed_commit_span)` is not invertible — which is why it has to
        // travel separately at all.
        let observed_commit_span: u64 = meta.observed_commit_span;

        tracing::debug!(
            object_id,
            snap_seq,
            chain_len,
            outcome = lineage_outcome,
            "checkpoint recovery selected a restore source"
        );

        // Step 6: high-water probe (b.8y5, fail-closed since b.4et). When the
        // per-object stream survived the state loss this yields the exact
        // pre-crash sequence number, which both tightens the guard band and
        // keeps post-restore `XADD`s from colliding with a surviving stream
        // top. `snap_seq` is passed in as the anchor for the plausibility
        // bound; `meta` is what authorizes the destructive cleanup on the
        // refusal branch (the SR-2.4 identity guard) — see the probe's rustdoc.
        //
        // The `?`-shaped arm below is the load-bearing half of b.4et. Pre-b.4et
        // every failure mode folded to `None`, which seeds the header at
        // `snap_seq + band` — BELOW a stream top we never read. Post-b.ph9 that
        // is no longer corruption, but it is a full, silent-until-first-write
        // outage on the restored layer. Refusing here instead makes the failure
        // loud, bounded, and retriable on the `Transient` arm: the handlers map
        // `Transient` to `UNAVAILABLE` (`ServiceError::from_recovery_error`) and
        // their recovery dispatch is retry-once, so there is no spin. A
        // `Permanent` from any restore-path key read (three producers, each
        // already logged loudly at the read that raised it — see the note on
        // `probe_stream_high_water`) is deliberately terminal: it propagates
        // verbatim through this arm to `INTERNAL`.
        let max_seq_observed = match self
            .probe_stream_high_water(object_id, snap_seq, &meta)
            .await
        {
            Ok(observed) => observed,
            Err(err) => {
                // No `apply_restore` call on this arm — that is the whole
                // point. Nothing is written, no header is seeded, and the
                // surviving stream is left exactly as we found it.
                self.metrics
                    .inc_checkpoint_recovery_attempt("error", "layer", &self.service_label);
                return RecoveryOutcome::Error(err);
            }
        };

        // Step 7: apply, through the applied-outcome core.
        //
        // ONE call for both paths. `payload` is `Ompf(base bytes)` on a
        // base-only lineage — the pre-Epic-4 path verbatim, and `ext_values` is
        // empty there by construction, because the assembler serializes base
        // OMPFs with zero external parts — and `Composed { state, ext_values }`
        // otherwise.
        //
        // The `ExtPrestore` is wired whenever the hook holds a store, so the
        // SR-3.4 pre-store runs strictly before the single `write_restored_object`
        // MULTI/EXEC. Handing it in unconditionally (rather than only on the
        // composed arm) costs nothing on the base-only path — the core skips the
        // whole step on an empty ext set — and keeps the fail-closed guard the
        // ONLY place that decides what a missing store means.
        //
        // Applied-outcome core, not the `apply_checkpoint` compat wrapper: that
        // is what puts `AppliedCheckpoint` in scope at the `restored`
        // arm, which Task 4's SR-3.6 re-base consumes on EVERY restore —
        // base-only included, since every restore leaves `seeded_seq > snap_seq`
        // and SR-3.6's conditional no-gap arm is therefore dead.
        let ext = self.ext_value_store.as_ref().map(|store| {
            let prestore = ExtPrestore::new(store.as_ref());
            #[cfg(feature = "test-checkpoint-fault-injection")]
            let prestore = match self.restore_fault_injector.as_ref() {
                Some(injector) => prestore.with_fault_injector(injector.as_ref()),
                None => prestore,
            };
            prestore
        });

        // `Box::pin`, not a bare `.await`, for the same reason the residue wipe
        // inside `apply_restore` is boxed — see that call site. `apply_restore`
        // inlines the whole composed-restore engine and the SR-3.4 ext pre-store
        // on top of the wipe, so its state machine is the largest single
        // contributor to this hook's future, and this hook's future is in turn
        // inlined into `SubscribeReadLayer`'s catch-up chain on a 2 MiB tokio
        // worker. A debug build gives a future its own non-overlapping slot at
        // each branch and move site it passes through, so the saving from
        // boxing is paid several times over; boxing at BOTH sites is what takes
        // the catch-up chain's stack demand back under the worker stack in a
        // debug build.
        //
        // Any new `async fn` awaited inline anywhere on the catch-up path — not
        // only in checkpoint code — draws on that same debug-build stack, and
        // the several-times-over multiplication means a small addition can
        // consume a lot of it. The catch-up's regime futures are boxed at
        // their dispatch in `run_catch_up_inner`, and their debug sizes are
        // fenced by `grpc::subscribe`'s
        // `run_catch_up_future_stays_under_its_recorded_debug_ceiling`. A debug
        // overflow is a test-lane signal, not a product one: release overlaps
        // the slots. The product margin is measured in a release build by the
        // release stack guard's subscribe layer-recovery case
        // (`tests/release_stack_guard/`).
        match Box::pin(apply_restore(
            &self.layer_store,
            object_id,
            payload,
            snap_seq,
            &meta,
            max_seq_observed,
            &self.config,
            ext,
        ))
        .await
        {
            // ── SR-3.6 post-restore gap re-base — this arm, and ONLY this arm ──
            //
            // Both of the re-base's inputs are in scope here on BOTH the
            // base-only and composed paths, with no re-fetch and no re-compose:
            // the `AppliedCheckpoint` (the already-in-memory restored
            // `ObjectState`, its `seeded_seq`, and the `band` applied) and the
            // `observed_commit_span` pinned at the selection site above.
            //
            // `AlreadyPresent` (below) is deliberately NOT a spawn point, and
            // neither is the step-4 structurally-complete-header
            // short-circuit: both
            // return `Restored` without this replica having composed anything,
            // and re-basing there would mint a base for a state we never held.
            Ok(ApplyOutcome::Applied(applied)) => {
                tracing::debug!(
                    object_id,
                    snap_seq,
                    chain_len,
                    outcome = lineage_outcome,
                    seeded_seq = applied.seeded_seq,
                    band = applied.band,
                    // Logged so the value the re-base takes is visible in the
                    // same line as the band it cannot be recovered from.
                    observed_commit_span,
                    "checkpoint recovery restored a layer"
                );
                self.metrics.inc_checkpoint_recovery_attempt(
                    "restored",
                    "layer",
                    &self.service_label,
                );

                // ── SR-3.6: close the gap this restore just opened ──────────
                //
                // Runs on EVERY restore, base-only included. `apply_restore`
                // seeds `seeded_seq = restore_floor + band` with `band =
                // max(checkpoint_seq_safety_margin, meta.observed_commit_span)`
                // and the margin is zero-rejected, so `band >= 1` always and the
                // live header ALWAYS lands strictly above the generation the
                // restore was seeded from. SR-3.6's old conditional no-gap arm
                // is therefore dead, and no `seeded_seq > snap_seq` test appears
                // here. Without the re-base a restored layer has no generation
                // at its seeded sequence and tail capture refuses it
                // (`PostRestoreUnrebased`) until its first post-restore periodic
                // assembly lands a base and clears `restore_seq_bump`.
                //
                // Spawn, never await: this body holds the per-`object_id`
                // admission mutex and runs on the lazy-recovery read path
                // (`SubscribeReadLayer` included, which is exempt from the tower
                // deadline layer). See `spawn_restore_rebase`'s own rustdoc.
                //
                // ── How this composes with the gates downstream ─────────────
                //
                // **Fossil-free by construction.** `band >= 1` means `seeded_seq
                // > restore_floor >= max_seq_observed >=` every surviving
                // abandoned-timeline entry. The re-base puts S0 at `seeded_seq`,
                // so the capture's read window `{seeded_seq+1}-0` starts strictly
                // above every fossil and SR-1.2's ascending-jump rule has
                // nothing to fire on.
                //
                // **The skip gate, and why firing it is correct.** After a
                // successful re-base `obj:hdr seq == last_checkpoint_seq ==
                // seeded_seq`, so the next tick's skip-if-no-commits gate fires,
                // ticks `{result="skipped",kind="layer"}`, and continues — no
                // assembly, no capture attempt. There is nothing to capture. The
                // first post-restore commit lands at `seeded_seq+1` and the layer
                // is eligible again: its assembly's read proves a cut however hot
                // the layer is (atomic at or below the atomic-read limits, fenced
                // above them), so it gets a normal verified snapshot. Only a
                // header-sequence regression returns `SkippedNotQuiesced`, and then
                // the capture fallback finds S0 = `seeded_seq` from the listing
                // with a sound prefix at `{seeded_seq+1}-0`.
                //
                // **The third cursor column is what actually unblocks capture.**
                // `tail_capture` refuses with
                // `TailCaptureRefusal::PostRestoreUnrebased` while
                // `restore_seq_bump != 0`, BEFORE the stream read. Without the
                // re-base's `restore_seq_bump = 0` stamp a restored layer refuses
                // on every tick regardless of stream survival — which is why the
                // re-base matters even for a layer whose stream is intact.
                //
                // **Metrics.** The restore-time producer ticks the same
                // `checkpoint_writes_total` cells as the creation-time one:
                // `{success,layer}` on a completed re-base, `{error,layer}` on
                // identity-read / OMPF-build / sidecar-serialization / PUT /
                // pre-stamp-HMGET / HSET failure. The guard skips and the
                // admission shed tick nothing. No new cells.
                //
                // TWO RE-BASES, ONE CELL — a LIVE invariant, not a reservation.
                // This SR-3.6 post-restore re-base MUST NOT tick
                // `live_edit_checkpoint_tail_total{outcome="rebased"}`. That
                // cell now has a real producer: the SR-4.1 chain-collapse
                // re-base (`checkpoint::rebase::rebase_layer_lineage`, driven
                // from the checkpoint tick by
                // `checkpointer::rebase_on_chain_full`). The separation is what
                // gives the cell a single meaning — "a layer's tail chain
                // reached K and was collapsed into a fresh base" — and it is
                // load-bearing in BOTH directions: the tail-outcome family
                // counts what one *capture* decision did with a layer-tick, and
                // this producer makes no capture decision at all. It is a write,
                // reached from the restore hook rather than from the tick's
                // capture arm, so it belongs on `writes_total` and nowhere else.
                // Ticking it here would make `rebased` unusable for the thing it
                // does measure: a DR restore of N layers would move it N times
                // with no chain collapsed anywhere, and the chain-health signal
                // would be buried under the recovery it has nothing to do with.
                // `tests/integration/checkpoint/restore_rebase_test.rs` is where
                // the separation is pinned.
                //
                // **Failure / shed posture.** Nothing is stamped and no retry
                // machinery is added beyond what SR-7 defines. The layer's
                // capture then refuses observably as `refused_seq_gap` (the
                // bump is simply never cleared) until a verified snapshot, a
                // later bootstrap attempt, or a re-restore lands.
                if let Some(bootstrap) = self.base_bootstrap.as_ref() {
                    let AppliedCheckpoint {
                        state,
                        seeded_seq,
                        band,
                        ..
                    } = applied;
                    bootstrap.spawn_restore_rebase(
                        object_id,
                        state,
                        BootstrapTarget::RestoreRebase {
                            seeded_seq,
                            observed_commit_span,
                            absorbed_band: band,
                        },
                    );
                }

                self.maybe_force_post_restore_notfound_for_test(object_id)
                    .await;
                RecoveryOutcome::Restored
            }
            // NOT a Task 4 hook point: nothing was composed here, and the peer
            // that actually landed the restore owns its own re-base.
            //
            // The metric arm is IDENTICAL to today's. Pre-Epic-4
            // `apply_checkpoint` returned `Ok(())` for the same presence
            // short-circuit — a bare `EXISTS` then, the structural-header
            // predicate since b.f4t — and the hook counted it `restored`;
            // splitting an `already_present`
            // label out here would break `sum by (result) == attempts` and the
            // existing assertions. That cell is session-scope only
            // (`src/checkpoint/session_recovery.rs`); the layer scope's `result`
            // vocabulary stays `restored | not_recoverable | error`.
            Ok(ApplyOutcome::AlreadyPresent) => {
                self.metrics.inc_checkpoint_recovery_attempt(
                    "restored",
                    "layer",
                    &self.service_label,
                );
                self.maybe_force_post_restore_notfound_for_test(object_id)
                    .await;
                RecoveryOutcome::Restored
            }
            Err(CheckpointError::NotFound) => {
                // The apply's SR-2.4 live-mapping guard rejected: no
                // session→layer mapping resolves to this object_id
                // anymore (DeleteLayer-checkpoint-delete race, or
                // CreateLayer-overwrite during state-loss minted a fresh
                // object_id). Recovery is correctly skipped here;
                // resurrecting state would create a phantom layer. A base over
                // the per-delta entry limit gets this same answer from
                // `adjudicate_over_limit_base_on_access`, before its refusal.
                self.metrics.inc_checkpoint_recovery_attempt(
                    "not_recoverable",
                    "layer",
                    &self.service_label,
                );
                RecoveryOutcome::NotRecoverable
            }
            Err(err) => {
                self.metrics
                    .inc_checkpoint_recovery_attempt("error", "layer", &self.service_label);
                RecoveryOutcome::Error(err)
            }
        }
    }

    /// Pick the **newest restorable lineage** this restore starts from: a
    /// reset-decodable [`CheckpointKind::Base`] plus the contiguous run of
    /// [`CheckpointKind::Tail`] envelopes that link onto it, replayed to the head
    /// sequence `S` (SDD SR-3.1 / SR-3.2).
    ///
    /// # This was Epic 4's single swap point, and this is the swap
    ///
    /// Selection policy lives here and nowhere else. [`Self::try_restore_layer`]
    /// gained no lineage vocabulary: it sees one
    /// `Result<Option<SelectedRestore>, _>` and its admission machinery, fetch
    /// budget and metric arms are untouched. Keep it that way — a selection
    /// policy that leaks into the hook body is a policy the next Epic has to
    /// re-thread through all three.
    ///
    /// # The loop
    ///
    /// Drive [`LineageSelection::next_restorable`] until it answers:
    ///
    /// - **A base-only lineage** (`tails.is_empty()`) → today's exact answer.
    ///   Emit [`RestorePayload::Ompf`] carrying the base's bytes, `meta =
    ///   base.meta`, `head_seq = base.handle.seq()`, outcome `gathered_base`.
    ///
    ///   Base-only is deliberately **not** routed through the composition
    ///   engine, even though `compose_from_artifacts` accepts a zero-tail
    ///   lineage. A corrupt base must keep surfacing as it does today — a decode
    ///   failure *inside apply*, i.e. `Error` — rather than as a step-down to an
    ///   older generation. That is a deliberate divergence from a literal reading
    ///   of SR-3.5, and it is what makes "existing base-only restore behaviour
    ///   unchanged" trivially provable rather than argued.
    ///   A base over the per-delta entry limit is also emitted like any other
    ///   base (SR-14.7). The hook body refuses a chunked (LEBC) one right after
    ///   selection. A plain OMPF one from an earlier release fails at apply's
    ///   decode, like a corrupt base, before anything is written.
    /// - **A lineage with tails** → if its base is a chunked (LEBC) base, it is
    ///   answered before composing, with no step-down and no lineage cell
    ///   (SR-14.7, [`Self::adjudicate_over_limit_base_on_access`]): `Ok(None)`
    ///   when the identity verdict finds no live mapping resolving to this
    ///   `object_id` (the layer is gone, so the hook answers `NotRecoverable`),
    ///   otherwise `Err` — the refusal's `Permanent` when the layer is still
    ///   mapped, or the verdict's own error. Otherwise compose it. `Ok` emits
    ///   [`RestorePayload::Composed`] with outcome `gathered_composed`. A
    ///   [`ComposeError`](crate::checkpoint::compose::ComposeError) records
    ///   `stepped_down_compose` through
    ///   [`LineageSelection::record_compose_failure`], WARNs with the offending
    ///   generation the error attributes itself to, and **continues the loop** —
    ///   never `Error`, and never a retry against the same bytes, because every
    ///   `ComposeError` is `Permanent` by construction (the engine does no I/O,
    ///   so a retry reaches the same verdict). The one exception is a base
    ///   decode that failed because the base is a plain OMPF delta over the
    ///   per-delta entry limit (`apply::is_over_entry_limit_base`). That is
    ///   answered like the chunked case — the identity verdict, then
    ///   `Ok(None)` or `Err` — with no step-down and no lineage cell (SR-14.7).
    /// - **`Ok(None)`** → `Ok(None)`, landing on the hook's unchanged
    ///   `not_recoverable` arm. The over-limit cases above are the only other
    ///   `Ok(None)` this function returns, and they tick no exhaustion cell.
    /// - **`Err(_)`** from the store propagates verbatim to the unchanged
    ///   `Error` arm — but only *after* the step-downs the walk had already
    ///   accumulated are emitted. A single `next_restorable` call can break on
    ///   one candidate and then fail the store on the next, and those breaks are
    ///   real regardless of how the call ended.
    ///
    /// # The composed meta — three sourcing decisions, adjudicated here
    ///
    /// Gather hands over both metas. Apply reads exactly six fields off the one
    /// it is given, and they split three ways:
    ///
    /// 1. **Identity + header columns (`session_id`, `layer_name`, `created_by`,
    ///    `server_ts`) — from the BASE's meta, always.**
    ///    `live_mapping_resolves_blob_identity` refuses an empty `session_id` /
    ///    `layer_name` outright, and a tail's identity is base-sourced by
    ///    construction. [`GatheredTail::meta`] is diagnostics, never an
    ///    identity carrier. (The rule was originally motivated by the adapters'
    ///    identity-empty meta-absent tail placeholder; b.hka (d) replaced that
    ///    with an adapter-side refusal, and the rule is unaffected.)
    /// 2. **`observed_commit_span` — from the head tail's ENVELOPE HEADER,
    ///    `header.to_seq - header.from_seq`.** The band exists to cover commits
    ///    that landed *after the newest generation* and died with the crash, so
    ///    the honest input is this layer's **most recent** measured interval. The
    ///    base's span measures the interval before the base was written (stale),
    ///    and the whole-lineage span `S - S0` measures K intervals at once,
    ///    inflating the band. Taken from the header rather than the meta
    ///    because the header is the artifact's own framing and cannot disagree
    ///    with itself, while any meta that reaches here with a zeroed column
    ///    would silently degrade a hot layer's band back to the configured
    ///    margin — the smaller-band hazard direction (b.8y5). (The concrete
    ///    zeroing producer was the adapters' meta-absent tail placeholder,
    ///    which b.hka (d) replaced with a refusal; sourcing from the header
    ///    remains right because it removes the class, not just that instance.)
    /// 3. **`written_at_unix_ms` — the newest generation's own write time**
    ///    (SR-3.3), i.e. the head tail's `meta.written_at_unix_ms`, falling back
    ///    to the base's when the head's is the `0` placeholder sentinel (which
    ///    then falls through apply's existing `0 -> now_unix_ms()` arm).
    ///
    /// The composed meta is therefore `base.meta` with exactly those two
    /// non-identity columns overridden. **No other column is head-sourced.**
    /// The construction lives in [`adjudicate_composed_meta`], shared with the
    /// SR-4.1 chain-collapse re-base
    /// ([`rebase_layer_lineage`](crate::checkpoint::rebase::rebase_layer_lineage))
    /// so the base-sourced-identity MUST has one home; `observed_commit_span` is an argument there precisely because the
    /// two callers derive it differently, and that helper's doc records both
    /// derivations side by side.
    ///
    /// # Base-only keeps today's `snap_seq` expression
    ///
    /// On a base-only lineage `head_seq == base.handle.seq()`, and today's
    /// defensive `meta.snap_seq.unwrap_or(handle.snap_seq)` must agree with it.
    /// The agreement is a `debug_assert!` plus a production WARN that
    /// **prefers today's value** — deliberately not a hard `assert!`: a
    /// non-conforming adapter must not panic the recovery hook, and preferring
    /// the pre-Epic-4 expression is what keeps the base-only fence exact.
    ///
    /// # Metrics
    ///
    /// This function owns **every** increment of
    /// `live_edit_checkpoint_restore_lineage_total` —
    /// `checkpoint::lineage` holds no metrics handle. One `gathered_*` per
    /// resolved lineage, one `stepped_down_*` per [`StepDownRecord`] the
    /// selection accumulated (walk-side breaks and compose-driven ones alike),
    /// and one exhaustion cell — `exhausted` on an exhaustion's `Ok(None)`
    /// (the over-limit `Ok(None)` for a gone layer ticks none), `exhausted_budget`
    /// on the `Exhaustion::BudgetInconclusive` arm that returns
    /// `CheckpointError::Transient` instead (b.hxk). Every spelling comes from a
    /// decision type's `*_label()`, never a literal.
    /// `checkpoint_recovery_attempts_total` is untouched and stays
    /// one-tick-per-attempt: the retriable arm ticks `result="error"` through
    /// the caller's existing `Err` routing, not a second cell of its own.
    ///
    /// [`GatheredTail::meta`]: crate::checkpoint::lineage::GatheredTail::meta
    /// [`StepDownRecord`]: crate::checkpoint::lineage::StepDownRecord
    async fn select_restore_source(
        &self,
        object_id: u64,
        deadline: Instant,
        reserve: Duration,
    ) -> Result<Option<SelectedRestore>, CheckpointError> {
        // The selection borrows `self.store` for its whole life and is consumed
        // INLINE — never spawned, never stored on a struct field, never made
        // `'static`. `self.store.as_ref()` borrows fine for exactly that shape.
        let mut selection =
            gather_restore_lineage(self.store.as_ref(), object_id, deadline, reserve).await?;
        let mut emitted_step_downs = 0usize;

        loop {
            // The result is BOUND, drained, and only then `?`-propagated — the
            // order is load-bearing. One `next_restorable` call can walk several
            // candidates: each break pushes a `StepDownRecord` and the walk
            // continues, so a store error raised on a *later* candidate's fetch
            // unwinds with earlier candidates' records already accumulated.
            // Applying `?` at the call site would short-circuit past the drain
            // and lose them, undercounting `stepped_down_*` on exactly the
            // "generation broke, then the backend failed" sequence operators
            // most need to see. Pinned by
            // `select_restore_source_emits_step_downs_accumulated_before_a_store_error`.
            let next = selection.next_restorable().await;
            // Drain whatever the walk just abandoned. One increment per
            // `StepDownRecord`, whatever produced it.
            self.emit_new_step_downs(&selection, &mut emitted_step_downs);
            let next = next?;

            let Some(lineage) = next else {
                // Exhausted, and the variant decides which of two answers this
                // is. Three of them are CONCLUSIVE — empty listing, no base, or
                // every listed base adjudicated and every candidate broken — and
                // they fold to one metric cell and one `Ok(None)`, where the
                // remaining split is diagnostic and already WARNed by the
                // selection. `BudgetInconclusive` is not: the walk never read
                // some listed base, so whatever it did adjudicate is not
                // conclusive evidence that the layer is unrestorable, and
                // reporting `Ok(None)` there would tell the client "this layer
                // has no backup" on the strength of the store being slow
                // (b.hxk).
                let exhaustion = selection.exhaustion();
                let outcome = exhaustion
                    .map(Exhaustion::outcome_label)
                    // Unreachable on this loop — it returns the moment a lineage
                    // verifies, so a drained cursor here always carries an
                    // exhaustion record. Folded onto a CONCLUSIVE variant's label
                    // rather than spelled as a literal, so the vocabulary stays
                    // single-sourced AND an absent record can never be read as
                    // the retriable answer: with no record there is nothing that
                    // says a base went unread, and inventing a retry out of a
                    // contract violation would spin the caller forever.
                    .unwrap_or_else(|| Exhaustion::AllBroken { candidates: 0 }.outcome_label());
                self.metrics
                    .inc_checkpoint_restore_lineage(outcome, &self.service_label);
                if let Some(Exhaustion::BudgetInconclusive {
                    unadjudicated_bases,
                    ..
                }) = exhaustion
                {
                    // `Transient` rather than a fourth `RecoveryOutcome`
                    // variant: `RecoveryOutcome`'s own doc makes the error class
                    // decide the status code (b.33e), and `Transient` already
                    // routes `Error(_)` -> `recovery_error_is_retriable` ->
                    // `UNAVAILABLE` at every one of this hook's callers with no
                    // edit. The message lands verbatim inside the callers'
                    // operator-facing prefixes, so it names both numbers an
                    // operator needs to tell this apart from a real data loss.
                    let budget_ms = self.config.checkpoint_recovery_fetch_budget().as_millis();
                    return Err(CheckpointError::Transient {
                        source: format!(
                            "restore-source selection exhausted its {budget_ms}ms fetch budget \
                             with {unadjudicated_bases} of this layer's full checkpoint \
                             generations still unread; the layer may be restorable once the \
                             checkpoint store answers, so this is retriable rather than a \
                             missing backup"
                        )
                        .into(),
                    });
                }
                return Ok(None);
            };

            // Labels and shape facts are read BEFORE the lineage is destructured.
            let outcome = lineage.outcome_label();
            let chain_len = lineage.chain_len();
            let head_seq = lineage.head_seq;
            let base_only = lineage.is_base_only();
            // Per the gather contract: the head kind is `Tail` iff the lineage
            // has tails. Derived from the RETURNED lineage, never from a listing
            // handle — the listing head and the verified lineage head are not
            // the same generation once a step-down has happened.
            let head_kind = if base_only {
                CheckpointKind::Base
            } else {
                CheckpointKind::Tail
            };
            let GatheredLineage { base, tails, .. } = lineage;

            if base_only {
                // ── Base-only: today's exact answer ──────────────────────────
                //
                // `head_seq` and the pre-Epic-4 expression must agree. The
                // `debug_assert!` fails a test build on a divergence; production
                // WARNs and prefers today's value, because a defective adapter
                // must not panic the recovery hook and the fence requires the
                // pre-Epic-4 behaviour exactly.
                let today = base.meta.snap_seq.unwrap_or(base.handle.snap_seq);
                debug_assert_eq!(
                    today, head_seq,
                    "base-only lineage: GatheredLineage::head_seq must equal the blob's own \
                     meta.snap_seq.unwrap_or(handle.snap_seq)"
                );
                let head_seq = if today == head_seq {
                    head_seq
                } else {
                    tracing::warn!(
                        object_id,
                        head_seq,
                        legacy_snap_seq = today,
                        "base-only restore selection: the gathered head sequence disagrees with \
                         the blob's own meta.snap_seq/handle.snap_seq; preferring the latter so \
                         base-only restore behaviour stays exactly as it was. Suspect a \
                         non-conforming adapter"
                    );
                    today
                };

                self.metrics
                    .inc_checkpoint_restore_lineage(outcome, &self.service_label);
                return Ok(Some(SelectedRestore {
                    payload: RestorePayload::Ompf(base.bytes),
                    head_seq,
                    meta: base.meta,
                    chain_len,
                    outcome,
                }));
            }

            // ── Composed: replay the tails onto the base ─────────────────────
            //
            // Source the two head-generation meta columns BEFORE the artifacts
            // are moved into the composition, because `tails` does not survive
            // it (see the memory note below).
            let head_tail = tails
                .last()
                .expect("a non-base-only lineage has at least one tail");
            let head_span = head_tail
                .header
                .to_seq
                .saturating_sub(head_tail.header.from_seq);
            let head_written_at = head_tail.meta.written_at_unix_ms;

            let tail_bytes: Vec<Bytes> = tails.iter().map(|tail| tail.bytes.clone()).collect();
            // The `GatheredTail` records (handles, headers, metas) are done with;
            // only the artifact `Bytes` handles go forward, and only into the
            // blocking closure below.
            drop(tails);

            let CheckpointBlob {
                bytes: base_bytes,
                meta: base_meta,
                handle: base_handle,
            } = base;
            let base_snap_seq = base_handle.seq();

            // SR-14.7, composed arm: a chunked (LEBC) base is refused before its
            // bytes move into the composition. A chunked base decodes, so without
            // this check the lineage would simply compose and hand the whole layer
            // to `write_restored_object`'s single transaction, which SR-14.7
            // forbids.
            //
            // It is refused as `Err(Permanent)` rather than as a step-down,
            // following SR-14.7's rule. Normally a step-down would re-offer the
            // same base as a base-only lineage, which the hook body's LEBC check
            // would catch. In the jumped-base shape of SDD §22.20.3 it would reach
            // a different, older base, which the refusal gives up even when it is
            // restorable.
            //
            // The SDD §22.6 step-2 identity verdict precedes the refusal, on the
            // base-sourced identity in `base_meta`. A layer the verdict finds
            // gone (no live mapping resolves to this `object_id`) returns
            // `Ok(None)`, which the hook answers `NotRecoverable` with the
            // `not_recoverable` attempt — the answer its apply `NotFound` arm
            // gives a gone layer. Every other answer is an `Err` that reaches the
            // hook's `Err` arm, which ticks the `error` recovery attempt. No
            // lineage cell is ticked for either, exactly as for a store error.
            // The step-downs this walk already accumulated were emitted above.
            // Boxed for the debug-build stack reason the hook gives at its
            // `apply_restore` call: this future is inlined into the catch-up
            // chain.
            if is_chunked_base(&base_bytes) {
                return match Box::pin(
                    self.adjudicate_over_limit_base_on_access(object_id, head_seq, &base_meta),
                )
                .await
                {
                    CheckpointError::NotFound => Ok(None),
                    err => Err(err),
                };
            }

            // ── Why `spawn_blocking` and not an inline call ──────────────────
            //
            // `compose_from_artifacts` is fully synchronous and CPU-bound with
            // **no yield point**: at stock knobs it replays up to
            // `TAIL_MAX_CHAIN x TAIL_MAX_ENTRIES` = 8 x 4096 = 32,768 records —
            // an OMPF parse, a C++ FlatBuffers verification and an `apply_delta`
            // each — over up to `TAIL_MAX_BYTES x TAIL_MAX_CHAIN` = 512 MiB of
            // artifact bytes. Inline, that blocks a tokio worker for plausibly
            // seconds. That is a magnitude change over HEAD's single-blob
            // `decode_checkpoint_ompf`, not a like-for-like continuation of it,
            // so the inline precedent does not carry. The inputs are `Bytes`
            // handles and a `u64` pair, all trivially cheap to move across the
            // boundary, so the offload costs one task hand-off.
            //
            // Memory, and why the move is the enforcement (amendment 1): the
            // artifact `Bytes` are moved INTO the closure and dropped with its
            // frame, so nothing keeps them alive past composition — the compiler,
            // not a comment, is what stops them living across `write_restored_object`.
            // Under the landed split-cache design the resident-bytes ceiling is
            // ONE lineage (`1 base + K tails`) rather than the whole listing:
            // gather retains link *facts* selection-wide (tens of bytes each) and
            // artifact *bytes* only for the candidate it is currently walking.
            // What does survive here is `ComposedLineage::ext_values`, whose
            // zero-copy `Bytes` views pin the tail buffers they were sliced from;
            // the apply core drops that whole set immediately after the SR-3.4
            // pre-store and before `write_restored_object`, so peak footprint is not
            // doubled.
            let composed = tokio::task::spawn_blocking(move || {
                compose_from_artifacts(object_id, &base_bytes, base_snap_seq, &tail_bytes)
            })
            .await;

            let composed = match composed {
                Ok(result) => result,
                Err(join_err) => {
                    // A panic inside the composition engine is a code defect, not
                    // a lineage verdict: stepping down would quietly serve older
                    // state and hide it. Surface it loudly on the `Error` arm.
                    tracing::error!(
                        object_id,
                        head_seq,
                        chain_len,
                        error = %join_err,
                        "composition task failed to join during restore selection; this is an \
                         internal defect, not a broken lineage"
                    );
                    return Err(CheckpointError::Permanent {
                        source: format!(
                            "composition task for object {object_id} at head {head_seq} failed to \
                             join: {join_err}"
                        )
                        .into(),
                    });
                }
            };

            let composed = match composed {
                Ok(composed) => composed,
                Err(compose_err) => {
                    // SR-14.7, composed arm, plain OMPF base: a base over the
                    // per-delta entry limit that an earlier release wrote as one
                    // OMPF fails the composition's base decode with the parser's
                    // typed over-limit error. SR-14.7 makes it `Permanent`, refused
                    // like a chunked base, with no step-down and no lineage cell.
                    // Normally a step-down would re-offer the same base as a
                    // base-only lineage that then fails at apply's decode. In the
                    // jumped-base shape of SDD §22.20.3 it would reach a
                    // different, older base, which the refusal gives up even when
                    // it is restorable. Read off the error after the join, so
                    // nothing is added to the success path; the identity
                    // verdict's round trip runs only on this refusal path, and
                    // its answers map exactly as at the chunked-base check above.
                    if let ComposeError::BaseDecode { source, .. } = &compose_err {
                        if is_over_entry_limit_base(source) {
                            return match Box::pin(self.adjudicate_over_limit_base_on_access(
                                object_id, head_seq, &base_meta,
                            ))
                            .await
                            {
                                CheckpointError::NotFound => Ok(None),
                                err => Err(err),
                            };
                        }
                    }
                    // Step down. Every `ComposeError` is `Permanent` by
                    // construction, so retrying the same bytes is provably
                    // futile; the next older restorable generation is the only
                    // move. The engine attributes itself to the offending
                    // generation (a tail's `to_seq`, or its index when even the
                    // header would not peek) — carried verbatim here.
                    tracing::warn!(
                        object_id,
                        head_seq,
                        base_snap_seq,
                        chain_len,
                        error = %compose_err,
                        "composed restore rejected a lineage; stepping down to the next older \
                         restorable generation"
                    );
                    selection.record_compose_failure(head_seq, head_kind);
                    self.emit_new_step_downs(&selection, &mut emitted_step_downs);
                    continue;
                }
            };

            // Two independent derivations of S — the gather walk's link-chain
            // terminal and the replay cursor's landing point. Cross-checked so a
            // silent divergence becomes a test failure rather than a restore
            // stamped at the wrong sequence.
            debug_assert_eq!(
                composed.composed_seq, head_seq,
                "the composed cursor and the gathered head sequence are two derivations of S \
                 and must agree"
            );

            // The adjudicated meta: base-sourced identity and header columns,
            // with exactly two non-identity columns taken from the head
            // generation. See this function's `# The composed meta` block for
            // why each is sourced where it is, and
            // [`adjudicate_composed_meta`] for why the rule is shared with the
            // SR-4.1 chain-collapse re-base rather than restated here — including the one
            // column the two callers genuinely derive differently
            // (`observed_commit_span`, which is why it is an argument).
            let meta = adjudicate_composed_meta(base_meta, head_span, head_written_at);

            tracing::debug!(
                object_id,
                head_seq,
                base_snap_seq,
                chain_len,
                ext_values = composed.ext_values.len(),
                observed_commit_span = head_span,
                outcome,
                "composed a restorable lineage"
            );
            self.metrics
                .inc_checkpoint_restore_lineage(outcome, &self.service_label);
            return Ok(Some(SelectedRestore {
                payload: RestorePayload::Composed {
                    state: composed.state,
                    ext_values: composed.ext_values,
                },
                head_seq,
                meta,
                chain_len,
                outcome,
            }));
        }
    }

    /// Emit one `stepped_down_*` increment per [`StepDownRecord`] the selection
    /// has accumulated since the last call, advancing `emitted`.
    ///
    /// The selection's step-down list is append-only, so a high-water cursor is
    /// all the bookkeeping needed. Spellings come from
    /// [`LineageBreak::reason_label`], which delegates to
    /// `CHECKPOINT_RESTORE_LINEAGE_OUTCOMES`; this function must not spell one.
    /// Covers both walk-side breaks and the caller-recorded compose failure,
    /// which is exactly why `record_compose_failure` pushes into the same list.
    ///
    /// [`StepDownRecord`]: crate::checkpoint::lineage::StepDownRecord
    /// [`LineageBreak::reason_label`]: crate::checkpoint::lineage::LineageBreak::reason_label
    fn emit_new_step_downs(&self, selection: &LineageSelection<'_>, emitted: &mut usize) {
        let records = selection.step_downs();
        for record in &records[*emitted..] {
            self.metrics
                .inc_checkpoint_restore_lineage(record.reason.reason_label(), &self.service_label);
        }
        *emitted = records.len();
    }

    /// Test-only post-`Restored` hook (b.sxc). Consults the optional
    /// one-shot latch installed via
    /// [`Self::with_force_post_restore_notfound_for_test`]; if armed,
    /// SYNCHRONOUSLY `UNLINK`s the just-restored `obj:hdr` against the
    /// round-robin multiplexed data pool and `.await`s its completion
    /// before returning, then self-disarms via `swap`. Ordering against
    /// the originating handler's subsequent retry HGETALL is guaranteed
    /// by Valkey's single-threaded command loop, NOT by connection
    /// identity (the apply's `txn_pool` connection is already released
    /// by the time this helper runs).
    ///
    /// The wipe is best-effort by design — `UNLINK` on a missing key is
    /// a no-op, and a failure to `UNLINK` is irrelevant to the test
    /// contract because the retry-once integration tests fail-loud at
    /// the retry-assertion stage, not on the wipe call itself. We
    /// deliberately do not surface errors here: a stray transient
    /// failure on the test seam should not change the
    /// `RecoveryOutcome::Restored` already accounted for in metrics.
    ///
    /// Under `#[cfg(not(any(test, feature = "test-checkpoint-fault-injection")))]`
    /// the latch field, its builder
    /// [`Self::with_force_post_restore_notfound_for_test`], and the
    /// wipe-path branch of this helper are all elided by the compiler
    /// — they are gone from the production object code, not present-but-
    /// `None`. The production helper-call site in
    /// [`Self::try_restore_layer`] resolves to the feature-off no-op
    /// `async fn` defined below, which the optimizer trivially
    /// eliminates. With the feature enabled but the latch never armed
    /// by a test, the cost is one `Option::as_ref` per Restored outcome
    /// — negligible on a path that already does a multi-MB OMPF parse +
    /// Valkey apply round-trips.
    #[cfg(any(test, feature = "test-checkpoint-fault-injection"))]
    async fn maybe_force_post_restore_notfound_for_test(&self, object_id: u64) {
        if let Some(ref latch) = self.force_post_restore_notfound_for_test {
            if latch.swap(false, std::sync::atomic::Ordering::AcqRel) {
                let key = self.layer_store.keys().obj_header(&object_id.to_string());
                let _: Result<(), _> = self.layer_store.client().pool.unlink::<(), _>(key).await;
            }
        }
    }

    /// Production-mode no-op stub mirroring the cfg-gated test seam
    /// above. Defined so the unconditional call sites in
    /// [`Self::try_restore_layer`] compile in non-test builds without
    /// scattering `#[cfg(...)]` blocks across the production hot path
    /// (which the directive explicitly placed inline; we factor to a
    /// helper for clean source). The compiler trivially inlines this to
    /// nothing.
    #[cfg(not(any(test, feature = "test-checkpoint-fault-injection")))]
    #[inline]
    async fn maybe_force_post_restore_notfound_for_test(&self, _object_id: u64) {}

    /// Probe whether `le:{object_id}:obj:hdr` is currently present **and
    /// structurally complete**.
    ///
    /// Returns `Ok(true)` only when the header exists AND carries the fields
    /// its live readers require (`seq`, `created_by`); `Ok(false)` otherwise.
    /// Surfaces transient Valkey errors as [`CheckpointError::Transient`]. Two
    /// callers: the post-acquire re-check in [`Self::try_restore_layer`], and
    /// [`Self::refuse_restore`]'s peer-landed re-check, which uses the same
    /// signal to distinguish "we could not adjudicate the stream" from "a peer
    /// replica already landed this recovery while we were trying to". Both
    /// inherit the structural predicate.
    ///
    /// b.f4t: this used to be a bare `EXISTS`, which reported a
    /// structurally-partial header as "already restored". That is the one
    /// answer neither caller can act on: `read_object_header` /
    /// `stat_layer_for_object_id` map exactly that header to `NotFound`, and
    /// `StatLayer` / `SubscribeReadLayer` route their `NotFound` back into this
    /// hook — so the short-circuit ticked
    /// `checkpoint_recovery_attempts_total{result="restored"}` for a restore
    /// that never ran and handed the handler's one retry the identical error.
    /// Delegating to the shared
    /// [`header_is_structurally_complete`](crate::checkpoint::apply::header_is_structurally_complete)
    /// keeps this predicate and `apply_restore`'s step-1 gate from drifting: the
    /// two are the same question asked a few round trips apart — including its
    /// error classification, so a wrong-typed `obj:hdr` surfaces here as
    /// [`CheckpointError::Permanent`] rather than as a `Transient` this hook
    /// would re-drive forever.
    async fn is_layer_present(&self, object_id: u64) -> Result<bool, CheckpointError> {
        crate::checkpoint::apply::header_is_structurally_complete(
            self.layer_store.client(),
            self.layer_store.keys(),
            object_id,
        )
        .await
    }

    /// Fail-closed high-water probe for the SR-2.5 seq advance (b.8y5,
    /// b.4et).
    ///
    /// Runs `XINFO STREAM le:{object_id}:obj:stream` on the data pool and
    /// returns the sequence component of the reply's `last-generated-id`.
    ///
    /// # Why the stream is a valid source
    ///
    /// [`apply_checkpoint`] only runs when `obj:hdr` is absent, but header
    /// absence does NOT imply the per-object stream is gone: the stream
    /// key carries the same `{object_id}` hash tag and is an independent
    /// key, so a partial state loss (or an `obj:*` wipe that spared the
    /// stream) can leave the stream intact. When it survived, its
    /// high-water ID is the *exact* pre-crash sequence number — strictly
    /// better than the checkpoint's `snap_seq`, which lags by up to one
    /// RPO window.
    ///
    /// It also removes a hard post-restore failure mode **whenever the
    /// probe returns a value**. Both commit paths `XADD` at `{new_seq}-0`,
    /// so a surviving stream whose high-water ID is at or above the seeded
    /// sequence makes *every* subsequent `UpdateLayer` fail with `ERR The
    /// ID specified in XADD is equal or smaller than the target stream top
    /// item` until the writer cursor climbs past it. Folding the stream's
    /// high-water into the restore floor means the seeded sequence starts
    /// above it by at least `LIVE_EDIT_CHECKPOINT_SEQ_SAFETY_MARGIN`.
    ///
    /// When the probe *refuses* a high-water it did read (the plausibility
    /// gate below), seeding from `snap_seq + margin` would sit far *below*
    /// the surviving stream top, so the wedge would be live. That is why the
    /// gate additionally removes the stale stream — **verifiably**, and only
    /// after the identity guard authorizes it: see
    /// [`Self::unlink_stale_stream`].
    ///
    /// # The stream is NOT written exclusively by the commit paths
    ///
    /// The "ID space *is* the sequence space" equivalence above holds for
    /// the two commit paths, which pin an explicit `{new_seq}-0`. It does
    /// **not** hold for the stream as a whole: two tombstone sites append
    /// a `payload=deleted` sentinel with the **auto-ID `*`** rather than an
    /// explicit ID — `LayerStore::delete_layer` and the CreateLayer-
    /// overwrite cascade (`src/storage/layer.rs`). Valkey mints an auto-ID
    /// from the wall clock, so one tombstone permanently moves
    /// `last-generated-id` into unix-millisecond territory (~1.787e12 and
    /// rising). That is below `F64_SAFE_INT_MAX`, so the ceiling check in
    /// [`apply_checkpoint`] would *not* catch it and the layer would be
    /// materialized with a permanently timestamp-inflated `sequence_num`.
    ///
    /// A tombstoned object is normally unreachable here — the SR-2.4
    /// live-mapping guard rejects the apply first — but session-checkpoint
    /// restore (`crate::checkpoint::session::apply_session_checkpoint`)
    /// re-creates `layer_name → object_id` mappings verbatim, so a
    /// pre-delete session checkpoint can re-point a live mapping at a
    /// tombstoned object and defeat that guard. The `*`-sentinel case is
    /// therefore **handled, not merely unreachable**.
    ///
    /// # The plausibility gate
    ///
    /// One gate handles it, and everything else shaped like it: refuse to
    /// fold a high-water more than [`MAX_PLAUSIBLE_HIGH_WATER_GAP`] above
    /// `snap_seq`.
    ///
    /// An earlier revision paired this with a dedicated tombstone gate
    /// that asked the script whether the newest entry carried the `deleted`
    /// sentinel. That gate was **provably subsumed** and has been dropped:
    /// an auto-ID is a unix-millisecond value (measured 1.787e12 on this
    /// Valkey, and monotonically rising), while this gate rejects anything
    /// above `snap_seq + 1e9` — three orders of magnitude below. Every
    /// tombstone the dedicated gate could catch, this one catches, and both
    /// took identical action. Dropping it also let the script move from
    /// `XINFO STREAM` to `XINFO STREAM FULL COUNT 1`, halving the
    /// server-side entry materialization (see the script header).
    ///
    /// Being cause-agnostic is the point: the gate equally catches a
    /// trimmed-away tombstone, and any future writer that puts a
    /// non-sequence-shaped ID on this stream, rather than silently seeding
    /// from it.
    ///
    /// # Why `XINFO STREAM`, and why server-side
    ///
    /// `last-generated-id` is precisely the value the server compares an
    /// incoming `XADD` ID against, so it is the exact bound for the
    /// collision above. The `XREVRANGE` tail ID is only a *lower* bound on
    /// it: deleting the top entry lowers what `XREVRANGE` reports while
    /// `last-generated-id` holds (verified against Valkey — after
    /// `XDEL <top>`, `XREVRANGE` drops to the previous entry's ID while
    /// `last-generated-id` stays put, and the next `XADD` still rejects
    /// anything at or below the deleted ID). Using the tail ID would
    /// therefore under-report the floor in exactly the case that produces
    /// the wedged-writer failure.
    ///
    /// The `XINFO` call runs inside [`STREAM_HIGH_WATER_PROBE_SCRIPT`]
    /// rather than from the client, because no `XINFO STREAM` form is
    /// metadata-only — every form embeds at least one entry, and
    /// layer-stream entries carry the writer's raw OMPF bounded only by
    /// `LIVE_EDIT_MAX_LAYER_SIZE`. Read client-side, this probe could pull
    /// ~100 MB to extract a ~15-byte ID, inline on an RPC-triggered
    /// recovery under the K=1 admission mutex — blowing
    /// `LIVE_EDIT_VALKEY_COMMAND_TIMEOUT_MS` and folding to `None` on
    /// exactly the large layers where the wedge matters most. The script
    /// projects the one scalar server-side, so the reply is O(1) in layer
    /// size. The server-side materialization cost is *not* O(1) — see the
    /// script header for the measured numbers and why it is acceptable
    /// here and nowhere else.
    ///
    /// # Fail-closed contract (b.4et)
    ///
    /// Pre-b.4et the probe never returned an error: a missing key, a
    /// wrong-type key, an unexpected reply shape, an unparseable ID and any
    /// Valkey error ALL folded to `None`. That conflation is the defect.
    /// `None` seeds the header at `snap_seq + band`; if the true stream top
    /// was above that, every subsequent `UpdateLayer` is refused by the b.ph9
    /// stream-id collision guard and the restored layer is a write outage
    /// until an operator clears the key by hand. "We could not read the
    /// evidence" was being spent as "there is no evidence".
    ///
    /// The script's `nil` is unchanged (it cannot distinguish the cases —
    /// see its header); the conflation is resolved Rust-side, with one
    /// `TYPE le:{oid}:obj:stream` round-trip whenever the probe does not
    /// hand back a plausible sequence:
    ///
    /// | probe outcome | `TYPE` says | action on the key | action on the restore |
    /// |---|---|---|---|
    /// | plausible ID | *(not asked)* | none | proceed, floor = `max(snap_seq, high-water)` |
    /// | implausible ID | *(not asked)* | verified `UNLINK` — gated on the SR-2.4 identity guard AND on `obj:hdr` still being absent (b.h4m); left alone when either says no | proceed, floor = `snap_seq` |
    /// | `nil` / unparseable / error | `none` | none — there is no key | proceed, floor = `snap_seq` |
    /// | `nil` / unparseable / error | `stream` | none | **REFUSE** (`Transient`) |
    /// | `nil` / unparseable / error | anything else | none — not our value | proceed, floor = `snap_seq` |
    ///
    /// The third row is what keeps DR available on the shape that actually
    /// dominates production: a full keyspace loss, where the stream is gone
    /// with everything else. It is byte-identical to pre-b.4et behaviour,
    /// including when the `EVALSHA` itself failed — a broken script cache
    /// must not cost us a restore we can prove is safe.
    ///
    /// The fourth row is the fix. The stream is *there* and we cannot say
    /// where its top is, so we do not seed a header underneath it.
    ///
    /// The fifth row (`wrongtype_key`) proceeds deliberately: a foreign value
    /// on the stream key carries no surviving high-water, so the guard band
    /// covers the RPO window exactly as it does on a full wipe. It is NOT
    /// unlinked — this service never retypes a key it owns, so the value
    /// belongs to something else and deleting it is not ours to do. See
    /// [`Self::classify_unreadable_stream`] for the accompanying WARN.
    ///
    /// Every probe / `TYPE` / identity-gate / peer-presence / `UNLINK`
    /// round-trip is retried per [`RECOVERY_STREAM_PROBE_RETRY`] before its arm
    /// is decided, so a reconnect-shaped blip does not cost a restore. That
    /// includes the b.h4m peer-presence re-read: retrying it is window-neutral
    /// (the window it narrows runs from the last SUCCESSFUL read to the
    /// `UNLINK`), while leaving it un-retried would have made a single blip
    /// refuse the restore. The one single-shot read on this path is the
    /// peer-landed re-check inside [`Self::refuse_restore`], which runs after
    /// the verdict is already a refusal. The *whole* sequence —
    /// probe, classification, identity gate, peer-presence re-read, verified
    /// `UNLINK`, and the peer-landed re-check — is bounded by one
    /// [`RECOVERY_STREAM_ADJUDICATION_BUDGET`] deadline; on expiry it takes the
    /// fourth row's REFUSE arm. That deadline, not the retry count, is what
    /// bounds how long this holds the K=1 admission mutex, because a *hung*
    /// round-trip costs `LIVE_EDIT_VALKEY_COMMAND_TIMEOUT_MS`, not a backoff
    /// sleep. See the const for the derivation.
    ///
    /// # A `Permanent` a *successful* probe can still cause
    ///
    /// Unchanged from b.8y5, and orthogonal to the above:
    /// a folded high-water raises `restore_floor`, and
    /// [`apply_checkpoint`]'s `seeded_seq >= F64_SAFE_INT_MAX` check
    /// rejects as `Permanent`. So a successful probe can in principle turn
    /// an otherwise-recoverable layer into a `Permanent` failure. The
    /// plausibility bound confines that: `restore_floor <= snap_seq +
    /// MAX_PLAUSIBLE_HIGH_WATER_GAP`, so the probe can advance the floor by
    /// at most that much, and only a layer already within
    /// `MAX_PLAUSIBLE_HIGH_WATER_GAP + safety_margin` of `2^53 − 1` can be
    /// pushed over. Such a layer is not *immediately* doomed — up to a
    /// billion commits of headroom may remain — but it is within one
    /// probe-width of a ceiling it is going to reach on its current
    /// trajectory, and the `Permanent` is a loud, diagnostic surfacing of
    /// that rather than a silent materialization of state whose writable
    /// lifetime is nearly over. Without the bound, a timestamp-shaped ID
    /// could vault a layer with 2^53 worth of headroom over the ceiling
    /// outright.
    async fn probe_stream_high_water(
        &self,
        object_id: u64,
        snap_seq: u64,
        meta: &CheckpointMeta,
    ) -> Result<Option<u64>, CheckpointError> {
        // ONE deadline over the whole adjudication. Every arm below is itself
        // retried, but a retry budget bounds *sleep*, not *round-trips*: a
        // Valkey that accepts the command and never answers costs a full
        // `LIVE_EDIT_VALKEY_COMMAND_TIMEOUT_MS` per attempt, so the retry
        // counts alone admit a worst case tens of seconds long — held under
        // the K=1 per-`object_id` admission mutex, on an RPC
        // (`SubscribeReadLayer`) that is deadline-exempt, with every other
        // reader of the layer queued behind it. The timeout is what makes the
        // latency claim true by construction.
        //
        // Cancel-safety: the future below writes nothing that a cancellation
        // could half-finish. Its only metric ticks are strictly-synchronous
        // tails after their last `.await` (the `stream_refused` tick follows a
        // completed `UNLINK`; the `stream_unresolved` tick is inside
        // `refuse_restore`, after its only `.await`), so a cancelled attempt
        // ticks nothing and the expiry arm below owns the single tick for this
        // adjudication. THREE non-cancelled exits tick nothing either, and all
        // three are the same deliberate exception rather than a gap — a
        // `Permanent` from a restore-path key read propagating verbatim, whose
        // loud `error!` is emitted by `classify_object_key_read_error` at the
        // read that raised it: `refuse_restore`'s own peer-landed re-check and
        // `remove_stale_stream`'s b.h4m peer-presence re-read ahead of the
        // `UNLINK` (both structural `HGETALL obj:hdr`), plus
        // `remove_stale_stream`'s SR-2.4 identity gate (`GET` of the layer
        // mapping). See those functions.
        // A cancelled in-flight `UNLINK` may or may not have
        // landed server-side, which is inert: it is idempotent, and we refuse
        // the restore either way rather than seeding on the assumption.
        match tokio::time::timeout(
            RECOVERY_STREAM_ADJUDICATION_BUDGET,
            self.adjudicate_stream_high_water(object_id, snap_seq, meta),
        )
        .await
        {
            Ok(verdict) => verdict,
            Err(_elapsed) => self.refuse_over_budget(object_id).await,
        }
    }

    /// The adjudication proper, run under
    /// [`Self::probe_stream_high_water`]'s deadline. See that method's rustdoc
    /// for the arm table; this is only the un-timed body, split out so the
    /// timeout wraps every arm including the b.h4m peer-presence re-read, the
    /// `UNLINK`, and the peer-landed re-check.
    async fn adjudicate_stream_high_water(
        &self,
        object_id: u64,
        snap_seq: u64,
        meta: &CheckpointMeta,
    ) -> Result<Option<u64>, CheckpointError> {
        let stream_key = self.layer_store.keys().layer_stream(object_id);

        // One bounded retry budget around the EVALSHA. `Aborted` (a transient
        // fred kind, per `is_transient_fred_error`) is the in-band retry
        // signal; anything else terminates immediately and lands on the
        // `Err` arm below, where the `TYPE` round-trip decides what it means.
        let probed = retry_aborted_with_jitter(
            RECOVERY_SEAM_STREAM_HIGH_WATER_PROBE,
            RECOVERY_STREAM_PROBE_RETRY,
            |_attempt| self.probe_high_water_once(&stream_key),
        )
        .await;

        // Why every non-plausible outcome funnels into ONE string: the arm
        // that consumes it (`classify_unreadable_stream`) treats them
        // identically — what decides the verdict is the *key's* state, not
        // which flavour of unreadable we hit. The cause rides along so the
        // WARN names it.
        let unreadable_cause = match probed {
            Ok(Some(raw_id)) => match parse_stream_id_seq(&raw_id) {
                Some(seq) => {
                    // Plausibility gate. Cause-agnostic by design: the two
                    // populations it separates are a genuine sequence number
                    // (bounded by the commit volume one object can accumulate
                    // between checkpoints) and anything else — overwhelmingly
                    // an auto-ID wall-clock timestamp from a `deleted`
                    // tombstone, whether or not that tombstone is still
                    // present in the stream.
                    let ceiling = snap_seq.saturating_add(MAX_PLAUSIBLE_HIGH_WATER_GAP);
                    if seq <= ceiling {
                        tracing::debug!(
                            object_id,
                            snap_seq,
                            last_generated_id = %raw_id,
                            high_water = seq,
                            "recovery: per-object stream survived state loss; \
                             folding its last-generated-id into the restore floor"
                        );
                        return Ok(Some(seq));
                    }
                    tracing::warn!(
                        object_id,
                        snap_seq,
                        last_generated_id = %raw_id,
                        high_water = seq,
                        plausibility_ceiling = ceiling,
                        max_plausible_gap = MAX_PLAUSIBLE_HIGH_WATER_GAP,
                        "recovery: stream high-water is implausibly far above the checkpoint's \
                         snap_seq — either the stream was tombstoned (the `deleted` sentinel is \
                         written with an auto-ID, so the stream top is a wall-clock timestamp \
                         rather than a sequence number) or some non-sequence writer set a \
                         genuinely implausible ID. Refusing to seed from it: the restore would \
                         proceed from snap_seq + safety margin, which sits BELOW this stream top, \
                         so every subsequent UpdateLayer would be rejected by the stream-id \
                         collision guard. Removing the stale stream so the restored layer's first \
                         write recreates it cleanly"
                    );
                    return self.remove_stale_stream(object_id, &stream_key, meta).await;
                }
                None => format!("unparseable last-generated-id `{raw_id}`"),
            },
            Ok(None) => {
                "the probe script returned nil (no key, a non-stream key, or a reply carrying no \
                 last-generated-id)"
                    .to_string()
            }
            Err(e) => format!("EVALSHA stream_high_water_probe failed: {e}"),
        };

        self.classify_unreadable_stream(object_id, &stream_key, unreadable_cause)
            .await
    }

    /// One high-water probe attempt: the `EVALSHA`, classified for the retry
    /// driver.
    ///
    /// Split out so [`retry_aborted_with_jitter`] can re-run exactly this and
    /// nothing else. [`map_snapshot_load_err`] is the classifier the whole
    /// crate uses for a fred error inside a retry loop — transient kinds
    /// become [`ServiceError::Aborted`] (retriable), everything else
    /// `Internal` (terminal).
    async fn probe_high_water_once(
        &self,
        stream_key: &str,
    ) -> Result<Option<String>, ServiceError> {
        #[cfg(feature = "test-checkpoint-fault-injection")]
        self.inject_recovery_seam(RECOVERY_SEAM_STREAM_HIGH_WATER_PROBE)?;

        // Single-key EVALSHA. The script returns `nil` for a missing key, a
        // wrong-type key, or a reply without `last-generated-id`; otherwise
        // the ID string. `evalsha_with_reload` handles `NOSCRIPT`
        // transparently.
        STREAM_HIGH_WATER_PROBE_SCRIPT
            .evalsha_with_reload(
                self.layer_store.client().pool.next(),
                vec![stream_key.to_string()],
                Vec::<String>::new(),
            )
            .await
            .map_err(|e| {
                map_snapshot_load_err(
                    FredErrOrigin::client(self.layer_store.client(), POOL_LABEL_DATA),
                    "EVALSHA stream_high_water_probe",
                    e,
                )
            })
    }

    /// Decide what a probe that could not produce a plausible sequence means,
    /// by asking the key itself (b.4et).
    ///
    /// This is where the script's single `nil` is un-conflated. The script
    /// cannot separate "no key" from "wrong-type key" from "unreadable reply"
    /// (see its header); `TYPE` can, and it is a single O(1) round-trip on the
    /// same slot.
    ///
    /// See [`Self::probe_stream_high_water`]'s arm table for the full mapping.
    async fn classify_unreadable_stream(
        &self,
        object_id: u64,
        stream_key: &str,
        cause: String,
    ) -> Result<Option<u64>, CheckpointError> {
        match self.read_stream_key_type(stream_key).await {
            Ok(StreamKeyType::Absent) => {
                // The common case, and the one a full keyspace loss produces:
                // the stream did not survive. Debug, not warn — this is the
                // expected outcome, and it is exactly the pre-b.4et
                // behaviour.
                tracing::debug!(
                    object_id,
                    cause = %cause,
                    "recovery: no per-object stream survived the state loss (TYPE reports the key \
                     is absent); falling back to snap_seq as the restore floor"
                );
                Ok(None)
            }
            Ok(StreamKeyType::Foreign(actual)) => {
                // Not our data. `KEYSPACE_TYPE_ERROR_REMEDIATION`'s policy
                // (src/storage/layer.rs) is that nothing in this service
                // retypes a key it owns, so a foreign value is an out-of-band
                // writer's — we neither seed from it nor delete it.
                //
                // The restore still proceeds and that is correct: a non-stream
                // key carries no surviving high-water, so there is nothing for
                // the floor to fold in and the guard band covers the RPO
                // window exactly as it does on a full wipe. What the operator
                // needs to know is that the layer comes back READABLE but not
                // WRITABLE — the b.459 wrong-type refusal fires on the first
                // commit — hence one loud WARN naming the key and its type.
                //
                // The `remediation` string below intentionally mirrors
                // `KEYSPACE_TYPE_ERROR_REMEDIATION` (`src/storage/layer.rs`),
                // which is private to that module; keep the two in sync by hand
                // until b.bm5 (d) makes that const `pub(crate)` and this can
                // name it directly.
                tracing::warn!(
                    object_id,
                    key = %stream_key,
                    actual_type = %actual,
                    expected_type = "stream",
                    cause = %cause,
                    remediation = "writes to this layer will fail closed until the named key is \
                                   removed; nothing in this service retypes a key it owns, so \
                                   investigate the out-of-band writer that placed a foreign value \
                                   on the le: keyspace before deleting it",
                    "recovery: the per-object stream key holds a foreign, non-stream value. The \
                     restore proceeds (a non-stream key carries no surviving high-water, so the \
                     restore floor is snap_seq and the guard band covers the RPO window as it \
                     does on a full keyspace loss), but every subsequent UpdateLayer on this \
                     layer will be refused by the wrong-type guard until the key is removed"
                );
                self.metrics.inc_checkpoint_recovery_stream_unresolved(
                    "wrongtype_key",
                    &self.service_label,
                );
                Ok(None)
            }
            Ok(StreamKeyType::Stream) => {
                self.refuse_restore(
                    object_id,
                    "probe_unreadable",
                    format!(
                        "the per-object stream key exists and is a stream, but its \
                         last-generated-id could not be read: {cause}"
                    ),
                )
                .await
            }
            Err(type_err) => {
                // We could not even learn what the key is. Same verdict as an
                // unreadable stream: refuse. Folding this to `Ok(None)` would
                // reintroduce the exact b.4et conflation one round-trip later.
                self.refuse_restore(
                    object_id,
                    "probe_unreadable",
                    format!(
                        "the per-object stream key could not be classified: {cause}; \
                         TYPE also failed: {type_err}"
                    ),
                )
                .await
            }
        }
    }

    /// Refuse the restore, unless a peer replica already landed it.
    ///
    /// # The header re-check is not an optimization
    ///
    /// The probe runs BEFORE [`apply_restore`]'s own structural-header
    /// short-circuit, so on this path we have not yet consulted the one
    /// condition that makes the whole restore moot. A peer replica racing the
    /// same recovery may have landed it while we were spending our attempts on
    /// a stream we could not read — and its restore did whatever needed doing
    /// to that stream. Refusing then would turn a *completed* recovery into an
    /// `UNAVAILABLE` for the client.
    ///
    /// This re-check is itself inside
    /// [`RECOVERY_STREAM_ADJUDICATION_BUDGET`] on every arm that reaches here
    /// normally, and under [`RECOVERY_STREAM_ADJUDICATION_PEER_GRACE`] on the
    /// budget-expiry arm ([`Self::refuse_over_budget`]), so it can never be the
    /// thing that lengthens the mutex hold.
    ///
    /// `Ok(None)` on that arm reaches the same place `Restored` would, by the
    /// existing route: the caller proceeds to `apply_restore`, whose step-1
    /// structural-header check returns [`ApplyOutcome::AlreadyPresent`], which
    /// [`Self::try_restore_layer`] already maps to
    /// [`RecoveryOutcome::Restored`]. Nothing is written on that path, so the
    /// `None` floor is never spent. This keeps [`RecoveryOutcome`]'s three
    /// variants and this method's return shape untouched.
    ///
    /// # The refusal is not always `Transient`
    ///
    /// The unresolved-stream refusal this function exists to raise is
    /// [`CheckpointError::Transient`], and a propagated
    /// [`CheckpointError::Permanent`] is the **only** other outcome. The
    /// peer-landed re-check above the refusal can itself fail; when it fails
    /// `Permanent` — the wrong-typed `obj:hdr` class, which the read has
    /// already logged loudly on its own path — that error propagates
    /// **verbatim** rather than being re-reported as the stream refusal, and
    /// nothing here ticks or WARNs. A *transient* header-read failure is not
    /// propagated: it folds into the ordinary refusal below (counter tick,
    /// WARN, `Transient`), because it is the same "could not establish the
    /// stream's state" verdict and its own read reports nothing. See the
    /// arm-by-arm comment inside.
    async fn refuse_restore(
        &self,
        object_id: u64,
        reason: &'static str,
        detail: String,
    ) -> Result<Option<u64>, CheckpointError> {
        // The arms are spelled out rather than collapsed with `matches!`, so the
        // classification here is safe BY CONSTRUCTION — but only ONE error class
        // propagates. The re-check's `HGETALL obj:hdr` can raise `Permanent`
        // (the wrong-typed `obj:hdr` class — `WRONGTYPE` on the `HGETALL`), and
        // folding that into "not present" would re-report an operator-actionable
        // permanent fault as this function's `Transient`, which the caller
        // retries forever. It propagates verbatim instead.
        //
        // Every OTHER `Err` — a plain Valkey blip, a command timeout, a dropped
        // connection — falls through to the refusal below and is reported as
        // this function's own unresolved-stream verdict. That is deliberate and
        // it is the reason this is not a blanket `Err(e) => return Err(e)`:
        // `classify_object_key_read_error` (`apply.rs`) emits its loud
        // operator-facing `error!` ONLY on the `WRONGTYPE` branch and returns
        // the non-wrongtype case as a `Transient` with NO log of its own. So
        // propagating every `Err` would take a transient header-read failure out
        // of this path silently — no `checkpoint_recovery_stream_unresolved`
        // tick, no WARN — for an error that is exactly the "we could not
        // establish the stream's state" class the refusal exists to report.
        //
        // Ordering makes the `Permanent` arm hard to reach rather than
        // impossible: every caller runs downstream of
        // `probe_stream_high_water`, which runs downstream of the post-acquire
        // `is_layer_present` re-check whose `Err` arm already propagates as
        // `RecoveryOutcome::Error`, so a wrong-typed header is normally refused
        // before anything reaches here. The explicit arm is what keeps that a
        // defence in depth rather than the only defence — and what keeps the
        // classification correct if the stream adjudication is ever moved ahead
        // of that re-check.
        //
        // Note what the `Permanent` arm deliberately does NOT do: it ticks no
        // `checkpoint_recovery_stream_unresolved` cell and emits no WARN. The
        // verdict it carries is the header fault, not an unresolved stream, and
        // the read that produced it has already logged its own `error!` on that
        // branch. It is also the one non-cancelled exit from this function that
        // ticks nothing — see the exception noted at `refuse_over_budget`'s
        // grace-expiry arm and at `probe_stream_high_water`'s cancel-safety
        // note.
        match self.is_layer_present(object_id).await {
            Ok(true) => {
                tracing::info!(
                    object_id,
                    reason,
                    detail = %detail,
                    "recovery: could not establish the per-object stream's state, but obj:hdr is \
                     now present and structurally complete — a peer replica landed this recovery \
                     while we were probing. Proceeding to the apply, which short-circuits on the \
                     same structural check and reports the layer restored"
                );
                return Ok(None);
            }
            Ok(false) => {}
            Err(e @ CheckpointError::Permanent { .. }) => return Err(e),
            Err(_) => {}
        }

        self.metrics
            .inc_checkpoint_recovery_stream_unresolved(reason, &self.service_label);
        tracing::warn!(
            object_id,
            reason,
            attempts = RECOVERY_STREAM_PROBE_RETRY.max_attempts,
            detail = %detail,
            "recovery: REFUSING to restore this layer. Seeding obj:hdr underneath a per-object \
             stream whose top we cannot prove we cleared would silently re-issue sequence numbers \
             pre-restore subscribers already hold, and post-b.ph9 would wedge every write to the \
             layer. Failing loudly instead: the caller sees UNAVAILABLE and may retry. Remediation \
             — check Valkey health; if it persists, inspect le:{object_id}:obj:stream by hand"
        );
        Err(CheckpointError::Transient {
            source: format!(
                "recovery refused for object {object_id} ({reason}, \
                 {} attempts): {detail}",
                RECOVERY_STREAM_PROBE_RETRY.max_attempts
            )
            .into(),
        })
    }

    /// The [`RECOVERY_STREAM_ADJUDICATION_BUDGET`] expiry arm: refuse, on the
    /// same fail-closed terms as every other unresolved-stream verdict.
    ///
    /// # Why `probe_unreadable` and not a fourth reason
    ///
    /// The unresolved-stream vocabulary is pinned at three cells
    /// (`probe_unreadable`, `unlink_unverified`, `wrongtype_key`) and this is
    /// the first: "the evidence could not be obtained". Running out of budget
    /// is *how* it could not be obtained, not a different thing that happened —
    /// and the operator response is identical (check Valkey health, then
    /// inspect the key). The WARN carries the budget so a dashboard spike is
    /// still attributable.
    ///
    /// # The peer-landed re-check, under a grace
    ///
    /// We are already over budget, so [`Self::refuse_restore`]'s structural
    /// `HGETALL obj:hdr` re-check — which can itself block for a full command
    /// timeout —
    /// gets [`RECOVERY_STREAM_ADJUDICATION_PEER_GRACE`] and no more. A healthy
    /// Valkey answers that one-key `HGETALL` in well under a millisecond; a
    /// Valkey that
    /// does not answer within the grace is exactly the one whose budget just
    /// expired, and waiting on it a second time only lengthens the mutex hold.
    /// So the hard ceiling on this whole adjudication is `budget + grace`.
    ///
    /// If the grace expires we refuse *without* the re-check, which is safe in
    /// both directions: refusing a restore a peer already landed costs one
    /// `UNAVAILABLE` that the handlers' retry-once recovery dispatch re-runs,
    /// and that re-run hits `apply_restore`'s structural-header short-circuit
    /// and
    /// reports the layer restored. It cannot cost correctness — nothing is
    /// written on any arm of this path.
    async fn refuse_over_budget(&self, object_id: u64) -> Result<Option<u64>, CheckpointError> {
        let detail = format!(
            "the per-object stream adjudication (high-water probe, TYPE classification, \
             identity-gated UNLINK) did not reach a verdict within its overall \
             {budget_ms} ms budget",
            budget_ms = RECOVERY_STREAM_ADJUDICATION_BUDGET.as_millis()
        );

        match tokio::time::timeout(
            RECOVERY_STREAM_ADJUDICATION_PEER_GRACE,
            self.refuse_restore(object_id, "probe_unreadable", detail.clone()),
        )
        .await
        {
            Ok(verdict) => verdict,
            Err(_elapsed) => {
                // `refuse_restore` has exactly one `.await` (the structural
                // `HGETALL obj:hdr` re-check), and
                // its counter tick and WARN both sit after it, so a
                // cancellation here provably ticked nothing. This arm owns the
                // single tick for this adjudication — with one exception, on a
                // path this arm cannot reach: `refuse_restore` returning early
                // with a propagated `Permanent` from that re-check ticks
                // nothing, by design (the read logs its own `error!`). That is a
                // completion, not a cancellation, so it is not this arm's case.
                self.metrics.inc_checkpoint_recovery_stream_unresolved(
                    "probe_unreadable",
                    &self.service_label,
                );
                tracing::warn!(
                    object_id,
                    reason = "probe_unreadable",
                    budget_ms = RECOVERY_STREAM_ADJUDICATION_BUDGET.as_millis(),
                    peer_grace_ms = RECOVERY_STREAM_ADJUDICATION_PEER_GRACE.as_millis(),
                    detail = %detail,
                    "recovery: REFUSING to restore this layer — the stream adjudication blew its \
                     overall budget AND the obj:hdr re-check that would have detected a peer \
                     replica landing this recovery did not answer within the grace either. \
                     Seeding obj:hdr underneath a per-object stream whose top we cannot prove we \
                     cleared would silently re-issue sequence numbers pre-restore subscribers \
                     already hold, and post-b.ph9 would wedge every write to the layer. Failing \
                     loudly instead: the caller sees UNAVAILABLE and its retry-once recovery \
                     dispatch covers the case where a peer had in fact already landed it. \
                     Remediation — Valkey is not answering; check its health, then inspect \
                     le:{object_id}:obj:stream by hand"
                );
                Err(CheckpointError::Transient {
                    source: format!(
                        "recovery refused for object {object_id} (probe_unreadable): {detail}; \
                         the obj:hdr peer-landed re-check also exceeded its \
                         {grace_ms} ms grace",
                        grace_ms = RECOVERY_STREAM_ADJUDICATION_PEER_GRACE.as_millis()
                    )
                    .into(),
                })
            }
        }
    }

    /// `TYPE` the per-object stream key, under the same bounded retry as the
    /// probe.
    async fn read_stream_key_type(
        &self,
        stream_key: &str,
    ) -> Result<StreamKeyType, CheckpointError> {
        retry_aborted_with_jitter(
            RECOVERY_SEAM_STREAM_KEY_TYPE,
            RECOVERY_STREAM_PROBE_RETRY,
            |_attempt| self.read_stream_key_type_once(stream_key),
        )
        .await
        .map_err(service_err_to_checkpoint_err)
    }

    /// One `TYPE` attempt.
    ///
    /// Deliberately NOT `storage::lua::KEYSPACE_TYPE_PROBE_SCRIPT`: that
    /// script answers "is this key the type I am about to write", which folds
    /// `none` and the expected type into one "fine" verdict. Here `none` and
    /// `stream` are the two arms that must NOT be folded — one proceeds, the
    /// other refuses.
    ///
    /// This is the *single* `TYPE` helper on both paths that need one — the
    /// classification in [`Self::classify_unreadable_stream`] and the
    /// post-`UNLINK` confirmation in [`Self::unlink_stale_stream_once`] — so
    /// the [`RECOVERY_SEAM_STREAM_KEY_TYPE`] seam below makes both
    /// deterministic for tests. Which refusal reason an injection produces is
    /// decided by which path is live; see that const's rustdoc.
    async fn read_stream_key_type_once(
        &self,
        stream_key: &str,
    ) -> Result<StreamKeyType, ServiceError> {
        #[cfg(feature = "test-checkpoint-fault-injection")]
        self.inject_recovery_seam(RECOVERY_SEAM_STREAM_KEY_TYPE)?;

        let ty: String = self
            .layer_store
            .client()
            .pool
            .r#type(stream_key)
            .await
            .map_err(|e| {
                map_snapshot_load_err(
                    FredErrOrigin::client(self.layer_store.client(), POOL_LABEL_DATA),
                    "TYPE obj:stream",
                    e,
                )
            })?;
        Ok(match ty.as_str() {
            "none" => StreamKeyType::Absent,
            "stream" => StreamKeyType::Stream,
            other => StreamKeyType::Foreign(other.to_string()),
        })
    }

    /// The plausibility-refusal branch: authorize, then verifiably remove the
    /// stale stream, then let the restore run.
    ///
    /// # Why the identity guard runs HERE and not only inside `apply_restore`
    ///
    /// b.4et item 6. `apply_restore`'s SR-2.4 guard is what decides whether
    /// this restore may happen at all, but it runs AFTER this point — so
    /// pre-b.4et the `UNLINK` fired first and destroyed the stream of a layer
    /// the guard was about to refuse. That is not hypothetical: "header
    /// absent, stream present" is the *normal* state for up to
    /// `STREAM_DELETE_GRACE_MS` (30 s) on every `DeleteLayer`, whose
    /// retirement order is DEL mapping → XADD `deleted` → UNLINK header →
    /// PEXPIRE stream (`src/storage/layer.rs`). A recovery racing a delete
    /// would therefore blow away the graced stream — the very thing that lets
    /// a parked subscriber see the `deleted` sentinel and exit cleanly —
    /// for a restore that then returns `NotFound` anyway.
    ///
    /// So: no identity, no delete. The restore still proceeds to
    /// `apply_restore`, which reaches the existing `NotFound` /
    /// `NotRecoverable` verdict on its own. `max_seq_observed` is `None`
    /// there, and irrelevant: nothing is written.
    ///
    /// # Ordering — a peer's rebuild can be admitted before the UNLINK (b.h4m)
    ///
    /// This runs at step 6; `apply_restore`'s single `write_restored_object`
    /// MULTI/EXEC is step 7. The pre-b.h4m argument here was that the step-6 →
    /// step-7 window is unexploitable, because every writer of this stream
    /// requires an `obj:hdr` that does not yet exist — the Lua fast path aborts
    /// on an empty `HGETALL` of `obj:hdr` before its `XADD`
    /// (`src/storage/scripts/update_layer_fast.lua`), the WATCH/MULTI/EXEC slow
    /// path bails on the absent header before it opens `MULTI`
    /// (`src/storage/layer.rs`), and `create_layer` only `XADD`s `1-0` on a
    /// freshly-minted `object_id` inside its own MULTI.
    ///
    /// **That reasoning is single-replica, and the conclusion it draws is
    /// false.** Layer restore is lazy and per-layer, and the admission mutex
    /// the caller holds is per-process: a PEER replica can land the entire
    /// rebuild — header included — while this replica sits in step 5's
    /// multi-second checkpoint-store blob fetch. From that instant writers are
    /// admitted against the peer's header, a client's `UpdateLayer` `XADD`s
    /// onto the winner's **live** stream, and every gate on this path still
    /// passes: the plausibility gate reads that fresh top as stale (it is far
    /// above a `snap_seq` the peer's generation has since superseded), and the
    /// identity gate passes because the live mapping resolves to the very same
    /// `object_id`. The `UNLINK` then destroys a live stream, and step 7
    /// reports `Restored` off `apply_restore`'s `AlreadyPresent`
    /// short-circuit — no error, no counter, nothing written.
    ///
    /// # What the re-read closes, and what it does not
    ///
    /// The structural `HGETALL obj:hdr` below re-asks step 4's presence
    /// question immediately before the `UNLINK` and skips the removal when a
    /// peer's complete header is there. That narrows the exposure from "the
    /// whole blob fetch" to **one round trip in the common case** — the window
    /// between the re-read returning "absent" and the `UNLINK` landing — and to
    /// **at most the remaining adjudication budget** when the `UNLINK` itself
    /// has to retry, since [`Self::unlink_stale_stream`] re-drives its attempt
    /// up to [`RECOVERY_STREAM_PROBE_RETRY`] times and the window is measured
    /// to whichever attempt lands. The re-read's own retries do not widen it:
    /// the window starts at the LAST SUCCESSFUL read, so a retried attempt only
    /// moves that start later. It does not close it either: a peer header
    /// landing inside the window still has its live stream
    /// removed, because nothing here fences the `UNLINK` against the header
    /// key. Closing it means holding a `WATCH` on `obj:hdr` across the `UNLINK`
    /// (or folding the removal into `write_restored_object`'s MULTI, which is
    /// already fenced that way); a filed follow-up owns that work.
    ///
    /// The residual damage, stated precisely, is confined to **live
    /// subscribers of the peer-restored layer**, and at worst it is one resync
    /// per subscriber, never a silent skip:
    ///
    /// - Nothing errors until the next commit. `XREAD BLOCK` against an absent
    ///   key is legal and waits, so no stream reader failure stashes
    ///   `Unavailable`, no `Deleted` notification is produced, and
    ///   `src/grpc/subscribe.rs` neither re-bootstraps onto a new `object_id`
    ///   nor ends the RPC `NOT_FOUND`.
    /// - The next commit's `XADD` recreates the key at `{seq}-0`, one above
    ///   the removed top. A subscription already at that top (or whose
    ///   caught-up ack covers it) receives that commit and everything after it
    ///   and continues.
    /// - A subscription behind the removed top ends `stream_gap` when the
    ///   recreated stream's first entry reaches it: the stream multiplexer's
    ///   head check finds nothing below that entry, so the commits between the
    ///   subscription's position and it are no longer in the stream (SDD
    ///   §12.3). The client reconnects, and its catch-up reconstructs from
    ///   materialized `obj:*` state, which this path never touches — so the
    ///   missing commits' content reaches it that way instead of being
    ///   skipped on that subscription.
    ///
    /// That bounded outcome — a reconnect with a full catch-up for the
    /// subscribers behind the removed top, and no lost content — is why the
    /// remaining one-round-trip window is left unfenced here rather than
    /// fenced in this change; the follow-up above still owns closing it.
    ///
    /// A brand-new subscriber arriving after the removal is unaffected end to
    /// end: it pins its cursor from the header, its registration reads the
    /// absent key like any other, and its catch-up snapshot comes from that
    /// same state.
    ///
    /// The one residual writer under the ORIGINAL argument — a concurrent
    /// `DeleteLayer` appending its tombstone — is unchanged, and is precisely
    /// what the identity gate above turns into a `NotFound` rather than a
    /// wedge.
    async fn remove_stale_stream(
        &self,
        object_id: u64,
        stream_key: &str,
        meta: &CheckpointMeta,
    ) -> Result<Option<u64>, CheckpointError> {
        // Same bounded retry as its neighbours in this adjudication. The
        // identity gate is one more O(1) round-trip on the same store, and a
        // reconnect-shaped blip there is not evidence about the layer — without
        // the retry a single dropped connection refuses the restore as
        // `unlink_unverified`, which is exactly the "a blip must not cost us a
        // restore" property the probe / `TYPE` / `UNLINK` retries exist for.
        // Inside `adjudicate_stream_high_water`, so it is bounded by the one
        // `RECOVERY_STREAM_ADJUDICATION_BUDGET` deadline like everything else.
        let identity = retry_aborted_with_jitter(
            RECOVERY_STREAM_IDENTITY_GATE_OP,
            RECOVERY_STREAM_PROBE_RETRY,
            |_attempt| self.live_mapping_resolves_identity_once(object_id, meta),
        )
        .await
        .map_err(service_err_to_checkpoint_err);

        match identity {
            Ok(Ok(true)) => {}
            Ok(Ok(false)) => {
                tracing::warn!(
                    object_id,
                    session_id = %meta.session_id,
                    layer_name = %meta.layer_name,
                    "recovery: the stale per-object stream was NOT removed — no live layer \
                     mapping resolves to this object_id, so the restore's identity guard is \
                     about to refuse this restore anyway (the layer was deleted, overwritten, or is inside \
                     its 30s stream-delete grace window). Destroying the stream here would strand \
                     subscribers parked on the `deleted` sentinel for a restore that will not \
                     happen"
                );
                // Proceed: `apply_restore` reaches the existing NotFound
                // verdict. Not an error here, and deliberately not counted on
                // the unresolved counter — nothing about the stream was left
                // unresolved, we simply had no business touching it.
                return Ok(None);
            }
            // A wrong-typed layer-mapping key — `classify_object_key_read_error`
            // has already emitted its operator-facing `error!`, and no retry can
            // `GET` a key that is not a string. It therefore rides the retry
            // driver's `Ok` channel (see `live_mapping_resolves_identity_once`),
            // which ends the loop on the attempt that raised it, and propagates
            // verbatim rather than being re-reported as a retriable
            // `unlink_unverified` refusal — which would re-drive a restore that
            // can never succeed (b.f4t sub-task 2).
            Ok(Err(permanent)) => return Err(permanent),
            Err(err) => {
                // We hold positive evidence of a stale top and cannot
                // determine whether we may clear it. Refuse: letting the
                // restore run on the chance that apply's own guard passes is
                // exactly how the layer ends up wedged.
                return self
                    .refuse_restore(
                        object_id,
                        "unlink_unverified",
                        format!(
                            "the identity round-trip that authorizes removing the stale \
                             stream failed (retries exhausted or the failure was terminal), so \
                             the removal could not be attempted: {err}"
                        ),
                    )
                    .await;
            }
        }

        // b.h4m: the peer-presence re-read, immediately before the removal.
        //
        // Everything above was decided against evidence gathered BEFORE step
        // 5's blob fetch — a multi-second I/O during which a peer replica can
        // land the whole rebuild and re-admit writers. Re-ask the structural
        // presence question here so the common shape of that race (peer header
        // landed anywhere in the fetch, client already writing to the winner's
        // stream) is a skip rather than the destruction of a live stream. See
        // this function's "What the re-read closes, and what it does not" for
        // the residual window (one round trip in the common case) and its
        // precise blast radius.
        //
        // Retried on the same `RECOVERY_STREAM_PROBE_RETRY` budget as its
        // neighbours, and that is window-NEUTRAL: the exploitable window is
        // measured from the LAST SUCCESSFUL read to the `UNLINK`, and a retry
        // only ever replaces a failed read with a later successful one. What
        // the retry buys is the failure mode the un-retried form introduced —
        // a single reconnect-shaped `HGETALL obj:hdr` blip refusing the whole
        // restore as `unlink_unverified` (Transient → `UNAVAILABLE`), which
        // this stretch of the adjudication did not do before b.h4m. Still
        // inside the one `RECOVERY_STREAM_ADJUDICATION_BUDGET` deadline, so
        // the extra attempts cannot lengthen the mutex hold past the ceiling.
        match retry_aborted_with_jitter(
            RECOVERY_STREAM_PEER_PRESENCE_REREAD_OP,
            RECOVERY_STREAM_PROBE_RETRY,
            |_attempt| self.peer_presence_reread_once(object_id),
        )
        .await
        {
            Ok(Ok(false)) => {}
            Ok(Ok(true)) => {
                tracing::debug!(
                    object_id,
                    session_id = %meta.session_id,
                    layer_name = %meta.layer_name,
                    "recovery: skipping the stale-stream removal — obj:hdr is now present and \
                     structurally complete, so a peer replica landed this rebuild while we were \
                     fetching the checkpoint blob and the stream under this object_id may be its \
                     LIVE one. Proceeding to the apply, which short-circuits on the same \
                     structural check and reports the layer restored"
                );
                // Not a refusal and not a removal, so nothing is counted:
                // `checkpoint_recovery_stream_refused_total` means "a stream
                // was cleared ahead of a restore", which is exactly what did
                // not happen here.
                //
                // What the skip rests on, and what it leaves standing. The
                // implausible stream is left in place, so this restore is
                // fenced only by guards this arm INHERITS rather than owns:
                // `apply_restore` step 1's structural presence read, which
                // short-circuits on the peer's header, and failing that, step
                // 2's identity guard — both point-in-time reads on keys this
                // decision fenced nothing on. The residual is therefore a
                // double fault: if the peer's structurally-complete `obj:hdr`
                // is lost between this re-read and step 1 while
                // `le:{sid}:layer:{name}` stays live — the partial-loss shape
                // step 3b's residue wipe cannot see, since its inventory
                // excludes the stream key — step 1 stops short-circuiting,
                // step 2 passes on the surviving mapping, and this replica
                // materializes at `snap_seq + band` underneath a stream top
                // far above it: exactly the stale-stream wedge the step-4
                // plausibility gate exists to prevent, re-entered through the
                // one arm that declined to act on it. It is reachable only by
                // that double fault. Nothing this arm OWNS announces it: the
                // skip above logs at `debug!`, and no counter moves —
                // `live_edit_checkpoint_recovery_stream_refused_total` ticks
                // only after a successful `UNLINK`, which is precisely what
                // was skipped, and
                // `live_edit_checkpoint_restore_fenced_total{presence_probe}`
                // cannot tick when the peer header it fences on is the thing
                // that vanished. Two signals do land, in this order. First the
                // plausibility gate's WARN, already emitted by the only caller
                // that reaches this function, naming the `object_id`, the
                // `snap_seq` and the implausible high-water. Then the first
                // post-restore `UpdateLayer`, which the b.ph9 stream-id
                // collision guard rejects whole with a non-retriable
                // `INTERNAL` — so the wedge surfaces as a failing write rather
                // than as silent divergence, and it stays operator-clearable,
                // by clearing the stream key by hand.
                //
                // Neither close was taken: refusing here instead
                // of skipping would fence it, at the cost of the "loser is
                // served" property this arm buys (the loser would carry a
                // retriable `UNAVAILABLE` rather than the winner's layer), and
                // folding the implausible high-water into the restore floor
                // was rejected on its own grounds — a band of ~1.8e12 hangs
                // clients on gap-fill. SDD §22.6's residual list states it in
                // full.
                return Ok(None);
            }
            // A wrong-typed `obj:hdr` — `classify_object_key_read_error` has
            // already emitted its operator-facing `error!` on that branch, and
            // no retry can `HGETALL` a key that is not a hash. It therefore
            // rides the retry driver's `Ok` channel (see
            // `peer_presence_reread_once`), which ends the loop on the attempt
            // that raised it, and propagates verbatim — exactly as
            // `refuse_restore`'s copy of this re-check does — rather than being
            // re-reported as a stream verdict.
            Ok(Err(permanent)) => return Err(permanent),
            Err(err) => {
                // Same disposition as the identity gate's `Err` arm above, and
                // for the same reason: we hold positive evidence of a stale top
                // and cannot establish whether the stream under it is a peer's
                // live one. Removing it on that ignorance is the failure this
                // read exists to prevent, so refuse instead — `refuse_restore`
                // takes one more look for a peer-landed header before it does.
                let err = service_err_to_checkpoint_err(err);
                return self
                    .refuse_restore(
                        object_id,
                        "unlink_unverified",
                        format!(
                            "the obj:hdr peer-presence re-read that authorizes removing the stale \
                             stream failed (retries exhausted or the failure was terminal), so the \
                             removal could not be attempted: {err}"
                        ),
                    )
                    .await;
            }
        }

        match self.unlink_stale_stream(object_id, stream_key).await {
            Ok(()) => {
                self.metrics
                    .inc_checkpoint_recovery_stream_refused(&self.service_label);
                Ok(None)
            }
            Err(err) => {
                self.refuse_restore(object_id, "unlink_unverified", format!("{err}"))
                    .await
            }
        }
    }

    /// `UNLINK` a stale per-object stream, **verifiably**, **before** the
    /// restore materializes over it.
    ///
    /// # Why the stream must go
    ///
    /// Called by [`Self::remove_stale_stream`] when the probe read a stream
    /// top it refuses to fold into the restore floor, and the SR-2.4 identity
    /// guard authorized the delete. Leaving that stream in place
    /// makes the restored layer **unwritable**. `apply_checkpoint` seeds
    /// `snap_seq + margin`; the stale stream top is a wall-clock timestamp
    /// far above it; so every `UpdateLayer` against the restored layer picks
    /// a `{new_seq}-0` the stream refuses, and both commit paths reject the
    /// write.
    ///
    /// The rejection is clean: since b.ph9 both paths fail closed on a
    /// stream-ID collision — the Lua fast path validates everything it can
    /// reject (input shape, key types) and then issues its `XADD` as the
    /// first mutating call, so nothing is written when it is refused, and the
    /// WATCH/MULTI/EXEC path probes the stream before opening its
    /// transaction. No state mutation lands, the header sequence does not
    /// advance, and subscribers miss nothing that the client was told
    /// succeeded. The client gets a loud non-retriable `INTERNAL` naming the
    /// collision.
    ///
    /// The claim is exact for the Lua path (default backend): every error it
    /// can return is raised before its first write. On the WATCH/MULTI/EXEC
    /// path it holds for the *collision* — the probe runs pre-`MULTI` — but a
    /// `WRONGTYPE` on a mutated key still surfaces as a per-command error
    /// frame inside `EXEC`, which Valkey does not roll back. That is a
    /// pre-existing slow-path property unrelated to this cleanup, and it is
    /// not reachable from the condition here: a stale stream makes the
    /// *stream* key collide, which the probe catches before `MULTI`.
    ///
    /// Pre-b.ph9 this was worse than an outage: `update_layer_fast.lua`
    /// wrote fields/sections, `HSET` the header, and only *then* `XADD`ed —
    /// and a mid-script `redis.call` failure does not roll back the writes
    /// that preceded it. Every write half-committed: state advanced,
    /// no stream entry existed for it, the client was told the write failed,
    /// and live subscribers diverged permanently. That is why this cleanup
    /// exists; it is now a write-availability measure rather than a
    /// corruption-prevention one, and still worth doing eagerly.
    ///
    /// # Why immediate removal, not a grace window
    ///
    /// An earlier revision used `PEXPIRE` with `LayerStore`'s
    /// `STREAM_DELETE_GRACE_MS`, reasoning by analogy with the two
    /// tombstone-writing paths. That analogy does not hold: the grace window
    /// is a window in which every write to the restored layer fails.
    ///
    /// The grace exists so that a subscriber parked between `XREAD BLOCK`
    /// round-trips on a layer being **deleted** can still observe the
    /// `deleted` sentinel and exit cleanly rather than looping against a
    /// vanished key. This path is a **resurrection**, not a deletion: there
    /// is no sentinel worth preserving (the layer is coming back), and a
    /// parked subscriber needs no window to survive the removal. While the key
    /// is missing its `XREAD BLOCK` keeps blocking — nothing errors, no
    /// `Deleted` is produced, and nothing re-bootstraps. The next commit's
    /// `XADD` recreates the key at `{seq}-0`, above the restore's guard band.
    /// A subscription whose next edit is that entry continues; every other
    /// one ends `stream_gap` when that entry reaches it (the stream
    /// multiplexer's head check finds nothing below it) and reconnects, and
    /// its catch-up delivers the restored content, stamped at `seeded_seq`
    /// for exactly that reconnect (SDD §12.3, §22.6). After a restore that is
    /// every attached subscription, since the restore skips a guard band; the
    /// catch-up re-sends content the client already had only when the restore
    /// lost nothing it held. One residual: a subscription whose cursor sits
    /// above the recreated stream's ids (it read an out-of-band entry near the
    /// stale top, above the header) receives nothing until the sequence passes
    /// its cursor. Full residual-damage note on [`Self::remove_stale_stream`].
    /// Nothing reads the stale stream to its benefit, and everything that
    /// writes it is rejected.
    ///
    /// `UNLINK` rather than `DEL` or `PEXPIRE 1`: the reclaim happens on a
    /// background thread, which matters because a wedged stream can hold
    /// up to `LIVE_EDIT_LAYER_STREAM_MAXLEN` OMPF-bearing entries, and we
    /// are on an inline RPC-triggered path under the K=1 admission mutex.
    /// `PEXPIRE 1` would leave a millisecond-wide write outage for no
    /// benefit.
    ///
    /// # Scope
    ///
    /// Only the plausibility-refusal path calls this, and only once the
    /// identity guard has authorized it. A probe that could not read a
    /// high-water takes no action on the key — there we have no evidence the
    /// top is problematic, and unlinking a healthy stream would throw away
    /// live subscribers' in-window catch-up for nothing. That arm refuses the
    /// restore instead ([`Self::classify_unreadable_stream`]).
    ///
    /// # Verified, not best-effort (b.4et)
    ///
    /// Pre-b.4et the `UNLINK` reply was bound as `()` — fred discards the
    /// integer, so the call could not tell "deleted it" from "there was
    /// nothing there" — and any error was logged and swallowed. The restore
    /// then ran regardless, seeding a header underneath a stream that may
    /// still have been sitting above it: a full write outage on the restored
    /// layer, discovered only by the first client that tried to write.
    ///
    /// Now the reply is bound as `u64` and the count decides:
    ///
    /// - `>= 1` — verified, we removed it.
    /// - `== 0` — the key was already gone at that instant. Confirm with a
    ///   `TYPE` re-probe reading `none`, because a peer replica racing the
    ///   same recovery may legitimately have cleared it first. Any other type
    ///   means something re-created the key between our two round-trips,
    ///   which no in-service writer can do (see [`Self::remove_stale_stream`]'s
    ///   window argument) — not verified, and not retried, because a retry
    ///   has no transient signal to converge on.
    ///
    /// Transient failures are retried per [`RECOVERY_STREAM_PROBE_RETRY`]. On
    /// exhaustion this returns [`CheckpointError::Transient`] and the caller
    /// refuses the restore: the header is never seeded under a stream top we
    /// could not clear.
    async fn unlink_stale_stream(
        &self,
        object_id: u64,
        stream_key: &str,
    ) -> Result<(), CheckpointError> {
        let verified = retry_aborted_with_jitter(
            RECOVERY_SEAM_STALE_STREAM_UNLINK,
            RECOVERY_STREAM_PROBE_RETRY,
            |_attempt| self.unlink_stale_stream_once(stream_key),
        )
        .await
        .map_err(service_err_to_checkpoint_err)?;

        if verified {
            tracing::info!(
                object_id,
                key = %stream_key,
                "recovery: removed stale per-object stream ahead of the restore; \
                 the restored layer's first write recreates it from the seeded sequence"
            );
            return Ok(());
        }

        Err(CheckpointError::Transient {
            source: format!(
                "UNLINK of stale stream `{stream_key}` reported deleting nothing and the key is \
                 still present, so its removal could not be verified"
            )
            .into(),
        })
    }

    /// What a selected base over the per-delta entry limit answers on access:
    /// the SDD §22.6 step-2 identity verdict first, then — only for a layer
    /// still mapped — the SR-14.7 refusal.
    ///
    /// Both arms' refusal sites call this, and only once they have detected
    /// such a base, so neither arm's success path awaits anything new. `meta`
    /// is the selected lineage's meta; its identity columns are base-sourced on
    /// both arms (see `select_restore_source`'s `# The composed meta`).
    ///
    /// # Why the identity verdict comes first
    ///
    /// The refusal is a statement about the backup. The identity verdict is a
    /// statement about the layer, and it decides what the client is told. A
    /// layer deleted, or replaced by a `CreateLayer` that minted a new
    /// `object_id`, during the state loss has no live mapping resolving to this
    /// `object_id`; `apply_restore`'s identity guard answers every such layer
    /// not-found, so the handlers answer `NOT_FOUND` and a subscription
    /// re-bootstraps. Refusing first would answer a big layer `INTERNAL`
    /// instead, a terminal status that never tells the client the layer is
    /// gone.
    ///
    /// # Returns
    ///
    /// The error the caller answers with, by the verdict:
    ///
    /// - **No live mapping resolves to `object_id`** (or the meta's identity is
    ///   empty) → [`CheckpointError::NotFound`], the value `apply_restore`'s
    ///   guard returns. The base-only arm maps it to `NotRecoverable` and the
    ///   `not_recoverable` attempt, exactly as the hook's apply `NotFound` arm
    ///   does; the composed arm returns `Ok(None)` from the selection, which
    ///   lands on the same answer. No refusal WARN: the skip is logged the way
    ///   `apply_restore` logs its own.
    /// - **The mapping resolves** → [`refuse_over_limit_base_on_access`]'s
    ///   `Permanent`, logged there.
    /// - **The mapping key is wrong-typed** → that `Permanent`, verbatim, as
    ///   [`Self::remove_stale_stream`] propagates it: no retry re-types a key,
    ///   and the classifier has already logged it.
    /// - **The mapping read failed**, after [`RECOVERY_STREAM_PROBE_RETRY`]'s
    ///   attempts, terminally, or by outlasting
    ///   [`RECOVERY_STREAM_ADJUDICATION_BUDGET`] → the read's error (or the
    ///   elapse's own `Transient`), which
    ///   `service_err_to_checkpoint_err` makes `Transient`, so the attempt
    ///   answers `UNAVAILABLE` and the client retries. Not the refusal's
    ///   `Permanent`: which terminal answer is right is exactly what the failed
    ///   read would have decided, and answering `INTERNAL` would tell a gone
    ///   layer `INTERNAL` on a store blip — the answer this verdict exists to
    ///   replace — with no retry to correct it. The probe's other callers answer
    ///   a failed read the same way: `apply_restore`'s guard propagates its
    ///   `Transient`, and `remove_stale_stream` refuses `unlink_unverified`,
    ///   also `Transient`.
    ///
    /// # Bound
    ///
    /// The read is retried like `remove_stale_stream`'s, on the same op label
    /// and [`RECOVERY_STREAM_PROBE_RETRY`] budget, and the whole retried read
    /// sits under [`RECOVERY_STREAM_ADJUDICATION_BUDGET`] (10 s), the same
    /// single deadline the stream adjudication runs under. The retry count
    /// alone would admit three full `LIVE_EDIT_VALKEY_COMMAND_TIMEOUT_MS`
    /// waits under the admission mutex, so the hold would move with a
    /// per-command timeout; the deadline is what keeps it fixed. This matters
    /// on the base-only arm, which runs after the recovery fetch budget and
    /// before the stream adjudication, under neither; on the composed arm the
    /// fetch budget's timeout also encloses it. An elapse answers like a
    /// failed read: a WARN and a `Transient`, so the client retries.
    async fn adjudicate_over_limit_base_on_access(
        &self,
        object_id: u64,
        head_seq: u64,
        meta: &CheckpointMeta,
    ) -> CheckpointError {
        // Bounded by the adjudication's one deadline, not only by the retry
        // count: see `# Bound`. An elapse is the read-failure arm's answer.
        let identity = match tokio::time::timeout(
            RECOVERY_STREAM_ADJUDICATION_BUDGET,
            retry_aborted_with_jitter(
                RECOVERY_STREAM_IDENTITY_GATE_OP,
                RECOVERY_STREAM_PROBE_RETRY,
                |_attempt| self.live_mapping_resolves_identity_once(object_id, meta),
            ),
        )
        .await
        {
            Ok(result) => result.map_err(service_err_to_checkpoint_err),
            Err(_elapsed) => Err(CheckpointError::Transient {
                source: format!(
                    "restore identity gate for object {object_id} exceeded the {}ms \
                     adjudication budget",
                    RECOVERY_STREAM_ADJUDICATION_BUDGET.as_millis()
                )
                .into(),
            }),
        };

        match identity {
            Ok(Ok(true)) => refuse_over_limit_base_on_access(object_id, head_seq),
            Ok(Ok(false)) => {
                tracing::warn!(
                    object_id,
                    session_id = %meta.session_id,
                    layer_name = %meta.layer_name,
                    "checkpoint recovery: the selected backup's layer identity does not resolve to \
                     this object; skipping recovery (layer deleted, overwritten, or the object_id \
                     was reused by a different layer)"
                );
                CheckpointError::NotFound
            }
            // A wrong-typed layer-mapping key, already logged by its
            // classifier and carried on the retry driver's `Ok` channel (see
            // `live_mapping_resolves_identity_once`).
            Ok(Err(permanent)) => permanent,
            Err(err) => {
                // `service_err_to_checkpoint_err` keeps only a `NotFound`
                // verbatim, and the one-attempt helper never raises one, so
                // this is `Transient` — as is the budget elapse above — and
                // the caller cannot read it as the gone-layer verdict above.
                tracing::warn!(
                    object_id,
                    head_seq,
                    session_id = %meta.session_id,
                    layer_name = %meta.layer_name,
                    error = %err,
                    "checkpoint recovery selected a backup over the per-delta entry limit, and the \
                     layer-mapping read that decides between not-found and refusing it failed; \
                     answering retriably"
                );
                err
            }
        }
    }

    /// One SR-2.4 identity-gate attempt, classified for the retry driver.
    ///
    /// [`live_mapping_resolves_blob_identity`] speaks `CheckpointError` and
    /// delegates retry to its caller by contract (see its `# Errors` section);
    /// [`retry_aborted_with_jitter`] speaks `ServiceError` and converges on
    /// [`ServiceError::Aborted`] as its only in-band retriable signal. This is
    /// the adapter between the two, in the same shape as the probe / `TYPE` /
    /// `UNLINK` one-attempt helpers: a transient becomes `Aborted` and is
    /// retried, and an exhausted budget lands on its caller's `Err` arm after
    /// `service_err_to_checkpoint_err` converts it back: in
    /// [`Self::remove_stale_stream`] the `unlink_unverified` refusal, in
    /// [`Self::adjudicate_over_limit_base_on_access`] a retriable answer in
    /// place of the over-limit refusal.
    ///
    /// # Why the return type is a nested `Result`
    ///
    /// Same reason as [`Self::peer_presence_reread_once`], and the same shape.
    /// The inner `Result` carries the one error class that must NOT be retried
    /// and must reach the caller **unchanged**: the `Permanent` verdict
    /// [`classify_object_key_read_error`](crate::checkpoint::apply) raises for a
    /// wrong-typed layer-mapping key `le:{sid}:layer:{name}`. No retry re-types
    /// a key (b.f4t sub-task 2), and that classifier has already emitted the
    /// operator-facing `error!`.
    ///
    /// Routing it through `ServiceError` instead would strip the variant twice
    /// over — folded to [`ServiceError::Internal`] here, then folded back to
    /// `Transient` by `service_err_to_checkpoint_err` at the call site, since it
    /// preserves nothing but `NotFound` — so a permanently unsatisfiable read
    /// would be refused as a retriable `unlink_unverified` and the recovery hook
    /// would re-drive the identical restore forever. Returning it through the
    /// driver's `Ok` channel is what gets it to the caller intact.
    ///
    /// No fault-injection seam here, unlike its three siblings: this gate reads
    /// the live layer mapping, so a test drives either verdict by staging that
    /// key directly rather than by injecting.
    async fn live_mapping_resolves_identity_once(
        &self,
        object_id: u64,
        meta: &CheckpointMeta,
    ) -> Result<Result<bool, CheckpointError>, ServiceError> {
        match live_mapping_resolves_blob_identity(
            self.layer_store.client(),
            self.layer_store.keys(),
            object_id,
            meta,
        )
        .await
        {
            Ok(resolved) => Ok(Ok(resolved)),
            Err(e @ CheckpointError::Permanent { .. }) => Ok(Err(e)),
            Err(CheckpointError::Transient { source }) => Err(ServiceError::Aborted(format!(
                "restore identity gate for object {object_id}: {source}"
            ))),
            Err(other) => Err(ServiceError::Internal(format!(
                "restore identity gate for object {object_id}: {other}"
            ))),
        }
    }

    /// One b.h4m peer-presence re-read attempt — the structural
    /// `HGETALL obj:hdr` that guards the stale-stream `UNLINK` — classified for
    /// the retry driver.
    ///
    /// # Why the return type is a nested `Result`
    ///
    /// The outer `Result` is [`retry_aborted_with_jitter`]'s channel:
    /// [`ServiceError::Aborted`] is its only in-band retriable signal, so a
    /// transient header read maps onto it and is re-driven, exactly as the
    /// probe / `TYPE` / identity-gate / `UNLINK` one-attempt helpers do.
    ///
    /// The inner `Result` carries the one error class that must NOT be retried
    /// and must reach the caller **unchanged**: the `Permanent` wrong-typed
    /// `obj:hdr` verdict. No retry can `HGETALL` a key that is not a hash, and
    /// [`classify_object_key_read_error`](crate::checkpoint::apply) has already
    /// emitted the operator-facing `error!` for it, so the caller propagates it
    /// verbatim rather than re-reporting it as a stream verdict. Returning it
    /// through the driver's `Ok` channel is what gets it there intact —
    /// routing it through `ServiceError` instead would strip the variant
    /// (`service_err_to_checkpoint_err` folds everything but `NotFound` into
    /// `Transient`), and the caller would re-drive a restore that can never
    /// succeed.
    ///
    /// The seam fires per attempt, like its three `RECOVERY_SEAM_*` siblings:
    /// an injected `Transient` is a retriable blip, any other injected kind is
    /// terminal.
    async fn peer_presence_reread_once(
        &self,
        object_id: u64,
    ) -> Result<Result<bool, CheckpointError>, ServiceError> {
        #[cfg(feature = "test-checkpoint-fault-injection")]
        self.inject_recovery_seam(RESTORE_SEAM_BEFORE_STALE_STREAM_UNLINK)?;

        match self.is_layer_present(object_id).await {
            Ok(present) => Ok(Ok(present)),
            Err(e @ CheckpointError::Permanent { .. }) => Ok(Err(e)),
            Err(other) => Err(ServiceError::Aborted(format!(
                "peer-presence obj:hdr re-read for object {object_id}: {other}"
            ))),
        }
    }

    /// One `UNLINK` attempt plus, when it deleted nothing, the `TYPE`
    /// confirmation. Returns whether the key is verifiably gone.
    async fn unlink_stale_stream_once(&self, stream_key: &str) -> Result<bool, ServiceError> {
        #[cfg(feature = "test-checkpoint-fault-injection")]
        self.inject_recovery_seam(RECOVERY_SEAM_STALE_STREAM_UNLINK)?;

        // `u64`, not `()`: the count IS the verification. Binding the reply
        // this way follows `tests/integration/checkpoint/
        // e2e_ext_value_orphan_sweep_reconcile_test.rs`.
        let removed: u64 = self
            .layer_store
            .client()
            .pool
            .next()
            .unlink(stream_key)
            .await
            .map_err(|e| {
                map_snapshot_load_err(
                    FredErrOrigin::client(self.layer_store.client(), POOL_LABEL_DATA),
                    "UNLINK stale obj:stream",
                    e,
                )
            })?;
        if removed >= 1 {
            return Ok(true);
        }
        Ok(matches!(
            self.read_stream_key_type_once(stream_key).await?,
            StreamKeyType::Absent
        ))
    }

    /// Route an injected fault from the b.4et recovery seams (and the b.h4m
    /// peer-presence one, which is now driven the same way) into the retry
    /// driver's error vocabulary.
    ///
    /// `Transient` becomes [`ServiceError::Aborted`] — the in-band retriable
    /// signal [`retry_aborted_with_jitter`] converges on — so a test can fail
    /// the first N attempts and let attempt N+1 through. Every other injected
    /// kind is terminal, matching how a non-transient fred error behaves.
    #[cfg(feature = "test-checkpoint-fault-injection")]
    fn inject_recovery_seam(&self, op: &'static str) -> Result<(), ServiceError> {
        let Some(injector) = self.restore_fault_injector.as_ref() else {
            return Ok(());
        };
        match injector.maybe_inject(op) {
            Ok(()) => Ok(()),
            Err(CheckpointError::Transient { source }) => Err(ServiceError::Aborted(format!(
                "injected transient fault at seam `{op}`: {source}"
            ))),
            Err(other) => Err(ServiceError::Internal(format!(
                "injected fault at seam `{op}`: {other}"
            ))),
        }
    }
}

/// What a `TYPE le:{oid}:obj:stream` round-trip says about the key the
/// high-water probe could not read a usable sequence out of (b.4et).
///
/// The three arms are the whole point: the probe script's single `nil` covers
/// all of them, and they demand opposite actions. Folding any two back
/// together reintroduces the defect.
#[derive(Debug)]
enum StreamKeyType {
    /// `"none"` — no key at all. The stream did not survive the state loss;
    /// the restore floor is `snap_seq` alone. The dominant production shape.
    Absent,
    /// `"stream"` — the key is there and is ours, and we could not read its
    /// top. Evidence unknown; the restore must not proceed.
    Stream,
    /// Any other Valkey type — a foreign value on the `le:` keyspace. Carries
    /// the reported type so the WARN can name it.
    Foreign(String),
}

/// Hard ceiling on the b.4et stream adjudication as a whole — probe,
/// classification, identity gate, the b.h4m peer-presence `HGETALL obj:hdr`
/// re-read, verified `UNLINK`, and the peer-landed structural `HGETALL obj:hdr`
/// re-check that precedes a refusal, end to end. Everything but that last
/// re-check is retried underneath it per [`RECOVERY_STREAM_PROBE_RETRY`].
///
/// It also bounds, separately and for the same reason, the identity gate that
/// precedes the over-limit base refusal
/// ([`RecoveryHook::adjudicate_over_limit_base_on_access`]); the two never run
/// in one attempt.
///
/// # Why a wall-clock deadline and not just a retry count
///
/// [`RECOVERY_STREAM_PROBE_RETRY`] bounds *backoff sleep*, which is
/// milliseconds. It does not bound *round-trips*: a Valkey that accepts a
/// command and never answers costs the full
/// `LIVE_EDIT_VALKEY_COMMAND_TIMEOUT_MS` (default 5 s, applied as fred's
/// `default_command_timeout` in `src/storage/client.rs`) on every attempt. The
/// retry counts alone therefore admit, on the default timeout:
///
/// - refuse path — probe 3 × 5 s + `TYPE` 3 × 5 s + the refusal's own
///   single-shot `HGETALL obj:hdr` 5 s ≈ **35 s**;
/// - `UNLINK` path — probe 3 × 5 s + identity gate 3 × 5 s + the b.h4m
///   peer-presence `HGETALL obj:hdr` 3 × 5 s + `UNLINK` 3 ×
///   (5 s each, plus a confirming `TYPE` 5 s each) + the refusal's own
///   `HGETALL obj:hdr` 5 s ≈ **80 s**.
///
/// All of it inline on an RPC-triggered restore, under the K=1 per-`object_id`
/// admission mutex, with every other reader of that layer queued behind it —
/// and `SubscribeReadLayer`, the RPC that most often triggers a recovery, is
/// exempt from `LIVE_EDIT_RPC_DEADLINE_MS`, so nothing upstream would have cut
/// it short. A minute-long mutex hold is not a latency budget, it is an
/// outage.
///
/// # The number
///
/// 10 s = 2 × the default command timeout. That is the smallest value that
/// still lets the adjudication ride out the failure it is *supposed* to ride
/// out — one round-trip lost to a connection drop, plus one full re-attempt
/// after fred reconnects — and it refuses everything longer. That framing holds
/// only when the first failure is FAST: a connection drop fails in milliseconds,
/// leaving essentially the whole 10 s for the re-attempt. It does NOT hold when
/// the failures are HUNG rather than dropped — two attempts that each burn the
/// full default 5 s command timeout, plus the backoff sleep between them, are
/// already over budget, so the deadline fires during (not after) the second
/// attempt and the adjudication refuses. That is the intended trade: a backing
/// store answering nothing twice in a row is the outage case this ceiling exists
/// to cut short, not the blip it is sized to ride out. It is a ceiling,
/// not a target: the healthy path is two O(1) round-trips and completes in
/// single-digit milliseconds, so this deadline is unreachable except on a
/// backing store that is already failing.
///
/// Deliberately a const rather than a function of
/// `LIVE_EDIT_VALKEY_COMMAND_TIMEOUT_MS`: the point is a bound on how long one
/// client may wedge a layer's admission mutex, and that bound should not move
/// because an operator widened a per-command timeout for unrelated reasons. An
/// operator who raises the command timeout above this simply finds this
/// deadline binding first — which is the intent.
const RECOVERY_STREAM_ADJUDICATION_BUDGET: Duration = Duration::from_secs(10);

/// Grace for the peer-landed structural `HGETALL obj:hdr` re-check on the
/// [`RECOVERY_STREAM_ADJUDICATION_BUDGET`] expiry arm, so the hard ceiling on
/// the whole adjudication is `budget + grace`.
///
/// Three orders of magnitude above a healthy one-key `HGETALL` and a tenth of
/// the budget
/// it follows: long enough that a merely-slow store still gets its answer
/// (turning a spurious `UNAVAILABLE` into the correct "a peer already landed
/// this"), short enough that a store which has stopped answering — the case
/// that just burned the budget — is not waited on twice. See
/// [`RecoveryHook::refuse_over_budget`] for why refusing without the re-check
/// is safe when it expires.
const RECOVERY_STREAM_ADJUDICATION_PEER_GRACE: Duration = Duration::from_secs(1);

/// Op label the retry driver's backoff tracing uses for the SR-2.4 identity
/// gate inside the b.4et stream adjudication, and for the same gate where it
/// precedes the over-limit base refusal
/// ([`RecoveryHook::adjudicate_over_limit_base_on_access`]); the two never run
/// in one attempt, since the refusal path ends before the high-water probe.
///
/// Deliberately NOT a fault-injection seam const like its three neighbours
/// ([`RECOVERY_SEAM_STREAM_HIGH_WATER_PROBE`] and friends): nothing injects
/// here, because both verdicts of this gate are reachable by staging the live
/// layer mapping. It exists only so a retried blip on this round-trip is
/// attributable in the logs.
const RECOVERY_STREAM_IDENTITY_GATE_OP: &str = "recovery_stream_identity_gate";

/// Op label the retry driver's backoff tracing uses for the b.h4m peer-presence
/// `HGETALL obj:hdr` re-read that immediately precedes the stale-stream
/// `UNLINK`.
///
/// Separate from
/// [`RESTORE_SEAM_BEFORE_STALE_STREAM_UNLINK`](crate::checkpoint::apply::RESTORE_SEAM_BEFORE_STALE_STREAM_UNLINK),
/// which is the fault-injection seam name fired inside
/// [`RecoveryHook::peer_presence_reread_once`]: a seam const names the string a
/// test injects on, an op label names the string the retry driver logs, and
/// spending one const on both jobs means a rename for one purpose silently
/// retargets the other. Modelled on [`RECOVERY_STREAM_IDENTITY_GATE_OP`] rather
/// than on the three `RECOVERY_SEAM_*` round-trips that still pass their seam
/// const straight into the driver.
const RECOVERY_STREAM_PEER_PRESENCE_REREAD_OP: &str = "recovery_stream_peer_presence_reread";

/// Per-round-trip retry budget inside the b.4et stream adjudication: the
/// high-water `EVALSHA`, the `TYPE` classification, the SR-2.4 identity gate,
/// the b.h4m peer-presence `HGETALL obj:hdr` re-read, and the stale-stream
/// `UNLINK` each get this many attempts. The same identity gate, where it
/// precedes the over-limit base refusal outside that adjudication
/// ([`RecoveryHook::adjudicate_over_limit_base_on_access`]), gets the same
/// budget.
///
/// # What this bounds, and what it does not
///
/// It bounds **backoff sleep only** — three attempts with a 10 ms base and a
/// 50 ms cap add up to ~60 ms of sleep per round-trip. It does **not** bound
/// wall time: each attempt can additionally block for a full
/// `LIVE_EDIT_VALKEY_COMMAND_TIMEOUT_MS` (default 5 s) waiting on a store that
/// took the command and went quiet, so on retry counts alone the adjudication's
/// worst case is ~35 s (refuse path) to ~80 s (`UNLINK` path). The number that
/// makes the "this stays well inside a client's patience" claim true is
/// [`RECOVERY_STREAM_ADJUDICATION_BUDGET`], the single deadline wrapped around
/// the whole sequence; this const is only the shape of the retrying underneath
/// it. The over-limit refusal's identity gate, which runs outside the stream
/// adjudication, sits under its own copy of that deadline, for the same
/// reason (see [`RecoveryHook::adjudicate_over_limit_base_on_access`]'s
/// `# Bound`).
///
/// # Why three
///
/// The failure this is meant to absorb is a reconnect-shaped blip: one
/// round-trip lost to a dropped connection, retried after fred re-establishes
/// it. Three attempts covers that with one spare. More attempts do not buy
/// resilience to anything else — a store that fails three consecutive attempts
/// is not blipping — and each extra attempt is another command timeout of mutex
/// hold that the overall budget then has to cut short anyway. So the two
/// numbers are consistent by construction: 3 attempts × ~2 timeouts' worth of
/// wall time is what the 10 s budget affords.
///
/// # Deliberately not tunable
///
/// Raising it cannot make a restore succeed that the overall budget would let
/// finish — the budget, not the attempt count, is the binding constraint on a
/// store that is actually failing. So the only thing an operator could buy is a
/// slightly different distribution of attempts inside a fixed 10 s window,
/// paid for in their own clients' latency, against an outage that the caller's
/// retry (the handlers' retry-once recovery dispatch, plus the client's own
/// reconnect) already covers at a level where nobody holds a mutex.
const RECOVERY_STREAM_PROBE_RETRY: RetryConfig = RetryConfig {
    max_attempts: 3,
    initial_delay_ms: 10,
    cap_delay_ms: 50,
};

/// Largest gap above a checkpoint's `snap_seq` that
/// [`RecoveryHook::probe_stream_high_water`] will accept as a genuine
/// sequence number rather than a foreign ID.
///
/// A legitimate gap is the commit count that landed between the last
/// checkpoint and the crash, on **one object**. Even at a wildly generous
/// 10,000 commits/sec sustained for a full 24 hours of checkpointer outage
/// — orders of magnitude beyond what a per-object serialized OMPF-parse +
/// Lua-EVAL pipeline can do — that is under 9e8. One billion is therefore
/// generous for the true signal.
///
/// It is also unreachable by the noise it exists to reject. Valkey mints an
/// auto-ID (`*`) from unix-milliseconds, which has been above 1e12 since
/// 2001 and only rises, so a timestamp-shaped ID clears this bound by three
/// orders of magnitude and always will. The two populations do not overlap
/// and cannot start overlapping with time.
const MAX_PLAUSIBLE_HIGH_WATER_GAP: u64 = 1_000_000_000;

/// Parse the sequence component (the `ms` field, which this service uses as
/// the commit sequence number) out of a Valkey Stream entry ID.
///
/// Entry IDs are `<ms>-<seq>`; both commit paths `XADD` at `{sequence_num}-0`,
/// so the leading component IS the layer's sequence number. Returns `None`
/// for any ID that does not parse as `<u64>-<...>`, including a bare `<u64>`
/// with no separator (server replies always carry both components, so a
/// missing separator means the reply shape is not what we assumed).
///
/// Local to the checkpoint module by design: `src/storage/notifications.rs`
/// owns the subscriber-side cursor handling and its internals are not part
/// of this module's contract surface.
fn parse_stream_id_seq(entry_id: &str) -> Option<u64> {
    entry_id.split_once('-')?.0.parse::<u64>().ok()
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;
    use crate::checkpoint::adapters::in_memory::InMemoryCheckpointStore;
    use crate::checkpoint::store::{CheckpointBlob, CheckpointHandle, CheckpointMeta};
    use crate::config::{Config, PodMemorySource};
    use crate::domain::ompf::OmpfContainer;
    use crate::observability::metrics::metrics_scrape_only;
    use crate::storage::layer::LayerStore;
    use crate::storage::test_support::TestValkey;
    use async_trait::async_trait;
    use bytes::Bytes;
    use fred::interfaces::KeysInterface;
    use oodf_ffi::OwnedDelta;
    use std::collections::HashSet;
    use std::sync::Mutex as StdMutex;

    /// Build a `Config` with the fields the recovery hook + apply path
    /// consult, defaulted otherwise. Mirrors `apply.rs`'s `test_config`.
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
            service_label: "test-replica".to_string(),
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

    /// Build the smallest valid reset-delta OMPF (no commands, no fields).
    fn build_empty_reset_ompf() -> Bytes {
        let delta = OwnedDelta {
            is_diff: true,
            base_version: 0,
            ..OwnedDelta::default()
        };
        let delta_bytes = oodf_ffi::build_delta(&delta).expect("build_delta");
        let ompf = OmpfContainer::serialize(&delta_bytes, &[]).expect("OMPF serialize");
        Bytes::from(ompf)
    }

    fn make_metrics() -> SharedMetrics {
        Arc::new(metrics_scrape_only("test-replica"))
    }

    /// Probe the live label-cell counter value for a given outcome at
    /// `scope="layer"` (this module's recovery hook), read from the rendered
    /// scrape exposition. Epic-2's session-recovery counterpart uses
    /// `scope="session"`. The `result × scope × service` grid is
    /// pre-initialised in `Metrics::new`, so every cell renders from boot.
    fn metric_value(metrics: &SharedMetrics, result: &str) -> u64 {
        use crate::observability::metrics::test_util::{render, sample_value};
        sample_value(
            &render(metrics),
            "live_edit_checkpoint_recovery_attempts_total",
            &[
                ("result", result),
                ("scope", "layer"),
                ("service", "test-replica"),
            ],
        )
        .unwrap_or(0.0) as u64
    }

    /// EVERY `reason` cell of
    /// `live_edit_checkpoint_recovery_stream_unresolved_total{reason, service}`,
    /// for a before/after comparison.
    ///
    /// The whole pinned vocabulary rather than one cell, for the reason the
    /// integration lane's `stream_unresolved_cells` gives: an arm that ticks the
    /// WRONG reason is a different defect from one that does not tick at all,
    /// and only a full sweep tells them apart. It is also what makes "this arm
    /// ticks NOTHING" assertable as a single equality against the `before`
    /// snapshot.
    fn stream_unresolved_cells(metrics: &SharedMetrics) -> Vec<(&'static str, u64)> {
        use crate::observability::metrics::test_util::{render, sample_value};
        use crate::observability::metrics::CHECKPOINT_RECOVERY_STREAM_UNRESOLVED_REASONS;
        let text = render(metrics);
        CHECKPOINT_RECOVERY_STREAM_UNRESOLVED_REASONS
            .iter()
            .map(|reason| {
                (
                    *reason,
                    sample_value(
                        &text,
                        "live_edit_checkpoint_recovery_stream_unresolved_total",
                        &[("reason", reason), ("service", "test-replica")],
                    )
                    .unwrap_or(0.0) as u64,
                )
            })
            .collect()
    }

    /// Probe one `outcome` cell of
    /// `live_edit_checkpoint_restore_lineage_total{outcome, service}` — the
    /// SR-3.1 lineage family, orthogonal to the attempt counter above.
    ///
    /// Zero-seeded at `Metrics::new` from `CHECKPOINT_RESTORE_LINEAGE_OUTCOMES`,
    /// so an absent series means the vocabulary moved underneath the test rather
    /// than "this outcome has not happened"; the `unwrap_or(0.0)` matches
    /// [`metric_value`]'s posture in this module for the same reason (these
    /// tests construct a fresh `Metrics` per case and assert absolute values, so
    /// a genuinely-unseeded cell would fail the assertion anyway).
    fn lineage_metric_value(metrics: &SharedMetrics, outcome: &str) -> u64 {
        use crate::observability::metrics::test_util::{render, sample_value};
        sample_value(
            &render(metrics),
            "live_edit_checkpoint_restore_lineage_total",
            &[("outcome", outcome), ("service", "test-replica")],
        )
        .unwrap_or(0.0) as u64
    }

    /// What [`FakeCheckpointStore`] does when the hook reaches step 5 and
    /// runs [`RecoveryHook::select_restore_source`].
    ///
    /// The layer-checkpoint fetch is the *only* axis every outcome-arm test in
    /// this module varies — every other trait method either delegates to a real
    /// in-memory store or is off-limits to this hook entirely — so it is the
    /// only axis the shared double is parameterised on.
    ///
    /// # It governs all three layer-read methods, not just one
    ///
    /// Selection is `list_layer_checkpoints` → `get_layer_checkpoint` since
    /// SR-2.5; `get_latest_layer_checkpoint` is no longer on the recovery path
    /// at all. A posture scripted onto `get_latest` alone would therefore be
    /// silently vacated — a fixture minting `Transient` there while its `list`
    /// answered `Ok(vec![])` would turn every `Error(Transient)` assertion in
    /// this module into an unnoticed `NotRecoverable` one. So each variant
    /// below is honoured by **all three** methods, and the `get_latest` arm is
    /// kept (rather than deleted) so a future caller that reaches for it drives
    /// the same posture.
    enum LayerFetch {
        /// Answer from the inner [`InMemoryCheckpointStore`]: `Ok(Some(..))`
        /// for a seeded object, `Ok(None)` otherwise. The realistic path, and
        /// the one the `Restored` / `NotRecoverable` arms need.
        Delegate,
        /// Fail with a classified [`CheckpointError`], minted per call because
        /// the error type is not `Clone`.
        Fail(fn() -> CheckpointError),
        /// Never answer within any deadline a caller would tolerate.
        ///
        /// The empty / `Ok(None)` answer the stall eventually returns is
        /// deliberately the *benign* one: if the fetch budget ever stopped
        /// applying, the tests that use this would see `NotRecoverable`, not an
        /// error, so a regression cannot disguise itself as the outcome under
        /// test.
        Stall,
        /// Answer like [`Self::Delegate`], after closing `client`'s data pool,
        /// once, on whichever layer read comes first. For a hook whose
        /// `LayerStore` reads through `client`, every Valkey read before the
        /// selection succeeds (the step-4 `is_layer_present` re-check among
        /// them) and every one after its first store read fails. A
        /// selection-time Valkey read therefore meets a closed pool, which no
        /// other posture can arrange.
        DelegateAfterClosingPool {
            client: Box<crate::storage::client::ValkeyClient>,
            closed: std::sync::atomic::AtomicBool,
        },
    }

    /// The one `CheckpointStore` double this module's tests drive
    /// `try_restore_layer` against.
    ///
    /// It replaced five near-identical hand-written stores (transient /
    /// permanent / conflict / stalling / panic-on-session-methods), each ~90
    /// lines of the same fifteen trait methods, differing only in what
    /// `get_latest_layer_checkpoint` returned. That shape does not just cost
    /// lines: a method added to [`CheckpointStore`] forced five identical
    /// edits, and the fifth one being subtly different from the other four is
    /// exactly the kind of divergence nobody would notice.
    ///
    /// # Session methods panic — on *every* variant
    ///
    /// Layer-state recovery MUST NOT touch the Epic-2 session-checkpoint
    /// surface (`put_session_checkpoint` / `get_session_checkpoint` /
    /// `delete_session_checkpoint`); session-state DR is
    /// [`SessionRecoveryHook`](crate::checkpoint::SessionRecoveryHook)'s job
    /// and reaching it from here would be an unspecified cross-hook leak. That
    /// used to be pinned by one dedicated fixture used by one test; folding it
    /// into the shared double makes it a standing assertion under every
    /// recovery test in this module rather than a single tripwire.
    struct FakeCheckpointStore {
        inner: InMemoryCheckpointStore,
        fetch: LayerFetch,
    }

    impl FakeCheckpointStore {
        /// How long [`LayerFetch::Stall`] hangs for.
        ///
        /// Long enough that a test which timed out on the *stall* instead of on
        /// the recovery fetch budget would blow its own nextest budget rather
        /// than passing by accident.
        const STALL: Duration = Duration::from_secs(30);

        /// Fetches answer from a real (initially empty) in-memory store.
        fn delegating() -> Self {
            Self {
                inner: InMemoryCheckpointStore::new(),
                fetch: LayerFetch::Delegate,
            }
        }

        fn failing_with(mint: fn() -> CheckpointError) -> Self {
            Self {
                inner: InMemoryCheckpointStore::new(),
                fetch: LayerFetch::Fail(mint),
            }
        }

        fn always_transient() -> Self {
            Self::failing_with(|| CheckpointError::Transient {
                source: "synthetic transient failure from FakeCheckpointStore".into(),
            })
        }

        /// `Permanent` is a distinct fixture from `Transient` because the
        /// hook's step-4 outcome mapping must route BOTH onto the same
        /// `Error(_)` fall-through — a regression that special-cased only
        /// `Transient` and dropped `Permanent` into `NotRecoverable` would
        /// otherwise slip past.
        fn always_permanent() -> Self {
            Self::failing_with(|| CheckpointError::Permanent {
                source: "synthetic permanent failure from FakeCheckpointStore".into(),
            })
        }

        /// `Conflict` is the fourth classified variant and the trait contract
        /// permits it from any store method on an optimistically-raced write.
        /// It must reach the same `Error(_)` arm; landing in
        /// `not_recoverable` instead would make the conflict signal vanish
        /// from the cell operators alert on.
        fn always_conflict() -> Self {
            Self::failing_with(|| CheckpointError::Conflict)
        }

        fn stalling() -> Self {
            Self {
                inner: InMemoryCheckpointStore::new(),
                fetch: LayerFetch::Stall,
            }
        }

        /// [`LayerFetch::DelegateAfterClosingPool`]'s one close; a no-op on
        /// every other posture and on every later call.
        async fn close_pool_once(&self) {
            if let LayerFetch::DelegateAfterClosingPool { client, closed } = &self.fetch {
                if !closed.swap(true, std::sync::atomic::Ordering::SeqCst) {
                    // Scoped for the reason given at the same import in
                    // `refuse_restore_folds_a_transient_header_read_failure_into_its_own_refusal`.
                    use fred::interfaces::ClientLike;
                    client
                        .pool
                        .quit()
                        .await
                        .expect("closing the hook's own data pool must succeed");
                }
            }
        }
    }

    #[async_trait]
    impl CheckpointStore for FakeCheckpointStore {
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
            match &self.fetch {
                LayerFetch::Delegate => self.inner.get_latest_layer_checkpoint(object_id).await,
                LayerFetch::DelegateAfterClosingPool { .. } => {
                    self.close_pool_once().await;
                    self.inner.get_latest_layer_checkpoint(object_id).await
                }
                LayerFetch::Fail(mint) => Err(mint()),
                LayerFetch::Stall => {
                    tokio::time::sleep(Self::STALL).await;
                    Ok(None)
                }
            }
        }

        /// On the SR-2.5 selection path this is the *first* store call the
        /// hook makes, so it carries the scripted posture — see
        /// [`LayerFetch`]'s "It governs all three layer-read methods".
        async fn list_layer_checkpoints(
            &self,
            object_id: u64,
        ) -> Result<Vec<CheckpointHandle>, CheckpointError> {
            match &self.fetch {
                LayerFetch::Delegate => self.inner.list_layer_checkpoints(object_id).await,
                LayerFetch::DelegateAfterClosingPool { .. } => {
                    self.close_pool_once().await;
                    self.inner.list_layer_checkpoints(object_id).await
                }
                LayerFetch::Fail(mint) => Err(mint()),
                LayerFetch::Stall => {
                    tokio::time::sleep(Self::STALL).await;
                    Ok(Vec::new())
                }
            }
        }

        /// The second half of the SR-2.5 selection path. Carries the same
        /// posture as the listing so a fixture cannot answer half the
        /// selection honestly and half of it not.
        async fn get_layer_checkpoint(
            &self,
            handle: &CheckpointHandle,
        ) -> Result<Option<CheckpointBlob>, CheckpointError> {
            match &self.fetch {
                LayerFetch::Delegate => self.inner.get_layer_checkpoint(handle).await,
                LayerFetch::DelegateAfterClosingPool { .. } => {
                    self.close_pool_once().await;
                    self.inner.get_layer_checkpoint(handle).await
                }
                LayerFetch::Fail(mint) => Err(mint()),
                LayerFetch::Stall => {
                    tokio::time::sleep(Self::STALL).await;
                    Ok(None)
                }
            }
        }

        async fn delete_layer_checkpoint(
            &self,
            handle: CheckpointHandle,
        ) -> Result<(), CheckpointError> {
            self.inner.delete_layer_checkpoint(handle).await
        }

        async fn put_session_checkpoint(
            &self,
            _session_id: &str,
            _body: Bytes,
            _meta: CheckpointMeta,
        ) -> Result<CheckpointHandle, CheckpointError> {
            panic!(
                "Epic-1 invariant violated: try_restore_layer MUST NOT invoke \
                 put_session_checkpoint; the session-checkpoint surface belongs to \
                 SessionRecoveryHook"
            );
        }

        async fn get_session_checkpoint(
            &self,
            _session_id: &str,
        ) -> Result<Option<CheckpointBlob>, CheckpointError> {
            panic!(
                "Epic-1 invariant violated: try_restore_layer MUST NOT invoke \
                 get_session_checkpoint; the session-checkpoint surface belongs to \
                 SessionRecoveryHook"
            );
        }

        async fn delete_session_checkpoint(
            &self,
            _session_id: &str,
        ) -> Result<(), CheckpointError> {
            panic!(
                "Epic-1 invariant violated: try_restore_layer MUST NOT invoke \
                 delete_session_checkpoint; the session-checkpoint surface belongs to \
                 SessionRecoveryHook"
            );
        }

        async fn list_session_checkpoint_ids(&self) -> Result<Vec<String>, CheckpointError> {
            panic!(
                "Epic-1 invariant violated: try_restore_layer MUST NOT invoke \
                 list_session_checkpoint_ids; the session-checkpoint surface belongs to \
                 SessionRecoveryHook"
            );
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

    /// Disabled short-circuit: `try_restore_layer` MUST return
    /// `NotRecoverable` immediately, with zero side effects on the
    /// admission map, the store, and the metric.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn try_restore_layer_disabled_short_circuits() {
        let tv = TestValkey::setup().await;
        let layer_store = Arc::new(LayerStore::new(
            tv.client().clone(),
            tv.key_schema().clone(),
            4096,
        ));
        let store: Arc<dyn CheckpointStore> = Arc::new(InMemoryCheckpointStore::new());
        let metrics = make_metrics();
        let hook = RecoveryHook::new(
            store,
            layer_store,
            Arc::new(test_config(false)),
            metrics.clone(),
        );

        let outcome = hook.try_restore_layer(123).await;
        assert!(
            matches!(outcome, RecoveryOutcome::NotRecoverable),
            "disabled hook MUST return NotRecoverable, got {outcome:?}"
        );

        assert_eq!(
            hook.admission_len(),
            0,
            "disabled hook MUST NOT touch the admission map"
        );
        assert_eq!(
            metric_value(&metrics, "restored"),
            0,
            "disabled hook MUST NOT increment the metric"
        );
        assert_eq!(metric_value(&metrics, "not_recoverable"), 0);
        assert_eq!(metric_value(&metrics, "error"), 0);
    }

    /// `Restored` arm: a checkpoint exists, a live layer-mapping points at
    /// the target `object_id`, and `apply_checkpoint` succeeds. The hook
    /// MUST return `Restored` and bump the metric.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn try_restore_layer_restored_arm() {
        let tv = TestValkey::setup().await;
        let layer_store = Arc::new(LayerStore::new(
            tv.client().clone(),
            tv.key_schema().clone(),
            4096,
        ));

        // Install a live layer mapping so apply's SR-2.4 guard passes.
        let oid: u64 = 4242;
        let mapping_key = tv.key_schema().layer_mapping("s1", "l");
        let _: () = tv
            .client()
            .pool
            .set(&mapping_key, oid.to_string(), None, None, false)
            .await
            .expect("SET mapping");

        // Seed the in-memory checkpoint store.
        let in_mem = Arc::new(InMemoryCheckpointStore::new());
        let meta = CheckpointMeta {
            creator_replica_id: "test-replica".to_string(),
            written_at_unix_ms: 1_700_000_000_000,
            snap_seq: Some(7),
            session_id: "s1".to_string(),
            layer_name: "l".to_string(),
            created_by: "u".to_string(),
            server_ts: 123,
            ..CheckpointMeta::default()
        };
        in_mem
            .put_layer_checkpoint(oid, 7, build_empty_reset_ompf(), meta)
            .await
            .expect("put checkpoint");

        let store: Arc<dyn CheckpointStore> = in_mem;
        let metrics = make_metrics();
        let hook = RecoveryHook::new(
            store,
            layer_store,
            Arc::new(test_config(true)),
            metrics.clone(),
        );

        let outcome = hook.try_restore_layer(oid).await;
        assert!(
            matches!(outcome, RecoveryOutcome::Restored),
            "restored arm MUST return Restored, got {outcome:?}"
        );
        assert_eq!(
            metric_value(&metrics, "restored"),
            1,
            "Restored arm MUST bump the `restored` metric cell"
        );
        assert_eq!(metric_value(&metrics, "not_recoverable"), 0);
        assert_eq!(metric_value(&metrics, "error"), 0);

        // Admission map entry MUST be retired on the last drop.
        assert_eq!(
            hook.admission_len(),
            0,
            "admission map entry MUST be retired on the last waiter's drop"
        );

        // The header MUST now exist for the object.
        let header_key = tv.key_schema().obj_header(&oid.to_string());
        let exists: bool = tv
            .client()
            .pool
            .exists(&header_key)
            .await
            .expect("EXISTS obj:hdr");
        assert!(exists, "obj:hdr MUST be present after a Restored outcome");
    }

    /// Shared fixture for the SR-3.6 re-base tests: a live Valkey, a live
    /// mapping, one seeded base generation at `snap_seq`, and a hook wired with
    /// a real `BaseBootstrap` whose three-way gate is resolved by the
    /// caller.
    ///
    /// Returns the pieces each test asserts against, including the tracker the
    /// re-base task lands on — draining it is what makes an
    /// otherwise-fire-and-forget task deterministically observable.
    struct RebaseFixture {
        tv: TestValkey,
        store: Arc<InMemoryCheckpointStore>,
        layer_store: Arc<LayerStore>,
        hook: RecoveryHook,
        tracker: tokio_util::task::TaskTracker,
        oid: u64,
        snap_seq: u64,
    }

    async fn rebase_fixture(bootstrap_enabled: bool) -> RebaseFixture {
        let tv = TestValkey::setup().await;
        let layer_store = Arc::new(LayerStore::new(
            tv.client().clone(),
            tv.key_schema().clone(),
            4096,
        ));

        let oid: u64 = 5150;
        let snap_seq: u64 = 7;
        let mapping_key = tv.key_schema().layer_mapping("s1", "l");
        let _: () = tv
            .client()
            .pool
            .set(&mapping_key, oid.to_string(), None, None, false)
            .await
            .expect("SET mapping");

        let in_mem = Arc::new(InMemoryCheckpointStore::new());
        let meta = CheckpointMeta {
            creator_replica_id: "test-replica".to_string(),
            written_at_unix_ms: 1_700_000_000_000,
            snap_seq: Some(snap_seq),
            session_id: "s1".to_string(),
            layer_name: "l".to_string(),
            created_by: "u".to_string(),
            server_ts: 123,
            ..CheckpointMeta::default()
        };
        in_mem
            .put_layer_checkpoint(oid, snap_seq, build_empty_reset_ompf(), meta)
            .await
            .expect("put checkpoint");

        let metrics = make_metrics();
        let tracker = tokio_util::task::TaskTracker::new();
        let bootstrap = Arc::new(BaseBootstrap::new(
            in_mem.clone() as Arc<dyn CheckpointStore>,
            tv.client().clone(),
            tv.key_schema().clone(),
            tracker.clone(),
            metrics.clone(),
            "test-replica".to_string(),
            bootstrap_enabled,
        ));

        let hook = RecoveryHook::new(
            in_mem.clone() as Arc<dyn CheckpointStore>,
            Arc::clone(&layer_store),
            Arc::new(test_config(true)),
            metrics,
        )
        .with_base_bootstrap(bootstrap);

        RebaseFixture {
            tv,
            store: in_mem,
            layer_store,
            hook,
            tracker,
            oid,
            snap_seq,
        }
    }

    /// SR-3.6, the Epic's acceptance behaviour: a successful restore leaves a
    /// `Base` generation AT `seeded_seq` and an `obj:hdr` carrying all THREE
    /// cursor columns — `last_checkpoint_seq = seeded_seq`, a fresh
    /// `last_checkpoint_ms`, and `restore_seq_bump = 0`.
    ///
    /// The third column is the load-bearing one: `tail_capture` refuses with
    /// `PostRestoreUnrebased` while the bump is non-zero, *before* it reads the
    /// stream, so without this the restored layer refuses on every tick
    /// regardless of stream survival.
    ///
    /// This is a **base-only** lineage on purpose. `band >= 1` unconditionally,
    /// so even here `seeded_seq > snap_seq` and the re-base must run — SR-3.6's
    /// old conditional no-gap arm is dead and no `seeded_seq > snap_seq` test
    /// exists in the code.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn restore_spawns_a_gap_rebase_that_bases_at_seeded_seq_and_retires_the_bump() {
        use fred::interfaces::HashesInterface;

        let fx = rebase_fixture(true).await;

        let outcome = fx.hook.try_restore_layer(fx.oid).await;
        assert!(
            matches!(outcome, RecoveryOutcome::Restored),
            "restored arm MUST return Restored, got {outcome:?}"
        );

        // The re-base is a best-effort tracker task the hook never awaits;
        // draining the tracker is what makes it observable here.
        fx.tracker.close();
        fx.tracker.wait().await;

        // `band = max(checkpoint_seq_safety_margin, observed_commit_span)` with
        // no stream to probe, so `restore_floor == snap_seq` and the seeded
        // cursor is exactly one margin above the generation restored.
        let expected_seeded = fx.snap_seq + test_config(true).checkpoint_seq_safety_margin;

        let handles = fx
            .store
            .list_layer_checkpoints(fx.oid)
            .await
            .expect("list generations");
        assert!(
            handles
                .iter()
                .any(|h| h.snap_seq == expected_seeded && h.kind == CheckpointKind::Base),
            "a Base generation MUST exist at seeded_seq = {expected_seeded}; got {handles:?}"
        );

        let header_key = fx.tv.key_schema().obj_header(&fx.oid.to_string());
        let cursor: Vec<Option<String>> = fx
            .tv
            .client()
            .pool
            .hmget(
                &header_key,
                vec![
                    "seq",
                    "last_checkpoint_seq",
                    "last_checkpoint_ms",
                    "restore_seq_bump",
                ],
            )
            .await
            .expect("HMGET obj:hdr cursor columns");

        assert_eq!(
            cursor[0].as_deref(),
            Some(expected_seeded.to_string().as_str()),
            "the restore MUST have seeded the writer cursor at restore_floor + band"
        );
        assert_eq!(
            cursor[1].as_deref(),
            Some(expected_seeded.to_string().as_str()),
            "the re-base MUST advance last_checkpoint_seq to the seeded cursor, closing the gap"
        );
        assert!(
            cursor[2]
                .as_deref()
                .and_then(|ms| ms.parse::<u64>().ok())
                .is_some_and(|ms| ms > 0),
            "the re-base MUST stamp a fresh last_checkpoint_ms, got {:?}",
            cursor[2]
        );
        assert_eq!(
            cursor[3].as_deref(),
            Some("0"),
            "the re-base MUST RETIRE restore_seq_bump — that is what unblocks tail capture"
        );
    }

    /// The two `Restored` arms that did NOT apply anything must not re-base.
    ///
    /// Driven through the step-4 `is_layer_present` short-circuit: the second
    /// hook finds the structurally-complete header the first restore wrote and
    /// returns `Restored` without composing. Re-basing there would mint a base
    /// for a state that call never held. (`apply_restore`'s own step-1
    /// short-circuit reports `AlreadyPresent`, excluded by the same `match` arm
    /// structure.)
    ///
    /// The second hook's bootstrap writes to a **separate, empty** checkpoint
    /// store, so "no re-base was spawned" is proved by that store still being
    /// empty after its tracker is drained — no timing assumption, and no
    /// dependence on the re-base PUT being idempotent at the same key.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_restore_that_applied_nothing_spawns_no_rebase() {
        let fx = rebase_fixture(true).await;

        assert!(matches!(
            fx.hook.try_restore_layer(fx.oid).await,
            RecoveryOutcome::Restored
        ));
        fx.tracker.close();
        fx.tracker.wait().await;

        let probe_store = Arc::new(InMemoryCheckpointStore::new());
        let probe_tracker = tokio_util::task::TaskTracker::new();
        let metrics = make_metrics();
        let probe_bootstrap = Arc::new(BaseBootstrap::new(
            probe_store.clone() as Arc<dyn CheckpointStore>,
            fx.tv.client().clone(),
            fx.tv.key_schema().clone(),
            probe_tracker.clone(),
            metrics.clone(),
            "test-replica".to_string(),
            true,
        ));
        let second_hook = RecoveryHook::new(
            fx.store.clone() as Arc<dyn CheckpointStore>,
            Arc::clone(&fx.layer_store),
            Arc::new(test_config(true)),
            metrics,
        )
        .with_base_bootstrap(probe_bootstrap);

        assert!(matches!(
            second_hook.try_restore_layer(fx.oid).await,
            RecoveryOutcome::Restored
        ));
        probe_tracker.close();
        probe_tracker.wait().await;

        assert!(
            probe_store
                .list_layer_checkpoints(fx.oid)
                .await
                .expect("list generations")
                .is_empty(),
            "a Restored outcome that applied nothing MUST NOT spawn a re-base"
        );
    }

    /// Flag-off posture: with the unit's three-way gate resolved `false` (a
    /// `LIVE_EDIT_CHECKPOINT_TAIL_ENABLED=false` deployment), a restore spawns
    /// no task and leaves the header exactly as `write_restored_object` wrote it — the
    /// restore's own audit stamp at `snap_seq` with its `restore_seq_bump`
    /// intact.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn tail_disabled_restore_spawns_no_rebase_and_leaves_the_cursor_alone() {
        use fred::interfaces::HashesInterface;

        let fx = rebase_fixture(false).await;

        assert!(matches!(
            fx.hook.try_restore_layer(fx.oid).await,
            RecoveryOutcome::Restored
        ));
        fx.tracker.close();
        fx.tracker.wait().await;

        let handles = fx
            .store
            .list_layer_checkpoints(fx.oid)
            .await
            .expect("list generations");
        assert_eq!(
            handles.len(),
            1,
            "a gated-off bootstrap MUST NOT write a generation; got {handles:?}"
        );

        let header_key = fx.tv.key_schema().obj_header(&fx.oid.to_string());
        let cursor: Vec<Option<String>> = fx
            .tv
            .client()
            .pool
            .hmget(&header_key, vec!["last_checkpoint_seq", "restore_seq_bump"])
            .await
            .expect("HMGET obj:hdr cursor columns");
        assert_eq!(
            cursor[0].as_deref(),
            Some(fx.snap_seq.to_string().as_str()),
            "the restore's own audit stamp MUST be untouched when the bootstrap is gated off"
        );
        assert_eq!(
            cursor[1].as_deref(),
            Some(
                test_config(true)
                    .checkpoint_seq_safety_margin
                    .to_string()
                    .as_str()
            ),
            "restore_seq_bump MUST survive when no re-base runs — capture refuses observably"
        );
    }

    /// `NotRecoverable` arm — no checkpoint exists. The restore-source
    /// selection sees an empty listing and returns `Ok(None)`; the hook
    /// reports `NotRecoverable` and bumps the `not_recoverable` metric cell.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn try_restore_layer_not_recoverable_on_no_checkpoint() {
        let tv = TestValkey::setup().await;
        let layer_store = Arc::new(LayerStore::new(
            tv.client().clone(),
            tv.key_schema().clone(),
            4096,
        ));
        let store: Arc<dyn CheckpointStore> = Arc::new(InMemoryCheckpointStore::new());
        let metrics = make_metrics();
        let hook = RecoveryHook::new(
            store,
            layer_store,
            Arc::new(test_config(true)),
            metrics.clone(),
        );

        let outcome = hook.try_restore_layer(9999).await;
        assert!(
            matches!(outcome, RecoveryOutcome::NotRecoverable),
            "NotRecoverable arm (no checkpoint) MUST return NotRecoverable, got {outcome:?}"
        );
        assert_eq!(metric_value(&metrics, "not_recoverable"), 1);
        assert_eq!(metric_value(&metrics, "restored"), 0);
        assert_eq!(metric_value(&metrics, "error"), 0);
        assert_eq!(hook.admission_len(), 0);
    }

    /// `NotRecoverable` arm — apply's SR-2.4 live-mapping guard rejects.
    /// A checkpoint exists, but no live `le:{sid}:layer:{hex(name)}`
    /// mapping resolves to the object_id (DeleteLayer-checkpoint-delete
    /// race or CreateLayer-overwrite during state-loss). `apply_checkpoint`
    /// returns `CheckpointError::NotFound`; the hook MUST surface
    /// `NotRecoverable` and bump the `not_recoverable` cell.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn try_restore_layer_not_recoverable_on_apply_guard_reject() {
        let tv = TestValkey::setup().await;
        let layer_store = Arc::new(LayerStore::new(
            tv.client().clone(),
            tv.key_schema().clone(),
            4096,
        ));

        // Seed checkpoint but DO NOT install any layer-mapping. The
        // SR-2.4 guard in `apply_checkpoint` will reject with NotFound.
        let oid: u64 = 5555;
        let in_mem = Arc::new(InMemoryCheckpointStore::new());
        let meta = CheckpointMeta {
            creator_replica_id: "test-replica".to_string(),
            written_at_unix_ms: 1_700_000_000_000,
            snap_seq: Some(3),
            session_id: "s1".to_string(),
            layer_name: "l".to_string(),
            created_by: "u".to_string(),
            server_ts: 123,
            ..CheckpointMeta::default()
        };
        in_mem
            .put_layer_checkpoint(oid, 3, build_empty_reset_ompf(), meta)
            .await
            .expect("put checkpoint");

        let store: Arc<dyn CheckpointStore> = in_mem;
        let metrics = make_metrics();
        let hook = RecoveryHook::new(
            store,
            layer_store,
            Arc::new(test_config(true)),
            metrics.clone(),
        );

        let outcome = hook.try_restore_layer(oid).await;
        assert!(
            matches!(outcome, RecoveryOutcome::NotRecoverable),
            "guard-reject arm MUST return NotRecoverable, got {outcome:?}"
        );
        assert_eq!(metric_value(&metrics, "not_recoverable"), 1);
        assert_eq!(metric_value(&metrics, "restored"), 0);
        assert_eq!(metric_value(&metrics, "error"), 0);

        // Header MUST NOT have been written.
        let header_key = tv.key_schema().obj_header(&oid.to_string());
        let exists: bool = tv
            .client()
            .pool
            .exists(&header_key)
            .await
            .expect("EXISTS obj:hdr");
        assert!(
            !exists,
            "guard-reject arm MUST NOT write obj:hdr; got header present"
        );
    }

    /// Counting wrapper around `InMemoryCheckpointStore` — records every
    /// trait-method invocation so the disabled-short-circuit test can
    /// prove the hook performs ZERO store calls when `enabled=false`.
    /// The PM patch on this Subtask explicitly asks for this stronger
    /// proof on top of the existing `try_restore_layer_disabled_short_circuits`
    /// test (which only asserts admission map + metric stay untouched).
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

        async fn list_layer_checkpoints(
            &self,
            object_id: u64,
        ) -> Result<Vec<CheckpointHandle>, CheckpointError> {
            self.record("list_layer_checkpoints");
            self.inner.list_layer_checkpoints(object_id).await
        }

        /// Forwarded (and recorded) rather than left on the trait's
        /// `NotImplemented` default: since SR-2.5 this is half of the
        /// recovery path's restore-source selection, so inheriting the default
        /// would turn every enabled-mode recovery through this wrapper into
        /// `Error(NotImplemented)`.
        async fn get_layer_checkpoint(
            &self,
            handle: &CheckpointHandle,
        ) -> Result<Option<CheckpointBlob>, CheckpointError> {
            self.record("get_layer_checkpoint");
            self.inner.get_layer_checkpoint(handle).await
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

    /// `Error(Transient)` arm — the store returns a classified `Transient`
    /// error from the restore-source selection. The hook MUST surface
    /// `Error(Transient)` verbatim and bump the `error` metric cell.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn try_restore_layer_error_on_transient_store_failure() {
        let tv = TestValkey::setup().await;
        let layer_store = Arc::new(LayerStore::new(
            tv.client().clone(),
            tv.key_schema().clone(),
            4096,
        ));
        let store: Arc<dyn CheckpointStore> = Arc::new(FakeCheckpointStore::always_transient());
        let metrics = make_metrics();
        let hook = RecoveryHook::new(
            store,
            layer_store,
            Arc::new(test_config(true)),
            metrics.clone(),
        );

        let outcome = hook.try_restore_layer(1).await;
        match outcome {
            RecoveryOutcome::Error(CheckpointError::Transient { .. }) => {}
            other => panic!("expected Error(Transient); got {other:?}"),
        }
        assert_eq!(metric_value(&metrics, "error"), 1);
        assert_eq!(metric_value(&metrics, "restored"), 0);
        assert_eq!(metric_value(&metrics, "not_recoverable"), 0);
    }

    /// Admission discipline — concurrent `try_restore_layer` calls on the
    /// same `object_id` MUST serialize through the K=1 mutex. Only one
    /// `apply_checkpoint` MULTI/EXEC may run; the other waiters MUST
    /// observe the restored header via the post-acquire structural-header re-check
    /// (or via `apply_checkpoint`'s own step-1 short-circuit) and return
    /// `Restored` without re-running the full pipeline.
    ///
    /// We assert this indirectly by observing that the admission map
    /// retires the entry after every concurrent attempt drops out — if
    /// the K=1 mutex were broken or the refcount semantics were off, the
    /// map would either leak entries or remove the entry before the last
    /// waiter had a chance to drop, depending on the failure mode.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn concurrent_restorers_serialize_and_retire_admission_entry() {
        let tv = TestValkey::setup().await;
        let layer_store = Arc::new(LayerStore::new(
            tv.client().clone(),
            tv.key_schema().clone(),
            4096,
        ));

        // Install a live layer mapping so the apply guard passes.
        let oid: u64 = 8888;
        let mapping_key = tv.key_schema().layer_mapping("s1", "l");
        let _: () = tv
            .client()
            .pool
            .set(&mapping_key, oid.to_string(), None, None, false)
            .await
            .expect("SET mapping");

        // Seed a single checkpoint.
        let in_mem = Arc::new(InMemoryCheckpointStore::new());
        let meta = CheckpointMeta {
            creator_replica_id: "test-replica".to_string(),
            written_at_unix_ms: 1_700_000_000_000,
            snap_seq: Some(11),
            session_id: "s1".to_string(),
            layer_name: "l".to_string(),
            created_by: "u".to_string(),
            server_ts: 123,
            ..CheckpointMeta::default()
        };
        in_mem
            .put_layer_checkpoint(oid, 11, build_empty_reset_ompf(), meta)
            .await
            .expect("put checkpoint");

        let store: Arc<dyn CheckpointStore> = in_mem;
        let metrics = make_metrics();
        let hook = Arc::new(RecoveryHook::new(
            store,
            layer_store,
            Arc::new(test_config(true)),
            metrics.clone(),
        ));

        // Fire 8 concurrent restorers on the same object_id.
        let mut handles = Vec::new();
        let outcomes = Arc::new(StdMutex::new(Vec::new()));
        for _ in 0..8 {
            let hook = Arc::clone(&hook);
            let outcomes = Arc::clone(&outcomes);
            handles.push(tokio::spawn(async move {
                let r = hook.try_restore_layer(oid).await;
                outcomes
                    .lock()
                    .unwrap()
                    .push(matches!(r, RecoveryOutcome::Restored));
            }));
        }
        for h in handles {
            h.await.expect("join");
        }

        let outcomes = outcomes.lock().unwrap();
        assert_eq!(outcomes.len(), 8);
        assert!(
            outcomes.iter().all(|r| *r),
            "every concurrent restorer MUST observe Restored (winner via apply, losers via re-check); got {outcomes:?}"
        );

        // Map entry MUST be retired after the last waiter drops out.
        assert_eq!(
            hook.admission_len(),
            0,
            "admission map entry MUST be retired on the last waiter's drop"
        );

        // All 8 outcomes count toward the `restored` cell.
        assert_eq!(
            metric_value(&metrics, "restored"),
            8,
            "every restorer (winner + losers) MUST bump the `restored` cell"
        );
    }

    /// `Error(Permanent)` arm — the store returns a classified `Permanent`
    /// error from the restore-source selection. Same outcome-mapping shape as
    /// `try_restore_layer_error_on_transient_store_failure`, but flowing
    /// through with `Permanent` to pin both classified-error variants on
    /// the same `Error(_)` fall-through arm. Without this case, a
    /// regression that special-cased only `Transient` (and accidentally
    /// dropped `Permanent` into `NotRecoverable`) would slip past the
    /// existing test suite.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn try_restore_layer_permanent_adapter_error_returns_error() {
        let tv = TestValkey::setup().await;
        let layer_store = Arc::new(LayerStore::new(
            tv.client().clone(),
            tv.key_schema().clone(),
            4096,
        ));
        let store: Arc<dyn CheckpointStore> = Arc::new(FakeCheckpointStore::always_permanent());
        let metrics = make_metrics();
        let hook = RecoveryHook::new(
            store,
            layer_store,
            Arc::new(test_config(true)),
            metrics.clone(),
        );

        let outcome = hook.try_restore_layer(1).await;
        match outcome {
            RecoveryOutcome::Error(CheckpointError::Permanent { .. }) => {}
            other => panic!("expected Error(Permanent); got {other:?}"),
        }
        assert_eq!(metric_value(&metrics, "error"), 1);
        assert_eq!(metric_value(&metrics, "restored"), 0);
        assert_eq!(metric_value(&metrics, "not_recoverable"), 0);
    }

    /// Stronger disabled short-circuit — uses a counting `CheckpointStore`
    /// wrapper to prove `try_restore_layer` performs ZERO method calls
    /// against the underlying store when `enabled=false`, in addition to
    /// the existing assertions that the admission map and the recovery-
    /// attempts metric remain untouched. This pins the unit-level Step 1
    /// short-circuit from the engineer Subtask: an operator who disables
    /// checkpointing MUST observe no recovery-side activity at all (no
    /// adapter round-trips, no admission-map churn, no metric ticks).
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn try_restore_layer_disabled_returns_not_recoverable_without_touching_store() {
        let tv = TestValkey::setup().await;
        let layer_store = Arc::new(LayerStore::new(
            tv.client().clone(),
            tv.key_schema().clone(),
            4096,
        ));
        let counting = Arc::new(CountingCheckpointStore::new());
        let store: Arc<dyn CheckpointStore> = counting.clone();
        let metrics = make_metrics();
        let hook = RecoveryHook::new(
            store,
            layer_store,
            Arc::new(test_config(false)),
            metrics.clone(),
        );

        let outcome = hook.try_restore_layer(777).await;
        assert!(
            matches!(outcome, RecoveryOutcome::NotRecoverable),
            "disabled hook MUST return NotRecoverable, got {outcome:?}"
        );

        // Stronger than the existing disabled test: ZERO store method calls
        // recorded — the hook MUST short-circuit before any adapter
        // round-trip.
        assert_eq!(
            counting.call_count(),
            0,
            "disabled hook MUST NOT invoke any CheckpointStore method; recorded calls = {:?}",
            counting.calls.lock().unwrap()
        );

        // And the admission map MUST stay empty, and the metric MUST stay
        // at zero across every label cell — same assertions as the
        // existing test, retained here so this case stands alone as the
        // full short-circuit proof.
        assert_eq!(
            hook.admission_len(),
            0,
            "disabled hook MUST NOT touch the admission map"
        );
        assert_eq!(metric_value(&metrics, "restored"), 0);
        assert_eq!(metric_value(&metrics, "not_recoverable"), 0);
        assert_eq!(metric_value(&metrics, "error"), 0);
    }

    /// Metrics increment per outcome — drive all three `Result` arms
    /// (`restored`, `not_recoverable`, `error`) through a single hook and
    /// assert each label cell increments exactly once. This is the
    /// load-bearing operator-signal check for SR-2.4: a regression where
    /// two arms shared a label (or one arm silently dropped its metric
    /// inc) would falsify `live_edit_checkpoint_recovery_attempts_total`
    /// rate alerts. Exercising all three arms back-to-back in one test
    /// also pins that the metric is dispatched on the correct branch per
    /// outcome — not, for example, always under `restored`.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn try_restore_layer_metrics_increment_per_outcome() {
        let tv = TestValkey::setup().await;
        let layer_store = Arc::new(LayerStore::new(
            tv.client().clone(),
            tv.key_schema().clone(),
            4096,
        ));

        // ── Arm 1: Restored. Seed a checkpoint + live layer mapping for
        // `oid_a`; the hook MUST land Restored and bump the `restored`
        // cell. We compose a hook per arm so the underlying store can be
        // tailored to the arm's required failure mode (Restored needs an
        // InMemory store with a seeded entry; Error needs the transient
        // fault-injection store). The metric Registry is shared across
        // arms because the assertions read live cell values, not deltas.
        let metrics = make_metrics();

        let oid_a: u64 = 1111;
        let mapping_key_a = tv.key_schema().layer_mapping("s1", "l1");
        let _: () = tv
            .client()
            .pool
            .set(&mapping_key_a, oid_a.to_string(), None, None, false)
            .await
            .expect("SET mapping for arm 1");
        let in_mem = Arc::new(InMemoryCheckpointStore::new());
        let meta = CheckpointMeta {
            creator_replica_id: "test-replica".to_string(),
            written_at_unix_ms: 1_700_000_000_000,
            snap_seq: Some(2),
            session_id: "s1".to_string(),
            layer_name: "l1".to_string(),
            created_by: "u".to_string(),
            server_ts: 123,
            ..CheckpointMeta::default()
        };
        in_mem
            .put_layer_checkpoint(oid_a, 2, build_empty_reset_ompf(), meta)
            .await
            .expect("put checkpoint for arm 1");
        let hook_restored = RecoveryHook::new(
            in_mem as Arc<dyn CheckpointStore>,
            layer_store.clone(),
            Arc::new(test_config(true)),
            metrics.clone(),
        );
        let outcome = hook_restored.try_restore_layer(oid_a).await;
        assert!(
            matches!(outcome, RecoveryOutcome::Restored),
            "arm 1 (Restored) MUST yield Restored; got {outcome:?}"
        );

        // ── Arm 2: NotRecoverable. Empty in-memory store on a fresh
        // `oid_b` — the restore-source selection lists nothing and returns
        // `Ok(None)`.
        let oid_b: u64 = 2222;
        let empty_store: Arc<dyn CheckpointStore> = Arc::new(InMemoryCheckpointStore::new());
        let hook_not_recoverable = RecoveryHook::new(
            empty_store,
            layer_store.clone(),
            Arc::new(test_config(true)),
            metrics.clone(),
        );
        let outcome = hook_not_recoverable.try_restore_layer(oid_b).await;
        assert!(
            matches!(outcome, RecoveryOutcome::NotRecoverable),
            "arm 2 (NotRecoverable) MUST yield NotRecoverable; got {outcome:?}"
        );

        // ── Arm 3: Error. Transient-only store on a fresh `oid_c`.
        let oid_c: u64 = 3333;
        let transient_store: Arc<dyn CheckpointStore> =
            Arc::new(FakeCheckpointStore::always_transient());
        let hook_error = RecoveryHook::new(
            transient_store,
            layer_store,
            Arc::new(test_config(true)),
            metrics.clone(),
        );
        let outcome = hook_error.try_restore_layer(oid_c).await;
        assert!(
            matches!(outcome, RecoveryOutcome::Error(_)),
            "arm 3 (Error) MUST yield Error; got {outcome:?}"
        );

        // Cross-arm verification: each label cell incremented exactly
        // once across the three attempts. A regression that mis-routed
        // any arm's metric inc would show up as either a missing 1 or a
        // spurious 2 here.
        assert_eq!(
            metric_value(&metrics, "restored"),
            1,
            "`restored` cell MUST be 1 (incremented exactly once across the three arms)"
        );
        assert_eq!(
            metric_value(&metrics, "not_recoverable"),
            1,
            "`not_recoverable` cell MUST be 1"
        );
        assert_eq!(metric_value(&metrics, "error"), 1, "`error` cell MUST be 1");
    }

    /// Epic-1 boundary: `try_restore_layer` MUST NOT touch the Epic-2
    /// session-checkpoint surface (`put_session_checkpoint` /
    /// `get_session_checkpoint` / `delete_session_checkpoint`). A
    /// regression that broadened recovery to also restore session state
    /// would silently invoke those methods on the configured store —
    /// and Epic 2 has not yet wired session-state apply, so any such
    /// invocation would be an unspecified leak across Epics.
    /// [`FakeCheckpointStore`]'s session methods panic, which turns that
    /// quiet boundary crossing into a load-bearing test failure.
    ///
    /// That guard now rides on **every** test in this module, so this one
    /// is no longer the only tripwire — it is the case that walks the
    /// longest distance past it. It drives `try_restore_layer` through
    /// both the `Restored` arm (a checkpoint exists + a live mapping
    /// resolves, so the full apply pipeline runs) AND the
    /// `NotRecoverable` arm (no checkpoint exists, the post-acquire
    /// structural-header re-check sees no header) — the two branches most likely to
    /// attract a future "while we're here, also probe the
    /// session-checkpoint surface" change, and the two that the
    /// error-arm tests never reach because they bail at step 5.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn try_restore_layer_never_invokes_session_checkpoint_methods() {
        let tv = TestValkey::setup().await;
        let layer_store = Arc::new(LayerStore::new(
            tv.client().clone(),
            tv.key_schema().clone(),
            4096,
        ));

        // Arm A: drive the `Restored` path. Seed a checkpoint + live
        // mapping for `oid_restored` so apply lands successfully.
        let oid_restored: u64 = 7777;
        let mapping_key = tv.key_schema().layer_mapping("s1", "l");
        let _: () = tv
            .client()
            .pool
            .set(&mapping_key, oid_restored.to_string(), None, None, false)
            .await
            .expect("SET mapping");

        let seeded_store = Arc::new(FakeCheckpointStore::delegating());
        let meta = CheckpointMeta {
            creator_replica_id: "test-replica".to_string(),
            written_at_unix_ms: 1_700_000_000_000,
            snap_seq: Some(4),
            session_id: "s1".to_string(),
            layer_name: "l".to_string(),
            created_by: "u".to_string(),
            server_ts: 123,
            ..CheckpointMeta::default()
        };
        seeded_store
            .put_layer_checkpoint(oid_restored, 4, build_empty_reset_ompf(), meta)
            .await
            .expect("put checkpoint must succeed");

        let store: Arc<dyn CheckpointStore> = seeded_store;
        let metrics = make_metrics();
        let hook = RecoveryHook::new(
            store,
            layer_store,
            Arc::new(test_config(true)),
            metrics.clone(),
        );

        let outcome = hook.try_restore_layer(oid_restored).await;
        assert!(
            matches!(outcome, RecoveryOutcome::Restored),
            "Arm A (Restored) MUST yield Restored; got {outcome:?}"
        );

        // Arm B: drive the `NotRecoverable` path. A fresh object_id
        // with no seeded checkpoint exercises step 4's `Ok(None)` arm,
        // which also must not touch session methods. The hook still
        // runs through the post-acquire structural-header re-check and the
        // restore-source selection — every code path on the
        // happy + sad arms is exercised. If any of those code paths
        // gained a session-method invocation, this test would panic.
        let oid_no_checkpoint: u64 = 7778;
        let outcome2 = hook.try_restore_layer(oid_no_checkpoint).await;
        assert!(
            matches!(outcome2, RecoveryOutcome::NotRecoverable),
            "Arm B (NotRecoverable) MUST yield NotRecoverable; got {outcome2:?}"
        );

        // Sanity: the metric increments for both arms confirm the
        // hook actually executed the full body — a regression that
        // accidentally short-circuited before the store call would
        // leave the metrics at zero and falsely "pass" the no-panic
        // assertion (vacuous test). The metric is the proxy for "the
        // hook DID run; it just happened not to touch session
        // methods".
        assert_eq!(metric_value(&metrics, "restored"), 1);
        assert_eq!(metric_value(&metrics, "not_recoverable"), 1);
        assert_eq!(metric_value(&metrics, "error"), 0);
    }

    /// Lifecycle cleanup on `Error(_)`: a recovery attempt that
    /// surfaces `RecoveryOutcome::Error(_)` MUST still retire its
    /// admission map entry on drop. Mirrors the b.3gx / b.xet
    /// RAII-guard discipline on `LayerStore::semaphores` — error
    /// branches share the same `AdmissionMapGuard` drop path as the
    /// success branches, but a regression that early-returned the
    /// `Error(_)` arm via a separate code path (e.g. skipping the
    /// RAII guard's drop) would silently leak admission entries on
    /// every transient checkpoint-backend failure.
    ///
    /// Under sustained transient failures (the canonical scenario
    /// the cleanup invariant protects against) this would
    /// monotonically grow the DashMap until OOM. The exact-equals-0
    /// assertion makes the RAII discipline a bright-line falsifier
    /// rather than a memory-pressure observability problem.
    ///
    /// We pair the assertion with `metric_value(&metrics, "error") =
    /// 1` to prove the hook actually executed the `Error(_)` arm —
    /// a vacuous test (admission map empty because the hook
    /// short-circuited at step 1 / 2 before populating the map)
    /// would still see `admission_len() == 0`, but would have a
    /// zero error metric.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn try_restore_layer_error_outcome_retires_admission_entry() {
        let tv = TestValkey::setup().await;
        let layer_store = Arc::new(LayerStore::new(
            tv.client().clone(),
            tv.key_schema().clone(),
            4096,
        ));
        let store: Arc<dyn CheckpointStore> = Arc::new(FakeCheckpointStore::always_transient());
        let metrics = make_metrics();
        let hook = RecoveryHook::new(
            store,
            layer_store,
            Arc::new(test_config(true)),
            metrics.clone(),
        );

        let outcome = hook.try_restore_layer(4242).await;
        match outcome {
            RecoveryOutcome::Error(CheckpointError::Transient { .. }) => {}
            other => panic!("expected Error(Transient); got {other:?}"),
        }

        // The error-arm metric increment proves the hook actually
        // executed the Error(_) branch — vacuous test guard.
        assert_eq!(
            metric_value(&metrics, "error"),
            1,
            "Error(_) arm MUST bump the `error` metric cell (proves the hook ran)"
        );

        // Lifecycle invariant: the admission map MUST drain to zero
        // after the Error(_) outcome. The `AdmissionMapGuard`'s
        // Drop impl runs unconditionally — across Ok and Err returns
        // from `try_restore_layer` — so an entry remaining here
        // would indicate the guard was elided on the Error arm or
        // the refcount accounting is asymmetric across success/error.
        assert_eq!(
            hook.admission_len(),
            0,
            "Error(_) outcome MUST retire the admission map entry on the \
             last waiter's drop (mirror of b.3gx / b.xet RAII discipline)"
        );
    }

    /// `Conflict` variant routing: `CheckpointError::Conflict` from the
    /// store MUST surface as `RecoveryOutcome::Error(Conflict)` (not
    /// `NotRecoverable`, not silent fall-through to `Restored`), and
    /// MUST bump the `error` recovery metric cell. Pairs with the
    /// existing `Transient` and `Permanent` arm tests to cover three of
    /// the four classified `CheckpointError` variants on the
    /// step-4 fall-through. The fourth variant (`NotFound`) is exercised
    /// by the apply-guard-reject test
    /// (`try_restore_layer_not_recoverable_on_apply_guard_reject`) and
    /// maps to `NotRecoverable` rather than `Error(_)` because it has a
    /// dedicated arm in step 5's match — `NotFound` from `apply` means
    /// the live mapping retired, not a backend failure.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn try_restore_layer_conflict_adapter_error_returns_error() {
        let tv = TestValkey::setup().await;
        let layer_store = Arc::new(LayerStore::new(
            tv.client().clone(),
            tv.key_schema().clone(),
            4096,
        ));
        let store: Arc<dyn CheckpointStore> = Arc::new(FakeCheckpointStore::always_conflict());
        let metrics = make_metrics();
        let hook = RecoveryHook::new(
            store,
            layer_store,
            Arc::new(test_config(true)),
            metrics.clone(),
        );

        let outcome = hook.try_restore_layer(1).await;
        match outcome {
            RecoveryOutcome::Error(CheckpointError::Conflict) => {}
            other => panic!("expected Error(Conflict); got {other:?}"),
        }
        assert_eq!(metric_value(&metrics, "error"), 1);
        assert_eq!(metric_value(&metrics, "restored"), 0);
        assert_eq!(metric_value(&metrics, "not_recoverable"), 0);

        // Lifecycle invariant (paired with the dedicated test above):
        // `Conflict` flows through the same `Error(_)` exit as
        // `Transient` / `Permanent`, so the admission map MUST also
        // drain to zero here.
        assert_eq!(
            hook.admission_len(),
            0,
            "Conflict-arm `Error(_)` outcome MUST retire the admission \
             map entry on the last waiter's drop"
        );
    }

    // ── b.gg7: the step-5 fetch deadline ────────────────────────────────
    //
    // `RecoveryHook::select_restore_source` — the whole list-then-get pair,
    // not just its final get — runs under a
    // `tokio::time::timeout(Config::checkpoint_recovery_fetch_budget())` —
    // `max(rpc_deadline_ms, 2 × checkpoint_request_timeout_ms)`, NOT the RPC
    // deadline alone. The adapter's own ceilings are
    // knowable but long (540 s on StorageApi at the defaults), the unary
    // handlers are bounded by the tower deadline layer regardless, and
    // `SubscribeReadLayer` — the lazy-recovery read path — is deliberately
    // exempt from that layer. So on the streaming path this selection was the
    // one place a subscriber could wait out the adapter, while *holding* the
    // per-`object_id` admission mutex and queueing every other reader of the
    // same layer behind it. The stall fixture therefore scripts the stall onto
    // the listing as well as the get, so the budget is proven against the call
    // the selection actually makes first.

    /// A `Config` whose recovery fetch budget is small enough to assert
    /// against.
    ///
    /// The budget is `max(rpc_deadline_ms, 2 × checkpoint_request_timeout_ms)`
    /// (see `Config::checkpoint_recovery_fetch_budget`), so BOTH knobs have to
    /// be scaled down — the shared fixture's 30 s request timeout alone would
    /// pin the budget at 60 s no matter what the deadline said. Halving the
    /// request timeout relative to the deadline puts the `max` on the deadline
    /// term, which is the one these tests name.
    fn stall_test_config(budget_ms: u64) -> Config {
        Config {
            rpc_deadline_ms: budget_ms,
            checkpoint_request_timeout_ms: budget_ms / 2,
            ..test_config(true)
        }
    }

    /// The fetch deadline itself: a stalled adapter surfaces on the existing
    /// `Error(_)` arm rather than pinning the caller for the adapter's own
    /// (far longer) ceiling.
    ///
    /// Both halves matter. The **classification** must stay inside the arm the
    /// hook already had — the `error` metric cell and the caller's retry-once
    /// semantics are defined there, and a new outcome variant would have
    /// silently changed both. The **elapsed time** is what proves the deadline
    /// applied at all: without it this call takes 30 s.
    ///
    /// Named for the *budget*, not for `rpc_deadline_ms`: since review round 2
    /// the bound is `max(rpc_deadline_ms, 2 × checkpoint_request_timeout_ms)`,
    /// and [`stall_test_config`] scales both terms precisely because naming
    /// only one of them is what the expression exists to prevent.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn try_restore_layer_error_when_the_checkpoint_fetch_exceeds_the_recovery_fetch_budget() {
        const BUDGET_MS: u64 = 200;

        let tv = TestValkey::setup().await;
        let layer_store = Arc::new(LayerStore::new(
            tv.client().clone(),
            tv.key_schema().clone(),
            4096,
        ));
        let store: Arc<dyn CheckpointStore> = Arc::new(FakeCheckpointStore::stalling());
        let metrics = make_metrics();
        let hook = RecoveryHook::new(
            store,
            layer_store,
            Arc::new(stall_test_config(BUDGET_MS)),
            metrics.clone(),
        );

        let started = std::time::Instant::now();
        let outcome = hook.try_restore_layer(4242).await;
        let elapsed = started.elapsed();

        match outcome {
            RecoveryOutcome::Error(CheckpointError::Transient { .. }) => {}
            other => panic!(
                "a fetch that outran the recovery budget must land on the existing \
                 Error(Transient) arm — that arm is what defines the `error` metric and the \
                 caller's retry-once behaviour, and any other variant silently redefines both. \
                 Got {other:?}"
            ),
        }
        assert!(
            elapsed < FakeCheckpointStore::STALL / 2,
            "the deadline, not the adapter, must have ended the fetch: budget {BUDGET_MS}ms, \
             took {elapsed:?}"
        );
        assert_eq!(
            metric_value(&metrics, "error"),
            1,
            "the elapse must count as an error attempt like every other classified failure"
        );
        assert_eq!(metric_value(&metrics, "not_recoverable"), 0);
        assert_eq!(metric_value(&metrics, "restored"), 0);
        assert_eq!(
            hook.admission_len(),
            0,
            "and the map entry must retire on the way out, same as every other Error arm"
        );
    }

    /// The load-bearing half: the timeout releases the per-`object_id`
    /// admission mutex, so a second reader of the same layer is not queued
    /// behind the first one's stall.
    ///
    /// This is the reason the deadline exists rather than just being tidy.
    /// `try_restore_layer` holds a K=1 mutex per `object_id` across the fetch,
    /// so an unbounded fetch does not cost one subscriber the adapter's
    /// ceiling — it costs *every* subscriber of that layer the ceiling, in
    /// series. Two concurrent callers on one `object_id` is the smallest
    /// configuration in which a *second* holder exists at all.
    ///
    /// # What the assertions actually pin
    ///
    /// Stated plainly, because the wall-clock bound here is `STALL / 2` — the
    /// same loose bound the single-caller sibling uses — and therefore does
    /// **not** measure "the pair cost ~2 × budget":
    ///
    /// - **Order of magnitude.** 15 s is far below the ~30 s a *second* holder
    ///   would spend if it inherited the full adapter stall, so the arrangement
    ///   under test (queue behind the first holder, then stall again) fails
    ///   here. It is deliberately nowhere near tight enough to distinguish
    ///   ~2 × budget from ~1 × budget, and nothing here should be read as
    ///   asserting the callers serialized at all.
    /// - **`error == 2`.** Both callers reached step 5 and recorded their own
    ///   attempt, which is what rules out the vacuous pass where the second
    ///   caller short-circuits on the post-acquire structural-header re-check and never
    ///   touches the store.
    /// - **Both outcomes `Error(Transient)`.** The deadline is what ended each
    ///   of them, not the mutex and not the adapter.
    ///
    /// A genuinely tight per-holder latency assertion would need to observe the
    /// two callers' individual spans, and at these budgets that is a wall-clock
    /// race on a shared runner rather than a test.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn stalled_checkpoint_fetch_releases_the_admission_mutex_for_the_next_caller() {
        const BUDGET_MS: u64 = 200;
        const OBJECT_ID: u64 = 5151;

        let tv = TestValkey::setup().await;
        let layer_store = Arc::new(LayerStore::new(
            tv.client().clone(),
            tv.key_schema().clone(),
            4096,
        ));
        let store: Arc<dyn CheckpointStore> = Arc::new(FakeCheckpointStore::stalling());
        let metrics = make_metrics();
        let hook = Arc::new(RecoveryHook::new(
            store,
            layer_store,
            Arc::new(stall_test_config(BUDGET_MS)),
            metrics.clone(),
        ));

        // Both callers target the SAME object_id — that is what puts them on
        // one admission entry and makes the second one's latency a function of
        // the first one's.
        let started = std::time::Instant::now();
        let mut handles = Vec::new();
        for _ in 0..2 {
            let hook = Arc::clone(&hook);
            handles.push(tokio::spawn(async move {
                hook.try_restore_layer(OBJECT_ID).await
            }));
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
                    RecoveryOutcome::Error(CheckpointError::Transient { .. })
                ),
                "both callers must surface the deadline, not just the mutex winner; got \
                 {outcome:?}"
            );
        }
        assert!(
            elapsed < FakeCheckpointStore::STALL / 2,
            "an order-of-magnitude bound, not a latency budget (see the doc-comment): a second \
             holder that inherited the full {stall:?} adapter stall lands above this line, a \
             pair bounded at {BUDGET_MS}ms apiece lands two orders below it. Took {elapsed:?}",
            stall = FakeCheckpointStore::STALL
        );
        assert_eq!(
            metric_value(&metrics, "error"),
            2,
            "each caller must record its OWN attempt — one attempt would mean the second caller \
             short-circuited on the post-acquire structural-header re-check and never held the mutex \
             across a fetch, which is the shape that would make this whole test vacuous"
        );
        assert_eq!(
            hook.admission_len(),
            0,
            "and the entry retires after the last waiter drops"
        );
    }

    // ── SR-2.5: kind-aware restore-source selection ──────────────────────
    //
    // The four tests below are the hook-level pins on
    // `RecoveryHook::select_restore_source`: list the generations, take the
    // newest `CheckpointKind::Base`, fetch it by handle. They share the
    // seeding helpers immediately below, which mint metas addressed to the
    // `("s1", "l")` layer each test installs a live mapping for so
    // `apply_checkpoint`'s SR-2.4 identity guard passes.

    /// A base-generation meta at `snap_seq`, addressed to `("s1", "l")`.
    ///
    /// `observed_commit_span` is left at its `0` no-measurement default on
    /// purpose: `apply_checkpoint` sizes its guard band as
    /// `max(checkpoint_seq_safety_margin, observed_commit_span)`, so a
    /// non-zero span here would move the post-restore `sequence_num` off the
    /// `snap_seq + 1000` these tests assert against.
    fn selection_base_meta(snap_seq: u64) -> CheckpointMeta {
        CheckpointMeta {
            creator_replica_id: "test-replica".to_string(),
            written_at_unix_ms: 1_700_000_000_000,
            snap_seq: Some(snap_seq),
            session_id: "s1".to_string(),
            layer_name: "l".to_string(),
            created_by: "u".to_string(),
            server_ts: 123,
            ..CheckpointMeta::default()
        }
    }

    /// A tail-generation meta covering the commit span `(from_seq, to_seq]`
    /// off the base at `base_snap_seq`. Both lineage fields are populated
    /// because `validate_tail_put` rejects a tail put that leaves either
    /// `None`, and `snap_seq` carries `to_seq` — the number a tail orders and
    /// addresses by (SR-2.3).
    fn selection_tail_meta(base_snap_seq: u64, from_seq: u64, to_seq: u64) -> CheckpointMeta {
        CheckpointMeta {
            kind: CheckpointKind::Tail,
            base_snap_seq: Some(base_snap_seq),
            from_seq: Some(from_seq),
            ..selection_base_meta(to_seq)
        }
    }

    /// The payload the seeded tails carry: an OMPF whose OODF delta is an
    /// **incremental** diff (`is_diff = true`, `base_version != 0`) rather
    /// than a reset.
    ///
    /// This is what makes the regression-proof claim on the tests below
    /// concrete rather than notional. `decode_checkpoint_ompf` rejects
    /// exactly this shape as [`CheckpointError::Permanent`] ("the delta is
    /// not a reset delta"), so a restore path that fed a tail to the reset
    /// decoder — which is what the pre-SR-2.5 `get_latest_layer_checkpoint`
    /// selection did whenever the newest generation was a tail — surfaces
    /// `Error(Permanent)`, the stranded-recovery outcome SR-2.5 exists to
    /// prevent, instead of the outcome each test asserts.
    fn build_incremental_tail_ompf(base_version: u64) -> Bytes {
        assert_ne!(
            base_version, 0,
            "a tail payload must not be a reset delta, or it would decode and the \
             regression-proof property of these fixtures would be vacuous"
        );
        let delta = OwnedDelta {
            is_diff: true,
            base_version,
            ..OwnedDelta::default()
        };
        let delta_bytes = oodf_ffi::build_delta(&delta).expect("build_delta");
        let ompf = OmpfContainer::serialize(&delta_bytes, &[]).expect("OMPF serialize");
        Bytes::from(ompf)
    }

    /// Newest-base selection: a lineage whose newest generations are tails
    /// still restores, and restores from the newest **base** — not the newest
    /// generation, and not merely *some* base.
    ///
    /// Seeded lineage for one `object_id`: base@5, base@9, tail@12, tail@15.
    /// Two tails, both newer than every base, so skipping them cannot pass as
    /// a one-element accident.
    ///
    /// # Regression proof
    ///
    /// This test FAILS against the pre-SR-2.5 hook, which selected via
    /// `get_latest_layer_checkpoint` — that returns tail@15, whose incremental
    /// payload `decode_checkpoint_ompf` rejects, so the outcome there is
    /// `Error(Permanent)` with the `error` cell at 1 and no `obj:hdr` written.
    ///
    /// # Which base, not just any base
    ///
    /// The `sequence_num` assertion is the load-bearing one. `apply_checkpoint`
    /// seeds `restore_floor + band`, where `restore_floor` folds to `snap_seq`
    /// (`probe_stream_high_water` yields `None` for a fresh `object_id` that has
    /// no `obj:stream`) and `band` folds to `checkpoint_seq_safety_margin`
    /// (`observed_commit_span` is 0 on these metas). So the restored header
    /// carries `snap_seq + 1000`, which reads 1009 for base@9 and would read
    /// 1005 for base@5 — the only observable that distinguishes *newest* base
    /// from *any* base.
    ///
    /// # Flag independence
    ///
    /// This hook has no capture-flag input; when Epic 3 adds one, this test
    /// must keep passing with the flag in both positions. A store can already
    /// hold tails with capture off (written by a peer replica, written before
    /// the flag was turned back off, or restored from an older backup), so the
    /// read side must stay decoupled from the write side.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn try_restore_layer_restores_newest_base_when_newer_tails_present() {
        let tv = TestValkey::setup().await;
        let layer_store = Arc::new(LayerStore::new(
            tv.client().clone(),
            tv.key_schema().clone(),
            4096,
        ));

        let oid: u64 = 6101;
        let mapping_key = tv.key_schema().layer_mapping("s1", "l");
        let _: () = tv
            .client()
            .pool
            .set(&mapping_key, oid.to_string(), None, None, false)
            .await
            .expect("SET mapping");

        // Seed the lineage: two bases, then two strictly newer tails.
        let in_mem = Arc::new(InMemoryCheckpointStore::new());
        for base_seq in [5u64, 9] {
            in_mem
                .put_layer_checkpoint(
                    oid,
                    base_seq,
                    build_empty_reset_ompf(),
                    selection_base_meta(base_seq),
                )
                .await
                .expect("put base");
        }
        for (from_seq, to_seq) in [(9u64, 12u64), (12, 15)] {
            in_mem
                .put_layer_checkpoint(
                    oid,
                    to_seq,
                    build_incremental_tail_ompf(to_seq),
                    selection_tail_meta(9, from_seq, to_seq),
                )
                .await
                .expect("put tail");
        }

        let store: Arc<dyn CheckpointStore> = in_mem;
        let metrics = make_metrics();
        let hook = RecoveryHook::new(
            store,
            layer_store.clone(),
            Arc::new(test_config(true)),
            metrics.clone(),
        );

        let outcome = hook.try_restore_layer(oid).await;
        assert!(
            matches!(outcome, RecoveryOutcome::Restored),
            "a lineage whose newest generations are tails MUST still restore from its newest \
             base; got {outcome:?}"
        );
        assert_eq!(
            metric_value(&metrics, "restored"),
            1,
            "the newest-base restore MUST count as a `restored` attempt"
        );
        assert_eq!(
            metric_value(&metrics, "error"),
            0,
            "no tail may reach the reset decoder — an `error` tick here is the pre-SR-2.5 \
             Error(Permanent) regression"
        );
        assert_eq!(metric_value(&metrics, "not_recoverable"), 0);

        let header = layer_store
            .read_object_header(oid)
            .await
            .expect("restored object header must be readable");
        assert_eq!(
            header.sequence_num,
            9 + test_config(true).checkpoint_seq_safety_margin,
            "the restore MUST have started from base@9 (seeded 1009), not base@5 (1005) and not \
             any tail"
        );

        assert_eq!(
            hook.admission_len(),
            0,
            "admission map entry MUST be retired on the last waiter's drop"
        );
    }

    /// A store holding only tails for an object is unrecoverable **in exactly
    /// the way an empty store is** — `NotRecoverable`, not `Error(_)`.
    ///
    /// One test, two arms, so the equivalence is asserted rather than inferred
    /// across two test bodies:
    ///
    /// - **Arm A** — `oid_a` holds tail@7 + tail@11 and no base. The live layer
    ///   mapping IS installed, so a `NotRecoverable` here cannot be confused
    ///   with an apply-guard rejection.
    /// - **Arm B** — `oid_b` is untouched, so the same store lists nothing for
    ///   it.
    ///
    /// # Regression proof
    ///
    /// This test FAILS against the pre-SR-2.5 hook. There, arm A's
    /// `get_latest_layer_checkpoint` returns tail@11 and its incremental
    /// payload reaches `decode_checkpoint_ompf`, which rejects it as
    /// `Permanent` — so the verdict is `Error(Permanent)` and the two arms
    /// land in *different* metric cells. Hence `error == 0` is the
    /// load-bearing assertion here, not a courtesy one: it is what pins that a
    /// tails-only store is reported as un-restorable rather than as a backend
    /// failure operators would page on.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn try_restore_layer_tails_only_store_is_not_recoverable_like_empty_store() {
        let tv = TestValkey::setup().await;
        let layer_store = Arc::new(LayerStore::new(
            tv.client().clone(),
            tv.key_schema().clone(),
            4096,
        ));

        let oid_a: u64 = 6201;
        let oid_b: u64 = 6202;

        // Arm A's live mapping — present so the NotRecoverable verdict is
        // attributable to selection, not to apply's identity guard.
        let mapping_key = tv.key_schema().layer_mapping("s1", "l");
        let _: () = tv
            .client()
            .pool
            .set(&mapping_key, oid_a.to_string(), None, None, false)
            .await
            .expect("SET mapping");

        // Arm A's lineage: tails only, no base. Not producible by the shipped
        // write paths (a tail is only ever written after the base it names), so
        // this models manual pruning of the base or partial store loss.
        let in_mem = Arc::new(InMemoryCheckpointStore::new());
        for (from_seq, to_seq) in [(4u64, 7u64), (7, 11)] {
            in_mem
                .put_layer_checkpoint(
                    oid_a,
                    to_seq,
                    build_incremental_tail_ompf(to_seq),
                    selection_tail_meta(4, from_seq, to_seq),
                )
                .await
                .expect("put tail");
        }

        let store: Arc<dyn CheckpointStore> = in_mem;
        let metrics = make_metrics();
        let hook = RecoveryHook::new(
            store,
            layer_store,
            Arc::new(test_config(true)),
            metrics.clone(),
        );

        let outcome_a = hook.try_restore_layer(oid_a).await;
        assert!(
            matches!(outcome_a, RecoveryOutcome::NotRecoverable),
            "arm A (tails only, live mapping installed) MUST report NotRecoverable; got \
             {outcome_a:?}"
        );

        let outcome_b = hook.try_restore_layer(oid_b).await;
        assert!(
            matches!(outcome_b, RecoveryOutcome::NotRecoverable),
            "arm B (empty store) MUST report NotRecoverable; got {outcome_b:?}"
        );

        assert_eq!(
            metric_value(&metrics, "not_recoverable"),
            2,
            "both arms MUST land in the same `not_recoverable` cell — that shared cell IS the \
             equivalence this test asserts"
        );
        assert_eq!(
            metric_value(&metrics, "error"),
            0,
            "a tails-only store MUST NOT surface as a backend failure: pre-SR-2.5 the tail bytes \
             reached decode_checkpoint_ompf and the verdict was Error(Permanent)"
        );
        assert_eq!(metric_value(&metrics, "restored"), 0);

        for oid in [oid_a, oid_b] {
            let header_key = tv.key_schema().obj_header(&oid.to_string());
            let exists: bool = tv
                .client()
                .pool
                .exists(&header_key)
                .await
                .expect("EXISTS obj:hdr");
            assert!(
                !exists,
                "neither arm may write obj:hdr; got a header for object {oid}"
            );
        }

        assert_eq!(hook.admission_len(), 0);
    }

    /// A base and a tail colliding at one sequence: selection restores the
    /// base.
    ///
    /// The collision is a live case, not a hypothetical — SR-4.1 re-base
    /// composes a base at the newest consumed tail's `to_seq`, so `base@S` and
    /// `tail@to_seq == S` coexist until retention removes one.
    ///
    /// # What this pins, precisely
    ///
    /// A hook-level regression pin only. At an equal sequence the kind filter
    /// yields base@9 under *either* tie-break, so this test cannot detect a
    /// Tail-before-Base listing order; the authoritative cross-adapter
    /// tie-break pin is `generation_sort_key`'s own unit assertion. What this
    /// case does hold is that the hook keeps preferring the composable
    /// generation when both kinds are addressable at the same number — the
    /// arrangement most likely to be broken by a future selection rewrite that
    /// reaches for the sequence alone.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn try_restore_layer_prefers_base_over_tail_at_equal_sequence() {
        let tv = TestValkey::setup().await;
        let layer_store = Arc::new(LayerStore::new(
            tv.client().clone(),
            tv.key_schema().clone(),
            4096,
        ));

        let oid: u64 = 6301;
        let mapping_key = tv.key_schema().layer_mapping("s1", "l");
        let _: () = tv
            .client()
            .pool
            .set(&mapping_key, oid.to_string(), None, None, false)
            .await
            .expect("SET mapping");

        let in_mem = Arc::new(InMemoryCheckpointStore::new());
        in_mem
            .put_layer_checkpoint(oid, 9, build_empty_reset_ompf(), selection_base_meta(9))
            .await
            .expect("put base@9");
        in_mem
            .put_layer_checkpoint(
                oid,
                9,
                build_incremental_tail_ompf(9),
                selection_tail_meta(5, 5, 9),
            )
            .await
            .expect("put tail@9");

        let store: Arc<dyn CheckpointStore> = in_mem;
        let metrics = make_metrics();
        let hook = RecoveryHook::new(
            store,
            layer_store.clone(),
            Arc::new(test_config(true)),
            metrics.clone(),
        );

        let outcome = hook.try_restore_layer(oid).await;
        assert!(
            matches!(outcome, RecoveryOutcome::Restored),
            "base@9 colliding with tail@9 MUST restore from the base; got {outcome:?}"
        );
        assert_eq!(metric_value(&metrics, "restored"), 1);
        assert_eq!(metric_value(&metrics, "error"), 0);
        assert_eq!(metric_value(&metrics, "not_recoverable"), 0);

        let header = layer_store
            .read_object_header(oid)
            .await
            .expect("restored object header must be readable");
        assert_eq!(
            header.sequence_num,
            9 + test_config(true).checkpoint_seq_safety_margin,
            "the restore MUST have started from the base at sequence 9"
        );
    }

    /// A store whose listing advertises bases but whose fetch answers
    /// `Ok(None)` — the contract-documented concurrent-prune race between the
    /// list and the get.
    ///
    /// Deliberately asymmetric across the two selection entry points, which is
    /// why it is not a [`LayerFetch`] posture on the shared
    /// [`FakeCheckpointStore`]: that double's contract is that a posture is
    /// honoured by *all* its layer-read methods precisely so a posture cannot
    /// be silently vacated, and the race under test here is the one case where
    /// the two halves must legitimately disagree. Everything not on the
    /// selection path is delegated to a `FakeCheckpointStore::delegating()`,
    /// which also carries the module-wide "session methods panic" guard
    /// through to this fixture.
    ///
    /// It advertises **two** bases and records every layer read, so the "no
    /// step-down" assertion is falsifiable: a selection that stepped down to
    /// the next older base on an `Ok(None)` would issue a second
    /// `get_layer_checkpoint` (for base@5) and show up as a third entry in the
    /// recorded log. With a single advertised base such an implementation
    /// would terminate immediately and the assertion would be vacuous.
    struct VanishingBaseStore {
        inner: FakeCheckpointStore,
        generations: Vec<CheckpointHandle>,
        layer_reads: StdMutex<Vec<String>>,
        /// When `Some(seq)`, the fetch of that generation raises a `Transient`
        /// store error instead of vanishing. Opt-in (default `None`) so the
        /// vanishing-race test above is untouched; see
        /// `select_restore_source_emits_step_downs_accumulated_before_a_store_error`
        /// for the sequence it exists to build.
        transient_fetch_at: Option<u64>,
    }

    impl VanishingBaseStore {
        /// Advertise `base_seqs` (newest first) for `object_id`; every fetch
        /// of any of them answers `Ok(None)`.
        fn new(object_id: u64, base_seqs: &[u64]) -> Self {
            Self {
                inner: FakeCheckpointStore::delegating(),
                generations: base_seqs
                    .iter()
                    .map(|seq| {
                        CheckpointHandle::base(object_id, *seq, format!("{object_id}:{seq}"))
                    })
                    .collect(),
                layer_reads: StdMutex::new(Vec::new()),
                transient_fetch_at: None,
            }
        }

        /// Turn the fetch of the advertised generation at `seq` from "vanished"
        /// (`Ok(None)`, a step-down) into a backend failure (`Err(Transient)`,
        /// which propagates). Every other advertised generation still vanishes.
        fn with_transient_fetch_at(mut self, seq: u64) -> Self {
            self.transient_fetch_at = Some(seq);
            self
        }

        fn record(&self, read: String) {
            self.layer_reads.lock().unwrap().push(read);
        }

        fn layer_reads(&self) -> Vec<String> {
            self.layer_reads.lock().unwrap().clone()
        }
    }

    #[async_trait]
    impl CheckpointStore for VanishingBaseStore {
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

        /// Recorded but not scripted: the hook must not reach for this since
        /// SR-2.5, and recording it (rather than panicking) keeps the
        /// recorded-log assertion the single place this test says so.
        async fn get_latest_layer_checkpoint(
            &self,
            object_id: u64,
        ) -> Result<Option<CheckpointBlob>, CheckpointError> {
            self.record("get_latest".to_string());
            self.inner.get_latest_layer_checkpoint(object_id).await
        }

        async fn list_layer_checkpoints(
            &self,
            _object_id: u64,
        ) -> Result<Vec<CheckpointHandle>, CheckpointError> {
            self.record("list".to_string());
            Ok(self.generations.clone())
        }

        /// The race itself: the listing saw this generation, retention removed
        /// it before the get landed — unless this is the generation
        /// [`VanishingBaseStore::with_transient_fetch_at`] armed, which fails
        /// the backend instead.
        async fn get_layer_checkpoint(
            &self,
            handle: &CheckpointHandle,
        ) -> Result<Option<CheckpointBlob>, CheckpointError> {
            self.record(format!("get(seq={})", handle.seq()));
            if self.transient_fetch_at == Some(handle.seq()) {
                return Err(CheckpointError::Transient {
                    source: format!(
                        "synthetic transient fetch failure at generation {}",
                        handle.seq()
                    )
                    .into(),
                });
            }
            Ok(None)
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

    /// The list-then-get race: the selected base disappears between the
    /// listing and the fetch. Under SR-3.1 that is a **step-down**, not a
    /// terminal answer: selection advances to the next older restorable
    /// generation, and only when *every* candidate has broken does the attempt
    /// end `NotRecoverable` — still the same verdict as an empty store, still
    /// not an `error`.
    ///
    /// # Revisited deliberately, exactly as its predecessor asked
    ///
    /// The pre-Epic-4 form of this test pinned the opposite property —
    /// **single-shot**, no step-down fetch of base@5 — because at the time
    /// staying single-shot was what kept base-only recovery byte-identical to
    /// `get_latest_layer_checkpoint`'s internal list-then-read race. Its own
    /// doc-comment named this Epic as the moment to revisit it rather than
    /// discover it, and this is that revisit: SR-3.1's "a broken lineage steps
    /// down to the next older restorable generation, **observably**" is now the
    /// contract, and `LineageBreak::LinkAbsent { at_fetch: true }` is precisely
    /// the break this race produces.
    ///
    /// # What is still pinned, and why the fixture still has two bases
    ///
    /// - **The step-down really happens**: the recorded read log must show the
    ///   second fetch. With one advertised base the assertion would be vacuous.
    /// - **Exhausting every candidate is `NotRecoverable`, never `Error`.** That
    ///   is the half that did NOT change, and it is the load-bearing one: a
    ///   vanished generation is a data verdict, not a backend failure operators
    ///   should page on.
    /// - **The step-downs are observable** on the lineage family, which is where
    ///   "DR served older state than the store nominally holds" is supposed to
    ///   show up — and *not* as a fourth `result` value on the attempt counter,
    ///   whose `sum by (result) == attempts` identity still holds here.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn try_restore_layer_base_vanishing_between_list_and_get_steps_down_then_not_recoverable()
    {
        let tv = TestValkey::setup().await;
        let layer_store = Arc::new(LayerStore::new(
            tv.client().clone(),
            tv.key_schema().clone(),
            4096,
        ));

        let oid: u64 = 6401;
        // The live mapping is installed so a NotRecoverable verdict is
        // attributable to the vanished base, not to apply's identity guard —
        // the apply path is never reached here, and this rules out the reading
        // where it was reached and rejected.
        let mapping_key = tv.key_schema().layer_mapping("s1", "l");
        let _: () = tv
            .client()
            .pool
            .set(&mapping_key, oid.to_string(), None, None, false)
            .await
            .expect("SET mapping");

        let vanishing = Arc::new(VanishingBaseStore::new(oid, &[9, 5]));
        let store: Arc<dyn CheckpointStore> = vanishing.clone();
        let metrics = make_metrics();
        let hook = RecoveryHook::new(
            store,
            layer_store,
            Arc::new(test_config(true)),
            metrics.clone(),
        );

        let outcome = hook.try_restore_layer(oid).await;
        assert!(
            matches!(outcome, RecoveryOutcome::NotRecoverable),
            "a base that vanished between the list and the get MUST report NotRecoverable, \
             indistinguishable from an empty store; got {outcome:?}"
        );
        assert_eq!(
            metric_value(&metrics, "not_recoverable"),
            1,
            "the race MUST record exactly one `not_recoverable` attempt"
        );
        assert_eq!(
            metric_value(&metrics, "error"),
            0,
            "a vanished generation is not a backend failure"
        );
        assert_eq!(metric_value(&metrics, "restored"), 0);

        assert_eq!(
            vanishing.layer_reads(),
            vec![
                "list".to_string(),
                "get(seq=9)".to_string(),
                "get(seq=5)".to_string()
            ],
            "selection MUST step down: one listing, a fetch of the newest base, and then a \
             fetch of the next older one (and no `get_latest` fallback at any point)"
        );

        assert_eq!(
            lineage_metric_value(&metrics, "stepped_down_link_broken"),
            2,
            "each vanished base MUST record its own step-down — that is what makes SR-3.1's \
             'observably' true rather than aspirational"
        );
        assert_eq!(
            lineage_metric_value(&metrics, "exhausted"),
            1,
            "running out of candidates MUST land on the exhausted cell exactly once"
        );
        assert_eq!(
            lineage_metric_value(&metrics, "gathered_base")
                + lineage_metric_value(&metrics, "gathered_composed"),
            0,
            "nothing was gathered"
        );

        assert_eq!(hook.admission_len(), 0);
    }

    /// A step-down that happened is still a step-down when the *next* candidate
    /// fails the backend: `select_restore_source` must emit the accumulated
    /// `StepDownRecord`s before it propagates the store error.
    ///
    /// # The sequence, and why it is reachable
    ///
    /// One `LineageSelection::next_restorable` call walks candidates until one
    /// verifies. A candidate that breaks pushes a record and the walk
    /// *continues*; only a store error leaves the call. So the ordinary
    /// "generation vanished, and then the backend hiccuped" sequence produces a
    /// single `next_restorable` call that returns `Err` with a non-empty
    /// step-down list — here: base@9 vanishes between the list and the get
    /// (`LinkAbsent { at_fetch: true }`), then base@5's fetch answers
    /// `Transient`.
    ///
    /// # What regresses without the fix
    ///
    /// Applying `?` at the call site short-circuits past `emit_new_step_downs`,
    /// so base@9's break is recorded on the selection and never reaches the
    /// metric — the lineage family reports zero step-downs on precisely the
    /// incident where an operator is asking "did DR degrade before it failed?".
    /// The fix binds the result, drains, and only then propagates. This test
    /// fails on the pre-fix ordering (`stepped_down_link_broken` = 0) and is the
    /// only thing that distinguishes the two.
    ///
    /// The `error` outcome itself is pinned too, so a "fix" that swallowed the
    /// store error to reach the drain would not pass either.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn select_restore_source_emits_step_downs_accumulated_before_a_store_error() {
        let tv = TestValkey::setup().await;
        let layer_store = Arc::new(LayerStore::new(
            tv.client().clone(),
            tv.key_schema().clone(),
            4096,
        ));

        let oid: u64 = 6402;
        // No live mapping is installed: the apply path is never reached on this
        // sequence (selection errors first), so there is nothing for the
        // identity guard to adjudicate.
        let store_double =
            Arc::new(VanishingBaseStore::new(oid, &[9, 5]).with_transient_fetch_at(5));
        let store: Arc<dyn CheckpointStore> = store_double.clone();
        let metrics = make_metrics();
        let hook = RecoveryHook::new(
            store,
            layer_store,
            Arc::new(test_config(true)),
            metrics.clone(),
        );

        let outcome = hook.try_restore_layer(oid).await;
        assert!(
            matches!(
                outcome,
                RecoveryOutcome::Error(CheckpointError::Transient { .. })
            ),
            "a store error on the second candidate MUST still surface as Error(Transient); \
             got {outcome:?}"
        );
        assert_eq!(
            metric_value(&metrics, "error"),
            1,
            "the attempt counter MUST record exactly one `error`"
        );
        assert_eq!(metric_value(&metrics, "not_recoverable"), 0);
        assert_eq!(metric_value(&metrics, "restored"), 0);

        assert_eq!(
            store_double.layer_reads(),
            vec![
                "list".to_string(),
                "get(seq=9)".to_string(),
                "get(seq=5)".to_string()
            ],
            "the walk MUST reach the second candidate — otherwise the accumulated-record \
             precondition this test depends on would not exist"
        );

        assert_eq!(
            lineage_metric_value(&metrics, "stepped_down_link_broken"),
            1,
            "base@9's break MUST be emitted even though the call it happened in ended in a \
             store error — this is the assertion the `?`-at-the-call-site ordering fails"
        );
        assert_eq!(
            lineage_metric_value(&metrics, "exhausted"),
            0,
            "a store error is not exhaustion; the candidate list was never drained"
        );
        assert_eq!(
            lineage_metric_value(&metrics, "gathered_base")
                + lineage_metric_value(&metrics, "gathered_composed"),
            0,
            "nothing was gathered"
        );

        assert_eq!(hook.admission_len(), 0);
    }

    // ── SR-3.1 / SR-3.4: composed restore through the hook ───────────────

    /// The C1 fail-closed guard, at the **hook** level: a composed lineage whose
    /// replayed tails reference external values, restored through a hook that
    /// was never handed an [`ExternalValueStore`], reports
    /// `RecoveryOutcome::Error(Permanent)` and writes **no** `obj:hdr`.
    ///
    /// Production always wires the store (`main.rs` calls
    /// [`RecoveryHook::with_ext_value_store`] unconditionally), so this pins the
    /// *defect* behaviour: a mis-wired hook must fail loudly rather than
    /// materialize an object whose tail-carried values live nowhere reachable.
    /// The check itself lives in the apply core — the payload carries the ext
    /// set, so the core is the only place that can see both halves of the
    /// mismatch — and this test proves the hook surfaces it.
    ///
    /// The lineage is the composition engine's own canonical fixture: a real
    /// assembler-built base at `BASE_SNAP_SEQ` plus a real
    /// `encode_tail_envelope` tail covering `(BASE_SNAP_SEQ, HEAD_SEQ]`, whose
    /// commits genuinely reference two distinct content hashes. Synthetic
    /// artifacts would make the "non-empty ext set" precondition a fixture
    /// artefact rather than a property of real captured traffic.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn composed_restore_without_an_ext_value_store_fails_closed_and_writes_nothing() {
        use crate::checkpoint::compose::tests as compose_tests;

        let tv = TestValkey::setup().await;
        let layer_store = Arc::new(LayerStore::new(
            tv.client().clone(),
            tv.key_schema().clone(),
            4096,
        ));

        let oid = compose_tests::OBJECT_ID;
        let mapping_key = tv.key_schema().layer_mapping("s1", "l");
        let _: () = tv
            .client()
            .pool
            .set(&mapping_key, oid.to_string(), None, None, false)
            .await
            .expect("SET mapping so the SR-2.4 identity guard passes");

        let (history, base) = compose_tests::canonical();
        let records = history.records_in(compose_tests::BASE_SNAP_SEQ, compose_tests::HEAD_SEQ);
        let tail = compose_tests::tail(
            compose_tests::BASE_SNAP_SEQ,
            compose_tests::BASE_SNAP_SEQ,
            &records,
        );

        let in_mem = Arc::new(InMemoryCheckpointStore::new());
        in_mem
            .put_layer_checkpoint(
                oid,
                compose_tests::BASE_SNAP_SEQ,
                base,
                selection_base_meta(compose_tests::BASE_SNAP_SEQ),
            )
            .await
            .expect("put base");
        in_mem
            .put_layer_checkpoint(
                oid,
                compose_tests::HEAD_SEQ,
                tail,
                selection_tail_meta(
                    compose_tests::BASE_SNAP_SEQ,
                    compose_tests::BASE_SNAP_SEQ,
                    compose_tests::HEAD_SEQ,
                ),
            )
            .await
            .expect("put tail");

        let store: Arc<dyn CheckpointStore> = in_mem;
        let metrics = make_metrics();
        // Deliberately NO `.with_ext_value_store(..)` — the wiring defect.
        let hook = RecoveryHook::new(
            store,
            layer_store,
            Arc::new(test_config(true)),
            metrics.clone(),
        );

        let outcome = hook.try_restore_layer(oid).await;
        match outcome {
            RecoveryOutcome::Error(CheckpointError::Permanent { .. }) => {}
            other => panic!(
                "a composed lineage with a non-empty ext set and no ExternalValueStore MUST \
                 fail closed as Error(Permanent) — never a silent skip and never a restore; \
                 got {other:?}"
            ),
        }

        assert_eq!(
            lineage_metric_value(&metrics, "gathered_composed"),
            1,
            "selection itself succeeded — the lineage gathered and composed; the refusal is \
             the apply core's, which is what pins WHERE the guard lives"
        );
        assert_eq!(metric_value(&metrics, "error"), 1);
        assert_eq!(metric_value(&metrics, "restored"), 0);
        assert_eq!(metric_value(&metrics, "not_recoverable"), 0);

        let header_key = tv.key_schema().obj_header(&oid.to_string());
        let exists: bool = tv
            .client()
            .pool
            .exists(&header_key)
            .await
            .expect("EXISTS obj:hdr after the fail-closed refusal");
        assert!(
            !exists,
            "the fail-closed arm MUST fire before write_restored_object; obj:hdr was created"
        );

        assert_eq!(hook.admission_len(), 0);
    }

    /// The composed happy path through the hook: a base plus one contiguous
    /// tail restores to the **lineage head** `S`, not to the base's sequence,
    /// and its tail-carried external values are in the live ext store.
    ///
    /// The `seq` assertion is the load-bearing one and is what a base-only
    /// regression would fail: `HEAD_SEQ + margin` vs `BASE_SNAP_SEQ + margin`.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn composed_restore_seeds_from_the_lineage_head_and_pre_stores_ext_values() {
        use crate::checkpoint::compose::tests as compose_tests;
        use crate::storage::external_values::ValkeyExternalValueStore;

        let tv = TestValkey::setup().await;
        let layer_store = Arc::new(LayerStore::new(
            tv.client().clone(),
            tv.key_schema().clone(),
            4096,
        ));
        let ext_store: Arc<dyn ExternalValueStore> = Arc::new(ValkeyExternalValueStore::new(
            tv.client().clone(),
            tv.key_schema().clone(),
        ));

        let oid = compose_tests::OBJECT_ID;
        let mapping_key = tv.key_schema().layer_mapping("s1", "l");
        let _: () = tv
            .client()
            .pool
            .set(&mapping_key, oid.to_string(), None, None, false)
            .await
            .expect("SET mapping");

        let (history, base) = compose_tests::canonical();
        let records = history.records_in(compose_tests::BASE_SNAP_SEQ, compose_tests::HEAD_SEQ);
        let tail = compose_tests::tail(
            compose_tests::BASE_SNAP_SEQ,
            compose_tests::BASE_SNAP_SEQ,
            &records,
        );

        let in_mem = Arc::new(InMemoryCheckpointStore::new());
        in_mem
            .put_layer_checkpoint(
                oid,
                compose_tests::BASE_SNAP_SEQ,
                base,
                selection_base_meta(compose_tests::BASE_SNAP_SEQ),
            )
            .await
            .expect("put base");
        in_mem
            .put_layer_checkpoint(
                oid,
                compose_tests::HEAD_SEQ,
                tail,
                selection_tail_meta(
                    compose_tests::BASE_SNAP_SEQ,
                    compose_tests::BASE_SNAP_SEQ,
                    compose_tests::HEAD_SEQ,
                ),
            )
            .await
            .expect("put tail");

        let store: Arc<dyn CheckpointStore> = in_mem;
        let metrics = make_metrics();
        let hook = RecoveryHook::new(
            store,
            layer_store.clone(),
            Arc::new(test_config(true)),
            metrics.clone(),
        )
        .with_ext_value_store(Arc::clone(&ext_store));

        let outcome = hook.try_restore_layer(oid).await;
        assert!(
            matches!(outcome, RecoveryOutcome::Restored),
            "a base plus a contiguous tail MUST restore; got {outcome:?}"
        );
        assert_eq!(metric_value(&metrics, "restored"), 1);
        assert_eq!(metric_value(&metrics, "error"), 0);
        assert_eq!(metric_value(&metrics, "not_recoverable"), 0);
        assert_eq!(
            lineage_metric_value(&metrics, "gathered_composed"),
            1,
            "the lineage MUST be reported composed, not base-only"
        );
        assert_eq!(lineage_metric_value(&metrics, "gathered_base"), 0);

        let header = layer_store
            .read_object_header(oid)
            .await
            .expect("restored object header must be readable");
        // The band folds to the configured margin: the head tail's envelope
        // header spans `HEAD_SEQ - BASE_SNAP_SEQ` = 8 commits, well under the
        // fixture's 1000-commit margin, so `max(margin, span) == margin`.
        assert_eq!(
            header.sequence_num,
            compose_tests::HEAD_SEQ + test_config(true).checkpoint_seq_safety_margin,
            "the restore MUST seed from the lineage HEAD (S), not from the base's snap_seq — \
             the latter would re-issue every sequence the replayed tail already used"
        );

        // The tail era references two distinct content hashes; both must be in
        // the live ext store, written strictly before the object.
        let loaded = ext_store
            .load_values(&[&compose_tests::HASH_B, &compose_tests::HASH_C])
            .await
            .expect("load_values after the SR-3.4 pre-store");
        assert_eq!(
            loaded.len(),
            2,
            "both tail-carried external values MUST be present after the restore; got {loaded:?}"
        );

        assert_eq!(hook.admission_len(), 0);
    }

    // ── SR-14.7: on-access restore refuses a base over the entry limit ─────
    //
    // A layer over one delta's entry limit is backed up as a chunked base
    // (`checkpoint::chunked_base`); an earlier release backed it up as one
    // plain OMPF delta, which no decoder accepts. Until on-access restore is
    // retired, the hook answers either shape on both arms before the residue
    // wipe, the ext pre-store and `write_restored_object`, and records no
    // step-down (no `stepped_down_compose`). What it answers is decided by the
    // live layer mapping, which the SDD §22.6 step-2 identity verdict reads
    // first (`adjudicate_over_limit_base_on_access`):
    //
    // - still mapped to the object → `Error(Permanent)`, the refusal;
    // - absent, or mapped to a different object (the layer was deleted or
    //   replaced during the state loss) → `NotRecoverable` with the
    //   `not_recoverable` attempt, the answer apply's identity guard gives a
    //   gone layer, so the client hears NOT_FOUND rather than INTERNAL. The
    //   composed arm's selection returns `Ok(None)` for it and ticks no
    //   lineage cell, not even `exhausted`;
    // - wrong-typed → the classifier's `Error(Permanent)`, not the refusal;
    // - unreadable, or not read within the 10 s adjudication budget →
    //   `Error(Transient)`, so the client retries.
    //
    // The base-only arm's plain over-limit base never reaches that verdict: it
    // fails at `apply_restore`'s decode, behind apply's own identity guard. A
    // genuinely corrupt base keeps its own behaviour: the composed arm steps
    // down, and in these fixtures (which hold no jumped base) the next
    // candidate is the same base as a base-only lineage, which fails at apply's
    // decode.
    //
    // Every test below drives the hook over the same keyspace: the layer
    // mapping as `LiveMapping` stages it (resolving to the object unless the
    // test says otherwise, so apply's identity guard would pass), a wired ext
    // store (so a composed restore that got past the refusal would succeed
    // rather than fail closed on the missing store), surviving residue under
    // the object (an orphan section, which the residue wipe would unlink), and
    // a `LayerStore` whose restore-write seam counts entries into
    // `write_restored_object`'s `WATCH` window. The keyspace dump before and
    // after is the primary evidence that nothing was written; the seam fires
    // only after the pre-`WATCH` probe, so it is the secondary one. A chunked
    // base is the canonical compose fixture's, chunked at 2 entries per chunk
    // (`chunked_canonical_base`): that refusal keys on the artifact's format,
    // not on its size. A plain over-limit base is a real one, one entry over
    // `MAX_DELTA_ENTRIES` (`plain_over_cap_base`), since that refusal keys on
    // the parser's over-limit verdict.

    /// Every `outcome` cell of `live_edit_checkpoint_restore_lineage_total`,
    /// in vocabulary order, for one-equality comparisons.
    fn lineage_cells(metrics: &SharedMetrics) -> Vec<(&'static str, u64)> {
        use crate::observability::metrics::CHECKPOINT_RESTORE_LINEAGE_OUTCOMES;
        CHECKPOINT_RESTORE_LINEAGE_OUTCOMES
            .iter()
            .map(|outcome| (*outcome, lineage_metric_value(metrics, outcome)))
            .collect()
    }

    /// [`lineage_cells`] as it should read when only `ticked` cells moved.
    fn lineage_cells_with(ticked: &[(&str, u64)]) -> Vec<(&'static str, u64)> {
        use crate::observability::metrics::CHECKPOINT_RESTORE_LINEAGE_OUTCOMES;
        CHECKPOINT_RESTORE_LINEAGE_OUTCOMES
            .iter()
            .map(|outcome| {
                let value = ticked
                    .iter()
                    .find(|(name, _)| name == outcome)
                    .map_or(0, |(_, value)| *value);
                (*outcome, value)
            })
            .collect()
    }

    /// What one on-access restore over the SR-14.7 keyspace did.
    struct OnAccessRestore {
        outcome: RecoveryOutcome,
        metrics: SharedMetrics,
        /// Entries into `write_restored_object`'s `WATCH` window.
        restore_writes: usize,
        admission_len: usize,
        before: crate::storage::test_support::structural_commit::ObjectKeyspaceDump,
        after: crate::storage::test_support::structural_commit::ObjectKeyspaceDump,
    }

    /// What the layer mapping `le:{s1}:layer:l` (the layer every generation's
    /// meta names) holds when one on-access restore over the SR-14.7 keyspace
    /// runs, and whether the hook can read it. It is the key the SDD §22.6
    /// step-2 identity verdict reads before an over-limit refusal.
    enum LiveMapping {
        /// Resolves to the object under restore: the layer is still mapped.
        Resolves,
        /// Absent: the layer was deleted during the state loss.
        Gone,
        /// Resolves to a different object: a `CreateLayer` overwrite minted a
        /// new `object_id` during the state loss.
        Replaced,
        /// A hash where a string belongs, so its `GET` answers `WRONGTYPE`.
        WrongTyped,
        /// Resolves, but the hook reads Valkey through a second client whose
        /// data pool closes at the selection's first store read
        /// ([`LayerFetch::DelegateAfterClosingPool`]): after the step-4
        /// re-check, before the identity read. Each attempt of that read then
        /// fails by timing out at `command_timeout_ms`, the client's
        /// per-command timeout (see
        /// `refuse_restore_folds_a_transient_header_read_failure_into_its_own_refusal`
        /// for that failure's shape and why the fixture's own pool must stay
        /// open).
        Unreadable { command_timeout_ms: u64 },
    }

    /// Seed `generations` for `oid` (each `(snap_seq, artifact, meta)`) over
    /// the keyspace the section header describes, with the layer mapping
    /// resolving to `oid`, run one `try_restore_layer(oid)`, and report what
    /// it did.
    async fn restore_on_access(
        tv: &TestValkey,
        oid: u64,
        generations: Vec<(u64, Bytes, CheckpointMeta)>,
    ) -> OnAccessRestore {
        restore_on_access_with(tv, oid, generations, LiveMapping::Resolves).await
    }

    /// [`restore_on_access`] with the layer mapping staged as `mapping`.
    async fn restore_on_access_with(
        tv: &TestValkey,
        oid: u64,
        generations: Vec<(u64, Bytes, CheckpointMeta)>,
        mapping: LiveMapping,
    ) -> OnAccessRestore {
        use crate::storage::external_values::ValkeyExternalValueStore;
        use crate::storage::test_support::structural_commit::{dump_object_keyspace, CommitTokens};
        use crate::storage::watched_txn::WatchWindowHook;
        use fred::clients::Client;
        use fred::interfaces::{HashesInterface, SetsInterface};
        use futures::FutureExt;
        use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

        let mapping_key = tv.key_schema().layer_mapping("s1", "l");
        let mapped_to = match &mapping {
            LiveMapping::Resolves | LiveMapping::Unreadable { .. } => Some(oid),
            LiveMapping::Replaced => Some(oid + 1),
            LiveMapping::Gone | LiveMapping::WrongTyped => None,
        };
        if let Some(mapped_to) = mapped_to {
            let _: () = tv
                .client()
                .pool
                .set(&mapping_key, mapped_to.to_string(), None, None, false)
                .await
                .expect("SET the layer mapping");
        }
        if matches!(mapping, LiveMapping::WrongTyped) {
            let _: () = tv
                .client()
                .pool
                .hset(&mapping_key, [("object_id", oid.to_string())])
                .await
                .expect("HSET the layer mapping as a hash");
        }

        // The client the hook reads Valkey through, and the selection posture
        // that closes it. Everything this helper seeds and dumps goes through
        // the fixture's own client, which stays open.
        let (hook_client, close_at_selection) = match mapping {
            LiveMapping::Unreadable { command_timeout_ms } => {
                let mut doomed_config = test_config(true);
                doomed_config.valkey_url = std::env::var("TEST_VALKEY_URL")
                    .unwrap_or_else(|_| "redis://127.0.0.1:6379".to_string());
                doomed_config.valkey_command_timeout_ms = command_timeout_ms;
                let doomed = crate::storage::test_support::connect_test_client(&doomed_config)
                    .await
                    .expect("a second client to the same test Valkey must connect");
                (doomed.clone(), Some(doomed))
            }
            _ => (tv.client().clone(), None),
        };

        // Residue the wipe would unlink: an orphan section and its index.
        let oid_str = oid.to_string();
        let _: () = tv
            .client()
            .pool
            .sadd(tv.key_schema().obj_sections_index(&oid_str), "77")
            .await
            .expect("SADD obj:secs");
        let _: () = tv
            .client()
            .pool
            .hset(
                tv.key_schema().obj_section(&oid_str, "77"),
                [("name", "stale"), ("version", "7")],
            )
            .await
            .expect("HSET obj:sec");

        let in_mem = InMemoryCheckpointStore::new();
        for (snap_seq, artifact, meta) in generations {
            in_mem
                .put_layer_checkpoint(oid, snap_seq, artifact, meta)
                .await
                .expect("put generation");
        }

        let entered = Arc::new(AtomicUsize::new(0));
        let count: WatchWindowHook = {
            let entered = Arc::clone(&entered);
            Arc::new(move |_client: &Client| {
                entered.fetch_add(1, Ordering::SeqCst);
                async {}.boxed()
            })
        };
        let layer_store = Arc::new(
            LayerStore::new(hook_client, tv.key_schema().clone(), 4096)
                .with_restore_write_watch_window_hook_for_test(count),
        );
        let ext_store: Arc<dyn ExternalValueStore> = Arc::new(ValkeyExternalValueStore::new(
            tv.client().clone(),
            tv.key_schema().clone(),
        ));
        let metrics = make_metrics();
        let store: Arc<dyn CheckpointStore> = match close_at_selection {
            None => Arc::new(in_mem),
            Some(client) => Arc::new(FakeCheckpointStore {
                inner: in_mem,
                fetch: LayerFetch::DelegateAfterClosingPool {
                    client: Box::new(client),
                    closed: AtomicBool::new(false),
                },
            }),
        };
        let hook = RecoveryHook::new(
            store,
            layer_store,
            Arc::new(test_config(true)),
            metrics.clone(),
        )
        .with_ext_value_store(ext_store);

        let before =
            dump_object_keyspace(tv.client(), tv.key_schema(), oid, CommitTokens::Verbatim).await;
        let outcome = hook.try_restore_layer(oid).await;
        let after =
            dump_object_keyspace(tv.client(), tv.key_schema(), oid, CommitTokens::Verbatim).await;

        OnAccessRestore {
            outcome,
            metrics,
            restore_writes: entered.load(Ordering::SeqCst),
            admission_len: hook.admission_len(),
            before,
            after,
        }
    }

    /// Assert `run` is the SR-14.7 refusal: `Error(Permanent)` carrying
    /// `refuse_over_limit_base_on_access`'s own message, and everything
    /// [`assert_failed_permanent_writing_nothing`] asserts.
    fn assert_refused_on_access(run: &OnAccessRestore, oid: u64, what: &str) {
        let text = assert_failed_permanent_writing_nothing(run, oid, what);
        assert!(
            text.contains("over the per-delta entry limit"),
            "{what}: the Permanent must be the over-limit base refusal, not some other failure; \
             got `{text}`"
        );
    }

    /// Assert `run` ended `Error(Permanent)` (returning its rendered text)
    /// with one `error` attempt and nothing else, and everything
    /// [`assert_wrote_nothing`] asserts.
    fn assert_failed_permanent_writing_nothing(
        run: &OnAccessRestore,
        oid: u64,
        what: &str,
    ) -> String {
        let text = match &run.outcome {
            RecoveryOutcome::Error(err @ CheckpointError::Permanent { .. }) => err.to_string(),
            other => panic!("{what}: MUST end Error(Permanent); got {other:?}"),
        };
        assert_attempt_cells(run, (1, 0, 0), what);
        assert_wrote_nothing(run, oid, what);
        text
    }

    /// Assert `run` ended `Error(Transient)` (returning its rendered text)
    /// with one `error` attempt and nothing else, and everything
    /// [`assert_wrote_nothing`] asserts.
    fn assert_failed_transient_writing_nothing(
        run: &OnAccessRestore,
        oid: u64,
        what: &str,
    ) -> String {
        let text = match &run.outcome {
            RecoveryOutcome::Error(err @ CheckpointError::Transient { .. }) => err.to_string(),
            other => {
                panic!("{what}: MUST end Error(Transient), so the client retries; got {other:?}")
            }
        };
        assert_attempt_cells(run, (1, 0, 0), what);
        assert_wrote_nothing(run, oid, what);
        text
    }

    /// Assert `run` ended `NotRecoverable` — the answer apply's identity
    /// guard gives a gone layer, which the handlers send as NOT_FOUND — with
    /// one `not_recoverable` attempt and nothing else, and everything
    /// [`assert_wrote_nothing`] asserts.
    fn assert_not_recoverable_writing_nothing(run: &OnAccessRestore, oid: u64, what: &str) {
        assert!(
            matches!(run.outcome, RecoveryOutcome::NotRecoverable),
            "{what}: MUST end NotRecoverable (NOT_FOUND to the client), not the over-limit \
             refusal's Error(Permanent) (INTERNAL); got {:?}",
            run.outcome
        );
        assert_attempt_cells(run, (0, 0, 1), what);
        assert_wrote_nothing(run, oid, what);
    }

    /// Assert `run`'s recovery-attempt cells read `(error, restored,
    /// not_recoverable)`.
    fn assert_attempt_cells(run: &OnAccessRestore, expected: (u64, u64, u64), what: &str) {
        assert_eq!(
            (
                metric_value(&run.metrics, "error"),
                metric_value(&run.metrics, "restored"),
                metric_value(&run.metrics, "not_recoverable"),
            ),
            expected,
            "{what}: exactly one attempt, on the expected cell (error, restored, not_recoverable)"
        );
    }

    /// Assert `run` never entered the write path and left the object's
    /// keyspace — residue included — exactly as it was, with its admission
    /// entry retired.
    fn assert_wrote_nothing(run: &OnAccessRestore, oid: u64, what: &str) {
        assert_eq!(
            run.restore_writes, 0,
            "{what}: write_restored_object MUST never be entered"
        );
        assert!(
            run.before.get("obj:hdr").is_none() && run.after.get("obj:hdr").is_none(),
            "{what}: no obj:hdr may be written"
        );
        assert!(
            run.before.get("obj:sec:77").is_some(),
            "precondition: the residue was seeded under object {oid}"
        );
        assert_eq!(
            run.after, run.before,
            "{what}: the object's keyspace MUST be untouched — no key written, the residue not \
             wiped"
        );
        assert_eq!(run.admission_len, 0, "{what}: the admission entry retires");
    }

    /// The generations of a composed lineage over `base` at the canonical
    /// fixture's `BASE_SNAP_SEQ`, with the canonical tail up to `HEAD_SEQ`,
    /// plus — when `older` is given — a restorable plain base at
    /// `OLDER_BASE_SEQ` underneath it.
    fn composed_generations(
        base: Bytes,
        older: Option<Bytes>,
    ) -> Vec<(u64, Bytes, CheckpointMeta)> {
        use crate::checkpoint::compose::tests as compose_tests;

        let (history, _plain) = compose_tests::canonical();
        let tail = compose_tests::tail(
            compose_tests::BASE_SNAP_SEQ,
            compose_tests::BASE_SNAP_SEQ,
            &history.records_in(compose_tests::BASE_SNAP_SEQ, compose_tests::HEAD_SEQ),
        );
        let mut generations: Vec<(u64, Bytes, CheckpointMeta)> = older
            .map(|bytes| (OLDER_BASE_SEQ, bytes, selection_base_meta(OLDER_BASE_SEQ)))
            .into_iter()
            .collect();
        generations.push((
            compose_tests::BASE_SNAP_SEQ,
            base,
            selection_base_meta(compose_tests::BASE_SNAP_SEQ),
        ));
        generations.push((
            compose_tests::HEAD_SEQ,
            tail,
            selection_tail_meta(
                compose_tests::BASE_SNAP_SEQ,
                compose_tests::BASE_SNAP_SEQ,
                compose_tests::HEAD_SEQ,
            ),
        ));
        generations
    }

    /// Where [`composed_generations`] puts its older, restorable base: below
    /// the canonical fixture's `BASE_SNAP_SEQ`.
    const OLDER_BASE_SEQ: u64 = 3;

    /// The canonical fixture's plain base at [`OLDER_BASE_SEQ`].
    fn older_plain_base() -> Bytes {
        use crate::checkpoint::compose::tests as compose_tests;

        let (history, _plain) = compose_tests::canonical();
        compose_tests::base_ompf(history.state_at(OLDER_BASE_SEQ))
    }

    /// The canonical compose fixture's base at `BASE_SNAP_SEQ`, chunked at 2
    /// entries per chunk: the chunked base every chunked-base test in this
    /// section restores.
    fn chunked_canonical_base() -> Bytes {
        use crate::checkpoint::chunked_base::tests::chunked_base_of;
        use crate::checkpoint::compose::tests as compose_tests;

        let (history, _plain) = compose_tests::canonical();
        Bytes::from(chunked_base_of(
            history.state_at(compose_tests::BASE_SNAP_SEQ),
            2,
        ))
    }

    /// A lineage that is just [`chunked_canonical_base`].
    fn base_only_chunked_generations() -> Vec<(u64, Bytes, CheckpointMeta)> {
        use crate::checkpoint::compose::tests as compose_tests;

        vec![(
            compose_tests::BASE_SNAP_SEQ,
            chunked_canonical_base(),
            selection_base_meta(compose_tests::BASE_SNAP_SEQ),
        )]
    }

    /// Base-only arm: a lineage that is just a chunked base is refused right
    /// after selection. Selection itself succeeded — the `gathered_base` cell
    /// is ticked — and the refusal is the hook body's.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn on_access_restore_refuses_a_base_only_lineage_with_a_chunked_base() {
        use crate::checkpoint::compose::tests as compose_tests;

        let tv = TestValkey::setup().await;
        let oid = compose_tests::OBJECT_ID;

        let run = restore_on_access(&tv, oid, base_only_chunked_generations()).await;

        assert_refused_on_access(&run, oid, "a base-only lineage with a chunked base");
        assert_eq!(
            lineage_cells(&run.metrics),
            lineage_cells_with(&[("gathered_base", 1)]),
            "the base-only lineage was gathered, and nothing stepped down"
        );
    }

    /// Composed arm: a chunked base under the canonical tail is refused
    /// inside the selection, before composing, and records no step-down. The
    /// store also holds an older restorable plain base; the refusal leaves it
    /// alone too.
    ///
    /// What discriminates is `lineage_cells == []`. Had the composed arm
    /// stepped down, it would have recorded `stepped_down_compose`, and
    /// `next_restorable` would then have offered the next candidate below the
    /// tail's head — on this fixture, which holds no jumped base, the same
    /// chunked base as a base-only lineage (`gathered_base`), which the
    /// base-only arm refuses. The older base is therefore never a candidate
    /// here, stepping down or not; the step-down `lineage_cells == []` rules
    /// out is that one. The plain over-limit counterpart is
    /// `on_access_restore_refuses_a_composed_lineage_over_an_earlier_releases_plain_over_limit_base`.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn on_access_restore_refusing_a_chunked_base_on_the_composed_arm_records_no_step_down() {
        use crate::checkpoint::compose::tests as compose_tests;

        let tv = TestValkey::setup().await;
        let oid = compose_tests::OBJECT_ID;

        let run = restore_on_access(
            &tv,
            oid,
            composed_generations(chunked_canonical_base(), Some(older_plain_base())),
        )
        .await;

        assert_refused_on_access(
            &run,
            oid,
            "a chunked newest lineage over a restorable older base",
        );
        assert_eq!(
            lineage_cells(&run.metrics),
            lineage_cells_with(&[]),
            "no step-down (no `stepped_down_compose`) and nothing gathered"
        );
    }

    /// The control for the refusals in this section: the same keyspace and
    /// generations as the no-step-down tests, with the canonical below-cap
    /// plain base as the newest base, restore on access — composed, through
    /// `write_restored_object`, with the residue wiped. This shows that their
    /// "write path never entered" and "keyspace untouched" assertions can fail
    /// on this fixture. The older base is not used here either: the newest
    /// lineage restores.
    ///
    /// Below-cap restore on access is otherwise pinned by
    /// `try_restore_layer_restored_arm` (base-only) and
    /// `composed_restore_seeds_from_the_lineage_head_and_pre_stores_ext_values`
    /// (composed).
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn on_access_restore_of_the_same_lineage_with_a_plain_base_restores_through_the_write_path(
    ) {
        use crate::checkpoint::compose::tests as compose_tests;

        let tv = TestValkey::setup().await;
        let oid = compose_tests::OBJECT_ID;
        let (_history, plain) = compose_tests::canonical();

        let run = restore_on_access(
            &tv,
            oid,
            composed_generations(plain, Some(older_plain_base())),
        )
        .await;

        assert!(
            matches!(run.outcome, RecoveryOutcome::Restored),
            "the plain-base lineage MUST restore; got {:?}",
            run.outcome
        );
        assert_eq!(metric_value(&run.metrics, "restored"), 1);
        assert_eq!(
            lineage_cells(&run.metrics),
            lineage_cells_with(&[("gathered_composed", 1)])
        );
        assert!(
            run.restore_writes > 0,
            "the restore MUST have entered write_restored_object's WATCH window"
        );
        assert!(
            run.after.get("obj:hdr").is_some(),
            "the restore wrote obj:hdr"
        );
        assert!(
            run.before.get("obj:sec:77").is_some() && run.after.get("obj:sec:77").is_none(),
            "the residue was there and was wiped"
        );
        assert_eq!(run.admission_len, 0);
    }

    /// Composed arm, an earlier release's base: a plain OMPF base one entry
    /// over `MAX_DELTA_ENTRIES`, under the canonical tail, with an older
    /// restorable base below it. The composition's base decode fails with the
    /// parser's over-limit verdict, and the hook refuses the lineage there:
    /// the refusal's message, one `error`, no lineage cell (no
    /// `stepped_down_compose`, nothing gathered), nothing written.
    ///
    /// Without the over-limit check the composed arm would treat the failure
    /// as a corrupt base: record `stepped_down_compose`, then — this fixture
    /// holding no jumped base — take the same base as the next candidate, a
    /// base-only lineage (`gathered_base`), and fail at apply's
    /// decode — still `Error(Permanent)`, but a step-down, and not the
    /// refusal.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn on_access_restore_refuses_a_composed_lineage_over_an_earlier_releases_plain_over_limit_base(
    ) {
        use crate::checkpoint::chunked_base::tests::plain_over_cap_base;
        use crate::checkpoint::compose::tests as compose_tests;

        let tv = TestValkey::setup().await;
        let oid = compose_tests::OBJECT_ID;

        let run = restore_on_access(
            &tv,
            oid,
            composed_generations(Bytes::from(plain_over_cap_base()), Some(older_plain_base())),
        )
        .await;

        assert_refused_on_access(
            &run,
            oid,
            "a composed lineage over an earlier release's plain over-limit base",
        );
        assert_eq!(
            lineage_cells(&run.metrics),
            lineage_cells_with(&[]),
            "no step-down (no `stepped_down_compose`) and nothing gathered"
        );
    }

    /// Composed arm, a genuinely corrupt base keeps its behaviour: an OMPF
    /// whose delta is not OODF fails the composition's base decode with a
    /// parser error other than the over-limit one, so the composed arm steps
    /// down (`stepped_down_compose`). On this fixture, which holds no jumped
    /// base, the next candidate below the tail's head is the same base as a
    /// base-only lineage (`gathered_base`), which
    /// fails `Permanent` at apply's decode — the base-only arm's documented
    /// answer to a corrupt base, which never steps down. Nothing is written,
    /// and the older base below is not restored.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn on_access_restore_of_a_composed_lineage_over_a_corrupt_base_steps_down_and_fails_at_the_base_decode(
    ) {
        use crate::checkpoint::compose::tests as compose_tests;

        let tv = TestValkey::setup().await;
        let oid = compose_tests::OBJECT_ID;
        let corrupt = Bytes::from(
            OmpfContainer::serialize(&b"definitely not an OODF delta"[..], &[])
                .expect("OMPF serialize"),
        );

        let run = restore_on_access(
            &tv,
            oid,
            composed_generations(corrupt, Some(older_plain_base())),
        )
        .await;

        let text = assert_failed_permanent_writing_nothing(
            &run,
            oid,
            "a composed lineage over a corrupt base",
        );
        assert!(
            text.contains("OODF parse failed during checkpoint apply")
                && !text.contains("over the per-delta entry limit"),
            "the corrupt base fails at apply's decode, not as the over-limit refusal; got `{text}`"
        );
        assert_eq!(
            lineage_cells(&run.metrics),
            lineage_cells_with(&[("stepped_down_compose", 1), ("gathered_base", 1)]),
            "the composed arm stepped down, and the same base came back base-only"
        );
    }

    /// Base-only arm, an earlier release's base: a lone plain OMPF base one
    /// entry over `MAX_DELTA_ENTRIES` is not refused by the hook body (that
    /// check is for a chunked base); it fails `Permanent` at
    /// `apply_restore`'s decode with the parser's over-limit verdict, before
    /// the residue wipe, the ext pre-store and the write. Selection gathered
    /// it (`gathered_base`) and nothing stepped down.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn on_access_restore_of_a_base_only_lineage_over_an_earlier_releases_plain_over_limit_base_fails_permanent_and_writes_nothing(
    ) {
        use crate::checkpoint::chunked_base::tests::plain_over_cap_base;
        use crate::checkpoint::compose::tests as compose_tests;

        let tv = TestValkey::setup().await;
        let oid = compose_tests::OBJECT_ID;

        let run = restore_on_access(
            &tv,
            oid,
            vec![(
                compose_tests::BASE_SNAP_SEQ,
                Bytes::from(plain_over_cap_base()),
                selection_base_meta(compose_tests::BASE_SNAP_SEQ),
            )],
        )
        .await;

        assert_failed_permanent_writing_nothing(
            &run,
            oid,
            "a base-only lineage over an earlier release's plain over-limit base",
        );
        let RecoveryOutcome::Error(err) = &run.outcome else {
            unreachable!("asserted above");
        };
        assert!(
            is_over_entry_limit_base(err),
            "the failure is the decode's typed over-limit verdict; got {err:?}"
        );
        assert_eq!(
            lineage_cells(&run.metrics),
            lineage_cells_with(&[("gathered_base", 1)]),
            "the base-only lineage was gathered, and nothing stepped down"
        );
    }

    // ── SR-14.7: the identity verdict that precedes the refusal ────────────
    //
    // The tests above stage a layer that is still mapped, so they pin the
    // refusal itself: `on_access_restore_refuses_a_base_only_lineage_with_a_chunked_base`,
    // `on_access_restore_refusing_a_chunked_base_on_the_composed_arm_records_no_step_down`
    // and
    // `on_access_restore_refuses_a_composed_lineage_over_an_earlier_releases_plain_over_limit_base`,
    // one per refusal site. The tests below stage every other answer of the
    // layer-mapping read, on the same keyspace. A gone layer is driven at all
    // three sites, because each site maps the verdict itself; the other
    // answers pass through the sites unmapped and are driven on the base-only
    // arm.

    /// Base-only arm, chunked base, layer deleted during the state loss: the
    /// identity verdict finds no mapping, so the hook answers `NotRecoverable`
    /// on the `not_recoverable` attempt, as apply's identity guard answers a
    /// gone layer, instead of refusing. Selection gathered the lineage
    /// (`gathered_base`) and nothing stepped down.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn on_access_restore_of_a_base_only_chunked_lineage_whose_layer_is_gone_is_not_recoverable(
    ) {
        use crate::checkpoint::compose::tests as compose_tests;

        let tv = TestValkey::setup().await;
        let oid = compose_tests::OBJECT_ID;

        let run =
            restore_on_access_with(&tv, oid, base_only_chunked_generations(), LiveMapping::Gone)
                .await;

        assert_not_recoverable_writing_nothing(
            &run,
            oid,
            "a base-only chunked lineage whose layer is gone",
        );
        assert_eq!(
            lineage_cells(&run.metrics),
            lineage_cells_with(&[("gathered_base", 1)]),
            "the base-only lineage was gathered, and nothing stepped down"
        );
    }

    /// Base-only arm, chunked base, layer replaced during the state loss: the
    /// mapping resolves to a different object, which the identity verdict
    /// treats as a gone layer — `NotRecoverable`, not the refusal.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn on_access_restore_of_a_base_only_chunked_lineage_whose_layer_was_replaced_is_not_recoverable(
    ) {
        use crate::checkpoint::compose::tests as compose_tests;

        let tv = TestValkey::setup().await;
        let oid = compose_tests::OBJECT_ID;

        let run = restore_on_access_with(
            &tv,
            oid,
            base_only_chunked_generations(),
            LiveMapping::Replaced,
        )
        .await;

        assert_not_recoverable_writing_nothing(
            &run,
            oid,
            "a base-only chunked lineage whose layer now maps to another object",
        );
        assert_eq!(
            lineage_cells(&run.metrics),
            lineage_cells_with(&[("gathered_base", 1)]),
            "the base-only lineage was gathered, and nothing stepped down"
        );
    }

    /// Composed arm, chunked base, layer gone: the selection answers
    /// `Ok(None)` before composing, and the hook answers `NotRecoverable`.
    ///
    /// `lineage_cells == []` is what separates this `Ok(None)` from an
    /// exhaustion, which would tick `exhausted`, and from a step-down, which
    /// would tick `stepped_down_compose` and then reach the same base
    /// base-only (`gathered_base`). The older restorable base below is not
    /// restored either.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn on_access_restore_of_a_composed_chunked_lineage_whose_layer_is_gone_is_not_recoverable_with_no_lineage_cell(
    ) {
        use crate::checkpoint::compose::tests as compose_tests;

        let tv = TestValkey::setup().await;
        let oid = compose_tests::OBJECT_ID;

        let run = restore_on_access_with(
            &tv,
            oid,
            composed_generations(chunked_canonical_base(), Some(older_plain_base())),
            LiveMapping::Gone,
        )
        .await;

        assert_not_recoverable_writing_nothing(
            &run,
            oid,
            "a composed chunked lineage whose layer is gone",
        );
        assert_eq!(
            lineage_cells(&run.metrics),
            lineage_cells_with(&[]),
            "no lineage cell: not `exhausted`, no step-down, nothing gathered"
        );
    }

    /// Composed arm, an earlier release's plain OMPF base over the entry
    /// limit, layer gone: the composition's base decode fails with the
    /// over-limit verdict, the identity verdict finds no mapping, and the
    /// hook answers `NotRecoverable` with no lineage cell, as for the chunked
    /// base above.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn on_access_restore_of_a_composed_lineage_over_a_plain_over_limit_base_whose_layer_is_gone_is_not_recoverable_with_no_lineage_cell(
    ) {
        use crate::checkpoint::chunked_base::tests::plain_over_cap_base;
        use crate::checkpoint::compose::tests as compose_tests;

        let tv = TestValkey::setup().await;
        let oid = compose_tests::OBJECT_ID;

        let run = restore_on_access_with(
            &tv,
            oid,
            composed_generations(Bytes::from(plain_over_cap_base()), Some(older_plain_base())),
            LiveMapping::Gone,
        )
        .await;

        assert_not_recoverable_writing_nothing(
            &run,
            oid,
            "a composed lineage over a plain over-limit base whose layer is gone",
        );
        assert_eq!(
            lineage_cells(&run.metrics),
            lineage_cells_with(&[]),
            "no lineage cell: not `exhausted`, no step-down, nothing gathered"
        );
    }

    /// A wrong-typed layer mapping (a hash where the string belongs): the
    /// mapping read's classifier answers `Permanent`, and that `Permanent`
    /// reaches the client unchanged — not the refusal's, whose message names
    /// the entry limit, and not a retriable `Transient`, since no retry
    /// re-types a key.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn on_access_restore_of_a_chunked_base_with_a_wrong_typed_layer_mapping_fails_permanent_with_the_classifiers_error(
    ) {
        use crate::checkpoint::compose::tests as compose_tests;

        let tv = TestValkey::setup().await;
        let oid = compose_tests::OBJECT_ID;

        let run = restore_on_access_with(
            &tv,
            oid,
            base_only_chunked_generations(),
            LiveMapping::WrongTyped,
        )
        .await;

        let text = assert_failed_permanent_writing_nothing(
            &run,
            oid,
            "a chunked base whose layer mapping is wrong-typed",
        );
        assert!(
            text.contains("GET layer mapping (restore identity guard)")
                && !text.contains("over the per-delta entry limit"),
            "the Permanent MUST be the mapping read's WRONGTYPE verdict, not the over-limit \
             refusal; got `{text}`"
        );
    }

    /// The layer-mapping read fails: every attempt meets a closed pool and
    /// times out. The hook answers `Error(Transient)` on the `error` attempt,
    /// so the client retries — not the refusal's `Permanent`, which would
    /// tell a gone layer INTERNAL on a store blip, and not `NotRecoverable`,
    /// which the failed read cannot justify.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn on_access_restore_of_a_chunked_base_whose_layer_mapping_read_fails_answers_transient()
    {
        use crate::checkpoint::compose::tests as compose_tests;

        let tv = TestValkey::setup().await;
        let oid = compose_tests::OBJECT_ID;

        // 500 ms per attempt, for the reason the same value is given in
        // `refuse_restore_folds_a_transient_header_read_failure_into_its_own_refusal`:
        // the failure is a timeout, so this is the test's wall-clock cost per
        // attempt. Three attempts stay far inside the adjudication budget.
        let run = restore_on_access_with(
            &tv,
            oid,
            base_only_chunked_generations(),
            LiveMapping::Unreadable {
                command_timeout_ms: 500,
            },
        )
        .await;

        let text = assert_failed_transient_writing_nothing(
            &run,
            oid,
            "a chunked base whose layer-mapping read fails",
        );
        assert!(
            text.contains("GET layer mapping (restore identity guard)")
                && !text.contains("adjudication budget"),
            "the Transient MUST be the failed mapping read's, not the budget elapse's; got `{text}`"
        );
        assert_eq!(
            lineage_cells(&run.metrics),
            lineage_cells_with(&[("gathered_base", 1)]),
            "the base-only lineage was gathered, and nothing stepped down"
        );
    }

    /// The layer-mapping read never answers within the adjudication budget:
    /// each attempt's per-command timeout is twice the budget, so only the
    /// budget can end the wait. The hook answers the elapse's own
    /// `Error(Transient)` on the `error` attempt, at the budget rather than
    /// after the retries' full timeouts.
    ///
    /// Costs `RECOVERY_STREAM_ADJUDICATION_BUDGET` (10 s) of wall time: the
    /// budget is a const and the read a real Valkey round trip, so there is no
    /// clock to pause.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn on_access_restore_of_a_chunked_base_whose_layer_mapping_read_outlasts_the_adjudication_budget_answers_transient(
    ) {
        use crate::checkpoint::compose::tests as compose_tests;

        let tv = TestValkey::setup().await;
        let oid = compose_tests::OBJECT_ID;

        let run = restore_on_access_with(
            &tv,
            oid,
            base_only_chunked_generations(),
            LiveMapping::Unreadable {
                command_timeout_ms: 2 * RECOVERY_STREAM_ADJUDICATION_BUDGET.as_millis() as u64,
            },
        )
        .await;

        let text = assert_failed_transient_writing_nothing(
            &run,
            oid,
            "a chunked base whose layer-mapping read outlasts the adjudication budget",
        );
        assert!(
            text.contains("adjudication budget"),
            "the Transient MUST be the budget elapse's, not a failed read's; got `{text}`"
        );
    }

    // ── b.4et: the stream-key classification and its budgets ──────────────

    /// `TYPE` must separate all three states the probe script's single `nil`
    /// covers, against a real server.
    ///
    /// This is the round-trip the whole b.4et fix rests on: `Absent` proceeds,
    /// `Stream` refuses, `Foreign` proceeds-but-reports. Folding any two of
    /// them back together reintroduces the defect, and the folding would happen
    /// HERE — in the `match ty.as_str()` — where no integration test can see
    /// it directly (they observe the consequence, several steps downstream,
    /// and a mis-classification that happens to reach the same verdict on the
    /// arm they stage would pass).
    ///
    /// Driven against a live Valkey rather than a hand-built enum because the
    /// claim is about the exact reply strings Valkey emits, which is precisely
    /// what a unit test over the enum could not check.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn read_stream_key_type_separates_absent_stream_and_foreign() {
        use fred::interfaces::StreamsInterface;

        let tv = TestValkey::setup().await;
        let layer_store = Arc::new(LayerStore::new(
            tv.client().clone(),
            tv.key_schema().clone(),
            4096,
        ));
        let store: Arc<dyn CheckpointStore> = Arc::new(InMemoryCheckpointStore::new());
        let hook = RecoveryHook::new(
            store,
            layer_store,
            Arc::new(test_config(true)),
            make_metrics(),
        );

        // 1. Absent — nothing was ever written under this key.
        let absent_key = tv.key_schema().layer_stream(9_100_001);
        assert!(
            matches!(
                hook.read_stream_key_type(&absent_key).await,
                Ok(StreamKeyType::Absent)
            ),
            "a key that does not exist MUST classify as Absent — this is the arm that lets a \
             full keyspace loss restore at all"
        );

        // 2. Stream — ours, and its top is what we could not read.
        let stream_key = tv.key_schema().layer_stream(9_100_002);
        let _: String = tv
            .client()
            .pool
            .xadd(
                &stream_key,
                false,
                None,
                "1-0",
                vec![("payload", fred::types::Value::from("x"))],
            )
            .await
            .expect("XADD must succeed");
        assert!(
            matches!(
                hook.read_stream_key_type(&stream_key).await,
                Ok(StreamKeyType::Stream)
            ),
            "a real stream MUST classify as Stream — this is the ONLY arm that refuses a \
             restore, so folding it into Absent is the b.4et defect itself"
        );

        // 3. Foreign — someone else's value on our keyspace.
        let foreign_key = tv.key_schema().layer_stream(9_100_003);
        let _: () = tv
            .client()
            .pool
            .set(&foreign_key, "not-a-stream", None, None, false)
            .await
            .expect("SET must succeed");
        match hook.read_stream_key_type(&foreign_key).await {
            Ok(StreamKeyType::Foreign(actual)) => assert_eq!(
                actual, "string",
                "the Foreign variant MUST carry the type Valkey reported, so the operator WARN \
                 can name it; got {actual:?}"
            ),
            other => panic!(
                "a non-stream value MUST classify as Foreign, not Absent (which would silently \
                 proceed with no report) and not Stream (which would refuse a restore over a key \
                 that is not even ours). Got {other:?}"
            ),
        }
    }

    /// The two b.4et budgets are consistent with each other, with the
    /// *derivation* their rustdoc gives, and with the ceiling SDD §22.5
    /// records.
    ///
    /// Both are module consts rather than config knobs, so nothing else in the
    /// tree can witness a change to them — and the two are not independent: the
    /// retry count is only defensible *because* the overall deadline cuts the
    /// worst case short. A future edit that raised `max_attempts` without
    /// touching the budget would silently lengthen the per-`object_id`
    /// admission-mutex hold, which is the cost the whole design is bounding.
    ///
    /// # Relationships, not digits
    ///
    /// The budget is asserted against **2 × the default
    /// `LIVE_EDIT_VALKEY_COMMAND_TIMEOUT_MS`**, read off `Config::from_env`
    /// with that variable explicitly unset, rather than against the literal
    /// `10 s`. A bare `== Duration::from_secs(10)` restates the constant it is
    /// reading and can only fail when someone edits the very line above it; the
    /// derived form additionally fails when the *default command timeout*
    /// moves, which is the edit that would silently invalidate the rustdoc's
    /// "one round-trip lost to a connection drop plus one full re-attempt"
    /// justification while leaving the number that expresses it untouched.
    ///
    /// `budget + grace` is asserted at `<= 11 s` for the same reason: 11 s is
    /// the ceiling SDD §22.5 step 6 records (10 s plus a 1 s grace), so it is a
    /// promise with a reader outside this file. The previous `<= 15 s` form admitted a
    /// grace four times its documented size without going red.
    ///
    /// `max_attempts == 3` is deliberately NOT re-asserted here: it is a bare
    /// mirror of the constant, and the property that actually matters about it
    /// — that its whole backoff budget stays negligible against the deadline —
    /// is derived below.
    #[test]
    fn the_stream_adjudication_budgets_stay_within_their_documented_bounds() {
        // Backoff sleep is the only thing the retry config bounds, and it must
        // stay a rounding error against the deadline that bounds the rest.
        let worst_case_sleep_ms = u64::from(RECOVERY_STREAM_PROBE_RETRY.max_attempts - 1)
            * RECOVERY_STREAM_PROBE_RETRY.cap_delay_ms;
        assert!(
            worst_case_sleep_ms <= 100,
            "the retry's whole backoff budget ({worst_case_sleep_ms} ms) must stay negligible \
             against the adjudication deadline — it is there to space out reconnect attempts, \
             not to wait anything out"
        );

        // The default command timeout, read from the one place that defines it.
        // `LIVE_EDIT_VALKEY_COMMAND_TIMEOUT_MS` is unset inside the closure so
        // an ambient value in the developer's or runner's environment cannot
        // quietly redefine what "the default" means here.
        let default_command_timeout = temp_env::with_vars(
            [
                (
                    "LIVE_EDIT_STORAGE_SERVICE_ENDPOINT",
                    Some("http://auth.example.com"),
                ),
                (
                    "LIVE_EDIT_USERINFO_ENDPOINT",
                    Some("http://userinfo.example.com"),
                ),
                ("LIVE_EDIT_VALKEY_COMMAND_TIMEOUT_MS", None),
            ],
            || {
                Duration::from_millis(
                    Config::from_env_with(PodMemorySource::Reference)
                        .expect("Config::from_env must succeed with the two required endpoints set")
                        .valkey_command_timeout_ms,
                )
            },
        );

        assert_eq!(
            RECOVERY_STREAM_ADJUDICATION_BUDGET,
            2 * default_command_timeout,
            "the overall deadline is 2x the default LIVE_EDIT_VALKEY_COMMAND_TIMEOUT_MS \
             ({default_command_timeout:?}) — the smallest value that still rides out one \
             round-trip lost to a connection drop plus one full re-attempt after fred \
             reconnects. If the default command timeout moved, this deadline's whole \
             justification moved with it and the const has to be re-derived deliberately, not \
             left at whatever it happened to be"
        );
        assert!(
            RECOVERY_STREAM_ADJUDICATION_PEER_GRACE < RECOVERY_STREAM_ADJUDICATION_BUDGET,
            "the peer-landed re-check grace ({:?}) must be strictly smaller than the budget it \
             follows ({:?}). It exists to give a merely-slow store one last chance to report a \
             peer's completed restore; a grace at or above the budget would double the mutex \
             hold on exactly the store that just failed to answer within it",
            RECOVERY_STREAM_ADJUDICATION_PEER_GRACE,
            RECOVERY_STREAM_ADJUDICATION_BUDGET
        );
        assert!(
            RECOVERY_STREAM_ADJUDICATION_BUDGET + RECOVERY_STREAM_ADJUDICATION_PEER_GRACE
                <= Duration::from_secs(11),
            "budget + grace is the HARD ceiling on how long one client can hold a layer's \
             admission mutex inside this adjudication, and 11 s is the figure SDD §22.5 step 6 \
             records (10 s plus a 1 s grace). Exceeding it here does not just lengthen a mutex \
             hold, it makes the design record wrong"
        );
    }

    // ═════════════════════════════════════════════════════════════════════════
    // `refuse_restore`'s two `Err` arms on the peer-landed re-check
    // ═════════════════════════════════════════════════════════════════════════
    //
    // Driven by calling `refuse_restore` directly rather than through the hook,
    // and that is forced rather than convenient. Every caller runs downstream of
    // `probe_stream_high_water`, which runs downstream of the post-acquire
    // `is_layer_present` re-check whose own `Err` arm already propagates — so a
    // wrong-typed `obj:hdr` is refused before anything reaches this function.
    // The arm is defence in depth, and the ONLY way to observe it is to address
    // it directly. `refuse_restore` is a private method, so this lane is the
    // only one that can.
    //
    // All three arms are covered, one case each. The *transient* `Err(_) => {}`
    // arm needs no fault-injection seam and disturbs no shared node: closing
    // the fixture's OWN data pool (`ValkeyClient::pool`, which
    // `is_layer_present` → `apply::header_is_structurally_complete` reads
    // through) before the call makes that one `HGETALL obj:hdr` fail with a
    // non-`WRONGTYPE` fred error, which `classify_object_key_read_error` maps
    // to `Transient` — see
    // `..._folds_a_transient_header_read_failure_into_its_own_refusal`.
    //
    // Note what discriminates that case, because the return type does NOT: a
    // blanket `Err(e) => return Err(e)` regression would propagate a
    // `CheckpointError::Transient` too (the classification is already
    // `Transient`), so `matches!(err, Transient)` passes either way. What the
    // regression drops is the refusal's own `checkpoint_recovery_stream_
    // unresolved` tick and its WARN — the transient blip leaves this path
    // silently. The counter assertion is therefore the load-bearing one, and
    // the error-text assertion beside it names WHICH `Transient` came back.

    /// The `Permanent` arm: a wrong-typed `obj:hdr` propagates verbatim, and
    /// this function ticks NOTHING.
    ///
    /// Both halves are the contract. Folding the `Permanent` into the ordinary
    /// refusal would re-report an operator-actionable permanent fault as this
    /// function's `Transient`, which the caller retries forever. And the silence
    /// is deliberate: the verdict carried is the header fault, not an unresolved
    /// stream, and the read that produced it already emitted its own `error!` on
    /// the `WRONGTYPE` branch — a tick here would inflate the cell an operator
    /// alerts on with an event that is not what the cell means.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn refuse_restore_propagates_a_wrongtyped_header_as_permanent_and_ticks_nothing() {
        let tv = TestValkey::setup().await;
        let layer_store = Arc::new(LayerStore::new(
            tv.client().clone(),
            tv.key_schema().clone(),
            4096,
        ));
        let store: Arc<dyn CheckpointStore> = Arc::new(InMemoryCheckpointStore::new());
        let metrics = make_metrics();
        let hook = RecoveryHook::new(
            store,
            layer_store,
            Arc::new(test_config(true)),
            metrics.clone(),
        );

        let oid: u64 = 9_601;
        // A String where a Hash belongs: `HGETALL` answers `WRONGTYPE`, which
        // `classify_object_key_read_error` maps to `Permanent`. Retyping the key
        // is the whole fixture — nothing in this service ever writes `obj:hdr`
        // as a String, so this is precisely the out-of-band-writer shape the
        // arm exists for.
        let _: () = tv
            .client()
            .pool
            .set(
                tv.key_schema().obj_header(&oid.to_string()),
                "not-a-hash",
                None,
                None,
                false,
            )
            .await
            .expect("SET obj:hdr as a String");

        let before = stream_unresolved_cells(&metrics);
        let err = hook
            .refuse_restore(oid, "probe_unreadable", "synthetic detail".to_owned())
            .await
            .expect_err("a refusal never returns Ok(Some(..)) and this one cannot return Ok(None)");

        assert!(
            matches!(err, CheckpointError::Permanent { .. }),
            "a wrong-typed obj:hdr must propagate VERBATIM as Permanent. A Transient here is the \
             regression: the caller's retry loop would hammer a fault no retry can heal, and the \
             operator's remediation (remove the foreign key) never surfaces. Got {err:?}"
        );
        assert_eq!(
            stream_unresolved_cells(&metrics),
            before,
            "the Permanent arm ticks NO unresolved-stream cell — not the `probe_unreadable` it was \
             called with, and not any other. The verdict is the header fault, and the read that \
             produced it already logged it"
        );
    }

    /// The ordinary refusal: no header at all, so the peer-landed re-check
    /// answers `Ok(false)` and the function reports its own verdict — one tick
    /// on the reason it was called with, and `Transient`.
    ///
    /// The counterweight to the case above. Together they pin that the
    /// classification is per-arm rather than blanket: this one MUST tick and
    /// MUST be retriable, that one MUST NOT tick and MUST NOT be retried.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn refuse_restore_with_no_header_ticks_its_reason_and_returns_transient() {
        let tv = TestValkey::setup().await;
        let layer_store = Arc::new(LayerStore::new(
            tv.client().clone(),
            tv.key_schema().clone(),
            4096,
        ));
        let store: Arc<dyn CheckpointStore> = Arc::new(InMemoryCheckpointStore::new());
        let metrics = make_metrics();
        let hook = RecoveryHook::new(
            store,
            layer_store,
            Arc::new(test_config(true)),
            metrics.clone(),
        );

        // Nothing seeded: `obj:hdr` is absent, so the re-check reads an empty
        // field map and reports "not structurally complete" without erroring.
        let oid: u64 = 9_602;

        let before = stream_unresolved_cells(&metrics);
        let err = hook
            .refuse_restore(oid, "unlink_unverified", "synthetic detail".to_owned())
            .await
            .expect_err("an unresolved stream with no peer-landed header must refuse");

        assert!(
            matches!(err, CheckpointError::Transient { .. }),
            "the unresolved-stream refusal is retriable by contract — the caller sees UNAVAILABLE \
             and may re-drive. Got {err:?}"
        );

        let after = stream_unresolved_cells(&metrics);
        let expected: Vec<(&'static str, u64)> = before
            .iter()
            .map(|(reason, value)| {
                (
                    *reason,
                    if *reason == "unlink_unverified" {
                        value + 1
                    } else {
                        *value
                    },
                )
            })
            .collect();
        assert_eq!(
            after, expected,
            "EXACTLY the reason the caller passed ticks, exactly once. A different cell means the \
             reason is being re-derived inside the refusal rather than reported; two cells means \
             the arm ticks a summary alongside its reason, which would double-count every refusal \
             on an operator's dashboard"
        );
    }

    /// The `Err(_) => {}` fall-through: a *transient* header-read failure is
    /// NOT propagated — it folds into this function's own refusal, which ticks
    /// and WARNs before returning `Transient`.
    ///
    /// The third of the three arms, and the one a plausible "simplification"
    /// deletes. Collapsing the match to a blanket `Err(e) => return Err(e)`
    /// looks like a no-op here because the propagated error is *also*
    /// `Transient` — same variant, same retriability, same caller behaviour.
    /// What it silently removes is the observability: no
    /// `checkpoint_recovery_stream_unresolved` tick and no WARN for an event
    /// that is exactly what that cell means, because
    /// `classify_object_key_read_error` logs nothing on its non-`WRONGTYPE`
    /// branch. An operator watching the cell would see a Valkey blip during a
    /// stream adjudication as nothing at all.
    ///
    /// So the counter assertion below is the discriminator, not the type
    /// assertion. The `Transient` check is kept anyway as the floor the
    /// refusal shares with the sibling `Ok(false)` case; the error-text check
    /// names which of the two `Transient`s came back.
    ///
    /// # How the transient failure is induced
    ///
    /// By closing the data pool the re-check reads through — `is_layer_present`
    /// → `apply::header_is_structurally_complete`, one `HGETALL obj:hdr` on
    /// `ValkeyClient::pool`. No fault-injection seam exists on that path
    /// (`ObjectSnapshotReadFaultInjector` covers `read_object_snapshot`, not
    /// this read), and none is needed. The failure arrives as fred's
    /// `default_command_timeout` expiring on a command that can no longer be
    /// dispatched, which is the same class of error a real blip produces and
    /// the same one `classify_object_key_read_error` sends down its
    /// non-`WRONGTYPE` branch.
    ///
    /// Nothing shared is disturbed: the closed pool belongs to a second client
    /// this test connects for the purpose (see the comment at the connect), not
    /// to the `TestValkey` fixture and not to the Valkey node, so no sibling
    /// test can see it. And everything the refusal does after the failed read —
    /// counter, WARN, error construction — is in-process and needs no
    /// connection, which is why closing the pool does not also swallow the
    /// evidence this case exists to read.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn refuse_restore_folds_a_transient_header_read_failure_into_its_own_refusal() {
        let tv = TestValkey::setup().await;

        // The hook's `LayerStore` gets its OWN connection to the same Valkey,
        // under the same test-prefixed key schema — because it is this
        // connection that gets closed, and `TestValkey`'s must not be.
        // `TestValkey::drop` sweeps its prefix with SCAN + UNLINK through
        // `TestValkey::client().pool`, and that sweep does not merely fail
        // against a closed pool, it never returns: `Drop` runs it under
        // `block_on`, so closing the fixture's own pool wedges teardown
        // forever instead of breaking the one read this case is about.
        // Separate pool, same node, same prefix: the blast radius is exactly
        // the `HGETALL` under test.
        let mut doomed_config = test_config(true);
        doomed_config.valkey_url = std::env::var("TEST_VALKEY_URL")
            .unwrap_or_else(|_| "redis://127.0.0.1:6379".to_string());
        // The `HGETALL` against the closed pool fails by TIMING OUT, not by
        // being rejected outright, so this value is this test's wall-clock
        // cost — the stock 5 s makes a sub-second case take five seconds.
        // 500 ms is far above what a connected round-trip needs (well under a
        // millisecond for a one-key `HGETALL`), so it cannot make a HEALTHY
        // read fail and turn this into a flake generator; it only shortens the
        // wait for a read that has no connection left to make.
        doomed_config.valkey_command_timeout_ms = 500;
        let doomed = crate::storage::test_support::connect_test_client(&doomed_config)
            .await
            .expect("a second client to the same test Valkey must connect");

        let layer_store = Arc::new(LayerStore::new(
            doomed.clone(),
            tv.key_schema().clone(),
            4096,
        ));
        let store: Arc<dyn CheckpointStore> = Arc::new(InMemoryCheckpointStore::new());
        let metrics = make_metrics();
        let hook = RecoveryHook::new(
            store,
            layer_store,
            Arc::new(test_config(true)),
            metrics.clone(),
        );

        let oid: u64 = 9_603;

        // Nothing is seeded under `oid` — deliberately, so that a pool close
        // that somehow failed to break the read would land on the `Ok(false)`
        // route and this case would degenerate into a copy of the sibling
        // rather than silently pass for the wrong reason. The error-text
        // assertion below is what separates the two.
        //
        // Closed AFTER the hook is built: construction itself touches no
        // connection, but keeping the order explicit means a future
        // `RecoveryHook::new` that DID probe on construction would fail loudly
        // here rather than sit inside the window this test needs closed.
        {
            // Scoped rather than module-level: `ClientLike` is needed by this
            // one line and nothing else here, and its blanket methods overlap
            // names the rest of the module reaches through other traits.
            use fred::interfaces::ClientLike;
            doomed
                .pool
                .quit()
                .await
                .expect("closing the hook's own data pool must succeed");
        }

        let before = stream_unresolved_cells(&metrics);
        let err = hook
            .refuse_restore(oid, "probe_unreadable", "synthetic detail".to_owned())
            .await
            .expect_err(
                "a refusal never returns Ok(Some(..)), and with the pool closed the peer-landed \
                 re-check cannot return Ok(true) either",
            );

        assert!(
            matches!(err, CheckpointError::Transient { .. }),
            "a transient header-read failure must stay retriable — the caller sees UNAVAILABLE and \
             may re-drive once Valkey is back. Got {err:?}"
        );
        // Asserted BEFORE the error text, deliberately: this is the assertion
        // the case exists for, and an assertion that fires second only ever
        // reports after something else has already failed.
        let after = stream_unresolved_cells(&metrics);
        let expected: Vec<(&'static str, u64)> = before
            .iter()
            .map(|(reason, value)| {
                (
                    *reason,
                    if *reason == "probe_unreadable" {
                        value + 1
                    } else {
                        *value
                    },
                )
            })
            .collect();
        assert_eq!(
            after, expected,
            "THE assertion of this case: the reason the caller passed ticks exactly once, and no \
             other cell moves. A flat grid means the transient `Err` was propagated instead of \
             folded — the regression is invisible in the return type (still `Transient`) and \
             invisible in the logs (the classifier is silent on its non-WRONGTYPE branch), so \
             this counter is the only place it shows"
        );

        let rendered = err.to_string();
        assert!(
            rendered.contains("recovery refused for object"),
            "corroborates the tick above by naming WHICH `Transient` came back: this function's \
             own unresolved-stream refusal, not the header read's error propagated verbatim (that \
             one renders as `restore path read failed for object ...`). Same variant either way, \
             so the wording is the only thing that distinguishes them at the type level. \
             Got: {rendered}"
        );
    }
}

/// Unit tests for [`parse_stream_id_seq`], the one piece of reply decoding
/// the high-water probe still does on the client side.
///
/// # Why only this function
///
/// The probe used to walk a client-side `XINFO STREAM` reply with three
/// helpers (`find_member` / `last_generated_id` / `last_entry_payload`),
/// each with RESP2, RESP3 and malformed-reply arms that no integration test
/// could reach — hence a large unit-test block here. That walking now
/// happens **server-side** inside [`STREAM_HIGH_WATER_PROBE_SCRIPT`], which
/// projects the reply down to `nil` or the bare `last-generated-id` string
/// before it crosses the wire. Wire-shape decoding is no longer this
/// module's concern at all: scripts run in RESP2 conversion mode regardless
/// of the connection's negotiated protocol, so the RESP3 divergence the old
/// tests existed to pin cannot occur, and fred decodes the scalar reply into
/// `Option<String>` directly.
///
/// The behaviour those tests covered is now pinned end-to-end by the gate
/// tests in `tests/integration/checkpoint/e2e_object_id_preservation_test.rs`
/// (`seq_floor_folds_in_surviving_stream_high_water`,
/// `seq_floor_ignores_tombstoned_stream_with_auto_id`,
/// `seq_floor_ignores_implausible_stream_high_water`), which drive a real
/// Valkey and therefore exercise the script's own extraction rather than a
/// hand-built reply value.
///
/// # One decode nuance deliberately left untested
///
/// fred maps **both** `Value::Null` and a zero-length bulk string to `None`
/// when decoding into `Option<String>`, so the probe cannot distinguish "the
/// script returned `nil`" from "the script returned an empty string". That is
/// inert: the script's only non-`nil` return is a server-minted
/// `last-generated-id`, which is never empty, and both fold to the same caller
/// behaviour anyway — since b.4et that is "hand the case to
/// `classify_unreadable_stream`", whose `TYPE` round-trip then decides between
/// falling the restore floor back to `snap_seq` (`none` / a foreign type) and
/// refusing the restore outright (`stream`); the ambiguity is invisible either
/// way, because the verdict turns on the key's state and not on which flavour
/// of unreadable produced it. Pinning
/// it would test fred rather than this module. Noted here so a future script
/// edit that can return an empty value knows the ambiguity exists.
#[cfg(test)]
mod stream_id_parsing_tests {
    use super::*;

    #[test]
    fn parse_stream_id_seq_well_formed_id_returns_ms_component() {
        assert_eq!(parse_stream_id_seq("42-0"), Some(42));
        assert_eq!(parse_stream_id_seq("42-7"), Some(42));
        // Both commit paths XADD at `{seq}-0`, so the ms component IS the
        // sequence number — including at the f64 ceiling apply guards on.
        assert_eq!(
            parse_stream_id_seq("9007199254740991-0"),
            Some(9_007_199_254_740_991)
        );
        // The wall-clock auto-ID shape the plausibility gate exists to
        // reject still parses — rejection is the gate's job, not the
        // parser's.
        assert_eq!(
            parse_stream_id_seq("1790000000000-0"),
            Some(1_790_000_000_000)
        );
    }

    #[test]
    fn parse_stream_id_seq_bare_id_without_separator_returns_none() {
        // Server replies always carry both components, so a missing
        // separator means the reply shape is not what the probe assumed —
        // treat it as unparseable rather than guessing.
        assert_eq!(parse_stream_id_seq("42"), None);
    }

    #[test]
    fn parse_stream_id_seq_non_numeric_ms_returns_none() {
        assert_eq!(parse_stream_id_seq("abc-0"), None);
        assert_eq!(parse_stream_id_seq("4 2-0"), None);
        // A leading `-` splits into an empty ms component.
        assert_eq!(parse_stream_id_seq("-1-0"), None);
    }

    #[test]
    fn parse_stream_id_seq_accepts_rusts_leading_plus_sign() {
        // Not a shape any server emits, pinned because it is the one
        // "obviously malformed" input the parser does NOT reject:
        // `u64::from_str` accepts a leading `+`. The parser is therefore not
        // the defense against a weird-looking ID — the plausibility gate in
        // `probe_stream_high_water` is, and it runs on the parsed value
        // regardless of how it was spelled.
        assert_eq!(parse_stream_id_seq("+42-0"), Some(42));
    }

    #[test]
    fn parse_stream_id_seq_empty_string_returns_none() {
        assert_eq!(parse_stream_id_seq(""), None);
        assert_eq!(parse_stream_id_seq("-"), None);
    }
}
