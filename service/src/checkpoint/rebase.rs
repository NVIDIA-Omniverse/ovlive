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

//! The chain-collapse **re-base unit** (SDD SR-4.1 / SR-4.3 / SR-4.5, SR-9.1 /
//! SR-9.2).
//!
//! One invocation takes a layer whose tail chain has reached K, composes the
//! whole lineage in memory through the *same* composition implementation the
//! restore path uses, backfills every inline external value to the checkpoint
//! store, and PUTs one new [`CheckpointKind::Base`] at the collapsed head `S`.
//! The old lineage is left completely intact — this unit deletes **nothing**;
//! lineage-aware retention owns every deletion.
//!
//! Without it, a layer whose ticks keep falling back to tail capture grows its
//! chain to K and then refuses `refused_chain_full` forever: restore cost stays
//! bounded but coverage stops advancing. Since every periodic read proves its
//! cut (the atomic read at or below the atomic-read limits, the fenced read
//! above them), only a header-sequence regression sends a tick to tail capture,
//! so a chain grows only one tick per regression; collapsing it is still what
//! keeps the bounded-chain design sustainable wherever one grows.
//!
//! # Why this is its own module and not part of the capture unit
//!
//! [`capture_layer_tail`](crate::checkpoint::tail_capture::capture_layer_tail)'s
//! entire cost contract is *"one listing, zero blob fetches"* — pinned by a
//! landed test (`capture_addressing_makes_zero_blob_reads_and_one_listing`) and
//! argued four ways in `lineage_shape`'s rustdoc. A re-base fetches `1 base + K
//! tails` of multi-MB artifacts and replays every commit in them. Those are
//! different cost classes and they do not belong behind one entry point.
//!
//! # `pub`, for the same reason the capture unit is
//!
//! SR-9.2's duplicate-leader test drives two **concurrent** re-bases from the
//! integration-test crate. Two checkpoint loops would serialize on
//! `le:checkpoint:leader` and prove nothing, so the unit has to be directly
//! invokable from outside the crate.
//!
//! # Division of duty — metrics-free, flag-agnostic, `Config`-free
//!
//! Mirrors the capture unit verbatim. This unit *decides* and *persists*; it
//! returns a [`RebaseDecision`] whose [`RebaseDecision::outcome_label`] the tick
//! wiring maps onto the counter family. Nothing here touches an instrument,
//! writes `obj:hdr`, or stamps a checkpoint cursor. It likewise never consults
//! the tail feature toggle: the gate lives in the tick wiring, which is also the
//! only thing that knows a re-base is what the `ChainFull` refusal turns into.
//!
//! The signature honours
//! [`gather_restore_lineage`]'s stated cross-task contract — `&dyn
//! CheckpointStore` and nothing else: no `RecoveryHook`, no `LayerStore`, no
//! [`Config`](crate::config::Config), no metrics handle, no Valkey client, no
//! `TailSliceLoader`. It talks to the checkpoint store and to pure code, so it
//! opens **no** transactional Valkey checkout and has no `acquire_txn_for`
//! obligation.
//!
//! # One composition implementation (SR-4.1)
//!
//! [`compose_from_artifacts`] is the single engine restore and re-base share.
//! There is no re-base-flavoured variant and there must never be one; the
//! committed golden fixtures under `tests/fixtures/tail_lineage/` and the AC-4
//! live-backend property therefore already fence this unit's composition. This
//! module consequently never calls `decode_tail_envelope` either — decoding is
//! the engine's, bytes-in.
//!
//! # Determinism boundary (SR-9.2)
//!
//! Two leaders re-basing the same lineage write **byte-identical** base OMPF and
//! byte-identical ext-refs sidecar JSON: every input is the immutable lineage,
//! and `plan_full_content` emits fields, time samples and orphan sections on a
//! total order. The two values that are *not* derived that way —
//! `now_unix_ms()` and the replica label — live in [`CheckpointMeta`], which is
//! deliberately outside the bytes. **Byte-equality assertions therefore scope to
//! the OMPF blob and the sidecar bytes only**: `CheckpointMeta`'s
//! `creator_replica_id` and `written_at_unix_ms` differ between two leaders *by
//! design*, so comparing whole metas would be wrong, not flaky.

use bytes::Bytes;
use tokio::time::Instant;

use crate::checkpoint::assembler::{build_reset_ompf_from_state, ext_refs_sidecar_json};
use crate::checkpoint::compose::{compose_from_artifacts, ComposeError, ComposedLineage, TailSite};
use crate::checkpoint::lineage::{gather_restore_lineage, no_deadline, GatheredLineage};
use crate::checkpoint::recovery::adjudicate_composed_meta;
use crate::checkpoint::store::{
    CheckpointBlob, CheckpointError, CheckpointHandle, CheckpointKind, CheckpointMeta,
    CheckpointStore,
};
use crate::checkpoint::{now_unix_ms, TailCaptureConfig};
use crate::storage::keys::sha256_to_hex;

// ── Result types ────────────────────────────────────────────────────────────

/// What one re-base invocation decided to do.
///
/// Three outcomes, not two — collapsed, deliberately did nothing because the
/// bound was not really reached, or refused. Modelled on
/// [`TailCaptureDecision`](crate::checkpoint::tail_capture::TailCaptureDecision),
/// whose split of "persisted / persisted nothing / failed" this mirrors.
#[derive(Debug, Clone)]
pub enum RebaseDecision {
    /// A new base was composed and persisted at the collapsed head.
    Rebased {
        /// The handle the adapter assigned. Its `snap_seq` is `to_seq`.
        handle: CheckpointHandle,
        /// The lineage root the collapse started from — the OLD base's
        /// sequence. It is not deleted here; retention prunes the superseded
        /// lineage as a unit.
        base_snap_seq: u64,
        /// The true link-walk chain length that triggered the collapse, i.e.
        /// how many tails were replayed. Always `>= cfg.max_chain`.
        from_chain_len: usize,
        /// **S** — the sequence the new base restores to, and the sequence
        /// component of its adapter address.
        to_seq: u64,
        /// How many distinct external-value blobs were `put_ext_blob`-acked as
        /// the SR-4.3 precondition. `0` for a lineage whose replayed commits
        /// referenced no inline ext parts.
        ext_backfilled: usize,
    },
    /// The **true** link-walk chain is shorter than K, so the bound the caller
    /// acted on was the listing's over-approximation rather than a real
    /// chain-at-K.
    ///
    /// This is SR-9.2's divergent-`S` duplicate-leader fork seen from the
    /// listing side: overlapping non-member tails written by a second leader sit
    /// above the base and inflate the positional count, while the link walk
    /// addresses members by exact sequence and never sees them.
    ///
    /// Re-basing anyway would burn a full lineage GET plus a full-snapshot
    /// build-and-PUT that the bound never demanded — repeatedly, for as long as
    /// the orphans persist. Nothing is persisted on this arm and **no metric
    /// cell moves**; the caller's move is to re-run the capture with a verified
    /// bound.
    UnderBound {
        /// The link-walk chain length, from [`GatheredLineage::chain_len`].
        true_chain_len: usize,
    },
    /// Nothing was persisted and the lineage is untouched; the next tick
    /// re-attempts (SR-4.5).
    Refused(RebaseRefusal),
}

/// Why a re-base persisted nothing — SR-4.5's refusal arm, one variant per
/// cause.
///
/// Field-free on purpose. Every variant's diagnostic detail (which generation,
/// which store error, which hash) rides a distinct `tracing` event at the arm
/// that produced it, which is where it is actionable and where it can carry a
/// `Display` that no metric label may. What the enum has to do is let a caller —
/// and the test lane — name the cause exactly, which a `Copy + Eq` unit variant
/// does best.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RebaseRefusal {
    /// The store failed the listing or a member fetch (or the deadline ran out
    /// during the walk). Nothing about the lineage is implied.
    Gather,
    /// The lineage did not verify at the listing head: the walk stepped down,
    /// or it had nothing to offer at all. **Never** a walk to an older lineage —
    /// see the no-step-down rule on [`rebase_layer_lineage`].
    LineageBroken,
    /// The composition engine rejected the lineage, **or** the two derivations
    /// of S disagreed once it had not — the composed cursor did not land on the
    /// gathered head sequence, so a base PUT would have claimed more coverage
    /// than it carried. Both causes raise the SR-4.5 suspect-lineage tripwire,
    /// under the same `SUSPECT LINEAGE` marker; the events distinguish them.
    Compose,
    /// An external-value blob could not be durably acknowledged, so the SR-4.3
    /// hard precondition for the base PUT is unmet. Includes the per-blob
    /// timeout elapsing.
    ExtBackfill,
    /// The new base could not be produced or persisted: the reset-OMPF build
    /// failed, the ext-refs sidecar would not serialize, or the single
    /// `put_layer_checkpoint` failed. Nothing landed at the address in any of
    /// the three; the two pre-PUT sub-arms are distinguished by their own WARNs.
    BasePut,
}

impl RebaseDecision {
    /// This decision's `outcome` label on `live_edit_checkpoint_tail_total`, or
    /// `None` when the decision deliberately moves no cell.
    ///
    /// Every returned value is a member of
    /// [`CHECKPOINT_TAIL_OUTCOMES`](crate::observability::metrics::CHECKPOINT_TAIL_OUTCOMES),
    /// which is the canonical vocabulary; this method must not re-spell it, and
    /// the re-base adds **no new family and no new cell**.
    ///
    /// # Why every refusal folds onto `refused_chain_full`
    ///
    /// That cell already means "the chain bound was reached and no tail was
    /// captured". Once the re-base is wired, its honest re-reading is *"the
    /// chain bound was reached and the re-base that should have collapsed it did
    /// not land"* — the same operator-facing condition with the same
    /// remediation surface, caught at the same point in the tick. Minting one
    /// cell per SR-4.5 cause would fork the dashboard vocabulary for
    /// distinctions the enum and the per-arm tracing events already carry, which
    /// is the argument
    /// [`TailCaptureDecision::outcome_label`](crate::checkpoint::tail_capture::TailCaptureDecision::outcome_label)
    /// makes for its own post-restore fold.
    ///
    /// # Why [`RebaseDecision::UnderBound`] returns `None`
    ///
    /// It is not an outcome of the *capture* the family counts: nothing was
    /// captured, nothing was refused, and the caller's next move is to re-run
    /// the capture with a verified bound — whose own decision ticks the cell.
    /// Returning a label here would double-count that tick. An `Option` rather
    /// than a sentinel string is what makes "tick nothing" impossible to get
    /// wrong at the call site.
    pub fn outcome_label(&self) -> Option<&'static str> {
        match self {
            RebaseDecision::Rebased { .. } => Some("rebased"),
            RebaseDecision::UnderBound { .. } => None,
            // The fold — see this method's doc.
            RebaseDecision::Refused(_) => Some("refused_chain_full"),
        }
    }
}

// ── Entry point ─────────────────────────────────────────────────────────────

/// Collapse `object_id`'s tail chain into one new base, or report why not.
///
/// # The sequence
///
/// 1. **Gather** the lineage at the listing head (one listing + `1 base + K`
///    member fetches), on the budget derived below.
/// 2. **Bound check** against the TRUE link-walk chain length.
/// 3. **Compose** through [`compose_from_artifacts`] on `spawn_blocking`, then
///    cross-check the two derivations of `S` unconditionally — a composed cursor
///    that did not land on the gathered head refuses on the same
///    [`RebaseRefusal::Compose`] arm rather than PUTting a base that claims more
///    coverage than it carries.
/// 4. **Backfill** every inline external value with durable acknowledgement —
///    SR-4.3's hard precondition.
/// 5. **Emit and PUT** one base at `(object_id, S)`.
///
/// Every refusal arm leaves the lineage completely intact, so the next tick
/// re-attempts from the same place. Nothing here deletes anything.
///
/// # No step-down (SR-4.5) — do not reuse `select_restore_source`'s loop
///
/// [`LineageSelection::next_restorable`](crate::checkpoint::lineage::LineageSelection::next_restorable)
/// is called **once**. If the walk stepped down, or answered `Ok(None)`, the
/// re-base refuses; it never walks to an older lineage, and it never calls
/// `record_compose_failure`.
///
/// A later reader *will* be tempted to reuse the restore-side loop. The reason
/// it is wrong here is that the two callers want opposite things from a broken
/// lineage. Restore steps down because older state still beats no state — it is
/// reading. A re-base is *writing*: minting a base from an older lineage would
/// (a) silence the very SR-4.5 tripwire the broken head is supposed to raise,
/// and (b) collapse the wrong thing — publishing a new head that supersedes a
/// lineage nobody adjudicated as unrestorable, and inviting retention to prune
/// the newer generations behind it. Refusing costs one tick; stepping down costs
/// the evidence.
///
/// Because a candidate is abandoned only by recording a
/// [`StepDownRecord`](crate::checkpoint::lineage::StepDownRecord), "the returned
/// lineage heads the listing" and "the selection recorded no step-down" are the
/// same predicate — the second is the one this unit can observe without a second
/// listing, and it is what the check below reads.
///
/// # The ext-mirror queue is NOT this path
///
/// `checkpoint::ext_mirror`'s bounded queue drops a copy when its item count
/// or its byte budget is full and is WARN-and-continue. That posture cannot
/// express SR-4.3's *hard precondition*,
/// so this unit does not enqueue and does not touch `ExtMirrorSender`. Do not
/// "optimise" the backfill onto the queue later: a partially-backfilled re-base
/// lands a base whose by-hash refs are backed only by live Valkey, and retention
/// then prunes the tails that were the last durable copy.
///
/// # SR-4.3 scope, and its recorded residual
///
/// The hard precondition covers **tail-inline** ext parts —
/// [`ComposedLineage::ext_values`], which is SR-4.3's literal wording ("every
/// inline ext part encountered in the tail entries"). Base-era hashes that the
/// new base's sidecar also lists are not backfilled here: the write path
/// already mirrored them, exactly as `base_bootstrap` states for its own base
/// ("the blob bytes behind these hashes are NOT re-mirrored here"), and a copy
/// the mirror lost is re-sent by the external-value backup repair
/// (`checkpoint::ext_backup_repair`) while Valkey holds the value. What
/// remains is a value Valkey loses before the repair reaches it, or one larger
/// than the mirror's whole byte budget; that residual is deliberate.
///
/// The backfill is **sequential**, not bounded-parallel: predictable lease burn,
/// no new machinery, and the loop is a hard precondition rather than a
/// throughput path. Parallelising it is a possible later optimisation, not a
/// correctness input.
///
/// # Errors
///
/// [`CheckpointError::NotFound`] **only** — the layer vanished mid-re-base and
/// the caller drops it from its working set, exactly as on the capture unit.
/// Every SR-4.5 cause is a [`RebaseDecision`], not an `Err`: a store failure
/// that would be `Transient` on a read path is, here, simply a refusal that the
/// next tick retries.
///
/// # No liveness probe
///
/// This form passes `None` for the pre-PUT `obj:hdr` liveness probe (b.hka)
/// because it holds no Valkey client. That leaves its base PUT unguarded —
/// byte-for-byte the behaviour it has always had — and is why the checkpoint
/// loop calls [`rebase_layer_lineage_with_probe`] instead. The split mirrors
/// [`capture_layer_tail`](crate::checkpoint::tail_capture::capture_layer_tail)
/// / `capture_layer_tail_with`: one entry point for the production path that
/// holds a client, one for the fake-store lane that does not.
pub async fn rebase_layer_lineage(
    store: &dyn CheckpointStore,
    object_id: u64,
    replica_id: &str,
    cfg: &TailCaptureConfig,
) -> Result<RebaseDecision, CheckpointError> {
    rebase_layer_lineage_with_probe(None, store, object_id, replica_id, cfg).await
}

/// [`rebase_layer_lineage`] carrying the pre-PUT `obj:hdr` liveness probe
/// (b.hka).
///
/// The checkpoint loop's entry point. A `Dead` verdict skips the base PUT
/// entirely and returns [`CheckpointError::NotFound`] — the loop's existing
/// benign-and-terminal arm for "the layer vanished mid-re-base", which it
/// already handles at `debug!` with no metric tick, so no new refusal cell is
/// minted for an object that no longer exists. See
/// [`crate::checkpoint::put_guard`] for the probe's residual.
pub(crate) async fn rebase_layer_lineage_with_probe(
    probe: Option<crate::checkpoint::put_guard::ObjectLivenessProbe<'_>>,
    store: &dyn CheckpointStore,
    object_id: u64,
    replica_id: &str,
    cfg: &TailCaptureConfig,
) -> Result<RebaseDecision, CheckpointError> {
    // ── 1. Gather ───────────────────────────────────────────────────────────
    //
    // THE GATHER BUDGET — derived here, deliberately NOT inherited from the
    // restore path.
    //
    // `reserve` is one adapter round-trip's expected cost
    // (`LIVE_EDIT_CHECKPOINT_REQUEST_TIMEOUT_MS`, 30 s stock; see
    // `TailCaptureConfig::rebase_fetch_reserve`). The whole-gather deadline is
    // `(max_chain + 2) x reserve`:
    //
    //   +1 for `gather_restore_lineage`'s own `list_layer_checkpoints`, which
    //      is issued before the first member fetch and spends the same
    //      `Instant`-based budget;
    //   +K for the tails and +1 for the base — the `1 base + K tails` a
    //      chain-at-K lineage is by definition.
    //
    // The `+2` is load-bearing, not slack. At `(max_chain + 1) x reserve`, an
    // adapter that is slow-but-HEALTHY and whose listing costs a full reserve
    // funds only K fetches for K+1 members. The gather's allowance rule then
    // abandons a member with `LineageBreak::Budget` — post-b.bu8 the last TAIL
    // rather than the base, since the base's fetch is the one the accounting
    // reserves, but a member either way — which under the
    // no-step-down rule above is a refusal, and it refuses on EVERY attempt,
    // recreating exactly the permanent chain-full wedge SR-4.1 exists to
    // remove, this time through the budget instead of the bound. (Anchoring the
    // deadline after the listing returns is the equivalent fix; this form keeps
    // the whole budget derivable from the knobs at one place.)
    //
    // Not inherited from the restore path because the cost model differs: this
    // caller fetches a whole lineage, always, where restore may answer from the
    // newest base alone. b.bu8 settled the restore path's arithmetic without
    // changing its budget formula — the base's fetch is now RESERVED out of the
    // budget by member kind inside `gather_restore_lineage` rather than gated
    // against a threshold — and this caller REQUIRES that `Base` arm. At
    // `(max_chain + 2) x reserve` with a full-reserve listing and K
    // full-reserve tails, the chain's own base is reached at exactly
    // `remaining == reserve`; the `Base` allowance `min(reserve, remaining)`
    // funds it, whereas a uniform speculative rule (`remaining - reserve`)
    // would decline it and recreate the chain-full wedge the `+2` above exists
    // to remove.
    //
    // THE HEAD-OF-LINE TRADE, accepted. At stock knobs the budget is
    // (8 + 2) x 30 s = 300 s against a 30 s `LEADER_LOCK_TTL_MS`, so a re-base against
    // a degraded adapter can outlive the lease. SR-9.1 already accepts exactly
    // that — a lease-burning re-base is "demoted next tick exactly as a
    // lease-burning snapshot PUT is today" — and the per-fetch
    // `timeout_at(bound)` inside the gather, whose bound is at most one reserve
    // wide, is what stops ONE wedged fetch running to the StorageApi adapter's
    // enforced 180 s per-call ceiling.
    //
    // REJECTED ALTERNATIVE, recorded so it is not re-litigated: capping the
    // gather at the lease TTL. That makes a K=8 re-base structurally impossible
    // to complete on a slow-but-healthy adapter (8 members cannot be fetched in
    // one 30 s lease at a 30 s per-fetch cost), i.e. a permanent chain-full
    // wedge — the same failure the `+2` correction above removes, reintroduced
    // deliberately.
    let reserve = cfg.rebase_fetch_reserve;
    let budget_units = u32::try_from(cfg.max_chain.saturating_add(2)).unwrap_or(u32::MAX);
    let budget = reserve.saturating_mul(budget_units);
    // A pathological knob pair can saturate the multiplication; `no_deadline()`
    // is the module's own 24 h horizon and is strictly SHORTER than a saturated
    // `Duration`, so the fallback is the safe direction and cannot panic the
    // `Instant` arithmetic.
    let deadline = Instant::now()
        .checked_add(budget)
        .unwrap_or_else(no_deadline);

    let mut selection = match gather_restore_lineage(store, object_id, deadline, reserve).await {
        Ok(selection) => selection,
        Err(CheckpointError::NotFound) => return Err(CheckpointError::NotFound),
        Err(err) => {
            tracing::warn!(
                object_id,
                reserve_ms = reserve.as_millis() as u64,
                is_transient = err.is_transient(),
                is_permanent = err.is_permanent(),
                error = %err,
                outcome = "refused_chain_full",
                "re-base could not list this layer's generations; the lineage is untouched and \
                 the next tick re-attempts"
            );
            return Ok(RebaseDecision::Refused(RebaseRefusal::Gather));
        }
    };

    // ONE call. See the no-step-down rule on this function.
    let lineage = match selection.next_restorable().await {
        Ok(Some(lineage)) => lineage,
        Ok(None) => {
            tracing::warn!(
                object_id,
                outcome = "refused_chain_full",
                "re-base found no verifiable lineage at the listing head; refusing rather than \
                 walking to an older lineage. The lineage is untouched"
            );
            return Ok(RebaseDecision::Refused(RebaseRefusal::LineageBroken));
        }
        Err(CheckpointError::NotFound) => return Err(CheckpointError::NotFound),
        Err(err) => {
            tracing::warn!(
                object_id,
                reserve_ms = reserve.as_millis() as u64,
                is_transient = err.is_transient(),
                is_permanent = err.is_permanent(),
                error = %err,
                outcome = "refused_chain_full",
                "re-base could not fetch this layer's lineage members; the lineage is untouched \
                 and the next tick re-attempts"
            );
            return Ok(RebaseDecision::Refused(RebaseRefusal::Gather));
        }
    };

    // SR-4.5, the no-step-down fence. A recorded step-down means the verified
    // lineage is NOT the listing head's — the walk already moved down, which is
    // the thing this unit must not act on.
    if !selection.step_downs().is_empty() {
        tracing::warn!(
            object_id,
            head_seq = lineage.head_seq,
            step_downs = selection.step_downs().len(),
            first_step_down_seq = selection.step_downs()[0].head_seq,
            first_step_down_reason = selection.step_downs()[0].reason.reason_label(),
            outcome = "refused_chain_full",
            "re-base declines a lineage the walk had to step down to: the listing head did not \
             verify, and minting a base from an older lineage would silence that signal and \
             collapse the wrong generations. The lineage is untouched"
        );
        return Ok(RebaseDecision::Refused(RebaseRefusal::LineageBroken));
    }

    // ── 2. The bound, read from the TRUE link-walk chain length ─────────────
    //
    // `GatheredLineage::chain_len()` is the binding K bound per SR-9.2 — the
    // number of tails actually linked onto the base, addressed by exact
    // sequence. `lineage_shape`'s listing-position count is deliberately NOT
    // consulted anywhere in this unit: it over-approximates (correct for the
    // capture unit's per-tick pre-filter, wrong for a re-base trigger).
    let true_chain_len = lineage.chain_len();
    if true_chain_len < cfg.max_chain {
        tracing::debug!(
            object_id,
            true_chain_len,
            max_chain = cfg.max_chain,
            head_seq = lineage.head_seq,
            "re-base declines: the true link-walk chain is under K, so the caller's bound came \
             from the listing's over-approximation (overlapping tails a second checkpoint \
             leader left behind). Nothing \
             persisted, no metric cell moved"
        );
        return Ok(RebaseDecision::UnderBound { true_chain_len });
    }

    // Shape facts read before the lineage is destructured.
    let head_seq = lineage.head_seq;
    let GatheredLineage { base, tails, .. } = lineage;

    let tail_bytes: Vec<Bytes> = tails.iter().map(|tail| tail.bytes.clone()).collect();
    // The `GatheredTail` records (handles, headers, advisory metas) are done
    // with; only the artifact `Bytes` handles go forward, and only into the
    // blocking closure below.
    drop(tails);

    let CheckpointBlob {
        bytes: base_bytes,
        meta: base_meta,
        handle: base_handle,
    } = base;
    let base_snap_seq = base_handle.seq();

    // ── 3. Compose, off the runtime ─────────────────────────────────────────
    //
    // Same reason and same memory argument as the restore driver's own offload.
    // `compose_from_artifacts` is fully synchronous, CPU-bound, and has NO
    // yield point: at stock knobs it replays up to `TAIL_MAX_CHAIN x
    // TAIL_MAX_ENTRIES` = 8 x 4096 = 32,768 records — an OMPF parse, a C++
    // FlatBuffers verification and an `apply_delta` each — over up to
    // `TAIL_MAX_BYTES x TAIL_MAX_CHAIN` = 512 MiB of artifact bytes. Inline,
    // that blocks a tokio worker for plausibly seconds, and this one runs on the
    // checkpoint tick where a blocked worker also stalls the leader walk.
    //
    // The artifact `Bytes` are moved INTO the closure and dropped with its
    // frame, so the COMPILER — not a comment — is what stops them living past
    // composition. What survives is `ComposedLineage::ext_values`, whose
    // zero-copy views pin the tail buffers they were sliced from; the backfill
    // below runs while they are alive and they are dropped before the base OMPF
    // is built, so the peak is not doubled.
    let composed = tokio::task::spawn_blocking(move || {
        compose_from_artifacts(object_id, &base_bytes, base_snap_seq, &tail_bytes)
    })
    .await;

    let composed = match composed {
        Ok(result) => result,
        Err(join_err) => {
            // A panic inside the composition engine is an INTERNAL DEFECT, not a
            // lineage verdict. Refusing would file it under SR-4.5 and hide it
            // behind a cell that reads as an adapter problem, so it surfaces as
            // `Permanent` — the one non-`NotFound` `Err` this unit produces, and
            // deliberately so.
            tracing::error!(
                object_id,
                head_seq,
                base_snap_seq,
                chain_len = true_chain_len,
                error = %join_err,
                "composition task failed to join during a chain-collapse re-base; this is an \
                 internal defect, not a broken lineage"
            );
            return Err(CheckpointError::Permanent {
                source: format!(
                    "re-base composition task for object {object_id} at head {head_seq} failed to \
                     join: {join_err}"
                )
                .into(),
            });
        }
    };

    let composed = match composed {
        Ok(composed) => composed,
        Err(compose_err) => {
            // ── The SR-4.5 suspect-lineage tripwire ─────────────────────────
            //
            // This is the ONE thing that makes "refuse and retry next tick" safe
            // rather than silent. A persisted tail that will not compose is
            // unrestorable NOW; without this WARN the condition would first
            // surface at DR-restore time as a step-down (SR-3.5) — i.e. during
            // the incident, on state an operator can no longer do anything
            // about. Every retry re-raises it, which is correct: the lineage
            // stays suspect until something replaces it.
            let (offending_to_seq, offending_artifact_index) =
                match offending_generation(&compose_err) {
                    Some(site) => (site.to_seq, Some(site.index)),
                    None => (None, None),
                };
            tracing::warn!(
                object_id,
                base_snap_seq,
                chain_len = true_chain_len,
                to_seq = head_seq,
                offending_to_seq = ?offending_to_seq,
                offending_artifact_index = ?offending_artifact_index,
                error = %compose_err,
                outcome = "refused_chain_full",
                "SUSPECT LINEAGE: a chain-collapse re-base could not compose this layer's \
                 persisted lineage. The lineage is unrestorable as it stands and will surface as \
                 a restore step-down if a DR restore selects it. Nothing was written, nothing \
                 was deleted, and the re-base will re-attempt (and re-raise this) every tick \
                 until the lineage is replaced"
            );
            return Ok(RebaseDecision::Refused(RebaseRefusal::Compose));
        }
    };

    // ── The S cross-check — UNCONDITIONAL, not a `debug_assert!` ────────────
    //
    // Two independent derivations of S: the gather walk's link-chain terminal
    // (`head_seq`, which is also the PUT address and `meta.snap_seq`) and the
    // replay cursor's landing point (`composed.composed_seq`, which is what the
    // composed state actually contains).
    //
    // WHY IT CANNOT BE A `debug_assert!`: that is compiled out of the release
    // binary, and what it guards is the single most expensive failure in this
    // subsystem. A divergence PUTs a base at address `head_seq` whose meta
    // claims `snap_seq = head_seq` while carrying state materialized only at
    // `composed_seq` — a generation whose claimed coverage exceeds its content,
    // i.e. the b.pw6 class the whole checkpoint subsystem exists to forbid. A
    // DR restore would then silently serve state older than the sequence it
    // reports, with nothing left to detect it. The check costs one integer
    // comparison per re-base.
    //
    // It is unreachable by construction today (the gather's link checks and
    // `compose_from_artifacts`'s own terminal-cursor defences both fence it),
    // so reaching it means an invariant broke somewhere upstream — which is
    // exactly the SR-4.5 "suspect lineage" condition, and it is raised with the
    // same `SUSPECT LINEAGE` marker and the same `outcome` as the
    // compose-failure arm above so one log query catches the whole tripwire.
    // Refuse whole, exactly like that arm: nothing PUT, lineage untouched, next
    // tick re-attempts. `record_compose_failure` is NOT called here (D6) and the
    // PUT address is not "corrected" to `composed_seq` — a derivation
    // disagreement is not a licence to pick one of them.
    if composed.composed_seq != head_seq {
        tracing::warn!(
            object_id,
            base_snap_seq,
            chain_len = true_chain_len,
            composed_seq = composed.composed_seq,
            head_seq,
            outcome = "refused_chain_full",
            "SUSPECT LINEAGE: a chain-collapse re-base's two derivations of S disagree — the \
             composed cursor and the gathered head sequence are two derivations of S and must \
             agree. Nothing was written: PUTting the base would have addressed it at head_seq \
             while it carries only the state composed at composed_seq, a generation claiming \
             more coverage than it holds. Suspect a broken lineage or a composition-engine \
             defect; the re-base will re-attempt (and re-raise this) every tick until the \
             lineage is replaced"
        );
        return Ok(RebaseDecision::Refused(RebaseRefusal::Compose));
    }

    let ComposedLineage {
        state, ext_values, ..
    } = composed;

    // ── 4. SR-4.3 ext backfill — a HARD precondition, before the base PUT ───
    //
    // `Ok(())` from `put_ext_blob` IS the durable acknowledgement: the trait
    // contract makes the write atomic and idempotent by hash, so a re-put of
    // bytes already present succeeds without rewriting and two leaders racing on
    // the same hash cannot conflict.
    //
    // The FIRST `Err` — or the first elapsed timeout — refuses the WHOLE
    // re-base: nothing PUT, nothing stamped, lineage intact, retry next tick.
    // Anything weaker lands a base whose by-hash refs are backed only by live
    // Valkey while retention prunes the tails that carried the bytes.
    //
    // THE PER-BLOB TIMEOUT is not optional. `ext_values` is bounded only by
    // `TAIL_MAX_BYTES x MAX_CHAIN` (512 MiB at defaults), so N can reach the
    // hundreds, and an unbounded `put_ext_blob` is capped only by the StorageApi
    // adapter's ENFORCED per-call ceiling of `request_timeout x 3 retries x 2
    // auth attempts` = 180 s. N x 180 s is materially worse than the 540 s
    // single-assembly worst case the loop already calls reachable. Capping each
    // call at one reserve makes the residual worst case `N x reserve` (N x 30 s
    // stock) — still large by construction, and still bounded by the same
    // lease-burn / head-of-line trade the gather budget records above: a re-base
    // that overruns is demoted next tick, per SR-9.1.
    let ext_backfilled = ext_values.len();
    for (hash, bytes) in &ext_values {
        let hash_hex = sha256_to_hex(hash);
        match tokio::time::timeout(reserve, store.put_ext_blob(&hash_hex, bytes.clone())).await {
            Ok(Ok(())) => {}
            Ok(Err(err)) => {
                // NotFound is deliberately NOT propagated from this call: it is
                // content-addressed and object-agnostic, so a NotFound here is
                // an adapter-contract violation rather than "the layer vanished"
                // — and refusing (retry next tick) is safer than dropping the
                // layer out of the caller's working set on it.
                tracing::warn!(
                    object_id,
                    head_seq,
                    base_snap_seq,
                    hash_hex = hash_hex.as_str(),
                    blob_bytes = bytes.len(),
                    ext_total = ext_backfilled,
                    is_transient = err.is_transient(),
                    is_permanent = err.is_permanent(),
                    error = %err,
                    outcome = "refused_chain_full",
                    "re-base ext-blob backfill failed; a base is written only after every \
                     external value carried inline in the collapsed tails is durably stored in \
                     the checkpoint store, so no base was written. The lineage is untouched"
                );
                return Ok(RebaseDecision::Refused(RebaseRefusal::ExtBackfill));
            }
            Err(_elapsed) => {
                tracing::warn!(
                    object_id,
                    head_seq,
                    base_snap_seq,
                    hash_hex = hash_hex.as_str(),
                    blob_bytes = bytes.len(),
                    ext_total = ext_backfilled,
                    reserve_ms = reserve.as_millis() as u64,
                    outcome = "refused_chain_full",
                    "re-base ext-blob backfill exceeded its per-blob budget; a base is written \
                     only after every external value carried inline in the collapsed tails is \
                     durably stored in the checkpoint store, so no base was written. The \
                     lineage is untouched"
                );
                return Ok(RebaseDecision::Refused(RebaseRefusal::ExtBackfill));
            }
        }
    }

    // Memory ordering, load-bearing: the backfill above ran while `ext_values`
    // was alive because its `Bytes` are zero-copy views that PIN the tail
    // artifact buffers. Dropping it here releases up to 512 MiB before the base
    // OMPF is built, so the two peaks do not add.
    drop(ext_values);

    // ── 5. Base emission ────────────────────────────────────────────────────
    //
    // Byte-deterministic for a given `ObjectState` (`plan_full_content` emits on
    // a total order), which is what makes the SR-9.2 duplicate-leader
    // byte-equality assertions sound for the blob and the sidecar.
    let (ompf_bytes, ordered_hashes) = match build_reset_ompf_from_state(&state) {
        Ok(built) => built,
        Err(err) => {
            tracing::warn!(
                object_id,
                head_seq,
                base_snap_seq,
                chain_len = true_chain_len,
                is_transient = err.is_transient(),
                is_permanent = err.is_permanent(),
                error = %err,
                outcome = "refused_chain_full",
                "re-base could not build the collapsed base OMPF; no base written, lineage \
                 untouched"
            );
            return Ok(RebaseDecision::Refused(RebaseRefusal::BasePut));
        }
    };
    // The composed state is dead after the build; drop it before the PUT rather
    // than carrying it across the await.
    drop(state);

    // ALWAYS `Some`, even when the hash list is empty: the b.s5y publish-order
    // obligation is what makes a sidecar visible no later than its OMPF, and it
    // applies to EVERY base. `validate_tail_put`'s "tails carry no sidecar"
    // rejection guards tail puts only and must not be read across to this one.
    let sidecar_bytes = match ext_refs_sidecar_json(&ordered_hashes) {
        Ok(bytes) => bytes,
        Err(err) => {
            tracing::warn!(
                object_id,
                head_seq,
                base_snap_seq,
                ext_hash_count = ordered_hashes.len(),
                is_transient = err.is_transient(),
                is_permanent = err.is_permanent(),
                error = %err,
                outcome = "refused_chain_full",
                "re-base could not serialize the collapsed base's ext-refs sidecar; no base \
                 written, lineage untouched"
            );
            return Ok(RebaseDecision::Refused(RebaseRefusal::BasePut));
        }
    };

    let ompf_byte_len = ompf_bytes.len();
    let sidecar_byte_len = sidecar_bytes.len();
    let ext_hash_count = ordered_hashes.len();

    // The collapsed lineage's TOTAL span: this base genuinely covers every
    // commit from the old root to S, so "commits this generation covers that the
    // previous did not" is `S - base_snap_seq`. That is the one column
    // `adjudicate_composed_meta`'s two callers derive differently, which is
    // exactly why it is an argument there — restore wants the head tail's
    // interval instead. Identity (`session_id` / `layer_name` / `created_by` /
    // `server_ts`) is BASE-sourced by the gather's MUST: a tail meta must never
    // reach `apply::live_mapping_resolves_blob_identity`.
    let observed_commit_span = head_seq.saturating_sub(base_snap_seq);
    let meta = CheckpointMeta {
        creator_replica_id: replica_id.to_string(),
        snap_seq: Some(head_seq),
        // Spelled out even though it is the `Default` and even though the
        // source meta is already a base: a base-vs-tail put must never be
        // implicit at a write site.
        kind: CheckpointKind::Base,
        // Likewise defensive against a non-conforming adapter meta — a Base
        // carries neither, by the `CheckpointMeta` three-field invariant.
        base_snap_seq: None,
        from_seq: None,
        sidecar_bytes: Some(Bytes::from(sidecar_bytes)),
        ..adjudicate_composed_meta(base_meta, observed_commit_span, now_unix_ms())
    };

    // ── 6. The single PUT ───────────────────────────────────────────────────
    //
    // FENCE: no read of the existing artifact at this address, under any
    // circumstance. SR-9.2's `(object_id, S)` idempotency contract is blind
    // last-writer-wins over byte-identical content — two leaders composing the
    // same lineage produce the same base bytes and the same sidecar bytes — and
    // a read-modify-write here would destroy that safety argument outright.
    // `head_seq` is the snap_seq argument. There is no retry loop; the tick is
    // the retry.
    //
    // Gated on the shared pre-PUT liveness probe (b.hka): the gather, the
    // compose and the ext-backfill above are SR-9.1's declared HEAVY unit, so
    // the window in which a `DeleteLayer` can land under this call is the
    // widest of the four durable-generation producers. `Ok(None)` folds onto
    // the `NotFound` arm immediately below — the same verdict as an adapter
    // `NotFound`, which this function already documents as its ONLY `Err`.
    // `crate::checkpoint::put_guard` carries the residual.
    let handle = match crate::checkpoint::put_guard::put_layer_checkpoint_if_live(
        probe,
        store,
        object_id,
        head_seq,
        Bytes::from(ompf_bytes),
        meta,
        "re-base",
    )
    .await
    {
        Ok(Some(handle)) => handle,
        Ok(None) => return Err(CheckpointError::NotFound),
        Err(CheckpointError::NotFound) => return Err(CheckpointError::NotFound),
        Err(err) => {
            // `NotImplemented` folds here too: only the `Disabled` adapter
            // produces it and the tick's enable gate makes it unreachable, so it
            // needs no arm of its own — and on this path it means the same
            // thing every other error does, that no base landed.
            tracing::warn!(
                object_id,
                head_seq,
                base_snap_seq,
                chain_len = true_chain_len,
                ompf_byte_len,
                is_transient = err.is_transient(),
                is_permanent = err.is_permanent(),
                error = %err,
                outcome = "refused_chain_full",
                "re-base base PUT failed; no base landed at the collapsed head. The lineage is \
                 untouched and the next tick re-attempts"
            );
            return Ok(RebaseDecision::Refused(RebaseRefusal::BasePut));
        }
    };

    let adapter_key = handle.adapter_key.clone();
    let decision = RebaseDecision::Rebased {
        handle,
        base_snap_seq,
        from_chain_len: true_chain_len,
        to_seq: head_seq,
        ext_backfilled,
    };
    tracing::info!(
        object_id,
        base_snap_seq,
        to_seq = head_seq,
        from_chain_len = true_chain_len,
        observed_commit_span,
        ext_backfilled,
        ext_hash_count,
        ompf_byte_len,
        sidecar_byte_len,
        adapter_key = adapter_key.as_str(),
        // Spelled as the literal rather than `?decision.outcome_label()`: the
        // `Debug` of the `Option` renders `Some("rebased")`, which no
        // `outcome="rebased"` log query matches, while every other `outcome`
        // field in the subsystem is a bare string. The value is statically known
        // on this arm — `outcome_label()` answers `Some("rebased")` for
        // `Rebased` by construction — and the `Option` return itself is right
        // and stays (it is what makes `UnderBound`'s "tick nothing" impossible
        // to get wrong at the metrics call site).
        outcome = "rebased",
        "layer lineage collapsed into a new base; the superseded lineage is left intact for \
         retention to prune"
    );
    Ok(decision)
}

/// Which generation a [`ComposeError`] attributes itself to, for the SR-4.5
/// tripwire's structured fields.
///
/// `None` means the BASE is the offender ([`ComposeError::BaseDecode`]), which
/// the tripwire already names through `base_snap_seq`. The exhaustive match is
/// deliberate: a new `ComposeError` variant must be classified here rather than
/// silently losing its attribution.
///
/// The error's own `Display` carries the same site plus the record-level detail
/// and rides the event as `error`; these two fields exist so the offending
/// generation is queryable without parsing a message.
fn offending_generation(err: &ComposeError) -> Option<TailSite> {
    match err {
        ComposeError::BaseDecode { .. } => None,
        ComposeError::TailDecode { site, .. }
        | ComposeError::ObjectIdMismatch { site, .. }
        | ComposeError::BaseMismatch { site, .. }
        | ComposeError::LineageGap { site, .. }
        | ComposeError::RecordSequenceMismatch { site, .. }
        | ComposeError::TerminalCursor { site, .. } => Some(*site),
        ComposeError::OmpfParse { site, .. }
        | ComposeError::OodfParse { site, .. }
        | ComposeError::ExtValueIndex { site, .. }
        | ComposeError::ApplyDelta { site, .. }
        | ComposeError::PostApplySequence { site, .. } => Some(site.tail),
    }
}
