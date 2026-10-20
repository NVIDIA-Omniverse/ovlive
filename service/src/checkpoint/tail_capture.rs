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

//! The tail **capture unit**: turns Task 2's classified sound prefix into a
//! persisted tail generation (SDD SR-1.2 / SR-1.3 / SR-1.4 / SR-1.5, SR-2.1,
//! SR-2.4, SR-7.1 / SR-7.2, SR-9.2).
//!
//! One invocation performs, in this order and at most once each:
//!
//! 1. **One** `list_layer_checkpoints` — the lineage addressing (S0, the base's
//!    `snap_seq`, the chain length). No blob is ever fetched.
//! 2. **At most one** `obj:hdr` read — the post-restore fossil guard, whose
//!    four identity columns are reused for the tail's `CheckpointMeta`.
//! 3. **At most one** bounded stream read + the pure classification.
//! 4. **At most one** `put_layer_checkpoint`.
//!
//! Every refusal arm sits *before* the read it precedes, so a layer that cannot
//! produce a tail costs a listing and nothing else.
//!
//! # Division of duty — deliberately metrics-free and flag-agnostic
//!
//! Mirrors [`crate::checkpoint::assembler`] verbatim. This unit *decides* and
//! *persists*; it returns a [`TailCaptureDecision`] whose
//! [`TailCaptureDecision::outcome_label`] the tick wiring (Task 4) maps onto the
//! counter family. Nothing here touches an instrument, writes `obj:hdr`, or
//! stamps a checkpoint cursor.
//!
//! It is likewise **flag-agnostic**: this unit never consults the tail feature
//! toggle on [`TailCaptureConfig`]. The gate lives in the tick wiring, which is
//! also the only thing that knows the tail path is a fallback after an
//! assembly that persisted nothing (a header-sequence regression). Reading the toggle here would put the gate in two
//! places and make the unit untestable without setting it.
//!
//! # Determinism boundary (SR-9.2)
//!
//! SR-9.2's idempotent-PUT argument is that two leaders racing on the same
//! `(object_id, to_seq, tail)` address write **byte-identical envelopes**, so
//! blind last-writer-wins is safe. That holds because every input to
//! [`encode_tail_envelope`] is derived from the immutable slice and the lineage
//! head: record sequences and `server_ts` come from the stream entries, `ompf`
//! is the writer's verbatim payload, and the header's four numbers are the
//! object id, the listing's base sequence, the listing's S0, and the
//! classification's `to_seq`.
//!
//! The two values that are *not* derived that way — the wall clock
//! (`now_unix_ms()`) and the replica label — live in [`CheckpointMeta`], which
//! is deliberately **outside** the bytes the argument depends on. Keep it that
//! way: a clock read or a replica identifier that reached the envelope would
//! make two leaders' artifacts differ and turn an idempotent overwrite into a
//! coin flip.

use crate::checkpoint::put_guard::ObjectLivenessProbe;
use crate::checkpoint::store::{
    CheckpointError, CheckpointHandle, CheckpointKind, CheckpointMeta, CheckpointStore,
};
use crate::checkpoint::tail_classify::{
    classify_tail_slice, TailClassification, TailRefusal, TailSliceLoader, ValkeyTailSliceLoader,
};
use crate::checkpoint::tail_codec::{
    encode_tail_envelope, TailHeader, TAIL_HEADER_SIZE, TAIL_RECORD_PREFIX_SIZE,
};
use crate::checkpoint::TailCaptureConfig;
use crate::storage::client::ValkeyClient;
use crate::storage::keys::KeySchema;
use crate::storage::tail_slice::{TailSliceCaps, TAIL_SLICE_PAGE_COUNT};

// ── Result types ────────────────────────────────────────────────────────────

/// What one capture invocation decided to do.
///
/// Three outcomes, not two — persisted, deliberately persisted nothing, or
/// failed. Only the third is a [`CheckpointError`], exactly as
/// [`crate::checkpoint::assembler::AssemblyDecision`] splits them.
#[derive(Debug, Clone)]
pub enum TailCaptureDecision {
    /// A tail envelope was encoded and persisted.
    Captured {
        /// The handle the adapter assigned. Its `snap_seq` is `to_seq`.
        handle: CheckpointHandle,
        /// S0 — the **exclusive** lower bound of the covered range, i.e. the
        /// lineage head this tail layers onto. The first captured record's own
        /// sequence is `from_seq + 1`.
        from_seq: u64,
        /// The last captured commit's sequence; the generation's ordering
        /// number and the adapter address's sequence component.
        to_seq: u64,
        /// Commits carried. Always `>= 1` — SR-1.3 forbids persisting an empty
        /// prefix, and this unit is the enforcement point.
        record_count: usize,
        /// A cap truncated the read, so the prefix is shorter than the stream
        /// had available. The claimed range still equals the content.
        capped: bool,
    },
    /// Nothing was persisted; the next tick re-attempts.
    Refused(TailCaptureRefusal),
}

/// Why a capture persisted nothing.
///
/// The first four arms are this unit's own call-site refusals; the fifth
/// passes Task 2's classifier verdict through unchanged so the four refusal
/// spellings keep a single owner.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TailCaptureRefusal {
    /// The listing is EMPTY — this object has no durable generation at all, so
    /// there is nothing for a tail to extend (SR-1.5).
    ///
    /// Routine and self-healing: the layer was never checkpointed, or retention
    /// pruned its last generation, and the first full backup that lands clears
    /// it. On a fleet that has just enabled the feature this is the bulk
    /// rollout population, which is why its arm logs at DEBUG.
    ///
    /// **Deliberately narrow.** A listing that holds generations none of which
    /// can root a lineage is [`TailCaptureRefusal::LineageUnrootable`], not
    /// this — see that variant for why the two do not share a cell.
    NoBase,
    /// Generations exist for this object but NONE of them can root a lineage:
    /// every generation in the listing is a tail, or the lineage head sits at
    /// sequence 0 (SR-1.5 / SR-7.1 / SR-7.2).
    ///
    /// Neither shape is producible by any shipped write path — a tail is only
    /// ever written after the base it names, and SR-7.1 roots every lineage at
    /// `snap_seq = 1` — so this means the checkpoint store was damaged: a base
    /// manually pruned out from under its tails, partial store loss, or a
    /// hand-written store. Both arms log at WARN and name which.
    ///
    /// The two arms are deliberately ONE variant and one cell. They share a
    /// remediation (repair the checkpoint store; no
    /// `LIVE_EDIT_CHECKPOINT_TAIL_*` knob touches either) and their distinct
    /// tracing events carry the diagnosis, which is the separability convention
    /// this subsystem's folds are held to.
    ///
    /// **That convention obliges a POSITIVE discriminator**, and the field sets
    /// below are load-bearing, not incidental — an operator who has filtered a
    /// log stream to this one cell separates the arms by selecting a field that
    /// is PRESENT, never by noticing one that is missing (a query for "the event
    /// without `newest_sequence`" also matches a truncated or re-formatted line):
    ///
    /// - tails-only — `object_id`, `generations`, `newest_sequence`, `outcome`.
    /// - head-at-zero — `object_id`, `generations`, `head_sequence`, `outcome`.
    ///   `head_sequence` IS the diagnosis, and it is why the field is not folded
    ///   into `newest_sequence`: the two arms would then differ in no field at
    ///   all, and this arm's whole content is *which* sequence the head sits at.
    ///
    /// Changing either set is a contract change (SR-7 addendum item 14), not a
    /// log tidy-up.
    LineageUnrootable,
    /// The chain bound was reached — see [`lineage_shape`]'s chain-count
    /// discussion (SR-2.5 / SR-4.1) and [`ChainBound`] for which measurement the
    /// bound was tested against.
    ///
    /// This unit's answer is not necessarily the tick's. The checkpoint loop
    /// treats this return as the trigger for the SR-4.1 chain-collapse re-base
    /// (`checkpointer::rebase_on_chain_full`): the layer either collapses
    /// (`rebased`), or re-runs this capture under [`ChainBound::Verified`] and
    /// captures, or — only when the re-base itself could not land — reports
    /// `refused_chain_full` after all.
    ChainFull {
        /// Generations above the newest base **in the listing** — always the
        /// listing count, including on a [`ChainBound::Verified`] run, so the
        /// reported number is the one an operator's own listing reproduces.
        chain_len: usize,
        /// K, from `LIVE_EDIT_CHECKPOINT_TAIL_MAX_CHAIN`.
        max_chain: usize,
    },
    /// A DR restore has landed for this object and no post-restore generation
    /// has cleared its stamp yet, so the stream may still hold
    /// abandoned-timeline fossils the slice cannot see. Refuses conservatively.
    PostRestoreUnrebased {
        /// `obj:hdr restore_seq_bump` as read. `0` here means the column itself
        /// was untrustworthy (present but unparseable) rather than non-zero.
        restore_seq_bump: u64,
    },
    /// Task 2's classifier found no sound prefix.
    Classified(TailRefusal),
}

impl TailCaptureDecision {
    /// This decision's `outcome` label on `live_edit_checkpoint_tail_total`.
    ///
    /// Every returned value is a member of `CHECKPOINT_TAIL_OUTCOMES`
    /// (`src/observability/metrics.rs`), which is the canonical vocabulary.
    /// This unit mints the two capture spellings (it is the only code that
    /// knows the capped/uncapped distinction at the outcome level) plus its own
    /// three call-site refusals; the four classifier spellings are **delegated**
    /// to [`TailRefusal::outcome_label`] and never re-spelled here.
    ///
    /// # The post-restore fold, and why it stays a fold
    ///
    /// [`TailCaptureRefusal::PostRestoreUnrebased`] reports itself with the
    /// same label as [`TailRefusal::SeqGap`]. That is deliberate. The guard
    /// fires on exactly the state SR-1.2 / SR-3.6 name as the post-restore
    /// fossil boundary — it is the *same operator-facing condition*, caught one
    /// read earlier because a jump beyond the cap is invisible to the slice.
    /// Minting a separate label would fork the dashboard vocabulary Task 1
    /// pinned for a distinction with no distinct remediation. The diagnostic
    /// difference is carried by the distinct enum variant and by a distinct
    /// tracing event, which is where it is actionable.
    ///
    /// # `refused_lineage_unrootable` is NOT a fold, and why the two rules agree
    ///
    /// [`TailCaptureRefusal::LineageUnrootable`] earns its own label under the
    /// same test the fold above passes, taken in the other direction: the bar
    /// is **a distinct operator remediation**, not a distinct internal state.
    /// A `refused_no_base` layer needs nothing done to it — a full backup is
    /// already coming, and on a post-enable fleet a bulk population of them
    /// drains on its own. A `LineageUnrootable` layer needs a **human to repair
    /// the checkpoint store**, and no amount of waiting or knob-turning will
    /// clear it. Folding the two would put a condition that never self-heals
    /// inside a cell whose documented healthy reading is a draining rollout
    /// population — precisely where an operator would not look for it.
    ///
    /// Splitting it also leaves `refused_no_base` meaning exactly one thing
    /// ("this object has no durable generation at all").
    pub fn outcome_label(&self) -> &'static str {
        match self {
            TailCaptureDecision::Captured { capped: false, .. } => "captured",
            TailCaptureDecision::Captured { capped: true, .. } => "captured_capped",
            TailCaptureDecision::Refused(TailCaptureRefusal::NoBase) => "refused_no_base",
            TailCaptureDecision::Refused(TailCaptureRefusal::LineageUnrootable) => {
                "refused_lineage_unrootable"
            }
            TailCaptureDecision::Refused(TailCaptureRefusal::ChainFull { .. }) => {
                "refused_chain_full"
            }
            // The fold — see this method's doc.
            TailCaptureDecision::Refused(TailCaptureRefusal::PostRestoreUnrebased { .. }) => {
                "refused_seq_gap"
            }
            TailCaptureDecision::Refused(TailCaptureRefusal::Classified(refusal)) => {
                refusal.outcome_label()
            }
        }
    }
}

// ── Chain bound ─────────────────────────────────────────────────────────────

/// Which measurement of the lineage's chain length the K bound is tested
/// against on this invocation.
///
/// The unit has exactly **one** bound comparison — the
/// `>= cfg.max_chain` test that mints [`TailCaptureRefusal::ChainFull`] — and
/// this enum is the only thing that chooses its left-hand side. Nothing else
/// moves: [`LineageShape::chain_len`], the listing count, remains what the
/// refusal reports and what the success line logs, so an operator's own
/// `list` still reproduces every number the unit prints.
///
/// # Why a second bound source exists — the deliberate divergence
///
/// [`lineage_shape`]'s doc carries the four grounds for the listing count being
/// a conservative over-approximation of the `from`/`to` link-walk SR-9.2
/// describes: no meta-only read exists on [`CheckpointStore`], SR-4.2's
/// per-tick cost model budgets the tick as one listing / one bounded XRANGE /
/// one small PUT, the approximation errs safe, and the torn-put meta case is
/// out of reach on this path. All four still hold and the approximation
/// **stays**, as the per-tick pre-filter here.
///
/// Over-counting has exactly one source: SR-9.2's divergent-S duplicate-leader
/// fork, which can leave overlapping tails above the base that are not members
/// of the walked chain. On that fork the listing reads `>= K` while the true
/// walk reads `< K` — the layer is genuinely under the bound and should
/// capture, not refuse every tick forever. The checkpoint loop answers that
/// question, because on the `ChainFull` return
/// `checkpointer::rebase_on_chain_full` gathers the lineage anyway (to compose
/// the SR-4.1 re-base) and thereby learns the true length.
/// `ChainBound::Verified(n)` is how it says *"I have walked it; the bound is
/// not reached"*, and the capture re-runs under it.
///
/// **This is not a licence to skip the listing count.** `Verified` is only ever
/// produced downstream of a gather, and a gather is only ever paid for because
/// the listing-count pre-filter already said the chain was at K. Minting one on
/// a tick that has not paid for the walk would put a `1 base + K tails` blob
/// fetch inside the per-tick budget ground 2 forbids.
///
/// # Accepted residual — a stale `Verified(n)` can overshoot by one tail
///
/// `n` is measured by a gather that, under
/// [`rebase_layer_lineage`](crate::checkpoint::rebase::rebase_layer_lineage)'s
/// own budget derivation, may run for up to
/// `(max_chain + 2) × rebase_fetch_reserve`. A peer leader can append a tail
/// inside that window, so a capture re-run driven by a `Verified(n)` measured
/// before that append can capture while the true walk has since reached K —
/// **overshooting the bound by one tail for one tick**. It is self-correcting:
/// the next eligible tick re-lists, re-measures, and re-bases. It sits in the
/// same conservative family as the listing over-approximation above — a
/// bounded, self-healing disagreement between a measurement and the store it
/// was taken from — but it is a real window, and is stated here rather than
/// left for a later reader to discover.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ChainBound {
    /// Test the bound against the listing count ([`LineageShape::chain_len`]).
    ///
    /// The pre-existing behaviour, byte for byte, and what every entry point
    /// other than [`capture_layer_tail_with_bound`] passes.
    FromListing,
    /// Test the bound against a true link-walk length the caller has already
    /// measured.
    ///
    /// Only the checkpoint loop can produce one, and only downstream of a
    /// gather — see the enum doc for why, and for the staleness residual. Its
    /// one production constructor is `checkpointer::attempt_tail_capture_fallback`'s
    /// `RebaseDecision::UnderBound` fork.
    Verified(usize),
}

impl ChainBound {
    /// The number the K comparison reads, given this invocation's listing count.
    fn measured(self, listing_chain_len: usize) -> usize {
        match self {
            ChainBound::FromListing => listing_chain_len,
            ChainBound::Verified(verified) => verified,
        }
    }

    /// Log-field descriptor naming which measurement [`ChainBound::measured`]
    /// returned, so `bound_chain_len` in a log line is never ambiguous about
    /// its provenance.
    fn source_label(self) -> &'static str {
        match self {
            ChainBound::FromListing => "listing",
            ChainBound::Verified(_) => "verified",
        }
    }
}

// ── Entry points ────────────────────────────────────────────────────────────

/// Capture one tail generation for `object_id`, or report why not.
///
/// Constructs the production `ValkeyTailSliceLoader` and forwards to
/// `capture_layer_tail_with`, exactly as `assemble_layer_checkpoint` builds a
/// `ValkeySnapshotLoader` and forwards to its inner function. Reads use the
/// default Valkey data pool (read-only, benefits from multiplexing).
///
/// `pub` rather than `pub(crate)` on purpose, and the asymmetry with the seam
/// form below is worth one sentence: Task 7's SR-9.2 duplicate-leader test
/// drives two *concurrent* captures from the integration-test crate — two
/// checkpoint loops would serialize on `le:checkpoint:leader`, so the unit has
/// to be directly invokable from outside the crate, the same way
/// `tests/integration/checkpoint/assembler_test.rs` invokes the assembler. The
/// seam form stays `pub(crate)` because its `loader` parameter names a
/// `pub(crate)` trait, which a `pub` signature could not carry.
///
/// # Errors
///
/// - [`CheckpointError::NotFound`] — `obj:hdr` is absent (the layer was deleted
///   or overwritten); the caller drops the object from its working set.
/// - [`CheckpointError::Transient`] — a transient backend failure on the
///   listing, the header read, the slice read, or the put.
/// - [`CheckpointError::Permanent`] — envelope encoding failed, or the adapter
///   put failed permanently.
/// - [`CheckpointError::Conflict`] — the adapter surfaced an
///   optimistic-concurrency race on the put.
pub async fn capture_layer_tail(
    client: &ValkeyClient,
    keys: &KeySchema,
    object_id: u64,
    store: &dyn CheckpointStore,
    replica_id: &str,
    cfg: &TailCaptureConfig,
) -> Result<TailCaptureDecision, CheckpointError> {
    let loader = ValkeyTailSliceLoader { client, keys };
    capture_layer_tail_with_bound(
        &loader,
        store,
        object_id,
        replica_id,
        cfg,
        ChainBound::FromListing,
        // This form holds a live client, so the pre-PUT liveness probe is
        // available here. `capture_layer_tail_with` — the fake-loader seam —
        // passes `None`; see that function's doc.
        Some(ObjectLivenessProbe::new(client, keys)),
    )
    .await
}

/// [`capture_layer_tail`] over an injected [`TailSliceLoader`].
///
/// The unit-testable form: the in-file test lane drives every arm through a
/// fake with no `TestValkey`.
///
/// Like every form, it takes the lineage listing itself and derives its
/// addressing through [`lineage_shape`], whose doc carries the load-bearing S0
/// and chain-count reasoning.
///
/// It likewise always tests the K bound against [`ChainBound::FromListing`],
/// and that asymmetry is deliberate rather than an omission: the only caller
/// that can hold a verified link-walk length is the checkpoint loop, which uses
/// the `_with_bound` form. Widening this signature (or [`capture_layer_tail`]'s
/// `pub` one) to carry a bound would offer a `Verified` variant to callers that
/// have not paid for the walk that earns it — see [`ChainBound`]'s doc.
///
/// # `S0 >= 1` is a guard, not a formality
///
/// `CreateLayer` XADDs a seed entry at `1-0` that is `update`-typed and
/// deliberately carries no `ompf` field **while accompanying the full initial
/// layer content**. The classifier would decode it as an empty commit and
/// replay it as a no-op over real content — silent corruption. It is
/// unreachable only because the read window opens at `{S0+1}-0` and SR-7.1
/// roots every lineage at `snap_seq = 1`. The `s0 == 0` arm below is the
/// production guard that keeps it so.
///
/// # Caps: this unit owns the envelope framing reservation
///
/// [`TailSliceCaps::max_payload_bytes`] counts **raw verbatim payload bytes
/// only**. Framing is the caller's to reserve, so the derivation below subtracts
/// [`TAIL_HEADER_SIZE`] plus [`TAIL_RECORD_PREFIX_SIZE`] per admissible entry
/// from the operator's configured byte cap before handing the primitive a
/// budget. Without it a prefix sized exactly to the cap would encode into an
/// envelope `44 + 21·N` bytes *over* it.
///
/// At stock defaults the reservation is roughly 0.13 % of the cap — it is a
/// correctness reservation, not a meaningful shrink of what a tail may carry.
///
/// Under a pathological configuration the whole chain is observable rather than
/// silent: the derivation refuses before any read, WARNing with both knobs
/// named. (Config rejects a byte cap below `TAIL_HEADER_SIZE +
/// TAIL_RECORD_PREFIX_SIZE + 1` = 66 at startup, so the arm is
/// defense-in-depth.) If a cap admits entries but excludes the first *commit*,
/// the primitive stops with a byte cap and no entries and the classifier returns
/// its own refusal — the layer then refuses observably every tick, which is the
/// intended operator signal.
///
/// # No liveness probe
///
/// It passes `None` for the pre-PUT `obj:hdr` liveness probe (b.hka): this form
/// exists precisely for the lane that has no Valkey, so there is nothing to
/// probe. A `None` probe leaves the PUT unguarded, which is byte-for-byte the
/// behaviour every caller of this form already had — the probe is an
/// optimisation over a race that a fake store cannot exhibit, and the b.vn6
/// conditional cursor stamp remains the actual guard on the production path.
///
/// `#[cfg(test)]` since b.hka. It is `pub(crate)`, so no integration test could
/// ever reach it, and the production entry point above now calls
/// [`capture_layer_tail_with_bound`] directly in order to pass the liveness
/// probe — leaving this form used only by the in-file test lane it was written
/// for. Gating it says so rather than letting it read as a production seam.
#[cfg(test)]
pub(crate) async fn capture_layer_tail_with(
    loader: &dyn TailSliceLoader,
    store: &dyn CheckpointStore,
    object_id: u64,
    replica_id: &str,
    cfg: &TailCaptureConfig,
) -> Result<TailCaptureDecision, CheckpointError> {
    capture_layer_tail_with_bound(
        loader,
        store,
        object_id,
        replica_id,
        cfg,
        ChainBound::FromListing,
        None,
    )
    .await
}

/// [`capture_layer_tail_with`] with the K bound's source and the liveness probe
/// chosen by the caller. Like every form, it takes the lineage listing itself,
/// at the moment it addresses the tail, so the listing is never older than this
/// call.
///
/// # `chain_bound`
///
/// This is the only form that carries a [`ChainBound`]: the checkpoint loop is
/// the sole caller that can hold a *verified* link-walk length, having paid for
/// the gather on this layer's `ChainFull` return. Pass
/// [`ChainBound::FromListing`] unless you are that caller; it selects the
/// pre-existing behaviour byte for byte. [`ChainBound`]'s doc carries the whole
/// argument for why a second bound source exists, why the listing count is
/// still the pre-filter, and the one-tail staleness residual `Verified` accepts.
///
/// `probe` is the pre-PUT `obj:hdr` liveness probe (b.hka). `Some` on the
/// production path (the checkpoint loop and [`capture_layer_tail`] both hold a
/// client), `None` from the fake-loader lane — see
/// [`crate::checkpoint::put_guard`] for what the probe does and, more
/// importantly, what it does not close.
// Flat signature, matching the sibling forms above: every parameter is
// distinctly typed, so a transposition is a compile error. Bundling them into a
// context struct would obscure which arguments are per-tick and which are
// per-layer.
#[allow(clippy::too_many_arguments)]
pub(crate) async fn capture_layer_tail_with_bound(
    loader: &dyn TailSliceLoader,
    store: &dyn CheckpointStore,
    object_id: u64,
    replica_id: &str,
    cfg: &TailCaptureConfig,
    chain_bound: ChainBound,
    probe: Option<ObjectLivenessProbe<'_>>,
) -> Result<TailCaptureDecision, CheckpointError> {
    let result = capture_inner(
        loader,
        store,
        object_id,
        replica_id,
        cfg,
        chain_bound,
        probe,
    )
    .await;
    if let Err(err) = &result {
        match err {
            // A vanished header is a routine DeleteLayer / cleanup race, not a
            // capture failure — matching the assembler's posture on the same
            // race, so the WARN arm stays a signal operators can act on.
            CheckpointError::NotFound => {
                tracing::debug!(
                    object_id,
                    "tail capture skipped: object header vanished mid-capture"
                );
            }
            other => {
                tracing::warn!(object_id, error = %other, "tail capture failed");
            }
        }
    }
    result
}

// ── Lineage shape ───────────────────────────────────────────────────────────

/// The four numbers a checkpoint-store listing answers about a layer's lineage.
///
/// Derived in exactly one place ([`lineage_shape`]), which this unit's
/// addressing and refusal arms read. A second copy of the derivation is
/// precisely the drift the centralized `adapters::generation_sort_key` exists
/// to prevent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct LineageShape {
    /// The kind of the newest generation — the head, of *either* kind.
    pub head_kind: CheckpointKind,
    /// S0 — the head's sequence. May be `0`; see [`capture_layer_tail_with`]'s
    /// `S0 >= 1` guard, which is a refusal arm and not this helper's business.
    pub head_seq: u64,
    /// The sequence of the newest [`CheckpointKind::Base`] — the lineage root.
    /// Equal to `head_seq` only when the head *is* that base.
    pub base_snap_seq: u64,
    /// Generations above the newest base — the conservative chain count below.
    pub chain_len: usize,
}

/// Derive the [`LineageShape`] a checkpoint-store listing implies.
///
/// `None` when the listing is empty **or** carries no [`CheckpointKind::Base`]
/// at all. The two cases are deliberately not distinguished here: they carry
/// different operator meanings (routine lifecycle vs. suspected store loss) and
/// different log levels, so the caller keeps its own `is_empty` check and traces
/// them apart. Folding them into one arm here would lose that distinction; a
/// two-variant error type would put the caller's vocabulary in this helper.
///
/// # Lineage addressing — S0 comes from the store listing, never from `obj:hdr`
///
/// `S0` is `handles[0].seq()` — the **lineage head**, the newest generation of
/// *either* kind. Three things about that are load-bearing:
///
/// - **Never the `obj:hdr last_checkpoint_seq` cursor.** SDD §22.7 step 8
///   tolerates that cursor lagging when its write fails, and a lagging cursor
///   would re-read commits an existing generation already covers, minting
///   overlapping tails. The listing is the single source of truth.
/// - **The head, not the newest base.** SR-2.1 requires
///   `tail_{n+1}.from_seq == tail_n.to_seq`, so once one tail exists the head is
///   the only sound anchor; rooting every tail at the base is the spec's
///   rejected alternative (b), which re-captures already-captured commits. Do
///   not "repair" `handles[0]` to the newest base. SR-1.2's word *restorable*
///   describes the lineage's restorability **as a unit** (base + contiguous
///   tails), which the landed lineage-aware *restore selection*
///   (`recovery::select_restore_source` → `gather_restore_lineage`) now
///   consumes; that selection is a separate question and deliberately does not
///   move S0.
/// - **The ordering is not re-derived.** The listing arrives newest-first with
///   Base pinned before Tail at an equal sequence, guaranteed by the shared
///   `generation_sort_key` every adapter sorts through. A second copy of that
///   rule here is exactly the drift centralizing the key exists to prevent. It
///   is also what makes `position(kind == Base)` the newest base and its index
///   the count of generations above it — do NOT re-sort.
///
/// # The chain count is a documented conservative over-approximation
///
/// [`LineageShape::chain_len`] is the *position of the newest base in the
/// listing* — a raw count of generations above it — not the `from`/`to`
/// link-walk SR-9.2 describes. The deviation is deliberate:
///
/// 1. **No meta-only read exists on [`CheckpointStore`].** A true link-walk
///    would fetch up to K full generations, each a multi-MB blob, on every
///    capture.
/// 2. **SR-4.2's cost model budgets this tick as one listing, one bounded
///    XRANGE, and one small PUT** — no blob fetch, all inside the leader lease
///    b.s5y protects. A K-blob walk does not fit that budget.
/// 3. **The approximation errs safe.** Every true chain member ending at the
///    newest tail lies above the newest base, so the raw count is `>=` the true
///    walk length: it can refuse *earlier* than K, never admit an over-long
///    chain. Over-counting arises only from SR-9.2's divergent-S duplicate-leader
///    fork, which can leave overlapping non-member tails — a rare, self-healing
///    state whose worst consequence is one unearned gather, which the loop's
///    re-base resolves by re-running the capture under [`ChainBound::Verified`].
/// 4. **The torn-put meta case is out of reach here, not forgotten.** A
///    generation whose meta companion is absent is refused by the adapter
///    (`MissingMetaCompanion`, b.hka (d)) and stepped over by Epic 4's gather
///    (`t1.dfh.yj`), which reads meta; this unit reads no meta at all, so
///    neither the refusal nor the lineage-`None` it replaced can arise on this
///    path.
///
/// The restore path's true link-walk (`GatheredLineage::chain_len`) is
/// deliberately **not** an option for a per-tick *capture* caller, for grounds 1
/// and 2: it fetches up to `1 base + K tails` of multi-MB blobs, which the
/// per-tick budget does not admit. Where the two disagree, this count is the
/// larger, so the tick reaches the bound earlier.
///
/// **It IS an option for the re-base**, and that is not a contradiction. The
/// chain-collapse re-base ([`crate::checkpoint::rebase::rebase_layer_lineage`])
/// pays exactly that walk — but only on a tick this cheap pre-filter has already
/// said is at K, and only because it must fetch the whole lineage anyway to
/// compose the new base. The walk is therefore free at the point it is consumed,
/// and it is reached at most every K-th eligible tick rather than per tick.
/// Grounds 1 and 2 fence the *capture*; they were never a claim that the walk is
/// unaffordable to a unit whose cost class is already `1 base + K tails`.
///
/// Having paid for it, that caller hands the length back *in*: on the
/// `ChainFull` return, `checkpointer::rebase_on_chain_full` gathers the lineage
/// and either collapses it or re-runs the capture with [`ChainBound::Verified`],
/// resolving ground 3's divergent-S over-count in the direction that captures.
/// No *per-tick* walk is added. [`ChainBound`] is where that divergence is
/// argued in full.
pub(crate) fn lineage_shape(handles: &[CheckpointHandle]) -> Option<LineageShape> {
    let base_index = handles
        .iter()
        .position(|handle| handle.kind == CheckpointKind::Base)?;
    let head = handles.first()?;
    Some(LineageShape {
        head_kind: head.kind,
        head_seq: head.seq(),
        base_snap_seq: handles[base_index].seq(),
        chain_len: base_index,
    })
}

/// The capture itself. Split out so the wrapper's failure tracing is one match
/// without nesting per step; every success / refusal arm traces at its own site,
/// where the arm-specific context lives.
// Flat signature, matching the sibling forms above: every parameter is
// distinctly typed, so a transposition is a compile error. Bundling them into a
// context struct would obscure which arguments are per-tick and which are
// per-layer.
#[allow(clippy::too_many_arguments)]
async fn capture_inner(
    loader: &dyn TailSliceLoader,
    store: &dyn CheckpointStore,
    object_id: u64,
    replica_id: &str,
    cfg: &TailCaptureConfig,
    chain_bound: ChainBound,
    probe: Option<ObjectLivenessProbe<'_>>,
) -> Result<TailCaptureDecision, CheckpointError> {
    // ── 1. Lineage addressing: one listing, no blob fetch ───────────────────
    // Taken here, at the moment the tail is addressed, never handed in: a
    // listing taken earlier in the tick would be one proven snapshot read old
    // — a fenced read's wait and hold, for a layer above the atomic-read
    // limits — and in that window generations can land without the leader
    // lock (the creation-time and post-restore bootstrap bases).
    //
    // Cost residual, and it is NOT the same listing: `prune_after_write`'s
    // listing (`src/checkpoint/checkpointer.rs`, the retention prune) still runs
    // once more per capture success. That one is deliberately not shareable, and
    // it carries TWO jobs this listing cannot do, both because this one is taken
    // BEFORE the write: it is the prune's observation of the generation the tick
    // just wrote, and it is the input `plan_retention` groups into lineage units
    // (SR-5.1). Collapsing it onto this one would retain one generation too many
    // on every eligible tick AND group the units from a pre-write shape — which
    // on a capture success means grouping the lineage without the very tail this
    // call is about to add to it.
    //
    // THE RE-BASE TICK PAYS MORE, in two distinct shapes (SR-4.1):
    //
    //  - `RebaseDecision::Rebased` — THREE listings: this call's,
    //    `gather_restore_lineage`'s own (`lineage.rs` — its signature is Epic
    //    4's cross-task contract and deliberately takes no pre-taken listing),
    //    and the prune's.
    //  - `RebaseDecision::UnderBound` — FOUR: the same three, plus the
    //    `ChainBound::Verified` re-run's own, since every call lists for itself.
    //
    // Both shapes are off the per-capture cost model on purpose: a re-base is
    // SR-9.1's declared HEAVY unit, reached at most every K-th eligible tick.
    let handles = store.list_layer_checkpoints(object_id).await?;

    if handles.is_empty() {
        let decision = TailCaptureDecision::Refused(TailCaptureRefusal::NoBase);
        // Routine: never checkpointed, or retention pruned the last generation.
        tracing::debug!(
            object_id,
            outcome = decision.outcome_label(),
            "tail capture refused: the object has no durable generation to root a lineage at"
        );
        return Ok(decision);
    }

    // One derivation — see `lineage_shape`. `None` here is the tails-only
    // listing: the empty case
    // already returned above, so the two keep their distinct log levels,
    // operator meanings AND — since the SR-7 adjudication — their distinct
    // outcome cells. This arm is one of the two `refused_lineage_unrootable`
    // producers; the empty one above is the sole `refused_no_base` producer.
    let Some(shape) = lineage_shape(&handles) else {
        let decision = TailCaptureDecision::Refused(TailCaptureRefusal::LineageUnrootable);
        // Mirrors `recovery::select_restore_source`'s no-base arm: the shipped
        // write paths cannot produce a tails-only lineage (a tail is only ever
        // written after the base it names), so this means manual pruning or
        // partial store loss and deserves a human look.
        //
        // `newest_sequence` is this arm's POSITIVE discriminator against the
        // head-at-zero arm below, which folds onto the same
        // `refused_lineage_unrootable` cell and carries `head_sequence`
        // instead. See `TailCaptureRefusal::LineageUnrootable` for the pinned
        // field sets.
        tracing::warn!(
            object_id,
            generations = handles.len(),
            newest_sequence = handles[0].seq(),
            outcome = decision.outcome_label(),
            "tail capture refused: every durable generation for this object is a tail, so there \
             is no base to root the lineage at. Not producible by the shipped write paths — \
             suspect manual pruning of the base or partial checkpoint-store loss"
        );
        return Ok(decision);
    };

    let base_snap_seq = shape.base_snap_seq;
    // S0 is the LINEAGE HEAD — newest generation of either kind. Deliberately
    // not named `base_snap_seq`: for a chain length >= 1 the two numbers differ,
    // and conflating them re-captures already-captured commits and breaks
    // SR-2.1's `tail_{n+1}.from_seq == tail_n.to_seq` contiguity.
    let s0 = shape.head_seq;
    // Generations above the newest base — the conservative chain count. See
    // `lineage_shape` for all four grounds.
    let chain_len = shape.chain_len;

    if s0 == 0 {
        let decision = TailCaptureDecision::Refused(TailCaptureRefusal::LineageUnrootable);
        // The read window would open at `1-0`, which is CreateLayer's
        // `update`-typed, deliberately ompf-less seed entry — and that entry
        // accompanies the FULL initial layer content, so the classifier would
        // record it as an empty commit and replay it as a no-op over real
        // state. SR-7.1 roots every lineage at snap_seq = 1, so a head at 0 is
        // a store nobody's shipped write path can produce.
        //
        // `head_sequence` is this arm's POSITIVE discriminator against the
        // tails-only arm above, which folds onto the same
        // `refused_lineage_unrootable` cell and carries `newest_sequence`
        // instead. It reads 0 by construction — that IS the diagnosis, and the
        // field's presence is what lets an operator select this arm out of the
        // cell rather than infer it from an absent field. See
        // `TailCaptureRefusal::LineageUnrootable` for the pinned field sets.
        tracing::warn!(
            object_id,
            generations = handles.len(),
            head_sequence = s0,
            outcome = decision.outcome_label(),
            "tail capture refused: the lineage head sits at sequence 0, so the read window would \
             start at CreateLayer's seed stream entry. The service roots every lineage at \
             snap_seq = 1; suspect a hand-written checkpoint store"
        );
        return Ok(decision);
    }

    // ── The ONE bound comparison in this unit ───────────────────────────────
    //
    // Its left-hand side is the only thing `ChainBound` moves. By default that
    // is `chain_len`, the listing count — a deliberate conservative
    // over-approximation of SR-9.2's `from`/`to` link-walk, for the four grounds
    // on `lineage_shape` (no meta-only read on the trait; SR-4.2's per-tick cost
    // model; it errs safe; the torn-put meta case is out of reach here). Those
    // grounds all still hold and the approximation stays as the per-tick
    // pre-filter.
    //
    // `ChainBound::Verified(n)` substitutes a true link-walk length the caller
    // has ALREADY paid a gather for — the divergent-S duplicate-leader fork,
    // where the listing over-counts overlapping non-member tails to `>= K` while
    // the layer is genuinely under the bound and should capture rather than
    // refuse forever. It reaches this comparison and nothing else: `chain_len`
    // stays what the refusal reports and what the success line logs, so the
    // operator-visible number is always the one their own `list` reproduces,
    // with the compared number alongside it. See `ChainBound`'s doc for the full
    // argument, for why this is not a licence to skip the listing pre-filter,
    // and for the accepted one-tail staleness residual.
    let bound_chain_len = chain_bound.measured(chain_len);
    if bound_chain_len >= cfg.max_chain {
        // THIS RETURN IS THE RE-BASE TRIGGER (SR-4.1), not the tick's verdict.
        // `checkpointer::rebase_on_chain_full` gathers the lineage on it and
        // either composes a new base that counts `rebased` (true length >= K) or
        // re-runs this capture under `ChainBound::Verified` (true length < K).
        // The refusal below is unchanged and remains the honest answer whenever
        // no verified length says otherwise and no re-base landed.
        //
        // READ THE EVENT BELOW WITH THAT ORDERING IN MIND — it is why the event
        // is DEBUG and why its label field is named `capture_outcome`:
        //
        // It is emitted on the way IN to a re-base attempt, before the loop has
        // decided anything, so it fires on every collapse tick including the
        // ones that succeed. At WARN that is one WARN per collapse tick — noise
        // on a designed path — and it is
        // never the terminal verdict in production: `capture_layer_tail` has no
        // production caller, and the only production caller of
        // `capture_layer_tail_with_bound` is
        // `checkpointer::attempt_tail_capture_fallback`, which ALWAYS intercepts
        // `ChainFull`. Every downstream non-success path emits its own WARN
        // carrying the true verdict (the five `RebaseRefusal` arms in
        // `rebase.rs`, plus the fossil and lease gates in
        // `rebase_on_chain_full`), and the success path is observable through
        // `tail_total{rebased}` and `rebase_on_chain_full`'s `info!` — so
        // demoting loses no signal.
        //
        // `capture_outcome`, NOT `outcome`: this is the capture DECISION's
        // label, not the tick's. Only the `live_edit_checkpoint_tail_total` cell
        // the loop finally moves says whether the chain actually collapsed, and
        // the `outcome` key has to keep one meaning across logs and metrics. The
        // wording below is chosen to survive the same way: it states the
        // condition, not a verdict.
        let decision = TailCaptureDecision::Refused(TailCaptureRefusal::ChainFull {
            chain_len,
            max_chain: cfg.max_chain,
        });
        tracing::debug!(
            object_id,
            chain_len,
            bound_chain_len,
            chain_bound_source = chain_bound.source_label(),
            max_chain = cfg.max_chain,
            base_snap_seq,
            s0,
            capture_outcome = decision.outcome_label(),
            "tail capture declined: the lineage already carries the maximum number of tails above \
             its base, so the checkpoint loop now attempts a chain-collapse re-base for \
             this layer. No action is needed unless the tail-outcome counter also moves \
             refused_chain_full for it, which means the re-base did not land — the re-base logs \
             its own reason. Raising LIVE_EDIT_CHECKPOINT_TAIL_MAX_CHAIN remains the manual lever"
        );
        return Ok(decision);
    }

    // ── 2. Caps derivation — the framing reservation this unit owns ─────────
    //
    // Constants imported from the codec, never re-spelled as 44 / 21.
    let framing_budget = cfg.max_bytes.saturating_sub(TAIL_HEADER_SIZE);
    let entries_by_bytes = framing_budget / TAIL_RECORD_PREFIX_SIZE;
    // `.max(1)` and the saturating subtractions are the pathological-config
    // clamps; the zero-budget arm just below is what makes that state
    // observable instead of silent.
    let max_entries = usize::try_from(cfg.max_entries)
        .unwrap_or(usize::MAX)
        .min(entries_by_bytes)
        .max(1);
    let reserve = TAIL_HEADER_SIZE + TAIL_RECORD_PREFIX_SIZE * max_entries;
    let max_payload_bytes =
        u64::try_from(cfg.max_bytes.saturating_sub(reserve)).unwrap_or(u64::MAX);

    if max_payload_bytes == 0 {
        // REFUSE, do not silently clamp: a byte cap that cannot fit one framed
        // record is an operator error, and a zero payload budget would have the
        // primitive exclude the first entry on every tick forever.
        //
        // Folded onto the classifier's `MissingBytes` rather than a new label
        // because the honest statement is identical to the oversize-first-entry
        // case Task 2 already folds there — "no admissible first entry".
        tracing::warn!(
            object_id,
            max_bytes = cfg.max_bytes,
            max_entries,
            reserve,
            "tail capture refused: LIVE_EDIT_CHECKPOINT_TAIL_MAX_BYTES leaves no room for a \
             framed record after the envelope-framing reservation (a fixed header plus one \
             per-record prefix per admissible entry). Raise \
             LIVE_EDIT_CHECKPOINT_TAIL_MAX_BYTES or lower \
             LIVE_EDIT_CHECKPOINT_TAIL_MAX_ENTRIES"
        );
        return Ok(TailCaptureDecision::Refused(
            TailCaptureRefusal::Classified(TailRefusal::MissingBytes),
        ));
    }

    let caps = TailSliceCaps {
        max_entries,
        max_payload_bytes,
        page_count: TAIL_SLICE_PAGE_COUNT,
    };

    // ── 3. Post-restore fossil guard, before the stream read ────────────────
    //
    // `restore_seq_bump != 0` means "a DR restore happened for this object and
    // no post-restore generation has cleared the stamp yet" — exactly the window
    // in which the stream can still hold fossils of an abandoned timeline below
    // an ascending jump. Task 2's cap lookahead closes only the cap-BOUNDARY
    // case; a jump further beyond the cap is invisible to the slice by
    // construction, so a cap landing inside a fossil region would otherwise
    // capture abandoned-timeline commits into a generation that claims to
    // extend the live lineage.
    //
    // One read, two consumers: the guard below and the tail meta's four
    // identity fields at step 5.
    let Some(identity) = loader.read_header_identity(&object_id.to_string()).await? else {
        // The header vanished — or exists with an empty `session_id` /
        // `layer_name`, which `read_header_identity` reports the same way
        // (b.hka review round 2), since a generation stamped from one is
        // permanently unrestorable and the adapter refuses it with a
        // `Permanent` that would tick adapter-health every tick. Either way,
        // surface `NotFound` so the loop's existing arm drops the layer
        // silently. Nothing has been built yet, and there is nothing here to
        // fabricate: the same read supplies the fossil guard's own input.
        return Err(CheckpointError::NotFound);
    };

    // `cursor_columns_trusted == false` means a cursor column was present but
    // unparseable and folded to 0, so a fabricated 0 is indistinguishable from
    // a truthful one. Refusing is the conservative direction, matching
    // `assembler::measure_observed_commit_span`'s own decline posture on the
    // same input.
    if identity.restore_seq_bump != 0 || !identity.cursor_columns_trusted {
        let decision = TailCaptureDecision::Refused(TailCaptureRefusal::PostRestoreUnrebased {
            restore_seq_bump: identity.restore_seq_bump,
        });
        tracing::warn!(
            object_id,
            restore_seq_bump = identity.restore_seq_bump,
            cursor_columns_trusted = identity.cursor_columns_trusted,
            s0,
            outcome = decision.outcome_label(),
            "tail capture refused: a DR restore is outstanding for this object (or its cursor \
             columns are untrustworthy), so the stream may hold abandoned-timeline commits the \
             bounded read cannot see. Capture resumes once a post-restore generation lands"
        );
        return Ok(decision);
    }

    // ── 4. Bounded slice read + pure classification ─────────────────────────
    let slice = loader.read_tail_slice(object_id, s0, caps).await?;

    let captured = match classify_tail_slice(&slice, s0) {
        TailClassification::Captured(captured) => captured,
        TailClassification::Refused(refusal) => {
            let decision = TailCaptureDecision::Refused(TailCaptureRefusal::Classified(refusal));
            match refusal {
                // Routine lifecycle states on an idle or retired layer.
                TailRefusal::Trimmed | TailRefusal::Sentinel => tracing::debug!(
                    object_id,
                    s0,
                    entries = slice.entries.len(),
                    outcome = decision.outcome_label(),
                    "tail capture refused: no sound prefix in the read window"
                ),
                // Both say the evidence itself is unusable — worth a look.
                TailRefusal::MissingBytes | TailRefusal::SeqGap => tracing::warn!(
                    object_id,
                    s0,
                    entries = slice.entries.len(),
                    outcome = decision.outcome_label(),
                    "tail capture refused: the read window's evidence is not a sound replayable \
                     continuation of the lineage"
                ),
            }
            return Ok(decision);
        }
    };

    // Zero-record enforcement — the pinned point. Two facts make this
    // belt-and-braces: the codec deliberately encodes zero-record envelopes for
    // its own round-trips, and `TailCaptured` keeps its record vector private so
    // an empty capture is unconstructible outside the classifier. SR-1.3 still
    // forbids persisting one, and this file must have NO path to the put with
    // zero records, so the guard is written anyway.
    if captured.records().is_empty() {
        debug_assert!(
            false,
            "classifier returned Captured with zero records — an empty capture is \
             unconstructible outside the classifier"
        );
        return Ok(TailCaptureDecision::Refused(
            TailCaptureRefusal::Classified(TailRefusal::MissingBytes),
        ));
    }

    // The off-by-one, spelled out: `TailCaptured::from_seq` is the INCLUSIVE
    // first record sequence, while `TailHeader::from_seq` is the EXCLUSIVE lower
    // bound of `(from_seq, to_seq]` (the encoder enforces
    // `expected = header.from_seq + 1 + count`). The header is therefore built
    // from this unit's own S0, never from the classification's `from_seq`.
    debug_assert_eq!(
        captured.from_seq,
        s0 + 1,
        "the classifier's inclusive first sequence must be S0 + 1; the envelope header's \
         exclusive from_seq is S0 itself"
    );

    let to_seq = captured.to_seq;
    let capped = captured.capped;
    let record_count = captured.record_count();

    // ── 5. Encode ───────────────────────────────────────────────────────────
    //
    // Determinism boundary — see the module header: every input below is
    // slice-derived or listing-derived, so two leaders over the same immutable
    // slice and lineage head emit byte-identical bytes.
    let header = TailHeader {
        object_id,
        base_snap_seq,
        from_seq: s0,
        to_seq,
    };
    let envelope = encode_tail_envelope(&header, captured.into_records()).map_err(|e| {
        // Every reachable codec error is a caller bug that a retry
        // reproduces exactly — same classification as the assembler's OODF
        // build failures.
        CheckpointError::Permanent {
            source: format!("tail envelope encoding failed for object {object_id}: {e}").into(),
        }
    })?;
    // Bounded by the caps derivation at step 2: payload <= max_payload_bytes and
    // entries <= max_entries, so the total is at most
    // `reserve + (max_bytes - reserve) == max_bytes`.
    debug_assert!(
        envelope.len() <= cfg.max_bytes,
        "encoded tail envelope ({} bytes) exceeds the configured cap ({} bytes); the framing \
         reservation is wrong",
        envelope.len(),
        cfg.max_bytes
    );
    let envelope_byte_len = envelope.len();

    let meta = CheckpointMeta {
        creator_replica_id: replica_id.to_string(),
        written_at_unix_ms: crate::checkpoint::now_unix_ms(),
        snap_seq: Some(to_seq),
        // Identity is base-sourced and deliberately NOT blanked: `apply`'s
        // live-mapping identity resolution refuses identity-empty blobs, so a
        // tail meta carries the same four header columns a base does.
        session_id: identity.session_id,
        layer_name: identity.layer_name,
        created_by: identity.created_by,
        server_ts: identity.server_ts,
        // Exactly the defined quantity — "commits this generation covers that
        // the previous did not" — and honest by construction: step 3 refuses
        // whenever `restore_seq_bump != 0`, so the bump is always 0 here and
        // there is nothing to subtract. Stamping the no-measurement sentinel
        // instead would silently degrade every composed restore of a hot layer
        // back to the configured margin, which is the smaller-band hazard
        // direction (b.8y5). Which meta a composed restore reads is Epic 4's to
        // decide; a truthful value keeps that choice open.
        observed_commit_span: to_seq - s0,
        kind: CheckpointKind::Tail,
        base_snap_seq: Some(base_snap_seq),
        from_seq: Some(s0),
        // MANDATORY, not stylistic: the shared put-time gate rejects a
        // Some-sidecar tail put as `Permanent` (a sidecar under a tail stem is
        // never reaped and would pin its ext hashes in the live set), so reusing
        // the base meta builder unmodified would fail every capture at runtime.
        sidecar_bytes: None,
    };

    // ── 6. PUT ──────────────────────────────────────────────────────────────
    //
    // FENCE: no read of the existing artifact at this address, under any
    // circumstance. SR-9.2's `(object_id, to_seq, tail)` idempotency contract is
    // blind last-writer-wins over byte-identical content; a read-modify-write
    // here would destroy the duplicate-leader safety argument outright. `to_seq`
    // is the snap_seq argument — the trait's tail-caller obligation. Errors
    // propagate classified; there is no retry loop, because the tick is the
    // retry.
    //
    // Gated on the shared pre-PUT liveness probe (b.hka). Step 3's
    // `read_header_identity` already refused an absent header, but the slice
    // read and the envelope encode sit between that read and this PUT — up to
    // `LIVE_EDIT_CHECKPOINT_TAIL_MAX_BYTES` (64 MiB stock) of framing work — so
    // the window is real. `Ok(None)` folds onto `NotFound`, which is the arm
    // step 3 already uses for the identical condition and which the loop drops
    // silently with no metric tick. Not a duplicate of step 3's read: that one
    // feeds the fossil guard and the tail meta's identity columns and happens
    // before the encode; this one asks a single question immediately before the
    // write. `crate::checkpoint::put_guard` carries the residual.
    let Some(handle) = crate::checkpoint::put_guard::put_layer_checkpoint_if_live(
        probe,
        store,
        object_id,
        to_seq,
        envelope,
        meta,
        "tail capture",
    )
    .await?
    else {
        return Err(CheckpointError::NotFound);
    };

    let adapter_key = handle.adapter_key.clone();
    let decision = TailCaptureDecision::Captured {
        handle,
        from_seq: s0,
        to_seq,
        record_count,
        capped,
    };
    tracing::info!(
        object_id,
        base_snap_seq,
        from_seq = s0,
        to_seq,
        record_count,
        envelope_byte_len,
        capped,
        // The LISTING count, always — the number an operator's own
        // `list_layer_checkpoints` reproduces. `bound_chain_len` is what the K
        // comparison above actually read; the two differ only on a
        // `ChainBound::Verified` run, which is exactly the case worth seeing in
        // the log.
        chain_len,
        bound_chain_len,
        chain_bound_source = chain_bound.source_label(),
        adapter_key = adapter_key.as_str(),
        outcome = decision.outcome_label(),
        "layer tail captured and persisted"
    );
    Ok(decision)
}

#[cfg(test)]
mod tests {
    //! Unit lane for the tail **capture unit** ([`capture_layer_tail_with`]).
    //!
    //! Every case drives the seam form with a fake [`TailSliceLoader`] — the
    //! shared one from [`crate::checkpoint::tail_classify`], extended there
    //! rather than re-hand-rolled here — plus a real
    //! [`InMemoryCheckpointStore`] behind a call-counting decorator. Nothing
    //! here touches Valkey, an instrument, the checkpoint tick loop, or a
    //! clock: the unit is deliberately metrics-free and flag-agnostic (module
    //! header), so the outcome vocabulary is asserted as
    //! [`TailCaptureDecision::outcome_label`] **strings** and never as metric
    //! cells.
    //!
    //! Scope fences worth stating, because each has an owner elsewhere:
    //!
    //! - The classifier arm matrix belongs to `tail_classify.rs`'s own module.
    //!   Refusals appear here only as passthroughs, to prove the label and the
    //!   zero-PUT.
    //! - SR-9.2's *same-address duplicate-leader* race (two concurrent captures
    //!   onto one store) is the Epic's integration Task. The determinism group
    //!   below asserts the property that makes that race safe — byte-identical
    //!   envelopes — over two independent stores.
    //! - Tick wiring, cursor stamping, and the feature flag are Task 4's.
    //!
    //! Every "persists nothing" claim is a `put_layer_checkpoint` **call-count**
    //! assertion on the counting store, never an inspection of the listing
    //! alone: a put that was attempted and failed is a different bug from one
    //! that was never attempted, and only the count tells them apart.

    use super::*;
    use crate::checkpoint::adapters::in_memory::InMemoryCheckpointStore;
    use crate::checkpoint::assembler::HeaderIdentity;
    use crate::checkpoint::store::CheckpointBlob;
    use crate::checkpoint::tail_classify::tests::{
        FakeTailSliceLoader, READ_HEADER_IDENTITY, READ_TAIL_SLICE,
    };
    use crate::checkpoint::tail_codec::{
        decode_tail_envelope, TailRecord, TailRecordKind, TAIL_MAX_SEQUENCE,
    };
    use crate::observability::metrics::CHECKPOINT_TAIL_OUTCOMES;
    use crate::storage::notifications::LayerNotification;
    use crate::storage::tail_slice::{TailEntryPayload, TailSlice, TailSliceCap, TailStreamEntry};
    use async_trait::async_trait;
    use bytes::Bytes;
    use std::collections::HashSet;
    use std::sync::Mutex as StdMutex;

    // ── Constants ───────────────────────────────────────────────────────────

    const OBJECT_ID: u64 = 4242;
    const REPLICA: &str = "replica-a";
    /// Base of the synthetic `server_ts` series. Distinct per sequence so a
    /// record-to-entry mix-up cannot pass a timestamp assertion.
    const SERVER_TS_BASE: i64 = 1_700_000_000_000;

    // The four `obj:hdr` identity columns the fake header read serves. Distinct,
    // non-empty values: `apply`'s live-mapping identity resolution refuses
    // identity-empty blobs at restore, so a tail meta that silently inherited
    // `String::new()` would restore-fail months later rather than fail here.
    const SESSION_ID: &str = "sess-tail-capture";
    const LAYER_NAME: &str = "world/root.live";
    const CREATED_BY: &str = "user-42";
    const HEADER_SERVER_TS: u64 = 1_700_000_123_456;

    // ── Fixtures: config ────────────────────────────────────────────────────

    /// Stock knobs with an explicit chain bound K.
    ///
    /// `enabled` is left at its `Default` (`false`) throughout this module on
    /// purpose: the unit is flag-agnostic, and every capture case below
    /// succeeding under `enabled: false` is the proof.
    fn tail_cfg(max_chain: usize) -> TailCaptureConfig {
        TailCaptureConfig {
            max_chain,
            ..TailCaptureConfig::default()
        }
    }

    /// Stock chain bound with explicit cap knobs (group I).
    fn tail_cfg_caps(max_entries: u64, max_bytes: usize) -> TailCaptureConfig {
        TailCaptureConfig {
            max_entries,
            max_bytes,
            ..TailCaptureConfig::default()
        }
    }

    // ── Fixtures: header identity ───────────────────────────────────────────

    /// A header the fossil guard lets through: no outstanding restore bump and
    /// trustworthy cursor columns.
    ///
    /// `cursor_columns_trusted: true` is load-bearing and NOT the default —
    /// [`HeaderIdentity::default`] carries `false`, which the guard treats as a
    /// refusal, so a fixture built from the default would make every "captures"
    /// case below silently assert a refusal instead.
    fn trusted_identity() -> HeaderIdentity {
        HeaderIdentity {
            session_id: SESSION_ID.to_string(),
            layer_name: LAYER_NAME.to_string(),
            created_by: CREATED_BY.to_string(),
            server_ts: HEADER_SERVER_TS,
            last_checkpoint_seq: 0,
            restore_seq_bump: 0,
            cursor_columns_trusted: true,
        }
    }

    /// A header carrying an outstanding DR-restore stamp.
    fn fossil_identity(restore_seq_bump: u64) -> HeaderIdentity {
        HeaderIdentity {
            restore_seq_bump,
            ..trusted_identity()
        }
    }

    /// A header whose cursor columns were present but unparseable.
    fn untrusted_identity() -> HeaderIdentity {
        HeaderIdentity {
            cursor_columns_trusted: false,
            ..trusted_identity()
        }
    }

    // ── Fixtures: stream entries and slices ─────────────────────────────────

    fn server_ts_for(seq: u64) -> i64 {
        SERVER_TS_BASE + seq as i64
    }

    /// Distinct, self-identifying verbatim bytes per sequence, so a
    /// byte-identity assertion cannot pass by comparing the wrong record
    /// against the wrong entry.
    fn ompf_for(seq: u64) -> Vec<u8> {
        format!("ompf-payload-{seq}").into_bytes()
    }

    fn update_entry(seq: u64, relay_ompf: Option<Bytes>) -> TailStreamEntry {
        let payload_bytes = relay_ompf.as_ref().map_or(0, |b| b.len() as u64);
        TailStreamEntry {
            stream_id: format!("{seq}-0"),
            payload: TailEntryPayload::Notification(LayerNotification::Update {
                object_id: OBJECT_ID,
                sequence_num: seq,
                client_id: "capture-client".to_string(),
                client_ts: 0,
                server_ts: server_ts_for(seq),
                relay_ompf,
            }),
            payload_bytes,
        }
    }

    /// A well-formed commit at `{seq}-0` carrying the writer's verbatim bytes.
    fn verbatim_entry(seq: u64, ompf: &[u8]) -> TailStreamEntry {
        update_entry(seq, Some(Bytes::copy_from_slice(ompf)))
    }

    /// The §8.4 empty-commit shape: `update` at `{seq}-0`, no `ompf` field.
    fn empty_delta_entry(seq: u64) -> TailStreamEntry {
        update_entry(seq, None)
    }

    /// A slice the stream ran out of: no cap, no lookahead.
    fn slice_of(entries: Vec<TailStreamEntry>) -> TailSlice {
        let payload_bytes = entries.iter().map(|e| e.payload_bytes).sum();
        TailSlice {
            entries,
            lookahead: None,
            cap_hit: None,
            payload_bytes,
            pages_read: 1,
        }
    }

    /// A slice a cap truncated, with the one entry immediately past it.
    fn capped_slice_of(
        entries: Vec<TailStreamEntry>,
        cap: TailSliceCap,
        lookahead: TailStreamEntry,
    ) -> TailSlice {
        TailSlice {
            cap_hit: Some(cap),
            lookahead: Some(lookahead),
            ..slice_of(entries)
        }
    }

    /// Contiguous verbatim commits over `(from_seq, to_seq]`.
    fn verbatim_run(from_seq: u64, to_seq: u64) -> Vec<TailStreamEntry> {
        (from_seq + 1..=to_seq)
            .map(|seq| verbatim_entry(seq, &ompf_for(seq)))
            .collect()
    }

    /// The loader every PROCEED fixture wants: one canned slice, a header that
    /// clears the fossil guard.
    fn loader_with(slice: TailSlice) -> FakeTailSliceLoader {
        FakeTailSliceLoader::new(vec![Ok(slice)]).with_header_identity(trusted_identity())
    }

    // ── Fixtures: lineage seeders ───────────────────────────────────────────
    //
    // The meta shapes below are PINNED, and each seeder asserts the kind of the
    // handle it got back. A mis-shaped seed does not fail loudly on its own: a
    // `CheckpointMeta::default()` carries `kind: Base`, so a "tail" seeded
    // through the default would list as a base, move `position(newest Base)`,
    // and make the whole lineage-addressing and chain-bound groups vacuously
    // green. The assert is what turns that into a fixture failure.

    async fn seed_base(
        store: &dyn CheckpointStore,
        object_id: u64,
        snap_seq: u64,
    ) -> CheckpointHandle {
        let meta = CheckpointMeta {
            creator_replica_id: "seeder".to_string(),
            written_at_unix_ms: 1,
            snap_seq: Some(snap_seq),
            session_id: SESSION_ID.to_string(),
            layer_name: LAYER_NAME.to_string(),
            created_by: CREATED_BY.to_string(),
            server_ts: HEADER_SERVER_TS,
            kind: CheckpointKind::Base,
            // A base carries NO lineage fields (the trait's caller obligation).
            base_snap_seq: None,
            from_seq: None,
            sidecar_bytes: None,
            ..CheckpointMeta::default()
        };
        let handle = store
            .put_layer_checkpoint(
                object_id,
                snap_seq,
                Bytes::from_static(b"seeded-base-ompf"),
                meta,
            )
            .await
            .expect("seeding a base generation");
        assert_eq!(
            handle.kind,
            CheckpointKind::Base,
            "seed_base must produce a Base generation; a Tail here would silently move the \
             lineage addressing every case in this module depends on"
        );
        handle
    }

    /// Seed a tail generation carrying a REAL envelope over `(from_seq, to_seq]`
    /// — never junk bytes, so no case can come to depend on a seeded artifact
    /// being undecodable.
    async fn seed_tail(
        store: &dyn CheckpointStore,
        object_id: u64,
        base_snap_seq: u64,
        from_seq: u64,
        to_seq: u64,
    ) -> CheckpointHandle {
        let records: Vec<TailRecord> = (from_seq + 1..=to_seq)
            .map(|seq| TailRecord::Verbatim {
                sequence_num: seq,
                server_ts: server_ts_for(seq),
                ompf: Bytes::from(ompf_for(seq)),
            })
            .collect();
        let envelope = encode_tail_envelope(
            &TailHeader {
                object_id,
                base_snap_seq,
                from_seq,
                to_seq,
            },
            records,
        )
        .expect("seeded tail envelope encodes");
        let meta = CheckpointMeta {
            creator_replica_id: "seeder".to_string(),
            written_at_unix_ms: 1,
            snap_seq: Some(to_seq),
            session_id: SESSION_ID.to_string(),
            layer_name: LAYER_NAME.to_string(),
            created_by: CREATED_BY.to_string(),
            server_ts: HEADER_SERVER_TS,
            kind: CheckpointKind::Tail,
            // Both lineage fields `Some` — the adapter rejects a Tail meta
            // missing either as `Permanent`.
            base_snap_seq: Some(base_snap_seq),
            from_seq: Some(from_seq),
            sidecar_bytes: None,
            ..CheckpointMeta::default()
        };
        let handle = store
            .put_layer_checkpoint(object_id, to_seq, envelope, meta)
            .await
            .expect("seeding a tail generation");
        assert_eq!(
            handle.kind,
            CheckpointKind::Tail,
            "seed_tail must produce a Tail generation; a Base here would change \
             `position(newest Base)` and make the chain-bound group vacuous"
        );
        handle
    }

    // ── Fixtures: the counting store ────────────────────────────────────────

    /// [`CheckpointStore`] decorator over a real [`InMemoryCheckpointStore`],
    /// recording every trait-method invocation.
    ///
    /// Two claims in this module can only be made as call counts:
    /// "persists nothing" (zero `put_layer_checkpoint` calls) and the SR-4.2
    /// cost contract "addressing is listing-only" (zero `get_layer_checkpoint`
    /// calls, one `list_layer_checkpoints`).
    struct CountingStore {
        inner: InMemoryCheckpointStore,
        calls: StdMutex<Vec<&'static str>>,
        /// One-shot injected `put_layer_checkpoint` failure (group J).
        /// One-shot because [`CheckpointError`] is not `Clone`, and the unit
        /// performs at most one put per invocation anyway.
        put_error: StdMutex<Option<CheckpointError>>,
        /// One-shot injected `list_layer_checkpoints` failure (group J).
        /// One-shot for the same reason as `put_error`, and the unit performs
        /// exactly one listing per invocation — the lineage addressing.
        list_error: StdMutex<Option<CheckpointError>>,
    }

    impl CountingStore {
        fn new() -> Self {
            Self {
                inner: InMemoryCheckpointStore::new(),
                calls: StdMutex::new(Vec::new()),
                put_error: StdMutex::new(None),
                list_error: StdMutex::new(None),
            }
        }

        fn record(&self, method: &'static str) {
            self.calls
                .lock()
                .expect("calls mutex poisoned")
                .push(method);
        }

        /// Forget the seeders' own calls, so every count in a test describes
        /// the capture invocation alone.
        fn reset_counts(&self) {
            self.calls.lock().expect("calls mutex poisoned").clear();
        }

        fn count(&self, method: &str) -> usize {
            self.calls
                .lock()
                .expect("calls mutex poisoned")
                .iter()
                .filter(|m| **m == method)
                .count()
        }

        fn fail_next_put(&self, err: CheckpointError) {
            *self.put_error.lock().expect("put_error mutex poisoned") = Some(err);
        }

        fn fail_next_list(&self, err: CheckpointError) {
            *self.list_error.lock().expect("list_error mutex poisoned") = Some(err);
        }
    }

    #[async_trait]
    impl CheckpointStore for CountingStore {
        async fn put_layer_checkpoint(
            &self,
            object_id: u64,
            snap_seq: u64,
            ompf: Bytes,
            meta: CheckpointMeta,
        ) -> Result<CheckpointHandle, CheckpointError> {
            // Recorded BEFORE the injection check: an attempted-then-failed put
            // is a different fact from a put that never happened, and the count
            // must keep them distinguishable.
            self.record("put_layer_checkpoint");
            let injected = self
                .put_error
                .lock()
                .expect("put_error mutex poisoned")
                .take();
            if let Some(err) = injected {
                return Err(err);
            }
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

        /// Forwarded, not inherited: the trait default answers
        /// `NotImplemented`, which would turn this module's artifact
        /// assertions into errors instead of reads — and would make the
        /// zero-blob-read count meaningless.
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
            // Recorded BEFORE the injection check, for the same reason the put
            // is: an attempted-then-failed listing and a listing that never
            // happened are different facts.
            self.record("list_layer_checkpoints");
            let injected = self
                .list_error
                .lock()
                .expect("list_error mutex poisoned")
                .take();
            if let Some(err) = injected {
                return Err(err);
            }
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

    // ── Assertion helpers ───────────────────────────────────────────────────

    /// Destructure a `Captured` decision, or fail naming the refusal.
    #[track_caller]
    fn captured_parts(
        decision: &TailCaptureDecision,
    ) -> (&CheckpointHandle, u64, u64, usize, bool) {
        match decision {
            TailCaptureDecision::Captured {
                handle,
                from_seq,
                to_seq,
                record_count,
                capped,
            } => (handle, *from_seq, *to_seq, *record_count, *capped),
            TailCaptureDecision::Refused(refusal) => {
                panic!("expected a capture, got Refused({refusal:?})")
            }
        }
    }

    /// Assert the decision is exactly this refusal — the specific variant, not
    /// merely "not a capture": each arm has a different operator remediation.
    #[track_caller]
    fn assert_refused(decision: &TailCaptureDecision, expected: TailCaptureRefusal) {
        match decision {
            TailCaptureDecision::Refused(refusal) => {
                assert_eq!(*refusal, expected, "wrong refusal arm")
            }
            TailCaptureDecision::Captured { to_seq, .. } => {
                panic!("expected Refused({expected:?}), got a capture up to {to_seq}")
            }
        }
    }

    #[track_caller]
    fn assert_tail_handle(handle: &CheckpointHandle, object_id: u64, to_seq: u64) {
        assert_eq!(handle.object_id, object_id, "handle names the wrong object");
        assert_eq!(
            handle.snap_seq, to_seq,
            "a tail's ordering sequence IS its `to_seq` (CheckpointHandle::snap_seq)"
        );
        assert_eq!(
            handle.kind,
            CheckpointKind::Tail,
            "capture must address the generation as a Tail; a Base handle here would collide \
             with the snapshot path's addressing"
        );
    }

    /// Fetch a persisted artifact by handle, failing loudly on absence.
    async fn fetch(store: &dyn CheckpointStore, handle: &CheckpointHandle) -> CheckpointBlob {
        store
            .get_layer_checkpoint(handle)
            .await
            .expect("fetching the persisted artifact")
            .expect("the capture's handle must resolve to a stored artifact")
    }

    #[track_caller]
    fn assert_verbatim(record: &TailRecord, seq: u64, ompf: &[u8]) {
        assert_eq!(record.kind(), TailRecordKind::Verbatim);
        assert_eq!(record.sequence_num(), seq);
        assert_eq!(record.server_ts(), server_ts_for(seq));
        assert_eq!(
            record.ompf().map(Bytes::as_ref),
            Some(ompf),
            "verbatim bytes must be byte-identical to the stream entry's `ompf` field — SR-2.2 \
             stores the writer's raw inbound container with no re-encode at the capture boundary"
        );
    }

    #[track_caller]
    fn assert_empty_delta(record: &TailRecord, seq: u64) {
        assert_eq!(record.kind(), TailRecordKind::EmptyDelta);
        assert_eq!(record.sequence_num(), seq);
        assert_eq!(record.server_ts(), server_ts_for(seq));
        assert!(
            record.ompf().is_none(),
            "an empty-delta record carries no payload"
        );
    }

    // ── Group V: outcome-label vocabulary membership ────────────────────────

    /// Every label this unit can mint is a member of the vocabulary Task 1
    /// pinned, including the two capture spellings and the post-restore fold.
    ///
    /// The const is referenced, never re-spelled as literals: a literal would
    /// drift from the very vocabulary it is supposed to be checked against. A
    /// label outside it renders as a metric series the dashboards never seed.
    #[test]
    fn every_outcome_label_is_a_member_of_the_pinned_metric_vocabulary() {
        let handle = CheckpointHandle::tail(OBJECT_ID, 12, "key".to_string());
        let decisions = vec![
            TailCaptureDecision::Captured {
                handle: handle.clone(),
                from_seq: 10,
                to_seq: 12,
                record_count: 2,
                capped: false,
            },
            TailCaptureDecision::Captured {
                handle,
                from_seq: 10,
                to_seq: 12,
                record_count: 2,
                capped: true,
            },
            TailCaptureDecision::Refused(TailCaptureRefusal::NoBase),
            TailCaptureDecision::Refused(TailCaptureRefusal::LineageUnrootable),
            TailCaptureDecision::Refused(TailCaptureRefusal::ChainFull {
                chain_len: 8,
                max_chain: 8,
            }),
            TailCaptureDecision::Refused(TailCaptureRefusal::PostRestoreUnrebased {
                restore_seq_bump: 7,
            }),
            TailCaptureDecision::Refused(TailCaptureRefusal::Classified(TailRefusal::Trimmed)),
            TailCaptureDecision::Refused(TailCaptureRefusal::Classified(TailRefusal::MissingBytes)),
            TailCaptureDecision::Refused(TailCaptureRefusal::Classified(TailRefusal::Sentinel)),
            TailCaptureDecision::Refused(TailCaptureRefusal::Classified(TailRefusal::SeqGap)),
        ];

        for decision in &decisions {
            let label = decision.outcome_label();
            assert!(
                CHECKPOINT_TAIL_OUTCOMES.contains(&label),
                "{decision:?} labels itself `{label}`, which is not a member of \
                 CHECKPOINT_TAIL_OUTCOMES — the metric cell would never be seeded"
            );
        }

        // Anti-vacuity: the constructed set must actually exercise the breadth
        // of the vocabulary, not label everything the same way.
        let distinct: HashSet<&str> = decisions.iter().map(|d| d.outcome_label()).collect();
        assert!(
            distinct.len() >= 8,
            "expected at least 8 distinct labels across the constructed decisions, got {}: {:?}",
            distinct.len(),
            distinct
        );

        // The pinned fold, asserted as an identity rather than inferred: the
        // post-restore guard reports itself with the classifier's `SeqGap`
        // spelling. It is the same operator-facing condition caught one read
        // earlier, and a tenth label would fork the dashboard vocabulary.
        assert_eq!(
            TailCaptureDecision::Refused(TailCaptureRefusal::PostRestoreUnrebased {
                restore_seq_bump: 7
            })
            .outcome_label(),
            TailRefusal::SeqGap.outcome_label()
        );
    }

    // ── Group A: lineage addressing (S0) ────────────────────────────────────

    /// A base-only lineage roots the first tail at the base's sequence.
    #[tokio::test]
    async fn capture_first_tail_against_base_only_lineage_roots_from_seq_at_the_base() {
        let store = CountingStore::new();
        seed_base(&store, OBJECT_ID, 10).await;
        store.reset_counts();
        let loader = loader_with(slice_of(verbatim_run(10, 13)));

        let decision = capture_layer_tail_with(&loader, &store, OBJECT_ID, REPLICA, &tail_cfg(8))
            .await
            .expect("capture succeeds");
        let puts = store.count("put_layer_checkpoint");

        let (handle, from_seq, to_seq, record_count, capped) = captured_parts(&decision);
        assert_eq!(decision.outcome_label(), "captured");
        assert_eq!((from_seq, to_seq, record_count, capped), (10, 13, 3, false));
        assert_tail_handle(handle, OBJECT_ID, 13);
        assert_eq!(puts, 1, "exactly one artifact is persisted per capture");

        // Landed step order: the header read (fossil guard) precedes the slice
        // read, and nothing reads twice.
        assert_eq!(
            loader.call_log(),
            vec![READ_HEADER_IDENTITY, READ_TAIL_SLICE]
        );

        let blob = fetch(&store, handle).await;
        let envelope = decode_tail_envelope(&blob.bytes).expect("persisted artifact decodes");
        assert_eq!(
            *envelope.header(),
            TailHeader {
                object_id: OBJECT_ID,
                base_snap_seq: 10,
                from_seq: 10,
                to_seq: 13,
            }
        );
        assert_eq!(envelope.records().len(), 3);
    }

    /// **The discriminating case.** Once a tail exists, S0 is the newest
    /// generation — the lineage HEAD — not the base.
    ///
    /// The seam's parameter is spelled `base_snap_seq` but MEANS S0, which is
    /// exactly the trap: rooting the second tail at the base would re-read
    /// commits the first tail already covers, minting overlapping generations
    /// and breaking SR-2.1's `tail_{n+1}.from_seq == tail_n.to_seq` contiguity.
    /// Three assertions, because a fix could be right in one place and wrong in
    /// another: the argument the loader was driven with, the persisted meta's
    /// `from_seq` / `base_snap_seq` split, and the encoded header's same split.
    #[tokio::test]
    async fn capture_second_tail_addresses_s0_at_the_newest_generation_not_the_lineage_base() {
        let store = CountingStore::new();
        seed_base(&store, OBJECT_ID, 10).await;
        seed_tail(&store, OBJECT_ID, 10, 10, 20).await;
        store.reset_counts();
        let loader = loader_with(slice_of(verbatim_run(20, 22)));

        let decision = capture_layer_tail_with(&loader, &store, OBJECT_ID, REPLICA, &tail_cfg(8))
            .await
            .expect("capture succeeds");

        // (a) the read window opened at `{21}-0`, not `{11}-0`.
        let calls = loader.calls();
        assert_eq!(calls.len(), 1, "exactly one bounded slice read per capture");
        assert_eq!(
            calls[0].1, 20,
            "S0 must be the lineage HEAD (the tail at 20), not the base at 10"
        );
        assert_eq!(calls[0].0, OBJECT_ID);

        let (handle, from_seq, to_seq, ..) = captured_parts(&decision);
        assert_eq!((from_seq, to_seq), (20, 22));

        // (b) the persisted meta carries the two DIFFERENT numbers.
        let blob = fetch(&store, handle).await;
        assert_eq!(blob.meta.from_seq, Some(20), "meta.from_seq is S0");
        assert_eq!(
            blob.meta.base_snap_seq,
            Some(10),
            "meta.base_snap_seq stays the lineage ROOT, which is not S0 once a tail exists"
        );

        // (c) the envelope header carries the same split.
        let envelope = decode_tail_envelope(&blob.bytes).expect("persisted artifact decodes");
        assert_eq!(envelope.header().from_seq, 20);
        assert_eq!(envelope.header().base_snap_seq, 10);
        assert_eq!(envelope.header().to_seq, 22);
    }

    /// An empty listing refuses before any read — the layer was never
    /// checkpointed, or retention pruned its last generation.
    ///
    /// **The sole `refused_no_base` producer.** The other two arms that used to
    /// share that cell — a tails-only listing and a head at sequence 0 — now
    /// report `refused_lineage_unrootable` (the two tests below), which is what
    /// leaves this cell meaning exactly "this object has no durable generation
    /// at all".
    #[tokio::test]
    async fn capture_empty_listing_refuses_no_base_and_persists_nothing() {
        let store = CountingStore::new();
        let loader = loader_with(slice_of(verbatim_run(10, 12)));

        let decision = capture_layer_tail_with(&loader, &store, OBJECT_ID, REPLICA, &tail_cfg(8))
            .await
            .expect("a refusal is not an error");

        assert_refused(&decision, TailCaptureRefusal::NoBase);
        assert_eq!(decision.outcome_label(), "refused_no_base");
        assert_eq!(store.count("put_layer_checkpoint"), 0);
        // Ordering evidence: the refusal precedes BOTH reads it fences, so a
        // layer that cannot produce a tail costs one listing and nothing else.
        assert!(
            loader.call_log().is_empty(),
            "no-base must refuse before the header read and the slice read"
        );
    }

    /// A tails-only listing has no root to layer onto. Not producible by the
    /// shipped write paths (a tail is only ever written after the base it
    /// names), so this mirrors `recovery::select_restore_source`'s no-base arm.
    ///
    /// **It reports `refused_lineage_unrootable`, not `refused_no_base`.** The
    /// listing is *not* empty — generations exist, none of them can root a
    /// lineage — and the remediation is a human repairing the checkpoint store,
    /// which is nothing like the empty-listing case's "wait for the first full
    /// backup". Once lineage-aware retention is live this shape is also the
    /// PRD AC-7 fingerprint (a base deleted while a retained tail references
    /// it), so the cell must not be re-folded onto the rollout tracker's.
    #[tokio::test]
    async fn capture_listing_without_any_base_generation_refuses_lineage_unrootable() {
        let store = CountingStore::new();
        seed_tail(&store, OBJECT_ID, 10, 10, 20).await;
        store.reset_counts();
        let loader = loader_with(slice_of(verbatim_run(20, 22)));

        let decision = capture_layer_tail_with(&loader, &store, OBJECT_ID, REPLICA, &tail_cfg(8))
            .await
            .expect("a refusal is not an error");

        assert_refused(&decision, TailCaptureRefusal::LineageUnrootable);
        assert_eq!(decision.outcome_label(), "refused_lineage_unrootable");
        assert_eq!(store.count("put_layer_checkpoint"), 0);
        assert!(loader.call_log().is_empty());
    }

    /// A lineage head at sequence 0 refuses: the read window would open at
    /// `1-0`, which is `CreateLayer`'s `update`-typed, deliberately ompf-less
    /// seed entry — and that entry ACCOMPANIES the full initial layer content,
    /// so the classifier would record it as an empty commit and replay it as a
    /// no-op over real state. SR-7.1 roots every lineage at `snap_seq = 1`, so
    /// `S0 >= 1` is a production guard, not a formality.
    ///
    /// **It reports `refused_lineage_unrootable`**, the second of that cell's
    /// two arms: a base exists here, so the listing is not empty — it is the
    /// *sequence* that no shipped write path can produce, and the remediation
    /// is the same checkpoint-store repair the tails-only arm needs.
    #[tokio::test]
    async fn capture_with_a_zero_sequence_lineage_head_refuses_lineage_unrootable() {
        let store = CountingStore::new();
        seed_base(&store, OBJECT_ID, 0).await;
        store.reset_counts();
        let loader = loader_with(slice_of(verbatim_run(0, 3)));

        let decision = capture_layer_tail_with(&loader, &store, OBJECT_ID, REPLICA, &tail_cfg(8))
            .await
            .expect("a refusal is not an error");

        assert_refused(&decision, TailCaptureRefusal::LineageUnrootable);
        assert_eq!(decision.outcome_label(), "refused_lineage_unrootable");
        assert_eq!(store.count("put_layer_checkpoint"), 0);
        assert!(
            loader.call_log().is_empty(),
            "the s0 == 0 guard must fire before any read reaches the seed entry"
        );
    }

    // ── Group B: the chain bound ────────────────────────────────────────────
    //
    // These drive the UNIT, which still returns `ChainFull` at K — that return
    // is the SR-4.1 re-base trigger, not the tick's verdict. The loop-level
    // behaviour (collapse, verified re-run, or a real `refused_chain_full`)
    // belongs to `checkpointer`'s lane and is asserted there.

    /// K tails above the base refuses `chain_full` and persists nothing.
    #[tokio::test]
    async fn capture_with_chain_at_k_refuses_chain_full_and_persists_nothing() {
        let store = CountingStore::new();
        seed_base(&store, OBJECT_ID, 10).await;
        seed_tail(&store, OBJECT_ID, 10, 10, 20).await;
        seed_tail(&store, OBJECT_ID, 10, 20, 30).await;
        store.reset_counts();
        let loader = loader_with(slice_of(verbatim_run(30, 32)));

        let decision = capture_layer_tail_with(&loader, &store, OBJECT_ID, REPLICA, &tail_cfg(2))
            .await
            .expect("a refusal is not an error");

        assert_refused(
            &decision,
            TailCaptureRefusal::ChainFull {
                chain_len: 2,
                max_chain: 2,
            },
        );
        assert_eq!(decision.outcome_label(), "refused_chain_full");
        assert_eq!(store.count("put_layer_checkpoint"), 0);
        assert!(
            loader.call_log().is_empty(),
            "chain-full must refuse before the header read and the slice read"
        );
    }

    /// Pins the LANDED conservative count: an overlapping non-member tail —
    /// the leftover of SR-9.2's divergent-S duplicate-leader fork — still
    /// counts toward the bound, because `chain_len` is `position(newest Base)`
    /// in the listing, a raw count of generations above the base rather than
    /// the `from`/`to` link-walk the spec describes.
    ///
    /// The deviation is deliberate and errs SAFE, per the unit's four-part
    /// argument: no meta-only read exists on `CheckpointStore`, so a true walk
    /// would fetch up to K multi-MB blobs per eligible tick; SR-4.2 budgets the
    /// tick as one listing, one bounded XRANGE and one small PUT; every true
    /// chain member lies above the newest base, so the raw count is `>=` the
    /// true walk length and can only refuse EARLIER than K, never admit an
    /// over-long chain; and over-counting arises only from a rare, self-healing
    /// fork whose worst consequence is one early chain-full tick.
    #[tokio::test]
    async fn capture_counts_overlapping_tails_above_the_newest_base_toward_the_bound() {
        let store = CountingStore::new();
        seed_base(&store, OBJECT_ID, 10).await;
        seed_tail(&store, OBJECT_ID, 10, 10, 20).await;
        // The divergent-S leftover: covers (10, 15], overlapping the tail
        // above. NOT a member of the walked chain ending at the newest tail.
        seed_tail(&store, OBJECT_ID, 10, 10, 15).await;
        store.reset_counts();
        let loader = loader_with(slice_of(verbatim_run(20, 22)));

        let decision = capture_layer_tail_with(&loader, &store, OBJECT_ID, REPLICA, &tail_cfg(2))
            .await
            .expect("a refusal is not an error");

        assert_refused(
            &decision,
            TailCaptureRefusal::ChainFull {
                chain_len: 2,
                max_chain: 2,
            },
        );
        assert_eq!(store.count("put_layer_checkpoint"), 0);
    }

    /// Anti-vacuity twin of the case above: the SAME overlapping-tail fixture
    /// at K = 3 captures, so the refusal there is the bound firing and not the
    /// fixture being unusable for some other reason.
    #[tokio::test]
    async fn capture_with_an_overlapping_tail_below_the_bound_still_captures() {
        let store = CountingStore::new();
        seed_base(&store, OBJECT_ID, 10).await;
        seed_tail(&store, OBJECT_ID, 10, 10, 20).await;
        seed_tail(&store, OBJECT_ID, 10, 10, 15).await;
        store.reset_counts();
        let loader = loader_with(slice_of(verbatim_run(20, 22)));

        let decision = capture_layer_tail_with(&loader, &store, OBJECT_ID, REPLICA, &tail_cfg(3))
            .await
            .expect("capture succeeds");

        let (handle, from_seq, to_seq, ..) = captured_parts(&decision);
        assert_eq!(decision.outcome_label(), "captured");
        assert_eq!(
            (from_seq, to_seq),
            (20, 22),
            "S0 is the listing head (20), unaffected by the overlapping leftover at 15"
        );
        assert_tail_handle(handle, OBJECT_ID, 22);
    }

    /// The SR-4.2 cost contract: lineage addressing is LISTING-ONLY. One
    /// `list_layer_checkpoints`, zero `get_layer_checkpoint` — no blob is ever
    /// fetched to answer "where does this lineage start and how long is it".
    /// Asserted on a success run AND on a chain-full run, since a walk added to
    /// either path would blow the same budget.
    #[tokio::test]
    async fn capture_addressing_makes_zero_blob_reads_and_one_listing() {
        // Success run.
        let store = CountingStore::new();
        seed_base(&store, OBJECT_ID, 10).await;
        seed_tail(&store, OBJECT_ID, 10, 10, 20).await;
        store.reset_counts();
        let loader = loader_with(slice_of(verbatim_run(20, 22)));

        capture_layer_tail_with(&loader, &store, OBJECT_ID, REPLICA, &tail_cfg(8))
            .await
            .expect("capture succeeds");

        assert_eq!(
            store.count("get_layer_checkpoint"),
            0,
            "addressing must not fetch a blob"
        );
        assert_eq!(store.count("list_layer_checkpoints"), 1);
        assert_eq!(store.count("get_latest_layer_checkpoint"), 0);

        // Chain-full run.
        let store = CountingStore::new();
        seed_base(&store, OBJECT_ID, 10).await;
        seed_tail(&store, OBJECT_ID, 10, 10, 20).await;
        store.reset_counts();
        let loader = loader_with(slice_of(verbatim_run(20, 22)));

        let decision = capture_layer_tail_with(&loader, &store, OBJECT_ID, REPLICA, &tail_cfg(1))
            .await
            .expect("a refusal is not an error");

        assert_eq!(decision.outcome_label(), "refused_chain_full");
        assert_eq!(store.count("get_layer_checkpoint"), 0);
        assert_eq!(store.count("list_layer_checkpoints"), 1);
    }

    // ── Group B2: the bound-carrying entry point lists for itself ───────────
    //
    // `capture_layer_tail_with_bound` — the checkpoint loop's entry point —
    // takes no listing from its caller: it lists once, at the moment it
    // addresses the tail, so a generation that landed earlier in the tick (a
    // creation-time or post-restore bootstrap base, written without the leader
    // lock while the assembler's fenced read ran) is addressed from, never
    // missed. The first two rows pin that; the last is a Group B case wearing
    // this group's call shape: the entry point is the only one carrying a
    // `ChainBound`, so the `Verified` side of the bound is asserted here.

    /// The helper's two `None` cases are the two the caller must keep apart —
    /// pinned here because `lineage_shape` deliberately folds them and
    /// `capture_inner`'s `handles.is_empty()` check is what un-folds them into
    /// a `debug!` (routine) and a `warn!` (suspected store loss).
    #[test]
    fn lineage_shape_is_none_for_an_empty_listing_and_for_a_tails_only_one() {
        assert_eq!(lineage_shape(&[]), None);
        let tails_only = vec![
            CheckpointHandle::tail(OBJECT_ID, 20, "k20".to_string()),
            CheckpointHandle::tail(OBJECT_ID, 15, "k15".to_string()),
        ];
        assert_eq!(lineage_shape(&tails_only), None);

        // Anti-vacuity: the same listing with a base under it derives a shape,
        // so the `None`s above are the base test and not an arity accident.
        let mut with_base = tails_only;
        with_base.push(CheckpointHandle::base(OBJECT_ID, 10, "k10".to_string()));
        assert_eq!(
            lineage_shape(&with_base),
            Some(LineageShape {
                head_kind: CheckpointKind::Tail,
                head_seq: 20,
                base_snap_seq: 10,
                chain_len: 2,
            })
        );
    }

    /// The cost claim: the bound-carrying entry point takes exactly ONE
    /// `list_layer_checkpoints` of its own — there is no caller-supplied listing
    /// to save it — and still zero blob reads, so addressing stays inside the
    /// SR-4.2 per-capture budget (one listing, one bounded XRANGE, one PUT).
    #[tokio::test]
    async fn capture_with_bound_takes_one_listing_of_its_own_and_no_blob_read() {
        let store = CountingStore::new();
        seed_base(&store, OBJECT_ID, 10).await;
        seed_tail(&store, OBJECT_ID, 10, 10, 20).await;
        store.reset_counts();
        let loader = loader_with(slice_of(verbatim_run(20, 22)));

        let decision = capture_layer_tail_with_bound(
            &loader,
            &store,
            OBJECT_ID,
            REPLICA,
            &tail_cfg(8),
            ChainBound::FromListing,
            // No Valkey in the fake-loader lane, so no liveness probe —
            // the pre-existing unguarded-PUT behaviour.
            None,
        )
        .await
        .expect("capture succeeds");

        let (handle, from_seq, to_seq, ..) = captured_parts(&decision);
        assert_eq!(decision.outcome_label(), "captured");
        assert_eq!(
            (from_seq, to_seq),
            (20, 22),
            "S0 is addressed at the lineage HEAD"
        );
        assert_tail_handle(handle, OBJECT_ID, 22);
        assert_eq!(
            store.count("list_layer_checkpoints"),
            1,
            "the unit lists once, for itself: no caller hands it one, and a second listing \
             would be a cost the per-capture budget does not carry"
        );
        assert_eq!(
            store.count("get_layer_checkpoint"),
            0,
            "addressing must not fetch a blob on this path either"
        );
        assert_eq!(store.count("put_layer_checkpoint"), 1);
    }

    /// The listing is taken at the moment the tail is addressed, so a base that
    /// landed earlier in the tick is the S0 the capture addresses from.
    ///
    /// The case the removed same-tick listing hand-off would have got wrong: the
    /// tick's assembler runs a proven read before this call — for a layer above
    /// the atomic-read limits a fenced read, its fence wait and hold — and in
    /// that window a base can land without the leader lock (the creation-time
    /// or post-restore bootstrap). A listing taken before the read names the
    /// older tail as the head; capturing from it would re-cover `(20, 25]`,
    /// which the new base already holds, and root the tail on a superseded
    /// lineage. Here the base at 25 is written after the older lineage, as the
    /// bootstrap would, and the capture must address from it.
    #[tokio::test]
    async fn capture_with_bound_addresses_from_a_base_that_landed_before_the_call() {
        let store = CountingStore::new();
        seed_base(&store, OBJECT_ID, 10).await;
        seed_tail(&store, OBJECT_ID, 10, 10, 20).await;
        // The bootstrap base that landed while the tick's read ran.
        seed_base(&store, OBJECT_ID, 25).await;
        store.reset_counts();
        let loader = loader_with(slice_of(verbatim_run(25, 27)));

        let decision = capture_layer_tail_with_bound(
            &loader,
            &store,
            OBJECT_ID,
            REPLICA,
            &tail_cfg(8),
            ChainBound::FromListing,
            None,
        )
        .await
        .expect("capture succeeds");

        let (handle, from_seq, to_seq, ..) = captured_parts(&decision);
        assert_eq!(
            (from_seq, to_seq),
            (25, 27),
            "S0 must be the newest base the store holds when the capture addresses, not the \
             older tail head a listing taken earlier in the tick would name"
        );
        assert_tail_handle(handle, OBJECT_ID, 27);
        let blob = fetch(&store, handle).await;
        assert_eq!(
            blob.meta.base_snap_seq,
            Some(25),
            "the tail is rooted on the base it was addressed from"
        );
    }

    /// A `ChainBound::Verified(n)` with `n >= max_chain` still refuses
    /// `chain_full`, and the refusal reports the **listing** count, not `n`.
    ///
    /// The mirror image of the divergent-S fork the bound exists for: there a
    /// verified `n` BELOW K admits a layer the listing over-counted up to K;
    /// here a verified `n` at or above K refuses one the listing alone would
    /// have admitted. `ChainBound::measured` is a substitution at one comparison
    /// site, not a per-direction escape hatch, so both directions have to hold.
    /// It rides `capture_layer_tail_with_bound` because that is the only form
    /// carrying a bound — which also lets the cost contract be re-asserted here.
    ///
    /// # Why these three numbers
    ///
    /// The fixture is Group B's overlapping-tail one — base at 10, tails
    /// `(10, 20]` and `(10, 15]` — whose LISTING count is 2 while its true walk
    /// is 1. Run at `max_chain = 3` under `Verified(3)`:
    ///
    /// - `3 >= 3` fires the bound and `2 >= 3` does not, so a refusal can only
    ///   be the verified number reaching the comparison. Its other half is the
    ///   sibling `capture_with_an_overlapping_tail_below_the_bound_still_captures`,
    ///   which CAPTURES on this exact fixture at this exact K under
    ///   `FromListing` — so the refusal is the bound source moving, not the
    ///   fixture being unusable.
    /// - `n = 3` and the listing count 2 are DISTINCT, so the reported
    ///   `chain_len` names one of them unambiguously. A fixture whose listing
    ///   already read K (`n == chain_len`) would make the second assertion
    ///   vacuous: it would pass just as well with the verified value leaking
    ///   into the refusal.
    ///
    /// That second assertion is the load-bearing one. It pins the enum's
    /// "reaches exactly one comparison site" contract, so an edit that let `n`
    /// into the operator-visible number — the one an operator's own `list` has
    /// to reproduce — goes red here. A `ChainFull` whose `chain_len` reads BELOW
    /// its `max_chain` is the intended shape on this arm, not a contradiction:
    /// the compared number travels alongside in the log's `bound_chain_len`.
    ///
    /// The arm is defensive today — the loop mints `Verified` only on the
    /// `UnderBound` fork (`n < max_chain`) — but `ChainBound`'s accepted
    /// one-tail staleness residual keeps it conceptually live, and the contract
    /// is asserted by its rustdoc either way.
    #[tokio::test]
    async fn a_verified_bound_at_or_above_k_refuses_chain_full_reporting_the_listing_count() {
        const MAX_CHAIN: usize = 3;
        // Above K, and deliberately != the listing count below.
        const VERIFIED_CHAIN_LEN: usize = 3;
        const LISTING_CHAIN_LEN: usize = 2;

        let store = CountingStore::new();
        seed_base(&store, OBJECT_ID, 10).await;
        seed_tail(&store, OBJECT_ID, 10, 10, 20).await;
        // The divergent-S leftover: overlaps the tail above, so the listing
        // counts two generations where the true walk reads one.
        seed_tail(&store, OBJECT_ID, 10, 10, 15).await;
        let listing = store
            .list_layer_checkpoints(OBJECT_ID)
            .await
            .expect("the fixture's own listing");
        let shape = lineage_shape(&listing).expect("a base is present");
        assert_eq!(
            shape.chain_len, LISTING_CHAIN_LEN,
            "the two numbers under test must differ or the chain_len assertion below is vacuous: \
             listing count vs verified {VERIFIED_CHAIN_LEN}"
        );
        store.reset_counts();
        let loader = loader_with(slice_of(verbatim_run(20, 22)));

        let decision = capture_layer_tail_with_bound(
            &loader,
            &store,
            OBJECT_ID,
            REPLICA,
            &tail_cfg(MAX_CHAIN),
            ChainBound::Verified(VERIFIED_CHAIN_LEN),
            // No Valkey in the fake-loader lane, so no liveness probe —
            // the pre-existing unguarded-PUT behaviour.
            None,
        )
        .await
        .expect("a refusal is not an error");

        // Half two first, so a leak reports itself by name rather than as a
        // whole-arm mismatch.
        match &decision {
            TailCaptureDecision::Refused(TailCaptureRefusal::ChainFull { chain_len, .. }) => {
                assert_eq!(
                    *chain_len, LISTING_CHAIN_LEN,
                    "the refusal must report the LISTING count; {VERIFIED_CHAIN_LEN} here means \
                     the verified bound leaked past its one comparison site into the \
                     operator-visible number"
                );
            }
            other => panic!("expected a chain_full refusal, got {other:?}"),
        }
        // Half one, and the whole arm: `max_chain` travels with it.
        assert_refused(
            &decision,
            TailCaptureRefusal::ChainFull {
                chain_len: LISTING_CHAIN_LEN,
                max_chain: MAX_CHAIN,
            },
        );
        assert_eq!(decision.outcome_label(), "refused_chain_full");

        // The cost contract the FromListing rows above pin, restated on the
        // verified bound: one listing of the unit's own, no blob fetched to
        // answer the bound, and a refusal persists nothing.
        assert_eq!(
            store.count("list_layer_checkpoints"),
            1,
            "a verified bound must cost no listing beyond the one the unit addresses from"
        );
        assert_eq!(store.count("get_layer_checkpoint"), 0);
        assert_eq!(store.count("put_layer_checkpoint"), 0);
        assert!(
            loader.call_log().is_empty(),
            "chain-full must refuse before the header read and the slice read on this bound too"
        );
    }

    // ── Group C: envelope content ───────────────────────────────────────────

    /// Both record shapes survive the capture boundary in stream order, and a
    /// verbatim payload is byte-identical to the entry's `ompf` field —
    /// SR-2.2's no-re-encode claim, asserted on the persisted artifact rather
    /// than on the in-memory classification.
    #[tokio::test]
    async fn capture_encodes_verbatim_and_empty_delta_records_in_stream_order() {
        let store = CountingStore::new();
        seed_base(&store, OBJECT_ID, 10).await;
        store.reset_counts();
        let entries = vec![
            verbatim_entry(11, &ompf_for(11)),
            empty_delta_entry(12),
            verbatim_entry(13, &ompf_for(13)),
        ];
        let loader = loader_with(slice_of(entries));

        let decision = capture_layer_tail_with(&loader, &store, OBJECT_ID, REPLICA, &tail_cfg(8))
            .await
            .expect("capture succeeds");

        let (handle, .., record_count, _) = captured_parts(&decision);
        assert_eq!(record_count, 3);
        let blob = fetch(&store, handle).await;
        let envelope = decode_tail_envelope(&blob.bytes).expect("persisted artifact decodes");
        let records = envelope.records();
        assert_eq!(records.len(), 3);
        assert_verbatim(&records[0], 11, &ompf_for(11));
        assert_empty_delta(&records[1], 12);
        assert_verbatim(&records[2], 13, &ompf_for(13));
    }

    /// §22.19 / the b.pw6 rule: the claimed range is exactly filled by content.
    /// The header's `from_seq` is EXCLUSIVE, so the record count equals
    /// `to_seq - from_seq`.
    #[tokio::test]
    async fn capture_claimed_range_equals_persisted_record_content() {
        let store = CountingStore::new();
        seed_base(&store, OBJECT_ID, 10).await;
        store.reset_counts();
        let loader = loader_with(slice_of(verbatim_run(10, 18)));

        let decision = capture_layer_tail_with(&loader, &store, OBJECT_ID, REPLICA, &tail_cfg(8))
            .await
            .expect("capture succeeds");

        let (handle, ..) = captured_parts(&decision);
        let blob = fetch(&store, handle).await;
        let envelope = decode_tail_envelope(&blob.bytes).expect("persisted artifact decodes");
        let header = *envelope.header();
        assert_eq!(
            envelope.records().len() as u64,
            header.to_seq - header.from_seq,
            "claimed range ({}, {}] is not filled by {} records",
            header.from_seq,
            header.to_seq,
            envelope.records().len()
        );
        for (index, record) in envelope.records().iter().enumerate() {
            assert_eq!(record.sequence_num(), header.from_seq + 1 + index as u64);
        }
    }

    // ── Group D: cap ────────────────────────────────────────────────────────

    /// A cap-bounded prefix is still persisted, reports `captured_capped`, and
    /// claims only what it carries: `to_seq` is the last INCLUDED entry, never
    /// the lookahead's.
    #[tokio::test]
    async fn capture_of_a_cap_bounded_prefix_persists_and_reports_captured_capped() {
        let store = CountingStore::new();
        seed_base(&store, OBJECT_ID, 10).await;
        store.reset_counts();
        // Benign lookahead: the next entry sits exactly at `expected` (13), so
        // it is a cap stop and not the boundary-fossil ascending jump.
        let slice = capped_slice_of(
            verbatim_run(10, 12),
            TailSliceCap::Bytes,
            verbatim_entry(13, &ompf_for(13)),
        );
        let loader = loader_with(slice);

        let decision = capture_layer_tail_with(&loader, &store, OBJECT_ID, REPLICA, &tail_cfg(8))
            .await
            .expect("capture succeeds");

        let (handle, from_seq, to_seq, record_count, capped) = captured_parts(&decision);
        assert_eq!(decision.outcome_label(), "captured_capped");
        assert!(capped);
        assert_eq!((from_seq, to_seq, record_count), (10, 12, 2));
        assert_tail_handle(handle, OBJECT_ID, 12);
        assert_eq!(store.count("put_layer_checkpoint"), 1);

        let blob = fetch(&store, handle).await;
        let envelope = decode_tail_envelope(&blob.bytes).expect("persisted artifact decodes");
        assert_eq!(
            envelope.header().to_seq,
            12,
            "the lookahead entry (13) is counted against neither cap and must never be claimed"
        );
        assert_eq!(envelope.records().len(), 2);
    }

    /// Anti-vacuity twin: the same entries without a cap stop report
    /// `captured`, so the case above is pinning the cap flag rather than some
    /// property of the fixture's shape.
    #[tokio::test]
    async fn capture_of_the_same_prefix_without_a_cap_reports_captured() {
        let store = CountingStore::new();
        seed_base(&store, OBJECT_ID, 10).await;
        store.reset_counts();
        let loader = loader_with(slice_of(verbatim_run(10, 12)));

        let decision = capture_layer_tail_with(&loader, &store, OBJECT_ID, REPLICA, &tail_cfg(8))
            .await
            .expect("capture succeeds");

        let (_, _, to_seq, _, capped) = captured_parts(&decision);
        assert_eq!(decision.outcome_label(), "captured");
        assert!(!capped);
        assert_eq!(to_seq, 12);
    }

    // ── Group E: zero-record-never-PUT ──────────────────────────────────────

    /// SR-1.3's enforcement point: a classifier refusal reaches the adapter as
    /// **nothing at all**.
    ///
    /// The assertion is at the STORE, not at the type, and that is the whole
    /// point of the case. `TailCaptured` keeps its record vector private so an
    /// empty capture is already unconstructible outside the classifier, and the
    /// unit's own belt-and-braces guard folds a hypothetical empty one onto
    /// `refused_missing_bytes`. Neither fact is what this Epic owes: the
    /// obligation is that no zero-record artifact ever reaches an adapter, and
    /// the codec deliberately PERMITS encoding one (a 44-byte header with
    /// `record_count == 0`) for its own round-trips. Only a store-side
    /// call-count can carry that claim.
    #[tokio::test]
    async fn capture_of_an_empty_sound_prefix_never_puts_an_artifact() {
        let store = CountingStore::new();
        seed_base(&store, OBJECT_ID, 10).await;
        store.reset_counts();
        let before = store
            .list_layer_checkpoints(OBJECT_ID)
            .await
            .expect("listing before");
        // A genuinely empty read window: the base sits at or ahead of the
        // stream head, nothing to capture.
        let loader = loader_with(slice_of(vec![]));

        let decision = capture_layer_tail_with(&loader, &store, OBJECT_ID, REPLICA, &tail_cfg(8))
            .await
            .expect("a refusal is not an error");

        assert_refused(
            &decision,
            TailCaptureRefusal::Classified(TailRefusal::Trimmed),
        );
        assert_eq!(decision.outcome_label(), "refused_trimmed");
        assert_eq!(
            store.count("put_layer_checkpoint"),
            0,
            "an empty prefix must not reach the adapter at all"
        );
        let after = store
            .list_layer_checkpoints(OBJECT_ID)
            .await
            .expect("listing after");
        assert_eq!(before, after, "the lineage is unchanged by a refusal");
    }

    // ── Group F: meta shape ─────────────────────────────────────────────────

    /// The five pinned lineage/kind fields on the persisted meta.
    ///
    /// `sidecar_bytes: None` is a regression guard, not tidiness: the shared
    /// put-time gate rejects a `Some`-sidecar tail as `Permanent`, so a capture
    /// that reused the base meta builder unmodified would fail at runtime on
    /// every tick.
    #[tokio::test]
    async fn capture_meta_carries_tail_kind_complete_lineage_and_no_sidecar() {
        let store = CountingStore::new();
        seed_base(&store, OBJECT_ID, 10).await;
        seed_tail(&store, OBJECT_ID, 10, 10, 20).await;
        store.reset_counts();
        let loader = loader_with(slice_of(verbatim_run(20, 22)));

        let decision = capture_layer_tail_with(&loader, &store, OBJECT_ID, REPLICA, &tail_cfg(8))
            .await
            .expect("capture succeeds");

        let (handle, ..) = captured_parts(&decision);
        let meta = fetch(&store, handle).await.meta;
        assert_eq!(meta.kind, CheckpointKind::Tail);
        assert_eq!(meta.snap_seq, Some(22), "a tail's snap_seq IS its to_seq");
        assert_eq!(meta.base_snap_seq, Some(10));
        assert_eq!(meta.from_seq, Some(20));
        assert!(
            meta.sidecar_bytes.is_none(),
            "a tail put carrying a sidecar is rejected `Permanent` by the shared put-time gate"
        );
    }

    /// Identity is header-sourced and deliberately NOT blanked: `apply`'s
    /// live-mapping identity resolution refuses identity-empty blobs at
    /// restore, so a tail meta carries the same four `obj:hdr` columns a base
    /// does.
    #[tokio::test]
    async fn capture_meta_carries_the_header_sourced_identity_fields() {
        let store = CountingStore::new();
        seed_base(&store, OBJECT_ID, 10).await;
        store.reset_counts();
        let loader = loader_with(slice_of(verbatim_run(10, 12)));

        let decision = capture_layer_tail_with(&loader, &store, OBJECT_ID, REPLICA, &tail_cfg(8))
            .await
            .expect("capture succeeds");

        let (handle, ..) = captured_parts(&decision);
        let meta = fetch(&store, handle).await.meta;
        assert_eq!(meta.session_id, SESSION_ID);
        assert_eq!(meta.layer_name, LAYER_NAME);
        assert_eq!(meta.created_by, CREATED_BY);
        assert_eq!(meta.server_ts, HEADER_SERVER_TS);
        assert_eq!(meta.creator_replica_id, REPLICA);
    }

    /// `observed_commit_span == to_seq - s0` — the ADJUDICATED value, not the
    /// `0` no-measurement sentinel.
    ///
    /// It is honest by construction: the fossil guard refuses whenever
    /// `restore_seq_bump != 0`, so the bump is always 0 by the time the meta is
    /// built and there is nothing to subtract. Stamping the sentinel instead
    /// would silently degrade every composed restore of a hot layer back to the
    /// configured safety margin — the smaller-band hazard direction. Epic 4
    /// inherits this value when it decides which meta a composed restore reads,
    /// so a truthful number here keeps that choice open.
    #[tokio::test]
    async fn capture_meta_observed_commit_span_equals_to_seq_minus_s0() {
        let store = CountingStore::new();
        seed_base(&store, OBJECT_ID, 10).await;
        seed_tail(&store, OBJECT_ID, 10, 10, 20).await;
        store.reset_counts();
        let loader = loader_with(slice_of(verbatim_run(20, 25)));

        let decision = capture_layer_tail_with(&loader, &store, OBJECT_ID, REPLICA, &tail_cfg(8))
            .await
            .expect("capture succeeds");

        let (handle, from_seq, to_seq, ..) = captured_parts(&decision);
        let meta = fetch(&store, handle).await.meta;
        assert_eq!(meta.observed_commit_span, to_seq - from_seq);
        assert_eq!(meta.observed_commit_span, 5);
        assert_ne!(
            meta.observed_commit_span, 0,
            "0 is the no-measurement sentinel and would fold a composed restore's guard band \
             back to the configured margin"
        );
    }

    // ── Group G: determinism (SR-9.2 / §22.19 Invariant 2) ──────────────────

    /// Two captures over equivalent fixtures produce **byte-identical**
    /// envelopes — the property that makes two leaders racing on the same
    /// `(object_id, to_seq, tail)` address safe under blind
    /// last-writer-wins.
    ///
    /// TWO independent stores, deliberately: re-running against the SAME store
    /// is a different scenario, because S0 would advance to the new tail's
    /// `to_seq` and the canned slice would then fail the index-0 first-entry
    /// rule (`Trimmed`). The same-key duplicate-leader race is the Epic's
    /// integration Task.
    ///
    /// The assertion is on `blob.bytes` ONLY. `written_at_unix_ms` and
    /// `creator_replica_id` live on the meta, are per-instant and per-writer by
    /// convention, and are deliberately OUTSIDE the bytes the determinism
    /// argument depends on — the two runs below even use different replica ids
    /// to prove neither reaches the envelope.
    #[tokio::test]
    async fn two_captures_over_the_same_fixture_produce_byte_identical_artifacts() {
        async fn run(replica: &str) -> CheckpointBlob {
            let store = CountingStore::new();
            seed_base(&store, OBJECT_ID, 10).await;
            seed_tail(&store, OBJECT_ID, 10, 10, 20).await;
            let loader = loader_with(slice_of(vec![
                verbatim_entry(21, &ompf_for(21)),
                empty_delta_entry(22),
                verbatim_entry(23, &ompf_for(23)),
            ]));
            let decision =
                capture_layer_tail_with(&loader, &store, OBJECT_ID, replica, &tail_cfg(8))
                    .await
                    .expect("capture succeeds");
            let (handle, ..) = captured_parts(&decision);
            assert_tail_handle(handle, OBJECT_ID, 23);
            fetch(&store, handle).await
        }

        let first = run("replica-a").await;
        let second = run("replica-b").await;

        assert_eq!(first.handle, second.handle, "both address (oid, 23, Tail)");
        assert_eq!(
            first.bytes, second.bytes,
            "SR-9.2 determinism is an ENVELOPE property: identical slice + identical lineage \
             head must encode to identical bytes, whatever the writer or the wall clock"
        );

        // Anti-vacuity: two empty artifacts would also compare equal.
        assert!(!first.bytes.is_empty());
        let envelope = decode_tail_envelope(&first.bytes).expect("artifact decodes");
        assert_eq!(envelope.records().len(), 3);
        // And the per-writer meta really did differ, so the equality above is
        // a statement about the bytes rather than about the whole blob.
        assert_ne!(
            first.meta.creator_replica_id,
            second.meta.creator_replica_id
        );
    }

    // ── Group H: post-restore fossil guard ──────────────────────────────────

    /// An outstanding DR-restore stamp refuses conservatively: the stream may
    /// still hold abandoned-timeline fossils below an ascending jump that a
    /// bounded read cannot see.
    #[tokio::test]
    async fn capture_with_a_nonzero_restore_seq_bump_refuses_and_persists_nothing() {
        let store = CountingStore::new();
        seed_base(&store, OBJECT_ID, 10).await;
        store.reset_counts();
        let loader = FakeTailSliceLoader::new(vec![Ok(slice_of(verbatim_run(10, 12)))])
            .with_header_identity(fossil_identity(7));

        let decision = capture_layer_tail_with(&loader, &store, OBJECT_ID, REPLICA, &tail_cfg(8))
            .await
            .expect("a refusal is not an error");

        assert_refused(
            &decision,
            TailCaptureRefusal::PostRestoreUnrebased {
                restore_seq_bump: 7,
            },
        );
        // The pinned fold — a member of the existing vocabulary, not a tenth
        // label.
        assert_eq!(decision.outcome_label(), "refused_seq_gap");
        assert_eq!(store.count("put_layer_checkpoint"), 0);
        // Ordering evidence: the guard precedes the XRANGE it fences.
        assert_eq!(loader.header_reads(), 1);
        assert!(
            loader.calls().is_empty(),
            "the fossil guard must refuse before the slice read"
        );
    }

    /// `cursor_columns_trusted == false` means a cursor column was present but
    /// unparseable and folded to 0, so a fabricated 0 is indistinguishable from
    /// a truthful one. Refusing is the conservative direction, matching the
    /// assembler's decline posture on the same input.
    #[tokio::test]
    async fn capture_with_untrusted_cursor_columns_refuses_the_same_way() {
        let store = CountingStore::new();
        seed_base(&store, OBJECT_ID, 10).await;
        store.reset_counts();
        let loader = FakeTailSliceLoader::new(vec![Ok(slice_of(verbatim_run(10, 12)))])
            .with_header_identity(untrusted_identity());

        let decision = capture_layer_tail_with(&loader, &store, OBJECT_ID, REPLICA, &tail_cfg(8))
            .await
            .expect("a refusal is not an error");

        assert_refused(
            &decision,
            TailCaptureRefusal::PostRestoreUnrebased {
                // 0 here means "the column itself was untrustworthy", not "no
                // restore bump" — the variant's documented reading.
                restore_seq_bump: 0,
            },
        );
        assert_eq!(decision.outcome_label(), "refused_seq_gap");
        assert_eq!(store.count("put_layer_checkpoint"), 0);
        assert!(loader.calls().is_empty());
    }

    /// Anti-vacuity twin for the two fossil cases: the identical fixture with a
    /// zero bump and trusted columns proceeds to a capture, so the refusals
    /// above are the guard firing rather than the fixture being uncapturable.
    #[tokio::test]
    async fn capture_with_a_zero_restore_seq_bump_proceeds_to_capture() {
        let store = CountingStore::new();
        seed_base(&store, OBJECT_ID, 10).await;
        store.reset_counts();
        let loader = FakeTailSliceLoader::new(vec![Ok(slice_of(verbatim_run(10, 12)))])
            .with_header_identity(trusted_identity());

        let decision = capture_layer_tail_with(&loader, &store, OBJECT_ID, REPLICA, &tail_cfg(8))
            .await
            .expect("capture succeeds");

        let (handle, from_seq, to_seq, ..) = captured_parts(&decision);
        assert_eq!(decision.outcome_label(), "captured");
        assert_eq!((from_seq, to_seq), (10, 12));
        assert_tail_handle(handle, OBJECT_ID, 12);
        assert_eq!(store.count("put_layer_checkpoint"), 1);
    }

    /// A vanished `obj:hdr` is `NotFound`, deliberately NOT the assembler's
    /// fold to identity defaults: nothing has been built yet at that point, and
    /// the fold would fabricate the very guard input this step exists to check.
    #[tokio::test]
    async fn capture_with_an_absent_object_header_returns_not_found() {
        let store = CountingStore::new();
        seed_base(&store, OBJECT_ID, 10).await;
        store.reset_counts();
        let loader =
            FakeTailSliceLoader::new(vec![Ok(slice_of(verbatim_run(10, 12)))]).with_absent_header();

        let err = capture_layer_tail_with(&loader, &store, OBJECT_ID, REPLICA, &tail_cfg(8))
            .await
            .expect_err("an absent header is an error, not a refusal");

        assert!(
            matches!(err, CheckpointError::NotFound),
            "expected NotFound, got {err:?}"
        );
        assert_eq!(store.count("put_layer_checkpoint"), 0);
        assert!(
            loader.calls().is_empty(),
            "the header read fences the slice read on the absent-header arm too"
        );
    }

    // ── Group I: caps derivation ────────────────────────────────────────────

    /// The caps handed to the slice read reserve envelope framing.
    ///
    /// `TailSliceCaps::max_payload_bytes` counts RAW VERBATIM PAYLOAD bytes
    /// only; framing is the caller's to reserve. Without the reservation a
    /// prefix sized exactly to the operator's byte cap would encode into an
    /// envelope `44 + 21·N` bytes OVER it.
    ///
    /// Both the formula (spelled through the imported constants) and the
    /// resulting literals are asserted: the formula catches a reservation that
    /// was dropped, the literals catch a formula that was copied and then
    /// drifted.
    #[tokio::test]
    async fn caps_passed_to_the_slice_read_reserve_envelope_framing() {
        // (a) byte-bound: the entries knob is far above what the byte cap can
        // frame, so `entries_by_bytes` decides.
        let caps = observed_caps(4096, 1000).await;
        let entries_by_bytes = (1000 - TAIL_HEADER_SIZE) / TAIL_RECORD_PREFIX_SIZE;
        assert_eq!(caps.max_entries, entries_by_bytes);
        assert_eq!(caps.max_entries, 45);
        assert_eq!(
            caps.max_payload_bytes,
            (1000 - TAIL_HEADER_SIZE - TAIL_RECORD_PREFIX_SIZE * caps.max_entries) as u64
        );
        assert_eq!(caps.max_payload_bytes, 11);
        assert!(caps.max_entries >= 1);

        // (b) entries-bound: the operator's entry cap is the binding one, and
        // the reservation shrinks with it rather than staying at the byte-cap
        // worst case.
        let caps = observed_caps(10, 10_000).await;
        assert_eq!(caps.max_entries, 10);
        assert_eq!(
            caps.max_payload_bytes,
            (10_000 - TAIL_HEADER_SIZE - TAIL_RECORD_PREFIX_SIZE * 10) as u64
        );
        assert_eq!(caps.max_payload_bytes, 9_746);
        assert!(caps.max_entries >= 1);
    }

    /// Drive one capture and return the [`TailSliceCaps`] the unit derived.
    async fn observed_caps(max_entries: u64, max_bytes: usize) -> TailSliceCaps {
        let store = CountingStore::new();
        seed_base(&store, OBJECT_ID, 10).await;
        let loader = loader_with(slice_of(verbatim_run(10, 11)));

        capture_layer_tail_with(
            &loader,
            &store,
            OBJECT_ID,
            REPLICA,
            &tail_cfg_caps(max_entries, max_bytes),
        )
        .await
        .expect("capture succeeds");

        let calls = loader.calls();
        assert_eq!(calls.len(), 1, "exactly one bounded slice read per capture");
        calls[0].2
    }

    /// Boundary: a prefix filling the derived payload cap exactly encodes to an
    /// envelope that is exactly the operator's byte cap — never over it.
    #[tokio::test]
    async fn capture_at_the_derived_cap_never_exceeds_max_bytes() {
        // 4 entries, 1000-byte cap → reserve = 44 + 21·4 = 128, payload = 872,
        // i.e. 218 bytes per record.
        let cfg = tail_cfg_caps(4, 1000);
        let per_record = 218usize;
        let payload = vec![b'x'; per_record];
        let entries: Vec<TailStreamEntry> =
            (11..=14).map(|seq| verbatim_entry(seq, &payload)).collect();

        let store = CountingStore::new();
        seed_base(&store, OBJECT_ID, 10).await;
        store.reset_counts();
        let loader = loader_with(slice_of(entries));

        let decision = capture_layer_tail_with(&loader, &store, OBJECT_ID, REPLICA, &cfg)
            .await
            .expect("capture succeeds");

        // The fixture is genuinely AT the derived cap, not comfortably under.
        let caps = loader.calls()[0].2;
        assert_eq!(caps.max_entries, 4);
        assert_eq!(caps.max_payload_bytes, (4 * per_record) as u64);

        let (handle, ..) = captured_parts(&decision);
        let blob = fetch(&store, handle).await;
        assert_eq!(
            blob.bytes.len(),
            cfg.max_bytes,
            "a payload filling the derived cap must encode to exactly the configured byte cap"
        );
        assert!(blob.bytes.len() <= cfg.max_bytes);
        let envelope = decode_tail_envelope(&blob.bytes).expect("persisted artifact decodes");
        assert_eq!(envelope.records().len(), 4);
    }

    /// A byte cap that cannot fit one framed record refuses BEFORE any read.
    ///
    /// `max_bytes = 65` is one byte under the 44 + 21 + 1 floor `Config`
    /// enforces at startup, so this arm is defense-in-depth — but it refuses
    /// observably rather than silently clamping, because a zero payload budget
    /// would otherwise have the read primitive exclude the first entry on every
    /// tick forever. The label folds onto `refused_missing_bytes`: the honest
    /// statement is identical to the oversize-first-entry case ("no admissible
    /// first entry").
    #[tokio::test]
    async fn pathological_byte_cap_refuses_before_any_read() {
        let store = CountingStore::new();
        seed_base(&store, OBJECT_ID, 10).await;
        store.reset_counts();
        let loader = loader_with(slice_of(verbatim_run(10, 12)));

        let decision = capture_layer_tail_with(
            &loader,
            &store,
            OBJECT_ID,
            REPLICA,
            &tail_cfg_caps(4096, TAIL_HEADER_SIZE + TAIL_RECORD_PREFIX_SIZE),
        )
        .await
        .expect("a refusal is not an error");

        assert_refused(
            &decision,
            TailCaptureRefusal::Classified(TailRefusal::MissingBytes),
        );
        assert_eq!(decision.outcome_label(), "refused_missing_bytes");
        assert_eq!(store.count("put_layer_checkpoint"), 0);
        assert!(
            loader.call_log().is_empty(),
            "the zero-budget arm must refuse before the header read and the slice read"
        );
    }

    // ── Group J: error propagation ──────────────────────────────────────────

    /// A `Transient` from the lineage LISTING — the unit's *first* fallible
    /// call — surfaces as `Transient`, classified, and short-circuits the whole
    /// unit: no header read, no slice read, no put.
    ///
    /// The loader assertion is the load-bearing half. The listing is what
    /// yields S0, so a listing failure that still let the seam run would mean
    /// the unit read a tail against an unaddressed (or defaulted) base — the
    /// failure mode `call_log().is_empty()` rules out, and one an
    /// error-kind-only assertion could not see.
    #[tokio::test]
    async fn capture_propagates_a_classified_listing_error() {
        let store = CountingStore::new();
        seed_base(&store, OBJECT_ID, 10).await;
        store.reset_counts();
        store.fail_next_list(CheckpointError::Transient {
            source: "injected listing failure".into(),
        });
        let loader = loader_with(slice_of(verbatim_run(10, 12)));

        let err = capture_layer_tail_with(&loader, &store, OBJECT_ID, REPLICA, &tail_cfg(8))
            .await
            .expect_err("the listing failed");

        assert!(
            matches!(err, CheckpointError::Transient { .. }),
            "expected Transient, got {err:?}"
        );
        assert_eq!(
            store.count("list_layer_checkpoints"),
            1,
            "the listing was attempted once and no retry loop re-drove it"
        );
        assert_eq!(
            store.count("put_layer_checkpoint"),
            0,
            "a failed lineage addressing must persist nothing"
        );
        assert!(
            loader.call_log().is_empty(),
            "the seam must not be touched at all when addressing failed, got {:?}",
            loader.call_log()
        );
    }

    /// A `Transient` from the HEADER read propagates classified, and fences the
    /// slice read behind it: the fossil guard reads `obj:hdr` before any
    /// `XRANGE`, so a header failure must cost zero slice reads and zero puts.
    #[tokio::test]
    async fn capture_propagates_a_classified_header_read_error() {
        let store = CountingStore::new();
        seed_base(&store, OBJECT_ID, 10).await;
        store.reset_counts();
        let loader = FakeTailSliceLoader::new(vec![Ok(slice_of(verbatim_run(10, 12)))])
            .with_header_error(CheckpointError::Transient {
                source: "injected obj:hdr read failure".into(),
            });

        let err = capture_layer_tail_with(&loader, &store, OBJECT_ID, REPLICA, &tail_cfg(8))
            .await
            .expect_err("the header read failed");

        assert!(
            matches!(err, CheckpointError::Transient { .. }),
            "expected Transient, got {err:?}"
        );
        assert_eq!(
            store.count("put_layer_checkpoint"),
            0,
            "a failed header read must persist nothing"
        );
        assert_eq!(
            loader.call_log(),
            vec![READ_HEADER_IDENTITY],
            "the header read ran and fenced the slice read behind it"
        );
    }

    /// A `Transient` from the PUT surfaces as `Transient`, classified — the
    /// tick is the retry, so the classification is what decides whether the
    /// next tick re-attempts.
    #[tokio::test]
    async fn capture_propagates_a_classified_store_put_error() {
        let store = CountingStore::new();
        seed_base(&store, OBJECT_ID, 10).await;
        store.reset_counts();
        store.fail_next_put(CheckpointError::Transient {
            source: "injected backend failure".into(),
        });
        let loader = loader_with(slice_of(verbatim_run(10, 12)));

        let err = capture_layer_tail_with(&loader, &store, OBJECT_ID, REPLICA, &tail_cfg(8))
            .await
            .expect_err("the put failed");

        assert!(
            matches!(err, CheckpointError::Transient { .. }),
            "expected Transient, got {err:?}"
        );
        assert_eq!(
            store.count("put_layer_checkpoint"),
            1,
            "the put was attempted once and no retry loop re-drove it"
        );
        assert!(
            store
                .list_layer_checkpoints(OBJECT_ID)
                .await
                .expect("listing")
                .iter()
                .all(|h| h.kind == CheckpointKind::Base),
            "a failed put must leave no tail generation behind"
        );
    }

    /// A `Transient` from the slice read propagates unchanged, and nothing is
    /// persisted on the way out.
    #[tokio::test]
    async fn capture_propagates_a_classified_slice_read_error() {
        let store = CountingStore::new();
        seed_base(&store, OBJECT_ID, 10).await;
        store.reset_counts();
        let loader = FakeTailSliceLoader::new(vec![Err(CheckpointError::Transient {
            source: "injected XRANGE failure".into(),
        })])
        .with_header_identity(trusted_identity());

        let err = capture_layer_tail_with(&loader, &store, OBJECT_ID, REPLICA, &tail_cfg(8))
            .await
            .expect_err("the slice read failed");

        assert!(
            matches!(err, CheckpointError::Transient { .. }),
            "expected Transient, got {err:?}"
        );
        assert_eq!(store.count("put_layer_checkpoint"), 0);
    }

    /// A prefix whose `to_seq` exceeds the codec's 2^53 − 1 ceiling surfaces a
    /// classified `Permanent` (the §10 mapping — a retry reproduces it
    /// exactly), never a panic and never a persisted artifact.
    #[tokio::test]
    async fn capture_of_a_slice_at_the_sequence_ceiling_surfaces_a_classified_error() {
        let store = CountingStore::new();
        seed_base(&store, OBJECT_ID, TAIL_MAX_SEQUENCE).await;
        store.reset_counts();
        let over_ceiling = TAIL_MAX_SEQUENCE + 1;
        let loader = loader_with(slice_of(vec![verbatim_entry(
            over_ceiling,
            &ompf_for(over_ceiling),
        )]));

        let err = capture_layer_tail_with(&loader, &store, OBJECT_ID, REPLICA, &tail_cfg(8))
            .await
            .expect_err("encoding above the sequence ceiling fails");

        assert!(
            matches!(err, CheckpointError::Permanent { .. }),
            "expected Permanent, got {err:?}"
        );
        assert_eq!(store.count("put_layer_checkpoint"), 0);
    }
}
