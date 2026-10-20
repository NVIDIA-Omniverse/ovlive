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

//! Lineage gather and restore selection (SR-3.1).
//!
//! Given a layer's checkpoint listing, pick the newest **restorable**
//! generation. When that generation is a [`CheckpointKind::Base`] the answer is
//! byte-identical to the pre-Epic-4 `recovery::select_restore_source` result.
//! When it is a [`CheckpointKind::Tail`], its whole lineage — the base it roots
//! on plus the contiguous tail run through it — is gathered with every link
//! verified against the artifacts' own envelope headers, and a lineage that
//! does not verify **steps down** to the next older restorable generation
//! instead of failing the restore.
//!
//! # What lives here and what does not
//!
//! This module is a free-standing policy unit over `&dyn CheckpointStore`,
//! mirroring the shape of Epic 3's capture unit
//! ([`crate::checkpoint::tail_capture::capture_layer_tail_with`]): it returns a
//! *decision* the caller records. Deliberately absent from every signature
//! here: the recovery hook, `LayerStore`, [`crate::config::Config`], and any
//! metrics handle. The SR-4.1 chain-collapse re-base
//! ([`crate::checkpoint::rebase::rebase_layer_lineage`]) calls the same entry
//! point from the checkpoint tick, where none of those are the same objects,
//! and the hook / tick wiring is what turns a decision into a metric increment.
//!
//! Three fences worth stating because each has an owner elsewhere:
//!
//! - **No decode.** [`decode_tail_envelope`](crate::checkpoint::tail_codec::decode_tail_envelope)
//!   is O(records) and allocates; this module only ever calls
//!   [`peek_tail_header`], which reads 44 bytes and allocates nothing. Turning
//!   the gathered bytes into replayable records is the composition engine's job
//!   (Task 1's `compose_from_artifacts`, which takes bytes in).
//! - **No `Config`.** The chain bound `LIVE_EDIT_CHECKPOINT_TAIL_MAX_CHAIN` is a
//!   *capture-side* knob. A store can legitimately hold a chain written when K
//!   was larger, and gating the walk on the current K would silently orphan a
//!   restorable lineage. The walk is bounded structurally instead (see
//!   [`LineageSelection`]). This is the same flag-independence argument
//!   `select_restore_source` already makes for itself.
//! - **No metrics.** Label spellings live on the decision types
//!   ([`GatheredLineage::outcome_label`], [`LineageBreak::reason_label`],
//!   [`Exhaustion::outcome_label`]) and delegate to
//!   [`CHECKPOINT_RESTORE_LINEAGE_OUTCOMES`](crate::observability::metrics::CHECKPOINT_RESTORE_LINEAGE_OUTCOMES),
//!   which is canonical.
//!
//! # The meta is an index, the envelope header is the source of truth
//!
//! Every fetched tail is peeked and its [`TailHeader`] is the lineage of
//! record. `CheckpointMeta`'s own lockstep doc says it outright: "a meta that
//! disagrees with its envelope loses — the meta is an index, not a source of
//! truth". A meta that is `Some` and *disagrees* WARNs here (operator signal)
//! and loses.
//!
//! Both shipped adapters used to answer a torn-put residual (a tail whose meta
//! artifact never landed) with `Ok(Some(..))` carrying a placeholder meta —
//! `kind: Tail`, `base_snap_seq: None`, `from_seq: None`, empty identity
//! fields — plus a WARN, and the header-is-authoritative rule was what made
//! that safe. Since b.hka (d) they refuse it instead, as a `Permanent` carrying
//! a typed
//! [`MissingMetaCompanion`](crate::checkpoint::store::MissingMetaCompanion);
//! the walk catches that one `Permanent` and steps down on its own
//! [`LineageBreak::MissingMetaCompanion`] (outcome label
//! `stepped_down_missing_meta` — a torn put is not a decode failure). The
//! header-is-authoritative rule
//! is unchanged and still load-bearing — it is what makes a meta that
//! *disagrees* harmless — but it is no longer what carries the meta-absent
//! case.
//!
//! # `from_seq` is EXCLUSIVE
//!
//! A tail's span is `(from_seq, to_seq]`; the first carried commit is
//! `from_seq + 1` (b.rrx). Link verification therefore compares
//! `predecessor.to_seq == successor.from_seq` with **no** `+1` / `-1` anywhere.

use std::collections::{HashMap, HashSet};
use std::time::Duration;

use bytes::Bytes;
use tokio::time::Instant;

use crate::checkpoint::store::{
    CheckpointBlob, CheckpointError, CheckpointHandle, CheckpointKind, CheckpointMeta,
    CheckpointStore,
};
use crate::checkpoint::tail_codec::{peek_tail_header, TailHeader};

// ── Output shape ────────────────────────────────────────────────────────────

/// One tail member of a gathered lineage.
///
/// `bytes` are the **undecoded** envelope. Task 1's composition engine decodes
/// them; this module never does (module header, "No decode").
#[derive(Debug, Clone)]
pub struct GatheredTail {
    /// The generation's handle, as the adapter minted it — so `adapter_key` is
    /// the adapter's own canonical key, never a caller fabrication.
    pub handle: CheckpointHandle,
    /// The raw tail-envelope bytes, opaque here.
    pub bytes: Bytes,
    /// The authoritative lineage, peeked out of `bytes`. Verified against the
    /// handle (`to_seq == handle.seq()`, `object_id` matches) and against the
    /// link that reached this member.
    pub header: TailHeader,
    /// The artifact's on-wire `format_version`, as it rode in
    /// [`peek_tail_header`]'s returned tuple.
    ///
    /// Always
    /// [`TAIL_FORMAT_VERSION`](crate::checkpoint::tail_codec::TAIL_FORMAT_VERSION)
    /// on a gathered member — the peek validates the version immediately after
    /// the magic and *before* any length check, so an artifact written at a
    /// version this build cannot read never becomes a member; it becomes a
    /// [`LineageBreak::Decode`] step-down (SR-6.1: surfaced, never guessed).
    /// Carried anyway so the composition engine need not re-derive it, and so
    /// a future v2 lineage has a place to report the mix.
    pub format_version: u32,
    /// The generation's meta, carried for completeness. **Advisory only** — see
    /// the module header, and see [`GatheredLineage`] for why it must never be
    /// used as an identity carrier.
    pub meta: CheckpointMeta,
}

/// One verified restorable lineage: a base plus the contiguous tails replayed
/// on top of it.
///
/// # Consumer
///
/// Task 1's `compose_from_artifacts` consumes this, bytes-in: `base.bytes` is
/// the reset OMPF, each `tails[i].bytes` an undecoded envelope. Decoding is the
/// composition engine's, per this module's no-`decode_tail_envelope` rule.
///
/// # Replay order — pinned
///
/// `tails` is **oldest-first**: `tails[0]` is the tail rooted directly at the
/// base, `tails[n-1]` the head. The walk that produces them runs head-down, so
/// they are reversed before being returned; pinning the direction here is what
/// lets Task 1's replay iterate the slice without re-deriving it.
///
/// # Identity is base-sourced — MUST
///
/// `base.meta` is the **only** identity carrier a caller may hand to
/// `apply::live_mapping_resolves_blob_identity`, which refuses a blob with an
/// empty `session_id` / `layer_name` outright. [`GatheredTail::meta`] exists
/// for diagnostics, not for identity.
///
/// The rule predates and outlives its original motivation: it was written when
/// both adapters answered a meta-absent tail with an identity-empty
/// placeholder. b.hka (d) removed those placeholders (the adapters refuse now),
/// so no shipped read path produces an identity-empty tail meta — but the MUST
/// stands on its own, because a tail's identity is base-sourced by
/// construction and reading it off the head generation would be wrong even
/// when the head's copy happens to be populated.
///
/// # The floor property
///
/// A base-only lineage is `tails.is_empty()`, and it is byte-equivalent to the
/// pre-Epic-4 selection result. Composed restore can only ever *add* coverage:
/// gather can never resolve to something worse than today's newest-base answer.
/// [`LineageSelection`]'s in-module table test is the executable form of that
/// claim.
#[derive(Debug, Clone)]
pub struct GatheredLineage {
    /// The reset-decodable full-state snapshot this lineage roots at.
    pub base: CheckpointBlob,
    /// The contiguous tail run, **oldest-first**. Empty for a base-only
    /// lineage.
    pub tails: Vec<GatheredTail>,
    /// **S** — the sequence this lineage restores to: the head tail's `to_seq`,
    /// or `base.handle.seq()` when `tails` is empty.
    ///
    /// Exposed rather than left to callers to re-derive because
    /// `try_restore_layer` computes its `snap_seq` as
    /// `blob.meta.snap_seq.unwrap_or(blob.handle.snap_seq)`, which for a
    /// composed lineage would yield the *base's* sequence — not S.
    pub head_seq: u64,
}

impl GatheredLineage {
    /// The number of tails replayed onto the base.
    ///
    /// This is the **true** `from`/`to` link-walk chain length SR-9.2
    /// specifies, as opposed to the conservative listing-position
    /// approximation Epic 3's capture unit uses (`tail_capture.rs`, "The chain
    /// count is a documented conservative over-approximation").
    ///
    /// # The consumer
    ///
    /// [`rebase_layer_lineage`](crate::checkpoint::rebase::rebase_layer_lineage)
    /// is the binding one (SR-4.1 / SR-9.2): the K bound that decides whether a
    /// chain-at-K layer collapses is read from **here**, never from the
    /// listing-position count. The divergence is deliberate on both sides — the
    /// listing count over-approximates, which is the safe direction for the
    /// capture unit's per-tick pre-filter but the wrong direction for a re-base
    /// trigger, where an over-count would burn a full GET-compose-PUT the bound
    /// never demanded. The re-base's `UnderBound` verdict is exactly the observable
    /// for that gap.
    pub fn chain_len(&self) -> usize {
        self.tails.len()
    }

    /// Whether this lineage is the base alone — today's exact restore answer.
    pub fn is_base_only(&self) -> bool {
        self.tails.is_empty()
    }

    /// The lineage root's sequence: `base.handle.seq()`.
    pub fn base_snap_seq(&self) -> u64 {
        self.base.handle.seq()
    }

    /// This lineage's `outcome` label on
    /// `live_edit_checkpoint_restore_lineage_total`.
    ///
    /// Every returned value is a member of
    /// [`CHECKPOINT_RESTORE_LINEAGE_OUTCOMES`](crate::observability::metrics::CHECKPOINT_RESTORE_LINEAGE_OUTCOMES),
    /// which is the canonical vocabulary; this method must not re-spell it.
    pub fn outcome_label(&self) -> &'static str {
        if self.is_base_only() {
            "gathered_base"
        } else {
            "gathered_composed"
        }
    }
}

// ── Step-down reasons ───────────────────────────────────────────────────────

/// Why one candidate lineage was abandoned.
///
/// Every variant is a **verdict**, never an `Err`: a broken lineage is a
/// step-down signal, and the store's own `Transient` / `Permanent` /
/// `NotImplemented` errors propagate verbatim past all of this.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LineageBreak {
    /// A link's generation is not reachable: it is absent from the listing, or
    /// `get_layer_checkpoint` answered `Ok(None)` (the contract-documented
    /// concurrent-prune race).
    LinkAbsent {
        /// The sequence the link named.
        seq: u64,
        /// The kind the link named — `Base` iff `seq == base_snap_seq`.
        kind: CheckpointKind,
        /// `true` when the listing held the handle but the fetch returned
        /// `Ok(None)`; `false` when the listing never held it.
        at_fetch: bool,
    },
    /// A member's authoritative header disagreed with the expected link, the
    /// lineage root, the addressed kind, or the object id.
    ///
    /// Also the shape the per-member **adapter-contract-violation** guard takes
    /// — the guard `select_restore_source` carries once for its single fetch,
    /// applied here to every member.
    LinkMismatch {
        /// The offending member's sequence.
        seq: u64,
        /// Which quantity disagreed (`"kind"`, `"object_id"`, `"snap_seq"`,
        /// `"header.object_id"`, `"header.to_seq"`, `"base_snap_seq"`,
        /// `"from_seq"`, `"lineage_cycle"`).
        field: &'static str,
        /// What the link required.
        expected: String,
        /// What the artifact carried.
        observed: String,
    },
    /// The member's bytes could not be turned into a usable generation:
    /// [`peek_tail_header`] rejected the envelope — an unknown `format_version`
    /// (SR-6.1 — surfaced, never guessed), a bad magic, a truncated artifact,
    /// or a header whose framing contradicts its declared span.
    ///
    /// Strictly a *bytes* verdict. A generation whose artifact reads fine but
    /// whose meta companion never landed is [`Self::MissingMetaCompanion`], not
    /// this — the two carry different operator remediations (bit-rot / version
    /// skew vs. a torn put) and therefore different outcome labels.
    Decode {
        /// The offending member's sequence.
        seq: u64,
        /// The codec error's `Display`, which names the version on the SR-6.1
        /// arm.
        message: String,
    },
    /// The adapter refused the member because its `.meta.json` companion is
    /// absent while its artifact is durable — the typed `Permanent` carrying
    /// [`MissingMetaCompanion`](crate::checkpoint::store::MissingMetaCompanion),
    /// recognised through [`CheckpointError::is_missing_meta_companion`]
    /// (b.hka (d)).
    ///
    /// Its own break rather than a [`Self::Decode`] with a different message:
    /// the break is what the operator-facing outcome label is derived from, and
    /// a torn put (the artifact landed, its index never did) is a different
    /// fault with a different remediation from an unreadable envelope. Folding
    /// the two would report a routine write-path interruption as storage
    /// corruption or a version skew.
    MissingMetaCompanion {
        /// The offending member's sequence.
        seq: u64,
        /// The adapter's `MissingMetaCompanion` `Display`, which names the
        /// artifact whose companion is missing.
        message: String,
    },
    /// The reserve-bounded abandonment: the remaining deadline could not fund
    /// one more member fetch at its kind-dependent allowance, a single fetch
    /// overran that allowance, or the member is one an earlier candidate
    /// already watched overrun.
    Budget {
        /// The member the walk gave up on.
        seq: u64,
        /// Budget left when the check ran, in milliseconds — the **live**
        /// remainder, sampled when the allowance was computed, or at mint time
        /// on the overrun and cached-`Wedged` arms.
        ///
        /// Including the two overrun arms: the fetch that ate its whole cap
        /// without answering, and every later candidate that declines the same
        /// member on the cached `LinkFact::Wedged` fact. Those used to report a
        /// flat `0` to say "what was abandoned is the *member*, not the clock";
        /// that distinction now lives in `seq` alone (which names the member)
        /// and this field reports the clock honestly instead. It has to: under
        /// the kind-dependent allowance (see
        /// [`gather_restore_lineage`]'s `reserve` discussion) a wedged tail
        /// leaves a real, non-zero remainder that is precisely the base's
        /// reservation, and a `0` here would misreport a selection that is
        /// still able to fund its floor answer as one with nothing left.
        remaining_ms: u64,
        /// The per-member reserve, in milliseconds.
        reserve_ms: u64,
    },
    /// The caller's composition marked this lineage `Permanent` (Task 1's
    /// per-generation classification). Recorded through
    /// [`LineageSelection::record_compose_failure`] so the caller's step-down
    /// and this module's step-downs share one vocabulary.
    Compose {
        /// The abandoned lineage's `head_seq`.
        head_seq: u64,
    },
}

impl LineageBreak {
    /// This break's `outcome` label on
    /// `live_edit_checkpoint_restore_lineage_total`.
    ///
    /// Members of
    /// [`CHECKPOINT_RESTORE_LINEAGE_OUTCOMES`](crate::observability::metrics::CHECKPOINT_RESTORE_LINEAGE_OUTCOMES);
    /// that constant is canonical and this method must not re-spell it.
    pub fn reason_label(&self) -> &'static str {
        match self {
            LineageBreak::LinkAbsent { .. } => "stepped_down_link_broken",
            LineageBreak::LinkMismatch { .. } => "stepped_down_link_mismatch",
            LineageBreak::Decode { .. } => "stepped_down_decode",
            LineageBreak::MissingMetaCompanion { .. } => "stepped_down_missing_meta",
            LineageBreak::Budget { .. } => "stepped_down_budget",
            LineageBreak::Compose { .. } => "stepped_down_compose",
        }
    }

    /// The sequence of the generation the break is about — the offending link
    /// for every walk-side variant, the abandoned head for
    /// [`LineageBreak::Compose`].
    pub fn link_seq(&self) -> u64 {
        match self {
            LineageBreak::LinkAbsent { seq, .. }
            | LineageBreak::LinkMismatch { seq, .. }
            | LineageBreak::Decode { seq, .. }
            | LineageBreak::MissingMetaCompanion { seq, .. }
            | LineageBreak::Budget { seq, .. } => *seq,
            LineageBreak::Compose { head_seq } => *head_seq,
        }
    }

    /// `(expected, observed)` for the tracing pair, or `None` where the break
    /// has no such pair.
    fn expectation(&self) -> Option<(&str, &str)> {
        match self {
            LineageBreak::LinkMismatch {
                expected, observed, ..
            } => Some((expected.as_str(), observed.as_str())),
            _ => None,
        }
    }

    /// The field name a [`LineageBreak::LinkMismatch`] names, or `""`.
    fn field(&self) -> &'static str {
        match self {
            LineageBreak::LinkMismatch { field, .. } => field,
            _ => "",
        }
    }
}

/// One abandoned candidate: which generation headed it and why it was
/// abandoned. Accumulated across a whole selection so the caller can report the
/// full step-down chain on exhaustion.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StepDownRecord {
    /// The abandoned candidate's head sequence.
    pub head_seq: u64,
    /// The abandoned candidate's head kind.
    pub head_kind: CheckpointKind,
    /// Why it was abandoned.
    pub reason: LineageBreak,
}

/// Why a selection ended with nothing restorable.
///
/// The split is **not** purely diagnostic. Three of the four variants are
/// *conclusive* — the selection holds positive evidence about every listed
/// [`CheckpointKind::Base`], so "nothing is restorable" is a statement about
/// the layer — and they fold onto the single `exhausted` metric outcome, where
/// the variant is the diagnostic distinction (the same split Epic 3 uses for
/// its post-restore fold). [`Exhaustion::BudgetInconclusive`] is the fourth,
/// and it is a statement about the *selection* rather than a conclusive one
/// about the layer — it may well hold evidence about the layer, just never
/// enough to prove the layer unrestorable. It
/// carries its own `exhausted_budget` outcome, and the restore consumer
/// ([`select_restore_source`]) is the one that maps it to a retriable
/// `Transient`. That mapping is that consumer's, not a property of the variant
/// — this module holds no caller knowledge (see [`gather_restore_lineage`],
/// "Signature — a cross-task contract"), and its other consumer, the SR-4.1
/// re-base, is kind-blind and refuses on every exhaustion alike.
///
/// **The classification rule, exactly.** `Ok(None)` → `NotFound` is licensed
/// only when the selection adjudicated every `Base` in the listing. A listed
/// `Base` is *adjudicated* when the selection learned something about the
/// artifact itself — it fetched and classified it, found it unusable, or found
/// it absent at fetch (the concurrent-prune race is positive evidence: the
/// generation is gone). It is *unadjudicated* when the walk never got an answer
/// about it: the fetch was declined at a zero allowance, or it wedged and ate
/// its allowance without answering. Neither of those says anything about the
/// artifact, so a selection holding one cannot claim the layer is unrestorable.
///
/// [`select_restore_source`]: crate::checkpoint::recovery
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Exhaustion {
    /// The object has no durable generation at all: never checkpointed (the
    /// pre-first-tick window after `CreateLayer`), or retention pruned the
    /// last one. Routine.
    EmptyListing,
    /// Durable generations exist but none of them is a
    /// [`CheckpointKind::Base`], so no lineage has a reset-decodable root.
    /// Absorbs `select_restore_source`'s tails-only arm.
    NoBase {
        /// How many generations the listing held.
        generations: usize,
        /// The listing head's sequence.
        newest_sequence: u64,
    },
    /// Every candidate was walked and at least one was abandoned, leaving
    /// nothing restorable below the cursor.
    ///
    /// Recorded only when the selection actually abandoned something. A cursor
    /// that drains after a successful candidate with an empty step-down list
    /// records no exhaustion at all — see [`LineageSelection::exhaustion`].
    AllBroken {
        /// How many candidates were tried.
        candidates: usize,
    },
    /// The budget ran out before the walk could adjudicate every listed
    /// [`CheckpointKind::Base`], so the selection cannot claim the layer is
    /// unrestorable — only that it ran out of time before it could tell.
    ///
    /// Distinct from [`Self::AllBroken`] because the two carry different
    /// evidence, and so license different answers to a consumer that
    /// distinguishes them. `AllBroken` is *conclusive* evidence that the layer
    /// is unrestorable; this one is not. It may well hold evidence about the
    /// layer — at the stock `retain_generations = 2` a selection can adjudicate
    /// one base as unusable and never read the other, which is why the WARN
    /// reports `step_downs` and `unadjudicated_bases` side by side — but the
    /// base it never read may be perfectly restorable once the store recovers,
    /// so the *unrestorable* verdict itself is unproven. The restore
    /// consumer (`select_restore_source`) acts on that split — terminal
    /// `NOT_FOUND` for `AllBroken`, retriable `UNAVAILABLE`
    /// (`CheckpointError::Transient`) for this one. That routing is that
    /// consumer's; the SR-4.1 re-base is kind-blind and refuses on either.
    ///
    /// **Why the predicate keys on bases and not on step-down reasons.**
    /// [`LineageBreak::Budget`] records a question the walk never asked, so
    /// counting Budget step-downs over-triggers: a speculative *tail* cut short
    /// above a set of fully-adjudicated broken bases leaves the verdict
    /// conclusive, because a tail can only extend a lineage, never root one.
    /// Only a `Base` can root one, so only an unread `Base` can hide a
    /// restorable answer.
    BudgetInconclusive {
        /// Candidates tried, as [`Self::AllBroken`] reports it.
        candidates: usize,
        /// How many listed `Base` generations were never adjudicated.
        unadjudicated_bases: usize,
    },
}

impl Exhaustion {
    /// The `outcome` label on `live_edit_checkpoint_restore_lineage_total`.
    ///
    /// The three conclusive variants fold onto one label — their split is
    /// diagnostic (see the type doc). [`Self::BudgetInconclusive`] gets its own,
    /// because it is the one arm the restore consumer answers differently
    /// (`select_restore_source`, client-visibly), and an operator triaging a
    /// restore wave needs to separate "these layers have no usable backup" from
    /// "this store was too slow to tell".
    ///
    /// Both spellings are members of
    /// [`CHECKPOINT_RESTORE_LINEAGE_OUTCOMES`](crate::observability::metrics::CHECKPOINT_RESTORE_LINEAGE_OUTCOMES),
    /// which stays canonical — see the module header, "No metrics".
    pub fn outcome_label(&self) -> &'static str {
        match self {
            Exhaustion::EmptyListing | Exhaustion::NoBase { .. } | Exhaustion::AllBroken { .. } => {
                "exhausted"
            }
            Exhaustion::BudgetInconclusive { .. } => "exhausted_budget",
        }
    }
}

// ── Deadline helpers ────────────────────────────────────────────────────────

/// How far ahead [`no_deadline`] places its instant.
///
/// Long enough that no gather can reach it, short enough that the arithmetic
/// cannot overflow an `Instant`.
pub const NO_DEADLINE_HORIZON: Duration = Duration::from_secs(24 * 60 * 60);

/// The `deadline` argument for a caller that has no RPC budget to spend.
///
/// The SR-4.1 chain-collapse re-base
/// ([`rebase_layer_lineage`](crate::checkpoint::rebase::rebase_layer_lineage))
/// is exactly such a caller: it runs on the checkpoint tick,
/// not under an RPC, so it must not have to synthesize an RPC-shaped budget
/// just to call [`gather_restore_lineage`]. The per-member reserve still
/// applies to that caller — it is what caps a single over-running adapter
/// fetch, and it is a separate argument for that reason.
pub fn no_deadline() -> Instant {
    Instant::now() + NO_DEADLINE_HORIZON
}

// ── The selector ────────────────────────────────────────────────────────────

/// What one probe of a generation learned. Tens of bytes; retained for the
/// whole selection.
#[derive(Debug, Clone)]
enum LinkFact {
    /// No handle at this `(seq, kind)` in the listing, or the fetch answered
    /// `Ok(None)`.
    Absent {
        /// See [`LineageBreak::LinkAbsent::at_fetch`].
        at_fetch: bool,
    },
    /// A base generation whose blob passed the per-member adapter-contract
    /// guards.
    Base { handle: CheckpointHandle },
    /// A tail generation: the authoritative header plus the artifact's
    /// `format_version` (which rides in `peek_tail_header`'s returned tuple).
    Tail {
        handle: CheckpointHandle,
        format_version: u32,
        header: TailHeader,
    },
    /// The artifact exists but can never participate in *any* lineage: its
    /// envelope did not peek, or it disagreed with its own handle. Cached as a
    /// fact so a later candidate re-reaching it costs no fetch.
    Unusable { detail: UnusableDetail },
    /// A fetch of this member overran its per-fetch cap.
    ///
    /// Not a statement about the artifact — it may be perfectly well-formed —
    /// but about the *selection*: this member has already eaten its whole
    /// allowance without answering, and a degraded backend that wedged once
    /// wedges again. Cached selection-wide for the same reason
    /// [`LinkFact::Unusable`] is, and with more at stake: candidates overlap
    /// heavily (a broken candidate's members are themselves the next
    /// candidates), so without the fact a single wedged member is re-fetched
    /// once per candidate that reaches it and each re-fetch costs another
    /// allowance. Every later candidate that reaches a wedged member steps down
    /// on [`LineageBreak::Budget`] with **zero** fetches.
    ///
    /// The allowance it ate is *not* necessarily a whole reserve: an
    /// allowanced `Tail` fetch is capped at `remaining - reserve` so it cannot
    /// spend the base's reservation (see [`gather_restore_lineage`]'s `reserve`
    /// discussion), and a tail wedged there therefore leaves one reserve behind
    /// less only the scheduling epsilon (that discussion's property (ii)),
    /// rather than consuming one. Only a `Base` fetch, or the un-gated first
    /// fetch of an untroubled selection — which is not charged against the
    /// reservation and so may wedge for a full reserve even on a `Tail` —
    /// spends up to a whole one. Every allowance is nonetheless bounded by the
    /// live remainder — see [`gather_restore_lineage`]'s property (i) for the
    /// exact deadline-precision statement.
    Wedged,
}

/// Why a [`LinkFact::Unusable`] member is unusable.
#[derive(Debug, Clone)]
enum UnusableDetail {
    Decode {
        message: String,
    },
    /// The adapter refused the member with the typed artifact-present /
    /// meta-companion-absent `Permanent`
    /// ([`CheckpointError::is_missing_meta_companion`]).
    ///
    /// A distinct variant rather than folding into [`Self::Decode`], because
    /// the two are reached by completely different code — one from
    /// `peek_tail_header`, one from `get_layer_checkpoint`'s error — and they
    /// name different faults: an unreadable envelope (bit-rot, version skew)
    /// versus a torn put whose index artifact never landed.
    ///
    /// Maps onto its own [`LineageBreak::MissingMetaCompanion`], hence its own
    /// operator-facing outcome label (`stepped_down_missing_meta`), so the
    /// metric preserves the remediation split rather than presenting a torn put
    /// as storage corruption.
    MissingMetaCompanion {
        message: String,
    },
    Mismatch {
        field: &'static str,
        expected: String,
        observed: String,
    },
}

impl UnusableDetail {
    fn into_break(self, seq: u64) -> LineageBreak {
        match self {
            UnusableDetail::Decode { message } => LineageBreak::Decode { seq, message },
            UnusableDetail::MissingMetaCompanion { message } => {
                LineageBreak::MissingMetaCompanion { seq, message }
            }
            UnusableDetail::Mismatch {
                field,
                expected,
                observed,
            } => LineageBreak::LinkMismatch {
                seq,
                field,
                expected,
                observed,
            },
        }
    }
}

/// Outcome of one probe: a fact, or "the budget would not fund this fetch".
enum MemberProbe {
    Known(LinkFact),
    BudgetExhausted { remaining: Duration },
}

/// Outcome of one candidate walk.
enum CandidateOutcome {
    Gathered(Box<GatheredLineage>),
    Broken(LineageBreak),
}

/// A resumable newest-first walk over one object's restorable generations.
///
/// Construct with [`gather_restore_lineage`] (which performs the single
/// `list_layer_checkpoints`), then drive [`LineageSelection::next_restorable`].
/// Each call returns the newest lineage that verifies **strictly below** the
/// previously-returned candidate's head sequence, so a caller whose
/// composition of one lineage failed `Permanent` can simply ask for the next —
/// which is the seam that let Task 1 (composition) and Task 2 (selection) be
/// built in parallel.
///
/// # Fetch discipline
///
/// - **One listing per selection.** It is passed into every walk and never
///   re-read mid-walk: the listing is the single source of truth, the same
///   posture Epic 3's capture unit takes.
/// - **Members only.** A walk fetches at most `1 base + the walked tails`. The
///   walk addresses each predecessor by **exact sequence**, never by listing
///   position, so SR-9.2's divergent-S fork residue — overlapping non-member
///   tails left by two leaders reading at different times — is excluded *by
///   construction*, not by a filter.
/// - **Probed once.** Every `(seq, kind)` this selection touches is probed at
///   most once: link facts (peeked header + `format_version`, kind, handle,
///   known-absence, and a fetch that overran its per-fetch cap) are cached
///   selection-wide, so a link shared by several candidates — the common case,
///   since a broken candidate's members are themselves the next candidates —
///   costs one `get_layer_checkpoint` in total, not one per candidate. The
///   wedged-fetch fact is the load-bearing one under a degraded backend: a
///   member that ate its whole allowance — at most one reserve — without
///   answering is *never* addressed again, so one wedged member costs at most
///   one reserve for the whole selection rather than one per candidate that
///   reaches it.
/// - **Bounded by the listing.** Across all candidates the total number of
///   generation *probes* is bounded by the listing length. Against SR-4.2's
///   `<= 1 base + K tails` per-lineage cost model, the whole-listing ceiling at
///   stock knobs (`retain_generations = 2`, `max_chain = 8`) is ~18 artifacts.
///
/// # Resident bytes — the ceiling, and why the cache is split
///
/// `CheckpointBlob::bytes` is a refcounted [`Bytes`], so anything held in a
/// cache is *pinned*. Caching bytes selection-wide would pin the whole listing:
/// ~18 artifacts at stock knobs — 2 bases x 100 MB plus 16 tails x 64 MiB ≈
/// **1.2 GiB**, per concurrently-restoring object. So the cache is split:
///
/// - **facts** are retained for the whole selection (tens of bytes each);
/// - **bytes** are retained only for the candidate currently being walked, and
///   dropped the moment that candidate is abandoned.
///
/// The resident-bytes ceiling is therefore **one lineage — `1 base + K tails`**
/// — which is the working set Task 1's composition needs anyway. The accepted
/// cost is that a member of an abandoned candidate which later becomes a member
/// of a *verified* candidate is fetched a second time to re-materialize its
/// bytes; its header is not re-probed.
///
/// # Structural bound, not a config bound
///
/// The walk terminates because each step must resolve to a listed handle at a
/// non-increasing sequence — strictly decreasing except for the terminal base,
/// which ends the walk — and a `visited` set forbids revisiting. It is
/// deliberately **not** bounded by `LIVE_EDIT_CHECKPOINT_TAIL_MAX_CHAIN`: see
/// the module header.
pub struct LineageSelection<'a> {
    store: &'a dyn CheckpointStore,
    object_id: u64,
    /// The single listing, newest-first with Base pinned before Tail at an
    /// equal sequence (`adapters::generation_sort_key`). NEVER re-sorted here.
    listing: Vec<CheckpointHandle>,
    /// Index of the next candidate to try.
    next_index: usize,
    /// Selection-wide link facts. See the type doc, "Resident bytes".
    facts: HashMap<(u64, CheckpointKind), LinkFact>,
    /// Bytes for the candidate currently being walked. Cleared per candidate.
    candidate_bytes: HashMap<(u64, CheckpointKind), CheckpointBlob>,
    /// Fetches issued for the candidate currently being walked.
    ///
    /// The kind-dependent allowance is skipped while this is zero **and**
    /// nothing has been abandoned yet, so an untroubled base-newest selection
    /// costs exactly one un-gated fetch — the pre-Epic-4 cost, with no budget
    /// precondition on it, at up to a full reserve and bounded only by the live
    /// remainder. Once the selection has stepped down at least once, first
    /// fetches are allowanced too, and from then on the allowance is what funds
    /// the floor answer: an allowanced `Tail` fetch may never take the
    /// remainder below one reserve, so whatever the *allowanced* walk above it
    /// does, the first `Base` the walk reaches is funded with a **full**
    /// `reserve` — the tail's cap is taken against the *live* remainder, so the
    /// LIST and every other non-fetch cost above it are already netted out of
    /// it and are not charged to the base a second time. The one un-gated first
    /// fetch is the stated exception: when it lands on a `Tail` and that tail
    /// wedges, the base behind it gets `min(reserve, remaining)` instead —
    /// short of a full reserve by the LIST plus that fetch's own epsilon. See
    /// [`gather_restore_lineage`]'s `reserve` discussion for the full contract,
    /// its two properties, and its stated precondition.
    candidate_fetches: usize,
    /// Every abandoned candidate, in the order they were abandoned.
    step_downs: Vec<StepDownRecord>,
    /// Set once the selection has nothing left to offer.
    exhaustion: Option<Exhaustion>,
    /// Candidates tried so far, for the exhaustion record.
    candidates_tried: usize,
    deadline: Instant,
    reserve: Duration,
}

/// Begin a newest-first restore selection for `object_id`.
///
/// Performs the one `list_layer_checkpoints` and returns the resumable cursor;
/// drive it with [`LineageSelection::next_restorable`].
///
/// # Signature — a cross-task contract
///
/// The parameters are `&dyn CheckpointStore`, `object_id`, and the two
/// budget arguments below. There is deliberately **no** `RecoveryHook`, no
/// `LayerStore`, no [`Config`](crate::config::Config), and no metrics handle:
/// the SR-4.1 chain-collapse re-base
/// ([`rebase_layer_lineage`](crate::checkpoint::rebase::rebase_layer_lineage))
/// calls this from the checkpoint tick, where none of those are the same
/// objects.
///
/// # `deadline` and `reserve` — cooperative, under a preemptive backstop
///
/// `RecoveryHook::try_restore_layer` wraps its whole selection in one
/// preemptive `tokio::time::timeout` sized by
/// `Config::checkpoint_recovery_fetch_budget()` (60 s at stock defaults). That
/// wrapper stays; `deadline` is the *cooperative* half, and its whole purpose
/// is that **running out of time becomes a step-down rather than an error**.
/// Without it, a healthy-but-slow multi-tail lineage would elapse the outer
/// timeout, map to `CheckpointError::Transient` → `RecoveryOutcome::Error`, and
/// strand a perfectly restorable base sitting one generation behind it — the
/// exact stranded-recovery class SR-2.5 and FR-6 forbid, reintroduced through
/// the timeout instead of the decoder.
///
/// `reserve` is the budget one generation fetch is assumed to need
/// (`Config::checkpoint_request_timeout_ms` is the honest unit; it is passed in
/// because this module reads no `Config`). It does **two** jobs, and both are
/// required:
///
/// 1. **The base's reservation.** One `reserve` of the budget is held back for
///    the floor answer's fetch, and only a [`CheckpointKind::Base`] member may
///    spend it. Concretely, every member fetch is capped at an **allowance**
///    derived from the live remainder rather than gated against a threshold,
///    over three arms:
///
///    - The **un-gated first fetch** — `min(reserve, remaining)`, taken while
///      the current candidate has fetched nothing *and* the selection has
///      abandoned nothing. "Un-gated" means it is not charged against the
///      base's reservation and carries no budget precondition — whatever its
///      kind, it may spend up to a full `reserve`. It is *not* unbounded: the
///      `min` keeps it under the live remainder like every other arm.
///    - `Base` — `min(reserve, remaining)`. Never declined while any budget is
///      left at all.
///    - `Tail` — `min(reserve, remaining - reserve)`, saturating at zero. A
///      tail is *speculative*: it can only improve on the floor answer, so it
///      may be cut short — even when the adapter is merely slow rather than
///      broken — or declined outright, to protect the reservation.
///
///    An allowance of zero is the abandonment: the candidate records
///    [`LineageBreak::Budget`] and the walk steps down, a verdict and never an
///    error.
///
///    **What this guarantees about the floor answer.** Two properties fall out,
///    and together they are the contract:
///
///    (i) **The deadline-precision statement.** This is its canonical site;
///    every other mention of it in this crate is a pointer here and states no
///    independent claim, so that the imprecise absolute form cannot be
///    reintroduced at one site alone.
///
///    Every allowance, on **all three** arms, is `<= remaining` *at the moment
///    `remaining` is sampled*, and — this is the part that has to be absolute —
///    every one is **spent as an instant, not a duration**. `fetch_member`
///    derives a `bound` of `min(now + reserve, deadline)` (base, and the
///    un-gated first fetch) or `min(now + reserve, deadline - reserve)`
///    (allowanced tail) and passes it to `tokio::time::timeout_at`. Those are
///    the same numbers the bullets above give as durations, but anchored on the
///    cooperative `deadline` itself, so **no fetch's timer can elapse after
///    that deadline** — not merely "is not scheduled to". A duration-anchored
///    `timeout(allowance, ..)` could not say that: it re-anchors where the
///    timer is constructed, a few statements after `remaining` was sampled, so
///    a fetch clamped to the remainder elapsed at `deadline + delta` and could
///    outlive the outer preemptive wrapper's own deadline.
///
///    That is the whole of what the accounting buys, and stating more than it
///    is how the imprecise form gets back in. It is **not** a strict ordering
///    against the outer preemptive wrapper: the cooperative deadline and the
///    wrapper's are both `tokio::time::Instant::now()` plus the same budget,
///    sampled two statements apart — equal outright under
///    `tokio::time::pause()` (which is every `start_paused` fence below), and
///    nanoseconds apart on a live clock. At that tie the cooperative verdict
///    wins for a reason outside this crate's arithmetic: `tokio::time::Timeout`
///    polls its inner future before its own delay, and the walk's post-elapse
///    work is synchronous, so the step-down is ready in the same poll. See
///    `RecoveryHook::try_restore_layer`'s tie-break note in `recovery.rs` for
///    that argument, for its tokio-version upgrade obligation, for what the two
///    Group C cases below do and do not fence, and for its one scoping caveat —
///    the bound covers when a FETCH ends, so a **composed** lineage's
///    post-fetch `spawn_blocking` composition can still overrun the wrapper and
///    strand as the retriable `Transient`, while the `Ok(None)` step-down and a
///    **base-only** restore cannot. Within that scope the outer preemptive
///    timeout stays a backstop rather than the mechanism. The b.gg7
///    head-of-line ceiling is untouched — the guarantee is bought with
///    *accounting*, never with a bigger number.
///
///    (ii) An **allowanced** `Tail` fetch — every `Tail` but one taken on the
///    un-gated arm — cannot take `remaining` below one `reserve`: its cap is
///    `remaining - reserve`, sampled from the *live* remainder, so the LIST and
///    every other un-capped cost above it are already netted out before the cap
///    is taken and cannot be charged to the base a second time. So however many
///    such members wedge above it, the first `Base` the walk reaches is funded
///    with a **full** `reserve` — less only the scheduling epsilon between that
///    tail's return and the base's own sample of `remaining`. The one un-gated
///    fetch a selection may take is the stated exception: it is not reserved
///    against, so when it lands on a `Tail` and that tail wedges, the base
///    behind it is funded with `min(reserve, remaining)` for a remainder short
///    of a full reserve by the LIST plus that first fetch's own epsilon. State
///    the property with that exception spelled out, and **not** as a flat "the
///    base is always guaranteed a full reserve": on that one arm the residue is
///    real, it is just not proportional to how badly the adapter is behaving,
///    which is what the pre-b.bu8 arithmetic got wrong (one wedged member spent
///    a full reserve *plus* the residue, dropping the remainder under the
///    threshold the base's own fetch was then gated on, and the selection
///    exhausted with a healthy base one step below it).
///
///    A **stated precondition**, deliberately not validated here:
///    `budget >= 2 x reserve`. Every shipped caller satisfies it structurally —
///    `RecoveryHook::select_restore_source` passes
///    `max(rpc_deadline_ms, 2 x request_timeout)` against a `request_timeout`
///    reserve, the SR-4.1 re-base passes `(max_chain + 2) x reserve`, and
///    [`no_deadline()`]'s 24 h horizon dwarfs any reserve. Below it tails are
///    **cut short, not eliminated**: across
///    `reserve < budget < 2 x reserve`, a fast enough first fetch leaves
///    `remaining > reserve` and the `Tail` allowance
///    `min(reserve, remaining - reserve)` is positive, so tails are still
///    fetched, just at a truncated allowance. `Tail` allowances collapse to
///    zero — the selection degrading to base-only fetches — only once
///    `remaining <= reserve`. Both are a safe direction; neither is the
///    intended one.
///
///    The precondition is on the budget, not on the remainder the walk actually
///    sees, and the gap between them is the LIST — which is bounded only by the
///    adapter's own enumeration ceiling (540 s at stock knobs), not by this
///    deadline. A slow enough LIST therefore leaves `remaining < reserve`
///    before the first fetch is even attempted, and property (i) is what covers
///    it: the un-gated arm's `min` clamps that first fetch to the remainder
///    instead of a flat reserve, so it stays bounded by the live remainder —
///    see property (i) above for the exact deadline-precision statement. What
///    follows is the degenerate outcome — if that
///    clamped fetch wedges, the remainder saturates to zero and the `Base`
///    behind it is declined with a zero allowance, i.e. one more
///    [`LineageBreak::Budget`] step-down and an exhaustion, never an error; and
///    if it instead fails
///    a hair inside its allowance, the base is funded with a near-zero
///    allowance and its doomed fetch records a `Wedged` fact against what may
///    be a perfectly healthy generation. Both are verdicts, both are bounded,
///    and neither is worth spending budget the LIST has already consumed.
///
///    **What that degenerate case reports to the caller.** It is not
///    classification-neutral, and the classification is the
///    [`Exhaustion`] variant's job rather than this arithmetic's. Worked at
///    stock knobs: a 45 s LIST under a 60 s budget with a 30 s reserve leaves
///    15 s, arm 1's `min` clamps the first fetch to it, that fetch wedges at
///    the cooperative deadline, `remaining` saturates to zero, and every later
///    candidate — the base among them — declines at a zero allowance. The
///    selection ends at [`Exhaustion::BudgetInconclusive`], because the listed
///    base was never adjudicated, and
///    `RecoveryHook::select_restore_source` turns that into
///    `CheckpointError::Transient` → `RecoveryOutcome::Error` → `UNAVAILABLE`,
///    which the standard client retry policy retries once the store recovers.
///    An exhaustion that DID adjudicate every listed base is
///    [`Exhaustion::AllBroken`] and still ends at the terminal
///    `Ok(None)` → `NotRecoverable` → `NOT_FOUND`, because there the selection
///    holds positive evidence about the layer rather than about the clock. So
///    the clamp costs the caller a retry on this trace, not a restorable layer
///    written off — which is what makes the clamp's own guarantee (the
///    reservation is real, and no fetch outlives the cooperative deadline)
///    affordable.
///
///    On the arm where nothing has gone wrong the floor is unconditional and
///    unchanged: an untroubled selection whose newest generation is a base pays
///    exactly one un-gated fetch of up to a full `reserve` — the `min` above is
///    a no-op whenever the LIST left a full reserve standing, which is every
///    non-degenerate selection — which is the pre-Epic-4 cost exactly.
/// 2. **A per-fetch cap.** Every individual `get_layer_checkpoint` is wrapped
///    in its own `tokio::time::timeout_at(bound, ..)` — the allowance from (1)
///    expressed as an instant (property (i)), at most one `reserve` wide and
///    never past the cooperative deadline. This is **not** redundant with the
///    accounting: the StorageApi adapter's *enforced* per-call ceiling is
///    `request_timeout x 3 retries x 2 auth attempts` = 180 s, which is 6x a
///    30 s reserve, so on a degraded backend a single uncapped fetch outlives
///    the whole 60 s outer budget and no cooperative check ever gets to run.
///    The outer preemptive timeout would fire first and reproduce precisely the
///    `Transient` → `Error` stranding this design exists to eliminate. Capping
///    each fetch converts that into a budget step-down.
///
/// Both are safe to abandon mid-flight for the reason the hook's own comment
/// already gives: this is a pure read — LIST plus GETs, no PUT, no DELETE — so
/// dropping a future cancels an in-flight request and nothing else.
///
/// Callers with no RPC budget pass [`no_deadline()`].
///
/// # Errors
///
/// Store errors propagate **verbatim**: `Transient`, `Permanent`,
/// `NotImplemented`. A broken lineage is never an error — it is a verdict.
pub async fn gather_restore_lineage(
    store: &dyn CheckpointStore,
    object_id: u64,
    deadline: Instant,
    reserve: Duration,
) -> Result<LineageSelection<'_>, CheckpointError> {
    let listing = store.list_layer_checkpoints(object_id).await?;
    Ok(LineageSelection {
        store,
        object_id,
        listing,
        next_index: 0,
        facts: HashMap::new(),
        candidate_bytes: HashMap::new(),
        candidate_fetches: 0,
        step_downs: Vec::new(),
        exhaustion: None,
        candidates_tried: 0,
        deadline,
        reserve,
    })
}

/// Hand-written because the `&dyn CheckpointStore` field is not `Debug`, and
/// because the byte scratch must never render: a derived impl would dump a
/// multi-MB `Bytes` into a log line or a test failure message.
impl std::fmt::Debug for LineageSelection<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LineageSelection")
            .field("object_id", &self.object_id)
            .field("generations", &self.listing.len())
            .field("next_index", &self.next_index)
            .field("probed", &self.facts.len())
            .field("resident_blobs", &self.candidate_bytes.len())
            .field("step_downs", &self.step_downs)
            .field("exhaustion", &self.exhaustion)
            .finish()
    }
}

impl<'a> LineageSelection<'a> {
    /// The object this selection is for.
    pub fn object_id(&self) -> u64 {
        self.object_id
    }

    /// Every candidate abandoned so far, oldest abandonment first.
    pub fn step_downs(&self) -> &[StepDownRecord] {
        &self.step_downs
    }

    /// Why the selection ended **badly**, if it did.
    ///
    /// `None` is not "the selection has not finished": a cursor drained after a
    /// successful candidate — the caller asked for one more generation below
    /// the one it took, and there is none — answers `Ok(None)` with no
    /// exhaustion recorded, because nothing went wrong. A `Some` here is
    /// therefore exactly the "the selection ended with nothing in hand" signal
    /// — the verdict, and nothing beyond it. It is **not** a single metric
    /// outcome and **not** a single routing:
    ///
    /// - **Label it** through [`Exhaustion::outcome_label`], never by folding
    ///   every `Some` onto `exhausted`. That accessor folds the three
    ///   conclusive variants onto `exhausted` and mints
    ///   [`Exhaustion::BudgetInconclusive`]'s own `exhausted_budget`.
    /// - **Route it** in the consumer. The restore consumer
    ///   (`select_restore_source`) answers `BudgetInconclusive` with a retriable
    ///   `CheckpointError::Transient` and the conclusive variants with a
    ///   terminal `NotFound`; the SR-4.1 re-base is kind-blind and refuses on
    ///   every exhaustion alike. Both routings are those consumers', not
    ///   properties of the variant — this module holds no caller knowledge (see
    ///   [`gather_restore_lineage`], "Signature — a cross-task contract").
    pub fn exhaustion(&self) -> Option<&Exhaustion> {
        self.exhaustion.as_ref()
    }

    /// Record that the caller's composition classified the lineage headed at
    /// `head_seq` as `Permanent` (Task 1's per-generation verdict).
    ///
    /// Exists so a composition failure steps down through the *same* vocabulary
    /// as a broken link: the caller records, this module spells. Call it before
    /// the next [`Self::next_restorable`].
    pub fn record_compose_failure(&mut self, head_seq: u64, head_kind: CheckpointKind) {
        let reason = LineageBreak::Compose { head_seq };
        self.warn_step_down(head_seq, head_kind, &reason);
        self.step_downs.push(StepDownRecord {
            head_seq,
            head_kind,
            reason,
        });
    }

    /// The newest lineage that verifies at or below the current cursor, or
    /// `Ok(None)` when nothing is left.
    ///
    /// Calling it again resumes **strictly below** the previously-returned
    /// candidate's `head_seq`: a generation at the same sequence restores to
    /// the same state, so re-offering it would not be a step *down*.
    ///
    /// # Errors
    ///
    /// Store errors propagate verbatim. Every lineage defect is a step-down,
    /// never an error.
    pub async fn next_restorable(&mut self) -> Result<Option<GatheredLineage>, CheckpointError> {
        // Idempotent re-entry. An exhausted selection has nothing left to
        // offer, and re-deriving the verdict would re-emit its WARN — turning
        // one operator signal into one per extra call by a caller that merely
        // asked twice.
        if self.exhaustion.is_some() {
            return Ok(None);
        }
        if self.listing.is_empty() {
            return Ok(self.exhaust(Exhaustion::EmptyListing));
        }
        if !self
            .listing
            .iter()
            .any(|handle| handle.kind == CheckpointKind::Base)
        {
            // Absorbs `select_restore_source`'s tails-only arm rather than
            // deleting its signal: the base is the only reset-decodable
            // artifact, so a tails-only listing is unrestorable under composed
            // restore too.
            let exhaustion = Exhaustion::NoBase {
                generations: self.listing.len(),
                newest_sequence: self.listing[0].seq(),
            };
            return Ok(self.exhaust(exhaustion));
        }

        while self.next_index < self.listing.len() {
            let index = self.next_index;
            self.next_index += 1;
            self.candidates_tried += 1;

            // Bytes are per-candidate (see the type doc, "Resident bytes").
            self.candidate_bytes.clear();
            self.candidate_fetches = 0;

            let head = self.listing[index].clone();
            let outcome = self.walk_candidate(&head).await?;
            match outcome {
                CandidateOutcome::Gathered(lineage) => {
                    let lineage = *lineage;
                    tracing::debug!(
                        object_id = self.object_id,
                        head_seq = lineage.head_seq,
                        base_snap_seq = lineage.base_snap_seq(),
                        chain_len = lineage.chain_len(),
                        outcome = lineage.outcome_label(),
                        "restore selection resolved a lineage"
                    );
                    // Resume strictly below this candidate's head sequence.
                    let head_seq = lineage.head_seq;
                    while self
                        .listing
                        .get(self.next_index)
                        .is_some_and(|handle| handle.seq() >= head_seq)
                    {
                        self.next_index += 1;
                    }
                    self.candidate_bytes.clear();
                    return Ok(Some(lineage));
                }
                CandidateOutcome::Broken(reason) => {
                    self.warn_step_down(head.seq(), head.kind, &reason);
                    self.step_downs.push(StepDownRecord {
                        head_seq: head.seq(),
                        head_kind: head.kind,
                        reason,
                    });
                    // Drop the abandoned candidate's bytes immediately; the
                    // facts learned along the way are what survive.
                    self.candidate_bytes.clear();
                }
            }
        }

        // The cursor ran past the last candidate. WHY it did is the whole
        // distinction here, and only the step-down record can tell:
        //
        // - With nothing abandoned, the caller simply drained a selection that
        //   already answered — asked for one more generation below the one it
        //   took, and there is none. Nothing is wrong, so nothing is recorded:
        //   no `Exhaustion` (the caller's own outcome was `gathered_*`, and an
        //   exhaustion record would have Task 3's hook report `exhausted` for a
        //   layer that restored) and no WARN (claiming "the layer is
        //   unrecoverable" about a layer that was just recovered is a false
        //   page).
        // - With something abandoned, every candidate really was walked and
        //   really did break, and the operator wants to hear about it.
        if self.step_downs.is_empty() {
            tracing::debug!(
                object_id = self.object_id,
                candidates = self.candidates_tried,
                "restore selection has no generation below the one it already resolved"
            );
            return Ok(None);
        }
        // WHICH exhaustion is the second distinction, and unlike the first it
        // carries evidence a consumer may act on rather than only diagnostics:
        // a selection that never adjudicated some listed base cannot claim the
        // layer is unrestorable, only that it ran out of time before it could
        // tell. See `Exhaustion`'s type doc for the exact rule, and for which
        // consumer acts on the split.
        let unadjudicated_bases = self.unadjudicated_bases();
        let exhaustion = if unadjudicated_bases > 0 {
            Exhaustion::BudgetInconclusive {
                candidates: self.candidates_tried,
                unadjudicated_bases,
            }
        } else {
            Exhaustion::AllBroken {
                candidates: self.candidates_tried,
            }
        };
        Ok(self.exhaust(exhaustion))
    }

    /// How many listed [`CheckpointKind::Base`] generations this selection never
    /// got an answer about. The [`Exhaustion::BudgetInconclusive`] predicate.
    ///
    /// A pure read over the facts the walk already accumulated — it introduces
    /// no state and costs no store call. Adjudicated is
    /// [`LinkFact::Base`] (fetched and classified), [`LinkFact::Unusable`] (the
    /// artifact can never participate in a lineage) or
    /// `LinkFact::Absent { at_fetch: true }` (gone at fetch — the
    /// contract-documented concurrent-prune race, which is positive evidence
    /// that the generation is not there). The `at_fetch` field is matched
    /// exactly rather than wildcarded: `at_fetch: false` says only that the
    /// listing held no handle for the key, which under a key taken *from* the
    /// listing is unreachable today (it is inserted on the one `facts_of` arm
    /// where [`Self::handle_at`] finds no handle), so the narrow pattern costs
    /// nothing now and makes the unreachable direction count as unadjudicated —
    /// the fail-safe one, retriable rather than terminal. Unadjudicated is a
    /// *missing* key (the fetch was declined at a zero allowance, so `facts_of`
    /// never inserted) or [`LinkFact::Wedged`] (the fetch ate its allowance
    /// without answering). Only those two leave a base that might still restore.
    /// [`LinkFact::Tail`] cannot appear under a `Base` key — [`Self::classify`]'s
    /// kind guard turns a kind-disagreeing blob into an `Unusable` mismatch —
    /// and were a future change to make it reachable it would count as
    /// unadjudicated, which is the safe direction (retriable, not terminal).
    ///
    /// Counted over distinct sequences: a non-conforming adapter that lists one
    /// `(seq, Base)` twice must not inflate the number an operator reads off the
    /// WARN.
    fn unadjudicated_bases(&self) -> usize {
        let mut seen: HashSet<u64> = HashSet::new();
        self.listing
            .iter()
            .filter(|handle| handle.kind == CheckpointKind::Base)
            .filter(|handle| seen.insert(handle.seq()))
            .filter(|handle| {
                !matches!(
                    self.facts.get(&(handle.seq(), CheckpointKind::Base)),
                    Some(
                        LinkFact::Base { .. }
                            | LinkFact::Unusable { .. }
                            | LinkFact::Absent { at_fetch: true }
                    )
                )
            })
            .count()
    }

    /// Record the exhaustion verdict, trace it, and answer `None`.
    fn exhaust(&mut self, exhaustion: Exhaustion) -> Option<GatheredLineage> {
        match &exhaustion {
            // Routine and expected: the layer was never checkpointed (the
            // pre-first-tick window after CreateLayer), or retention pruned its
            // last generation. `debug!` rather than `warn!` follows this
            // module's established severity convention — an *expected* outcome
            // does not get a WARN — and matches the pre-Epic-4 behaviour, which
            // logged nothing at all on this arm. The two arms below are the
            // ones that mean something is wrong.
            Exhaustion::EmptyListing => tracing::debug!(
                object_id = self.object_id,
                outcome = exhaustion.outcome_label(),
                "restore selection found no durable generation for this object"
            ),
            Exhaustion::NoBase {
                generations,
                newest_sequence,
            } => tracing::warn!(
                object_id = self.object_id,
                generations = *generations,
                newest_sequence = *newest_sequence,
                outcome = exhaustion.outcome_label(),
                "restore selection found no base generation for this object; every durable \
                 generation is a tail, and a lineage has no reset-decodable root without one. \
                 Not producible by the shipped write paths — suspect manual pruning of the base \
                 or partial checkpoint-store loss. Reporting the layer as unrecoverable, exactly \
                 as for an empty store"
            ),
            Exhaustion::AllBroken { candidates } => tracing::warn!(
                object_id = self.object_id,
                candidates = *candidates,
                step_downs = self.step_downs.len(),
                outcome = exhaustion.outcome_label(),
                "restore selection stepped down through every candidate generation without \
                 finding a verifiable lineage; the layer is unrecoverable despite holding \
                 durable checkpoints"
            ),
            // Deliberately NOT the AllBroken wording. That arm claims the layer
            // is unrecoverable; this one cannot, because at least one base was
            // never read. Saying so would page an operator to a data-loss
            // investigation for an adapter that was merely slow. The message
            // states this selection's EVIDENCE, not any consumer's answer —
            // this module has more than one consumer and holds no knowledge of
            // what any of them does with the verdict (see
            // [`gather_restore_lineage`], "Signature — a cross-task contract").
            Exhaustion::BudgetInconclusive {
                candidates,
                unadjudicated_bases,
            } => tracing::warn!(
                object_id = self.object_id,
                candidates = *candidates,
                unadjudicated_bases = *unadjudicated_bases,
                step_downs = self.step_downs.len(),
                outcome = exhaustion.outcome_label(),
                "restore selection ran out of budget before it could read every full generation \
                 for this object; this selection holds no conclusive evidence that the layer is \
                 unrestorable, so it must not be read as reporting the layer unrecoverable. \
                 Remediation: adapter latency — cross-check \
                 `checkpoint_adapter_ceiling_elapsed_total{{phase=\"call\"}}`"
            ),
        }
        self.exhaustion = Some(exhaustion);
        None
    }

    fn warn_step_down(&self, head_seq: u64, head_kind: CheckpointKind, reason: &LineageBreak) {
        let (expected, observed) = reason.expectation().unwrap_or(("", ""));
        // WARN, not debug: a step-down means DR is about to serve older state
        // than the store nominally holds. That is worth a human look even
        // though the restore itself still succeeds.
        tracing::warn!(
            object_id = self.object_id,
            head_seq,
            head_kind = head_kind.as_str(),
            link_seq = reason.link_seq(),
            field = reason.field(),
            expected,
            observed,
            detail = ?reason,
            outcome = reason.reason_label(),
            "restore selection abandoned a candidate lineage and stepped down"
        );
    }

    // ── The head-down membership walk ───────────────────────────────────────

    /// Walk one candidate's lineage head-down and materialize it, or report why
    /// it cannot be.
    async fn walk_candidate(
        &mut self,
        head: &CheckpointHandle,
    ) -> Result<CandidateOutcome, CheckpointError> {
        // Phase 1 — verify the links over FACTS, fetching only what is not
        // already known. A candidate that shares a broken link with an
        // already-abandoned candidate resolves here with zero fetches.
        let chain = match self.verify_links(head).await? {
            Ok(chain) => chain,
            Err(reason) => return Ok(CandidateOutcome::Broken(reason)),
        };

        // Phase 2 — materialize. Members probed during THIS candidate's phase 1
        // already have their bytes parked; a member whose facts were learned
        // under an abandoned candidate is fetched again here (see the type
        // doc's resident-bytes ceiling).
        match self.materialize(chain).await? {
            Ok(lineage) => Ok(CandidateOutcome::Gathered(Box::new(lineage))),
            Err(reason) => Ok(CandidateOutcome::Broken(reason)),
        }
    }

    /// Phase 1. Returns the chain **newest-first**, terminated by its base.
    async fn verify_links(
        &mut self,
        head: &CheckpointHandle,
    ) -> Result<Result<Vec<LinkFact>, LineageBreak>, CheckpointError> {
        let mut visited: HashSet<(u64, CheckpointKind)> = HashSet::new();
        let mut chain: Vec<LinkFact> = Vec::new();

        let head_fact = match self.facts_of(head.seq(), head.kind).await? {
            MemberProbe::BudgetExhausted { remaining } => {
                return Ok(Err(self.budget_break(head.seq(), remaining)))
            }
            MemberProbe::Known(fact) => fact,
        };
        visited.insert((head.seq(), head.kind));

        // A base-headed candidate is a one-member lineage — identical work to
        // the pre-Epic-4 path.
        let (lineage_root, mut current) = match &head_fact {
            LinkFact::Absent { at_fetch } => {
                return Ok(Err(LineageBreak::LinkAbsent {
                    seq: head.seq(),
                    kind: head.kind,
                    at_fetch: *at_fetch,
                }))
            }
            LinkFact::Unusable { detail } => return Ok(Err(detail.clone().into_break(head.seq()))),
            // Reached only from the fact cache — a fetch that overran its
            // allowance under an earlier candidate. The verdict is the same
            // Budget step-down that candidate recorded, at zero fetches, but
            // the remainder is measured fresh here (see `remaining_budget`).
            LinkFact::Wedged => {
                return Ok(Err(self.budget_break(head.seq(), self.remaining_budget())))
            }
            LinkFact::Base { .. } => {
                chain.push(head_fact);
                return Ok(Ok(chain));
            }
            LinkFact::Tail { header, .. } => (header.base_snap_seq, *header),
        };
        chain.push(head_fact);

        loop {
            // The predecessor sits at EXACTLY `from_seq` — the exclusive lower
            // bound of the current member's `(from_seq, to_seq]` span, so no
            // `+1` / `-1` appears anywhere in this comparison (b.rrx). Its kind
            // is `Base` iff that sequence IS the lineage root, which makes a
            // base and a tail colliding on one sequence unambiguous.
            let want_seq = current.from_seq;
            let want_kind = if want_seq == lineage_root {
                CheckpointKind::Base
            } else {
                CheckpointKind::Tail
            };

            // Structural termination: sequences are non-increasing, and
            // equality is admissible only for the terminal base (a zero-length
            // tail whose `from_seq == to_seq` sitting directly on its root).
            if want_seq > current.to_seq
                || (want_seq == current.to_seq && want_kind == CheckpointKind::Tail)
            {
                return Ok(Err(LineageBreak::LinkMismatch {
                    seq: current.to_seq,
                    field: "from_seq",
                    expected: format!("< {}", current.to_seq),
                    observed: want_seq.to_string(),
                }));
            }
            if !visited.insert((want_seq, want_kind)) {
                return Ok(Err(LineageBreak::LinkMismatch {
                    seq: want_seq,
                    field: "lineage_cycle",
                    expected: "an unvisited generation".to_string(),
                    observed: format!("{} at sequence {want_seq}", want_kind.as_str()),
                }));
            }

            let fact = match self.facts_of(want_seq, want_kind).await? {
                MemberProbe::BudgetExhausted { remaining } => {
                    return Ok(Err(self.budget_break(want_seq, remaining)))
                }
                MemberProbe::Known(fact) => fact,
            };

            match &fact {
                LinkFact::Absent { at_fetch } => {
                    return Ok(Err(LineageBreak::LinkAbsent {
                        seq: want_seq,
                        kind: want_kind,
                        at_fetch: *at_fetch,
                    }))
                }
                LinkFact::Unusable { detail } => {
                    return Ok(Err(detail.clone().into_break(want_seq)))
                }
                // As above: a member already known to have eaten its allowance
                // without answering is skipped, not re-fetched. Remainder
                // measured fresh.
                LinkFact::Wedged => {
                    return Ok(Err(self.budget_break(want_seq, self.remaining_budget())))
                }
                LinkFact::Base { handle } => {
                    // `facts_of` already proved `handle.seq() == want_seq`, and
                    // `want_seq == lineage_root` is how this arm was selected,
                    // so the terminal-base check is discharged by construction.
                    debug_assert_eq!(handle.seq(), lineage_root);
                    chain.push(fact);
                    return Ok(Ok(chain));
                }
                LinkFact::Tail { header, .. } => {
                    if header.base_snap_seq != lineage_root {
                        return Ok(Err(LineageBreak::LinkMismatch {
                            seq: want_seq,
                            field: "base_snap_seq",
                            expected: lineage_root.to_string(),
                            observed: header.base_snap_seq.to_string(),
                        }));
                    }
                    current = *header;
                    chain.push(fact);
                }
            }
        }
    }

    /// Phase 2. Consumes the newest-first chain and produces the lineage with
    /// its tails reversed into replay order.
    async fn materialize(
        &mut self,
        chain: Vec<LinkFact>,
    ) -> Result<Result<GatheredLineage, LineageBreak>, CheckpointError> {
        let mut tails: Vec<GatheredTail> = Vec::with_capacity(chain.len().saturating_sub(1));
        let mut base: Option<CheckpointBlob> = None;
        let mut head_seq: Option<u64> = None;

        for fact in chain {
            let (seq, kind) = match &fact {
                LinkFact::Base { handle } => (handle.seq(), CheckpointKind::Base),
                LinkFact::Tail { handle, .. } => (handle.seq(), CheckpointKind::Tail),
                // `verify_links` only ever pushes Base / Tail facts.
                LinkFact::Absent { .. } | LinkFact::Unusable { .. } | LinkFact::Wedged => continue,
            };
            let blob = match self.blob_of(seq, kind).await? {
                Ok(blob) => blob,
                Err(reason) => return Ok(Err(reason)),
            };
            match fact {
                LinkFact::Base { .. } => base = Some(blob),
                LinkFact::Tail {
                    handle,
                    header,
                    format_version,
                } => {
                    head_seq.get_or_insert(header.to_seq);
                    tails.push(GatheredTail {
                        handle,
                        bytes: blob.bytes,
                        header,
                        format_version,
                        meta: blob.meta,
                    });
                }
                LinkFact::Absent { .. } | LinkFact::Unusable { .. } | LinkFact::Wedged => {
                    unreachable!(
                        "verify_links pushes only Base / Tail facts; the match above already \
                         `continue`d on the others"
                    )
                }
            }
        }

        let Some(base) = base else {
            // Unreachable: `verify_links` returns `Ok` only when it terminated
            // on a Base. Guarded rather than `expect`ed because the cost is one
            // branch on a cold path and the alternative is a panic inside DR.
            return Ok(Err(LineageBreak::LinkMismatch {
                seq: 0,
                field: "lineage_root",
                expected: "a terminal base generation".to_string(),
                observed: "none".to_string(),
            }));
        };

        // The walk produced tails newest-first; replay order is oldest-first.
        tails.reverse();
        let head_seq = head_seq.unwrap_or_else(|| base.handle.seq());
        Ok(Ok(GatheredLineage {
            base,
            tails,
            head_seq,
        }))
    }

    // ── Probing ─────────────────────────────────────────────────────────────

    /// Facts for one generation, fetching only if they are not already known.
    ///
    /// This is the probe-dedup point: a `(seq, kind)` already known `Absent` or
    /// `Unusable` costs **no** store call, however many candidates re-reach it.
    async fn facts_of(
        &mut self,
        seq: u64,
        kind: CheckpointKind,
    ) -> Result<MemberProbe, CheckpointError> {
        if let Some(fact) = self.facts.get(&(seq, kind)) {
            return Ok(MemberProbe::Known(fact.clone()));
        }
        let Some(handle) = self.handle_at(seq, kind) else {
            let fact = LinkFact::Absent { at_fetch: false };
            self.facts.insert((seq, kind), fact.clone());
            return Ok(MemberProbe::Known(fact));
        };
        self.fetch_member(&handle).await
    }

    /// The blob for one already-verified member: parked bytes if this
    /// candidate's own walk fetched it, otherwise one re-fetch.
    async fn blob_of(
        &mut self,
        seq: u64,
        kind: CheckpointKind,
    ) -> Result<Result<CheckpointBlob, LineageBreak>, CheckpointError> {
        if let Some(blob) = self.candidate_bytes.remove(&(seq, kind)) {
            return Ok(Ok(blob));
        }
        let Some(handle) = self.handle_at(seq, kind) else {
            return Ok(Err(LineageBreak::LinkAbsent {
                seq,
                kind,
                at_fetch: false,
            }));
        };
        match self.fetch_member(&handle).await? {
            MemberProbe::BudgetExhausted { remaining } => {
                Ok(Err(self.budget_break(seq, remaining)))
            }
            MemberProbe::Known(fact) => match fact {
                LinkFact::Absent { at_fetch } => Ok(Err(LineageBreak::LinkAbsent {
                    seq,
                    kind,
                    at_fetch,
                })),
                LinkFact::Unusable { detail } => Ok(Err(detail.into_break(seq))),
                // Not produced here — `fetch_member` answers an overrun with
                // `BudgetExhausted`, handled above — but spelled as a verdict
                // rather than an `unreachable!`, because the cost is one branch
                // on a cold path and the alternative is a panic inside DR.
                LinkFact::Wedged => Ok(Err(self.budget_break(seq, self.remaining_budget()))),
                LinkFact::Base { .. } | LinkFact::Tail { .. } => {
                    match self.candidate_bytes.remove(&(seq, kind)) {
                        Some(blob) => Ok(Ok(blob)),
                        // Unreachable: a usable fact minted by `fetch_member`
                        // always parks its bytes.
                        None => Ok(Err(LineageBreak::LinkAbsent {
                            seq,
                            kind,
                            at_fetch: true,
                        })),
                    }
                }
            },
        }
    }

    fn handle_at(&self, seq: u64, kind: CheckpointKind) -> Option<CheckpointHandle> {
        self.listing
            .iter()
            .find(|handle| handle.seq() == seq && handle.kind == kind)
            .cloned()
    }

    /// The live remaining budget, saturating at zero once the deadline passes.
    ///
    /// Read at the moment a [`LineageBreak::Budget`] is minted, never cached:
    /// the arms that mint one off a *cached* [`LinkFact::Wedged`] have no fresh
    /// measurement of their own, and a flat zero there would understate a
    /// remainder that still holds the base's reservation.
    fn remaining_budget(&self) -> Duration {
        self.deadline.saturating_duration_since(Instant::now())
    }

    fn budget_break(&self, seq: u64, remaining: Duration) -> LineageBreak {
        LineageBreak::Budget {
            seq,
            remaining_ms: remaining.as_millis().min(u128::from(u64::MAX)) as u64,
            reserve_ms: self.reserve.as_millis().min(u128::from(u64::MAX)) as u64,
        }
    }

    /// Fetch one member, verify it against its handle, cache the fact, and park
    /// its bytes for the current candidate.
    async fn fetch_member(
        &mut self,
        handle: &CheckpointHandle,
    ) -> Result<MemberProbe, CheckpointError> {
        // ── The allowance: reserve accounting BY MEMBER KIND (b.bu8) ────────
        //
        // Not a threshold gate. Each fetch gets a kind-dependent allowance
        // derived from the live remainder, and an allowance of zero IS the
        // abandonment. The old form — decline when `remaining < reserve`, then
        // run the fetch for a flat `reserve` — could not deliver the floor
        // answer at the shipped knobs it was written for: with `budget =
        // 2 x reserve`, one wedged member spent a full reserve PLUS the real
        // wall clock of the LIST and the walk around it, so the remainder was
        // `reserve - epsilon`, the base's own fetch was declined by the very
        // gate meant to protect it, and the selection exhausted with a healthy
        // base sitting one step below. Reserving the base's fetch out of the
        // budget at the point of spend fixes that without touching the budget
        // (raising it would hand back the b.gg7 head-of-line stall).
        //
        // Three arms:
        //
        // 1. The UN-GATED first fetch — `min(reserve, remaining)`. While this
        //    candidate has fetched nothing AND the selection has abandoned
        //    nothing, up to one full `reserve` is spent with no budget
        //    precondition: this fetch alone is not charged against the base's
        //    reservation, which is what keeps an untroubled base-newest
        //    selection at exactly the pre-Epic-4 cost. Un-gated is not
        //    unbounded — the `min` still holds it under the live remainder
        //    (property (i) in `gather_restore_lineage`'s `reserve` discussion
        //    is the exact deadline-precision statement; do not restate it
        //    here). In every selection meeting the stated `budget >= 2 x
        //    reserve` precondition that `min` is a no-op (the remainder here
        //    is the budget less the LIST, so >= reserve unless the un-capped
        //    LIST itself ate the budget); it bites only in that slow-LIST
        //    case, which is exactly where a flat `reserve` would have run past
        //    the deadline and made the preemptive wrapper the mechanism rather
        //    than the backstop.
        // 2. `Base` — `min(reserve, remaining)`. THE RESERVATION. Never
        //    declined while any budget is left, because this is the fetch the
        //    whole floor-answer guarantee is about. It is bounded by
        //    `remaining` like every other arm, and the base may legitimately
        //    end AT the deadline — see property (i) in
        //    `gather_restore_lineage`'s `reserve` discussion for the exact
        //    deadline-precision statement and the cooperative-beats-preemptive
        //    tie-break in `recovery.rs` it points at.
        // 3. `Tail` — `min(reserve, remaining - reserve)`, saturating. A tail
        //    is SPECULATIVE: it can only improve on the floor answer, never
        //    produce it. So it may be cut short (a slow-but-healthy tail is an
        //    accepted casualty) or declined un-fetched, and it can never take
        //    the remainder below the base's reservation. However many
        //    ALLOWANCED tails wedge, the first Base the walk reaches is still
        //    funded with a full reserve; a tail that wedges on arm 1 instead
        //    can leave the base less than that (see the `reserve` discussion's
        //    property (ii)), but never nothing while budget is left.
        //
        // The un-gated first fetch is otherwise paid ONCE PER CANDIDATE, which
        // is why arm 1 turns off as soon as anything has been abandoned: a
        // listing of N candidates would burn N reserves of wall clock and blow
        // the outer preemptive timeout — the `Transient` -> `Error` stranding
        // this whole design exists to eliminate.
        //
        // ── Expressed as an INSTANT, not a duration (b.bu8 follow-up) ──────
        //
        // Each arm's cap below is an absolute `Instant` derived from the
        // cooperative `deadline`, spent through `tokio::time::timeout_at`. The
        // duration forms in the prose above are the same numbers — `bound −
        // now` reproduces each `min(..)` exactly — but the anchoring differs,
        // and that difference is the point. A `timeout(allowance, ..)` re-anchors
        // at the moment the timer is CONSTRUCTED, a few statements after
        // `remaining` was sampled, so a fetch clamped to the remainder actually
        // elapses at `deadline + delta`: LATER than the cooperative deadline,
        // and possibly later than the preemptive wrapper's own deadline in
        // `RecoveryHook::try_restore_layer`. `timeout_at` cannot drift: the
        // bound is `min(now + reserve, deadline[ − reserve])`, so it is never
        // after `deadline`. That removes the case where the inner timer elapses
        // PAST the wrapper's deadline; it does not order the two at a tie, which
        // is still settled by `Timeout`'s poll order and the walk's synchronous
        // post-elapse path. See that hook's tie-break note — and its upgrade
        // obligation — before leaning on the tie.
        let now = Instant::now();
        let remaining = self.deadline.saturating_duration_since(now);
        // `checked_sub` rather than a bare `-`: `Instant - Duration` panics when
        // the result is not representable, which a `deadline` less than one
        // `reserve` above the monotonic origin would produce. `None` falls back
        // to `now`, i.e. the zero-allowance decline the saturating duration form
        // reached for the same input.
        let bound = if self.candidate_fetches == 0 && self.step_downs.is_empty() {
            (now + self.reserve).min(self.deadline)
        } else {
            match handle.kind {
                CheckpointKind::Base => (now + self.reserve).min(self.deadline),
                CheckpointKind::Tail => {
                    (now + self.reserve).min(self.deadline.checked_sub(self.reserve).unwrap_or(now))
                }
            }
        };
        if bound <= now {
            return Ok(MemberProbe::BudgetExhausted { remaining });
        }
        self.candidate_fetches += 1;

        let key = (handle.seq(), handle.kind);
        // Per-fetch cap, at the `bound` derived above. REQUIRED, not
        // redundant with that arithmetic: the StorageApi adapter's enforced
        // per-call ceiling is `request_timeout x 3 retries x 2 auth attempts`
        // = 180 s at stock knobs — 6x a 30 s reserve — so a single uncapped
        // degraded fetch outlives the 60 s outer preemptive budget and no
        // cooperative check ever runs. Without this cap the outer timeout fires
        // first and reproduces the `Transient` -> `Error` stranding
        // deadline-awareness exists to eliminate. It is also what makes the
        // reservation real rather than notional: a wedged tail stops at its
        // allowance, so the budget it leaves behind is the base's.
        // `timeout_at`, not `timeout` — see the anchoring note above the bound.
        // See `gather_restore_lineage`'s doc.
        let fetched =
            match tokio::time::timeout_at(bound, self.store.get_layer_checkpoint(handle)).await {
                // The ONE `Permanent` this walk is allowed to convert into a
                // step-down: the adapter has a durable artifact whose meta
                // companion is absent, so the generation carries no identity
                // and can never be restored (b.hka (d)). Recorded as an
                // `Unusable` fact — cached per selection exactly like the
                // `Wedged` path below, so a later candidate that re-reaches
                // this member costs no second fetch — which becomes a
                // `LineageBreak::MissingMetaCompanion`
                // (`stepped_down_missing_meta`, its own outcome so a torn put
                // is not reported as a decode failure), and the existing
                // step-down then picks the next-older restorable generation.
                //
                // Recognised by downcast, never by message match: see
                // `CheckpointError::is_missing_meta_companion`.
                //
                // FENCE: every OTHER `Permanent` — auth failure, malformed
                // response, store outage — still propagates unchanged via the
                // `result?` below. Widening this arm would report a credential
                // outage as "this layer's checkpoint is unusable", which is the
                // wrong verdict in the one situation where an operator most
                // needs the right one. If no candidate survives the step-down,
                // the honest `Ok(None)` is the answer — and it reads
                // `NotRecoverable` only where the exhaustion behind it is
                // *conclusive*. A walk that ran out of budget with a listed
                // `Base` still unread ends `BudgetInconclusive`, which the
                // restore consumer routes to a retriable `Transient` instead
                // (see `Exhaustion`'s type doc).
                Ok(Err(err)) if err.is_missing_meta_companion() => {
                    tracing::warn!(
                        object_id = handle.object_id,
                        seq = handle.seq(),
                        kind = handle.kind.as_str(),
                        error = %err,
                        "restore-lineage member is unrestorable (artifact durable, meta \
                         companion absent); stepping down to the next-older generation"
                    );
                    let fact = LinkFact::Unusable {
                        detail: UnusableDetail::MissingMetaCompanion {
                            message: err.to_string(),
                        },
                    };
                    self.facts.insert(key, fact.clone());
                    return Ok(MemberProbe::Known(fact));
                }
                Ok(result) => result?,
                Err(_elapsed) => {
                    // Record the overrun as a selection-wide fact BEFORE
                    // answering, so the next candidate that reaches this member
                    // consults the fact and never re-fetches it. Without this,
                    // a listing of [T2@20, T1@14, base@10] whose T1 is wedged
                    // burns one allowance inside candidate T2's walk and a
                    // second inside candidate T1's own first fetch, and the
                    // outer preemptive timeout fires before base@10 is ever
                    // addressed. With it, exactly one allowance is spent on T1
                    // in total and the floor answer lands one allowance in.
                    //
                    // The remainder reported is the LIVE one, not a flat zero:
                    // an allowanced Tail stops at `remaining - reserve`, so
                    // what is left is precisely the base's reservation, and
                    // even a Tail wedged on the un-gated first fetch leaves
                    // whatever the budget still holds. Saying "0" would
                    // misreport a selection that can still fund its floor
                    // answer. See `LineageBreak::Budget::remaining_ms`.
                    self.facts.insert(key, LinkFact::Wedged);
                    return Ok(MemberProbe::BudgetExhausted {
                        remaining: self.remaining_budget(),
                    });
                }
            };

        let Some(blob) = fetched else {
            // The contract-documented concurrent-prune race: the listing
            // observed this generation and retention removed it before the get
            // landed. A verdict, not an error.
            let fact = LinkFact::Absent { at_fetch: true };
            self.facts.insert(key, fact.clone());
            return Ok(MemberProbe::Known(fact));
        };

        let fact = self.classify(handle, &blob);
        self.facts.insert(key, fact.clone());
        if matches!(fact, LinkFact::Base { .. } | LinkFact::Tail { .. }) {
            self.candidate_bytes.insert(key, blob);
        }
        Ok(MemberProbe::Known(fact))
    }

    /// The per-member verification: the adapter-contract guards
    /// `select_restore_source` carries once for its single fetch, applied to
    /// every member, plus the authoritative header peek for a tail.
    fn classify(&self, handle: &CheckpointHandle, blob: &CheckpointBlob) -> LinkFact {
        let mismatch =
            |field: &'static str, expected: String, observed: String| LinkFact::Unusable {
                detail: UnusableDetail::Mismatch {
                    field,
                    expected,
                    observed,
                },
            };

        // Unreachable against a conforming adapter — `get_layer_checkpoint`
        // MUST return a blob whose `(object_id, snap_seq, kind)` equals the
        // addressed handle's. Guarded anyway because the failure it prevents is
        // handing a tail envelope to the reset-only decoder.
        if blob.handle.kind != handle.kind {
            return mismatch(
                "kind",
                handle.kind.as_str().to_string(),
                blob.handle.kind.as_str().to_string(),
            );
        }
        if blob.handle.object_id != self.object_id {
            return mismatch(
                "object_id",
                self.object_id.to_string(),
                blob.handle.object_id.to_string(),
            );
        }
        if blob.handle.seq() != handle.seq() {
            return mismatch(
                "snap_seq",
                handle.seq().to_string(),
                blob.handle.seq().to_string(),
            );
        }

        match handle.kind {
            CheckpointKind::Base => LinkFact::Base {
                handle: blob.handle.clone(),
            },
            CheckpointKind::Tail => {
                // The envelope header is the lineage of record. `peek_tail_header`
                // validates the magic and then `format_version` BEFORE any length
                // check, so an artifact written at a version this build cannot read
                // surfaces as `UnknownFormatVersion` (SR-6.1) rather than being
                // guessed at or mis-parsed. The version rides in the returned tuple.
                let (format_version, header) = match peek_tail_header(&blob.bytes) {
                    Ok(peeked) => peeked,
                    Err(err) => {
                        return LinkFact::Unusable {
                            detail: UnusableDetail::Decode {
                                message: err.to_string(),
                            },
                        }
                    }
                };
                if header.object_id != self.object_id {
                    return mismatch(
                        "header.object_id",
                        self.object_id.to_string(),
                        header.object_id.to_string(),
                    );
                }
                // The handle's sequence IS the envelope's `to_seq`.
                if header.to_seq != handle.seq() {
                    return mismatch(
                        "header.to_seq",
                        handle.seq().to_string(),
                        header.to_seq.to_string(),
                    );
                }
                self.cross_check_meta(handle, &blob.meta, &header);
                LinkFact::Tail {
                    handle: blob.handle.clone(),
                    format_version,
                    header,
                }
            }
        }
    }

    /// Advisory meta cross-check. A disagreement WARNs and loses; `None` is
    /// used silently.
    ///
    /// `None` was reachable in production as the adapters' torn-put
    /// placeholder, which the adapter itself WARNed about; b.hka (d) replaced
    /// that placeholder with a refusal, so the silent arm now covers only a
    /// meta a hand-built fixture or a future adapter leaves unset. Silent is
    /// still right for it: an absent lineage field is not a disagreement, and
    /// the envelope header is consulted unconditionally either way.
    fn cross_check_meta(
        &self,
        handle: &CheckpointHandle,
        meta: &CheckpointMeta,
        header: &TailHeader,
    ) {
        if let Some(meta_base) = meta.base_snap_seq {
            if meta_base != header.base_snap_seq {
                tracing::warn!(
                    object_id = self.object_id,
                    seq = handle.seq(),
                    meta_base_snap_seq = meta_base,
                    header_base_snap_seq = header.base_snap_seq,
                    "tail meta disagrees with its envelope header on base_snap_seq; the header \
                     wins (the meta is an index, not a source of truth). Suspect a torn or \
                     hand-edited meta sidecar"
                );
            }
        }
        if let Some(meta_from) = meta.from_seq {
            if meta_from != header.from_seq {
                tracing::warn!(
                    object_id = self.object_id,
                    seq = handle.seq(),
                    meta_from_seq = meta_from,
                    header_from_seq = header.from_seq,
                    "tail meta disagrees with its envelope header on from_seq; the header wins \
                     (the meta is an index, not a source of truth). Suspect a torn or \
                     hand-edited meta sidecar"
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    //! Unit lane for lineage gather and restore selection.
    //!
    //! Every case drives the real entry points over a fake [`CheckpointStore`]
    //! whose layer surface is a hand-built generation table — the only way to
    //! seed the shapes this module must tolerate but no shipped writer
    //! produces: a tail whose meta lost its lineage fields (the adapters reject
    //! that at put time, by design), a member that lists but answers `Ok(None)`
    //! at fetch, an artifact at an unreadable `format_version`, and an adapter
    //! that returns a generation of the wrong kind. The listing is sorted
    //! through the shared `handle_generation_sort_key` rather than re-derived,
    //! so no case can come to depend on an order the adapters do not produce.
    //!
    //! Nothing here touches Valkey, an instrument, or the recovery hook: the
    //! unit is deliberately metrics-free and `Config`-free, so the outcome
    //! vocabulary is asserted as label **strings** and never as metric cells.
    //!
    //! Every "was not consulted" claim is a `get_layer_checkpoint`
    //! **call-count** assertion, never an inspection of the returned lineage: a
    //! member that was fetched and then discarded is a different fact from one
    //! that was never addressed, and only the count tells them apart.

    use super::*;
    use crate::checkpoint::adapters::handle_generation_sort_key;
    use crate::checkpoint::adapters::in_memory::InMemoryCheckpointStore;
    use crate::checkpoint::store::LayerArtifact;
    use crate::checkpoint::tail_codec::{
        encode_tail_envelope, TailRecord, TAIL_FORMAT_VERSION_OFFSET,
    };
    use async_trait::async_trait;
    use std::sync::{Arc, Mutex as StdMutex};
    use tracing_subscriber::fmt::MakeWriter;

    const OBJECT_ID: u64 = 4242;
    const SESSION_ID: &str = "sess-lineage";
    const LAYER_NAME: &str = "world/root.live";
    const CREATED_BY: &str = "user-42";
    const HEADER_SERVER_TS: u64 = 1_700_000_123_456;
    /// Stock-shaped reserve: `LIVE_EDIT_CHECKPOINT_REQUEST_TIMEOUT_MS`. Named
    /// here rather than read from `Config`, which this module never touches.
    const RESERVE: Duration = Duration::from_secs(30);

    // ── Fixtures: artifacts ─────────────────────────────────────────────────

    fn ompf_for(seq: u64) -> Bytes {
        Bytes::from(format!("ompf-payload-{seq}"))
    }

    /// A real, decodable envelope over `(from_seq, to_seq]` — never junk bytes,
    /// so no case can come to depend on a seeded artifact being undecodable.
    fn tail_bytes(object_id: u64, base_snap_seq: u64, from_seq: u64, to_seq: u64) -> Bytes {
        let records: Vec<TailRecord> = (from_seq + 1..=to_seq)
            .map(|seq| TailRecord::Verbatim {
                sequence_num: seq,
                server_ts: 1_700_000_000_000 + seq as i64,
                ompf: ompf_for(seq),
            })
            .collect();
        encode_tail_envelope(
            &TailHeader {
                object_id,
                base_snap_seq,
                from_seq,
                to_seq,
            },
            records,
        )
        .expect("fixture tail envelope encodes")
    }

    /// The same bytes with the on-wire `format_version` forged to a value this
    /// build cannot read (SR-6.1).
    fn at_unreadable_format_version(bytes: &Bytes) -> Bytes {
        let mut raw = bytes.to_vec();
        raw[TAIL_FORMAT_VERSION_OFFSET..TAIL_FORMAT_VERSION_OFFSET + 4]
            .copy_from_slice(&99u32.to_le_bytes());
        Bytes::from(raw)
    }

    fn base_meta(snap_seq: u64) -> CheckpointMeta {
        CheckpointMeta {
            creator_replica_id: "seeder".to_string(),
            written_at_unix_ms: 1,
            snap_seq: Some(snap_seq),
            session_id: SESSION_ID.to_string(),
            layer_name: LAYER_NAME.to_string(),
            created_by: CREATED_BY.to_string(),
            server_ts: HEADER_SERVER_TS,
            kind: CheckpointKind::Base,
            base_snap_seq: None,
            from_seq: None,
            ..CheckpointMeta::default()
        }
    }

    fn tail_meta(base_snap_seq: u64, from_seq: u64, to_seq: u64) -> CheckpointMeta {
        CheckpointMeta {
            kind: CheckpointKind::Tail,
            base_snap_seq: Some(base_snap_seq),
            from_seq: Some(from_seq),
            snap_seq: Some(to_seq),
            ..base_meta(to_seq)
        }
    }

    /// A tail meta that carries no lineage and no identity: kind-only,
    /// `base_snap_seq: None`, `from_seq: None`, empty `session_id` /
    /// `layer_name`.
    ///
    /// This *was* the torn-put residual both adapters synthesized when a
    /// tail's meta artifact never landed. b.hka (d) replaced that fabrication
    /// with an adapter-side refusal, so no shipped read path produces this
    /// shape any more — but the fixture is kept, and the cases below with it,
    /// because the rule it exercises is not about that producer. The gather
    /// consults the envelope header **unconditionally** and treats
    /// [`CheckpointMeta`] as advisory; `cross_check_meta`'s `None` arm is
    /// still live production code, documented as covering "a meta a hand-built
    /// fixture or a future adapter leaves unset". This is that fixture, and
    /// these are the cases that keep the arm from rotting into a mis-link the
    /// day a fourth adapter appears.
    fn lineage_less_tail_meta() -> CheckpointMeta {
        CheckpointMeta {
            kind: CheckpointKind::Tail,
            base_snap_seq: None,
            from_seq: None,
            ..CheckpointMeta::default()
        }
    }

    // ── Fixtures: the fake store ────────────────────────────────────────────

    /// One seeded generation, plus the per-generation fault switches this
    /// module needs.
    #[derive(Clone)]
    struct Generation {
        handle: CheckpointHandle,
        bytes: Bytes,
        meta: CheckpointMeta,
        /// `get_layer_checkpoint` answers `Ok(None)` although the listing
        /// advertises the handle — the contract-documented concurrent-prune
        /// race.
        missing_at_fetch: bool,
        /// The returned blob's handle carries this kind instead of the listed
        /// one — the adapter-contract violation.
        returned_kind: Option<CheckpointKind>,
        /// The returned blob's handle carries this object id.
        returned_object_id: Option<u64>,
        /// The fetch never completes. Models a degraded backend whose enforced
        /// per-call ceiling (180 s at stock knobs) is 6x the reserve.
        hangs: bool,
    }

    fn base_gen(snap_seq: u64) -> Generation {
        Generation {
            handle: CheckpointHandle::base(OBJECT_ID, snap_seq, format!("{OBJECT_ID}.{snap_seq}")),
            bytes: Bytes::from_static(b"seeded-base-ompf"),
            meta: base_meta(snap_seq),
            missing_at_fetch: false,
            returned_kind: None,
            returned_object_id: None,
            hangs: false,
        }
    }

    fn tail_gen(base_snap_seq: u64, from_seq: u64, to_seq: u64) -> Generation {
        Generation {
            handle: CheckpointHandle::tail(OBJECT_ID, to_seq, format!("{OBJECT_ID}.{to_seq}.tail")),
            bytes: tail_bytes(OBJECT_ID, base_snap_seq, from_seq, to_seq),
            meta: tail_meta(base_snap_seq, from_seq, to_seq),
            missing_at_fetch: false,
            returned_kind: None,
            returned_object_id: None,
            hangs: false,
        }
    }

    /// [`CheckpointStore`] whose **layer** surface is the seeded table above
    /// and whose every other method forwards to a real
    /// [`InMemoryCheckpointStore`].
    struct FakeStore {
        inner: InMemoryCheckpointStore,
        generations: Vec<Generation>,
        gets: StdMutex<Vec<(u64, CheckpointKind)>>,
        lists: StdMutex<usize>,
        list_error: StdMutex<Option<CheckpointError>>,
        get_error: StdMutex<Option<CheckpointError>>,
        /// Wall-clock each fetch consumes. Under `tokio::time::pause()` this
        /// advances the paused clock deterministically.
        fetch_delay: Duration,
        /// Wall-clock the LISTING consumes, before any member is addressed.
        ///
        /// Separate from [`Self::fetch_delay`] because the two are bounded by
        /// different things and the gap between them is a real production
        /// shape: the adapter's enumeration is bounded by its own ceiling
        /// (540 s at stock knobs) and by nothing this module owns, so a LIST
        /// can return having already spent the caller's whole budget. No
        /// per-fetch cap covers it — the caller's deadline is sampled before
        /// the LIST and consulted only after it.
        list_delay: Duration,
    }

    impl FakeStore {
        fn new(generations: Vec<Generation>) -> Self {
            Self {
                inner: InMemoryCheckpointStore::new(),
                generations,
                gets: StdMutex::new(Vec::new()),
                lists: StdMutex::new(0),
                list_error: StdMutex::new(None),
                get_error: StdMutex::new(None),
                fetch_delay: Duration::ZERO,
                list_delay: Duration::ZERO,
            }
        }

        fn with_fetch_delay(mut self, delay: Duration) -> Self {
            self.fetch_delay = delay;
            self
        }

        fn with_list_delay(mut self, delay: Duration) -> Self {
            self.list_delay = delay;
            self
        }

        fn fail_next_list(&self, err: CheckpointError) {
            *self.list_error.lock().expect("list_error poisoned") = Some(err);
        }

        fn fail_next_get(&self, err: CheckpointError) {
            *self.get_error.lock().expect("get_error poisoned") = Some(err);
        }

        fn get_count(&self, seq: u64, kind: CheckpointKind) -> usize {
            self.gets
                .lock()
                .expect("gets poisoned")
                .iter()
                .filter(|probe| **probe == (seq, kind))
                .count()
        }

        fn total_gets(&self) -> usize {
            self.gets.lock().expect("gets poisoned").len()
        }

        fn list_count(&self) -> usize {
            *self.lists.lock().expect("lists poisoned")
        }
    }

    #[async_trait]
    impl CheckpointStore for FakeStore {
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
            let handles = self.list_layer_checkpoints(object_id).await?;
            match handles.first() {
                Some(handle) => self.get_layer_checkpoint(handle).await,
                None => Ok(None),
            }
        }

        async fn get_layer_checkpoint(
            &self,
            handle: &CheckpointHandle,
        ) -> Result<Option<CheckpointBlob>, CheckpointError> {
            // Recorded BEFORE any injection: an attempted-then-failed fetch is
            // a different fact from one that never happened.
            self.gets
                .lock()
                .expect("gets poisoned")
                .push((handle.seq(), handle.kind));
            if let Some(err) = self.get_error.lock().expect("get_error poisoned").take() {
                return Err(err);
            }
            if !self.fetch_delay.is_zero() {
                tokio::time::sleep(self.fetch_delay).await;
            }
            let Some(generation) = self
                .generations
                .iter()
                .find(|g| g.handle.seq() == handle.seq() && g.handle.kind == handle.kind)
            else {
                return Ok(None);
            };
            if generation.hangs {
                std::future::pending::<()>().await;
            }
            if generation.missing_at_fetch {
                return Ok(None);
            }
            let mut returned = generation.handle.clone();
            if let Some(kind) = generation.returned_kind {
                returned.kind = kind;
            }
            if let Some(object_id) = generation.returned_object_id {
                returned.object_id = object_id;
            }
            Ok(Some(CheckpointBlob {
                handle: returned,
                bytes: generation.bytes.clone(),
                meta: generation.meta.clone(),
            }))
        }

        async fn list_layer_checkpoints(
            &self,
            object_id: u64,
        ) -> Result<Vec<CheckpointHandle>, CheckpointError> {
            *self.lists.lock().expect("lists poisoned") += 1;
            if let Some(err) = self.list_error.lock().expect("list_error poisoned").take() {
                return Err(err);
            }
            if !self.list_delay.is_zero() {
                tokio::time::sleep(self.list_delay).await;
            }
            let mut handles: Vec<CheckpointHandle> = self
                .generations
                .iter()
                .filter(|g| g.handle.object_id == object_id)
                .map(|g| g.handle.clone())
                .collect();
            // The canonical mixed-kind order, never re-derived here.
            handles.sort_unstable_by_key(handle_generation_sort_key);
            Ok(handles)
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

        async fn list_layer_artifacts(&self) -> Result<Vec<LayerArtifact>, CheckpointError> {
            self.inner.list_layer_artifacts().await
        }
    }

    // ── Assertion helpers ───────────────────────────────────────────────────

    /// Run one selection to its first answer against a far-future deadline.
    async fn first_lineage(store: &FakeStore) -> (Option<GatheredLineage>, Vec<StepDownRecord>) {
        let mut selection = gather_restore_lineage(store, OBJECT_ID, no_deadline(), RESERVE)
            .await
            .expect("listing succeeds");
        let lineage = selection
            .next_restorable()
            .await
            .expect("selection never errors on a healthy store");
        (lineage, selection.step_downs().to_vec())
    }

    #[track_caller]
    fn assert_tails(lineage: &GatheredLineage, spans: &[(u64, u64)]) {
        let observed: Vec<(u64, u64)> = lineage
            .tails
            .iter()
            .map(|tail| (tail.header.from_seq, tail.header.to_seq))
            .collect();
        assert_eq!(
            observed, spans,
            "tails must be OLDEST-first — the walk runs head-down and reverses before returning, \
             and Task 1's replay depends on the pinned direction"
        );
    }

    /// Sink for [`capture_logs`].
    #[derive(Clone)]
    struct CapturingWriter(Arc<StdMutex<Vec<u8>>>);

    impl<'a> MakeWriter<'a> for CapturingWriter {
        type Writer = CapturingWriter;
        fn make_writer(&'a self) -> Self::Writer {
            self.clone()
        }
    }

    impl std::io::Write for CapturingWriter {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0
                .lock()
                .expect("capture buffer poisoned")
                .extend_from_slice(buf);
            Ok(buf.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    /// Drive `f` under a thread-local tracing subscriber and return its value
    /// alongside everything it emitted.
    ///
    /// Needed because *whether a WARN was emitted* is the entire observable
    /// difference between the two ways a selection can run out of candidates:
    /// `next_restorable` answers `Ok(None)` either way, and only the log tells
    /// "the layer is unrecoverable" apart from "the layer restored and there is
    /// nothing older to offer". Asserting on the returned `exhaustion()` alone
    /// would not catch a re-emitted WARN on a second call.
    ///
    /// `set_default` is thread-local and `#[tokio::test]` drives its future on
    /// the calling thread, so the guard covers exactly this future with no
    /// cross-test bleed.
    async fn capture_logs<T>(f: impl std::future::Future<Output = T>) -> (T, String) {
        let buf = Arc::new(StdMutex::new(Vec::new()));
        let subscriber = tracing_subscriber::fmt()
            .with_writer(CapturingWriter(buf.clone()))
            .with_max_level(tracing::Level::TRACE)
            // ANSI off: the assertions match on plain substrings, and the
            // colour escapes land inside them.
            .with_ansi(false)
            .finish();
        let out = {
            let _guard = tracing::subscriber::set_default(subscriber);
            f.await
        };
        let captured = buf.lock().expect("capture buffer poisoned").clone();
        (
            out,
            String::from_utf8(captured).expect("captured tracing output must be UTF-8"),
        )
    }

    /// The `AllBroken` WARN, keyed on a distinctive slice of its message so a
    /// count is exact. Deliberately not a prefix of `warn_step_down`'s message,
    /// which also contains the words "stepped down".
    const ALL_BROKEN_WARN: &str = "stepped down through every candidate generation";

    #[track_caller]
    fn assert_break(records: &[StepDownRecord], index: usize, expected: &LineageBreak) {
        let record = records.get(index).unwrap_or_else(|| {
            panic!(
                "expected at least {} step-downs, got {records:?}",
                index + 1
            )
        });
        assert_eq!(
            &record.reason, expected,
            "wrong step-down reason at index {index}"
        );
    }

    // ── Group A: gathering one candidate lineage ────────────────────────────

    #[tokio::test]
    async fn a_base_headed_candidate_gathers_the_base_alone() {
        let store = FakeStore::new(vec![base_gen(10)]);
        let (lineage, breaks) = first_lineage(&store).await;
        let lineage = lineage.expect("a base is always restorable");

        assert!(lineage.is_base_only());
        assert_eq!(lineage.chain_len(), 0);
        assert_eq!(lineage.head_seq, 10, "head_seq is the base's own sequence");
        assert_eq!(lineage.base_snap_seq(), 10);
        assert_eq!(lineage.outcome_label(), "gathered_base");
        assert!(breaks.is_empty());
        // Identical work to the pre-Epic-4 path: one listing, one get.
        assert_eq!(store.list_count(), 1);
        assert_eq!(store.total_gets(), 1);
    }

    #[tokio::test]
    async fn a_tail_headed_lineage_gathers_its_base_and_every_contiguous_tail_in_replay_order() {
        let store = FakeStore::new(vec![
            base_gen(10),
            tail_gen(10, 10, 14),
            tail_gen(10, 14, 20),
            tail_gen(10, 20, 26),
        ]);
        let (lineage, breaks) = first_lineage(&store).await;
        let lineage = lineage.expect("an intact lineage gathers");

        assert_eq!(lineage.base_snap_seq(), 10);
        assert_tails(&lineage, &[(10, 14), (14, 20), (20, 26)]);
        assert_eq!(
            lineage.head_seq, 26,
            "head_seq is S — the head tail's to_seq, not the base's sequence"
        );
        assert_eq!(lineage.chain_len(), 3);
        assert_eq!(lineage.outcome_label(), "gathered_composed");
        assert!(breaks.is_empty());
        // <= 1 base + the walked tails, and one listing for the whole gather.
        assert_eq!(store.list_count(), 1);
        assert_eq!(store.total_gets(), 4);
    }

    #[tokio::test]
    async fn identity_on_a_gathered_lineage_is_base_sourced() {
        // `apply::live_mapping_resolves_blob_identity` refuses a blob with an
        // empty `session_id` / `layer_name` outright, so the base's meta is the
        // only identity carrier a caller may hand to that guard. The rule is
        // structural — a tail's identity is base-sourced by construction — and
        // does NOT rest on the adapters' identity-empty meta-absent
        // placeholder, which b.hka (d) removed. This case pins the rule by
        // giving the head tail an identity-empty meta anyway and requiring the
        // gather to source identity from the base regardless.
        let mut head = tail_gen(10, 10, 14);
        head.meta = lineage_less_tail_meta();
        let store = FakeStore::new(vec![base_gen(10), head]);
        let (lineage, _) = first_lineage(&store).await;
        let lineage = lineage.expect("gathers");

        assert_eq!(lineage.base.meta.session_id, SESSION_ID);
        assert_eq!(lineage.base.meta.layer_name, LAYER_NAME);
        assert!(
            lineage.tails[0].meta.session_id.is_empty(),
            "the fixture must actually carry the identity-empty placeholder, or this case is \
             vacuous"
        );
    }

    #[tokio::test]
    async fn a_lineage_less_meta_tail_gathers_via_its_envelope_header() {
        // `base_snap_seq: None`, `from_seq: None` — the shape
        // `cross_check_meta`'s silent `None` arm covers. The header is
        // consulted unconditionally, so a meta that names no lineage can
        // neither skip nor mis-link the member.
        let mut middle = tail_gen(10, 10, 14);
        middle.meta = lineage_less_tail_meta();
        let store = FakeStore::new(vec![base_gen(10), middle, tail_gen(10, 14, 20)]);
        let (lineage, breaks) = first_lineage(&store).await;
        let lineage = lineage.expect("a lineage-less meta must not break the lineage");

        assert_tails(&lineage, &[(10, 14), (14, 20)]);
        assert_eq!(lineage.head_seq, 20);
        assert!(breaks.is_empty());
    }

    #[tokio::test]
    async fn a_meta_that_disagrees_with_its_envelope_loses_to_the_header() {
        // The meta claims a lineage rooted at 999 and a span starting at 777.
        // The header says (10, 14] on base 10, and the header is the source of
        // truth — the walk resolves through it (and WARNs at the site).
        let mut middle = tail_gen(10, 10, 14);
        middle.meta = CheckpointMeta {
            base_snap_seq: Some(999),
            from_seq: Some(777),
            ..middle.meta
        };
        let store = FakeStore::new(vec![base_gen(10), middle, tail_gen(10, 14, 20)]);
        let (lineage, breaks) = first_lineage(&store).await;
        let lineage = lineage.expect("the envelope header resolves the lineage");

        assert_tails(&lineage, &[(10, 14), (14, 20)]);
        assert!(breaks.is_empty());
    }

    #[tokio::test]
    async fn a_link_absent_from_the_listing_steps_down_naming_the_offending_sequence() {
        // (14, 20] names a predecessor at 14; nothing is listed there.
        let store = FakeStore::new(vec![base_gen(10), tail_gen(10, 14, 20)]);
        let (lineage, breaks) = first_lineage(&store).await;

        assert_break(
            &breaks,
            0,
            &LineageBreak::LinkAbsent {
                seq: 14,
                kind: CheckpointKind::Tail,
                at_fetch: false,
            },
        );
        assert_eq!(breaks[0].reason.reason_label(), "stepped_down_link_broken");
        assert_eq!(
            store.get_count(14, CheckpointKind::Tail),
            0,
            "a link the listing does not hold must not be fetched"
        );
        let lineage = lineage.expect("steps down to the base");
        assert!(lineage.is_base_only());
        assert_eq!(lineage.head_seq, 10);
    }

    #[tokio::test]
    async fn a_member_pruned_between_the_list_and_the_get_steps_down() {
        let mut middle = tail_gen(10, 10, 14);
        middle.missing_at_fetch = true;
        let store = FakeStore::new(vec![base_gen(10), middle, tail_gen(10, 14, 20)]);
        let (lineage, breaks) = first_lineage(&store).await;

        assert_break(
            &breaks,
            0,
            &LineageBreak::LinkAbsent {
                seq: 14,
                kind: CheckpointKind::Tail,
                at_fetch: true,
            },
        );
        assert!(lineage.expect("steps down").is_base_only());
    }

    #[tokio::test]
    async fn an_unreadable_format_version_steps_down_as_a_decode_break() {
        // SR-6.1: an artifact written at a version this build cannot read is
        // surfaced, never guessed at.
        let mut head = tail_gen(10, 10, 14);
        head.bytes = at_unreadable_format_version(&head.bytes);
        let store = FakeStore::new(vec![base_gen(10), head]);
        let (lineage, breaks) = first_lineage(&store).await;

        match &breaks[0].reason {
            LineageBreak::Decode { seq, message } => {
                assert_eq!(*seq, 14);
                assert!(
                    message.contains("format version") && message.contains("99"),
                    "the break must name the version the artifact carried; got {message:?}"
                );
            }
            other => panic!("expected a Decode break, got {other:?}"),
        }
        assert_eq!(breaks[0].reason.reason_label(), "stepped_down_decode");
        assert!(lineage.expect("steps down").is_base_only());
    }

    #[tokio::test]
    async fn a_corrupt_envelope_steps_down_as_a_decode_break() {
        let mut head = tail_gen(10, 10, 14);
        head.bytes = Bytes::from_static(b"not a tail envelope at all");
        let store = FakeStore::new(vec![base_gen(10), head]);
        let (lineage, breaks) = first_lineage(&store).await;

        assert_eq!(breaks[0].reason.reason_label(), "stepped_down_decode");
        assert_eq!(breaks[0].reason.link_seq(), 14);
        assert!(lineage.expect("steps down").is_base_only());
    }

    #[tokio::test]
    async fn an_adapter_returning_the_wrong_kind_steps_down_per_member() {
        // The guard `select_restore_source` carries once for its single fetch,
        // applied to every member: refusing to route a tail envelope into the
        // reset-only decoder.
        let mut head = tail_gen(10, 10, 14);
        head.returned_kind = Some(CheckpointKind::Base);
        let store = FakeStore::new(vec![base_gen(10), head]);
        let (lineage, breaks) = first_lineage(&store).await;

        assert_break(
            &breaks,
            0,
            &LineageBreak::LinkMismatch {
                seq: 14,
                field: "kind",
                expected: "tail".to_string(),
                observed: "base".to_string(),
            },
        );
        assert_eq!(
            breaks[0].reason.reason_label(),
            "stepped_down_link_mismatch"
        );
        assert!(lineage.expect("steps down").is_base_only());
    }

    #[tokio::test]
    async fn an_adapter_returning_the_wrong_object_id_steps_down() {
        let mut head = tail_gen(10, 10, 14);
        head.returned_object_id = Some(OBJECT_ID + 1);
        let store = FakeStore::new(vec![base_gen(10), head]);
        let (_, breaks) = first_lineage(&store).await;

        assert_break(
            &breaks,
            0,
            &LineageBreak::LinkMismatch {
                seq: 14,
                field: "object_id",
                expected: OBJECT_ID.to_string(),
                observed: (OBJECT_ID + 1).to_string(),
            },
        );
    }

    #[tokio::test]
    async fn a_header_naming_a_foreign_object_steps_down() {
        let mut head = tail_gen(10, 10, 14);
        head.bytes = tail_bytes(OBJECT_ID + 7, 10, 10, 14);
        let store = FakeStore::new(vec![base_gen(10), head]);
        let (_, breaks) = first_lineage(&store).await;

        assert_break(
            &breaks,
            0,
            &LineageBreak::LinkMismatch {
                seq: 14,
                field: "header.object_id",
                expected: OBJECT_ID.to_string(),
                observed: (OBJECT_ID + 7).to_string(),
            },
        );
    }

    #[tokio::test]
    async fn a_header_whose_to_seq_disagrees_with_its_handle_steps_down() {
        let mut head = tail_gen(10, 10, 14);
        // The handle is addressed at 14; the envelope claims it ends at 13.
        head.bytes = tail_bytes(OBJECT_ID, 10, 10, 13);
        let store = FakeStore::new(vec![base_gen(10), head]);
        let (_, breaks) = first_lineage(&store).await;

        assert_break(
            &breaks,
            0,
            &LineageBreak::LinkMismatch {
                seq: 14,
                field: "header.to_seq",
                expected: "14".to_string(),
                observed: "13".to_string(),
            },
        );
    }

    #[tokio::test]
    async fn a_member_rooted_at_a_different_base_steps_down() {
        // (14, 20] roots at base 10; its predecessor (5, 14] roots at base 5.
        // The link's sequences chain, but the lineage roots do not.
        let store = FakeStore::new(vec![
            base_gen(5),
            base_gen(10),
            tail_gen(5, 5, 14),
            tail_gen(10, 14, 20),
        ]);
        let (lineage, breaks) = first_lineage(&store).await;

        assert_break(
            &breaks,
            0,
            &LineageBreak::LinkMismatch {
                seq: 14,
                field: "base_snap_seq",
                expected: "10".to_string(),
                observed: "5".to_string(),
            },
        );
        // Steps down to the tail-headed lineage rooted at base 5.
        let lineage = lineage.expect("an older lineage verifies");
        assert_eq!(lineage.base_snap_seq(), 5);
        assert_tails(&lineage, &[(5, 14)]);
    }

    #[tokio::test]
    async fn store_errors_propagate_verbatim_from_the_listing() {
        let store = FakeStore::new(vec![base_gen(10)]);
        store.fail_next_list(CheckpointError::Transient {
            source: "backend brownout".into(),
        });
        let err = gather_restore_lineage(&store, OBJECT_ID, no_deadline(), RESERVE)
            .await
            .expect_err("a listing failure is an error, not a verdict");
        assert!(err.is_transient());
    }

    #[tokio::test]
    async fn store_errors_propagate_verbatim_from_a_member_fetch() {
        let store = FakeStore::new(vec![base_gen(10), tail_gen(10, 10, 14)]);
        store.fail_next_get(CheckpointError::Permanent {
            source: "malformed persisted meta".into(),
        });
        let mut selection = gather_restore_lineage(&store, OBJECT_ID, no_deadline(), RESERVE)
            .await
            .expect("listing succeeds");
        let err = selection
            .next_restorable()
            .await
            .expect_err("an adapter error is never folded into a step-down");
        assert!(err.is_permanent());
    }

    /// SR-9.2's divergent-S fork: two leaders reading at different times mint
    /// tails that share a `from_seq` and differ in `to_seq`. The loser is not a
    /// member of the newest lineage, and the walk — which addresses each
    /// predecessor by EXACT SEQUENCE, never by listing position — must never
    /// address it, even though it sits between two members in the listing.
    #[tokio::test]
    async fn the_divergent_s_fork_residue_is_never_addressed() {
        const S0: u64 = 10;
        const A: u64 = 14;
        const B: u64 = 17;
        const C: u64 = 22;
        // The interleaving is the whole point of the fixture: T2 must sit
        // BETWEEN the two members in the listing, or a position-driven walk
        // would skip it for the wrong reason.
        const { assert!(A < B && B < C) };

        let store = FakeStore::new(vec![
            base_gen(S0),
            tail_gen(S0, S0, A), // T1 — a member
            tail_gen(S0, S0, B), // T2 — the fork residue, NOT a member
            tail_gen(S0, A, C),  // T3 — the head
        ]);
        let (lineage, breaks) = first_lineage(&store).await;
        let lineage = lineage.expect("the newest lineage verifies");

        assert_tails(&lineage, &[(S0, A), (A, C)]);
        assert_eq!(lineage.head_seq, C);
        assert!(breaks.is_empty());

        assert_eq!(store.get_count(C, CheckpointKind::Tail), 1);
        assert_eq!(store.get_count(A, CheckpointKind::Tail), 1);
        assert_eq!(store.get_count(S0, CheckpointKind::Base), 1);
        assert_eq!(
            store.get_count(B, CheckpointKind::Tail),
            0,
            "T2 sits between T3 and T1 in the listing but is not addressed by any link; a \
             position-driven walk would have fetched it"
        );
        assert_eq!(store.total_gets(), 3, "exactly 1 base + the 2 walked tails");
    }

    /// The gather must work against a shipped adapter, not only the fake.
    #[tokio::test]
    async fn an_intact_lineage_gathers_against_the_in_memory_adapter() {
        let store = InMemoryCheckpointStore::new();
        store
            .put_layer_checkpoint(
                OBJECT_ID,
                10,
                Bytes::from_static(b"seeded-base-ompf"),
                base_meta(10),
            )
            .await
            .expect("seeding the base");
        for (from_seq, to_seq) in [(10u64, 14u64), (14, 20)] {
            store
                .put_layer_checkpoint(
                    OBJECT_ID,
                    to_seq,
                    tail_bytes(OBJECT_ID, 10, from_seq, to_seq),
                    tail_meta(10, from_seq, to_seq),
                )
                .await
                .expect("seeding a tail");
        }

        let mut selection = gather_restore_lineage(&store, OBJECT_ID, no_deadline(), RESERVE)
            .await
            .expect("listing succeeds");
        let lineage = selection
            .next_restorable()
            .await
            .expect("no error")
            .expect("the lineage verifies");
        assert_tails(&lineage, &[(10, 14), (14, 20)]);
        assert_eq!(lineage.head_seq, 20);
        assert_eq!(lineage.base.handle.kind, CheckpointKind::Base);
    }

    // ── Group B: newest-restorable selection and the step-down loop ─────────

    /// Corrupt an already-seeded generation's envelope in place.
    fn corrupted(mut generation: Generation) -> Generation {
        generation.bytes = Bytes::from_static(b"not a tail envelope at all");
        generation
    }

    #[tokio::test]
    async fn selection_returns_the_newest_base_when_the_newest_generation_is_a_base() {
        // Post-re-base stores routinely interleave: a fresh base sits above an
        // older lineage.
        let store = FakeStore::new(vec![base_gen(5), tail_gen(5, 5, 14), base_gen(20)]);
        let (lineage, breaks) = first_lineage(&store).await;
        let lineage = lineage.expect("the newest base is always restorable");

        assert_eq!(lineage.base_snap_seq(), 20);
        assert!(lineage.is_base_only());
        assert_eq!(lineage.outcome_label(), "gathered_base");
        assert!(breaks.is_empty());
        assert_eq!(
            store.total_gets(),
            1,
            "a base candidate costs exactly one fetch"
        );
    }

    #[tokio::test]
    async fn a_broken_newest_lineage_steps_down_to_an_older_tail_headed_lineage() {
        // NOT to the next older base: advancing by listing index is what lets
        // an older *tail*-headed lineage still win.
        let store = FakeStore::new(vec![
            base_gen(10),
            tail_gen(10, 10, 14),
            corrupted(tail_gen(10, 14, 20)),
            tail_gen(10, 20, 26),
        ]);
        let mut selection = gather_restore_lineage(&store, OBJECT_ID, no_deadline(), RESERVE)
            .await
            .expect("listing succeeds");
        let lineage = selection
            .next_restorable()
            .await
            .expect("no error")
            .expect("an older lineage verifies");

        assert_eq!(lineage.base_snap_seq(), 10);
        assert_tails(&lineage, &[(10, 14)]);
        assert_eq!(lineage.head_seq, 14);
        assert_eq!(
            lineage.chain_len(),
            1,
            "chain_len is the TRUE link-walk length, not the listing position of the base"
        );
        assert_eq!(lineage.outcome_label(), "gathered_composed");

        // Two candidates abandoned: the head, and the corrupt member itself
        // once it became a candidate in its own right.
        let breaks = selection.step_downs();
        assert_eq!(breaks.len(), 2, "got {breaks:?}");
        assert_eq!(breaks[0].head_seq, 26);
        assert_eq!(breaks[1].head_seq, 20);
        for record in breaks {
            assert_eq!(record.reason.reason_label(), "stepped_down_decode");
            assert_eq!(record.reason.link_seq(), 20);
        }

        assert_eq!(
            store.get_count(20, CheckpointKind::Tail),
            1,
            "the shared broken link must be probed once, not once per candidate"
        );
        assert_eq!(
            store.list_count(),
            1,
            "one listing per selection, never re-read"
        );
    }

    #[tokio::test]
    async fn every_lineage_broken_falls_to_the_oldest_base() {
        let mut newest_base = base_gen(10);
        newest_base.missing_at_fetch = true;
        let store = FakeStore::new(vec![
            base_gen(5),
            newest_base,
            corrupted(tail_gen(10, 10, 14)),
        ]);
        let mut selection = gather_restore_lineage(&store, OBJECT_ID, no_deadline(), RESERVE)
            .await
            .expect("listing succeeds");
        let lineage = selection
            .next_restorable()
            .await
            .expect("no error")
            .expect("the oldest base still restores");

        assert_eq!(lineage.base_snap_seq(), 5);
        assert!(lineage.is_base_only());
        let breaks = selection.step_downs();
        assert_eq!(breaks.len(), 2, "got {breaks:?}");
        assert_eq!(breaks[0].reason.reason_label(), "stepped_down_decode");
        assert_eq!(
            breaks[1].reason,
            LineageBreak::LinkAbsent {
                seq: 10,
                kind: CheckpointKind::Base,
                at_fetch: true,
            }
        );
    }

    #[tokio::test]
    async fn a_tails_only_listing_yields_none_with_the_no_base_exhaustion() {
        // Absorbs `select_restore_source`'s tails-only arm: the base is the
        // only reset-decodable artifact, so this stays unrestorable under
        // composed restore.
        let store = FakeStore::new(vec![tail_gen(10, 10, 14), tail_gen(10, 14, 20)]);
        let mut selection = gather_restore_lineage(&store, OBJECT_ID, no_deadline(), RESERVE)
            .await
            .expect("listing succeeds");

        assert!(selection
            .next_restorable()
            .await
            .expect("no error")
            .is_none());
        assert_eq!(
            selection.exhaustion(),
            Some(&Exhaustion::NoBase {
                generations: 2,
                newest_sequence: 20,
            })
        );
        assert_eq!(
            selection.exhaustion().expect("set").outcome_label(),
            "exhausted"
        );
        assert_eq!(
            store.total_gets(),
            0,
            "a listing with no base is answerable from the listing alone"
        );
    }

    #[tokio::test]
    async fn an_empty_listing_yields_none_with_the_empty_listing_exhaustion() {
        let store = FakeStore::new(vec![]);
        let mut selection = gather_restore_lineage(&store, OBJECT_ID, no_deadline(), RESERVE)
            .await
            .expect("listing succeeds");

        assert!(selection
            .next_restorable()
            .await
            .expect("no error")
            .is_none());
        assert_eq!(selection.exhaustion(), Some(&Exhaustion::EmptyListing));
        assert_eq!(store.total_gets(), 0);
    }

    #[tokio::test]
    async fn next_restorable_resumes_strictly_below_the_previous_candidate_and_terminates() {
        let store = FakeStore::new(vec![
            base_gen(10),
            tail_gen(10, 10, 14),
            corrupted(tail_gen(10, 14, 20)),
            tail_gen(10, 20, 26),
        ]);
        let mut selection = gather_restore_lineage(&store, OBJECT_ID, no_deadline(), RESERVE)
            .await
            .expect("listing succeeds");

        let mut heads = Vec::new();
        // Bounded drive: the listing is 4 long, so more iterations than that
        // would mean the cursor failed to advance.
        for _ in 0..8 {
            match selection.next_restorable().await.expect("no error") {
                Some(lineage) => heads.push(lineage.head_seq),
                None => break,
            }
        }

        assert_eq!(
            heads,
            vec![14, 10],
            "each answer must sit strictly below the previous one"
        );
        assert!(
            heads.windows(2).all(|w| w[0] > w[1]),
            "resumption is strictly below the previously-returned head sequence"
        );
        assert!(matches!(
            selection.exhaustion(),
            Some(Exhaustion::AllBroken { .. })
        ));
        assert!(
            !selection.step_downs().is_empty(),
            "the accumulated reason record travels with the exhaustion"
        );
    }

    #[tokio::test]
    async fn a_shared_broken_link_is_probed_once_across_a_multi_step_down_selection() {
        // T1 lists but answers `Ok(None)` at fetch. Three candidates above it
        // (T3, T2, T1 itself) all reach the same link; only the first pays.
        let mut vanished = tail_gen(10, 10, 14);
        vanished.missing_at_fetch = true;
        let store = FakeStore::new(vec![
            base_gen(10),
            vanished,
            tail_gen(10, 14, 20),
            tail_gen(10, 20, 26),
        ]);
        let mut selection = gather_restore_lineage(&store, OBJECT_ID, no_deadline(), RESERVE)
            .await
            .expect("listing succeeds");
        let lineage = selection
            .next_restorable()
            .await
            .expect("no error")
            .expect("the base still restores");

        assert!(lineage.is_base_only());
        assert_eq!(selection.step_downs().len(), 3);
        for (seq, kind) in [
            (26, CheckpointKind::Tail),
            (20, CheckpointKind::Tail),
            (14, CheckpointKind::Tail),
            (10, CheckpointKind::Base),
        ] {
            assert_eq!(
                store.get_count(seq, kind),
                1,
                "{kind:?}@{seq} must be probed exactly once across the whole selection"
            );
        }
        assert_eq!(store.total_gets(), 4);
        assert_eq!(store.list_count(), 1);
    }

    #[tokio::test]
    async fn record_compose_failure_steps_down_through_the_shared_vocabulary() {
        // Task 1 classifies a composition `Permanent`; Task 3's loop records it
        // here so a composition failure and a broken link step down through one
        // vocabulary.
        let store = FakeStore::new(vec![base_gen(10), tail_gen(10, 10, 14)]);
        let mut selection = gather_restore_lineage(&store, OBJECT_ID, no_deadline(), RESERVE)
            .await
            .expect("listing succeeds");

        let composed = selection
            .next_restorable()
            .await
            .expect("no error")
            .expect("the lineage verifies");
        assert_eq!(composed.head_seq, 14);
        assert!(selection.exhaustion().is_none());

        selection.record_compose_failure(composed.head_seq, CheckpointKind::Tail);
        let fallback = selection
            .next_restorable()
            .await
            .expect("no error")
            .expect("the base is still reachable after a composition failure");
        assert!(fallback.is_base_only());
        assert_eq!(fallback.base_snap_seq(), 10);

        let breaks = selection.step_downs();
        assert_eq!(breaks.len(), 1);
        assert_eq!(breaks[0].reason, LineageBreak::Compose { head_seq: 14 });
        assert_eq!(breaks[0].reason.reason_label(), "stepped_down_compose");
    }

    /// Draining a selection that ALREADY ANSWERED is not an exhaustion.
    ///
    /// A caller drives this cursor until it stops offering generations, so the
    /// final `Ok(None)` on a healthy store is the routine end of a successful
    /// restore — the caller took a lineage and there is nothing older below it.
    /// Reporting `AllBroken` there would have Task 3's hook fold an `exhausted`
    /// outcome onto a layer that restored, and WARN "the layer is unrecoverable
    /// despite holding durable checkpoints" about a layer that was just
    /// recovered. Both are false, and the WARN is the kind of false page that
    /// teaches an operator to ignore the real one.
    #[tokio::test]
    async fn a_drain_after_a_successful_candidate_is_quiet_and_records_no_exhaustion() {
        let store = FakeStore::new(vec![base_gen(10), tail_gen(10, 10, 14)]);
        let ((heads, exhaustion, step_downs), logs) = capture_logs(async {
            let mut selection = gather_restore_lineage(&store, OBJECT_ID, no_deadline(), RESERVE)
                .await
                .expect("listing succeeds");
            let mut heads = Vec::new();
            // Bounded drive: the listing is 2 long, so more iterations than
            // that would mean the cursor failed to advance.
            for _ in 0..4 {
                match selection.next_restorable().await.expect("no error") {
                    Some(lineage) => heads.push(lineage.head_seq),
                    None => break,
                }
            }
            (
                heads,
                selection.exhaustion().cloned(),
                selection.step_downs().to_vec(),
            )
        })
        .await;

        assert_eq!(
            heads,
            vec![14, 10],
            "both candidates verify; the drive ends because nothing is left, not because \
             something broke"
        );
        assert!(
            step_downs.is_empty(),
            "nothing was abandoned; got {step_downs:?}"
        );
        assert_eq!(
            exhaustion, None,
            "a drain after a successful candidate is not an exhaustion — `exhaustion()` is the \
             'nothing restorable' signal, and this selection restored twice"
        );
        assert!(
            !logs.contains("WARN"),
            "a clean selection must emit no WARN at all; got:\n{logs}"
        );
        assert!(
            !logs.contains(ALL_BROKEN_WARN),
            "the unrecoverable-layer WARN must not fire for a layer that just restored; got:\n\
             {logs}"
        );
    }

    /// Asking an exhausted selection again is a no-op, not a second page.
    ///
    /// `next_restorable` is driven in a loop, and a caller that calls once more
    /// after `Ok(None)` — or retries its own step-down loop — must not multiply
    /// the operator signal. The record is minted once and the WARN with it.
    #[tokio::test]
    async fn next_restorable_after_exhaustion_is_idempotent_and_does_not_re_warn() {
        let store = FakeStore::new(vec![base_gen(10), corrupted(tail_gen(10, 10, 14))]);
        let ((first, exhausted_once, again, exhausted_twice, gets), logs) = capture_logs(async {
            let mut selection = gather_restore_lineage(&store, OBJECT_ID, no_deadline(), RESERVE)
                .await
                .expect("listing succeeds");
            let first = selection
                .next_restorable()
                .await
                .expect("no error")
                .expect("the base behind the corrupt tail restores");
            let drained = selection.next_restorable().await.expect("no error");
            assert!(drained.is_none(), "nothing sits below the base");
            let exhausted_once = (
                selection.exhaustion().cloned(),
                selection.step_downs().len(),
            );
            let gets = store.total_gets();

            let again = selection.next_restorable().await.expect("no error");
            let exhausted_twice = (
                selection.exhaustion().cloned(),
                selection.step_downs().len(),
            );
            (
                first.head_seq,
                exhausted_once,
                again.is_none(),
                exhausted_twice,
                (gets, store.total_gets()),
            )
        })
        .await;

        assert_eq!(first, 10);
        assert!(again, "an exhausted selection has nothing left to offer");
        assert_eq!(
            exhausted_once,
            (Some(Exhaustion::AllBroken { candidates: 2 }), 1),
            "a candidate WAS abandoned here, so this is the loud arm"
        );
        assert_eq!(
            exhausted_twice, exhausted_once,
            "the verdict is minted once; a second call must not re-derive or re-accumulate it"
        );
        assert_eq!(
            gets.0, gets.1,
            "the re-entry must not re-walk the listing either"
        );
        assert_eq!(
            logs.matches(ALL_BROKEN_WARN).count(),
            1,
            "the unrecoverable-layer WARN is one operator signal per selection, not one per \
             call; got:\n{logs}"
        );
    }

    /// The floor property, as a TABLE test — not a `proptest!` (the
    /// proptest-isolation rule confines those to `tests/proptests.rs`; property
    /// evidence, if wanted, belongs to Task 6's proptest lane).
    ///
    /// The claim: composed restore can only ever ADD coverage. Whenever a
    /// `Base` is present, selection resolves to *something*; with a base-only
    /// listing it resolves to exactly the newest base — the handle the
    /// pre-Epic-4 `position(kind == Base)` selection picks; and in every case
    /// the answer's `head_seq` is at least the newest base's sequence, i.e.
    /// never worse than today's answer.
    #[tokio::test]
    async fn the_floor_property_holds_over_mixed_kind_listings() {
        let cases: Vec<(&str, Vec<Generation>)> = vec![
            ("one base", vec![base_gen(10)]),
            ("two bases", vec![base_gen(10), base_gen(20)]),
            ("base + one tail", vec![base_gen(10), tail_gen(10, 10, 14)]),
            (
                "base + a chain",
                vec![base_gen(10), tail_gen(10, 10, 14), tail_gen(10, 14, 20)],
            ),
            (
                "an older base beneath a lineage",
                vec![base_gen(5), base_gen(10), tail_gen(10, 10, 14)],
            ),
            (
                "a tail whose link is absent from the listing",
                vec![base_gen(10), tail_gen(10, 14, 20)],
            ),
            (
                "the divergent-S fork residue",
                vec![
                    base_gen(10),
                    tail_gen(10, 10, 14),
                    tail_gen(10, 10, 17),
                    tail_gen(10, 14, 22),
                ],
            ),
            (
                "a base colliding with a tail at one sequence",
                vec![base_gen(10), tail_gen(10, 10, 14), base_gen(14)],
            ),
            (
                "every tail corrupt",
                vec![
                    base_gen(10),
                    corrupted(tail_gen(10, 10, 14)),
                    corrupted(tail_gen(10, 14, 20)),
                ],
            ),
            ("tails only — no base", vec![tail_gen(10, 10, 14)]),
            ("empty", vec![]),
        ];

        for (name, generations) in cases {
            let base_only = generations
                .iter()
                .all(|g| g.handle.kind == CheckpointKind::Base);
            let store = FakeStore::new(generations);
            let listing = store
                .list_layer_checkpoints(OBJECT_ID)
                .await
                .expect("listing succeeds");
            // The pre-Epic-4 oracle: the first Base in the canonical
            // newest-first order IS the newest base.
            let newest_base = listing
                .iter()
                .find(|handle| handle.kind == CheckpointKind::Base)
                .cloned();

            let (lineage, _) = first_lineage(&store).await;

            let Some(newest_base) = newest_base else {
                assert!(
                    lineage.is_none(),
                    "[{name}] a listing with no base has no reset-decodable root"
                );
                continue;
            };
            let lineage = lineage.unwrap_or_else(|| {
                panic!("[{name}] a Base is present, so selection MUST resolve to a lineage")
            });
            assert!(
                lineage.head_seq >= newest_base.seq(),
                "[{name}] composed restore resolved to {} — BELOW the newest base at {}; it may \
                 only ever add coverage",
                lineage.head_seq,
                newest_base.seq()
            );
            if base_only {
                assert_eq!(
                    lineage.base.handle, newest_base,
                    "[{name}] a base-only listing must resolve to exactly the handle the \
                     pre-Epic-4 selection picks"
                );
                assert!(lineage.is_base_only());
                assert_eq!(lineage.outcome_label(), "gathered_base");
            }
        }
    }

    // ── Group C: the deadline / allowance budget ────────────────────────────
    //
    // Every case here runs on a PAUSED clock, so the budget arithmetic is
    // deterministic rather than a wall-clock race: `tokio::time::sleep` inside
    // the fake advances the paused clock by exactly the configured per-fetch
    // (and, where set, per-LISTING) cost, and `tokio::time::timeout`
    // auto-advances when the runtime idles, which is what lets a
    // never-returning fetch be tested at all. Every elapsed-time assertion
    // below is therefore an exact arithmetic claim, not a timing heuristic —
    // the guide's "no timing assertions in a gating suite" rule is about
    // wall-clock races, and a paused clock has none.
    //
    // The arithmetic they pin, at the shipped defaults: budget
    // `= max(rpc_deadline_ms, 2 x checkpoint_request_timeout_ms) = 60 s`,
    // reserve `= checkpoint_request_timeout_ms = 30 s` — so the budget is
    // always at least TWO reserves, which is the gather's stated precondition.
    //
    // WHAT THE RULE IS (post-b.bu8). Not a threshold gate. Every fetch is
    // capped at a kind-dependent ALLOWANCE taken off the LIVE remainder, and a
    // zero allowance IS the abandonment:
    //
    //   arm 1, the un-gated first fetch  `min(reserve, remaining)`
    //   `Base`                           `min(reserve, remaining)`
    //   `Tail`                           `min(reserve, remaining - reserve)`
    //
    // The base's reserve is held back by ACCOUNTING rather than bought with a
    // bigger budget, so these cases pin consequences the old gate could not
    // deliver at the very knobs it shipped under:
    //
    // - one wedged tail leaves the base fundable at budget == 2 x reserve
    //   (`one_wedged_member_cannot_starve_the_floor_at_the_shipped_two_reserve_budget`),
    //   and N wedged tails still do — the later ones decline at zero cost
    //   (`two_wedged_members_still_leave_the_base_its_reserve`);
    // - the COST side of the same rule, and the only behaviour it takes away: a
    //   perfectly HEALTHY tail is cut short by a positive-but-truncated
    //   allowance and the answer degrades from composed to base-only
    //   (`a_slow_but_healthy_tail_is_cut_short_to_protect_the_bases_reserve`);
    // - below two reserves tails are declined or cut short while the base is
    //   still funded at `min(reserve, remaining)`
    //   (`below_two_reserves_tails_are_declined_but_the_base_is_still_funded`);
    // - the re-base caller's `(max_chain + 2) x reserve` boundary funds the
    //   chain's own base at exactly `remaining == reserve`
    //   (`the_rebase_budget_boundary_funds_the_chains_own_base_at_exactly_one_reserve`);
    // - a wedged member is addressed at most once for the whole selection
    //   (`a_wedged_member_costs_at_most_one_reserve_for_the_whole_selection`);
    // - and the ONE degenerate case — a LIST slow enough to leave under one
    //   reserve — exhausts honestly inside the deadline instead of overrunning
    //   it
    //   (`a_slow_listing_that_eats_the_budget_exhausts_honestly_rather_than_overrunning`).
    //
    // WHICH EXHAUSTION a run-out-of-budget selection reports is a separate
    // claim from the arithmetic above, and it does NOT key on the step-down
    // reasons: it keys on whether every listed `Base` was adjudicated. A
    // speculative tail cut short above fully-adjudicated broken bases leaves
    // the verdict conclusive (`Exhaustion::AllBroken` -> `exhausted` ->
    // terminal `NOT_FOUND`); a base the walk never read leaves it inconclusive
    // (`Exhaustion::BudgetInconclusive` -> `exhausted_budget` -> retriable
    // `UNAVAILABLE`), because an unread base is no evidence that the layer has
    // no backup. The three cases at the end of this group pin that split:
    //
    // - the stock shape, where the answer is inconclusive
    //   (`a_wedged_base_under_a_completed_fetch_exhausts_inconclusively_not_all_broken`);
    // - the negative, where budget step-downs coexist with fully-adjudicated
    //   bases and the verdict stays conclusive
    //   (`budget_step_downs_over_fully_adjudicated_bases_stay_conclusively_all_broken`);
    // - and the two boundaries — a base pruned between LIST and fetch counts as
    //   adjudicated, and a tails-only listing is `NoBase` however the budget
    //   went
    //   (`a_base_pruned_between_list_and_fetch_counts_as_adjudicated`,
    //   `a_tails_only_listing_is_no_base_even_when_the_budget_is_already_gone`).
    //
    // Several cases below assert a NON-ZERO `remaining_ms` on a `Budget` break.
    // That is load-bearing, not incidental: pre-b.bu8 the overrun arms reported
    // a flat `0`, which under the allowance rule would misreport a selection
    // still holding precisely the base's reservation as one with nothing left.
    //
    // A non-zero per-fetch epsilon (`with_fetch_delay`) is likewise
    // load-bearing wherever it appears. The pre-b.bu8 arithmetic failed by
    // exactly the residue the LIST and the walk consume, so a fixture whose
    // fetches are free cannot see the defect at all.

    /// The stock recovery fetch budget, spelled from the same two terms
    /// `Config::checkpoint_recovery_fetch_budget` composes — this module reads
    /// no `Config`, so the fixture states the arithmetic instead of importing
    /// the answer.
    const STOCK_BUDGET: Duration = Duration::from_secs(60);

    /// A per-fetch scheduling epsilon: small against [`RESERVE`], and — the
    /// point — NOT zero.
    ///
    /// The pre-b.bu8 gate declined the base by precisely the residue the LIST
    /// and the walk around a wedged member consumed, so a fixture whose fetches
    /// cost nothing leaves that residue at zero and the defect invisible. Any
    /// small non-zero value works; 1 s keeps the arithmetic in the cases below
    /// readable.
    const FETCH_EPSILON: Duration = Duration::from_secs(1);

    /// The only case below the gather's stated `budget >= 2 x reserve`
    /// precondition, and the one that pins what the allowance rule degrades to
    /// there: **tails are declined, the base is still funded.**
    ///
    /// A sub-two-reserve budget is not a shipped shape — every caller satisfies
    /// the precondition structurally — but the rule must not fall off a cliff
    /// below it, and the direction it degrades in is the whole claim. `Tail`
    /// allowances collapse to zero (speculative work is what gets cut), while
    /// the `Base` allowance `min(reserve, remaining)` keeps funding the floor
    /// answer out of whatever is left.
    ///
    /// Pre-b.bu8 this was the `..._abandons_every_remaining_candidate_unfetched`
    /// case, which asserted `None`: the old threshold gate declined the base
    /// alongside the tails, because the remainder was under one reserve and the
    /// gate could not tell the floor answer's fetch from a speculative one.
    /// That is the defect, so the assertion is re-pointed rather than kept.
    #[tokio::test(start_paused = true)]
    async fn below_two_reserves_tails_are_declined_but_the_base_is_still_funded() {
        // 40 s of budget against a 30 s reserve — under two reserves — and 15 s
        // per fetch.
        //
        //   LIST                     free            remaining 40
        //   T2@20, arm 1             min(30, 40) = 30, costs 15   -> 25
        //   T1@14 as T2's link       min(30, 25 - 30) = 0         -> DECLINED
        //   T1@14 as its own head    min(30, 25 - 30) = 0         -> DECLINED
        //   base@10                  min(30, 25) = 25, costs 15   -> 10, ANSWER
        //
        // The two declines cost no fetch at all, which is the other half of the
        // rule: the un-gated first fetch is paid once per CANDIDATE, so a
        // selection that kept starting speculative fetches it could not fund
        // would burn N truncated reserves of wall clock and blow the outer
        // preemptive timeout — the `Transient` -> `Error` stranding
        // deadline-awareness exists to eliminate.
        let store = FakeStore::new(vec![
            base_gen(10),
            tail_gen(10, 10, 14),
            tail_gen(10, 14, 20),
        ])
        .with_fetch_delay(Duration::from_secs(15));
        let deadline = Instant::now() + Duration::from_secs(40);
        let started = Instant::now();

        let mut selection = gather_restore_lineage(&store, OBJECT_ID, deadline, RESERVE)
            .await
            .expect("listing succeeds");
        let lineage = selection
            .next_restorable()
            .await
            .expect("running out of time is a step-down signal, NOT an error")
            .expect(
                "the floor answer is funded out of what is left even below two reserves — the \
                 base's allowance is `min(reserve, remaining)` and is never declined while any \
                 budget remains",
            );

        assert!(lineage.is_base_only());
        assert_eq!(
            lineage.base.handle,
            CheckpointHandle::base(OBJECT_ID, 10, "4242.10".to_string()),
            "the floor answer is the newest base — the same handle the pre-Epic-4 selection picks"
        );

        let breaks = selection.step_downs();
        assert_eq!(
            breaks.len(),
            2,
            "one record per abandoned candidate, both declined without a fetch; got {breaks:?}"
        );
        for (index, expected_head) in [(0usize, 20u64), (1, 14)] {
            assert_eq!(breaks[index].head_seq, expected_head);
            match &breaks[index].reason {
                LineageBreak::Budget {
                    seq,
                    remaining_ms,
                    reserve_ms,
                } => {
                    assert_eq!(*seq, 14, "the break names the member the walk gave up on");
                    assert_eq!(*reserve_ms, RESERVE.as_millis() as u64);
                    assert!(
                        *remaining_ms < *reserve_ms,
                        "a TAIL is declined precisely because the remainder is under one reserve \
                         — there is nothing left above the base's reservation to fund \
                         speculative work with"
                    );
                }
                other => panic!("expected a Budget break, got {other:?}"),
            }
            assert_eq!(breaks[index].reason.reason_label(), "stepped_down_budget");
        }
        assert_eq!(
            store.get_count(14, CheckpointKind::Tail),
            0,
            "a declined tail must cost NO fetch — a Budget step-down recorded off the clock, not \
             off a probe"
        );
        assert_eq!(
            store.total_gets(),
            2,
            "the head tail and the base, nothing else"
        );
        assert_eq!(
            Instant::now().saturating_duration_since(started),
            Duration::from_secs(30),
            "exactly the two fetches that were funded; the declines must not spend a further \
             truncated reserve per remaining candidate"
        );
    }

    /// The shipped ratio, and the case the pre-b.bu8 arithmetic got wrong.
    ///
    /// `budget = 2 x reserve` is not a corner — it is what
    /// `max(rpc_deadline_ms, 2 x checkpoint_request_timeout_ms)` yields at the
    /// shipped defaults, so this is the posture every production restore runs
    /// in. One wedged tail above a healthy base must still reach the floor.
    ///
    /// It did not. The old rule ran the wedged fetch for a flat `reserve` and
    /// then gated the base on `remaining >= reserve`, so the base was declined
    /// by exactly the residue the LIST and the walk had consumed — the
    /// [`FETCH_EPSILON`] here — and the selection exhausted `AllBroken` with a
    /// perfectly restorable base one step below it. The allowance rule caps the
    /// tail at `remaining - reserve` instead, so what it leaves behind IS the
    /// base's reservation.
    #[tokio::test(start_paused = true)]
    async fn one_wedged_member_cannot_starve_the_floor_at_the_shipped_two_reserve_budget() {
        let mut wedged = tail_gen(10, 10, 14);
        wedged.hangs = true;
        // [T2@20 -> T1@14 -> base@10], T1 wedged, at 60 s / 30 s:
        //
        //   T2@20, arm 1        min(30, 60) = 30, costs 1  -> 59
        //   T1@14 (Tail)        min(30, 59 - 30) = 29, WEDGES for 29 -> 30
        //   T1@14 as own head   cached Wedged, 0 fetches   -> 30
        //   base@10 (Base)      min(30, 30) = 30, costs 1  -> 29, ANSWER
        let store = FakeStore::new(vec![base_gen(10), wedged, tail_gen(10, 14, 20)])
            .with_fetch_delay(FETCH_EPSILON);
        let deadline = Instant::now() + STOCK_BUDGET;
        let started = Instant::now();

        let mut selection = gather_restore_lineage(&store, OBJECT_ID, deadline, RESERVE)
            .await
            .expect("listing succeeds");
        let lineage = selection
            .next_restorable()
            .await
            .expect("a wedged adapter must not surface as an error here")
            .expect(
                "at the SHIPPED budget of two reserves, one wedged tail must leave the base \
                 fundable — an allowanced tail is capped at `remaining - reserve`, so it cannot \
                 spend the reservation",
            );

        assert!(lineage.is_base_only());
        assert_eq!(lineage.base_snap_seq(), 10);

        let breaks = selection.step_downs();
        assert_eq!(
            breaks.len(),
            2,
            "T2 abandons on its wedged link and T1 abandons on the cached fact; the base answers \
             rather than abandoning, so there is no third record. Got {breaks:?}"
        );
        for record in breaks {
            assert_eq!(
                record.reason,
                LineageBreak::Budget {
                    seq: 14,
                    remaining_ms: RESERVE.as_millis() as u64,
                    reserve_ms: RESERVE.as_millis() as u64,
                },
                "both candidates name the same member, and BOTH report the LIVE remainder — \
                 which is exactly the base's reservation. A flat `remaining_ms: 0` here (the \
                 pre-b.bu8 spelling) would report a selection that is about to fund its floor \
                 answer as one with nothing left"
            );
            assert_eq!(record.reason.reason_label(), "stepped_down_budget");
        }

        assert_eq!(
            store.get_count(14, CheckpointKind::Tail),
            1,
            "the wedged member is addressed exactly once for the whole selection"
        );
        assert_eq!(
            store.total_gets(),
            3,
            "T2, the wedged T1 once, and the base that answered"
        );
        let elapsed = Instant::now().saturating_duration_since(started);
        assert!(
            elapsed <= STOCK_BUDGET,
            "the whole selection must land inside the cooperative budget, or the outer \
             preemptive wrapper is the mechanism rather than the backstop; took {elapsed:?} of \
             {STOCK_BUDGET:?}"
        );
    }

    /// The COST side of the reservation, and the only behaviour b.bu8 takes
    /// away: a **healthy** tail cut short by a positive-but-truncated allowance,
    /// degrading a composed answer to the floor.
    ///
    /// Production rustdoc claims this outright — `fetch_member`'s arm 3 says a
    /// `Tail` "may be cut short (a slow-but-healthy tail is an accepted
    /// casualty)" — so it is a contract, not a side effect, and it needs a case
    /// that fails when someone later "optimizes" arm 3 back to
    /// `min(reserve, remaining)` on the grounds that the tail would have
    /// succeeded.
    ///
    /// Nothing here is degraded. The fixture is the sibling above's chain with
    /// the wedge removed and every fetch merely SLOW — `[T2@20 -> T1@14 ->
    /// base@10]`, all three healthy and decodable, 20 s per fetch at the shipped
    /// 60 s / 30 s:
    ///
    /// ```text
    ///   LIST                free                                  remaining 60
    ///   T2@20, arm 1        min(30, 60) = 30, costs 20          -> 40
    ///   T1@14 (Tail)        min(30, 40 - 30) = 10, TRUNCATED    -> 30
    ///   T1@14 as own head   cached Wedged, 0 fetches            -> 30
    ///   base@10 (Base)      min(30, 30) = 30, costs 20          -> 10, ANSWER
    /// ```
    ///
    /// Two consequences worth naming because both look like defects in
    /// isolation. A [`LinkFact::Wedged`] is cached against a generation that was
    /// never wedged — truncation and never-returning are the same fact to this
    /// module, deliberately, because the remedy is identical and re-fetching the
    /// slow one would cost a second allowance. And the returned lineage is
    /// base-only where the same fixture composes under a uniform
    /// `min(reserve, remaining)` rule: T1 would complete at 40 s, the walk would
    /// reach base@10 with 20 s left, and the selection would return the whole
    /// chain.
    ///
    /// That trade is the design. The tail is speculative — it can only improve
    /// on the floor, never produce it — and the 30 s it declined to spend is
    /// exactly what funds the base's own fetch. Buying the composed answer here
    /// means spending the reservation, which is the
    /// `..._cannot_starve_the_floor_...` sibling's failure mode one fixture over.
    #[tokio::test(start_paused = true)]
    async fn a_slow_but_healthy_tail_is_cut_short_to_protect_the_bases_reserve() {
        /// Long enough that a `Tail`'s `remaining - reserve` allowance cannot
        /// cover it, short enough that the un-gated first fetch and the `Base`
        /// arm — both `min(reserve, remaining)` — both can. That gap IS the case.
        const SLOW_FETCH: Duration = Duration::from_secs(20);

        let store = FakeStore::new(vec![
            base_gen(10),
            tail_gen(10, 10, 14),
            tail_gen(10, 14, 20),
        ])
        .with_fetch_delay(SLOW_FETCH);
        let deadline = Instant::now() + STOCK_BUDGET;
        let started = Instant::now();

        let mut selection = gather_restore_lineage(&store, OBJECT_ID, deadline, RESERVE)
            .await
            .expect("listing succeeds");
        let lineage = selection
            .next_restorable()
            .await
            .expect("a truncated fetch is a budget VERDICT, never an error")
            .expect("the base is funded out of the reservation the truncated tail left standing");

        assert!(
            lineage.is_base_only(),
            "a cut-short speculative tail degrades the answer to the floor. A COMPOSED lineage \
             here means the `Tail` allowance was widened back to `min(reserve, remaining)` and \
             the base is no longer reserved out of the budget — the b.bu8 regression this case \
             exists to catch"
        );
        assert_eq!(lineage.base_snap_seq(), 10);
        assert_eq!(
            lineage.base.bytes,
            base_gen(10).bytes,
            "the base's own fetch ran to COMPLETION inside its 30 s allowance against a 20 s \
             cost — it is the tail that was cut short, not the floor answer"
        );

        let breaks = selection.step_downs();
        assert_eq!(
            breaks.len(),
            2,
            "T2 gives up on its truncated link and T1 gives up on the cached fact; the base \
             answers rather than abandoning, so there is no third record. Got {breaks:?}"
        );
        for record in breaks {
            assert_eq!(
                record.reason,
                LineageBreak::Budget {
                    seq: 14,
                    // 60 s less T2's 20 s less T1's 10 s truncated allowance:
                    // the base's whole reservation, still standing.
                    remaining_ms: RESERVE.as_millis() as u64,
                    reserve_ms: RESERVE.as_millis() as u64,
                },
                "the member named is the HEALTHY one that was cut short, and the remainder \
                 reported is the live one — which is precisely the reservation the truncation \
                 preserved"
            );
            assert_eq!(record.reason.reason_label(), "stepped_down_budget");
        }

        assert_eq!(
            store.get_count(14, CheckpointKind::Tail),
            1,
            "the truncated member is addressed exactly ONCE for the whole selection: the \
             `LinkFact::Wedged` fact is cached on truncation just as it is on a never-returning \
             fetch, so the candidate that heads at it does not pay a second allowance"
        );
        assert_eq!(
            store.get_count(10, CheckpointKind::Base),
            1,
            "and the base is addressed once"
        );
        assert_eq!(
            store.total_gets(),
            3,
            "T2, the truncated T1 once, and the base that answered"
        );
        assert_eq!(
            Instant::now().saturating_duration_since(started),
            Duration::from_secs(50),
            "20 s (T2) + 10 s (T1, cut at its `remaining - reserve` allowance) + 20 s (the base, \
             in full). The middle term is the load-bearing one: 30 s of budget the tail was not \
             allowed to spend is what leaves the base fundable at all"
        );
    }

    /// The reservation is a floor, not a one-shot: **N** wedged tails still
    /// leave the base its reserve, because every tail after the first declines
    /// at zero cost.
    ///
    /// This is the property a per-member cost model does not give you. If each
    /// wedged member spent an allowance, two of them would spend two and the
    /// base would be starved at the shipped two-reserve budget however the
    /// first one was capped. What actually bounds it is that the `Tail`
    /// allowance is taken off the LIVE remainder: once the remainder is down to
    /// the reservation, every later tail's allowance is zero, and a zero
    /// allowance is a decline WITHOUT a fetch.
    #[tokio::test(start_paused = true)]
    async fn two_wedged_members_still_leave_the_base_its_reserve() {
        let mut wedged_low = tail_gen(10, 10, 14);
        wedged_low.hangs = true;
        let mut wedged_high = tail_gen(10, 14, 20);
        wedged_high.hangs = true;
        // [T3@30 -> T2@20 -> T1@14 -> base@10], T2 and T1 wedged, at 60 s / 30 s:
        //
        //   T3@30, arm 1        min(30, 60) = 30, costs 1  -> 59
        //   T2@20 (Tail)        min(30, 59 - 30) = 29, WEDGES for 29 -> 30
        //   T2@20 as own head   cached Wedged, 0 fetches   -> 30
        //   T1@14 (Tail)        min(30, 30 - 30) = 0       -> DECLINED, 0 fetches
        //   base@10 (Base)      min(30, 30) = 30, costs 1  -> 29, ANSWER
        let store = FakeStore::new(vec![
            base_gen(10),
            wedged_low,
            wedged_high,
            tail_gen(10, 20, 30),
        ])
        .with_fetch_delay(FETCH_EPSILON);
        let deadline = Instant::now() + STOCK_BUDGET;
        let started = Instant::now();

        let mut selection = gather_restore_lineage(&store, OBJECT_ID, deadline, RESERVE)
            .await
            .expect("listing succeeds");
        let lineage = selection
            .next_restorable()
            .await
            .expect("a wedged adapter must not surface as an error here")
            .expect("a SECOND wedged member must not cost the base its reservation either");

        assert!(lineage.is_base_only());
        assert_eq!(lineage.base_snap_seq(), 10);

        assert_eq!(
            store.get_count(14, CheckpointKind::Tail),
            0,
            "the second wedged member must be declined UN-FETCHED. This is the reservation doing \
             its work: by the time the walk reaches it the remainder is exactly one reserve, so \
             its `min(reserve, remaining - reserve)` allowance is zero and it costs nothing. A \
             `1` here means wedged members cost an allowance EACH and the base is starved by the \
             second one"
        );
        assert_eq!(
            store.get_count(20, CheckpointKind::Tail),
            1,
            "the first wedged member is the only one that gets to spend anything, and it spends \
             it once"
        );
        assert_eq!(
            store.total_gets(),
            3,
            "T3, the first wedged member once, and the base"
        );
        assert_eq!(
            selection.step_downs().len(),
            3,
            "T3 on its wedged link, T2 on the cached fact, T1 on a zero allowance; got {:?}",
            selection.step_downs()
        );
        for record in selection.step_downs() {
            assert_eq!(record.reason.reason_label(), "stepped_down_budget");
        }
        let elapsed = Instant::now().saturating_duration_since(started);
        assert!(
            elapsed <= STOCK_BUDGET,
            "N wedged members must not cost N allowances of wall clock; took {elapsed:?} of \
             {STOCK_BUDGET:?}"
        );
    }

    /// The accepted degenerate case, fenced: a LIST slow enough to eat most of
    /// the budget must **exhaust honestly inside the deadline**, never overrun
    /// it.
    ///
    /// The LIST is the one cost in the walk that no cap covers. The caller's
    /// deadline is sampled before it and consulted only after, and the
    /// adapter's enumeration is bounded by its own ceiling (540 s at stock
    /// knobs) rather than by anything this module owns. So a LIST can return
    /// having already spent most of the budget, and the un-gated first fetch —
    /// the one arm carrying no budget precondition — is what runs next.
    ///
    /// Pre-b.bu8 that fetch ran a **flat** reserve regardless of the remainder:
    /// 45 s of LIST plus 30 s of wedged fetch = 75 s against a 60 s budget, so
    /// the outer preemptive wrapper elapsed first and the caller got
    /// `Transient` -> `RecoveryOutcome::Error`. Arm 1's `min(reserve,
    /// remaining)` clamps it to the 15 s actually left, so the **cooperative**
    /// verdict is ready first and the walk ends at `Ok(None)`.
    ///
    /// Two things this case pins, and they are separable:
    ///
    /// 1. **The mechanism**, which b.bu8 fixed and nothing since has changed:
    ///    the selection stays *inside its own deadline*, so the cooperative
    ///    step-down is what answers and the preemptive wrapper stays the
    ///    backstop — the invariant `RecoveryHook::try_restore_layer`'s
    ///    tie-break note depends on. That is the `elapsed == STOCK_BUDGET`
    ///    equality and the `stepped_down_budget` reasons below, and neither
    ///    moves.
    /// 2. **The verdict**, which b.hxk re-pointed. b.bu8 shipped this trace as
    ///    [`Exhaustion::AllBroken`], and accepted the resulting terminal
    ///    `NOT_FOUND` where a flat reserve had produced a retriable
    ///    `UNAVAILABLE`. That acceptance was wrong on the evidence: base@10 was
    ///    declined at a zero allowance, so the selection never read the only
    ///    base in the listing and held nothing that says the layer is
    ///    unrestorable. The verdict is now
    ///    [`Exhaustion::BudgetInconclusive`], whose `exhausted_budget` outcome
    ///    `select_restore_source` maps back to `Transient` -> `UNAVAILABLE`.
    ///    So the outcome swap b.bu8 documented as accepted no longer exists,
    ///    and the clamp costs the caller a retry rather than a restorable layer
    ///    written off.
    #[tokio::test(start_paused = true)]
    async fn a_slow_listing_that_eats_the_budget_exhausts_honestly_rather_than_overrunning() {
        let mut wedged = tail_gen(10, 10, 14);
        wedged.hangs = true;
        // 45 s of LIST against a 60 s budget and a 30 s reserve:
        //
        //   LIST                costs 45                   -> 15
        //   T1@14, arm 1        min(30, 15) = 15, WEDGES for 15 -> 0
        //   base@10 (Base)      min(30, 0) = 0             -> DECLINED, 0 fetches
        let store = FakeStore::new(vec![base_gen(10), wedged])
            .with_list_delay(Duration::from_secs(45))
            .with_fetch_delay(Duration::ZERO);
        let deadline = Instant::now() + STOCK_BUDGET;
        let started = Instant::now();

        let mut selection = gather_restore_lineage(&store, OBJECT_ID, deadline, RESERVE)
            .await
            .expect("listing succeeds");
        let answer = selection
            .next_restorable()
            .await
            .expect("running out of time is a step-down signal, NOT an error");

        assert!(
            answer.is_none(),
            "the budget really is gone — this case is about HOW it runs out, not about salvaging \
             an answer from it"
        );
        // THE CLAIM, as an equality rather than a bound.
        //
        // `<=` was the honest fence while the clamp was a DURATION: a
        // `timeout(min(reserve, remaining))` re-anchors where the timer is
        // constructed, so the elapse landed at `deadline + delta` for some
        // unknowable anchor gap and only "at most the budget, near enough" was
        // assertable. `timeout_at(min(now + reserve, deadline))` makes the
        // number exact — the timer IS the deadline here, because 45 s of LIST
        // leaves 15 s and `now + 30 > deadline` — so under the paused clock the
        // selection returns at precisely `started + STOCK_BUDGET`.
        //
        // Equality is therefore the tightest TRUE assertion, and it is strictly
        // stronger in the direction `<=` was blind to: an arm-1 fetch that
        // declined outright (spending 0 rather than the 15 s it was clamped to)
        // satisfies `<=` while breaking the "clamped, not skipped" half of the
        // claim. `<` would simply be false.
        let elapsed = Instant::now().saturating_duration_since(started);
        assert_eq!(
            elapsed, STOCK_BUDGET,
            "THE CLAIM. The un-gated first fetch must be clamped to the 15 s the LIST left — \
             spending exactly that and no more. Above {STOCK_BUDGET:?} means it ran a flat \
             {RESERVE:?} regardless, which puts the selection at 75 s, elapses the outer \
             preemptive wrapper first, and hands the caller the `Transient` -> `Error` stranding \
             the cooperative verdict exists to replace. Below it means the fetch was skipped \
             rather than clamped. Observed {elapsed:?}"
        );
        let breaks = selection.step_downs();
        assert_eq!(
            breaks.len(),
            2,
            "the wedged head, then the base declined at a zero allowance; got {breaks:?}"
        );
        for record in breaks {
            assert_eq!(
                record.reason.reason_label(),
                "stepped_down_budget",
                "an exhausted budget is a Budget verdict on EVERY candidate — nothing here is \
                 reclassified as a broken link or an error"
            );
        }
        assert_eq!(
            store.get_count(10, CheckpointKind::Base),
            0,
            "with nothing left, even the base is declined un-fetched — the `Base` arm is \
             `min(reserve, remaining)`, and `remaining` is zero"
        );
        assert_eq!(
            selection.exhaustion(),
            Some(&Exhaustion::BudgetInconclusive {
                candidates: 2,
                unadjudicated_bases: 1,
            }),
            "candidates were abandoned, so the exhaustion is a loud arm — and it is the \
             INCONCLUSIVE one, because base@10 was declined at a zero allowance and so was never \
             read. A selection holding no answer about the only base in the listing cannot claim \
             the layer is unrestorable (b.hxk); `AllBroken` here would report a store that was \
             merely slow as a layer with no usable backup, terminally. Got {:?}",
            selection.exhaustion()
        );
        assert_eq!(
            selection
                .exhaustion()
                .expect("the selection exhausted")
                .outcome_label(),
            "exhausted_budget",
            "and it lands on its OWN metric cell, not the conclusive `exhausted` one"
        );
    }

    #[tokio::test(start_paused = true)]
    async fn the_floor_answer_is_reached_while_one_reserve_of_budget_remains() {
        // The honest floor bound, positive arm. The head candidate is abandoned
        // (a decode break, one fetch), and because the walk above left more
        // than one reserve the floor answer is funded and returned. This is the
        // bound the budget's rustdoc states — NOT "the floor always fits
        // however many candidates were abandoned above it", which no budget
        // value can buy.
        let store = FakeStore::new(vec![base_gen(10), corrupted(tail_gen(10, 10, 14))])
            .with_fetch_delay(Duration::from_secs(10));
        let deadline = Instant::now() + STOCK_BUDGET;
        let started = Instant::now();

        let mut selection = gather_restore_lineage(&store, OBJECT_ID, deadline, RESERVE)
            .await
            .expect("listing succeeds");
        let lineage = selection
            .next_restorable()
            .await
            .expect("no error")
            .expect("a walk that leaves one reserve above the floor still reaches the floor");

        assert!(lineage.is_base_only());
        assert_eq!(
            lineage.base.handle,
            CheckpointHandle::base(OBJECT_ID, 10, "4242.10".to_string()),
            "the floor answer is the newest base — the same handle the pre-Epic-4 selection picks"
        );
        assert_eq!(
            selection.step_downs()[0].reason.reason_label(),
            "stepped_down_decode"
        );
        let elapsed = Instant::now().saturating_duration_since(started);
        assert!(
            elapsed + RESERVE <= STOCK_BUDGET,
            "the fixture must actually leave a reserve above the floor, or it pins the wrong \
             bound; the selection took {elapsed:?} of {STOCK_BUDGET:?}"
        );
    }

    /// The b.gg7 reconciliation this Subtask exists for. The StorageApi
    /// adapter's ENFORCED per-call ceiling is `request_timeout x 3 retries x 2
    /// auth attempts` = 180 s — 6x the reserve — so on a degraded backend a
    /// single fetch outlives the whole 60 s outer preemptive budget and the
    /// cooperative reserve check never gets to run. Without the per-fetch cap,
    /// the hook's `tokio::time::timeout` would fire first and map to
    /// `Transient` -> `RecoveryOutcome::Error`, stranding a restorable base one
    /// generation behind. With it, the same store yields a budget step-down.
    #[tokio::test(start_paused = true)]
    async fn a_fetch_that_never_returns_becomes_a_budget_step_down_not_an_error() {
        let mut wedged = tail_gen(10, 10, 14);
        wedged.hangs = true;
        let store = FakeStore::new(vec![base_gen(10), wedged]);
        let deadline = Instant::now() + STOCK_BUDGET;
        let started = Instant::now();

        let mut selection = gather_restore_lineage(&store, OBJECT_ID, deadline, RESERVE)
            .await
            .expect("listing succeeds");
        let lineage = selection
            .next_restorable()
            .await
            .expect("a wedged adapter must not surface as an error here")
            .expect("the base behind the wedged tail is still restorable");

        assert!(lineage.is_base_only());
        assert_eq!(lineage.base_snap_seq(), 10);
        assert_eq!(
            selection.step_downs()[0].reason.reason_label(),
            "stepped_down_budget"
        );
        assert_eq!(selection.step_downs()[0].reason.link_seq(), 14);
        let elapsed = Instant::now().saturating_duration_since(started);
        assert!(
            elapsed >= RESERVE && elapsed < STOCK_BUDGET,
            "the wedged fetch must be cut at the RESERVE ({RESERVE:?}), well inside the outer \
             preemptive budget ({STOCK_BUDGET:?}); observed {elapsed:?}"
        );
    }

    /// The multi-candidate half of the same reconciliation, and the reason the
    /// overrun is cached as a selection-wide FACT rather than merely returned.
    ///
    /// A wedged member is almost never reached by one candidate only: a broken
    /// candidate's members are themselves the next candidates, so T1 here is
    /// both a link of T2's lineage and the head of its own. Pre-fix that cost
    /// two reserves — one inside T2's walk, a second on candidate T1's own
    /// un-gated first fetch — and against a 60 s budget the outer preemptive
    /// timeout fired before base@10 was ever addressed: `Transient` -> `Error`,
    /// with a perfectly restorable base sitting right there. With the fact
    /// cached, T1's candidate consults it, spends nothing, and the floor lands
    /// one reserve in.
    ///
    /// "At most one reserve" rather than "one": post-b.bu8 an allowanced `Tail`
    /// is capped at `remaining - reserve`, so what a wedged tail actually
    /// spends is bounded by the reserve and equals it only when the remainder
    /// is at least two of them — which it is here, with free fetches against
    /// the shipped two-reserve budget. The
    /// `..._at_the_shipped_two_reserve_budget` sibling above runs the same
    /// topology with a non-zero per-fetch epsilon, where the cap bites and the
    /// spend is strictly less.
    #[tokio::test(start_paused = true)]
    async fn a_wedged_member_costs_at_most_one_reserve_for_the_whole_selection() {
        let mut wedged = tail_gen(10, 10, 14);
        wedged.hangs = true;
        // [T2@20 -> T1@14 -> base@10], T1 wedged. T1 is a member of T2's
        // lineage AND the next candidate in its own right.
        let store = FakeStore::new(vec![base_gen(10), wedged, tail_gen(10, 14, 20)]);
        let deadline = Instant::now() + STOCK_BUDGET;
        let started = Instant::now();

        let mut selection = gather_restore_lineage(&store, OBJECT_ID, deadline, RESERVE)
            .await
            .expect("listing succeeds");
        let lineage = selection
            .next_restorable()
            .await
            .expect("a wedged adapter must not surface as an error here")
            .expect("the base below the wedged member is still restorable");

        assert!(lineage.is_base_only());
        assert_eq!(lineage.base_snap_seq(), 10);

        assert_eq!(
            store.get_count(14, CheckpointKind::Tail),
            1,
            "the wedged member must be addressed EXACTLY once across the whole selection — the \
             second candidate to reach it consults the cached overrun and fetches nothing"
        );
        assert_eq!(
            store.total_gets(),
            3,
            "T2, the wedged T1 once, and the base"
        );

        // Both candidates that reached T1 record the same verdict about the
        // same member, and the one that reached it through the fact records it
        // without a probe.
        let breaks = selection.step_downs();
        assert_eq!(breaks.len(), 2, "got {breaks:?}");
        assert_eq!(breaks[0].head_seq, 20);
        assert_eq!(breaks[1].head_seq, 14);
        for record in breaks {
            assert_eq!(
                record.reason,
                LineageBreak::Budget {
                    seq: 14,
                    remaining_ms: RESERVE.as_millis() as u64,
                    reserve_ms: RESERVE.as_millis() as u64,
                },
                "a fact-cached skip is still a Budget-reason step-down for its own candidate, \
                 recorded identically to the overrun that discovered it — INCLUDING the \
                 remainder, which both measure fresh. With free fetches against a two-reserve \
                 budget the wedged tail's allowance is a full reserve, so exactly one reserve is \
                 left when either record is minted. Pre-b.bu8 both spelled this `0`, which under \
                 the allowance rule would report a selection about to fund its floor answer as \
                 one with nothing left"
            );
            assert_eq!(record.reason.reason_label(), "stepped_down_budget");
        }

        // "At most one reserve" is the CONTRACT; on this fixture the spend is
        // exactly one, and that is what gets asserted.
        //
        // The rustdoc above already states when the two coincide: an allowanced
        // `Tail` is capped at `remaining - reserve`, which equals a full reserve
        // precisely when the remainder is at least two of them — free fetches
        // against the shipped two-reserve budget, i.e. here. Under
        // `timeout_at(min(now + reserve, deadline - reserve))` that cap is an
        // instant, so the wedged fetch elapses at exactly `started + RESERVE`
        // and the base (free, and reached with a reserve still standing)
        // returns without advancing the clock further.
        //
        // Equality rather than `<=` for the same reason as the slow-LIST case:
        // it additionally catches a regression that stopped spending anything
        // on the wedged member — a cached-fact lookup that fired on the FIRST
        // candidate rather than the second would satisfy `<=` while proving
        // nothing was ever probed. The `store.get_count(14, ..) == 1` assertion
        // above covers that one from the other side; keeping both is what makes
        // "addressed exactly once, for exactly one reserve" a single readable
        // claim. The sibling with a non-zero fetch epsilon is where the cap
        // bites and the spend is strictly less — it asserts its own number.
        let elapsed = Instant::now().saturating_duration_since(started);
        assert_eq!(
            elapsed, RESERVE,
            "EXACTLY one reserve is spent on the wedged member. Above it, a second candidate paid \
             its own truncated reserve for the same member — which leaves the floor unfunded and, \
             in production, elapses the outer preemptive timeout. Observed {elapsed:?} against a \
             {RESERVE:?} reserve"
        );
        assert!(
            elapsed + RESERVE <= STOCK_BUDGET,
            "and a reserve is still standing when the base is reached. That is the PRECONDITION \
             this case runs under — the shipped budget is always at least two reserves — not the \
             mechanism: what actually holds the base's reserve back is the allowance rule, which \
             caps a speculative tail at `remaining - reserve`. See the \
             `two_wedged_members_still_leave_the_base_its_reserve` sibling, where the \
             precondition alone would not survive a second wedged member"
        );
    }

    /// The **re-base caller's boundary**, and the case a uniform speculative
    /// rule would break.
    ///
    /// `rebase_layer_lineage_with_probe` passes `(max_chain + 2) x reserve`,
    /// and its own comment states why the `+2` is load-bearing rather than
    /// slack. Post-b.bu8 that caller additionally REQUIRES the `Base` arm to be
    /// `min(reserve, remaining)` and not the speculative `min(reserve,
    /// remaining - reserve)`: at the boundary — a full-reserve listing plus K
    /// full-reserve tails — the chain's own base is reached at exactly
    /// `remaining == reserve`, where the speculative form yields zero and
    /// declines it. A decline there is a `LineageBreak::Budget`, which the
    /// re-base's no-step-down rule turns into a refusal, on EVERY attempt —
    /// recreating precisely the permanent chain-full wedge SR-4.1 exists to
    /// remove.
    ///
    /// Deliberately a **design fence** rather than a regression fence: the
    /// pre-b.bu8 gate also funded this base (`remaining >= reserve` holds at
    /// the boundary, non-strictly), so this case does not fail without the fix.
    /// It fails against the tempting simplification of applying one uniform
    /// allowance to every member, which is why it is cheap to keep.
    /// `e2e_rebase_lifecycle_test.rs` fences the caller end-to-end, but its
    /// gathers run under `no_deadline()`, where every allowance is a full
    /// reserve by construction and this boundary is unreachable.
    #[tokio::test(start_paused = true)]
    async fn the_rebase_budget_boundary_funds_the_chains_own_base_at_exactly_one_reserve() {
        /// `Config::checkpoint_tail_max_chain`'s shipped default. Named rather
        /// than imported because this module reads no `Config`.
        const MAX_CHAIN: u32 = 8;
        /// `rebase_layer_lineage_with_probe`'s budget formula, spelled out.
        const REBASE_BUDGET: Duration = Duration::from_secs((MAX_CHAIN as u64 + 2) * 30);

        // K = 8 slow-but-HEALTHY tails over one base, every fetch and the
        // listing costing a full reserve — the worst case the `+2` is sized
        // for:
        //
        //   LIST                 costs 30                  -> 270
        //   T8 (head), arm 1     min(30, 270) = 30, costs 30 -> 240
        //   T7..T1 (Tail)        min(30, remaining - 30) = 30 each, 7 x 30 -> 30
        //   base@10 (Base)       min(30, 30) = 30 -- THE BOUNDARY -- costs 30 -> 0
        //
        // A uniform `remaining - reserve` rule gives the base `min(30, 0) = 0`
        // at that last step and declines it.
        let mut generations = vec![base_gen(10)];
        for index in 0..MAX_CHAIN as u64 {
            let from_seq = 10 + index * 2;
            generations.push(tail_gen(10, from_seq, from_seq + 2));
        }
        let store = FakeStore::new(generations)
            .with_list_delay(RESERVE)
            .with_fetch_delay(RESERVE);
        let started = Instant::now();

        let mut selection =
            gather_restore_lineage(&store, OBJECT_ID, Instant::now() + REBASE_BUDGET, RESERVE)
                .await
                .expect("listing succeeds");
        let lineage = selection
            .next_restorable()
            .await
            .expect("a slow-but-healthy adapter must not surface as an error")
            .expect(
                "the whole chain must compose at the re-base caller's own budget boundary — a \
                 refusal here is the permanent chain-full wedge SR-4.1 exists to remove",
            );

        assert_tails(
            &lineage,
            &[
                (10, 12),
                (12, 14),
                (14, 16),
                (16, 18),
                (18, 20),
                (20, 22),
                (22, 24),
                (24, 26),
            ],
        );
        assert_eq!(lineage.base_snap_seq(), 10);
        assert!(
            selection.step_downs().is_empty(),
            "a slow-but-healthy chain at the caller's own boundary must abandon NOTHING; got {:?}",
            selection.step_downs()
        );
        assert_eq!(
            store.total_gets(),
            MAX_CHAIN as usize + 1,
            "K tails plus the base, each fetched exactly once"
        );
        assert_eq!(
            Instant::now().saturating_duration_since(started),
            REBASE_BUDGET,
            "the walk consumes the budget EXACTLY — which is the boundary claim: the base's own \
             fetch is sampled at `remaining == reserve` and funded with a full one, so it ends \
             precisely at the deadline. Anything less and the fixture is not at the boundary and \
             the case pins nothing"
        );
    }

    #[tokio::test]
    async fn budget_exhaustion_is_a_distinct_reason_from_a_broken_link() {
        // Separable in the returned record and, because the label is what the
        // tracing event carries, in the log line too.
        let budget = LineageBreak::Budget {
            seq: 14,
            remaining_ms: 10,
            reserve_ms: 30_000,
        };
        let absent = LineageBreak::LinkAbsent {
            seq: 14,
            kind: CheckpointKind::Tail,
            at_fetch: false,
        };
        assert_ne!(budget.reason_label(), absent.reason_label());
        assert_eq!(budget.reason_label(), "stepped_down_budget");
        assert_ne!(budget, absent);
    }

    /// The SR-4.1 chain-collapse re-base's shape (`rebase_layer_lineage`): it
    /// runs on the checkpoint tick, not under an
    /// RPC, so it must be able to call the entry point without synthesizing an
    /// RPC-shaped budget. The per-member reserve still applies — it is what
    /// caps one over-running adapter fetch — but nothing is abandoned for time.
    #[tokio::test(start_paused = true)]
    async fn a_caller_with_no_deadline_gathers_a_long_lineage_without_an_rpc_budget() {
        let store = FakeStore::new(vec![
            base_gen(10),
            tail_gen(10, 10, 14),
            tail_gen(10, 14, 20),
            tail_gen(10, 20, 26),
            tail_gen(10, 26, 32),
        ])
        // 20 s per fetch — inside the per-member reserve (which still applies
        // to a deadline-free caller), but five of them total 100 s, well past
        // the 60 s an RPC-shaped budget would have granted.
        .with_fetch_delay(Duration::from_secs(20));
        let started = Instant::now();

        let mut selection = gather_restore_lineage(&store, OBJECT_ID, no_deadline(), RESERVE)
            .await
            .expect("listing succeeds");
        let lineage = selection
            .next_restorable()
            .await
            .expect("no error")
            .expect("the whole lineage gathers");

        assert_tails(&lineage, &[(10, 14), (14, 20), (20, 26), (26, 32)]);
        assert_eq!(lineage.head_seq, 32);
        assert!(
            selection.step_downs().is_empty(),
            "a deadline-free caller must never see a budget step-down"
        );
        assert!(
            Instant::now().saturating_duration_since(started) > STOCK_BUDGET,
            "the walk must outlast an RPC-shaped budget, or this case does not distinguish a \
             deadline-free caller from a budgeted one"
        );
        assert!(
            NO_DEADLINE_HORIZON > STOCK_BUDGET * 100,
            "the horizon must be far past any RPC-shaped budget to be meaningful"
        );
    }

    /// The b.hxk regression, on the **stock** store shape: a selection that
    /// completed a member fetch and still never read a listed base exhausts
    /// [`Exhaustion::BudgetInconclusive`], not [`Exhaustion::AllBroken`].
    ///
    /// The shape is `retain_generations = 2` plus a tail — `[T@20, base@10,
    /// base@5]` — which is what a store under the shipped knobs actually holds.
    /// The slow LIST is the production trigger: it is the one cost no cap
    /// covers, so a degraded adapter routinely hands the walk a remainder under
    /// one reserve.
    ///
    /// ```text
    ///   LIST                costs 25                          -> 35
    ///   T@20, arm 1         min(30, 35) = 30, costs 10        -> 25   COMPLETES
    ///   base@10 (Base)      min(30, 25) = 25, WEDGES for 25   -> 0    Wedged
    ///   base@10 as own head cached Wedged, 0 fetches          -> 0
    ///   base@5 (Base)       min(30, 0) = 0                    -> DECLINED, 0 fetches
    /// ```
    ///
    /// **Why `unadjudicated_bases == 2` and not 1.** The predicate counts every
    /// listed `Base` the walk got no answer about, and the two bases here are
    /// unadjudicated for *different* reasons that the rule deliberately treats
    /// alike: base@10 carries a [`LinkFact::Wedged`] (its fetch ate the whole
    /// allowance without answering) and base@5 carries no fact at all (declined
    /// at a zero allowance, so `facts_of` never inserted). Neither says
    /// anything about the artifact, so neither licenses a terminal verdict.
    ///
    /// **Why the completed fetch is asserted.** The ticket's first-cut
    /// predicate was "no fetch ever completed", and this fixture is exactly the
    /// case it gets wrong: T@20's fetch completed and classified fine — proven
    /// below by the first step-down naming base@10 rather than T@20, which is
    /// only reachable once T@20's envelope header has been read — yet the
    /// selection still holds nothing about either base. Keying on bases rather
    /// than on fetch completion is what makes the stock shape retriable.
    #[tokio::test(start_paused = true)]
    async fn a_wedged_base_under_a_completed_fetch_exhausts_inconclusively_not_all_broken() {
        /// Leaves 35 s of a 60 s budget — over one reserve, so the head fetch
        /// is not itself clamped, and under two, so the base's wedge can eat
        /// the rest. That gap IS the case.
        const SLOW_LIST: Duration = Duration::from_secs(25);
        /// Small against [`RESERVE`] and non-zero, so the wedge is reached with
        /// a strictly-positive-but-sub-reserve remainder.
        const FETCH_COST: Duration = Duration::from_secs(10);

        let mut wedged_base = base_gen(10);
        wedged_base.hangs = true;
        let store = FakeStore::new(vec![base_gen(5), wedged_base, tail_gen(10, 10, 20)])
            .with_list_delay(SLOW_LIST)
            .with_fetch_delay(FETCH_COST);
        let deadline = Instant::now() + STOCK_BUDGET;
        let started = Instant::now();

        let mut selection = gather_restore_lineage(&store, OBJECT_ID, deadline, RESERVE)
            .await
            .expect("listing succeeds");
        let answer = selection
            .next_restorable()
            .await
            .expect("running out of time is a step-down signal, NOT an error");

        assert!(
            answer.is_none(),
            "the budget really is gone — this case is about WHICH exhaustion that is, not about \
             salvaging an answer from it"
        );
        assert_eq!(
            selection.exhaustion(),
            Some(&Exhaustion::BudgetInconclusive {
                candidates: 3,
                unadjudicated_bases: 2,
            }),
            "THE CLAIM (b.hxk). Neither base was ever adjudicated — base@10 wedged, base@5 was \
             declined un-fetched — so this selection holds no evidence that the layer is \
             unrestorable. `AllBroken` here is the defect: it folds onto `exhausted`, answers \
             `Ok(None)`, and hands the client a TERMINAL `NOT_FOUND` claiming the layer has no \
             usable backup, on the strength of the store having been slow. Got {:?}",
            selection.exhaustion()
        );
        assert_eq!(
            selection
                .exhaustion()
                .expect("the selection exhausted")
                .outcome_label(),
            "exhausted_budget",
            "and the operator-visible cell is the one that does NOT mean checkpoint loss"
        );

        // The completed fetch, from two sides.
        assert_eq!(
            store.get_count(20, CheckpointKind::Tail),
            1,
            "the head tail was addressed once"
        );
        let breaks = selection.step_downs();
        assert_eq!(
            breaks.len(),
            3,
            "T@20 on its wedged base, base@10 on the cached fact, base@5 on a zero allowance; got \
             {breaks:?}"
        );
        assert_eq!(breaks[0].head_seq, 20);
        assert_eq!(
            breaks[0].reason.link_seq(),
            10,
            "T@20's break names the BASE it walked down to, not T@20 itself — which is only \
             reachable once T@20's own fetch completed and its envelope header was read. This is \
             the assertion that distinguishes the shipped predicate from the ticket's first-cut \
             `no fetch ever completed` one"
        );
        for record in breaks {
            assert_eq!(record.reason.reason_label(), "stepped_down_budget");
        }

        assert_eq!(
            store.get_count(10, CheckpointKind::Base),
            1,
            "the wedged base is addressed exactly once for the whole selection"
        );
        assert_eq!(
            store.get_count(5, CheckpointKind::Base),
            0,
            "and the older base is never addressed at all — with the remainder at zero its \
             `min(reserve, remaining)` allowance is zero. That is precisely why the selection \
             cannot speak for it"
        );
        assert_eq!(
            store.total_gets(),
            2,
            "the head tail and the wedged base, nothing else"
        );
        assert_eq!(
            Instant::now().saturating_duration_since(started),
            STOCK_BUDGET,
            "and the whole selection still lands EXACTLY on its own deadline — the b.bu8 fence is \
             untouched by this reclassification"
        );
    }

    /// The predicate is **SOME listed base unadjudicated**, not *all* of them: a
    /// selection that read one base and got no answer about another still
    /// exhausts [`Exhaustion::BudgetInconclusive`], carrying a count of 1.
    ///
    /// Every other b.hxk case is all-or-nothing — 2-of-2 unread directly above,
    /// 0-of-2 in the negative below, 1-of-1 in the single-base cases, one base
    /// apiece in both integration arms — so a mis-implementation reading "ALL
    /// listed bases unadjudicated ⇒ inconclusive" satisfies the entire set
    /// without this case. It is not an exotic mis-reading to guard against
    /// either: it reinstates the defect at exactly the shape production holds.
    /// `retain_generations = 2` (the shipped default) means two bases are
    /// listed, and one of them readable-and-broken while the other goes unread
    /// is an ordinary degraded-store trace. Under the wrong reading that trace
    /// answers [`Exhaustion::AllBroken`] → terminal `NOT_FOUND`, on the strength
    /// of a base nobody ever read.
    ///
    /// `[base@10 wedged, base@5 read and unusable]`, at the stock ratio:
    ///
    /// ```text
    ///   base@10 (Base), arm 1   min(30, 60) = 30, WEDGES for 30  -> 30   Wedged
    ///   base@5  (Base)          min(30, 30) = 30, costs 10       -> 20   Unusable
    /// ```
    ///
    /// The two bases are adjudicated **differently on purpose**, and the mixture
    /// is the whole case. base@5 was fetched and classified — an adapter
    /// returning a foreign `object_id`, which is positive evidence about that
    /// artifact — while base@10 ate its entire allowance without answering. One
    /// answer beside one silence is still a silence, so the verdict stays
    /// retriable and the count reports the one base the operator's store never
    /// spoke for.
    ///
    /// Note the fixture leaves budget on the table (20 s of 60 s): base@5 was
    /// not declined for want of allowance, it was genuinely read. That is what
    /// separates this trace from the 2-of-2 case above, where the second base's
    /// allowance was zero.
    #[tokio::test(start_paused = true)]
    async fn one_read_base_beside_one_wedged_base_still_exhausts_inconclusively() {
        /// Short against [`RESERVE`], so base@5's own `min(reserve, remaining)`
        /// allowance comfortably funds a completed fetch after the wedge above
        /// it has spent a full reserve.
        const FETCH_COST: Duration = Duration::from_secs(10);

        let foreign = |mut generation: Generation| {
            generation.returned_object_id = Some(OBJECT_ID + 1);
            generation
        };
        let mut wedged_base = base_gen(10);
        wedged_base.hangs = true;
        let store =
            FakeStore::new(vec![foreign(base_gen(5)), wedged_base]).with_fetch_delay(FETCH_COST);
        let deadline = Instant::now() + STOCK_BUDGET;

        let mut selection = gather_restore_lineage(&store, OBJECT_ID, deadline, RESERVE)
            .await
            .expect("listing succeeds");
        let answer = selection
            .next_restorable()
            .await
            .expect("a wedged fetch is a step-down signal, NOT an error");

        assert!(
            answer.is_none(),
            "one base wedged and the other is unusable, so nothing restores"
        );
        assert_eq!(
            selection.exhaustion(),
            Some(&Exhaustion::BudgetInconclusive {
                candidates: 2,
                unadjudicated_bases: 1,
            }),
            "THE CLAIM. The predicate is SOME listed base unadjudicated, not ALL of them: base@5 \
             was read and is genuinely unusable, but base@10 wedged, and a selection that never \
             read base@10 cannot claim the layer is unrestorable. An `AllBroken` here is the \
             mis-implementation this case exists to catch — it reads `1 of 2 unread` as `not all \
             unread` and hands a client a terminal `NOT_FOUND` at the stock \
             `retain_generations = 2` shape. Got {:?}",
            selection.exhaustion()
        );
        assert_eq!(
            selection
                .exhaustion()
                .expect("the selection exhausted")
                .outcome_label(),
            "exhausted_budget",
            "and it lands on the cell that does NOT mean checkpoint loss"
        );

        // Anti-vacuity: the two bases really were adjudicated by different
        // mechanisms. Without this the fixture could degenerate into the
        // all-unread shape the case above already owns, and the discrimination
        // would be lost.
        let labels: Vec<&str> = selection
            .step_downs()
            .iter()
            .map(|record| record.reason.reason_label())
            .collect();
        assert_eq!(
            labels,
            ["stepped_down_budget", "stepped_down_link_mismatch"],
            "base@10 stepped down on its budget and base@5 on its foreign `object_id` — a MIXED \
             adjudication. Two budget verdicts would make this the all-unread case; two mismatches \
             would make it the fully-adjudicated negative"
        );

        assert_eq!(
            store.get_count(10, CheckpointKind::Base),
            1,
            "the wedged base was addressed exactly once and answered nothing — the unadjudicated \
             one"
        );
        assert_eq!(
            store.get_count(5, CheckpointKind::Base),
            1,
            "and the older base WAS read to completion — the adjudicated one. If this were 0 the \
             fixture would be the all-unread shape and could not distinguish the two predicates"
        );
        assert_eq!(store.total_gets(), 2, "nothing else was addressed");
    }

    /// [`RestoreSelection::unadjudicated_bases`] counts **distinct sequences**:
    /// a listing holding the same `(seq, Base)` handle twice contributes ONE.
    ///
    /// The `seen.insert(handle.seq())` clause is the only part of that predicate
    /// with no other coverage, and it is live rather than defensive:
    /// `gather_restore_lineage` stores the adapter's listing VERBATIM — no
    /// dedup, no uniqueness assertion — so whatever `list_layer_checkpoints`
    /// returns is what the predicate iterates. A non-conforming or
    /// double-listing adapter therefore reaches this code in production, and
    /// the number it produces is not internal: it rides out on
    /// [`Exhaustion::BudgetInconclusive`]'s `unadjudicated_bases` field and into
    /// the operator-facing WARN. Counting one physical generation twice would
    /// tell an operator that two backups went unread when only one exists —
    /// inflating the apparent scale of a store problem while they are triaging
    /// one.
    ///
    /// The fixture is the minimum that distinguishes the two counts: one wedged
    /// base, listed twice, and nothing else.
    ///
    /// ```text
    ///   base@10 (Base), arm 1   min(30, 60) = 30, WEDGES for 30  -> 30   Wedged
    ///   base@10 as own head     cached Wedged, 0 fetches         -> 30   Budget
    /// ```
    ///
    /// Both listing entries are adjudicated by ONE fact, keyed
    /// `(10, Base)` — which is exactly why the walk's own bookkeeping cannot
    /// supply the count and the predicate has to dedup for itself. With the
    /// dedup removed the same run reports `unadjudicated_bases: 2` against a
    /// store holding a single generation.
    ///
    /// The verdict stays [`Exhaustion::BudgetInconclusive`] either way, so this
    /// case is deliberately NOT a claim about which exhaustion was chosen — the
    /// two cases above own that split. It is a claim about the number the chosen
    /// one carries.
    #[tokio::test(start_paused = true)]
    async fn a_base_listed_twice_counts_once_among_the_unadjudicated() {
        let mut wedged_base = base_gen(10);
        wedged_base.hangs = true;
        // The SAME generation handed to the fake twice: `list_layer_checkpoints`
        // maps one handle per seeded generation, so this is a listing with two
        // identical `(10, Base)` members — the non-conforming adapter shape.
        let store = FakeStore::new(vec![wedged_base.clone(), wedged_base]);
        let deadline = Instant::now() + STOCK_BUDGET;

        let mut selection = gather_restore_lineage(&store, OBJECT_ID, deadline, RESERVE)
            .await
            .expect("listing succeeds");
        let answer = selection
            .next_restorable()
            .await
            .expect("a wedged fetch is a step-down signal, NOT an error");

        assert!(
            answer.is_none(),
            "the only base wedged, so there is nothing to restore from"
        );
        assert_eq!(
            selection.exhaustion(),
            Some(&Exhaustion::BudgetInconclusive {
                candidates: 2,
                unadjudicated_bases: 1,
            }),
            "THE CLAIM. Two listing entries, ONE physical unadjudicated base, so the \
             operator-facing count is 1 — while `candidates` stays 2 because the walk really did \
             try two listed members. Dropping the distinct-sequence dedup makes this read 2 and \
             overstates a store problem to the operator triaging it. Got {:?}",
            selection.exhaustion()
        );

        assert_eq!(
            store.get_count(10, CheckpointKind::Base),
            1,
            "and the duplicate is not re-fetched — the second candidate is adjudicated off the \
             cached `(10, Base)` fact, which is what leaves the predicate as the only place the \
             duplication could be counted twice"
        );
        assert_eq!(store.total_gets(), 1, "nothing else was addressed");
        assert_eq!(
            selection.step_downs().len(),
            2,
            "both listing entries were walked and both stepped down"
        );
    }

    /// The negative: budget step-downs are **not** the predicate. A selection
    /// full of them that nonetheless adjudicated every listed base stays
    /// conclusively [`Exhaustion::AllBroken`] — terminal `NOT_FOUND`, correctly.
    ///
    /// `[T@30 -> T@20 -> base@10]` plus an older base@5, with T@20 wedged and
    /// **both** bases unusable (an adapter returning a foreign `object_id`):
    ///
    /// ```text
    ///   T@30, arm 1         min(30, 60) = 30, costs 10       -> 50
    ///   T@20 (Tail)         min(30, 50 - 30) = 20, WEDGES    -> 30    Budget
    ///   T@20 as own head    cached Wedged, 0 fetches         -> 30    Budget
    ///   base@10 (Base)      min(30, 30) = 30, costs 10       -> 20    Unusable
    ///   base@5 (Base)       min(30, 20) = 20, costs 10       -> 10    Unusable
    /// ```
    ///
    /// Half the step-downs are `stepped_down_budget`, and the verdict is still
    /// conclusive, because a `Tail` can only extend a lineage — never root one.
    /// The walk read both artifacts that *could* have rooted one and both are
    /// broken, so "this layer is unrestorable" is a claim about the layer and
    /// the client's terminal `NOT_FOUND` is the right answer. A predicate that
    /// counted `LineageBreak::Budget` records instead would over-trigger here
    /// and turn a real unrestorable layer into an endless client retry.
    #[tokio::test(start_paused = true)]
    async fn budget_step_downs_over_fully_adjudicated_bases_stay_conclusively_all_broken() {
        /// Long enough that the speculative tail's `remaining - reserve`
        /// allowance is spent by the wedge, short enough that both `Base`
        /// fetches complete inside `min(reserve, remaining)`.
        const FETCH_COST: Duration = Duration::from_secs(10);

        let foreign = |mut generation: Generation| {
            generation.returned_object_id = Some(OBJECT_ID + 1);
            generation
        };
        let mut wedged_tail = tail_gen(10, 10, 20);
        wedged_tail.hangs = true;
        let store = FakeStore::new(vec![
            foreign(base_gen(5)),
            foreign(base_gen(10)),
            wedged_tail,
            tail_gen(10, 20, 30),
        ])
        .with_fetch_delay(FETCH_COST);
        let deadline = Instant::now() + STOCK_BUDGET;

        let mut selection = gather_restore_lineage(&store, OBJECT_ID, deadline, RESERVE)
            .await
            .expect("listing succeeds");
        let answer = selection
            .next_restorable()
            .await
            .expect("a wedged adapter must not surface as an error here");

        assert!(answer.is_none(), "every candidate really is broken");
        assert_eq!(
            selection.exhaustion(),
            Some(&Exhaustion::AllBroken { candidates: 4 }),
            "THE NEGATIVE. Both listed bases were READ and both are unusable, so the selection \
             has positive evidence about the layer and the `Ok(None)` -> `NOT_FOUND` is honest — \
             even though two of its four step-downs are Budget verdicts. Reporting \
             `BudgetInconclusive` here would tell the client to retry a layer that can never \
             restore. Got {:?}",
            selection.exhaustion()
        );
        assert_eq!(
            selection
                .exhaustion()
                .expect("the selection exhausted")
                .outcome_label(),
            "exhausted",
            "the conclusive cell — a rise in it DOES indicate checkpoint loss, which is the whole \
             reason the inconclusive arm needed a cell of its own"
        );

        // The fixture really does contain both classes of step-down, or the
        // negative proves nothing.
        let labels: Vec<&str> = selection
            .step_downs()
            .iter()
            .map(|record| record.reason.reason_label())
            .collect();
        assert_eq!(
            labels,
            [
                "stepped_down_budget",
                "stepped_down_budget",
                "stepped_down_link_mismatch",
                "stepped_down_link_mismatch",
            ],
            "anti-vacuity: two Budget step-downs must actually be present, or this case does not \
             distinguish the shipped base-keyed predicate from a step-down-reason-keyed one"
        );
        assert_eq!(
            store.get_count(10, CheckpointKind::Base),
            1,
            "both bases are ADJUDICATED, which is what makes the verdict conclusive"
        );
        assert_eq!(store.get_count(5, CheckpointKind::Base), 1);
    }

    /// Boundary: a base pruned between the LIST and the fetch counts as
    /// **adjudicated**, so a listing whose only base vanished that way stays
    /// [`Exhaustion::AllBroken`].
    ///
    /// The concurrent-prune race is the one `Ok(None)`-at-fetch the store
    /// contract documents, and it is positive evidence: the generation is not
    /// there. Retention removed it, and no amount of retrying brings it back —
    /// so the terminal `NOT_FOUND` is correct and `BudgetInconclusive` here
    /// would spin the client on a layer that really has no backup.
    ///
    /// The `at_fetch` field is matched exactly by the predicate rather than
    /// wildcarded, so this case pins the `true` direction; the `false`
    /// direction (no handle in the listing for a key taken *from* the listing)
    /// is unreachable today and deliberately counts as unadjudicated, which is
    /// the fail-safe direction.
    #[tokio::test]
    async fn a_base_pruned_between_list_and_fetch_counts_as_adjudicated() {
        let mut pruned = base_gen(10);
        pruned.missing_at_fetch = true;
        let store = FakeStore::new(vec![pruned]);

        let mut selection = gather_restore_lineage(&store, OBJECT_ID, no_deadline(), RESERVE)
            .await
            .expect("listing succeeds");
        let answer = selection.next_restorable().await.expect("no error");

        assert!(answer.is_none(), "the only base vanished under the walk");
        assert_eq!(
            selection.step_downs().len(),
            1,
            "got {:?}",
            selection.step_downs()
        );
        assert_eq!(
            selection.step_downs()[0].reason,
            LineageBreak::LinkAbsent {
                seq: 10,
                kind: CheckpointKind::Base,
                at_fetch: true,
            },
            "the prune race, not a budget verdict — this selection had all the time it wanted"
        );
        assert_eq!(
            selection.exhaustion(),
            Some(&Exhaustion::AllBroken { candidates: 1 }),
            "`Absent {{ at_fetch: true }}` IS an answer about the artifact, so the base counts as \
             adjudicated and the verdict stays conclusive. Got {:?}",
            selection.exhaustion()
        );
    }

    /// Boundary: a tails-only listing is [`Exhaustion::NoBase`] however the
    /// budget went — never [`Exhaustion::BudgetInconclusive`].
    ///
    /// The predicate counts listed `Base` generations, so a listing with none
    /// can never produce a positive count. This case fences that from the other
    /// side by making the budget the loudest thing in the fixture — a LIST that
    /// outlives the whole deadline — and asserting the verdict does not move.
    /// The structural check runs before the walk, so no member is ever
    /// addressed and there is no budget verdict to confuse it with.
    #[tokio::test(start_paused = true)]
    async fn a_tails_only_listing_is_no_base_even_when_the_budget_is_already_gone() {
        let store = FakeStore::new(vec![tail_gen(10, 10, 14), tail_gen(10, 14, 20)])
            // The LIST alone outlives the budget: nothing at all is left by the
            // time the walk could run.
            .with_list_delay(STOCK_BUDGET + Duration::from_secs(30));
        let deadline = Instant::now() + STOCK_BUDGET;

        let mut selection = gather_restore_lineage(&store, OBJECT_ID, deadline, RESERVE)
            .await
            .expect("listing succeeds even having outlived the deadline");
        let answer = selection.next_restorable().await.expect("no error");

        assert!(answer.is_none());
        assert_eq!(
            selection.exhaustion(),
            Some(&Exhaustion::NoBase {
                generations: 2,
                newest_sequence: 20,
            }),
            "a listing holding no base is unrestorable for a STRUCTURAL reason, and no amount of \
             budget would change that — so it must not be reported as a selection that merely ran \
             out of time. Got {:?}",
            selection.exhaustion()
        );
        assert_eq!(
            selection
                .exhaustion()
                .expect("the selection exhausted")
                .outcome_label(),
            "exhausted",
            "and it folds onto the conclusive cell, terminal `NOT_FOUND` — retrying a tails-only \
             store forever would be the failure mode"
        );
        assert!(
            selection.step_downs().is_empty(),
            "the structural check short-circuits before any candidate is walked; got {:?}",
            selection.step_downs()
        );
        assert_eq!(
            store.total_gets(),
            0,
            "and no member is addressed, so there is no Budget verdict in play at all"
        );
    }

    // ── Group D: the four selection arms, in the SR-3.1 arm framing ─────────
    //
    // Groups A–C above cover the mechanism (one candidate's walk, the step-down
    // loop, the budget). This group pins the four *arms* the requirement is
    // stated in — base-newest, tail-newest-intact, newest-broken, all-broken —
    // each over a listing that ALSO holds an older base, which is what makes
    // "the answer is generation X and not generation Y" a distinguishing claim
    // rather than a tautology over a one-base fixture.

    #[tokio::test]
    async fn selection_newest_generation_is_a_base_selects_it_without_gathering() {
        // base@5, base@9 — nothing newer. The regression fence for base-only
        // stores: the answer, the fetch count, and the untouched older
        // generation are each byte-identical to the pre-Epic-4 path's.
        let store = FakeStore::new(vec![base_gen(5), base_gen(9)]);
        let (lineage, breaks) = first_lineage(&store).await;
        let lineage = lineage.expect("a base is always restorable");

        assert_eq!(lineage.base_snap_seq(), 9, "the NEWEST base wins");
        assert_eq!(lineage.head_seq, 9);
        assert!(
            lineage.tails.is_empty(),
            "a base-headed candidate has no tail members; got {:?}",
            lineage.tails.len()
        );
        assert_eq!(lineage.chain_len(), 0);
        assert!(lineage.is_base_only());
        assert_eq!(lineage.outcome_label(), "gathered_base");
        assert!(breaks.is_empty(), "nothing was abandoned; got {breaks:?}");

        assert_eq!(store.list_count(), 1);
        assert_eq!(
            store.total_gets(),
            1,
            "a base-newest listing costs exactly one fetch — no gathering; observed {} fetches",
            store.total_gets()
        );
        assert_eq!(
            store.get_count(5, CheckpointKind::Base),
            0,
            "the older base is not a member of the selected lineage and must never be addressed"
        );
    }

    #[tokio::test]
    async fn selection_newest_generation_is_a_tail_with_intact_lineage_gathers_the_whole_run() {
        // base@5, base@9, tail(9,12], tail(12,15] — the head roots on base@9,
        // so base@5 sits BELOW the selected lineage and is neither a member nor
        // addressed.
        let store = FakeStore::new(vec![
            base_gen(5),
            base_gen(9),
            tail_gen(9, 9, 12),
            tail_gen(9, 12, 15),
        ]);
        let (lineage, breaks) = first_lineage(&store).await;
        let lineage = lineage.expect("an intact lineage gathers");

        assert_eq!(lineage.base_snap_seq(), 9);
        assert_tails(&lineage, &[(9, 12), (12, 15)]);
        assert_eq!(lineage.chain_len(), 2);
        assert_eq!(
            lineage.head_seq, 15,
            "head_seq is S — the head tail's to_seq"
        );
        assert_eq!(lineage.outcome_label(), "gathered_composed");

        // The resolved links, read off the members themselves: the head's
        // `from_seq` names its predecessor, and the oldest tail's `from_seq` IS
        // the lineage root (`from_seq` is EXCLUSIVE — no +/-1 anywhere).
        let head = lineage.tails.last().expect("a head tail");
        assert_eq!(head.header.from_seq, 12);
        assert_eq!(head.header.base_snap_seq, 9);
        assert_eq!(
            lineage.tails[0].header.from_seq,
            lineage.base_snap_seq(),
            "the oldest tail links directly onto the base it roots at"
        );

        assert!(breaks.is_empty(), "nothing was abandoned; got {breaks:?}");
        assert_eq!(store.list_count(), 1);
        assert_eq!(
            store.get_count(5, CheckpointKind::Base),
            0,
            "base@5 is not a member of the lineage rooted at base@9 and must never be addressed"
        );
        assert_eq!(
            store.total_gets(),
            3,
            "1 base + the 2 walked tails; observed {} fetches over a 4-generation listing",
            store.total_gets()
        );
    }

    #[tokio::test]
    async fn selection_newest_lineage_broken_steps_down_to_the_next_older_restorable_generation() {
        // base@5, base@9, tail(11,15] — the head's `from_seq` names 11, which
        // no generation supplies: a link into nothing. The step-down must land
        // on the NEXT older restorable generation (base@9), not skip to the
        // oldest.
        let store = FakeStore::new(vec![base_gen(5), base_gen(9), tail_gen(9, 11, 15)]);
        let (lineage, breaks) = first_lineage(&store).await;

        assert_eq!(
            breaks.len(),
            1,
            "exactly one candidate was abandoned; got {breaks:?}"
        );
        assert_break(
            &breaks,
            0,
            &LineageBreak::LinkAbsent {
                seq: 11,
                kind: CheckpointKind::Tail,
                at_fetch: false,
            },
        );
        assert_eq!(
            breaks[0].head_seq, 15,
            "the record names the abandoned head"
        );
        assert_eq!(breaks[0].head_kind, CheckpointKind::Tail);
        assert_eq!(breaks[0].reason.reason_label(), "stepped_down_link_broken");
        assert_eq!(breaks[0].reason.link_seq(), 11);

        let lineage = lineage.expect("the step-down still resolves");
        assert!(lineage.is_base_only());
        assert_eq!(
            lineage.base_snap_seq(),
            9,
            "the step-down lands on the NEXT older restorable generation, not the oldest"
        );
        assert_eq!(
            store.get_count(11, CheckpointKind::Tail),
            0,
            "a link the listing does not hold must not be fetched"
        );
        assert_eq!(
            store.get_count(5, CheckpointKind::Base),
            0,
            "selection stopped at base@9, so base@5 was never reached"
        );
    }

    #[tokio::test]
    async fn selection_every_lineage_broken_falls_to_the_oldest_restorable_generation_then_to_none()
    {
        // Arm (a) — every TAIL lineage broken, the base intact: selection walks
        // past each broken candidate and lands on the base, recording one
        // step-down per lineage walked past.
        let broken_tails = vec![corrupted(tail_gen(5, 5, 9)), corrupted(tail_gen(5, 9, 12))];
        let lineages_walked_past = broken_tails.len();
        let mut generations = vec![base_gen(5)];
        generations.extend(broken_tails);
        let store = FakeStore::new(generations);

        let (lineage, breaks) = first_lineage(&store).await;
        let lineage = lineage.expect("the intact base still restores");
        assert!(lineage.is_base_only());
        assert_eq!(lineage.base_snap_seq(), 5);
        assert_eq!(
            breaks.len(),
            lineages_walked_past,
            "one step-down per lineage walked past; got {breaks:?}"
        );
        for record in &breaks {
            assert_eq!(record.reason.reason_label(), "stepped_down_decode");
        }

        // Arm (b) — a TAILS-ONLY store is observably identical to an EMPTY one:
        // both answer `Ok(None)` under the one `exhausted` outcome label, and
        // NEITHER is an `Err`. Asserted side by side in one body so the
        // equivalence is stated rather than inferred across two test bodies —
        // it restates at the gather boundary the invariant
        // `recovery.rs::try_restore_layer_tails_only_store_is_not_recoverable_like_empty_store`
        // pins at the hook.
        let tails_only = FakeStore::new(vec![tail_gen(5, 5, 9), tail_gen(5, 9, 12)]);
        let empty = FakeStore::new(vec![]);
        let mut answers: Vec<(bool, &'static str)> = Vec::new();
        for store in [&tails_only, &empty] {
            let mut selection = gather_restore_lineage(store, OBJECT_ID, no_deadline(), RESERVE)
                .await
                .expect("listing succeeds");
            let answer = selection
                .next_restorable()
                .await
                .expect("an unrestorable store answers Ok(None), never Err");
            answers.push((
                answer.is_none(),
                selection
                    .exhaustion()
                    .expect("an Ok(None) answer always records why")
                    .outcome_label(),
            ));
        }
        assert_eq!(
            answers,
            vec![(true, "exhausted"), (true, "exhausted")],
            "a tails-only store must be observably identical to an empty one"
        );
    }

    // ── Group E: fetch-bound cost (SR-4.2) ──────────────────────────────────

    /// A store stratified into TWO complete lineages: an old one rooted at
    /// base@1 and the selected one rooted at base@9. Everything below base@9 is
    /// intact and restorable — so a fetch that addresses it is an *eager scan*,
    /// not a step-down.
    fn stratified_store() -> FakeStore {
        FakeStore::new(vec![
            base_gen(1),
            tail_gen(1, 1, 3),
            tail_gen(1, 3, 6),
            base_gen(9),
            tail_gen(9, 9, 12),
            tail_gen(9, 12, 15),
        ])
    }

    /// The generations strictly below the selected lineage's base.
    const BELOW_THE_SELECTED_BASE: [(u64, CheckpointKind); 3] = [
        (1, CheckpointKind::Base),
        (3, CheckpointKind::Tail),
        (6, CheckpointKind::Tail),
    ];

    #[tokio::test]
    async fn gather_bounds_its_fetches_to_the_selected_lineage() {
        const LISTED_GENERATIONS: usize = 6;
        let store = stratified_store();
        let (lineage, breaks) = first_lineage(&store).await;
        let lineage = lineage.expect("the newest lineage verifies");

        assert!(breaks.is_empty(), "nothing was abandoned; got {breaks:?}");
        assert_eq!(lineage.base_snap_seq(), 9);
        assert_eq!(lineage.chain_len(), 2);

        assert_eq!(
            store.list_count(),
            1,
            "one listing per selection, never re-read; observed {} listings",
            store.list_count()
        );
        // SR-4.2's cost model: <= 1 base + K tails, where K is the TRUE
        // link-walk chain length — never once per listed generation.
        let ceiling = 1 + lineage.chain_len();
        assert!(
            store.total_gets() <= ceiling,
            "the gather issued {} fetches against a ceiling of 1 base + chain_len = {ceiling}, \
             over a listing of {LISTED_GENERATIONS} generations; a per-listed-generation scan \
             would read {LISTED_GENERATIONS}",
            store.total_gets()
        );
    }

    #[tokio::test]
    async fn gather_does_not_fetch_generations_below_the_selected_base() {
        // The "do not scan-and-fetch the whole listing eagerly" clause, made
        // observable on the INTACT-head arm: the older lineage is perfectly
        // restorable, so its untouched-ness is a cost claim, not a defect one.
        let store = stratified_store();
        let (lineage, breaks) = first_lineage(&store).await;
        let lineage = lineage.expect("the newest lineage verifies");
        assert!(
            breaks.is_empty(),
            "the head arm must be intact; got {breaks:?}"
        );
        assert_eq!(lineage.base_snap_seq(), 9);

        for (seq, kind) in BELOW_THE_SELECTED_BASE {
            assert_eq!(
                store.get_count(seq, kind),
                0,
                "{kind:?}@{seq} sits below the selected base@9 and must never be addressed; it \
                 was fetched {} times",
                store.get_count(seq, kind)
            );
        }
    }

    // ── Group F: lineage integrity ──────────────────────────────────────────
    //
    // Groups A–E pin what the gather *finds*. This group pins the three ways a
    // gather can be WRONG rather than merely short: it can trust a fabricated
    // meta, it can accept a non-contiguous chain, or it can walk into a forked
    // sibling. Each is a mis-restore — commits layered onto a base that never
    // carried them — which no amount of step-down coverage detects, because a
    // mis-linked lineage *verifies*.
    //
    // The lineage-less-meta fixture is `lineage_less_tail_meta()` assigned onto
    // a seeded `Generation`, not a delegating store newtype: `Generation::meta`
    // is already the adapter's returned meta verbatim, so assigning it puts the
    // shape on the wire with the real envelope bytes underneath. A wrapper
    // store would reproduce the same shape through more machinery. (It used to
    // reproduce the two adapters' torn-put placeholder byte for byte; b.hka (d)
    // replaced that placeholder with a refusal, and the fixture now stands for
    // the `cross_check_meta` `None` arm on its own terms — see
    // `lineage_less_tail_meta`'s doc.)
    //
    // No metric cell is read anywhere here: the gather mints reason RECORDS
    // (`StepDownRecord` / `LineageBreak::reason_label`) and Task 3's hook is what
    // turns one into an increment.

    /// Rewrite an already-encoded envelope's on-wire `base_snap_seq`.
    ///
    /// The only way to seed a header whose declared root disagrees with its own
    /// span: both `encode_tail_envelope` and `read_tail_header` refuse
    /// `base_snap_seq > from_seq`, so such an artifact can only be hand-forged.
    /// The offset is the header layout `tail_codec`'s golden-layout test pins
    /// (magic 0..4, format_version 4..8, object_id 8..16, base_snap_seq 16..24);
    /// a drifted offset lands the patch on a different field and the case's
    /// expected codec message stops matching, so the assertion is its own guard.
    fn forged_lineage_root(mut generation: Generation, base_snap_seq: u64) -> Generation {
        const BASE_SNAP_SEQ_OFFSET: usize = 16;
        let mut raw = generation.bytes.to_vec();
        raw[BASE_SNAP_SEQ_OFFSET..BASE_SNAP_SEQ_OFFSET + 8]
            .copy_from_slice(&base_snap_seq.to_le_bytes());
        generation.bytes = Bytes::from(raw);
        generation
    }

    /// A gathered lineage is a fully-verified run: every tail links onto its
    /// predecessor with no `+1` / `-1` (`from_seq` is EXCLUSIVE, b.rrx), every
    /// member shares one lineage root, and `head_seq` is the head's `to_seq`.
    ///
    /// The load-bearing shared assertion for the whole integrity group: an
    /// answer is either this or a strictly older generation — a *partial* run
    /// (silently shortened, silently extended) is the mis-restore outcome.
    #[track_caller]
    fn assert_fully_verified_run(lineage: &GatheredLineage, context: &str) {
        let root = lineage.base_snap_seq();
        let mut expected_from = root;
        for tail in &lineage.tails {
            assert_eq!(
                tail.header.from_seq, expected_from,
                "[{context}] tail ending at {} links onto {expected_from}, not {}; a run whose \
                 links do not chain is a partial run",
                tail.header.to_seq, tail.header.from_seq
            );
            assert_eq!(
                tail.header.base_snap_seq, root,
                "[{context}] every member must share the lineage root {root}"
            );
            expected_from = tail.header.to_seq;
        }
        assert_eq!(
            lineage.head_seq, expected_from,
            "[{context}] head_seq is S — the head member's to_seq"
        );
    }

    // ── Group F1: the meta is advisory, the envelope header is authoritative ─

    #[tokio::test]
    async fn a_lineage_less_meta_tail_whose_envelope_is_also_unreadable_is_a_broken_lineage() {
        // The negative arm that keeps the lineage-less-meta case from being
        // satisfied by "a `None` lineage means intact". With BOTH lineage
        // sources gone — the meta naming none, the envelope past recovery —
        // there is nothing left to verify the member against, so it must never
        // be admitted: the lineage is broken and selection steps down
        // observably. Note the break is `Decode`, NOT
        // `MissingMetaCompanion`: the bytes are what failed here. The two are
        // deliberately different outcomes (b.hka (d)) because they carry
        // different operator remediations.
        let mut head = corrupted(tail_gen(9, 9, 12));
        head.meta = lineage_less_tail_meta();
        let store = FakeStore::new(vec![base_gen(9), head]);
        let (lineage, breaks) = first_lineage(&store).await;

        assert_eq!(breaks.len(), 1, "got {breaks:?}");
        assert_eq!(breaks[0].head_seq, 12);
        assert_eq!(breaks[0].head_kind, CheckpointKind::Tail);
        match &breaks[0].reason {
            LineageBreak::Decode { seq, message } => {
                assert_eq!(*seq, 12);
                assert!(
                    !message.is_empty(),
                    "the record must carry the codec's own reason, not an empty verdict"
                );
            }
            other => panic!("expected a Decode break, got {other:?}"),
        }
        // The RECORD is the observable step-down, not a metric cell: this module
        // mints labels and Task 3's hook increments.
        assert_eq!(breaks[0].reason.reason_label(), "stepped_down_decode");

        let lineage = lineage.expect("the base behind the unreadable tail still restores");
        assert!(lineage.is_base_only());
        assert_eq!(lineage.base_snap_seq(), 9);
        assert_eq!(lineage.head_seq, 9);
        assert_fully_verified_run(&lineage, "lineage-less meta + unreadable envelope");
    }

    #[tokio::test]
    async fn the_envelope_header_wins_when_the_meta_names_a_different_existing_generation() {
        // The MIS-LINKING arm. `a_meta_that_disagrees_with_its_envelope_loses_to
        // _the_header` points the meta at sequences no generation supplies, so a
        // meta-preferring implementation breaks loudly there. Here the meta
        // names base@5 — a real, listed, perfectly restorable generation — so a
        // meta-preferring implementation produces a lineage that VERIFIES and is
        // wrong: tail (9, 12]'s commits layered onto a base that never carried
        // sequences 6..9. That is the wrong-restore the advisory rule exists to
        // prevent, and it is invisible to every other assertion in this file.
        let mut head = tail_gen(9, 9, 12);
        head.meta = CheckpointMeta {
            base_snap_seq: Some(5),
            from_seq: Some(5),
            ..head.meta
        };
        let store = FakeStore::new(vec![base_gen(5), base_gen(9), head]);
        let (lineage, breaks) = first_lineage(&store).await;
        let lineage = lineage.expect("the envelope header resolves the lineage");

        assert!(breaks.is_empty(), "nothing was abandoned; got {breaks:?}");
        assert_eq!(
            lineage.base_snap_seq(),
            9,
            "the header names base@9; the meta's base@5 is an index entry, not a source of truth"
        );
        assert_tails(&lineage, &[(9, 12)]);
        assert_fully_verified_run(&lineage, "meta names a different existing base");
        assert_eq!(
            store.get_count(5, CheckpointKind::Base),
            0,
            "base@5 is what the meta names; addressing it at all means the meta drove the walk"
        );
        assert_eq!(
            lineage.tails[0].meta.base_snap_seq,
            Some(5),
            "the fixture must actually carry the disagreeing meta, or this case is vacuous"
        );
    }

    /// SR-6.1, and the executable detector for a `peek_tail_header` check-ORDER
    /// regression.
    ///
    /// `read_tail_header` checks magic, then `format_version`, then length. An
    /// implementation that reordered version behind the length check would tell
    /// the operator a v2 artifact is *truncated* — "your data is damaged" rather
    /// than "this build is too old" — which is the opposite remediation. The
    /// fourth case below is what fails on that regression: an artifact that is
    /// BOTH at an unreadable version and too short must report the version.
    #[tokio::test]
    async fn an_unknown_format_version_break_is_distinguishable_from_bad_magic_and_truncation() {
        let intact = tail_bytes(OBJECT_ID, 9, 9, 12);
        let bad_magic = {
            let mut raw = intact.to_vec();
            // A checkpoint OMPF handed to the tail decoder — the shape the
            // codec's own `BadMagic` doc names.
            raw[0..4].copy_from_slice(b"OMPF");
            Bytes::from(raw)
        };
        // Magic and a READABLE, supported version, then nothing: the length
        // check is what rejects it.
        let truncated = intact.slice(0..8);
        // Both defects at once. Version is checked first, so this must be
        // indistinguishable from the pure version case.
        let unreadable_version_and_truncated = at_unreadable_format_version(&intact).slice(0..8);

        let cases: [(&str, Bytes, &str); 4] = [
            (
                "unknown format_version",
                at_unreadable_format_version(&intact),
                "format version 99",
            ),
            ("bad magic", bad_magic, "bad tail magic"),
            ("truncated", truncated, "truncated tail at"),
            (
                "unknown format_version AND truncated",
                unreadable_version_and_truncated,
                "format version 99",
            ),
        ];

        let mut messages: Vec<String> = Vec::new();
        for (name, bytes, expected_substring) in cases {
            let mut head = tail_gen(9, 9, 12);
            head.bytes = bytes;
            let store = FakeStore::new(vec![base_gen(9), head]);
            let (lineage, breaks) = first_lineage(&store).await;

            let message = match &breaks[0].reason {
                LineageBreak::Decode { seq, message } => {
                    assert_eq!(*seq, 12, "[{name}] the break names the offending member");
                    message.clone()
                }
                other => panic!("[{name}] expected a Decode break, got {other:?}"),
            };
            assert!(
                message.contains(expected_substring),
                "[{name}] the verdict must carry the codec's own reason naming \
                 {expected_substring:?}; got {message:?}"
            );
            assert_eq!(breaks[0].reason.reason_label(), "stepped_down_decode");
            let lineage = lineage
                .unwrap_or_else(|| panic!("[{name}] the base behind the bad artifact restores"));
            assert!(lineage.is_base_only(), "[{name}] steps down to base@9");
            messages.push(message);
        }

        let distinct: HashSet<&str> = messages.iter().map(String::as_str).collect();
        assert_eq!(
            distinct.len(),
            3,
            "the three decode defects must be distinguishable in the returned record — an \
             operator reads the message to decide between 'roll the build forward' and 'the \
             artifact is damaged'; got {messages:?}"
        );
        assert_eq!(
            messages[3], messages[0],
            "version is checked BEFORE length: an artifact that is both must report the \
             unreadable version, not the truncation. A different message here means the \
             check order regressed and SR-6.1 now mis-diagnoses a forward-version artifact"
        );
    }

    // ── Group F2: contiguity violations ─────────────────────────────────────

    /// SR-2.1 linkage, as a table over one helper.
    ///
    /// `from_seq` is the EXCLUSIVE lower bound of `(from_seq, to_seq]`, so the
    /// rule is: the first tail has `from_seq == base.snap_seq`, and each later
    /// tail has `from_seq == previous.to_seq` — no `+1` / `-1` anywhere (b.rrx).
    /// Every case here violates exactly one link, and each must yield a
    /// broken-lineage verdict naming THAT link plus a step-down to a strictly
    /// older generation — never a silently shortened or silently extended
    /// member list.
    #[tokio::test]
    async fn contiguity_violations_step_down_naming_the_violated_link() {
        struct Case {
            name: &'static str,
            /// What the head's link requires, in the failure message.
            violated_link: &'static str,
            generations: Vec<Generation>,
            /// The head candidate that must be abandoned.
            abandoned_head: u64,
            expected: LineageBreak,
        }

        let cases = vec![
            Case {
                name: "gap between tails",
                violated_link: "tail (13, 17] names a predecessor ending at 13; the run below it \
                                ends at 12",
                generations: vec![base_gen(9), tail_gen(9, 9, 12), tail_gen(9, 13, 17)],
                abandoned_head: 17,
                expected: LineageBreak::LinkAbsent {
                    seq: 13,
                    kind: CheckpointKind::Tail,
                    at_fetch: false,
                },
            },
            Case {
                name: "head tail's from_seq skips the base",
                violated_link: "tail (10, 15] roots at base@9 but starts at 10, so sequence 10 \
                                is claimed by nothing",
                generations: vec![base_gen(9), tail_gen(9, 10, 15)],
                abandoned_head: 15,
                expected: LineageBreak::LinkAbsent {
                    seq: 10,
                    kind: CheckpointKind::Tail,
                    at_fetch: false,
                },
            },
            Case {
                name: "tail rooted at a base that does not exist",
                violated_link: "tail (4, 12] declares lineage root 4; no base@4 is listed. The \
                                link's kind is Base precisely because the sequence IS the root",
                generations: vec![base_gen(9), tail_gen(4, 4, 12)],
                abandoned_head: 12,
                expected: LineageBreak::LinkAbsent {
                    seq: 4,
                    kind: CheckpointKind::Base,
                    at_fetch: false,
                },
            },
            Case {
                // The ticket's "base_snap_seq: 9, from_seq: 7" shape is
                // UNREPRESENTABLE: `base_snap_seq > from_seq` is refused by
                // `encode_tail_envelope` AND by `read_tail_header`
                // (`BaseAfterRange`), so no such envelope can be built or read.
                // The reachable form is a hand-forged one, and the disagreement
                // therefore surfaces as a decode refusal rather than as a link
                // walk — which is the stronger outcome: the artifact never
                // reaches the walk at all.
                name: "the lineage root is above the span the tail claims",
                violated_link: "the header declares root 11 for a span starting at 9, so the \
                                tail does not layer onto the snapshot it names; the codec \
                                refuses to represent it in either direction",
                generations: vec![base_gen(9), forged_lineage_root(tail_gen(9, 9, 12), 11)],
                abandoned_head: 12,
                expected: LineageBreak::Decode {
                    seq: 12,
                    message: "tail base_snap_seq 11 is not below from_seq 9".to_string(),
                },
            },
            Case {
                name: "the named predecessor exists at the wrong kind",
                violated_link: "tail (12, 17] names a TAIL at 12; the listing holds a BASE there. \
                                Links are addressed by (sequence, kind), so a base cannot stand \
                                in for the tail a link names",
                generations: vec![base_gen(9), base_gen(12), tail_gen(9, 12, 17)],
                abandoned_head: 17,
                expected: LineageBreak::LinkAbsent {
                    seq: 12,
                    kind: CheckpointKind::Tail,
                    at_fetch: false,
                },
            },
        ];

        for case in cases {
            let Case {
                name,
                violated_link,
                generations,
                abandoned_head,
                expected,
            } = case;
            let store = FakeStore::new(generations);
            let (lineage, breaks) = first_lineage(&store).await;

            assert_eq!(
                breaks.len(),
                1,
                "[{name}] exactly one candidate is abandoned — {violated_link}; got {breaks:?}"
            );
            assert_eq!(
                breaks[0].head_seq, abandoned_head,
                "[{name}] the record names the abandoned head"
            );
            assert_eq!(breaks[0].head_kind, CheckpointKind::Tail);
            assert_eq!(
                breaks[0].reason, expected,
                "[{name}] the verdict must name the violated link: {violated_link}"
            );
            assert_eq!(
                breaks[0].reason.reason_label(),
                expected.reason_label(),
                "[{name}] the label travels with the verdict"
            );
            if let LineageBreak::LinkAbsent { seq, kind, .. } = expected {
                assert_eq!(
                    store.get_count(seq, kind),
                    0,
                    "[{name}] a link the listing does not hold must not be fetched"
                );
            }

            // The load-bearing shared claim: the answer is either the fully
            // verified run or a strictly older generation, NEVER a partial run.
            let lineage = lineage.unwrap_or_else(|| {
                panic!("[{name}] a base is listed, so selection must still resolve")
            });
            assert_fully_verified_run(&lineage, name);
            assert!(
                lineage.head_seq < abandoned_head,
                "[{name}] the answer must sit strictly below the abandoned head {abandoned_head}; \
                 got {}",
                lineage.head_seq
            );
            assert!(
                lineage
                    .tails
                    .iter()
                    .all(|tail| tail.header.to_seq != abandoned_head),
                "[{name}] the abandoned head must not reappear as a member of the answer"
            );
        }
    }

    // ── Group F3: divergent-S overlap exclusion (SR-9.2) ────────────────────

    /// The SR-9.2 fork residue in its canonical shape, mirroring
    /// `checkpointer_tail_duplicate_leader_test.rs`'s
    /// `divergent_s_fork_leaves_two_keys_with_the_newest_as_the_lineage_head`:
    /// two tails rooted at the SAME `from_seq == base.snap_seq` with different
    /// `to_seq`, both durable, neither overwriting the other.
    ///
    /// The gather is the first surface that can assert non-membership
    /// POSITIVELY — the capture side's `chain_len` is a deliberate conservative
    /// over-approximation that *counts* the overlapped tail — and the counting
    /// store makes the stronger claim available: the sibling is never even
    /// addressed.
    #[tokio::test]
    async fn a_forked_sibling_tail_is_neither_a_member_nor_fetched() {
        const BASE: u64 = 9;
        const EARLY: u64 = 12;
        const LATE: u64 = 15;
        const { assert!(EARLY < LATE) };

        let store = FakeStore::new(vec![
            base_gen(BASE),
            tail_gen(BASE, BASE, EARLY), // the overlapped loser
            tail_gen(BASE, BASE, LATE),  // the lineage head
        ]);
        let (lineage, breaks) = first_lineage(&store).await;
        let lineage = lineage.expect("the newest of the two forked tails heads a valid lineage");

        assert!(breaks.is_empty(), "neither tail is broken; got {breaks:?}");
        assert_tails(&lineage, &[(BASE, LATE)]);
        assert_eq!(lineage.head_seq, LATE, "the NEWEST fork arm is the head");
        assert_fully_verified_run(&lineage, "divergent-S fork");

        // The duplicate-leader test's own assertion vocabulary: the walk reaches
        // the base in ONE hop, and no chain walked back from the head traverses
        // the sibling.
        let head = lineage.tails.last().expect("a head tail");
        assert_eq!(head.header.from_seq, head.header.base_snap_seq);
        assert_eq!(head.header.from_seq, lineage.base_snap_seq());
        assert_ne!(
            head.header.from_seq, EARLY,
            "the head links onto the BASE, not onto its forked sibling"
        );
        assert_eq!(
            lineage.chain_len(),
            1,
            "the TRUE link-walk chain length is 1 — the capture side's conservative count would \
             be 2 because it counts the overlapped tail"
        );

        // Strictly stronger than "absent from the member list": provably not
        // consulted. A member that was fetched and then discarded is a different
        // fact from one that was never addressed, and only the count tells them
        // apart.
        assert_eq!(
            store.get_count(EARLY, CheckpointKind::Tail),
            0,
            "the forked sibling must never be addressed — the walk resolves predecessors by \
             EXACT sequence, so the fork residue is excluded by construction, not by a filter"
        );
        assert_eq!(
            store.total_gets(),
            2,
            "exactly 1 base + the 1 walked tail over a 3-generation listing; observed {}",
            store.total_gets()
        );
    }

    // ── Group F4: a composition failure steps down like a broken link ───────

    /// Task 1 classifies a composition `Permanent` per generation, and Task 3's
    /// loop consumes that verdict by driving this cursor past the candidate.
    /// The claim under test is the *equivalence*: from the abandonment onward, a
    /// composition failure and a broken link are indistinguishable — same
    /// remaining answers, same step-down record shape, same exhaustion — so the
    /// caller needs no second step-down path.
    #[tokio::test]
    async fn a_compose_failure_steps_down_identically_to_a_broken_link() {
        /// Drain a selection to exhaustion, returning the head sequences it
        /// still offers. Bounded: more iterations than the listing length would
        /// mean the cursor failed to advance.
        async fn drain(selection: &mut LineageSelection<'_>) -> Vec<u64> {
            let mut heads = Vec::new();
            for _ in 0..8 {
                match selection.next_restorable().await.expect("no error") {
                    Some(lineage) => heads.push(lineage.head_seq),
                    None => break,
                }
            }
            heads
        }

        // Arm 1 — the lineage VERIFIES here and fails in the caller's
        // composition, which records the verdict through this module.
        let compose_store = FakeStore::new(vec![base_gen(9), tail_gen(9, 9, 12)]);
        let mut compose = gather_restore_lineage(&compose_store, OBJECT_ID, no_deadline(), RESERVE)
            .await
            .expect("listing succeeds");
        let composed = compose
            .next_restorable()
            .await
            .expect("no error")
            .expect("the lineage verifies");
        assert_eq!(composed.head_seq, 12);
        compose.record_compose_failure(composed.head_seq, CheckpointKind::Tail);
        let compose_heads = drain(&mut compose).await;

        // Arm 2 — the same listing shape, but the head's envelope is corrupt, so
        // the WALK abandons it. Nothing else differs.
        let broken_store = FakeStore::new(vec![base_gen(9), corrupted(tail_gen(9, 9, 12))]);
        let mut broken = gather_restore_lineage(&broken_store, OBJECT_ID, no_deadline(), RESERVE)
            .await
            .expect("listing succeeds");
        let broken_heads = drain(&mut broken).await;

        assert_eq!(
            compose_heads, broken_heads,
            "from the abandonment onward the two arms must offer the same answers"
        );
        assert_eq!(compose_heads, vec![9], "both fall back to base@9");
        assert_eq!(
            compose.exhaustion(),
            broken.exhaustion(),
            "the exhaustion verdict must not depend on WHY the head was abandoned"
        );
        assert!(matches!(
            compose.exhaustion(),
            Some(Exhaustion::AllBroken { candidates: 2 })
        ));

        // The step-down records agree on everything except the reason — which is
        // the point: one vocabulary, two producers.
        let compose_record = &compose.step_downs()[0];
        let broken_record = &broken.step_downs()[0];
        assert_eq!(compose.step_downs().len(), 1);
        assert_eq!(broken.step_downs().len(), 1);
        assert_eq!(compose_record.head_seq, broken_record.head_seq);
        assert_eq!(compose_record.head_kind, broken_record.head_kind);
        assert_eq!(compose_record.head_seq, 12);
        assert_eq!(compose_record.head_kind, CheckpointKind::Tail);
        assert_eq!(
            compose_record.reason,
            LineageBreak::Compose { head_seq: 12 },
            "the caller's verdict is recorded verbatim, naming the abandoned head"
        );
        assert_eq!(
            compose_record.reason.link_seq(),
            broken_record.reason.link_seq()
        );
        assert_eq!(compose_record.reason.reason_label(), "stepped_down_compose");
        assert_eq!(broken_record.reason.reason_label(), "stepped_down_decode");
        assert_ne!(
            compose_record.reason.reason_label(),
            broken_record.reason.reason_label(),
            "the two step-downs are separable in the record even though the outcomes match"
        );
    }

    // ── Group V: outcome-label vocabulary membership ────────────────────────

    /// Every label this module can mint is a member of the vocabulary
    /// `metrics.rs` pins.
    ///
    /// The const is referenced, never re-spelled as literals: a literal here
    /// would drift from the very vocabulary it is supposed to be checked
    /// against. A label outside it renders as a metric series the dashboards
    /// never seed.
    #[test]
    fn every_label_this_module_mints_is_a_member_of_the_pinned_metric_vocabulary() {
        use crate::observability::metrics::CHECKPOINT_RESTORE_LINEAGE_OUTCOMES;

        let base = CheckpointBlob {
            handle: CheckpointHandle::base(OBJECT_ID, 10, "4242.10".to_string()),
            bytes: Bytes::from_static(b"base"),
            meta: base_meta(10),
        };
        let composed = GatheredLineage {
            base: base.clone(),
            tails: vec![GatheredTail {
                handle: CheckpointHandle::tail(OBJECT_ID, 14, "4242.14.tail".to_string()),
                bytes: tail_bytes(OBJECT_ID, 10, 10, 14),
                header: TailHeader {
                    object_id: OBJECT_ID,
                    base_snap_seq: 10,
                    from_seq: 10,
                    to_seq: 14,
                },
                format_version: 1,
                meta: tail_meta(10, 10, 14),
            }],
            head_seq: 14,
        };
        let base_only = GatheredLineage {
            base,
            tails: Vec::new(),
            head_seq: 10,
        };

        let mut labels: Vec<&'static str> = vec![
            base_only.outcome_label(),
            composed.outcome_label(),
            Exhaustion::EmptyListing.outcome_label(),
            Exhaustion::NoBase {
                generations: 2,
                newest_sequence: 20,
            }
            .outcome_label(),
            Exhaustion::AllBroken { candidates: 3 }.outcome_label(),
            // The one exhaustion that does NOT fold onto `exhausted` (b.hxk).
            // Without a producer here the anti-vacuity count below would fall
            // one short of the vocabulary and this test would be the thing that
            // notices — which is the point: every pinned outcome must have a
            // minting site in this module.
            Exhaustion::BudgetInconclusive {
                candidates: 3,
                unadjudicated_bases: 1,
            }
            .outcome_label(),
        ];
        labels.extend(
            [
                LineageBreak::LinkAbsent {
                    seq: 14,
                    kind: CheckpointKind::Tail,
                    at_fetch: false,
                },
                LineageBreak::LinkMismatch {
                    seq: 14,
                    field: "base_snap_seq",
                    expected: "10".to_string(),
                    observed: "5".to_string(),
                },
                LineageBreak::Decode {
                    seq: 14,
                    message: "bad magic".to_string(),
                },
                // A torn put, NOT a decode failure: the artifact reads fine
                // and its index never landed. Its own label is the whole
                // point — the two carry different operator remediations.
                LineageBreak::MissingMetaCompanion {
                    seq: 14,
                    message: "meta companion absent".to_string(),
                },
                LineageBreak::Budget {
                    seq: 14,
                    remaining_ms: 0,
                    reserve_ms: 30_000,
                },
                LineageBreak::Compose { head_seq: 14 },
            ]
            .iter()
            .map(LineageBreak::reason_label),
        );

        for label in &labels {
            assert!(
                CHECKPOINT_RESTORE_LINEAGE_OUTCOMES.contains(label),
                "`{label}` is not a member of CHECKPOINT_RESTORE_LINEAGE_OUTCOMES — the metric \
                 cell would never be seeded"
            );
        }

        // Anti-vacuity: the constructed set must exercise the whole
        // vocabulary, not label everything the same way. The three CONCLUSIVE
        // `Exhaustion` variants deliberately fold onto one label and
        // `BudgetInconclusive` carries its own, so four variants contribute two
        // distinct labels and the distinct count is the vocabulary size.
        let distinct: HashSet<&str> = labels.into_iter().collect();
        assert_eq!(
            distinct.len(),
            CHECKPOINT_RESTORE_LINEAGE_OUTCOMES.len(),
            "every pinned outcome must have a producer in this module; got {distinct:?}"
        );
    }
}
