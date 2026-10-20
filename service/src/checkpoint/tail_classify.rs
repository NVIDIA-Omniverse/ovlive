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

//! The tail-capture **completeness gate**: a crate-internal seam over the
//! storage primitive that reads a stream slice, and the pure classifier that
//! turns that slice into either a sound replayable prefix or exactly one refusal
//! (SDD SR-1.2 / SR-1.3 / SR-1.4).
//!
//! This module holds the *decision* half of tail capture. The *read* half is
//! [`crate::storage::tail_slice`]; the *capture unit* that orchestrates the two,
//! mints the artifact, and increments metrics is `tail_capture` (sibling Task 3)
//! — which is why this module is named `tail_classify` and leaves that name free.
//!
//! # Why the seam lives here, not in `storage`
//!
//! [`TailSliceLoader`] is declared in `checkpoint`, exactly as
//! [`crate::checkpoint::assembler::SnapshotLoader`] is (production impl
//! `ValkeySnapshotLoader`, which likewise delegates into `storage`). Two
//! reasons, both structural:
//!
//! 1. **The consumer owns the seam** — the established pattern in this codebase.
//!    The capture unit is what needs to inject a fake, so the trait it injects
//!    against belongs beside it.
//! 2. **A seam declared in `storage` could not carry the capture unit's other
//!    read.** That read — the checkpoint **store listing** which defines S0 — is
//!    a [`crate::checkpoint::CheckpointStore`] call. Putting it behind a
//!    storage-declared trait would mint a new concrete `storage → checkpoint`
//!    edge, which CLAUDE.md's *Module Dependency Rules* inversion rule forbids.
//!
//! The classifier is here for the mirror-image reason: its output is
//! [`TailRecord`] and its outcome vocabulary is the checkpoint metric family's.
//! `checkpoint → storage` is an allowed edge; the reverse is not.
//!
//! # The S0 contract, and the seam's deliberate minimality
//!
//! `base_snap_seq` (S0) is **the newest restorable generation's sequence per the
//! checkpoint STORE LISTING** — the lineage head, base or tail. It is *never*
//! the `le:{object_id}:obj:hdr` `last_checkpoint_seq` cursor: SDD §22.7 step 8
//! tolerates that cursor lagging when its write fails, and a lagging cursor
//! would re-read commits an existing generation already covers, minting
//! overlapping tails. **Enforcing that provenance is Task 3's call site**, the
//! only code holding the store listing; neither the seam nor the classifier can
//! check it.
//!
//! The seam carries **the slice read and nothing else**, deliberately. The
//! capture unit's other two reads are already served: the store listing sits
//! behind `CheckpointStore`, and whatever `obj:hdr` read the unit needs (see the
//! cap/jump complement below) is Task 3's to add to this trait when it lands.
//! Adding speculative methods now would leave unimplemented surface on the fake.
//!
//! ## S0 ≥ 1, and the CreateLayer seed-entry exclusion it buys
//!
//! `S0 ≥ 1` always: SR-7.1 roots every lineage at `snap_seq = 1`, and a layer
//! whose listing cannot root a lineage refuses at Task 3's call site before this
//! function is reached — `refused_no_base` when the listing is empty,
//! `refused_lineage_unrootable` when it holds only tails or its head sits at
//! sequence 0. That is **load-bearing, not incidental**.
//!
//! `CreateLayer` XADDs a seed entry at `1-0` (`LayerStore::write_object`'s
//! `initial_stream_payload` arm, `src/storage/layer.rs`). That entry is
//! `update`-typed and carries **no `ompf` field** — yet it is emphatically NOT
//! an empty commit: it accompanies the full initial layer content. Under this
//! classifier's rules it would decode as an [`TailRecord::EmptyDelta`] and
//! replay as a no-op over real content — a silent-corruption defect. It is
//! unreachable only because the read window starts at `{S0+1}-0` and `S0 ≥ 1`,
//! so the window never reaches back to `1-0`.
//!
//! The restore path adds no second ompf-less producer of that shape: it
//! deliberately XADDs nothing (`crate::checkpoint::apply`, the restore-seed
//! path — "make the static layer state visible", not "synthesize a synthetic
//! update event"). [`classify_tail_slice`] `debug_assert!`s `base_snap_seq >= 1`
//! so a caller that loses the precondition fails loudly in test builds; the
//! production guard remains Task 3's `s0 == 0` gate, which reports
//! `refused_lineage_unrootable`.
//!
//! # Empty commits are classified BY INVARIANT, not by byte shape
//!
//! An `update`-typed entry at an explicit `{seq}-0` ID with **no** `ompf` field
//! is captured as [`TailRecord::EmptyDelta`] — the §8.4 empty-commit shape that
//! advanced the sequence without changing state (SR-1.2).
//!
//! The classification cannot be byte-shape-based, because the byte shape alone
//! is *ambiguous*. `RELAY_OMPF_FIELD`'s documented absence list
//! (`src/storage/notifications.rs`) names three producers of an ompf-less
//! `update` entry: CreateLayer's `1-0` seed entry, an empty-OMPF update, and a
//! pre-b.gze entry still inside the retention window. A pre-b.gze non-verbatim
//! entry is byte-identical to the empty-commit shape.
//!
//! The invariant that resolves it is **every non-empty update commit XADDs the
//! `ompf` field** — already load-bearing for the b.gze steady-state verbatim
//! relay, not new to this feature. Under it, and given the seed-entry exclusion
//! above, an ompf-less `update` entry inside the read window *is* an empty
//! commit for every reachable stream: pre-verbatim-relay entries are
//! pre-release-unreachable (no store predating the b.gze relay exists). A
//! violation's failure mode is silent — a non-empty commit misclassified as
//! empty replays as a no-op over a real state change — which is why the
//! invariant is enforced on the write side rather than trusted here:
//! `LayerStore::update_layer_with_summary` refuses a delta without non-empty
//! relay OMPF before any store I/O, on every backend, and writes no zero-length
//! `ompf` field. `SubscribeReadLayer` relies on the same
//! classification: its steady state and its rollback-mode in-window replay send
//! an ompf-less entry above sequence 1 as the empty delta, with no read.
//!
//! # The stream ID is authoritative
//!
//! Every commit entry carries its sequence **twice**: in the explicit stream ID
//! `{seq}-0`, and in the pipe-delimited `payload` field surfaced by
//! `LayerNotification::parse`. Record sequences and `to_seq` are sourced from
//! the **stream ID**, never re-derived from the payload text.
//!
//! A single writer emits both halves in one `XADD`, so a disagreement between
//! them means corruption or a foreign writer. That entry's evidence is
//! self-contradictory, so it is a `MissingBytes` terminal — a release-safe
//! defense-in-depth arm, unlike the contiguity `debug_assert!`s which vanish in
//! release builds.
//!
//! # The b.pw6 rule
//!
//! **A generation's claimed range MUST equal its actual content.** Claiming
//! coverage the records do not supply is the b.pw6 silent-content-loss defect,
//! and it is forbidden in *every* arm — including both cap arms, where a shorter
//! sound tail is always preferable to a longer claimed one. Every
//! [`TailClassification::Captured`] construction path carries a `debug_assert!`
//! that the records are exactly contiguous over `from_seq..=to_seq` with
//! `records.len() == to_seq - from_seq + 1`. [`TailCaptured`] additionally keeps
//! its record vector private so a zero-record capture is unconstructible from
//! outside this module: SR-1.3 forbids persisting an empty prefix, and
//! [`crate::checkpoint::tail_codec`] happily encodes a zero-record envelope, so
//! the type is where that is caught.
//!
//! # The cap/jump boundary, and what this module does NOT close
//!
//! When a cap truncates the read, [`TailSlice::lookahead`] carries the one entry
//! immediately past it. The cap arm inspects it so SR-1.2's
//! refuse-everything-on-jump rule still applies **at the cap boundary**: a cap
//! landing inside a post-restore fossil region would otherwise hide the
//! ascending jump immediately beyond it, and the capped prefix would extend a
//! lineage with commits from an abandoned pre-restore timeline.
//!
//! **The lookahead closes only the boundary case.** A jump further beyond the
//! cap is invisible to the slice by construction. The complementary guard is the
//! `obj:hdr` `restore_seq_bump != 0` check — the column every restore stamps
//! (`crate::checkpoint::apply`) and a landed post-restore generation clears
//! (`crate::checkpoint::checkpointer`). That guard belongs to **Task 3's call
//! site and is deliberately NOT implemented here**; recorded so it cannot be
//! forgotten.
//!
//! # Outcome vocabulary — who owns which spelling
//!
//! The canonical list is `CHECKPOINT_TAIL_OUTCOMES`
//! (`src/observability/metrics.rs`); nothing re-spells it as a free literal.
//! Ownership of the ten values:
//!
//! - **This module** owns the four refusal spellings reachable from
//!   classification: `refused_trimmed`, `refused_missing_bytes`,
//!   `refused_sentinel`, `refused_seq_gap` — via
//!   [`TailRefusal::outcome_label`], each appearing in exactly one place.
//! - **The capture unit** (`tail_capture`, Task 3) owns `captured` /
//!   `captured_capped` (only it knows the capped/uncapped distinction at the
//!   *outcome* level), plus its three call-site refusals `refused_no_base`
//!   (the listing is empty), `refused_lineage_unrootable` (generations exist
//!   but none can root a lineage — Epic 5's SR-7 split of the
//!   checkpoint-store-damage arms out of `refused_no_base`) and
//!   `refused_chain_full`, plus its fold of this module's `SeqGap`.
//!   [`TailClassification`] deliberately carries **no** `outcome_label`.
//! - **The chain-collapse re-base unit** (`checkpoint::rebase`) owns `rebased`,
//!   minted by
//!   [`RebaseDecision::outcome_label`](crate::checkpoint::rebase::RebaseDecision::outcome_label)
//!   and ticked by `checkpointer::rebase_on_chain_full`. That unit also folds
//!   every SR-4.5 refusal cause onto the capture unit's `refused_chain_full`
//!   rather than minting its own spelling. (The SR-3.6 *post-restore* re-base in
//!   `checkpoint::recovery` mints nothing in this family — it is a write, not a
//!   capture outcome.)
//!
//! # Determinism contract
//!
//! Mirrors [`crate::checkpoint::tail_codec`]'s: [`classify_tail_slice`] is a
//! **total, synchronous, I/O-free, clock-free** function of its two arguments,
//! reviewable by inspection because the classification path may not contain any
//! of the following:
//!
//! - `std::time`, `Instant`, `SystemTime`, or any other clock read — in
//!   particular `now_unix_ms()` (`src/checkpoint/mod.rs`). Record timestamps are
//!   carried in from the captured stream entries, never read from the clock.
//! - Randomness of any kind.
//! - `async` functions or `await` points.
//! - `fred` (no Valkey), `cxx` / `oodf_ffi` (no FlatBuffers), or `OmpfContainer`
//!   — verbatim payload bytes stay opaque and are never parsed here.
//! - `HashMap` / `HashSet` or any other nondeterministic-iteration collection.
//!
//! The seam above it is async by necessity; nothing async may leak into the
//! classifier.

use async_trait::async_trait;

use crate::checkpoint::assembler::{
    service_err_to_checkpoint_err, HeaderIdentity, SnapshotLoader, ValkeySnapshotLoader,
};
use crate::checkpoint::tail_codec::TailRecord;
use crate::checkpoint::CheckpointError;
use crate::storage::client::ValkeyClient;
use crate::storage::keys::KeySchema;
use crate::storage::notifications::LayerNotification;
use crate::storage::tail_slice::{
    TailEntryPayload, TailSlice, TailSliceCap, TailSliceCaps, TailStreamEntry,
};

// ── The seam ────────────────────────────────────────────────────────────────

/// The storage read tail capture depends on, behind a trait so the capture unit
/// unit-tests against a fake without `TestValkey`.
///
/// See the module header for why this trait is declared in `checkpoint` rather
/// than `storage`, and for the **S0 contract** its `base_snap_seq` argument
/// carries.
///
/// Errors are classified per the load-bearing-error contract by
/// [`service_err_to_checkpoint_err`]. Everything the slice read can produce is
/// [`CheckpointError::Transient`] and retried on the next tick: the storage
/// primitive raises only a transient fred failure or its own ceiling guard, and
/// the shared mapper has no `NotFound` arm on that path. The one method that
/// *can* answer "this object has no usable header" is
/// [`Self::read_header_identity`], via `Ok(None)` — which covers both an absent
/// `obj:hdr` and a present one whose identity columns are empty. The capture
/// unit turns either into [`CheckpointError::NotFound`] so the leader-elected
/// loop can drop the object from its working set.
#[async_trait]
pub(crate) trait TailSliceLoader: Send + Sync {
    /// Read the bounded stream slice starting at `{base_snap_seq + 1}-0`.
    ///
    /// `base_snap_seq` MUST be S0 per the checkpoint store listing, and MUST be
    /// `>= 1` — see the module header; enforcing both is the caller's job.
    async fn read_tail_slice(
        &self,
        object_id: u64,
        base_snap_seq: u64,
        caps: TailSliceCaps,
    ) -> Result<TailSlice, CheckpointError>;

    /// Read the `obj:hdr` columns in one `HGETALL`, for the capture unit's two
    /// consumers of them: the post-restore fossil guard
    /// (`restore_seq_bump` / `cursor_columns_trusted`) and the tail
    /// `CheckpointMeta`'s four identity fields.
    ///
    /// `Ok(None)` has **two** causes, and the capture unit treats them
    /// identically — see
    /// [`SnapshotLoader::read_header_identity`](crate::checkpoint::assembler::SnapshotLoader::read_header_identity),
    /// whose contract this one inherits verbatim through the delegating
    /// production impl below:
    ///
    /// 1. `obj:hdr` is absent — the layer was deleted or overwritten out from
    ///    under the capture;
    /// 2. `obj:hdr` exists but its `session_id` or `layer_name` is missing or
    ///    empty, so any generation stamped from it would be permanently
    ///    unrestorable (one WARN at the delegate).
    ///
    /// Either way the capture unit maps it to [`CheckpointError::NotFound`]
    /// rather than folding to defaults: unlike the assembler's same-shaped read,
    /// nothing has been built yet at that point, and a fabricated identity would
    /// feed a *guard* whose whole job is to distrust fabricated columns.
    ///
    /// Deliberately NOT a second header struct: `HeaderIdentity` already carries
    /// everything both consumers need, and the production impl delegates to the
    /// assembler's rather than re-parsing the hash.
    async fn read_header_identity(
        &self,
        object_id: &str,
    ) -> Result<Option<HeaderIdentity>, CheckpointError>;
}

/// Production [`TailSliceLoader`], delegating to the storage primitive.
pub(crate) struct ValkeyTailSliceLoader<'a> {
    pub client: &'a ValkeyClient,
    pub keys: &'a KeySchema,
}

#[async_trait]
impl<'a> TailSliceLoader for ValkeyTailSliceLoader<'a> {
    async fn read_tail_slice(
        &self,
        object_id: u64,
        base_snap_seq: u64,
        caps: TailSliceCaps,
    ) -> Result<TailSlice, CheckpointError> {
        // Direction: checkpoint → storage, never the reverse. The storage
        // primitive owns the paging loop; this seam calls it once.
        crate::storage::tail_slice::read_tail_slice(
            self.client,
            self.keys,
            object_id,
            base_snap_seq,
            caps,
        )
        .await
        .map_err(service_err_to_checkpoint_err)
    }

    async fn read_header_identity(
        &self,
        object_id: &str,
    ) -> Result<Option<HeaderIdentity>, CheckpointError> {
        // Delegate to the assembler's production implementation rather than
        // copying it: one HGETALL parser, one cursor-trust policy, one place a
        // header-column change lands. Both loaders are `pub(crate)` structs of
        // the same two `pub` borrowed fields, so this is a construction, not a
        // refactor of `assembler.rs`.
        ValkeySnapshotLoader {
            client: self.client,
            keys: self.keys,
        }
        .read_header_identity(object_id)
        .await
    }
}

// ── Classification result types ─────────────────────────────────────────────

/// Why a tail capture found no sound prefix to persist.
///
/// `pub` (not `pub(crate)`) because it becomes a field of Task 3's `pub` capture
/// refusal type, and a `pub` type embedding a `pub(crate)` one trips
/// `private_interfaces` under the `-D warnings` gate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TailRefusal {
    /// The stream head was trimmed past the expected first ID — or, at index 0,
    /// the first entry simply is not `{S0+1}-0` (including an ID *above* it, the
    /// post-restore fresh-stream case of SR-3.6). Also the genuinely empty
    /// window. Operator levers: the stream-retention knobs
    /// `LIVE_EDIT_LAYER_STREAM_MAXLEN` (the entry cap) and
    /// `LIVE_EDIT_LAYER_STREAM_MAX_BYTES` (the relay byte budget); retention is
    /// the smaller of the two windows.
    Trimmed,
    /// The evidence for the next commit is unusable: an undecodable entry, a
    /// zero-length `ompf` field, an `update` at a non-`-0` (auto) ID, an entry
    /// whose stream ID and payload sequence disagree, an `update` below the
    /// expected sequence — or no admissible first entry at all, because the byte
    /// cap excluded it (SR-8.3's residual class; operator lever
    /// `LIVE_EDIT_CHECKPOINT_TAIL_MAX_BYTES`, *not* the retention knob).
    MissingBytes,
    /// A `deleted` layer-retirement sentinel was reached with no sound prefix
    /// before it.
    Sentinel,
    /// An ascending jump in the `+1` progression. The **only** refusal that
    /// discards an accumulated non-empty prefix — see [`classify_tail_slice`].
    SeqGap,
}

impl TailRefusal {
    /// This refusal's `outcome` label on
    /// `live_edit_checkpoint_tail_captures_total`.
    ///
    /// Every returned value is a member of `CHECKPOINT_TAIL_OUTCOMES`
    /// (`src/observability/metrics.rs`), which is the canonical vocabulary; the
    /// membership is asserted by a unit test below. The other five members of
    /// that const are minted elsewhere — see the module header's ownership map.
    pub fn outcome_label(&self) -> &'static str {
        match self {
            TailRefusal::Trimmed => "refused_trimmed",
            TailRefusal::MissingBytes => "refused_missing_bytes",
            TailRefusal::Sentinel => "refused_sentinel",
            TailRefusal::SeqGap => "refused_seq_gap",
        }
    }
}

/// A sound, replayable prefix — always non-empty (SR-1.3).
///
/// `records` is private, which is what makes a zero-record capture
/// unconstructible outside this module; read it through [`Self::records`] or
/// take ownership with [`Self::into_records`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct TailCaptured {
    /// The captured commits, in stream order, contiguous over
    /// `from_seq..=to_seq`. Private — see the type doc.
    records: Vec<TailRecord>,
    /// Sequence number of the first captured record — **inclusive; the tail
    /// envelope header's `from_seq` is this minus one.**
    ///
    /// The two spellings genuinely differ by one and the trap is easy to walk
    /// into: [`crate::checkpoint::tail_codec::TailHeader::from_seq`] is the
    /// **exclusive** lower bound of the covered range `(from_seq, to_seq]` (the
    /// encoder enforces `expected = header.from_seq + 1 + count`), whereas this
    /// field is the first record's own sequence, i.e. `base_snap_seq + 1`.
    /// Task 3's capture unit builds the envelope header from its own S0, never
    /// from this field.
    pub from_seq: u64,
    /// Sequence number of the last captured record.
    pub to_seq: u64,
    /// `true` when a cap truncated the read and more entries were pending — the
    /// `captured_capped` outcome. The claimed range still equals the content, so
    /// this is not an error signal.
    pub capped: bool,
}

impl TailCaptured {
    /// The captured commits, in stream order.
    pub fn records(&self) -> &[TailRecord] {
        &self.records
    }

    /// Take ownership of the captured commits, e.g. to feed
    /// [`crate::checkpoint::tail_codec::encode_tail_envelope`].
    pub fn into_records(self) -> Vec<TailRecord> {
        self.records
    }

    /// Number of captured commits. Always `>= 1`.
    pub fn record_count(&self) -> usize {
        self.records.len()
    }
}

/// What [`classify_tail_slice`] decided about a slice.
///
/// Deliberately carries **no** `outcome_label` — see the module header's
/// vocabulary-ownership map.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum TailClassification {
    /// A sound, replayable, non-empty prefix to persist.
    Captured(TailCaptured),
    /// No sound prefix; persist nothing and re-attempt next tick (SR-1.3).
    Refused(TailRefusal),
}

// ── The classifier ──────────────────────────────────────────────────────────

/// Per-entry verdict inside the walk.
enum Step {
    /// The entry is replayable and extends the prefix.
    Record(TailRecord),
    /// The prefix ends here, for this reason.
    Terminal(TailRefusal),
}

/// Classify a tail slice into a sound replayable prefix or one refusal.
///
/// Pure: synchronous, I/O-free, clock-free, total — see the module header's
/// determinism contract.
///
/// # Arm evaluation order (normative)
///
/// 1. **Empty-entries disposition.** `entries` empty + `cap_hit == None` →
///    `Trimmed` (a genuinely empty `XRANGE` window: the base is at or ahead of
///    the stream head). `entries` empty + `cap_hit == Some(_)` → `MissingBytes`:
///    the byte cap excluded the very first entry, so the honest statement is
///    "no admissible first entry". Folding *that* to `Trimmed` is **forbidden**
///    — trimmed's documented remediation is the stream-retention knobs
///    (`LIVE_EDIT_LAYER_STREAM_MAXLEN`, `LIVE_EDIT_LAYER_STREAM_MAX_BYTES`),
///    which are the wrong knobs for a commit that is simply larger than
///    `LIVE_EDIT_CHECKPOINT_TAIL_MAX_BYTES` (SR-8.3's reachable residual class).
///    `Some(Entries)` here is unreachable (`max_entries` is zero-rejected at
///    config) and is `debug_assert!`ed.
/// 2. **The index-0 first-entry rule**, which is spec-pinned and *replaces* the
///    ID-progression disposition below for the first entry only: a first
///    `update` entry whose ID is not exactly `{S0+1}-0` — **including one above
///    it** — is `Trimmed`, never `SeqGap`. SR-3.6 depends on this: on a fresh
///    post-restore stream the first commit XADDs at `{seeded_seq+1}-0` and must
///    read as `trimmed`, while the surviving-stream case puts the jump mid-slice
///    where `seq_gap` catches it.
/// 3. **Per-entry shape and progression rules**, walked in order with
///    `expected = base_snap_seq + 1`. Entry *kind* is judged before entry *ID*
///    (a `deleted` sentinel is `Sentinel` wherever it sits, including index 0
///    and including an auto ID numerically far above `expected`; an undecodable
///    entry is `MissingBytes` likewise) — the ID progression is only meaningful
///    for entries that claim to be commits.
/// 4. **The cap arm**, reached only when the prefix consumed every entry. It can
///    therefore never see an empty `entries` vec, which is what keeps
///    [`TailCaptured`] unconstructible with zero records.
///
/// # Terminal disposition (SR-1.3's asymmetry)
///
/// A terminal reached with a **non-empty** prefix yields `Captured` — a shorter
/// tail still advances the recovery point and its claimed range equals its
/// content. A terminal at index 0 yields the matching `Refused`. `SeqGap` is the
/// sole exception to both: it always refuses **everything**, discarding any
/// accumulated prefix. Entries below an ascending jump are fossils of an
/// abandoned pre-restore timeline (SR-1.2 / SR-3.6); composing them would mint a
/// generation whose restore resurrects state the live timeline has already
/// superseded. At a page or cap boundary the same jump is alternatively a
/// mid-capture trim race — refusal is sound under both readings.
///
/// # Panics
///
/// `debug_assert!`s that `base_snap_seq >= 1` (test builds only). The production
/// guard is Task 3's `s0 == 0` gate, which refuses
/// `refused_lineage_unrootable`: SR-1.5 / SR-7.1 root every lineage at
/// `snap_seq = 1`, and the precondition is what keeps the read window clear of
/// `CreateLayer`'s ompf-less `1-0` seed entry — see the module header.
pub(crate) fn classify_tail_slice(slice: &TailSlice, base_snap_seq: u64) -> TailClassification {
    debug_assert!(
        base_snap_seq >= 1,
        "classify_tail_slice requires S0 >= 1 (every lineage is rooted at snap_seq = 1; a \
         layer whose listing cannot root a lineage refuses `no_base` or `lineage_unrootable` at \
         the capture call site before this runs). A 0 here means the read window starts at \
         CreateLayer's ompf-less `1-0` seed entry, which this classifier would mis-capture as an \
         empty commit"
    );

    // (1) Empty-entries disposition. Runs first so the cap arm below can never
    //     see an empty `entries` vec.
    if slice.entries.is_empty() {
        debug_assert!(
            !matches!(slice.cap_hit, Some(TailSliceCap::Entries)),
            "an entries-cap stop with zero admitted entries requires max_entries == 0, which \
             LIVE_EDIT_CHECKPOINT_TAIL_MAX_ENTRIES zero-rejects at config"
        );
        return match slice.cap_hit {
            None => TailClassification::Refused(TailRefusal::Trimmed),
            Some(_) => TailClassification::Refused(TailRefusal::MissingBytes),
        };
    }

    let from_seq = base_snap_seq.saturating_add(1);
    let mut expected = from_seq;
    let mut records: Vec<TailRecord> = Vec::with_capacity(slice.entries.len());

    // (2) + (3) The walk. `index == 0` selects the first-entry rule inside
    //     `classify_entry`; everything else is the general progression.
    for (index, entry) in slice.entries.iter().enumerate() {
        match classify_entry(entry, index, expected) {
            Step::Record(record) => {
                records.push(record);
                expected = expected.saturating_add(1);
            }
            // SeqGap discards everything, including a non-empty prefix.
            Step::Terminal(TailRefusal::SeqGap) => {
                return TailClassification::Refused(TailRefusal::SeqGap);
            }
            Step::Terminal(refusal) => {
                return if records.is_empty() {
                    TailClassification::Refused(refusal)
                } else {
                    // Stopped early, so nothing was truncated by a cap.
                    captured(records, from_seq, false)
                };
            }
        }
    }

    // (4) Cap arm. Every entry was replayable; `records` is non-empty.
    let capped = if slice.cap_hit.is_some() {
        if let Some(lookahead) = slice.lookahead.as_ref() {
            if is_ascending_jump(lookahead, expected) {
                // The boundary fossil case: the cap landed immediately below an
                // ascending jump. Refuse the whole prefix, as a mid-slice jump
                // would. See the module header for what this does NOT reach.
                return TailClassification::Refused(TailRefusal::SeqGap);
            }
        }
        true
    } else {
        false
    };

    captured(records, from_seq, capped)
}

/// Build the `Captured` variant, enforcing the b.pw6 rule on the way through.
///
/// The only construction path for [`TailCaptured`]. Both call sites guard
/// `records` non-empty; the empty arm here is unreachable and refuses rather
/// than minting a claim no content backs.
fn captured(records: Vec<TailRecord>, from_seq: u64, capped: bool) -> TailClassification {
    let Some(last) = records.last() else {
        debug_assert!(
            false,
            "captured() called with zero records — a tail must never persist an empty prefix"
        );
        return TailClassification::Refused(TailRefusal::Trimmed);
    };
    let to_seq = last.sequence_num();

    // The b.pw6 rule: claimed range == actual content, in every arm.
    debug_assert!(
        records
            .iter()
            .enumerate()
            .all(|(i, r)| r.sequence_num() == from_seq.saturating_add(i as u64)),
        "captured records are not exactly contiguous from from_seq={from_seq}"
    );
    debug_assert_eq!(
        records.len() as u64,
        to_seq - from_seq + 1,
        "captured record count does not fill the claimed range ({from_seq}..={to_seq})"
    );

    TailClassification::Captured(TailCaptured {
        records,
        from_seq,
        to_seq,
        capped,
    })
}

/// Judge one entry: does it extend the prefix, or end it (and why)?
///
/// Kind before ID — see [`classify_tail_slice`]'s evaluation order.
fn classify_entry(entry: &TailStreamEntry, index: usize, expected: u64) -> Step {
    let notification = match &entry.payload {
        // Preserved-not-skipped by the read primitive precisely so the prefix
        // terminates here instead of stepping over a hole.
        TailEntryPayload::Undecodable { .. } => return Step::Terminal(TailRefusal::MissingBytes),
        TailEntryPayload::Notification(notification) => notification,
    };

    let LayerNotification::Update {
        sequence_num,
        server_ts,
        relay_ompf,
        ..
    } = notification
    else {
        // `LayerNotification::Deleted` — the layer-retirement sentinel, always
        // written at an auto ID. Terminal wherever it appears: Valkey rejects an
        // explicit XADD ID at or below the last delivered one, so no commit
        // entry can follow it. The at-any-index check is defense-in-depth, not a
        // live case.
        return Step::Terminal(TailRefusal::Sentinel);
    };

    // Commit entries carry an explicit `{seq}-0` ID. Anything else — an auto
    // `{ms}-{n}` ID, an explicit ID with a non-zero sub-counter, an unparseable
    // one — is not evidence this classifier can use.
    let Some(id_seq) = parse_commit_id(&entry.stream_id) else {
        return Step::Terminal(TailRefusal::MissingBytes);
    };

    // Defense in depth: the sequence is carried twice by one writer in one
    // XADD, so a disagreement is corruption or a foreign writer. The ID wins,
    // and a self-contradictory entry is not replayable.
    if id_seq != *sequence_num {
        return Step::Terminal(TailRefusal::MissingBytes);
    }

    if id_seq != expected {
        return Step::Terminal(if index == 0 {
            // The spec-pinned first-entry rule. An ID *below* `expected` is
            // additionally unreachable here: the read window's inclusive start
            // is `{expected}-0`, so `XRANGE` cannot return anything lower.
            TailRefusal::Trimmed
        } else if id_seq > expected {
            TailRefusal::SeqGap
        } else {
            // A repeat / regression mid-slice: the stream cannot supply the
            // commit the prefix needs next.
            TailRefusal::MissingBytes
        });
    }

    match relay_ompf {
        // Empty commit BY INVARIANT, not by byte shape — see the module header.
        None => Step::Record(TailRecord::EmptyDelta {
            sequence_num: id_seq,
            server_ts: *server_ts,
        }),
        // Defense in depth: no producer writes a present-but-empty `ompf` (both
        // commit paths add the field only when non-empty), and `tail_codec`
        // rejects a zero-payload `Verbatim` on encode — so admitting it would
        // build an un-encodable record.
        Some(ompf) if ompf.is_empty() => Step::Terminal(TailRefusal::MissingBytes),
        Some(ompf) => Step::Record(TailRecord::Verbatim {
            sequence_num: id_seq,
            server_ts: *server_ts,
            ompf: ompf.clone(),
        }),
    }
}

/// Parse a commit stream ID: `Some(seq)` iff the ID is literally `{seq}-0`.
///
/// `None` for auto IDs (`{ms}-{n}`), for an explicit ID with a non-zero
/// sub-counter, and for anything unparseable.
fn parse_commit_id(stream_id: &str) -> Option<u64> {
    let (seq, sub) = stream_id.split_once('-')?;
    if sub != "0" {
        return None;
    }
    seq.parse::<u64>().ok()
}

/// Is this lookahead entry an `update` at an explicit `{a}-0` with `a > expected`
/// — i.e. the boundary fossil case the cap arm refuses on?
fn is_ascending_jump(lookahead: &TailStreamEntry, expected: u64) -> bool {
    let TailEntryPayload::Notification(LayerNotification::Update { .. }) = &lookahead.payload
    else {
        return false;
    };
    parse_commit_id(&lookahead.stream_id).is_some_and(|id_seq| id_seq > expected)
}

#[cfg(test)]
pub(crate) mod tests {
    //! Seam smoke test, outcome-vocabulary membership, and the exhaustive
    //! classifier arm matrix.
    //!
    //! `pub(crate)` on the module, and on [`FakeTailSliceLoader`] below, so the
    //! capture unit's own test lane (`tail_capture.rs`) drives the seam through
    //! the SAME fake rather than a second one that could drift from this
    //! module's view of the trait. Nothing outside `#[cfg(test)]` can reach it.
    //!
    //! [`classify_tail_slice`] is a free function over a plain [`TailSlice`]
    //! value, so the arm matrix needs **no seam mock at all** — every case
    //! builds the slice it wants directly through the entry builders below and
    //! asserts on the returned [`TailClassification`]. Nothing in the matrix
    //! touches I/O, a clock, or a runtime, matching the module header's
    //! determinism contract. The [`TailSliceLoader`] fake is a separate concern
    //! kept here for Task 3's capture-unit tests to reuse.
    //!
    //! The read primitive's own paging / cap / preserve-don't-skip behaviours
    //! are **not** re-tested here: they are value-level inputs to this module,
    //! and their tests run against real Valkey in
    //! [`crate::storage::tail_slice`]'s own `#[cfg(test)]` module.
    //!
    //! **The load-bearing claim of the matrix is SR-1.3's asymmetry**, and
    //! `classify_tail_slice_identical_prefix_captures_on_sentinel_but_refuses_everything_on_seq_gap`
    //! is where it is pinned as a *pair* rather than inferred by reading two
    //! tests side by side: a non-empty sound prefix survives every terminal
    //! except `SeqGap`, which discards it whole.
    use super::*;
    use crate::observability::metrics::CHECKPOINT_TAIL_OUTCOMES;
    use bytes::Bytes;
    use std::sync::Mutex;

    /// Method name [`FakeTailSliceLoader::call_log`] records for a slice read.
    pub(crate) const READ_TAIL_SLICE: &str = "read_tail_slice";
    /// Method name [`FakeTailSliceLoader::call_log`] records for a header read.
    pub(crate) const READ_HEADER_IDENTITY: &str = "read_header_identity";

    /// Hand-rolled [`TailSliceLoader`] fake — one canned answer per call, in
    /// order, plus a record of the arguments it was driven with. Mirrors
    /// `assembler.rs`'s `FakeLoader` pattern.
    ///
    /// [`CheckpointError`] is not `Clone`, so answers are *taken* rather than
    /// copied; an exhausted queue answers `NotFound`, which is the arm a caller
    /// is least likely to mistake for a canned success.
    ///
    /// The header answer is a *repeatable* canned value rather than a queue: it
    /// describes a state of `obj:hdr`, not a sequence of events, and every
    /// caller reads it at most once per invocation.
    ///
    /// [`Self::call_log`] interleaves BOTH methods in invocation order, which is
    /// what lets the capture unit's tests assert that a refusal arm ran *before*
    /// the read it fences rather than merely instead of it.
    pub(crate) struct FakeTailSliceLoader {
        answers: Mutex<Vec<Result<TailSlice, CheckpointError>>>,
        calls: Mutex<Vec<(u64, u64, TailSliceCaps)>>,
        /// `Some(identity)` = the header is present and reads as this;
        /// `None` = `obj:hdr` is absent (`Ok(None)`).
        identity: Option<HeaderIdentity>,
        /// One-shot injected `read_header_identity` failure, taken on read and
        /// preferred over [`Self::identity`] when present. One-shot because
        /// [`CheckpointError`] is not `Clone`, and every caller reads the
        /// header at most once per invocation anyway.
        header_error: Mutex<Option<CheckpointError>>,
        call_log: Mutex<Vec<&'static str>>,
    }

    impl FakeTailSliceLoader {
        pub(crate) fn new(answers: Vec<Result<TailSlice, CheckpointError>>) -> Self {
            // Reversed so `pop` consumes in insertion order.
            let mut rev = answers;
            rev.reverse();
            Self {
                answers: Mutex::new(rev),
                calls: Mutex::new(Vec::new()),
                identity: Some(HeaderIdentity::default()),
                header_error: Mutex::new(None),
                call_log: Mutex::new(Vec::new()),
            }
        }

        /// Answer [`TailSliceLoader::read_header_identity`] with this identity.
        ///
        /// The default (see [`Self::new`]) is [`HeaderIdentity::default`], whose
        /// `cursor_columns_trusted: false` the capture unit's post-restore
        /// fossil guard treats as a REFUSAL. A fixture that wants the capture to
        /// proceed must pass an identity with that field `true`.
        pub(crate) fn with_header_identity(mut self, identity: HeaderIdentity) -> Self {
            self.identity = Some(identity);
            self
        }

        /// Answer [`TailSliceLoader::read_header_identity`] with `Ok(None)` —
        /// the layer was deleted or overwritten out from under the capture.
        pub(crate) fn with_absent_header(mut self) -> Self {
            self.identity = None;
            self
        }

        /// Fail [`TailSliceLoader::read_header_identity`] once with this error —
        /// the backing store was unreachable while resolving `obj:hdr`.
        ///
        /// Distinct from [`Self::with_absent_header`]: `Ok(None)` is a *state*
        /// (the layer is gone, a refusal), whereas this is a failed read, which
        /// a caller must propagate classified rather than treat as absence.
        pub(crate) fn with_header_error(self, err: CheckpointError) -> Self {
            *self
                .header_error
                .lock()
                .expect("header error mutex poisoned") = Some(err);
            self
        }

        /// Every `(object_id, base_snap_seq, caps)` triple the caller passed
        /// down, in call order.
        pub(crate) fn calls(&self) -> Vec<(u64, u64, TailSliceCaps)> {
            self.calls.lock().expect("calls mutex poisoned").clone()
        }

        /// Both seam methods' invocations, interleaved in call order.
        pub(crate) fn call_log(&self) -> Vec<&'static str> {
            self.call_log
                .lock()
                .expect("call log mutex poisoned")
                .clone()
        }

        /// How many times [`TailSliceLoader::read_header_identity`] was called.
        pub(crate) fn header_reads(&self) -> usize {
            self.call_log()
                .iter()
                .filter(|method| **method == READ_HEADER_IDENTITY)
                .count()
        }

        fn record(&self, method: &'static str) {
            self.call_log
                .lock()
                .expect("call log mutex poisoned")
                .push(method);
        }
    }

    #[async_trait]
    impl TailSliceLoader for FakeTailSliceLoader {
        async fn read_tail_slice(
            &self,
            object_id: u64,
            base_snap_seq: u64,
            caps: TailSliceCaps,
        ) -> Result<TailSlice, CheckpointError> {
            self.record(READ_TAIL_SLICE);
            self.calls
                .lock()
                .expect("calls mutex poisoned")
                .push((object_id, base_snap_seq, caps));
            self.answers
                .lock()
                .expect("answers mutex poisoned")
                .pop()
                .unwrap_or(Err(CheckpointError::NotFound))
        }

        /// The canned header state — see [`FakeTailSliceLoader::new`] and
        /// [`FakeTailSliceLoader::with_header_identity`] — unless a one-shot
        /// [`FakeTailSliceLoader::with_header_error`] is armed, which wins.
        async fn read_header_identity(
            &self,
            _object_id: &str,
        ) -> Result<Option<HeaderIdentity>, CheckpointError> {
            self.record(READ_HEADER_IDENTITY);
            let injected = self
                .header_error
                .lock()
                .expect("header error mutex poisoned")
                .take();
            if let Some(err) = injected {
                return Err(err);
            }
            Ok(self.identity.clone())
        }
    }

    #[tokio::test]
    async fn tail_slice_loader_fake_returns_canned_slice_and_records_arguments() {
        let caps = TailSliceCaps {
            max_entries: 64,
            max_payload_bytes: 1024,
            page_count: 8,
        };
        let loader = FakeTailSliceLoader::new(vec![Ok(TailSlice::default())]);

        let slice = loader
            .read_tail_slice(42, 7, caps)
            .await
            .expect("canned answer");

        assert_eq!(slice, TailSlice::default());
        assert_eq!(loader.calls(), vec![(42, 7, caps)]);
    }

    /// Case 18 — outcome ↔ label mapping.
    ///
    /// Scoped to [`TailRefusal`] on purpose: `outcome_label` exists on the
    /// refusal enum ONLY, and [`TailClassification`] deliberately carries none
    /// (module header's vocabulary-ownership map). The capture-side spellings
    /// (`captured` / `captured_capped` / `refused_no_base` /
    /// `refused_lineage_unrootable` / `refused_chain_full` and the capture
    /// unit's fold of `SeqGap`) belong to Task 3's membership
    /// test, not this one — duplicating them here would reintroduce the
    /// double-spelling this split closed.
    ///
    /// The `match` is what makes this exhaustive: adding a `TailRefusal`
    /// variant without extending it is a **compile error here**, not a
    /// silently-unlabelled outcome that renders as a missing metric series.
    /// The const is referenced, never re-spelled as string literals — a literal
    /// would drift from the vocabulary it is supposed to be checked against.
    #[test]
    fn refusal_outcome_labels_are_members_of_the_canonical_vocabulary() {
        let all = [
            TailRefusal::Trimmed,
            TailRefusal::MissingBytes,
            TailRefusal::Sentinel,
            TailRefusal::SeqGap,
        ];

        for refusal in all {
            // Exhaustiveness trigger — see the doc-comment.
            match refusal {
                TailRefusal::Trimmed
                | TailRefusal::MissingBytes
                | TailRefusal::Sentinel
                | TailRefusal::SeqGap => {}
            }

            let label = refusal.outcome_label();
            assert!(
                CHECKPOINT_TAIL_OUTCOMES.contains(&label),
                "{refusal:?} labels itself `{label}`, which is not a member of \
                 CHECKPOINT_TAIL_OUTCOMES — the metric cell would never be seeded"
            );
        }

        // Every refusal gets its OWN spelling. A copy-paste collapse in
        // `outcome_label` would merge two arms onto one metric cell and make an
        // operator unable to tell a retention problem (`refused_trimmed`) from a
        // fossil-region problem (`refused_seq_gap`) — different remediations.
        let mut labels: Vec<&str> = all.iter().map(TailRefusal::outcome_label).collect();
        let before = labels.len();
        labels.sort_unstable();
        labels.dedup();
        assert_eq!(
            labels.len(),
            before,
            "two TailRefusal variants share one outcome label: {labels:?}"
        );
    }

    // ── Arm matrix: shared builders ─────────────────────────────────────────
    //
    // Every case below constructs its slice through these, so no test
    // hand-rolls a `LayerNotification` field-by-field and a shape change lands
    // in one place.

    /// S0 for the matrix. Deliberately `> 1` — the production precondition
    /// (module header) — so the read window is `8-0` onward and never reaches
    /// `CreateLayer`'s ompf-less `1-0` seed entry.
    const S0: u64 = 7;
    /// First sequence the window may contain: `{S0 + 1}-0`.
    const FIRST: u64 = S0 + 1;
    const OBJECT_ID: u64 = 4242;
    const SERVER_TS_BASE: i64 = 1_700_000_000_000;

    /// An `update`-typed entry at an arbitrary stream ID, with an arbitrary
    /// payload-declared sequence — the general builder the ID/payload
    /// disagreement and auto-ID cases need.
    fn update_entry(
        stream_id: &str,
        sequence_num: u64,
        relay_ompf: Option<Bytes>,
    ) -> TailStreamEntry {
        let payload_bytes = relay_ompf.as_ref().map_or(0, |b| b.len() as u64);
        TailStreamEntry {
            stream_id: stream_id.to_string(),
            payload: TailEntryPayload::Notification(LayerNotification::Update {
                object_id: OBJECT_ID,
                sequence_num,
                client_id: "matrix-client".to_string(),
                client_ts: 0,
                server_ts: SERVER_TS_BASE + sequence_num as i64,
                relay_ompf,
            }),
            payload_bytes,
        }
    }

    /// A well-formed non-empty commit at `{seq}-0` carrying verbatim bytes.
    fn verbatim_entry(seq: u64, ompf: &[u8]) -> TailStreamEntry {
        update_entry(&format!("{seq}-0"), seq, Some(Bytes::copy_from_slice(ompf)))
    }

    /// The §8.4 empty-commit shape: `update` at `{seq}-0`, no `ompf` field.
    fn empty_commit_entry(seq: u64) -> TailStreamEntry {
        update_entry(&format!("{seq}-0"), seq, None)
    }

    /// A layer-retirement sentinel. Always an auto ID — the writers XADD `*`.
    fn sentinel_entry(auto_id: &str) -> TailStreamEntry {
        TailStreamEntry {
            stream_id: auto_id.to_string(),
            payload: TailEntryPayload::Notification(LayerNotification::Deleted),
            payload_bytes: 0,
        }
    }

    /// An entry the read primitive preserved rather than skipped.
    fn undecodable_entry(id: &str) -> TailStreamEntry {
        TailStreamEntry {
            stream_id: id.to_string(),
            payload: TailEntryPayload::Undecodable {
                reason: "missing `payload` field".to_string(),
            },
            payload_bytes: 0,
        }
    }

    /// Distinct, self-identifying verbatim bytes per sequence, so a
    /// byte-identity assertion cannot pass by accidentally comparing the wrong
    /// record against the wrong entry.
    fn ompf_for(seq: u64) -> Vec<u8> {
        format!("ompf-payload-{seq}").into_bytes()
    }

    /// A slice the stream ran out of: no cap, no lookahead.
    fn slice(entries: Vec<TailStreamEntry>) -> TailSlice {
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
    fn capped_slice(
        entries: Vec<TailStreamEntry>,
        cap: TailSliceCap,
        lookahead: TailStreamEntry,
    ) -> TailSlice {
        TailSlice {
            cap_hit: Some(cap),
            lookahead: Some(lookahead),
            ..slice(entries)
        }
    }

    #[track_caller]
    fn expect_captured(classification: TailClassification) -> TailCaptured {
        match classification {
            TailClassification::Captured(captured) => {
                // The b.pw6 rule, asserted at the value level on EVERY capture
                // this matrix makes: claimed range == actual content. The
                // classifier's own `debug_assert!`s vanish in release builds;
                // this one does not.
                assert_eq!(
                    captured.record_count() as u64,
                    captured.to_seq - captured.from_seq + 1,
                    "claimed range {}..={} is not filled by {} records",
                    captured.from_seq,
                    captured.to_seq,
                    captured.record_count()
                );
                for (i, record) in captured.records().iter().enumerate() {
                    assert_eq!(
                        record.sequence_num(),
                        captured.from_seq + i as u64,
                        "record {i} breaks contiguity from from_seq={}",
                        captured.from_seq
                    );
                }
                assert_eq!(
                    captured.records().last().map(TailRecord::sequence_num),
                    Some(captured.to_seq),
                    "to_seq must come from the last RETAINED record"
                );
                assert!(
                    captured.record_count() >= 1,
                    "SR-1.3 forbids persisting an empty prefix"
                );
                captured
            }
            TailClassification::Refused(refusal) => {
                panic!("expected a capture, got Refused({refusal:?})")
            }
        }
    }

    #[track_caller]
    fn expect_refused(classification: TailClassification, expected: TailRefusal) {
        match classification {
            TailClassification::Refused(refusal) => assert_eq!(
                refusal, expected,
                "wrong refusal arm: the operator remediation differs per arm"
            ),
            TailClassification::Captured(captured) => panic!(
                "expected Refused({expected:?}), got a capture of {} records over {}..={}",
                captured.record_count(),
                captured.from_seq,
                captured.to_seq
            ),
        }
    }

    #[track_caller]
    fn assert_verbatim(record: &TailRecord, seq: u64, ompf: &[u8]) {
        match record {
            TailRecord::Verbatim {
                sequence_num,
                ompf: bytes,
                ..
            } => {
                assert_eq!(*sequence_num, seq, "verbatim record carries the wrong seq");
                assert_eq!(
                    bytes.as_ref(),
                    ompf,
                    "verbatim bytes are not byte-identical to the stream entry's `ompf` field \
                     (SR-2.2 stores the writer's raw inbound container with no re-encode)"
                );
            }
            other => panic!("expected a verbatim record at {seq}, got {other:?}"),
        }
    }

    #[track_caller]
    fn assert_empty_delta(record: &TailRecord, seq: u64) {
        match record {
            TailRecord::EmptyDelta { sequence_num, .. } => {
                assert_eq!(
                    *sequence_num, seq,
                    "empty-delta record carries the wrong seq"
                );
            }
            other => panic!("expected an empty-delta record at {seq}, got {other:?}"),
        }
    }

    // ── 1-4: the capture arms ───────────────────────────────────────────────

    /// Case 1 — the happy path: consecutive `{seq}-0` ompf-bearing entries from
    /// `{S0+1}-0` capture whole, in stream order, byte-identical.
    #[test]
    fn classify_tail_slice_consecutive_verbatim_entries_capture_the_whole_slice() {
        let seqs = [FIRST, FIRST + 1, FIRST + 2, FIRST + 3];
        let entries: Vec<_> = seqs
            .iter()
            .map(|s| verbatim_entry(*s, &ompf_for(*s)))
            .collect();

        let captured = expect_captured(classify_tail_slice(&slice(entries), S0));

        assert_eq!(captured.from_seq, FIRST);
        assert_eq!(captured.to_seq, *seqs.last().expect("non-empty"));
        assert_eq!(captured.record_count(), seqs.len());
        assert!(!captured.capped, "no cap fired, so this is `captured`");
        for (record, seq) in captured.records().iter().zip(seqs) {
            assert_verbatim(record, seq, &ompf_for(seq));
        }
    }

    /// Case 2 — a one-entry slice is the minimal non-empty prefix. The
    /// degenerate end of case 1, kept separate because an off-by-one in the
    /// walk would show up here and nowhere else.
    #[test]
    fn classify_tail_slice_single_entry_slice_captures_the_minimal_prefix() {
        let captured = expect_captured(classify_tail_slice(
            &slice(vec![verbatim_entry(FIRST, &ompf_for(FIRST))]),
            S0,
        ));

        assert_eq!(captured.record_count(), 1);
        assert_eq!(captured.from_seq, FIRST);
        assert_eq!(captured.to_seq, FIRST);
        assert_verbatim(&captured.records()[0], FIRST, &ompf_for(FIRST));
    }

    /// Case 3 — a mid-slice ompf-less `update` at `{seq}-0` extends the prefix
    /// as an empty-delta record. Not dropped (which would leave a hole under a
    /// contiguous-looking claim) and not verbatim (there are no bytes).
    #[test]
    fn classify_tail_slice_mid_slice_ompf_less_update_extends_prefix_as_empty_delta() {
        let entries = vec![
            verbatim_entry(FIRST, &ompf_for(FIRST)),
            empty_commit_entry(FIRST + 1),
            verbatim_entry(FIRST + 2, &ompf_for(FIRST + 2)),
        ];

        let captured = expect_captured(classify_tail_slice(&slice(entries), S0));

        assert_eq!(captured.record_count(), 3);
        assert_eq!(captured.to_seq, FIRST + 2);
        assert_verbatim(&captured.records()[0], FIRST, &ompf_for(FIRST));
        assert_empty_delta(&captured.records()[1], FIRST + 1);
        assert_verbatim(&captured.records()[2], FIRST + 2, &ompf_for(FIRST + 2));
    }

    /// Case 4 — position-0 coverage for the same shape. An early-return bug on
    /// the first entry would freeze a hot layer's recovery point permanently:
    /// every tick would re-read a window whose first entry is an empty commit
    /// and refuse, so the lineage would never advance again.
    #[test]
    fn classify_tail_slice_first_entry_ompf_less_update_extends_prefix_as_empty_delta() {
        let entries = vec![
            empty_commit_entry(FIRST),
            verbatim_entry(FIRST + 1, &ompf_for(FIRST + 1)),
        ];

        let captured = expect_captured(classify_tail_slice(&slice(entries), S0));

        assert_eq!(captured.record_count(), 2);
        assert_empty_delta(&captured.records()[0], FIRST);
        assert_verbatim(&captured.records()[1], FIRST + 1, &ompf_for(FIRST + 1));
    }

    // ── 5-7: trimmed, and the index-0 precedence ────────────────────────────

    /// Case 5 — the head was trimmed past the window start, so the first entry
    /// sits above `{S0+1}-0`. Nothing is captured; the operator levers are the
    /// stream-retention knobs, `LIVE_EDIT_LAYER_STREAM_MAXLEN` and
    /// `LIVE_EDIT_LAYER_STREAM_MAX_BYTES`.
    #[test]
    fn classify_tail_slice_first_entry_above_window_start_refuses_trimmed() {
        let entries = vec![
            verbatim_entry(FIRST + 1, &ompf_for(FIRST + 1)),
            verbatim_entry(FIRST + 2, &ompf_for(FIRST + 2)),
        ];

        expect_refused(
            classify_tail_slice(&slice(entries), S0),
            TailRefusal::Trimmed,
        );
    }

    /// Case 6 — a genuinely empty window (`entries` empty, **`cap_hit: None`**)
    /// folds to `Trimmed`. `cap_hit` is fixed to `None` explicitly here because
    /// that is exactly what separates this arm from the oversize-first-entry
    /// arm below; the two must not collapse.
    #[test]
    fn classify_tail_slice_empty_slice_without_a_cap_hit_refuses_trimmed() {
        let empty = TailSlice {
            cap_hit: None,
            lookahead: None,
            ..TailSlice::default()
        };

        expect_refused(classify_tail_slice(&empty, S0), TailRefusal::Trimmed);
    }

    /// Case 7 — index-0 precedence: a first entry that *jumps* the progression
    /// is `Trimmed`, never `SeqGap`.
    ///
    /// SR-3.6 depends on this. After a DR restore the stream is fresh and the
    /// first commit lands at `{seeded_seq+1}-0`, far above the restored base's
    /// `{S0+1}-0` — that must read as a retention/window problem, not as the
    /// fossil-region refusal, because there is no accumulated prefix to discard
    /// and `seq_gap` would point the operator at the wrong diagnosis.
    #[test]
    fn classify_tail_slice_first_entry_far_above_window_start_refuses_trimmed_not_seq_gap() {
        let jumped = FIRST + 500;
        let entries = vec![
            verbatim_entry(jumped, &ompf_for(jumped)),
            verbatim_entry(jumped + 1, &ompf_for(jumped + 1)),
        ];

        let classification = classify_tail_slice(&slice(entries), S0);

        expect_refused(classification.clone(), TailRefusal::Trimmed);
        assert_ne!(
            classification,
            TailClassification::Refused(TailRefusal::SeqGap),
            "index-0 precedence: the first entry's ID never routes to SeqGap"
        );
    }

    // ── 8-10: the sentinel arm, and kind-before-ID ──────────────────────────

    /// Case 8 — a `deleted` sentinel as the first entry refuses `Sentinel` with
    /// an empty prefix. Kind is judged before ID, so the sentinel's auto ID does
    /// not get swept into `Trimmed` by the index-0 rule (which is scoped to
    /// `update`-typed entries).
    #[test]
    fn classify_tail_slice_first_entry_sentinel_refuses_sentinel() {
        let entries = vec![sentinel_entry("1700000000000-0")];

        expect_refused(
            classify_tail_slice(&slice(entries), S0),
            TailRefusal::Sentinel,
        );
    }

    /// Case 9 — SR-1.3's capture-then-stop: a sentinel AFTER a non-empty prefix
    /// captures that prefix and stops. A shorter sound tail still advances the
    /// recovery point, and its claimed range still equals its content.
    #[test]
    fn classify_tail_slice_sentinel_after_non_empty_prefix_captures_the_prefix() {
        let entries = vec![
            verbatim_entry(FIRST, &ompf_for(FIRST)),
            verbatim_entry(FIRST + 1, &ompf_for(FIRST + 1)),
            sentinel_entry("1700000000000-0"),
        ];

        let captured = expect_captured(classify_tail_slice(&slice(entries), S0));

        assert_eq!(captured.record_count(), 2, "the sentinel is not a record");
        assert_eq!(captured.to_seq, FIRST + 1);
        assert!(
            !captured.capped,
            "the walk stopped early, so nothing was truncated by a cap"
        );
    }

    /// Case 10 — kind-before-ID, pinned at its sharpest. This sentinel's auto ID
    /// is numerically enormous relative to `expected`, so an implementation that
    /// judged the ID first would call it `Trimmed` (index-0 rule) or `SeqGap`
    /// (ascending jump) or `MissingBytes` (non-`-0` sub-counter). All three are
    /// wrong: the entry announces that the layer was retired, which is the
    /// operator-visible fact.
    #[test]
    fn classify_tail_slice_sentinel_at_auto_id_far_above_expected_refuses_sentinel() {
        let entries = vec![sentinel_entry("1735689600000-7")];

        let classification = classify_tail_slice(&slice(entries), S0);

        expect_refused(classification.clone(), TailRefusal::Sentinel);
        for wrong in [
            TailRefusal::Trimmed,
            TailRefusal::SeqGap,
            TailRefusal::MissingBytes,
        ] {
            assert_ne!(
                classification,
                TailClassification::Refused(wrong),
                "entry KIND must be judged before entry ID"
            );
        }
    }

    // ── 11-12: the missing-bytes arms ───────────────────────────────────────

    /// Case 11a — an entry that is neither a usable commit nor an empty commit,
    /// at index 0, refuses `MissingBytes` with an empty prefix. The read
    /// primitive preserved it precisely so the prefix terminates here instead of
    /// stepping over a hole.
    #[test]
    fn classify_tail_slice_first_entry_undecodable_refuses_missing_bytes() {
        let entries = vec![undecodable_entry(&format!("{FIRST}-0"))];

        expect_refused(
            classify_tail_slice(&slice(entries), S0),
            TailRefusal::MissingBytes,
        );
    }

    /// Case 11b — the landed defense arm: a present-but-zero-length `ompf` field
    /// is `MissingBytes`, not an empty-delta record. No producer writes that
    /// shape, and `tail_codec` rejects a zero-payload `Verbatim` on encode, so
    /// admitting it would build an un-encodable record.
    #[test]
    fn classify_tail_slice_zero_length_ompf_update_refuses_missing_bytes() {
        let entries = vec![update_entry(
            &format!("{FIRST}-0"),
            FIRST,
            Some(Bytes::new()),
        )];

        expect_refused(
            classify_tail_slice(&slice(entries), S0),
            TailRefusal::MissingBytes,
        );
    }

    /// Case 12 — the same terminal after a non-empty prefix captures the prefix
    /// and stops (SR-1.3), rather than discarding it.
    #[test]
    fn classify_tail_slice_undecodable_after_non_empty_prefix_captures_the_prefix() {
        let entries = vec![
            verbatim_entry(FIRST, &ompf_for(FIRST)),
            verbatim_entry(FIRST + 1, &ompf_for(FIRST + 1)),
            undecodable_entry(&format!("{}-0", FIRST + 2)),
            verbatim_entry(FIRST + 3, &ompf_for(FIRST + 3)),
        ];

        let captured = expect_captured(classify_tail_slice(&slice(entries), S0));

        assert_eq!(
            captured.record_count(),
            2,
            "the walk stops AT the undecodable entry — the entry beyond it must \
             not be pulled in over the hole"
        );
        assert_eq!(captured.to_seq, FIRST + 1);
    }

    // ── 13: the contrast test ───────────────────────────────────────────────

    /// Case 13 — **the asymmetry, as a pair.** Two slices with a byte-identical
    /// three-entry replayable prefix, differing only in what terminates them:
    ///
    /// - a sentinel captures all three records (SR-1.3 capture-then-stop);
    /// - an ascending `{seq}-0` jump refuses everything, zero records.
    ///
    /// Both halves live in one test on purpose. Split across two tests the
    /// asymmetry can silently collapse — someone "harmonises" the terminal
    /// handling, one test is updated to match, and the pair no longer says
    /// anything. Entries below an ascending jump are fossils of an abandoned
    /// pre-restore timeline (SR-1.2 / SR-3.6): composing them mints a generation
    /// whose restore resurrects state the live timeline already superseded.
    #[test]
    fn classify_tail_slice_identical_prefix_captures_on_sentinel_but_refuses_everything_on_seq_gap()
    {
        let prefix = || {
            vec![
                verbatim_entry(FIRST, &ompf_for(FIRST)),
                verbatim_entry(FIRST + 1, &ompf_for(FIRST + 1)),
                verbatim_entry(FIRST + 2, &ompf_for(FIRST + 2)),
            ]
        };

        let mut with_sentinel = prefix();
        with_sentinel.push(sentinel_entry("1700000000000-0"));
        let captured = expect_captured(classify_tail_slice(&slice(with_sentinel), S0));
        assert_eq!(
            captured.record_count(),
            3,
            "sentinel keeps the sound prefix"
        );
        assert_eq!(captured.to_seq, FIRST + 2);

        let mut with_jump = prefix();
        with_jump.push(verbatim_entry(FIRST + 9, &ompf_for(FIRST + 9)));
        let classification = classify_tail_slice(&slice(with_jump), S0);
        expect_refused(classification.clone(), TailRefusal::SeqGap);
        assert!(
            !matches!(classification, TailClassification::Captured(_)),
            "SeqGap is the one refusal that discards a non-empty accumulated prefix"
        );
    }

    /// Complement to case 13, from the landed classifier's scoping note: a
    /// **duplicate / regressing** ID after a non-empty prefix is NOT an
    /// ascending jump, so it takes the ordinary capture-then-stop path. Only an
    /// *ascending* jump implies a fossil region worth discarding evidence over.
    #[test]
    fn classify_tail_slice_duplicate_id_after_non_empty_prefix_captures_the_prefix() {
        let entries = vec![
            verbatim_entry(FIRST, &ompf_for(FIRST)),
            verbatim_entry(FIRST + 1, &ompf_for(FIRST + 1)),
            // Same ID again: the stream cannot supply the commit the prefix
            // needs next, but nothing above the prefix is implicated.
            verbatim_entry(FIRST + 1, &ompf_for(FIRST + 1)),
        ];

        let captured = expect_captured(classify_tail_slice(&slice(entries), S0));

        assert_eq!(captured.record_count(), 2);
        assert_eq!(captured.to_seq, FIRST + 1);
    }

    // ── 14-17: the cap arms ─────────────────────────────────────────────────

    /// Case 14 — the page/cap-boundary fossil case. Every entry was replayable
    /// and a cap stopped the read, but the lookahead shows an ascending jump
    /// immediately beyond the boundary. That is the same evidence a mid-slice
    /// jump carries, so it gets the same verdict: refuse everything, discarding
    /// the accumulated prefix. Without the lookahead this slice would look like
    /// a clean `captured_capped`.
    #[test]
    fn classify_tail_slice_cap_lookahead_ascending_jump_refuses_seq_gap_discarding_the_prefix() {
        let entries = vec![
            verbatim_entry(FIRST, &ompf_for(FIRST)),
            verbatim_entry(FIRST + 1, &ompf_for(FIRST + 1)),
        ];
        let jumped = FIRST + 40;
        let slice = capped_slice(
            entries,
            TailSliceCap::Entries,
            verbatim_entry(jumped, &ompf_for(jumped)),
        );

        expect_refused(classify_tail_slice(&slice, S0), TailRefusal::SeqGap);
    }

    /// Case 15 — entry-cap shape. Truncated with a benign lookahead (the very
    /// next commit) yields `captured_capped`: exactly the admitted records, and
    /// `to_seq` from the last RETAINED one — never from the lookahead, which is
    /// not part of the content.
    #[test]
    fn classify_tail_slice_entries_cap_with_benign_lookahead_captures_capped() {
        let entries = vec![
            verbatim_entry(FIRST, &ompf_for(FIRST)),
            verbatim_entry(FIRST + 1, &ompf_for(FIRST + 1)),
            verbatim_entry(FIRST + 2, &ompf_for(FIRST + 2)),
        ];
        let next = FIRST + 3;
        let slice = capped_slice(
            entries,
            TailSliceCap::Entries,
            verbatim_entry(next, &ompf_for(next)),
        );

        let captured = expect_captured(classify_tail_slice(&slice, S0));

        assert!(
            captured.capped,
            "a cap truncated the read with more pending"
        );
        assert_eq!(captured.record_count(), 3);
        assert_eq!(
            captured.to_seq,
            FIRST + 2,
            "to_seq is the last retained record, not the lookahead's sequence"
        );
    }

    /// Case 16 — exactly-at-cap with `cap_hit: None` is `captured`, NOT
    /// `captured_capped`. Per the read primitive's contract, `cap_hit` means
    /// "genuinely truncated, **with more pending**"; a read that reached a cap
    /// exactly as the stream ended truncated nothing.
    #[test]
    fn classify_tail_slice_all_entries_consumed_without_a_cap_hit_captures_uncapped() {
        let entries = vec![
            verbatim_entry(FIRST, &ompf_for(FIRST)),
            verbatim_entry(FIRST + 1, &ompf_for(FIRST + 1)),
        ];

        let captured = expect_captured(classify_tail_slice(&slice(entries), S0));

        assert!(
            !captured.capped,
            "cap_hit == None means nothing was pending, so the outcome is \
             `captured`, not `captured_capped`"
        );
        assert_eq!(captured.record_count(), 2);
        assert_eq!(captured.to_seq, FIRST + 1);
    }

    /// Case 17 — byte-cap shape. Same disposition as the entry cap: the claimed
    /// range equals the content, and the excluded entry contributes nothing.
    #[test]
    fn classify_tail_slice_byte_cap_with_benign_lookahead_captures_capped() {
        let entries = vec![
            verbatim_entry(FIRST, &ompf_for(FIRST)),
            verbatim_entry(FIRST + 1, &ompf_for(FIRST + 1)),
        ];
        let next = FIRST + 2;
        let slice = capped_slice(
            entries,
            TailSliceCap::Bytes,
            verbatim_entry(next, &ompf_for(next)),
        );

        let captured = expect_captured(classify_tail_slice(&slice, S0));

        assert!(captured.capped);
        assert_eq!(captured.record_count(), 2);
        assert_eq!(captured.from_seq, FIRST);
        assert_eq!(captured.to_seq, FIRST + 1);
    }

    /// Cap-arm complement: a lookahead that is not an `update` at an ascending
    /// `{a}-0` is not a jump, so it does not trip the boundary refusal. Both
    /// non-`update` kinds are covered — a retirement sentinel and a preserved
    /// undecodable entry — because `is_ascending_jump` must reject on KIND
    /// before it ever parses the ID, exactly as the mid-slice walk does.
    #[test]
    fn classify_tail_slice_cap_lookahead_that_is_not_an_update_captures_capped() {
        let entries = || vec![verbatim_entry(FIRST, &ompf_for(FIRST))];

        for lookahead in [
            sentinel_entry("1735689600000-3"),
            undecodable_entry("1735689600000-4"),
        ] {
            let slice = capped_slice(entries(), TailSliceCap::Entries, lookahead);
            let captured = expect_captured(classify_tail_slice(&slice, S0));
            assert!(captured.capped);
            assert_eq!(captured.record_count(), 1);
            assert_eq!(captured.to_seq, FIRST);
        }
    }

    /// Cap-arm complement: a lookahead sitting exactly at `expected` is the
    /// ordinary "more commits pending" case, not a jump. The comparison must be
    /// strictly greater-than — an off-by-one to `>=` here would refuse every
    /// capped read in the fleet.
    #[test]
    fn classify_tail_slice_cap_lookahead_at_the_expected_sequence_captures_capped() {
        let entries = vec![
            verbatim_entry(FIRST, &ompf_for(FIRST)),
            verbatim_entry(FIRST + 1, &ompf_for(FIRST + 1)),
        ];
        let next = FIRST + 2;
        let slice = capped_slice(
            entries,
            TailSliceCap::Entries,
            verbatim_entry(next, &ompf_for(next)),
        );

        let captured = expect_captured(classify_tail_slice(&slice, S0));

        assert!(captured.capped);
        assert_eq!(captured.to_seq, FIRST + 1);
    }

    // ── PM amendment cases ──────────────────────────────────────────────────

    /// PM amendment 1 — the oversize-first-entry interaction (SR-8.3's residual
    /// class). `entries` empty **with** `cap_hit == Some(Bytes)` and a lookahead
    /// present is NOT the empty window of case 6: the byte cap excluded the very
    /// first entry, so the honest statement is "no admissible first entry".
    ///
    /// Folding it to `Trimmed` is forbidden because the two arms name **different
    /// knobs**: `trimmed`'s documented remediation is the stream-retention
    /// knobs (`LIVE_EDIT_LAYER_STREAM_MAXLEN`, `LIVE_EDIT_LAYER_STREAM_MAX_BYTES`),
    /// which do nothing for
    /// a commit that is simply larger than
    /// `LIVE_EDIT_CHECKPOINT_TAIL_MAX_BYTES` (the per-tail artifact budget). An
    /// operator sent to the retention knob would raise it, observe no change, and
    /// still be unable to checkpoint the layer.
    #[test]
    fn classify_tail_slice_empty_entries_with_byte_cap_hit_refuses_missing_bytes() {
        let oversize = TailSlice {
            entries: Vec::new(),
            // The read primitive anchors its probe on the window start when
            // nothing was admitted, so the lookahead IS the excluded entry.
            lookahead: Some(verbatim_entry(FIRST, &ompf_for(FIRST))),
            cap_hit: Some(TailSliceCap::Bytes),
            payload_bytes: 0,
            pages_read: 1,
        };

        let classification = classify_tail_slice(&oversize, S0);

        expect_refused(classification.clone(), TailRefusal::MissingBytes);
        assert_ne!(
            classification,
            TailClassification::Refused(TailRefusal::Trimmed),
            "the oversize-first-entry arm must not fold into the retention arm"
        );
    }

    /// PM amendment 4 — defense in depth against a self-contradictory entry: the
    /// stream ID says `{FIRST}-0` but the payload's own `sequence_num` says
    /// `FIRST + 5`. One writer emits both halves in one `XADD`, so a disagreement
    /// means corruption or a foreign writer. The ID stays authoritative and the
    /// entry is refused rather than replayed — a release-safe arm, unlike the
    /// contiguity `debug_assert!`s which vanish in release builds.
    #[test]
    fn classify_tail_slice_stream_id_disagreeing_with_payload_sequence_refuses_missing_bytes() {
        let entries = vec![update_entry(
            &format!("{FIRST}-0"),
            FIRST + 5,
            Some(Bytes::from_static(b"contradictory")),
        )];

        expect_refused(
            classify_tail_slice(&slice(entries), S0),
            TailRefusal::MissingBytes,
        );
    }

    /// PM amendment 5 — the complement of case 10: an `update`-typed entry at an
    /// **auto** ID. Commit entries always carry an explicit `{seq}-0`, so an auto
    /// ID on an update is not evidence this classifier can use — `MissingBytes`,
    /// not a record and not a sentinel.
    #[test]
    fn classify_tail_slice_update_at_an_auto_id_refuses_missing_bytes() {
        let entries = vec![update_entry(
            "1700000000000-3",
            FIRST,
            Some(Bytes::from_static(b"auto-id-update")),
        )];

        expect_refused(
            classify_tail_slice(&slice(entries), S0),
            TailRefusal::MissingBytes,
        );
    }

    /// Sibling of the case above: an explicit ID with a **non-zero sub-counter**
    /// (`{seq}-1`) is likewise not a commit ID. Pinned separately because the
    /// parse rejects it on a different branch than the auto-ID case.
    #[test]
    fn classify_tail_slice_explicit_id_with_non_zero_sub_counter_refuses_missing_bytes() {
        let entries = vec![update_entry(
            &format!("{FIRST}-1"),
            FIRST,
            Some(Bytes::from_static(b"sub-counter")),
        )];

        expect_refused(
            classify_tail_slice(&slice(entries), S0),
            TailRefusal::MissingBytes,
        );
    }

    // ── 19-20: the debug-assert pins ────────────────────────────────────────

    /// Case 19 — the `S0 >= 1` precondition is ENFORCED, not merely documented.
    ///
    /// With `base_snap_seq == 0` the read window would start at `1-0`, which is
    /// `CreateLayer`'s seed entry: `update`-typed, carrying **no `ompf` field**,
    /// and yet accompanying the full initial layer content. This classifier
    /// cannot tell it apart from a §8.4 empty commit — it would decode as an
    /// empty-delta record and replay as a no-op over real content, a b.pw6-class
    /// silent-corruption defect. So the classifier refuses to run at all in test
    /// builds; production's guard is the capture unit's `s0 == 0` gate, which
    /// refuses `refused_lineage_unrootable` before this function is ever
    /// reached (SR-1.5 / SR-7.1 root every lineage at `snap_seq = 1`).
    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "classify_tail_slice requires S0 >= 1")]
    fn classify_tail_slice_zero_base_snap_seq_panics_in_debug() {
        let entries = vec![empty_commit_entry(1)];
        let _ = classify_tail_slice(&slice(entries), 0);
    }

    /// Case 20a — SR-1.3's "never persist an empty prefix" is enforced by the
    /// single construction path, not just by its callers happening to guard.
    ///
    /// [`TailCaptured::records`] is private, so a zero-record capture is
    /// unconstructible from outside this module; `captured()` is the only way in
    /// and it refuses. This test reaches it directly (same-module privacy) to
    /// prove the guard is armed — `tail_codec` would happily encode a
    /// zero-record envelope, so this type is where it has to be caught.
    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "captured() called with zero records")]
    fn captured_with_zero_records_panics_in_debug() {
        let _ = captured(Vec::new(), FIRST, false);
    }

    /// Case 20b — the b.pw6 contiguity guard is armed: a record vector with a
    /// hole cannot be dressed up as a contiguous claimed range. The `Captured`
    /// path cannot produce this from a real slice (the walk only ever pushes at
    /// `expected`), so the assertion is reached directly.
    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "not exactly contiguous")]
    fn captured_with_non_contiguous_records_panics_in_debug() {
        let records = vec![
            TailRecord::EmptyDelta {
                sequence_num: FIRST,
                server_ts: SERVER_TS_BASE,
            },
            TailRecord::EmptyDelta {
                sequence_num: FIRST + 2,
                server_ts: SERVER_TS_BASE,
            },
        ];
        let _ = captured(records, FIRST, false);
    }

    /// The empty-entries arm's documented-unreachable case: an entries-cap stop
    /// that admitted nothing requires `max_entries == 0`, which
    /// `LIVE_EDIT_CHECKPOINT_TAIL_MAX_ENTRIES` zero-rejects at config. Pinned so
    /// the claim is checked rather than merely asserted in prose.
    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "requires max_entries == 0")]
    fn classify_tail_slice_empty_entries_with_entries_cap_panics_in_debug() {
        let impossible = TailSlice {
            entries: Vec::new(),
            lookahead: Some(verbatim_entry(FIRST, &ompf_for(FIRST))),
            cap_hit: Some(TailSliceCap::Entries),
            payload_bytes: 0,
            pages_read: 1,
        };

        let _ = classify_tail_slice(&impossible, S0);
    }
}
