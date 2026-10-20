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

//! Base bootstrap: the creation-time first base (SR-7.1 / SR-7.2) and the
//! post-restore gap re-base (SR-3.6).
//!
//! ## What this unit is for
//!
//! A layer acquires its first base from the periodic assembler only at its
//! first eligible tick — whose read proves a cut however hot the layer is, the
//! atomic read at or below the atomic-read limits and the fenced read above
//! them (Spec SDD SR-4.1, SR-11.5) — and not at all on a tick a header-sequence
//! regression skips. Without a base there is no lineage root, and tail capture
//! refuses `no_base`. This unit gives every layer its lineage root at
//! creation, independent of the periodic loop.
//!
//! `CreateLayer` is the one moment a layer's full state is in hand with no
//! stream read and no quiescence proof required — the handler holds the
//! fully-decoded [`ObjectState`] at commit time, and it is by construction the
//! committed creation state at `sequence_num == 1`. [`BaseBootstrap`]
//! persists exactly that state as a [`CheckpointKind::Base`] generation at
//! `snap_seq = 1` and stamps the header cursor columns, giving every layer a
//! lineage root regardless of how hot it is afterwards.
//!
//! A **DR restore** is the second such moment, and the reason this unit is no
//! longer creation-only. `apply_restore` materializes the composed state and
//! seeds the writer cursor at `seeded_seq = restore_floor + band` while stamping
//! `last_checkpoint_seq = snap_seq` (the recovery audit trail), and `band >= 1`
//! unconditionally — so **every** restore, base-only and composed alike, leaves
//! the live header a whole band above the newest generation. Until a base lands
//! at or above it — this unit's, or the periodic loop's at the layer's next
//! eligible tick — `tail_capture` refuses on every tick. The restore hands its
//! already-in-memory state straight back here, and this unit persists it as a
//! fresh base at `seeded_seq` at once.
//!
//! ## Why plain spawn semantics are correct here — and b.443's machinery is
//! deliberately absent
//!
//! This is the single most important thing to understand before adding
//! anything to this module. The sibling [`SessionCheckpointWriter`]
//! (`session_writer.rs`) carries three pieces of ordering machinery — a
//! per-session serialized dispatcher, a coalescing slot map, and a Valkey-side
//! publish watermark. None of them are needed here, and adding them would be
//! pure cost:
//!
//! - The session writer's PUT is keyed **per session** and its payload
//!   **changes with every lifecycle event** (a session's layer-mapping set
//!   grows and shrinks). Two events therefore race to one fixed path with
//!   *different* content, and the loser durably clobbers the winner — which is
//!   exactly what the dispatcher, the coalescing, and the watermark exist to
//!   arbitrate.
//! - This unit's PUT is keyed `(object_id, target.snap_seq(),
//!   CheckpointKind::Base)` and its payload is **idempotent-identical**. At the
//!   creation site `object_id` is freshly allocated by the `create_layer`
//!   commit that triggered the spawn, `snap_seq` is the literal `1`, and the
//!   bytes are a reset OMPF built from the creation state that commit
//!   persisted. At the restore site `snap_seq` is the `seeded_seq` that one
//!   `write_restored_object` MULTI/EXEC committed, and the bytes are a reset OMPF of the
//!   state it committed. At the restore site exactly one spawn happens per
//!   commit, on the replica that served it. At the creation site a commit can
//!   be spawned for twice — by the call that committed it and again by a
//!   client re-send carrying the same `operation_id`, possibly on another
//!   replica (see [`BaseBootstrap::spawn_creation_base`]) — but the re-send's
//!   state is rebuilt from byte-identical request bytes, so it is the same
//!   `ObjectState` and builds the same blob. Either way no second writer can
//!   ever target that key with different content. There is nothing for a
//!   serializer, a coalescer, or a watermark to protect.
//!
//! ## One attempt, no retry — and one bounded admission
//!
//! The unit runs **entirely** in the spawned task — the OMPF build included, so
//! the commit path pays nothing for it — and makes exactly ONE
//! `put_layer_checkpoint` attempt. There is no retry loop, and the unit never
//! re-spawns itself. (One caller can spawn a second run for an object: a
//! `CreateLayer` re-send answered `ALREADY_COMMITTED` re-spawns the recorded
//! object's creation base, because the original call may never have reached
//! its own spawn — see [`BaseBootstrap::spawn_creation_base`]. That second run
//! does not know how the first ended, makes its own single attempt, and is a
//! no-op when the first landed.) On any failure the layer simply has no base
//! at that target: tail capture refuses observably (SR-7.2 — the
//! defined-refusal arm of PRD AC-5) until
//! either the first eligible periodic tick checkpoints it normally or a later
//! generation lands. Which refusal depends on the site: `refused_no_base` at
//! the creation site (no lineage root exists at all), `refused_seq_gap` at the
//! restore site (the restore's `restore_seq_bump` is simply never cleared, so
//! the `PostRestoreUnrebased` guard folds there on BOTH stream shapes —
//! `Trimmed` is unreachable while the bump stands). A silent absence is never a
//! wedge, and a failure is always visible on
//! `checkpoint_writes_total{result="error",kind="layer"}`.
//!
//! Admission is bounded at [`BOOTSTRAP_MAX_INFLIGHT`] bodies per unit (and
//! production constructs exactly one unit, shared by both call sites), enforced
//! at the top of the body [`BaseBootstrap::run_once`] wraps. It **sheds** rather
//! than queues: a queued task would still be holding its `ObjectState`, which is
//! the retention the bound exists to cap, so queueing would convert a memory
//! bound into a latency bound and close nothing.
//!
//! A shed is **not** the same observable posture as a failed bootstrap, and the
//! two must not be read as one. They agree on the durable state — nothing
//! stamped, no blob written, so capture refuses observably either way
//! (`refused_no_base` at the creation site, `refused_seq_gap` at the restore
//! site, where the bump is simply never cleared) — but they are distinguishable
//! on the wire: a shed ticks
//! `live_edit_checkpoint_base_bootstrap_total{outcome="shed", site=…}` and
//! nothing else, while a failed bootstrap ticks `{outcome="failed", site=…}` on
//! that same family AND
//! `checkpoint_writes_total{result="error",kind="layer"}` (every failure arm but
//! the gate-unreachable `NotImplemented` one — see [`BootstrapOutcome::Failed`]).
//! Admission saturation
//! and adapter failure therefore separate at a glance, which matters because
//! their remediations differ — see [`BOOTSTRAP_MAX_INFLIGHT`] for the shed's.
//!
//! ## The SR-7 observability adjudication, as landed
//!
//! Epic-3 interim hazard #4 (t1.dfh.57) is discharged here. Hazards #4 and #5
//! were handed on to Epic 5's SR-7 pass (t1.dfh.be), which landed the following
//! and closes both:
//!
//! - **Every** [`BaseBootstrap::run_once`] invocation ticks exactly one
//!   `live_edit_checkpoint_base_bootstrap_total{outcome,site}` cell — the shed
//!   arm, the guard-(a) and guard-(b) skips, the untrusted-cursor skip and the
//!   `NotImplemented` arm included. No arm is silent any more. The tick happens
//!   at ONE site (the `run_once` wrapper around `run_once_inner`), which is what
//!   makes "one increment per invocation" a structural property rather than a
//!   per-`return` discipline.
//! - `checkpoint_writes_total` is deliberately **unchanged** by that pass. It
//!   counts generation-completion outcomes fleet-wide — `checkpointer.rs` ticks
//!   `{error,layer}` on the identical blob-landed / HSET-failed shape and
//!   `{success,layer}` only after the stamp — so a guard skip that declines to
//!   complete a generation must not read as a success there. The two families
//!   are additive, never substitutes.
//! - Hazard #5 — the guard-(b) under-count, i.e. skip arms that were
//!   indistinguishable from one another — is discharged by the vocabulary
//!   itself. [`BootstrapOutcome`] is now 1:1 with
//!   `CHECKPOINT_BASE_BOOTSTRAP_OUTCOMES`, so the pre-PUT skip
//!   (`skipped_generation_present`), the post-PUT seq skip
//!   (`unstamped_generation_present`) and the post-PUT foreign-restore skip
//!   (`unstamped_foreign_restore`) each own a cell.
//!
//! b.vn6 added a tenth member on the same discipline —
//! `unstamped_header_absent`, the post-PUT half of the deleted-header race —
//! and it is the cell that incident needed: `skipped_header_absent` (the
//! pre-PUT half) read a flat zero throughout, because the header was still
//! alive when this unit looked at it and died while the PUT was in flight.
//!
//! This unit produces no `checkpoint_tail_total` cell of any kind: SR-4.1's
//! chain-collapse re-base (`rebase.rs`) is the sole producer of `{rebased}`, and
//! the SR-3.6 post-restore re-base written here stays `writes_total`-only on
//! that axis.
//!
//! ## The two call sites, and everything that differs between them
//!
//! Epic 3 recorded a forward contract here for the SR-3.6 reuse; this section
//! replaces it with what actually landed. Every call-site difference is carried
//! by [`BootstrapTarget`] and nothing else — the body below has exactly one
//! `match` per difference, and no site-specific branch anywhere else:
//!
//! | | [`BootstrapTarget::CreationBase`] | [`BootstrapTarget::RestoreRebase`] |
//! |---|---|---|
//! | spawned by | `CreateLayer`: post-commit, or on an `ALREADY_COMMITTED` re-send (see [`BaseBootstrap::spawn_creation_base`]) | [`RecoveryHook::try_restore_layer`](crate::checkpoint::RecoveryHook::try_restore_layer), applied arm only |
//! | `snap_seq` | `1` (see the variant's doc) | the restore's `seeded_seq` |
//! | cursor stamp | 2 pairs; atomic floor at `1` | 3 pairs — adds `restore_seq_bump = 0`; atomic floor at `seeded_seq` |
//! | meta `observed_commit_span` | `NO_COMMIT_SPAN_MEASUREMENT` | the restored generation's own measurement |
//! | `cursor_columns_trusted` | ignored | a skip arm |
//! | guard (b) reads | `last_checkpoint_seq` | `last_checkpoint_seq` **and** `restore_seq_bump` |
//!
//! Two of those rows are load-bearing rather than cosmetic, and are argued in
//! full at their sites: retiring `restore_seq_bump` is what actually unblocks
//! tail capture after a restore (`tail_capture.rs`'s `PostRestoreUnrebased`
//! refusal keys on it *before* the stream read), and carrying the measured
//! commit span forward is what keeps the next restore of this layer from
//! falling back to the configured margin.
//!
//! Both cursor-regression guards are stated **relative to the target**
//! (`observed >= target.snap_seq()` → skip) rather than against zero. At
//! `CreationBase` that is bit-for-bit the pre-existing `!= 0` test, because the
//! target there is `1`; see `cursor_regression_skips` and the unit test that
//! pins the equivalence.
//!
//! [`SessionCheckpointWriter`]: crate::checkpoint::SessionCheckpointWriter

use std::sync::Arc;

use bytes::Bytes;
use fred::interfaces::HashesInterface;
use tokio::sync::{Semaphore, TryAcquireError};
use tokio_util::task::TaskTracker;

use super::assembler::{
    build_reset_ompf_from_state, ext_refs_sidecar_json, SnapshotLoader, ValkeySnapshotLoader,
    NO_COMMIT_SPAN_MEASUREMENT,
};
use super::store::{CheckpointError, CheckpointKind, CheckpointMeta, CheckpointStore};
use crate::checkpoint::now_unix_ms;
use crate::domain::types::ObjectState;
use crate::observability::metrics::{
    SharedMetrics, CHECKPOINT_BASE_BOOTSTRAP_OUTCOMES, CHECKPOINT_BASE_BOOTSTRAP_SITES,
};
use crate::storage::client::{FredErrOrigin, ValkeyClient, POOL_LABEL_DATA};
use crate::storage::keys::KeySchema;

/// How many bootstrap bodies may be in flight on one unit at a time
/// (Epic-3 interim hazard #4).
///
/// # Why a bound exists at all
///
/// The unit deliberately carries none of `session_writer.rs`'s ordering
/// machinery (see the module header), and that machinery was also the only
/// backpressure. Both spawners hand the task a full `ObjectState` and the task
/// then builds an OMPF from it, so N concurrent bootstraps hold N × (state +
/// OMPF) resident until each PUT resolves — and a degraded adapter's PUT can sit
/// for the 180 s per-call ceiling. SR-7.1's cost posture bounds write
/// amplification *per creation*, not concurrent in-memory retention; this const
/// closes that gap. The restore-time call site makes it worse in kind rather
/// than only in degree: a composed restore's state is already resident and the
/// re-base extends its lifetime across the PUT.
///
/// # The derivation — a sizing assumption against the shipped pod limit
///
/// Per in-flight unit the peak is `state` + the built OMPF: ≈ 2 × the layer's
/// state during the build, ≈ 1 × across the PUT await (the long part). The
/// bound was sized assuming a layer's state is about the size of one upload,
/// so at most `LIVE_EDIT_MAX_LAYER_SIZE` (default 100 MiB), and when the chart
/// shipped `resources.limits.memory: 1Gi`; it now ships `5Gi`
/// (`helm/live-edit-service/values.yaml`), so the figures below are the
/// conservative case of that assumption:
///
/// ```text
/// 2 × BOOTSTRAP_MAX_INFLIGHT × LIVE_EDIT_MAX_LAYER_SIZE ≤ pod memory headroom
/// 2 × 2 × 100 MB = 400 MB transient (≈ 200 MB steady across the PUT stall)
/// ```
///
/// against a 1 GiB limit that also has to hold the service's baseline, the
/// Valkey pools, and whatever else is resident. A bound of 4 — the naive first
/// choice — yields ~800 MB transient under the same assumption, i.e. an
/// OOMKill scenario, which is strictly worse than the shed it would have
/// prevented.
///
/// This is an assumption, not a guarantee. `LIVE_EDIT_MAX_LAYER_SIZE` caps one
/// CreateLayer or UpdateLayer upload, not a layer's state: a layer grown
/// through UpdateLayer has no size cap, so a bootstrap of such a layer can hold
/// more than the figures above, and nothing here bounds that.
///
/// # Operator-facing consequence
///
/// The bound is fixed. Neither the pod memory limit nor
/// `LIVE_EDIT_MAX_LAYER_SIZE` changes it, so no setting buys more concurrent
/// bootstrap coverage; a shed heals at the layer's next eligible periodic tick
/// (below).
///
/// # What an operator does about a shed
///
/// Usually nothing: a shed heals itself at either site. The layer's next
/// eligible periodic tick writes a verified base — its read proves a cut however
/// hot the layer is (the atomic read at or below the atomic-read limits, the
/// fenced read above them; Spec SDD SR-4.1, SR-11.5) — and, at the **restore**
/// site, that tick's cursor stamp also clears `restore_seq_bump`, which unblocks
/// tail capture. Until that tick a restore-site shed leaves the layer's newest
/// durable copy at the generation it was restored from, and a header-sequence
/// regression on the tick defers the heal by one tick per regression.
///
/// Restore-site sheds concentrate (a mass restore against a degraded adapter is
/// the shed's whole concentration case), and the population is identifiable
/// from `live_edit_checkpoint_base_bootstrap_total{outcome="shed",
/// site="restore_rebase"}` plus the shed WARN's `object_id`. When waiting for
/// the next eligible tick is too long, a second restore of those layers re-runs
/// the re-base against an unsaturated bound.
///
/// # Why this is a const and not an env knob
///
/// SR-8.1 pins this feature's operator surface at the four
/// `LIVE_EDIT_CHECKPOINT_TAIL_*` knobs, and the remediation above is not this
/// number: it is an operational action. A fifth knob would require an SDD §16
/// amendment plus the SDD §16 / `docs/service/docs/configuration.md` sync for a value
/// whose safe range is set by the memory arithmetic above rather than by
/// anything an operator can observe — the shed cell says saturation happened,
/// not that a higher bound would still fit in the pod.
///
/// `pub` because it is the source of truth the test lane reads rather than
/// re-spelling; [`BaseBootstrap::with_max_inflight`] is the only way to
/// deviate from it, and production never calls it.
pub const BOOTSTRAP_MAX_INFLIGHT: usize = 2;

/// Which call site a [`BaseBootstrap::run_once`] invocation is serving.
///
/// This is the *only* carrier of call-site variation in the unit. Everything
/// that differs between the creation-time first base and the post-restore gap
/// re-base is reachable from here (the table in the module header enumerates
/// the full set), so a third call site would extend this enum rather than
/// thread flags through the body.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BootstrapTarget {
    /// SR-7.1 creation-time first base, written at `snap_seq = 1`.
    ///
    /// The `1` is not a convention: `CreateLayer` applies its delta to a fresh
    /// `ObjectState::default()` exactly once on every path (empty content
    /// included), so the committed header sequence is `1` by construction, and
    /// `LayerStore::create_layer` persists that same state. The bootstrap base
    /// therefore claims coverage of exactly the commits it carries.
    CreationBase,
    /// SR-3.6 post-restore gap re-base, written at the restore's seeded writer
    /// cursor.
    ///
    /// Every field is taken from the restore that just committed — none is
    /// re-derived here, because re-deriving `seeded_seq` would duplicate the
    /// b.8y5 seeding formula and guarantee drift against it.
    RestoreRebase {
        /// `AppliedCheckpoint::seeded_seq` — `restore_floor + band`, and the
        /// live header's `seq` at the moment the restore committed. The base
        /// this unit writes claims coverage up to exactly that point, which is
        /// what closes the gap the restore opened.
        seeded_seq: u64,
        /// The restored generation's `CheckpointMeta::observed_commit_span`,
        /// carried forward onto the re-base's own meta. NOT recoverable from
        /// `band` (`band = max(margin, span)` is not invertible), which is why
        /// the hook reads it off the meta and hands it in.
        observed_commit_span: u64,
        /// `AppliedCheckpoint::band` — the guard band this restore added, i.e.
        /// the exact value it stamped into `obj:hdr restore_seq_bump`. Used by
        /// guard (b) to prove the bump this re-base is about to clear is
        /// *this* restore's bump and not a newer one's.
        absorbed_band: u64,
    },
}

impl BootstrapTarget {
    /// The `snap_seq` this generation is written at, and the value stamped into
    /// `obj:hdr last_checkpoint_seq` on success.
    fn snap_seq(self) -> u64 {
        match self {
            Self::CreationBase => 1,
            Self::RestoreRebase { seeded_seq, .. } => seeded_seq,
        }
    }

    /// What to stamp into [`CheckpointMeta::observed_commit_span`].
    ///
    /// - `CreationBase` → [`NO_COMMIT_SPAN_MEASUREMENT`]. There is no
    ///   predecessor generation to measure against — precisely the assembler's
    ///   `last_checkpoint_seq == 0` decline case. The restore band falls back to
    ///   the configured `LIVE_EDIT_CHECKPOINT_SEQ_SAFETY_MARGIN`, exactly as it
    ///   does for any first generation.
    /// - `RestoreRebase` → the restored generation's own measurement, carried
    ///   forward. By the strict definition ("commits this generation covers that
    ///   the previous did not") a re-base covers zero *new* commits, so the
    ///   sentinel is the literal reading — but stamping it would make the
    ///   **next** restore of this layer fall back to the configured margin and
    ///   throw away the b.vnw measurement, which is the smaller-band hazard
    ///   direction (b.8y5). This is the identical argument the landed tail
    ///   capture already makes for its own meta (`tail_capture.rs`, "Stamping
    ///   the no-measurement sentinel instead would silently degrade every
    ///   composed restore of a hot layer back to the configured margin"); the
    ///   two write sites are deliberately reasoned the same way.
    fn observed_commit_span(self) -> u64 {
        match self {
            Self::CreationBase => NO_COMMIT_SPAN_MEASUREMENT,
            Self::RestoreRebase {
                observed_commit_span,
                ..
            } => observed_commit_span,
        }
    }

    /// The `obj:hdr` field/value pairs of the success stamp — TWO pairs at the
    /// creation site, THREE at the restore site.
    ///
    /// `last_checkpoint_seq` is what makes the periodic loop's
    /// skip-if-no-commits gate (`seq == last_checkpoint_seq`) suppress a
    /// redundant re-snapshot of state we just persisted; `last_checkpoint_ms` is
    /// the cadence clock the interval arm measures against. They ride one HSET
    /// so the two can never disagree about whether a generation was persisted.
    ///
    /// `restore_seq_bump` is written **only** on `RestoreRebase`, and there it
    /// is written as `0` to RETIRE the correction the restore left behind —
    /// matching the periodic loop's own three-pair HSET (`checkpointer.rs`) and
    /// for the same two reasons:
    ///
    /// 1. The base this unit just wrote sits AT `seeded_seq` and has therefore
    ///    absorbed the restore's jump. Leaving the bump would make the next
    ///    assembly subtract a band that is already behind `last_checkpoint_seq`
    ///    (`assembler::measure_observed_commit_span`) and silently deflate every
    ///    later `observed_commit_span` by a full band.
    /// 2. `tail_capture` refuses capture outright with
    ///    `TailCaptureRefusal::PostRestoreUnrebased` while `restore_seq_bump !=
    ///    0`, *before* the stream read. Clearing this column is what actually
    ///    unblocks tail capture after a restore — arguably more load-bearing
    ///    than the seq/ms pair SR-3.6 names.
    ///
    /// At `CreationBase` the column is deliberately NOT written: a freshly
    /// created layer never had a restore, so the column is simply absent, and an
    /// absent column already reads as a truthful `0` with
    /// `cursor_columns_trusted = true`. Writing it there would assert that a
    /// restore happened.
    fn cursor_pairs(self, stamped_ms: u64) -> Vec<(&'static str, String)> {
        let mut pairs = vec![
            ("last_checkpoint_seq", self.snap_seq().to_string()),
            ("last_checkpoint_ms", stamped_ms.to_string()),
        ];
        if matches!(self, Self::RestoreRebase { .. }) {
            pairs.push(("restore_seq_bump", "0".to_string()));
        }
        pairs
    }

    /// This call site's `site` label — on the metric family AND on every
    /// structured log line this unit emits.
    ///
    /// The two returned values ARE
    /// [`CHECKPOINT_BASE_BOOTSTRAP_SITES`](crate::observability::metrics::CHECKPOINT_BASE_BOOTSTRAP_SITES),
    /// the pinned `site` vocabulary of
    /// `live_edit_checkpoint_base_bootstrap_total`: [`BaseBootstrap::run_once`]
    /// passes this straight into
    /// [`Metrics::inc_checkpoint_base_bootstrap`](crate::observability::metrics::Metrics::inc_checkpoint_base_bootstrap),
    /// so a value that is not a member of that constant would open an unseeded
    /// series. It is therefore not "only a log discriminator" — it is the label
    /// value, and the fact that the same string is also the `target` field on
    /// every log line here is what lets an operator join a cell to the lines
    /// that produced it. Re-spelling either side silently breaks that join.
    ///
    /// So the arms index the constant instead of spelling its strings — the same
    /// mechanization [`BootstrapOutcome::outcome_label`] applies on the other
    /// axis of this one metric family, with [`BOOTSTRAP_SITE_LABEL_PINS`] playing
    /// the role `BOOTSTRAP_OUTCOME_LABEL_PINS` plays there: a reordering of
    /// [`CHECKPOINT_BASE_BOOTSTRAP_SITES`] becomes a compile error rather than a
    /// silent swap of which call site a cell belongs to.
    fn label(self) -> &'static str {
        match self {
            Self::CreationBase => CHECKPOINT_BASE_BOOTSTRAP_SITES[0],
            Self::RestoreRebase { .. } => CHECKPOINT_BASE_BOOTSTRAP_SITES[1],
        }
    }
}

/// What one [`BaseBootstrap::run_once`] call did — and the mint of that call's
/// `outcome` metric label.
///
/// Returned rather than `()` for two reasons. It is still the testable surface
/// for the arms that log below WARN and write no blob, which is why it existed
/// at all. And since the SR-7 observability pass it is **1:1 with
/// [`CHECKPOINT_BASE_BOOTSTRAP_OUTCOMES`]**: [`Self::outcome_label`] is the one
/// place this enum meets the metric vocabulary, and the [`BaseBootstrap::run_once`]
/// wrapper ticks exactly one cell from it per invocation.
///
/// The 1:1-ness is the property to preserve, not an accident of the current
/// arm count. A variant covering two labels would make a cell un-attributable
/// to a code arm; two variants sharing one label would hide a distinction the
/// vocabulary paid a series for. When an arm is added, add its label to the
/// constant in the same change — the `run_once_inner` arm, the variant, the
/// label, and the seeded cell move together or not at all.
///
/// Variant declaration order deliberately mirrors the constant's order, so the
/// two lists can be read side by side.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BootstrapOutcome {
    /// The base landed on the adapter and the cursor stamp committed.
    /// `persisted`; also ticks `checkpoint_writes_total{success,layer}`.
    Persisted { snap_seq: u64 },
    /// Admission shed — the in-flight bound was already saturated. Nothing was
    /// read, built, written, or stamped. `shed`.
    Shed,
    /// `obj:hdr` was absent at the step-1 read — the layer was deleted or
    /// overwritten inside the bootstrap window. `skipped_header_absent`.
    SkippedHeaderAbsent,
    /// Guard (a), PRE-PUT: the step-1 identity read already showed a generation
    /// at or above `target.snap_seq()`, so nothing was built, PUT, or stamped.
    /// `skipped_generation_present`.
    ///
    /// Deliberately **not** shared with the post-PUT guard-(b) arms below: this
    /// one costs one header read and leaves the adapter untouched, while those
    /// leave a durable blob behind. Same predicate, materially different
    /// residual, so they are different cells.
    SkippedGenerationPresent,
    /// `RestoreRebase` only, PRE-PUT: the header's cursor columns were present
    /// but unparseable, so a blind stamp could not be justified.
    /// `skipped_untrusted_cursor`.
    SkippedUntrustedCursor,
    /// POST-PUT: the conditional cursor stamp was REFUSED because `obj:hdr` no
    /// longer carries a `seq` field — the layer was deleted or overwritten while
    /// the base PUT was in flight. `unstamped_header_absent` (b.vn6).
    ///
    /// Deliberately **not** [`Self::SkippedHeaderAbsent`], which is the same
    /// predicate one phase earlier: that arm costs one header read and leaves
    /// the adapter untouched, while this one has a durable base blob behind it
    /// for a layer that no longer exists. Exactly the pre-PUT / post-PUT split
    /// the `SkippedGenerationPresent` ↔ [`Self::UnstampedGenerationPresent`]
    /// pair already draws — and the reason `skipped_header_absent` read a flat
    /// zero all the way through the dev incident (b.vn6) that this arm exists
    /// to make visible.
    ///
    /// The arm re-arms the object on `le:ckpt:{recon}:dead` and then attempts a
    /// fast-path reap of the just-written generation before returning, reporting
    /// the reap's result on the log line's `reaped` field — INFO when it
    /// succeeded and nothing is left for an operator to do, WARN when it did not
    /// and reclamation is deferred. Both steps are best-effort: the
    /// reconciliation pass's dead-object phase is the correctness backstop and
    /// this is latency.
    UnstampedHeaderAbsent,
    /// Guard (b) seq half, POST-PUT: a generation at or above
    /// `target.snap_seq()` landed while the PUT was in flight — seen by the
    /// pre-stamp read, or refused atomically by the stamp's cursor-regression
    /// floor when it landed after that read — so every cursor column was left
    /// untouched. `unstamped_generation_present`.
    ///
    /// The blob IS durable at the target and the cursor was not stamped — the
    /// benign form of that split state, since the generation that overtook us
    /// covers what ours would have.
    UnstampedGenerationPresent,
    /// Guard (b) bump half, `RestoreRebase` only, POST-PUT: the outstanding
    /// `restore_seq_bump` is not the band this re-base absorbed, so a newer
    /// restore owns this header and its bump was left standing rather than
    /// cleared over its fossils. `unstamped_foreign_restore`.
    ///
    /// A risk indicator, not a harm count — see `raw_bump_blocks_stamp`, whose
    /// same-band residual is the guard's false negative and lives inside this
    /// population.
    UnstampedForeignRestore,
    /// The blob landed but the cursor stamp did not, because a Valkey command
    /// failed: the pre-stamp `HMGET` or the stamp `HSET` itself.
    /// `unstamped_write_error`; also ticks
    /// `checkpoint_writes_total{error,layer}`.
    ///
    /// Distinct from [`Self::Failed`] on exactly the fact an operator acts on:
    /// a durable base exists at the target and only the header disagrees, so
    /// the next eligible tick re-snapshots the layer rather than the layer
    /// having no lineage root at all.
    UnstampedWriteError,
    /// No base landed: the identity read, the OMPF build, the sidecar
    /// serialization, or the PUT itself failed. Every such arm logs WARN and
    /// ticks `checkpoint_writes_total{result="error",kind="layer"}`. `failed`.
    ///
    /// The `Disabled` adapter's `NotImplemented` return folds here too — it is
    /// unreachable behind the three-way gate, and it is the one arm that ticks
    /// this cell WITHOUT a matching `writes_total{error,layer}` (ticking an
    /// adapter error for "checkpointing is off" would manufacture a noise
    /// floor).
    Failed,
}

impl BootstrapOutcome {
    /// This outcome's `outcome` label on
    /// `live_edit_checkpoint_base_bootstrap_total`.
    ///
    /// Every returned value is a member of [`CHECKPOINT_BASE_BOOTSTRAP_OUTCOMES`]
    /// (`src/observability/metrics.rs`), which is the canonical vocabulary — the
    /// same discipline `GatheredLineage::outcome_label` and
    /// [`TailCaptureDecision::outcome_label`](crate::checkpoint::tail_capture::TailCaptureDecision::outcome_label)
    /// follow against theirs. Here it is mechanized rather than merely stated:
    /// the arms index the constant instead of spelling its strings, so a
    /// spelling can only be changed in one place, and `BOOTSTRAP_OUTCOME_LABEL_PINS`
    /// makes a reordering of the constant a compile error rather than a silently
    /// mis-attributed cell.
    pub fn outcome_label(&self) -> &'static str {
        match self {
            Self::Persisted { .. } => CHECKPOINT_BASE_BOOTSTRAP_OUTCOMES[0],
            Self::Shed => CHECKPOINT_BASE_BOOTSTRAP_OUTCOMES[1],
            Self::SkippedHeaderAbsent => CHECKPOINT_BASE_BOOTSTRAP_OUTCOMES[2],
            Self::SkippedGenerationPresent => CHECKPOINT_BASE_BOOTSTRAP_OUTCOMES[3],
            Self::SkippedUntrustedCursor => CHECKPOINT_BASE_BOOTSTRAP_OUTCOMES[4],
            Self::UnstampedHeaderAbsent => CHECKPOINT_BASE_BOOTSTRAP_OUTCOMES[5],
            Self::UnstampedGenerationPresent => CHECKPOINT_BASE_BOOTSTRAP_OUTCOMES[6],
            Self::UnstampedForeignRestore => CHECKPOINT_BASE_BOOTSTRAP_OUTCOMES[7],
            Self::UnstampedWriteError => CHECKPOINT_BASE_BOOTSTRAP_OUTCOMES[8],
            Self::Failed => CHECKPOINT_BASE_BOOTSTRAP_OUTCOMES[9],
        }
    }
}

/// Compile-time check that each index [`BootstrapOutcome::outcome_label`] uses
/// still names the spelling that arm means.
///
/// Indexing the constant is what keeps the spellings un-duplicated, but it moves
/// the risk: a reordering of [`CHECKPOINT_BASE_BOOTSTRAP_OUTCOMES`] would leave
/// every arm compiling and every label still a *member* of the vocabulary — so
/// even the exhaustive membership test cannot catch it — while silently
/// attributing each arm to the wrong cell. The pins below are the only place the
/// ten strings appear twice, and that is their whole job: they are assertions,
/// not a second mint.
const BOOTSTRAP_OUTCOME_LABEL_PINS: [(usize, &str); 10] = [
    (0, "persisted"),
    (1, "shed"),
    (2, "skipped_header_absent"),
    (3, "skipped_generation_present"),
    (4, "skipped_untrusted_cursor"),
    (5, "unstamped_header_absent"),
    (6, "unstamped_generation_present"),
    (7, "unstamped_foreign_restore"),
    (8, "unstamped_write_error"),
    (9, "failed"),
];

/// `str` equality, in const context (`PartialEq` is not const).
const fn str_eq(a: &str, b: &str) -> bool {
    let (a, b) = (a.as_bytes(), b.as_bytes());
    if a.len() != b.len() {
        return false;
    }
    let mut i = 0;
    while i < a.len() {
        if a[i] != b[i] {
            return false;
        }
        i += 1;
    }
    true
}

const _: () = {
    assert!(
        CHECKPOINT_BASE_BOOTSTRAP_OUTCOMES.len() == BOOTSTRAP_OUTCOME_LABEL_PINS.len(),
        "BootstrapOutcome is 1:1 with CHECKPOINT_BASE_BOOTSTRAP_OUTCOMES: a label was added to \
         or removed from the vocabulary without a matching variant + pin here (or vice versa). \
         Add the arm, the label, the pin, and the seeded cell in one change."
    );
    let mut i = 0;
    while i < BOOTSTRAP_OUTCOME_LABEL_PINS.len() {
        let (index, expected) = BOOTSTRAP_OUTCOME_LABEL_PINS[i];
        assert!(
            str_eq(CHECKPOINT_BASE_BOOTSTRAP_OUTCOMES[index], expected),
            "CHECKPOINT_BASE_BOOTSTRAP_OUTCOMES no longer holds the spelling that \
             BootstrapOutcome::outcome_label indexes at this position, so every arm from here on \
             would tick a cell belonging to a different outcome. Re-point the indices in \
             outcome_label and the pins in BOOTSTRAP_OUTCOME_LABEL_PINS together."
        );
        i += 1;
    }
};

/// Compile-time check that each index [`BootstrapTarget::label`] uses still names
/// the spelling that call site means.
///
/// The `site` axis of `live_edit_checkpoint_base_bootstrap_total` carries the
/// same discipline as the `outcome` axis above, for the same reason: `label()`
/// indexes [`CHECKPOINT_BASE_BOOTSTRAP_SITES`] rather than re-spelling it, so a
/// reordering of that constant would leave both arms compiling and both labels
/// still *members* of the vocabulary — the runtime membership test in this
/// module's `tests` cannot see it — while attributing every creation-time base
/// to `restore_rebase` and vice versa. That is not a cosmetic mix-up: `shed` at
/// `restore_rebase` is the alertable cell (SR-3.6 leaves that layer with no DR
/// lineage root) and `shed` at `creation_base` is not, so the swap would invert
/// an alert.
///
/// The two pins below are the only place these two strings appear twice, and
/// that is their whole job: they are assertions, not a second mint. What they
/// deliberately do NOT cover — two arms pointed at the same index — is covered
/// by the runtime distinctness assertion in
/// `every_bootstrap_target_label_is_a_distinct_member_of_the_pinned_site_vocabulary`.
const BOOTSTRAP_SITE_LABEL_PINS: [(usize, &str); 2] = [(0, "creation_base"), (1, "restore_rebase")];

const _: () = {
    assert!(
        CHECKPOINT_BASE_BOOTSTRAP_SITES.len() == BOOTSTRAP_SITE_LABEL_PINS.len(),
        "BootstrapTarget is 1:1 with CHECKPOINT_BASE_BOOTSTRAP_SITES: a site was added to or \
         removed from the vocabulary without a matching variant + pin here (or vice versa). Add \
         the variant, the label, the pin, and the seeded column in one change."
    );
    let mut i = 0;
    while i < BOOTSTRAP_SITE_LABEL_PINS.len() {
        let (index, expected) = BOOTSTRAP_SITE_LABEL_PINS[i];
        assert!(
            str_eq(CHECKPOINT_BASE_BOOTSTRAP_SITES[index], expected),
            "CHECKPOINT_BASE_BOOTSTRAP_SITES no longer holds the spelling that \
             BootstrapTarget::label indexes at this position, so each call site would tick — and \
             log — the other site's identity. Re-point the indices in label() and the pins in \
             BOOTSTRAP_SITE_LABEL_PINS together."
        );
        i += 1;
    }
};

/// Cursor-regression guard predicate, shared by guard (a) and guard (b).
///
/// The general rule is *skip when a generation at or above the sequence we are
/// about to stamp has already landed*. Writing a base below the observed cursor
/// and stamping the cursor back down to it is strictly harmful at both sites: it
/// un-suppresses the loop's skip-if-no-commits gate and inflates the next
/// assembly's `observed_commit_span` (measured from our lower number instead of
/// from the generation that actually landed), which widens the DR restore guard
/// band. At the restore site the same stamp would additionally regress the
/// cursor below `seeded_seq`, i.e. below the live header's own `seq`.
///
/// **The `CreationBase` equivalence.** At `target_snap_seq == 1` this is exactly
/// the pre-existing `observed != 0` test, because `observed >= 1 ⇔ observed !=
/// 0` over `u64`. The creation site is therefore bit-for-bit preserved by the
/// generalization, and
/// [`cursor_regression_guard_is_the_nonzero_test_at_the_creation_target`] pins
/// that so a future edit cannot silently change it.
///
/// **This is the BOOTSTRAP sites' predicate, and `>=` is right for exactly
/// them.** Both targets here are a sequence this replica is about to mint a
/// generation AT, and nothing this replica did put the cursor there — so
/// `observed == target` can only mean a peer already landed and stamped at our
/// target, and blocking is correct. The chain-collapse re-base arm in
/// [`checkpointer`](crate::checkpoint::checkpointer) deliberately does NOT reuse
/// this: its target is the head tail's `to_seq`, which the capture that produced
/// that tail already stamped itself, so equality there is the normal steady
/// state rather than a peer's interference and it guards with its own
/// `observed > to_seq` form. See `checkpointer::collapse_cursor_blocks_stamp`
/// for the full argument. `pub(super)` is retained so that divergence stays one
/// grep away from this predicate rather than being invisible from it.
pub(super) fn cursor_regression_skips(observed: u64, target_snap_seq: u64) -> bool {
    observed >= target_snap_seq
}

/// Guard (b)'s reading of the raw pre-stamp `last_checkpoint_seq` value.
///
/// `None` (column genuinely absent) is the only value that permits the stamp.
/// An unparseable value counts as "present" — the conservative direction, and
/// preserved verbatim from the creation-only version: declining to stamp costs
/// one redundant re-snapshot, while stamping over a landed generation costs a
/// regressed cursor.
///
/// `pub(super)` for the reason given on [`cursor_regression_skips`]. The
/// chain-collapse re-base arm's pre-stamp `HMGET` reads the SAME raw column but
/// asks a strictly weaker question of it (`observed > to_seq`, because equality
/// there is its own capture's stamp) and carries its own
/// `collapse_cursor_blocks_stamp` — including this function's two non-numeric
/// readings, verbatim, which are the half that genuinely is shared reasoning.
pub(super) fn raw_cursor_blocks_stamp(raw: Option<&str>, target_snap_seq: u64) -> bool {
    match raw {
        None => false,
        Some(raw) => raw
            .parse::<u64>()
            .map(|observed| cursor_regression_skips(observed, target_snap_seq))
            .unwrap_or(true),
    }
}

/// Guard (b)'s `RestoreRebase`-only second reading: does the outstanding
/// `restore_seq_bump` still belong to the restore this re-base absorbed?
///
/// The stamp is permitted **only** when the observed column is exactly the band
/// the caller's own restore applied. Anything else — a different band, an
/// absent column, an unparseable one — blocks.
///
/// # The hole this closes
///
/// The re-base's third stamp column clears `restore_seq_bump`, and clearing it
/// is precisely what lets `tail_capture` read the stream again. A *stale*
/// re-base clearing a NEWER restore's bump would therefore unblock capture over
/// that newer restore's abandoned-timeline fossils — the exact class SR-1.2 and
/// SR-3.6 exist to forbid.
///
/// It is reachable, not theoretical: our PUT can sit on a degraded adapter for
/// up to the per-call ceiling, and a second state loss inside that window is
/// restored by a peer that stamps its own `band'` and (having selected before
/// our base landed) a `last_checkpoint_seq` no higher than the one we already
/// stepped over — so the seq half of guard (b) can pass while the header
/// belongs to a restore we know nothing about. Comparing the band closes it
/// with one extra field on a read we were already issuing.
///
/// # The residual it accepts
///
/// The band comparison cannot distinguish two restores that applied the
/// *same* band — and identical bands are the common case, not the exception:
/// `band = max(checkpoint_seq_safety_margin, observed_commit_span)` and the
/// margin defaults to 1000, so every layer whose measured span sits under the
/// margin restores at the same band every time. In the double-restore
/// interleaving above, a peer restore that selects the same generation stamps
/// a byte-identical `(last_checkpoint_seq, restore_seq_bump)` pair, so a
/// stale re-base passes both halves of guard (b) and clears a bump it does
/// not own — stamping `last_checkpoint_seq` below the newer restore's
/// `seeded_seq`, exactly the harm this guard exists to prevent. There is no
/// cheap stronger discriminator: the peer's `last_checkpoint_ms` is the
/// restored blob's own recorded write time and is identical too when the same
/// generation is selected, and gating on `observed == seeded_seq` would block
/// the re-base of every layer written to since its restore — every hot
/// layer's. The same-band residual is
/// accepted deliberately, alongside the read-to-HSET microsecond gap guard
/// (b) already documents.
///
/// **It gets no metric cell of its own, and that is not an oversight.** The
/// residual is precisely this guard's *false negative*: the arm it takes is the
/// permitting one, and at the decision point the two restores are byte-identical
/// in every column read, so there is nothing to count — a cell would have to be
/// ticked by code that cannot tell whether it is in the residual. What IS
/// counted is the enclosing population: every time the guard *blocks*, the
/// invocation reports [`BootstrapOutcome::UnstampedForeignRestore`] and ticks
/// `live_edit_checkpoint_base_bootstrap_total{outcome="unstamped_foreign_restore"}`.
/// That cell is a **risk indicator, not a harm count** — it marks the
/// interleaving in which same-band restores are happening at all, so a non-zero
/// reading says the residual's precondition is live in this fleet, and a zero
/// reading says it is not. The realized harm is fenced downstream rather than
/// here: a stale re-base that slips through clears a bump it does not own, and
/// SR-1.2's classifier then refuses on its ascending-jump rule — one of the
/// three `refused_seq_gap` producers — for any abandoned-timeline fossil inside
/// the read window, ticking `checkpoint_tail_total{outcome="refused_seq_gap"}`
/// at the capture unit rather than anything here.
fn raw_bump_blocks_stamp(raw: Option<&str>, absorbed_band: u64) -> bool {
    match raw {
        // Absent reads as a truthful 0, and `band >= 1` always
        // (`checkpoint_seq_safety_margin` is zero-rejected), so an absent column
        // is never this re-base's own bump. In practice the only writer that
        // clears it is a periodic tick, whose `last_checkpoint_seq` stamp the
        // seq half of the guard has already caught.
        None => true,
        Some(raw) => raw
            .parse::<u64>()
            .map(|observed| observed != absorbed_band)
            .unwrap_or(true),
    }
}

/// Base bootstrap unit (SR-7.1 creation-time first base, SR-3.6 post-restore
/// gap re-base).
///
/// **Exactly one instance** is constructed at service startup and shared as an
/// `Arc` by both call sites — held on
/// [`LiveEditService`](crate::grpc::service::LiveEditService) for `CreateLayer`
/// ([`Self::spawn_creation_base`]) and on the
/// [`RecoveryHook`](crate::checkpoint::RecoveryHook) for the restore path
/// ([`Self::spawn_restore_rebase`]). The wiring costs nothing when the feature
/// is off. The single-instance property is a wiring invariant, not an
/// optimization: the admission bound below is a field, so a second instance
/// would be a second, independent bound.
///
/// `Clone` is derived because the `spawn_*` methods must hand an owned,
/// `'static` value to the spawned task; every field is an `Arc`, a
/// cheaply-cloneable handle, or a short `String`. Cloning shares the semaphore
/// (it is an `Arc`) — a bare `Semaphore` field would give every spawned task
/// its own bound and enforce nothing.
#[derive(Clone)]
pub struct BaseBootstrap {
    /// Adapter selected by `LIVE_EDIT_CHECKPOINT_ADAPTER`. May be the
    /// `NoopCheckpointStore` for `Disabled`, whose `NotImplemented` return the
    /// PUT arm swallows silently — though the three-way gate below means a
    /// `Disabled` deployment never spawns in the first place.
    store: Arc<dyn CheckpointStore>,
    /// Read-side Valkey handle. Used for the pre-PUT header identity read and
    /// the post-PUT cursor stamp, both against the data pool.
    client: ValkeyClient,
    /// Same key schema the rest of the service uses.
    keys: KeySchema,
    /// Shared tracker that owns the spawned task — the same one the
    /// session-checkpoint dispatcher uses, so graceful shutdown's existing
    /// `close()` + `wait()` drains bootstrap tasks before
    /// `ValkeyClient::quit()` with no new shutdown wiring.
    tracker: TaskTracker,
    /// Metrics handle. Two families: the per-invocation
    /// `inc_checkpoint_base_bootstrap(outcome, site)` tick in
    /// [`Self::run_once`], and the `inc_checkpoint_write(result, "layer")` ticks
    /// the generation-completion arms inside `run_once_inner` share with
    /// the periodic loop.
    metrics: SharedMetrics,
    /// Service / replica identifier persisted as
    /// [`CheckpointMeta::creator_replica_id`].
    creator_replica_id: String,
    /// The resolved three-way gate — see [`BaseBootstrap::new`].
    enabled: bool,
    /// Admission permits for [`Self::run_once`] bodies —
    /// [`BOOTSTRAP_MAX_INFLIGHT`] of them, or whatever
    /// [`Self::with_max_inflight`] set in a test.
    ///
    /// `Arc`, not a bare `Semaphore`: the struct derives `Clone` and every spawn
    /// clones it, so a bare field would hand each task a private bound and
    /// enforce nothing. Never `close()`d.
    inflight: Arc<Semaphore>,
    /// The bound `inflight` was sized to, retained only so the shed WARN can
    /// name the number an operator would otherwise have to look up in the
    /// source. `Semaphore` exposes `available_permits()`, which is 0 at exactly
    /// the moment the shed arm runs.
    max_inflight: usize,
    /// Test-only hook awaited once per run immediately before the conditional
    /// cursor stamp, after the pre-stamp `HMGET` and guard (b) have passed. See
    /// [`Self::with_pre_stamp_hook_for_test`].
    #[cfg(any(test, feature = "test-checkpoint-fault-injection"))]
    pre_stamp_hook_for_test: Option<PreStampHook>,
}

/// The hook type of [`BaseBootstrap::with_pre_stamp_hook_for_test`].
#[cfg(any(test, feature = "test-checkpoint-fault-injection"))]
pub type PreStampHook = Arc<dyn Fn() -> futures::future::BoxFuture<'static, ()> + Send + Sync>;

impl BaseBootstrap {
    /// Construct a bootstrap unit. Argument order/shape mirrors
    /// [`SessionCheckpointWriter::new`](crate::checkpoint::SessionCheckpointWriter::new)
    /// so the two `main.rs` wirings read the same.
    ///
    /// `creator_replica_id` is typically `LIVE_EDIT_SERVICE_LABEL` (falling
    /// back to `$HOSTNAME` / `unknown`) and is stamped into every persisted
    /// [`CheckpointMeta`] so an operator can attribute a base blob to its
    /// origin replica.
    ///
    /// **`enabled` is computed by the caller** and must be the full three-way
    /// conjunction, `Config::checkpoint_base_bootstrap_enabled` (which the pod
    /// memory plan also reads, to charge the base bootstraps' memory):
    ///
    /// ```text
    /// config.checkpoint_enabled
    ///     && !matches!(config.checkpoint_adapter, CheckpointAdapter::Disabled)
    ///     && config.checkpoint_tail_enabled
    /// ```
    ///
    /// This deliberately differs from
    /// [`SessionCheckpointWriter::new`](crate::checkpoint::SessionCheckpointWriter::new)'s
    /// `enabled`, which is the `!Disabled` conjunct alone: session checkpoints
    /// are written on any replica with a real adapter, even one that runs no
    /// checkpoint loop. SR-7.1 mandates all three conjuncts here, because this
    /// unit exists to serve the tail path — it is off whenever the tail path
    /// is.
    pub fn new(
        store: Arc<dyn CheckpointStore>,
        client: ValkeyClient,
        keys: KeySchema,
        tracker: TaskTracker,
        metrics: SharedMetrics,
        creator_replica_id: String,
        enabled: bool,
    ) -> Self {
        Self {
            store,
            client,
            keys,
            tracker,
            metrics,
            creator_replica_id,
            enabled,
            inflight: Arc::new(Semaphore::new(BOOTSTRAP_MAX_INFLIGHT)),
            max_inflight: BOOTSTRAP_MAX_INFLIGHT,
            #[cfg(any(test, feature = "test-checkpoint-fault-injection"))]
            pre_stamp_hook_for_test: None,
        }
    }

    /// Install a hook awaited once per run immediately before the conditional
    /// cursor stamp — after the pre-stamp `HMGET` read and both halves of
    /// guard (b) have let the run through.
    ///
    /// **Test-only.** This is the window the stamp's atomic cursor-regression
    /// floor exists for: two overlapping `CreationBase` runs for one object
    /// that BOTH pass the pre-stamp read before either stamps. A hook that
    /// holds each run here until both have arrived (a 2-party barrier) drives
    /// that interleaving deterministically, so a test can show the floor
    /// refuses the second stamp — one `persisted`, one
    /// `unstamped_generation_present`, and one
    /// `checkpoint_writes_total{success,layer}` tick — instead of racing two
    /// runs and hoping. The hook is infallible by type, so it adds no exit
    /// path. Shared by every clone of this unit, like the admission bound.
    ///
    /// Gated under `test-checkpoint-fault-injection`, the checkpoint lane's
    /// seam gate, so the field, the builder and the await compile out of
    /// production library builds.
    #[cfg(any(test, feature = "test-checkpoint-fault-injection"))]
    pub fn with_pre_stamp_hook_for_test(mut self, hook: PreStampHook) -> Self {
        self.pre_stamp_hook_for_test = Some(hook);
        self
    }

    /// Override the in-flight bound. **Test affordance; production never calls
    /// it.**
    ///
    /// Ungated by a cargo feature, deliberately and unlike
    /// `RecoveryHook::with_force_post_restore_notfound_for_test`: that one
    /// *injects fault behaviour* into the production path, while this one only
    /// resizes a semaphore and cannot change production behaviour unless it is
    /// invoked. Its purpose is to make the shed arm drivable at
    /// `max_inflight = 1` without needing [`BOOTSTRAP_MAX_INFLIGHT`] concurrent
    /// blocked PUTs to reproduce it.
    ///
    /// `0` would shed unconditionally, which is not a state any caller wants and
    /// not one production can reach, so it is clamped to 1 rather than silently
    /// disabling the unit.
    pub fn with_max_inflight(mut self, max_inflight: usize) -> Self {
        self.max_inflight = max_inflight.max(1);
        self.inflight = Arc::new(Semaphore::new(self.max_inflight));
        self
    }

    /// Spawn the creation-time base capture for a just-committed layer.
    ///
    /// **Synchronous** — returns as soon as the task is on the tracker. The
    /// CreateLayer RPC has already determined its response value by the time
    /// this is reached, so nothing here can fail the RPC or add measurable
    /// latency to the commit path: the only synchronous cost is the gate read,
    /// a handful of `Arc` clones, and one `tracker.spawn`.
    ///
    /// `state` is **moved**, never cloned — the whole reason the spawn lives in
    /// the gRPC handler rather than `LayerStore::create_layer` (which takes
    /// `&ObjectState`). Cloning a `LIVE_EDIT_MAX_LAYER_SIZE`-scale state on the
    /// commit path would break the zero-synchronous-work fence outright.
    ///
    /// The disabled short-circuit lives inside `Self::spawn` rather than at
    /// the call site (the [`RecoveryHook`](crate::checkpoint::RecoveryHook)
    /// posture), so the handler never branches on configuration. When disabled
    /// it drops `state` and does nothing else: no task, no Valkey command, no
    /// allocation, no metric.
    ///
    /// **Which `CreateLayer` outcomes spawn.** The handler spawns for the
    /// object that is the layer after its call:
    ///
    /// - a create that committed — the fresh-create and the overwrite path
    ///   alike: `create_layer_with_operation` returns the NEW `object_id` either
    ///   way, and the new object's own header stamps `seq = 1`, so keying on the
    ///   returned id with `snap_seq = 1` is correct for both. Nothing after the
    ///   commit point fails the call, so every create whose commit point the
    ///   call saw pass answers OK and spawns here;
    /// - an `ALREADY_COMMITTED` answer to a re-send carrying the same
    ///   `operation_id`: spawned for the RECORDED object, from the state the
    ///   handler rebuilds by parsing the re-sent bytes (byte-identical to the
    ///   original's — the request digests match), because the original call's
    ///   reply was lost and it may never have reached its own spawn.
    ///
    /// **A second spawn for one object is safe.** It targets the same
    /// `(object_id, snap_seq = 1, Base)` key with the same bytes
    /// (`build_reset_ompf_from_state` is byte-reproducible for equal state). If
    /// the first run already stamped the cursor, the second takes guard (a)
    /// before any PUT (`skipped_generation_present`); if they overlap, the
    /// adapter collapses the two PUTs into one generation (the store's
    /// same-key overwrite contract) and the later stamp is refused
    /// (`unstamped_generation_present`) — by guard (b)'s pre-stamp read, or,
    /// when both runs passed that read before either stamped, by the stamp's
    /// own atomic cursor-regression floor (`stamp_obj_header_cursor_below`).
    /// That read-then-stamp window was new with the re-send re-spawn — before
    /// it, no two `CreationBase` runs could target one object — and the floor
    /// closes it. So exactly one run stamps and ticks `persisted` and
    /// `checkpoint_writes_total{success,layer}`; each run ticks exactly one
    /// `base_bootstrap_total` cell, as every invocation does.
    ///
    /// **Residuals.** A deadline drop or crash between the commit point and the
    /// handler's spawn skips it, and only a re-send carrying the operation id
    /// re-spawns it — so a layer whose client never re-sends keeps no first
    /// base until it quiesces once or a later generation lands. The same holds
    /// when the call answers `UNAVAILABLE` with its commit in doubt (the commit
    /// script's reply lost while the script ran, or an overwrite whose commit
    /// its in-process finisher or the orphan sweep completes after the
    /// answer): nothing is spawned until a re-send carrying the operation id is
    /// answered `ALREADY_COMMITTED`; a client without an id, or one that gives
    /// up, leaves the layer base-less until it quiesces once or a later
    /// generation lands. And a layer
    /// overwritten or deleted *within* the bootstrap window (or before a
    /// re-send's re-spawn) leaves a base at `(old_object_id, 1)` that nothing
    /// will ever restore. That is a checkpoint-store orphan, and the §22.17.5
    /// reconciliation sweep is the existing owner of that class — no new
    /// cleanup path is introduced here.
    pub fn spawn_creation_base(&self, object_id: u64, state: ObjectState) {
        self.spawn(object_id, state, BootstrapTarget::CreationBase);
    }

    /// Spawn the SR-3.6 post-restore gap re-base for a just-restored layer.
    ///
    /// Called from [`RecoveryHook::try_restore_layer`](crate::checkpoint::RecoveryHook::try_restore_layer)'s
    /// applied arm, and from nowhere else. Same **synchronous** contract as
    /// [`Self::spawn_creation_base`], and here that contract is not a nicety: the
    /// recovery hook holds the per-`object_id` admission mutex across its whole
    /// body, and it runs on the lazy-recovery read path — including
    /// `SubscribeReadLayer`, which is exempt from the tower deadline layer.
    /// Awaiting a PUT bounded only by the adapter's 180 s per-call ceiling would
    /// reintroduce exactly the b.gg7 head-of-line stall the recovery fetch
    /// budget was added to cap, with every other waiter for that layer queued
    /// behind it. SR-3.6 specifies a best-effort tracker task; this is it.
    ///
    /// `state` is the restore's own already-in-memory `ObjectState`, moved out
    /// of `AppliedCheckpoint` — no re-fetch, no re-compose, no clone.
    ///
    /// `target` is taken rather than constructed here so the three restore-side
    /// numbers travel together from the one place that owns them. Passing a
    /// [`BootstrapTarget::CreationBase`] would be a wiring bug, not a behaviour
    /// choice, so it is a `debug_assert` rather than a silent fold: it would
    /// write a base at `1` for a restored layer whose header sits far above it,
    /// which guard (a) would then decline anyway.
    pub fn spawn_restore_rebase(
        &self,
        object_id: u64,
        state: ObjectState,
        target: BootstrapTarget,
    ) {
        debug_assert!(
            matches!(target, BootstrapTarget::RestoreRebase { .. }),
            "spawn_restore_rebase requires BootstrapTarget::RestoreRebase, got {target:?}"
        );
        self.spawn(object_id, state, target);
    }

    /// Shared spawn body for both call sites.
    ///
    /// The disabled short-circuit lives HERE rather than at either call site
    /// (the [`RecoveryHook`](crate::checkpoint::RecoveryHook) posture), so
    /// neither the handler nor the hook branches on configuration. When disabled
    /// this drops `state` and does nothing else: no task, no Valkey command, no
    /// allocation, no metric.
    fn spawn(&self, object_id: u64, state: ObjectState, target: BootstrapTarget) {
        if !self.enabled {
            // Checkpointing off, adapter `Disabled`, or the tail path off:
            // drop `state` and do nothing. Byte-identical to the pre-feature
            // behavior of both call sites.
            return;
        }

        let unit = self.clone();
        self.tracker.spawn(async move {
            unit.run_once(object_id, state, target).await;
        });
    }

    /// The spawned task body, exposed directly so it can be awaited.
    ///
    /// Deliberately a named `pub async fn` rather than a closure buried inside
    /// `Self::spawn`: awaiting it is what makes the failure arms and the
    /// duplicate-spawn idempotency property testable without driving a gRPC
    /// request or a restore.
    ///
    /// Never returns an error — every arm is terminal and best-effort, and the
    /// operator-visible signal is the metric plus the log line, not a return
    /// value nobody could act on. The [`BootstrapOutcome`] it *does* return
    /// names the arm that was taken; both `spawn_*` wrappers discard it.
    ///
    /// # The single tick site IS the sum identity
    ///
    /// This function is a thin wrapper whose only job is the
    /// `live_edit_checkpoint_base_bootstrap_total` increment, and the whole body
    /// lives in `run_once_inner`. That split is what makes "exactly one
    /// increment per invocation" — the identity the family's consumers read
    /// `sum(…)` against, and the reason its benign cells are a usable
    /// denominator — a **structural** property rather than a discipline spread
    /// over the inner function's fourteen exits. A fifteenth exit added
    /// tomorrow inherits the tick; it cannot forget it, and it cannot tick
    /// twice.
    ///
    /// `tail_capture.rs`'s `capture_layer_tail_with_bound` / `capture_inner`
    /// pair is the precedent and was adopted here for the same reason. Nothing
    /// else moved out: the owned admission `_permit` still covers the entire
    /// body it guards, and the shed arm still drops `state` before returning,
    /// because both live inside the inner function.
    pub async fn run_once(
        &self,
        object_id: u64,
        state: ObjectState,
        target: BootstrapTarget,
    ) -> BootstrapOutcome {
        let outcome = self.run_once_inner(object_id, state, target).await;
        self.metrics
            .inc_checkpoint_base_bootstrap(outcome.outcome_label(), target.label());
        outcome
    }

    /// The bootstrap itself. Split out so [`Self::run_once`] can own the metric
    /// tick for every exit path at one site; every arm traces and returns at its
    /// own site, where the arm-specific context lives.
    async fn run_once_inner(
        &self,
        object_id: u64,
        state: ObjectState,
        target: BootstrapTarget,
    ) -> BootstrapOutcome {
        let target_snap_seq = target.snap_seq();

        // ── Step 0: admission — acquire or shed, NEVER queue ─────────────
        //
        // The permit is owned and held for the whole body, so it releases on
        // every exit path including all the early returns below. Taken HERE
        // rather than at the `spawn_*` wrappers so the bound covers both call
        // sites and any direct `run_once` caller with one permit pool.
        //
        // `try_acquire_owned`, never `acquire().await`: a queued task still
        // holds its `state` — and would then go on to build the OMPF — so
        // queueing preserves exactly the in-memory retention the bound exists to
        // cap. It would convert a memory bound into a latency bound and close
        // nothing.
        let _permit = match Arc::clone(&self.inflight).try_acquire_owned() {
            Ok(permit) => permit,
            Err(e) => {
                // `Closed` is unreachable — nothing ever closes this semaphore —
                // and folding it onto the shed arm is the safe direction either
                // way (shedding is already a defined, observable posture).
                debug_assert!(
                    matches!(e, TryAcquireError::NoPermits),
                    "the bootstrap admission semaphore is never closed, got {e:?}"
                );
                tracing::warn!(
                    object_id,
                    target = target.label(),
                    snap_seq = target_snap_seq,
                    max_inflight = self.max_inflight,
                    "base bootstrap: shed — the in-flight bootstrap bound is saturated. No \
                     base was written for this layer and no cursor column was stamped, so \
                     its tail capture refuses observably until its next eligible periodic \
                     checkpoint tick lands a base, or it is restored again"
                );
                // `state` is dropped here, immediately — the whole point of
                // shedding rather than queueing.
                drop(state);
                return BootstrapOutcome::Shed;
            }
        };

        let object_id_str = object_id.to_string();
        let loader = ValkeySnapshotLoader {
            client: &self.client,
            keys: &self.keys,
        };

        // ── Step 1: identity + existence read ────────────────────────────
        //
        // Reading the persisted header beats threading identity down from the
        // caller for three reasons: it is the *persisted* identity rather
        // than the pre-commit arguments, it is the only way to obtain
        // `server_ts` (minted inside `create_layer` / restored by
        // `write_restored_object`, never returned to the caller), and it doubles as a
        // free existence check for the delete/overwrite race below.
        let identity = match loader.read_header_identity(&object_id_str).await {
            Ok(Some(identity)) => identity,
            Ok(None) => {
                // `obj:hdr` vanished between the commit and this read — the
                // layer was deleted or overwritten inside the bootstrap
                // window. A routine race, not a fault: mirror the assembler's
                // NotFound posture (no PUT, no stamp, no `writes_total` tick —
                // the wrapper's `skipped_header_absent` cell is the only mark
                // this arm leaves).
                //
                // Since b.hka review round 2 this arm also covers a header that
                // EXISTS with an empty `session_id` / `layer_name`:
                // `read_header_identity` reports the two identically because a
                // generation stamped from such a header is permanently
                // unrestorable, and the WARN naming the degenerate columns is
                // emitted there. Before that, this unit had no check at all and
                // wrote a base the adapter refused with a `Permanent`.
                tracing::debug!(
                    object_id,
                    target = target.label(),
                    "base bootstrap: obj:hdr absent or identity-empty (layer deleted or \
                     overwritten within the bootstrap window, or a partially written header); \
                     skipping"
                );
                return BootstrapOutcome::SkippedHeaderAbsent;
            }
            Err(e) => {
                tracing::warn!(
                    object_id,
                    target = target.label(),
                    is_transient = e.is_transient(),
                    is_permanent = e.is_permanent(),
                    error = %e,
                    "base bootstrap: obj:hdr identity read failed; no base written. \
                     Tail capture will refuse for this layer until its next eligible \
                     periodic checkpoint tick lands a base"
                );
                self.metrics.inc_checkpoint_write("error", "layer");
                return BootstrapOutcome::Failed;
            }
        };

        // Cursor-regression guard, part (a). An observed `last_checkpoint_seq`
        // at or above the sequence we are about to stamp means a generation
        // that already covers this one has landed — the lineage has a root at
        // least as high as ours and this unit's whole purpose is served.
        // Writing a base below it and stamping the cursor back down would be
        // strictly harmful: it un-suppresses the loop's skip-if-no-commits gate
        // and inflates the next assembly's `observed_commit_span` (measured
        // from our lower number instead of from the generation that actually
        // landed), which widens the DR restore guard band. Same posture as the
        // vanished-header arm: no PUT, no stamp, no `writes_total` tick — this
        // arm's only mark is the wrapper's `skipped_generation_present` cell,
        // which is PRE-PUT and therefore deliberately not the same cell as the
        // post-PUT guard-(b) skips below.
        //
        // Read the two site semantics off one predicate
        // (`cursor_regression_skips`):
        //
        //  - `CreationBase` (target 1) — "any generation has landed", i.e. the
        //    pre-existing `!= 0` test, bit for bit.
        //  - `RestoreRebase` (target `seeded_seq`) — "a generation at or above
        //    the restore's seeded cursor has landed". The restore's OWN stamp
        //    sits at `snap_seq`, strictly below `seeded_seq` (the band is
        //    unconditional and `>= 1`), so it correctly does NOT trip this
        //    guard — which is the whole reason the guard had to be generalized:
        //    the un-generalized `!= 0` form would abort every re-base.
        if cursor_regression_skips(identity.last_checkpoint_seq, target_snap_seq) {
            tracing::debug!(
                object_id,
                target = target.label(),
                target_snap_seq,
                last_checkpoint_seq = identity.last_checkpoint_seq,
                "base bootstrap: a checkpoint generation at or above this target already \
                 landed for this object; skipping"
            );
            return BootstrapOutcome::SkippedGenerationPresent;
        }

        // `cursor_columns_trusted == false` means a cursor column was present
        // but UNPARSEABLE and folded to 0 (genuine absence is trusted — see
        // `SnapshotLoader::read_header_identity`). This arm is deliberately
        // asymmetric, `RestoreRebase` only:
        //
        //  - At the restore site this read happens microseconds after our own
        //    `write_restored_object` stamped those very columns. An unparseable value
        //    there means another writer is putting garbage in the header —
        //    precisely when a blind stamp (and, worse, a blind CLEARING of
        //    `restore_seq_bump`) is most dangerous. Nothing is lost by
        //    skipping: `tail_capture` independently refuses on
        //    `!cursor_columns_trusted`, so the layer's capture is refusing
        //    either way, and the re-base is exactly what would have unblocked
        //    it.
        //  - At the creation site the field stays ignored, as it always has:
        //    a fresh layer carries no cursor columns at all (`create_layer`
        //    passes `checkpoint_cursor = None`), and this unit consults no span
        //    measurement there. Adding the arm would be a behaviour change to a
        //    landed, tested call site for no reachable benefit — the asymmetry
        //    is intentional and should not be "harmonized" away.
        if matches!(target, BootstrapTarget::RestoreRebase { .. })
            && !identity.cursor_columns_trusted
        {
            tracing::warn!(
                object_id,
                target = target.label(),
                target_snap_seq,
                "base bootstrap: obj:hdr cursor columns are untrustworthy (a column was \
                 present but unparseable), so the post-restore re-base declines to stamp \
                 over them; tail capture refuses for this layer either way"
            );
            return BootstrapOutcome::SkippedUntrustedCursor;
        }

        // ── Step 2: build the reset OMPF from the committed creation state ──
        let (ompf_bytes, ordered_hashes) = match build_reset_ompf_from_state(&state) {
            Ok(built) => built,
            Err(e) => {
                tracing::warn!(
                    object_id,
                    is_transient = e.is_transient(),
                    is_permanent = e.is_permanent(),
                    error = %e,
                    "base bootstrap: reset-OMPF build failed; no base written. \
                     Tail capture will refuse for this layer until its next eligible \
                     periodic checkpoint tick lands a base"
                );
                self.metrics.inc_checkpoint_write("error", "layer");
                return BootstrapOutcome::Failed;
            }
        };
        // The layer's state (of no capped size: a layer grown through
        // UpdateLayer can exceed `LIVE_EDIT_MAX_LAYER_SIZE`) is dead after the
        // build; dropping it here keeps it from sitting on the task frame
        // across the PUT / pre-stamp re-read / HSET awaits.
        drop(state);

        // ── Step 3: ext-refs sidecar ─────────────────────────────────────
        //
        // The blob bytes behind these hashes are NOT re-mirrored here: the
        // write path already did it. `store_external_values` runs before
        // `create_layer` in the CreateLayer handler, and the SR-3.1 mirror
        // enqueues from there. This unit needs the hash list only.
        //
        // `sidecar_bytes` is ALWAYS `Some`, even when the list is empty —
        // matching `assemble_inner`, which is what makes the adapter's b.s5y
        // publish-order obligation apply to bootstrap bases too. Note this is
        // a BASE: `validate_tail_put`'s "tails carry no sidecar" rejection
        // guards tail puts only, so the Epic's tail deferral does not apply
        // here and must not be read across.
        let sidecar_bytes = match ext_refs_sidecar_json(&ordered_hashes) {
            Ok(bytes) => bytes,
            Err(e) => {
                tracing::warn!(
                    object_id,
                    is_transient = e.is_transient(),
                    is_permanent = e.is_permanent(),
                    error = %e,
                    "base bootstrap: ext-refs sidecar serialization failed; no base \
                     written. Tail capture will refuse for this layer until its next \
                     eligible periodic checkpoint tick lands a base"
                );
                self.metrics.inc_checkpoint_write("error", "layer");
                return BootstrapOutcome::Failed;
            }
        };

        // ── Step 4: meta ─────────────────────────────────────────────────
        let ompf_byte_len = ompf_bytes.len();
        let sidecar_byte_len = sidecar_bytes.len();
        let ext_hash_count = ordered_hashes.len();
        let meta = CheckpointMeta {
            creator_replica_id: self.creator_replica_id.clone(),
            written_at_unix_ms: now_unix_ms(),
            snap_seq: Some(target_snap_seq),
            session_id: identity.session_id,
            layer_name: identity.layer_name,
            created_by: identity.created_by,
            server_ts: identity.server_ts,
            // Per-variant: the sentinel at the creation site (no predecessor
            // generation to measure against), the restored generation's own
            // measurement carried forward at the restore site. The full
            // argument — including why the strictly-literal sentinel is the
            // WRONG answer for a re-base — is on
            // `BootstrapTarget::observed_commit_span`.
            observed_commit_span: target.observed_commit_span(),
            // Spelled out even though it is the `Default`: a base-vs-tail put
            // must never be implicit at a write site.
            kind: CheckpointKind::Base,
            sidecar_bytes: Some(Bytes::from(sidecar_bytes)),
            ..CheckpointMeta::default()
        };

        // ── Step 4b: cheap pre-PUT liveness re-check (b.vn6) ─────────────
        //
        // **This is an optimisation, not the guard.** The guard is the
        // conditional stamp at step 6, which is atomic and therefore actually
        // closes the race; this check is racy by construction and buys nothing
        // for a small layer, where step 1's read and the PUT are microseconds
        // apart. What it does buy is the case the dev incident was made of: a
        // large layer whose OMPF build and sidecar serialization above take
        // long enough for a `DeleteLayer` to land in between. Catching that
        // here converts a would-be durable-garbage outcome — a blob written for
        // a layer that no longer exists, then reaped — into a clean pre-PUT
        // `SkippedHeaderAbsent` with nothing written at all.
        //
        // `skipped_header_absent` is the right cell precisely because this is
        // PRE-PUT and leaves the adapter untouched; the post-PUT refusal is a
        // different cell (`unstamped_header_absent`) for that exact reason.
        //
        // The probe body itself now lives in `checkpoint::put_guard`, which the
        // other three durable-generation producers share (b.hka). This is the
        // site it was written for (b.vn6); folding it into the shared helper
        // rather than leaving a bootstrap-local copy beside it is what keeps
        // the predicate — `HEXISTS ... "seq"`, and the advisory
        // proceed-on-error posture — from drifting between the four. That
        // module's header carries the full argument, including why the field
        // form is stronger than `EXISTS` and why this can never replace the
        // atomic conditional stamp at step 6.
        //
        // ── Step 5: the single PUT, skipped if the object is already dead ─
        //
        // `header_key` is for steps 6 and 7 below, which stamp and re-read the
        // same key; the guard's probe derives its own key internally.
        let header_key = self.keys.obj_header(&object_id_str);
        let handle = match crate::checkpoint::put_guard::put_layer_checkpoint_if_live(
            Some(crate::checkpoint::put_guard::ObjectLivenessProbe::new(
                &self.client,
                &self.keys,
            )),
            self.store.as_ref(),
            object_id,
            target_snap_seq,
            Bytes::from(ompf_bytes),
            meta,
            "base bootstrap",
        )
        .await
        {
            Ok(None) => {
                // The shared guard logs the skip in its own vocabulary
                // (`context = "base bootstrap"`), which cannot say WHICH
                // bootstrap target this was. Restoring that one field here
                // keeps the pre-b.hka observability intact without widening
                // `put_guard`'s `context` argument into a per-target string —
                // the guard's vocabulary stays a stable, groupable set of four
                // producer names.
                tracing::debug!(
                    object_id,
                    target = target.label(),
                    snap_seq = target_snap_seq,
                    "base bootstrap: pre-PUT liveness probe found obj:hdr gone; nothing written"
                );
                return BootstrapOutcome::SkippedHeaderAbsent;
            }
            Ok(Some(handle)) => handle,
            Err(CheckpointError::NotImplemented { .. }) => {
                // The `Disabled` adapter's `NoopCheckpointStore` is the only
                // producer, and the three-way gate makes it unreachable here —
                // handled silently for the same reason `session_writer` does:
                // ticking `error` on the "checkpointing is off" pathway would
                // manufacture a noise floor that represents no failure. Folded
                // onto `Failed` because it is an error return from the store
                // that produced no base; it is the one arm that reaches the
                // wrapper's `failed` cell WITHOUT a matching
                // `writes_total{error,layer}`, and the asymmetry is the point —
                // `writes_total` stays a count of real adapter failures.
                return BootstrapOutcome::Failed;
            }
            Err(e) => {
                tracing::warn!(
                    object_id,
                    target = target.label(),
                    snap_seq = target_snap_seq,
                    is_transient = e.is_transient(),
                    is_permanent = e.is_permanent(),
                    error = %e,
                    "base bootstrap: base PUT failed; no base landed at this target. Tail \
                     capture will refuse for it until its next eligible periodic checkpoint \
                     tick lands a base. Not retried"
                );
                // A bootstrap PUT is a layer-base adapter write, so its home
                // is the existing `checkpoint_writes_total{result,kind}`
                // family — no new metric names, no label-cardinality change.
                // It also makes a fleet-wide bootstrap failure visible
                // immediately, instead of only later as `refused_no_base`.
                self.metrics.inc_checkpoint_write("error", "layer");
                return BootstrapOutcome::Failed;
            }
        };

        // ── Step 6: cursor stamp — ONE HSET, TWO pairs (three on a re-base) ──
        //
        // `last_checkpoint_seq` is what makes the periodic loop's
        // skip-if-no-commits gate (`seq == last_checkpoint_seq`) suppress a
        // redundant re-snapshot of state we just persisted; `last_checkpoint_ms`
        // is the cadence clock the interval arm measures against; and on a
        // re-base `restore_seq_bump = 0` retires the restore's correction. They
        // ride ONE HSET so they can never disagree about the generation they
        // describe. `BootstrapTarget::cursor_pairs` owns which columns those
        // are and argues each one.
        //
        // The data pool, deliberately NOT the maintenance pool: that pool
        // defaults to `LIVE_EDIT_MAINTENANCE_POOL_SIZE = 1` and is dedicated
        // to sweep traffic, while this is lifecycle-event cadence — the same
        // reasoning the session-checkpoint dispatcher applies to its own
        // reads. (`header_key` is the one built just above step 5, whose only
        // consumers are this stamp and the step-7 re-read.)

        // Cursor-regression guard, part (b). The step-1 early-abort closes the
        // common case, but the PUT above can sit on a degraded adapter for up
        // to the per-call ceiling, during which a periodic tick can land a
        // verified snapshot at a higher `snap_seq` and stamp it — or, at the
        // restore site, a whole second restore can land. Re-read the cursor
        // columns immediately before the stamp and skip when either moved.
        // The seq half is re-checked atomically by the stamp itself (its
        // cursor-regression floor, below), so a `last_checkpoint_seq` that
        // reaches our target between this read and the stamp is refused there.
        // Residual, stated plainly: the bump half has no such twin, so for it
        // this narrows the window to the microseconds between the re-read and
        // the stamp.
        //
        // ONE `HMGET` rather than an `HGET`, so the two columns are read
        // atomically by Valkey's single command loop and cannot disagree about
        // which restore the header belongs to. `restore_seq_bump` is consulted
        // on the `RestoreRebase` arm only (see `raw_bump_blocks_stamp`); the
        // creation site reads it and ignores it, which costs one field on a
        // round-trip it was already making.
        let observed: Vec<Option<String>> = match self
            .client
            .pool
            .hmget(&header_key, vec!["last_checkpoint_seq", "restore_seq_bump"])
            .await
        {
            Ok(raw) => raw,
            Err(e) => {
                FredErrOrigin::client(&self.client, POOL_LABEL_DATA).observe_error(&e);
                tracing::warn!(
                    object_id,
                    target = target.label(),
                    snap_seq = target_snap_seq,
                    error = %e,
                    "base bootstrap: pre-stamp HMGET of the obj:hdr cursor columns \
                     failed; base blob landed but the cursor did not, so the next eligible \
                     tick re-snapshots this layer"
                );
                self.metrics.inc_checkpoint_write("error", "layer");
                // `UnstampedWriteError`, not `Failed`: the base IS durable at
                // the target and only the header disagrees, which is a
                // materially different operator posture from "no base landed".
                return BootstrapOutcome::UnstampedWriteError;
            }
        };
        let observed_seq = observed.first().and_then(|v| v.as_deref());
        let observed_bump = observed.get(1).and_then(|v| v.as_deref());

        if raw_cursor_blocks_stamp(observed_seq, target_snap_seq) {
            tracing::debug!(
                object_id,
                target = target.label(),
                snap_seq = target_snap_seq,
                observed_last_checkpoint_seq = observed_seq.unwrap_or("<absent>"),
                "base bootstrap: a checkpoint generation landed while the base PUT \
                 was in flight; leaving every cursor column untouched"
            );
            // POST-PUT, so a distinct cell from guard (a)'s
            // `skipped_generation_present`: our blob is durable at the target
            // and only the stamp was declined.
            return BootstrapOutcome::UnstampedGenerationPresent;
        }

        // C6, `RestoreRebase` only: prove the outstanding `restore_seq_bump` is
        // the one THIS re-base absorbed before clearing it. A stale re-base
        // clearing a newer restore's bump would unblock tail capture over that
        // restore's fossils — see `raw_bump_blocks_stamp` for the reachable
        // interleaving and why the seq half above does not cover it.
        if let BootstrapTarget::RestoreRebase { absorbed_band, .. } = target {
            if raw_bump_blocks_stamp(observed_bump, absorbed_band) {
                tracing::warn!(
                    object_id,
                    target = target.label(),
                    snap_seq = target_snap_seq,
                    absorbed_band,
                    observed_restore_seq_bump = observed_bump.unwrap_or("<absent>"),
                    "base bootstrap: obj:hdr carries a restore_seq_bump this re-base did \
                     not absorb, so a newer restore owns this header; leaving every cursor \
                     column untouched rather than clearing a bump that still guards fossils"
                );
                // Its own cell, not the seq half's: this is the only signal a
                // fleet has that same-band double restores — the residual
                // `raw_bump_blocks_stamp` documents — are live at all.
                return BootstrapOutcome::UnstampedForeignRestore;
            }
        }

        // ── The conditional stamp (b.vn6) ────────────────────────────────
        //
        // This used to be a bare `HSET`, with a comment arguing that guarding
        // one of two structurally identical stamp sites would be worse than
        // guarding neither. Field evidence falsified the premise and a shared
        // primitive dissolved the dilemma: `scripts/stamp_cursor_if_live.lua` is
        // one Lua script, and ALL FOUR stamp sites go through it — this one via
        // `stamp_obj_header_cursor_below` (with its cursor-regression floor), the
        // checkpoint loop's three via `stamp_obj_header_cursor` — so there is no
        // longer an unguarded sibling to be inconsistent with.
        //
        // The exposure the old comment called theoretical is what this ticket
        // was filed for: a bare `HSET` on an UNLINKed header CREATES it, and
        // what it creates is a cursor-only phantom that makes the checkpoint
        // loop WARN every tick and `diff_read` default fields until the Valkey
        // orphan sweep reaps it minutes later. `DeleteLayer` is a sequence of
        // commands, not a transaction, so only a server-side atomic conditional
        // write can close it. See `scripts/stamp_cursor_if_live.lua`.
        //
        // With a cursor-regression floor at our own target: the script ALSO
        // refuses when `last_checkpoint_seq` is already at or above it, in the
        // same atomic step as the write. That is guard (b)'s seq half made
        // atomic. The `HMGET` above still answers it in the common case (and
        // is what the bump half reads), but a read followed by a write leaves
        // a window in which two overlapping runs for ONE object both pass the
        // read and both stamp, both counting a persisted generation. That
        // window was opened by the `CreateLayer` re-send re-spawn — before
        // it, no two `CreationBase` runs could target one object — and the
        // floor closes it: the second run's stamp finds the first's cursor and
        // is refused.
        // Test-only: the read-then-stamp window the floor closes. See
        // `with_pre_stamp_hook_for_test`.
        #[cfg(any(test, feature = "test-checkpoint-fault-injection"))]
        if let Some(hook) = self.pre_stamp_hook_for_test.clone() {
            hook().await;
        }
        let stamped_ms = now_unix_ms();
        let stamp = crate::storage::lua::stamp_obj_header_cursor_below(
            self.client.pool.next(),
            &header_key,
            &target.cursor_pairs(stamped_ms),
            target_snap_seq,
        )
        .await;

        match stamp {
            Ok(crate::storage::lua::CursorStamp::Stamped) => {}
            Ok(crate::storage::lua::CursorStamp::CursorAtOrAbove) => {
                // The atomic twin of guard (b)'s seq half above: a generation
                // at or above our target was stamped between that read and
                // this stamp — on the creation site, typically a second run
                // for the same object. Same arm, same cell, same posture: the
                // blob is durable at the target and covered by the generation
                // that overtook us, and no `writes_total` tick on either axis.
                tracing::debug!(
                    object_id,
                    target = target.label(),
                    snap_seq = target_snap_seq,
                    "base bootstrap: a checkpoint generation at or above this target was stamped \
                     between the pre-stamp read and the stamp; leaving every cursor column \
                     untouched"
                );
                return BootstrapOutcome::UnstampedGenerationPresent;
            }
            Ok(crate::storage::lua::CursorStamp::HeaderAbsent) => {
                // The layer died while our PUT was in flight. Nothing was
                // written to the header — that is the whole point — so all
                // that is left is the blob we just made, which covers a layer
                // that no longer exists.
                //
                // ── Re-arm the dead ZSET FIRST, and unconditionally ────────
                //
                // This arm used to go straight to the inline reap and lean on
                // the delete path's own best-effort `ZADD NX` as its backstop
                // (`reap_dead_base`'s rustdoc asserted the member "is already in
                // place … whatever happens here"). That assertion was too
                // strong, and the counter-example is not exotic: the delete
                // path's `ZADD` is best-effort too, so one Valkey blip wide
                // enough to fail it can equally fail the tombstone-record step
                // of the reap below — and then the generation this call just
                // wrote is named by nothing, forever. Re-arming here is the
                // other half of the same defence the checkpoint loop's three
                // refused-stamp arms already carry (b.vn6 review round 2, C1).
                //
                // BEFORE the reap, not after: if this task dies between the two
                // the member is armed and the reconciliation pass finishes the
                // job, whereas the opposite order leaves a window in which
                // nothing names the blob.
                //
                // UNCONDITIONALLY, not just on `reaped == false`. `ZADD NX` is
                // idempotent, so the cost on the common path is one round-trip
                // on an arm that is already abandoning this layer; and
                // `reaped == true` is NOT evidence that re-arming was
                // unnecessary, because `reap_dead_base` deletes exactly the one
                // generation this call wrote. A `RestoreRebase` target can sit
                // on an object that carries older generations, and those are
                // reclaimed only by the dead-object phase — which needs the
                // member.
                //
                // The re-arm is best-effort in its own right (it WARNs and
                // returns), so it cannot turn a routine race into a failure.
                super::checkpointer::register_dead_object_after_refused_stamp(
                    &self.client,
                    &self.keys,
                    object_id,
                    &self.creator_replica_id,
                )
                .await;
                let reaped = self.reap_dead_base(object_id, &handle).await;
                // Level follows the reap result, not the refusal (b.vn6 review
                // round 1). The refusal itself is a routine race and is fully
                // cleaned up when `reaped = true` — nothing is left for an
                // operator to do, so INFO, matching the checkpoint loop's three
                // refused-stamp arms. `reaped = false` is the case the metric's
                // own rustdoc calls out as worth following up: the inline
                // cleanup failed and reclamation is deferred to the
                // reconciliation pass, so it earns a WARN.
                //
                // The two arms carry byte-identical message text, deliberately:
                // a log query written against one must find the other, and the
                // `reaped` field is what distinguishes them.
                const REFUSED_STAMP_MSG: &str =
                    "base bootstrap: obj:hdr was gone by the cursor stamp (layer deleted or \
                     overwritten while the base PUT was in flight), so the stamp was refused \
                     rather than resurrecting the header as a cursor-only phantom. The base \
                     blob covers a dead layer and was reaped inline when `reaped=true`; when \
                     false the reconciliation pass's dead-object phase reclaims it";
                if reaped {
                    tracing::info!(
                        object_id,
                        target = target.label(),
                        snap_seq = target_snap_seq,
                        reaped,
                        adapter_key = %handle.adapter_key,
                        "{REFUSED_STAMP_MSG}"
                    );
                } else {
                    tracing::warn!(
                        object_id,
                        target = target.label(),
                        snap_seq = target_snap_seq,
                        reaped,
                        adapter_key = %handle.adapter_key,
                        "{REFUSED_STAMP_MSG}"
                    );
                }
                // No `writes_total` tick on EITHER axis. `{success, layer}`
                // would be false — no generation was completed — and
                // `{error, layer}` would be false too: nothing failed, the
                // layer simply stopped existing. Same posture as the two
                // guard-(b) skips above, which also leave that family alone.
                return BootstrapOutcome::UnstampedHeaderAbsent;
            }
            Err(e) => {
                FredErrOrigin::client(&self.client, POOL_LABEL_DATA).observe_error(&e);
                // The durable base is on the adapter but Valkey does not know
                // it. Nothing is lost: every later reader (the periodic loop
                // included) sees a layer with un-persisted commits and
                // re-snapshots it on the next eligible tick. Counted as an
                // error so the divergence is visible.
                tracing::warn!(
                    object_id,
                    target = target.label(),
                    snap_seq = target_snap_seq,
                    error = %e,
                    "base bootstrap: the conditional obj:hdr cursor stamp failed; \
                     base blob landed but the cursor did not, so the next eligible tick \
                     re-snapshots this layer"
                );
                self.metrics.inc_checkpoint_write("error", "layer");
                // Same cell as the pre-stamp HMGET failure above, and for the
                // same reason: a durable base with an un-stamped header,
                // whichever of the two commands failed.
                return BootstrapOutcome::UnstampedWriteError;
            }
        }

        tracing::info!(
            object_id,
            target = target.label(),
            snap_seq = target_snap_seq,
            ext_hash_count,
            ompf_byte_len,
            sidecar_byte_len,
            adapter_key = %handle.adapter_key,
            "base bootstrap: base persisted and cursor stamped"
        );
        self.metrics.inc_checkpoint_write("success", "layer");
        BootstrapOutcome::Persisted {
            snap_seq: target_snap_seq,
        }
    }

    /// Fast-path reap of the base this invocation just wrote, after the
    /// conditional cursor stamp proved the layer is dead (b.vn6).
    ///
    /// Returns whether the generation is gone — the `reaped` field on the
    /// caller's WARN.
    ///
    /// # Scope: the just-written handle, and nothing else
    ///
    /// This deletes ONE generation, the one this call site created. It is not a
    /// per-object reclamation and must not become one: the object may carry
    /// older generations, and enumerating and deleting those is the
    /// reconciliation pass's dead-object phase — which runs under the
    /// checkpoint leader lease, re-proves it before every destructive step, and
    /// re-checks liveness. This path holds no lease and is not entitled to
    /// delete anything it did not itself write.
    ///
    /// # The ordering is mandatory
    ///
    /// Sidecar read → tombstone record → delete, exactly as the reconciliation
    /// pass's orphan arm does it. The `ext-refs.json` sidecar is the ONLY
    /// durable record of which ext blobs this generation references; deleting
    /// the generation first strands every mirrored blob it named, permanently,
    /// because nothing else will ever know to look for them.
    ///
    /// # Best-effort, and why that is safe
    ///
    /// Every failure arm WARNs and gives up. The correctness backstop is the
    /// dead-object phase of the reconciliation pass, which will enumerate and
    /// reap this generation — sidecar-first, under the lease — whatever happens
    /// here. This path exists to close the window in minutes rather than in up
    /// to one reconcile interval, not to be the thing that closes it.
    ///
    /// That backstop rests on `le:ckpt:{recon}:dead` naming this `object_id`,
    /// and this rustdoc used to assert it already did — the delete path that
    /// killed the layer registers it before returning, so "the member is already
    /// in place whatever happens here". That is one `ZADD NX` too few (b.vn6
    /// review round 2, C1): the delete path's registration is best-effort, and a
    /// Valkey blip wide enough to fail it can equally fail this function's own
    /// tombstone record, leaving the generation named by nothing. The caller
    /// therefore re-arms the member itself, unconditionally, immediately before
    /// calling this — see the comment at that call site. The claim here is now
    /// the weaker and true one: *by the time this runs, the caller has made its
    /// own attempt to put the member in place.*
    async fn reap_dead_base(
        &self,
        object_id: u64,
        handle: &super::store::CheckpointHandle,
    ) -> bool {
        let hashes = match self.store.read_ext_refs_sidecar(handle).await {
            Ok(set) => set,
            Err(e) => {
                tracing::warn!(
                    object_id,
                    snap_seq = handle.snap_seq,
                    error = %e,
                    "base bootstrap: could not read the ext-refs sidecar of the base it just \
                     wrote for a since-deleted layer, so the base is left in place rather than \
                     stranding the ext blobs it names; the reconciliation pass's dead-object \
                     phase reclaims it"
                );
                return false;
            }
        };

        let hash_count = hashes.len();
        if let Err(e) = self
            .tombstone_sink()
            .record(hashes.into_iter().collect::<Vec<String>>())
            .await
        {
            tracing::warn!(
                object_id,
                snap_seq = handle.snap_seq,
                hash_count,
                error = %e,
                "base bootstrap: could not tombstone the ext hashes of the base it just wrote \
                 for a since-deleted layer; leaving the base in place (deleting it now would \
                 strand those blobs). The reconciliation pass's dead-object phase reclaims it"
            );
            return false;
        }

        match self.store.delete_layer_checkpoint(handle.clone()).await {
            Ok(()) => true,
            Err(e) => {
                tracing::warn!(
                    object_id,
                    snap_seq = handle.snap_seq,
                    error = %e,
                    "base bootstrap: could not delete the base it just wrote for a \
                     since-deleted layer; its ext hashes are already tombstoned so nothing \
                     leaks meanwhile, and the reconciliation pass's dead-object phase \
                     reclaims the generation"
                );
                false
            }
        }
    }

    /// The shared ext-blob tombstone carrier, bound to this unit's client and
    /// key schema.
    ///
    /// Built per use rather than held as a field: `ExtTombstoneSink` borrows its
    /// client and metrics, so storing one would make [`BaseBootstrap`]
    /// self-referential for a struct that is three `String`/reference fields
    /// and a ZSET key format. The reconciliation pass constructs it the same
    /// way, once per pass.
    fn tombstone_sink(&self) -> super::checkpointer::ExtTombstoneSink<'_> {
        super::checkpointer::ExtTombstoneSink::new(
            &self.client,
            &self.keys,
            Some(&self.metrics),
            &self.creator_replica_id,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::observability::metrics::CHECKPOINT_BASE_BOOTSTRAP_SITES;
    use std::collections::HashSet;

    /// Every `BootstrapOutcome` labels itself with a member of the pinned metric
    /// vocabulary, and the nine arms map onto nine DISTINCT members.
    ///
    /// Sibling of `tail_capture`'s
    /// `every_outcome_label_is_a_member_of_the_pinned_metric_vocabulary`, and it
    /// covers what `BOOTSTRAP_OUTCOME_LABEL_PINS` deliberately cannot. That
    /// const-eval block pins index → spelling, so a *reordering* of
    /// `CHECKPOINT_BASE_BOOTSTRAP_OUTCOMES` is a compile error. What it says
    /// nothing about is a variant added later and pointed at an index that is
    /// already spoken for: every arm still compiles, every label is still a
    /// member, and the pins still hold — while two code arms now share one cell
    /// and the vocabulary has silently stopped being 1:1 with the enum. The
    /// distinctness assertion below is the only thing that fails on that.
    ///
    /// The membership half is the other direction and is not redundant with the
    /// indexing either: an arm could be re-written to return a literal rather
    /// than index the constant, which compiles and opens an unseeded series.
    #[test]
    fn every_bootstrap_outcome_label_is_a_distinct_member_of_the_pinned_vocabulary() {
        let outcomes = [
            BootstrapOutcome::Persisted { snap_seq: 1 },
            BootstrapOutcome::Shed,
            BootstrapOutcome::SkippedHeaderAbsent,
            BootstrapOutcome::SkippedGenerationPresent,
            BootstrapOutcome::SkippedUntrustedCursor,
            BootstrapOutcome::UnstampedHeaderAbsent,
            BootstrapOutcome::UnstampedGenerationPresent,
            BootstrapOutcome::UnstampedForeignRestore,
            BootstrapOutcome::UnstampedWriteError,
            BootstrapOutcome::Failed,
        ];

        for outcome in &outcomes {
            let label = outcome.outcome_label();
            assert!(
                CHECKPOINT_BASE_BOOTSTRAP_OUTCOMES.contains(&label),
                "{outcome:?} labels itself `{label}`, which is not a member of \
                 CHECKPOINT_BASE_BOOTSTRAP_OUTCOMES — the metric cell would never be seeded, so \
                 an arm that is supposed to stay silent becomes indistinguishable from a \
                 producer nobody wired"
            );
        }

        let distinct: HashSet<&str> = outcomes.iter().map(|o| o.outcome_label()).collect();
        assert_eq!(
            distinct.len(),
            outcomes.len(),
            "the enum is 1:1 with the vocabulary: two variants sharing one label would hide a \
             distinction the vocabulary paid a series for, and would break the per-arm \
             attribution `sum by (outcome)` is read for. Got {} distinct labels across {} \
             variants: {distinct:?}",
            distinct.len(),
            outcomes.len()
        );
        assert_eq!(
            distinct.len(),
            CHECKPOINT_BASE_BOOTSTRAP_OUTCOMES.len(),
            "and 1:1 in the other direction too — a vocabulary member no variant mints is a cell \
             that can only ever read zero, which is exactly the shape of a producer that was \
             removed and left its dashboard behind"
        );
    }

    /// Both `BootstrapTarget`s label themselves with a member of the pinned
    /// `site` vocabulary, and the two are distinct.
    ///
    /// Sibling of the outcome-axis test above, and it covers what
    /// `BOOTSTRAP_SITE_LABEL_PINS` deliberately cannot. Since `label()` indexes
    /// the constant, a *re-spelling or reordering* is already a compile error at
    /// that const block; what stays invisible there is two arms pointed at the
    /// same index — both labels would still be members, and the two call sites
    /// would collapse onto one cell. The distinctness assertion below is the only
    /// thing that fails on that. The same string is the `target` field on every
    /// log line this unit emits, so the collapse would take the cell-to-log join
    /// with it.
    #[test]
    fn every_bootstrap_target_label_is_a_distinct_member_of_the_pinned_site_vocabulary() {
        let targets = [
            BootstrapTarget::CreationBase,
            BootstrapTarget::RestoreRebase {
                seeded_seq: 40,
                observed_commit_span: 3,
                absorbed_band: 1000,
            },
        ];

        for target in targets {
            let label = target.label();
            assert!(
                CHECKPOINT_BASE_BOOTSTRAP_SITES.contains(&label),
                "{target:?} labels itself `{label}`, which is not a member of \
                 CHECKPOINT_BASE_BOOTSTRAP_SITES — `run_once` passes this straight into \
                 `inc_checkpoint_base_bootstrap`, so it would open an unseeded series"
            );
        }

        let distinct: HashSet<&str> = targets.iter().map(|t| t.label()).collect();
        assert_eq!(
            distinct.len(),
            CHECKPOINT_BASE_BOOTSTRAP_SITES.len(),
            "the two call sites must stay separable: `shed` at `restore_rebase` is the alertable \
             mass-restore signature, and collapsing the sites onto one label would bury it in \
             creation-time traffic. Got: {distinct:?}"
        );
    }

    /// The guard generalization is a REFACTOR at the creation site, and this is
    /// what pins that.
    ///
    /// Guard (a) and guard (b) both used to test `observed != 0`; they now test
    /// `observed >= target.snap_seq()`. Those are the same predicate exactly
    /// when the target is `1`, which is what `BootstrapTarget::CreationBase`
    /// always is. If a future edit changes either side — the creation target
    /// away from `1`, or the comparison to `>` — this fails rather than
    /// silently changing when the creation-time bootstrap declines to write.
    #[test]
    fn cursor_regression_guard_is_the_nonzero_test_at_the_creation_target() {
        let creation_target = BootstrapTarget::CreationBase.snap_seq();
        assert_eq!(
            creation_target, 1,
            "the equivalence below holds only at target 1"
        );

        for observed in [0u64, 1, 2, 3, 7, 4096, u64::MAX] {
            assert_eq!(
                cursor_regression_skips(observed, creation_target),
                observed != 0,
                "guard (a) at the creation target must be exactly the pre-existing \
                 `last_checkpoint_seq != 0` test (observed = {observed})"
            );
            assert_eq!(
                raw_cursor_blocks_stamp(Some(&observed.to_string()), creation_target),
                observed != 0,
                "guard (b) at the creation target must be exactly the pre-existing \
                 `parse != 0` test (observed = {observed})"
            );
        }

        // The two non-numeric readings guard (b) has always had, unchanged.
        assert!(
            !raw_cursor_blocks_stamp(None, creation_target),
            "an absent column is a truthful 0 and must still permit the stamp"
        );
        assert!(
            raw_cursor_blocks_stamp(Some("garbage"), creation_target),
            "an unparseable column must still count as present (conservative direction)"
        );
    }

    /// The restore site's target is the seeded cursor, and the restore's OWN
    /// audit stamp sits strictly below it — so the guard must NOT fire on it.
    /// This is the case the un-generalized `!= 0` form got wrong for every
    /// re-base.
    #[test]
    fn cursor_regression_guard_admits_the_restores_own_audit_stamp() {
        // A restore stamps `last_checkpoint_seq = snap_seq` while seeding
        // `seq = restore_floor + band`, and `band >= 1` unconditionally.
        let snap_seq = 40u64;
        let seeded_seq = 91u64;
        let target = BootstrapTarget::RestoreRebase {
            seeded_seq,
            observed_commit_span: 7,
            absorbed_band: 50,
        };
        assert_eq!(target.snap_seq(), seeded_seq);

        assert!(
            !cursor_regression_skips(snap_seq, target.snap_seq()),
            "the restore's own `last_checkpoint_seq` stamp must not abort its re-base"
        );
        assert!(
            cursor_regression_skips(seeded_seq, target.snap_seq()),
            "a generation AT the seeded cursor means the re-base already landed"
        );
        assert!(
            cursor_regression_skips(seeded_seq + 1, target.snap_seq()),
            "a generation above the seeded cursor must abort the re-base"
        );
        assert!(
            !cursor_regression_skips(0, target.snap_seq()),
            "a never-checkpointed header cannot block a re-base either"
        );
    }

    /// C6: the bump half of guard (b) permits the stamp for exactly one
    /// observed value — the band this re-base absorbed.
    ///
    /// This is also the fence on the guard's accepted residual, and the only
    /// one there can be. Epic 5's SR-7 adjudication (Task 5 `t2.dfh.be.s8`,
    /// 2026-09-01) ruled the same-band collision **accepted-and-documented**
    /// (log-only, no cell of its own), with
    /// `live_edit_checkpoint_base_bootstrap_total{outcome="unstamped_foreign_restore"}`
    /// — the complement, i.e. every time this guard BLOCKS — standing as its
    /// risk indicator: non-zero says the interleaving that makes the residual
    /// possible is live in this fleet. The accepted collision is precisely what
    /// the `Some("1000") ⇒ permits` case below encodes: at the decision point a
    /// peer restore that absorbed the same band is byte-identical in every
    /// column read, so permitting it is indistinguishable from permitting our
    /// own, and no assertion here (or counter in production) could separate the
    /// two — see [`raw_bump_blocks_stamp`]'s "# The residual it accepts". What
    /// the five blocking cases pin is that the permit is granted for that ONE
    /// reading and nothing else, which is the whole of the guard's contract.
    #[test]
    fn bump_guard_permits_only_the_absorbed_band() {
        let absorbed_band = 1000u64;

        assert!(
            !raw_bump_blocks_stamp(Some("1000"), absorbed_band),
            "our own restore's bump is the one value that may be cleared"
        );
        assert!(
            raw_bump_blocks_stamp(Some("999"), absorbed_band),
            "a different band means a different restore owns this header"
        );
        assert!(
            raw_bump_blocks_stamp(Some("1001"), absorbed_band),
            "a different band means a different restore owns this header"
        );
        assert!(
            raw_bump_blocks_stamp(Some("0"), absorbed_band),
            "an already-retired bump is not ours to retire again"
        );
        assert!(
            raw_bump_blocks_stamp(None, absorbed_band),
            "an absent column reads as 0, and `band >= 1` always"
        );
        assert!(
            raw_bump_blocks_stamp(Some("garbage"), absorbed_band),
            "unparseable blocks, matching the seq half's conservatism"
        );
    }

    /// The stamp is TWO pairs at the creation site and THREE at the restore
    /// site, and the third is the retirement of `restore_seq_bump`.
    #[test]
    fn cursor_pairs_are_two_at_creation_and_three_on_a_rebase() {
        let creation = BootstrapTarget::CreationBase.cursor_pairs(1_700_000_000_000);
        assert_eq!(
            creation,
            vec![
                ("last_checkpoint_seq", "1".to_string()),
                ("last_checkpoint_ms", "1700000000000".to_string()),
            ],
            "the creation site must not assert that a restore happened"
        );

        let rebase = BootstrapTarget::RestoreRebase {
            seeded_seq: 91,
            observed_commit_span: 7,
            absorbed_band: 50,
        }
        .cursor_pairs(1_700_000_000_000);
        assert_eq!(
            rebase,
            vec![
                ("last_checkpoint_seq", "91".to_string()),
                ("last_checkpoint_ms", "1700000000000".to_string()),
                ("restore_seq_bump", "0".to_string()),
            ],
            "retiring the bump is what unblocks tail capture after a restore"
        );
    }

    /// The meta span is the sentinel at creation and the carried-forward
    /// measurement on a re-base — never the literal zero-new-commits reading.
    #[test]
    fn observed_commit_span_is_carried_forward_on_a_rebase() {
        assert_eq!(
            BootstrapTarget::CreationBase.observed_commit_span(),
            NO_COMMIT_SPAN_MEASUREMENT
        );
        assert_eq!(
            BootstrapTarget::RestoreRebase {
                seeded_seq: 91,
                observed_commit_span: 7,
                absorbed_band: 50,
            }
            .observed_commit_span(),
            7,
            "stamping the sentinel would degrade the NEXT restore of this layer back to \
             the configured margin"
        );
    }
}
