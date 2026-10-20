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

//! Lineage-unit retention planning — the **pure, fetch-free decision half** of
//! SR-5.1 / SR-5.2.
//!
//! [`plan_retention`] takes a `list_layer_checkpoints` result and `retain`, and
//! returns the [`RetentionPlan`] the post-write prune executes: which lineage
//! units survive, which are deleted whole (tails first, base last), and which
//! stray tails are orphans. It performs no `await`, holds no store, reads no
//! [`CheckpointMeta`](super::store::CheckpointMeta), and fetches no blob.
//!
//! Nothing in this module changes behaviour on its own — the executor is
//! `checkpointer::prune_old_generations`.
//!
//! # Why fetch-free (the SR-4.2 cost model)
//!
//! The obvious grouping is the one Epic 4's gather uses: walk each tail's
//! `(base_snap_seq, from_seq, to_seq)` links. That is not available here. A
//! tail's header numbers live **inside the envelope bytes** —
//! `lineage::LineageSelection` gets them only by fetching the blob and calling
//! `peek_tail_header`, and there is no meta-only read on
//! [`CheckpointStore`](super::store::CheckpointStore). A link-walk would
//! therefore cost up to `1 base + K tails` multi-MB fetches on **every tick
//! that writes a generation**, against SR-4.2's per-tick budget of one listing,
//! one bounded `XRANGE`, and one small PUT, all inside the leader lease.
//!
//! The other candidate input — tail `CheckpointMeta` — is forbidden outright;
//! see *Meta is not keyed on* below.
//!
//! So retention groups by **adjacency over `(kind, sequence)`**: exactly the two
//! fields [`CheckpointHandle`] already carries.
//!
//! # The grouping, and why it is correct
//!
//! Scan the listing newest-first. Accumulate `Tail` handles; a `Base` **closes**
//! the accumulation into one [`LineageUnit`] rooted at that base. Handles still
//! accumulated when the scan ends — the tails below the *oldest* base — are the
//! orphan bucket.
//!
//! **The listing is consumed in the order it arrives and is never re-sorted.**
//! `list_layer_checkpoints` returns one strict descending order over the
//! generation sequence with `Base` pinned before `Tail` at an equal sequence,
//! produced by every adapter through the shared
//! [`handle_generation_sort_key`](super::adapters::handle_generation_sort_key).
//! Re-deriving that order here is precisely the drift centralizing the key
//! exists to prevent. (A `debug_assert!` checks the precondition rather than
//! restoring it — see [`plan_retention`].)
//!
//! ## The adjacency argument
//!
//! A tail's own header satisfies `base_snap_seq <= from_seq <= to_seq`, enforced
//! by `tail_codec` at **encode** (`InvalidRange`, `BaseAfterRange`) and again at
//! decode. A handle's ordering number is the base's `snap_seq` or the tail's
//! `to_seq`, so a member tail of the unit rooted at `base@B` has
//! `to_seq >= base_snap_seq == B`, i.e. it sorts at or above `base@B`; and every
//! member of the same unit shares that root, so the base is the **minimum
//! sequence of its unit** and always terminates its run. Scanning newest-first
//! and closing at each `Base` therefore reproduces the unit boundaries.
//!
//! **The re-base tie, named explicitly.** SR-4.1's chain-collapse re-base mints
//! a new `Base` at `S`, where `S` is the surviving head tail's `to_seq` — so
//! `base@S` and `tail@S` genuinely coexist, and the pinned tie-break sorts
//! `base@S` *before* `tail@S`. The scan closes an **empty** accumulation at the
//! new chain-0 base, and hands the surviving old head `tail@S` down to the older
//! unit it actually belongs to. That is exactly the artifact shape Task 3
//! (`t2.dfh.be.84`) records: the re-base deletes nothing, and the superseded
//! lineage stays complete and restorable until retention takes it as a unit.
//!
//! ## Producer agreement — this is not an approximation of the links
//!
//! Adjacency grouping reproduces the **writer's own base attribution**.
//! `tail_capture::lineage_shape` derives a new tail's `base_snap_seq` as
//! `handles[base_index].seq()`, where `base_index = position(kind == Base)` —
//! the newest base in the listing at capture time. So for every tail the capture
//! path produces, *"the base directly below me in the listing"* **is** *"the base
//! I recorded in my header"*. Grouping and attribution agree by construction;
//! the only way they diverge is a base published between a capture's listing and
//! its PUT — the duplicate-leader race recorded as the cross-jump residual
//! below.
//!
//! `lineage_shape`'s own *"the chain count is a documented conservative
//! over-approximation"* block is the sibling decision, taken for the identical
//! SR-4.2 cost reason: listing position instead of a link-walk, erring in the
//! conservative direction. This module is the second instance of that trade, not
//! a new one.
//!
//! ## The strictness precondition — the zero-width tail
//!
//! The step "a member tail sorts at or above its base" is only enough because,
//! at an *equal* sequence, the Base-before-Tail tie-break puts the base first.
//! For a tail to sort **above** its own base it needs `to_seq > base_snap_seq`
//! **strictly**. `tail_codec::encode_tail_envelope` enforces only
//! `from_seq <= to_seq` and `base_snap_seq <= from_seq`, so a **zero-width tail**
//! (`base_snap_seq == from_seq == to_seq == S`) is a legal encoding — and it
//! would sort *below* `base@S` and be swept into the next-older unit (or into
//! the orphan bucket), falsifying the argument above. It is the one shape that
//! does.
//!
//! Three things close that gap:
//!
//! 1. **The capture path never emits one.** A capture requires a non-empty
//!    replayable prefix — `tail_classify` yields a slice of at least one commit
//!    — so every tail this service writes has `to_seq > from_seq >= base_snap_seq`.
//!    The re-base path composes strictly forward for the same reason. No shipped
//!    writer can produce `to_seq == base_snap_seq`.
//! 2. **Its mis-grouping is provably harmless**, which is why this module does
//!    not act conservatively on it (see [`RetentionPlan::base_tied_tails`] for
//!    the full argument, including why a conservative arm would be *worse*).
//! 3. **The observable is recorded** in [`RetentionPlan::base_tied_tails`] so a
//!    future writer that starts emitting one leaves a number behind, and the
//!    in-file `a_sequence_tied_tail_is_counted_and_grouped_below_its_tied_base`
//!    case pins today's behaviour rather than leaving it to be discovered.
//!
//! # The recorded residual
//!
//! ## The grouping is a conservative SUPERSET of the gather's link-walk
//!
//! A unit may hold tails that the gather's walk would not visit — SR-9.2's
//! divergent-`S` duplicate-leader fork residue, which overlaps a real member's
//! range instead of linking behind it. Inside a **retained** unit that residue is
//! retained as if it were a member (storage cost only), and it is reaped when the
//! unit is pruned whole.
//!
//! **Superset safety, stated because a reader will otherwise assume the superset
//! is what creates the risk.** Supersetting is monotonically safe: a unit retains
//! *at least* every member of the gather's walked chain that lies within it, so a
//! superset can never cause the gather to want a deleted artifact. The only way
//! the gather can want a deleted artifact is if its chosen lineage extends
//! **below the newest base** — the cross-jump case below. The superset and the
//! hole are independent.
//!
//! ## The one hole — a chain that jumps over an intervening base
//!
//! Producible only by a duplicate-leader publish race that lands a base between a
//! capture's listing and its PUT (`lineage_shape` roots every tail at the newest
//! base as of its *own* listing, and SR-1.2 makes `S0` the newest restorable
//! generation per that listing).
//!
//! Worked example. Listing `[tail@7, base@5, tail@4, base@3]` with
//! `tail@7.base_snap_seq == 3`. The gather walks `tail@7 → seq 4 → tail@4 →
//! seq 3 → base@3`, spanning two adjacency units. At `retain = 1` this planner
//! keeps `{base@5, tail@7}` and prunes `{base@3, tail@4}`. The gather then hits
//! `LineageBreak::LinkAbsent` at seq 4 and steps down to `base@5`.
//!
//! **The provable bound:** the recovery point never falls below the **newest
//! base's sequence** — SR-2.5's own floor ("a store holding tails never regresses
//! recovery below what the newest base provides") — and never to an unrestorable
//! artifact. Both halves matter: the base is newer (7 → 5 is a real
//! recovery-point regression), and the coverage is older. It is the same
//! direction SR-5.2 already accepts for a crash mid-prune, and it is pinned by an
//! executable case rather than asserted absent (owner: `t3.dfh.be.mk.ah`).
//!
//! ## Residue bound — the arithmetic, not an adjective
//!
//! Fork residue is *bounded*, and the numbers are what make "storage cost only"
//! an answer rather than a hope — the Task-3 PM review found duplicate-leader
//! forks are **steady-state** for large layers, not exceptional.
//!
//! - **Lifetime** = `retain` × the re-base period. At stock knobs
//!   (`retain = 2`, `K = 8`, `interval = 60 s`) that is ≈ 16 minutes: residue
//!   inside a unit dies when the unit is pruned whole.
//! - **Worst-case volume**, with duplicate leadership on every eligible tick:
//!   ≤ `retain × K` ≈ 16 extra tail artifacts per unit, each individually bounded
//!   by SR-1.4's caps.
//! - **Self-limiting.** Residue inflates the *listing-position* chain count that
//!   `lineage_shape` returns, so the layer reaches `K` **earlier**, re-bases
//!   **sooner**, and rotates the unit out **sooner**. More residue shortens its
//!   own lifetime.
//!
//! # The third base producer — the `retain` arithmetic is deliberately unadjusted
//!
//! `retain_generations` counts **lineage units**, and units are in bijection
//! with bases (see `RetentionPlan::retained_units`). So every base minted by
//! anything, for any reason, consumes exactly one retain slot. Three producers
//! mint one outside the ordinary verified-snapshot arm, and the binding
//! 2026-08-31 deferral asks that their effect be a decision rather than an
//! accident:
//!
//! 1. **SR-7.1 creation base** — `base_bootstrap`'s
//!    `BootstrapTarget::CreationBase`, at `snap_seq = 1`. It is a layer's
//!    *first* base; there is no older lineage for it to displace, so it is
//!    listed for completeness and does not participate in the argument below.
//! 2. **SR-4.1 chain-collapse re-base** — `rebase::rebase_layer_lineage`, a new
//!    base at `S` = the surviving head tail's `to_seq`.
//! 3. **SR-3.6 post-restore re-base** — `base_bootstrap`'s
//!    `BootstrapTarget::RestoreRebase`, a lone base at
//!    `seeded_seq = restore_floor + band` (`band >= 1` unconditionally, so it
//!    sorts strictly above every generation already in the store).
//!
//! **The arithmetic is NOT adjusted, and this is the reason.** Each of (2) and
//! (3) is a *re-materialization of the coverage of the lineage it displaces*,
//! built from the composed or restored state — not an unrelated new generation
//! competing for the same window. (2) composes `base@B + tails` into `base@S`,
//! so `base@S` carries everything the superseded unit carried. (3) writes a
//! reset OMPF of exactly the state `apply_restore` committed, which is the
//! composed state of the newest restorable lineage, so it carries everything
//! *that* lineage carried. In both cases the lineage pushed out of the far end
//! of the window therefore holds only **strictly older** coverage — which
//! retention has always been permitted to drop. Spending a slot on a base that
//! subsumes the window's newest coverage is the arithmetic working, not a leak
//! in it.
//!
//! **(2) additionally frees the slot it spends.** Because `base@S` ties in
//! sequence with the surviving head `tail@S` and sorts Base-first, the scan
//! closes an **empty** (chain-0) unit at the new base and hands `tail@S` down to
//! the superseded unit — so the collapse consumes one slot *and* makes the
//! displaced unit whole and prunable in the same listing. Before the collapse
//! that unit's head tail was the newest generation and the unit could not be
//! pruned without losing the store's newest coverage; after it, it can.
//!
//! **The observable consequence, stated for an operator rather than implied.**
//! At the default `retain_generations = 2`, a DR restore immediately demotes the
//! pre-restore lineage to the **last** retained slot: the store goes from
//! `[L1, L2]` to `[R, L1]`, and the next post-write prune deletes `L2` whole.
//! Nothing restorable to a *newer* point is lost — `R` subsumes `L1`, which
//! subsumed `L2` — but the number of distinct *historical* recovery points does
//! fall by one for one generation-period after every restore. An operator who
//! needs `N` pre-restore points to survive a restore sets `retain_generations`
//! to `N + 1`; the knob is where that preference belongs.
//!
//! **Why an adjusted count is not merely unnecessary but unavailable here.** Any
//! "do not count restore-minted bases" rule needs to tell a restore-minted base
//! from an ordinary one, and nothing durable records that. A
//! [`CheckpointHandle`] carries no provenance — `(kind, sequence)` is congruent
//! across all three producers — and `CheckpointMeta` has no producer
//! discriminator either; `BootstrapTarget::label` (`base_bootstrap.rs:315-321`)
//! is a *structured-log* discriminator, not a persisted field. So an adjusted
//! arithmetic would need a new durable field on the tail-and-base meta **and** a
//! retention read of it — exactly the input *Meta is not keyed on* forbids — for
//! a correction the paragraphs above show is not owed.
//!
//! # Meta is not keyed on (D8) — a structural invariant, not a preference
//!
//! This function reads `handle.kind`, `handle.seq()` **and nothing else**. No
//! `CheckpointMeta`, no envelope header, no `read_ext_refs_sidecar`, no
//! `peek_tail_header`.
//!
//! A future reader must not "improve" the grouping by reading tail meta. The
//! binding 2026-08-28 Task-4 PM deferral forbids keying **any** destructive
//! decision on tail-meta presence or contents, because a duplicate leader whose
//! envelope PUT fails runs an unguarded best-effort DELETE of the tail meta —
//! including the meta of an **already-durable, byte-identical** envelope written
//! by the winner. Meta absence therefore does not mean "artifact absent", and a
//! retention rule that read it would delete live artifacts under an ordinary
//! duplicate-leader race. The envelope header is the authoritative lineage
//! record, and it costs a blob fetch, which is what closes the loop back to
//! *Why fetch-free* above.
//!
//! ## The verification — all three adapters, as citations
//!
//! "This module reads no meta" is only half the property. The other half is a
//! claim about the *input*: `handle.kind` must itself be minted from something a
//! lost or forged meta cannot move — otherwise meta would be keyed on
//! indirectly, one call frame further out. That is a property of the three
//! `list_layer_checkpoints` implementations, not of this file, so it is verified
//! against each of them rather than assumed. Cited rather than re-derived: a
//! citation stays checkable against a moving tree, a paraphrase rots.
//!
//! **(a) This planner reads `handle.kind` and `handle.seq()` and nothing else.**
//! `plan_retention`'s body is one `match handle.kind` plus one `handle.seq()`
//! comparison; `CheckpointHandle`'s other fields (`adapter_key`, `object_id`)
//! never reach a decision.
//!
//! **(b) The executor reads a sidecar for BASE handles only.**
//! `checkpointer::prune_old_generations` step 1 calls `read_ext_refs_sidecar`
//! on `unit.base` and on no other handle, so a tail handle cannot reach that
//! read from retention at all. Belt and braces below it, all three adapters
//! short-circuit the read on `handle.kind == Tail` before any lookup —
//! `in_memory.rs:680`, `storage_api.rs:1846`, `local_disk.rs:1388` — each pinned
//! by its own test: `read_ext_refs_sidecar_on_a_tail_handle_returns_empty_and_not_the_bases_hashes`
//! (`in_memory.rs:2218`, `local_disk.rs:4351`) and
//! `read_ext_refs_sidecar_on_a_tail_handle_returns_empty_without_fetching_the_base`
//! (`storage_api.rs:5566`).
//!
//! **(c) Every adapter mints `kind` from the artifact STEM, never from meta.**
//!
//! - **StorageApi.** `list_layer_checkpoints` (`storage_api.rs:1287-1317`) maps
//!   `list_layer_generations_via_enumerate` (`:2954-2969`), which filters the raw
//!   address LIST through `parse_generation_filename` (`:789-794`): `.ompf` ⇒
//!   `Base`, `.tail` ⇒ `Tail`. Pinned by
//!   `parse_generation_filename_round_trips_the_address_builders_own_output`
//!   (`:7201`), which drives the *address builders' own output* back through the
//!   parser and additionally asserts that `.meta.json`, `.tail.meta.json` and
//!   `.ext-refs.json` enumerate as **nothing**.
//! - **LocalDisk.** `list_layer_checkpoints` (`local_disk.rs:1051-1081`) maps
//!   `list_generations_for_object` (`:449-474`), a `read_dir` walk filtered by
//!   `parse_generation_filename` (`:752-757`) — `parse_ompf_filename` ⇒ `Base`,
//!   `parse_tail_filename` ⇒ `Tail`, over disjoint suffixes whose disjointness is
//!   argued at `:701-712`. Pinned by
//!   `tail_stem_round_trips_through_its_parsers_and_never_aliases_the_ompf_family`
//!   (`:4437`), including "the tail/base meta companion must never enumerate as a
//!   generation".
//! - **InMemory.** `list_layer_checkpoints` (`in_memory.rs:440-469`) emits from
//!   the **slot field** the entry occupies — `slot.base.is_some()` /
//!   `slot.tail.is_some()` — chosen at PUT from `meta.kind` and never re-read
//!   from `entry.meta` afterwards. Stronger than the other two rather than
//!   weaker: `LayerEntry` holds the artifact and its meta as one value, so this
//!   adapter has no independently addressable meta and therefore no best-effort
//!   meta DELETE at all (contrast `storage_api.rs:1091`, `local_disk.rs:834`) —
//!   the D8 hazard has no in-memory analogue to guard against.
//!
//! In all three the relation runs **artifact ⇒ listing**: losing a meta cannot
//! remove a generation from the listing, cannot change its kind, and cannot
//! conjure one (a companion never parses as a generation). That is exactly the
//! property D8 needs, and it is why nothing downstream has to compensate.
//!
//! ## The durable-envelope probe is DECLINED (adjudication, 2026-09-02)
//!
//! The 2026-08-28 deferral offered two ways to satisfy itself: verify the design
//! keys on no tail meta, or add a **durable-envelope probe** — a read that
//! confirms an envelope is really there before trusting the artifact. The
//! verification above holds, so the probe is **not added**, and the reason is
//! structural rather than budgetary: *retention never needs the envelope, so
//! there is nothing to probe*. Every input it consumes is already the
//! listing's own evidence that the envelope exists (a stem enumerates because
//! the artifact is there), and the one fact an envelope read would add — the
//! `(base_snap_seq, from_seq, to_seq)` links — is the link-walk this module
//! declined on the SR-4.2 cost model. A probe would therefore buy a fetch per
//! artifact per tick and answer a question no arm here asks.
//!
//! # Deletion keying
//!
//! Every handle this plan hands out carries `object_id`, `seq()` and `kind`, and
//! the executor deletes **through the handle**. That is load-bearing, not
//! incidental: a *sequence-keyed* delete would remove the newly minted re-base
//! `base@S` while trying to delete the superseded `tail@S`. Lineage deletion is
//! keyed `(object_id, sequence, kind)`.

use crate::checkpoint::store::{CheckpointHandle, CheckpointKind};

/// One retention unit: a base and the tails rooted at it, as the listing's
/// adjacency implies.
///
/// `tails` is **newest-first**, in listing order — which is also the order the
/// executor must delete them in, so that a crash mid-prune never leaves a tail
/// whose predecessor is already gone.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct LineageUnit {
    /// The unit's root — the `Base` that closed the accumulation.
    pub(crate) base: CheckpointHandle,
    /// The unit's tails, newest-first. Empty for a chain-0 lineage (a lone base,
    /// including the artifact SR-4.1's re-base has just minted).
    pub(crate) tails: Vec<CheckpointHandle>,
}

impl LineageUnit {
    /// Total artifacts in the unit — one base plus its tails. The executor's
    /// pruned-generation count.
    pub(crate) fn generation_count(&self) -> usize {
        1 + self.tails.len()
    }
}

/// The decision [`plan_retention`] produces. Purely descriptive: it names
/// artifacts, it does not delete them.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub(crate) struct RetentionPlan {
    /// How many lineage units survive — `min(units, retain)`.
    ///
    /// **This number is identical to today's base count**, and that is a finding
    /// rather than a coincidence: every lineage has exactly one base by
    /// construction, so units and bases are in bijection. The *counting* half of
    /// retention therefore does not change at all when lineage-unit retention
    /// lands — only delete-set membership does. In particular the existing
    /// base-count decision is preserved here by construction, including its
    /// refusal to early-return on an **artifact** count: `handles.len() <= retain`
    /// is the wrong test (2 bases + 5 tails at `retain = 2` would fall through it
    /// and then prune a base the quota should have kept), and this planner never
    /// forms it.
    pub(crate) retained_units: usize,
    /// The units to delete, **oldest unit first**. Within each unit the tails are
    /// newest-first and the base is deleted last.
    pub(crate) prune: Vec<LineageUnit>,
    /// Tails below the **oldest** base: Epic-3 interim residue, and fork residue
    /// whose base has already been pruned. Never members of any unit, never
    /// consulted by the gather, structurally unable to strand a base — so they
    /// are deleted individually and they never count against `retain`.
    ///
    /// Listing order (newest-first) is preserved.
    ///
    /// **Task-5 observability input, recorded not minted.** Orphan-tail pruning
    /// is a *log-only* verdict: the executor reports `pruned_orphan_tails` /
    /// `deleted_orphan_tails` as fields on one terminal `info!` and ticks no
    /// counter. An operator watching for the SR-9.2 fork rate, or for Epic-3
    /// residue draining after an upgrade, therefore has to parse logs to see a
    /// rate that is otherwise a one-line counter — and cannot alert on it at all.
    /// Recorded here for Task 5's SR-7 adjudication (`t2.dfh.be.s8`); this Task
    /// mints no metric family, label value, or env knob.
    pub(crate) orphan_tails: Vec<CheckpointHandle>,
    /// The listing was **non-empty and carried no `Base` at all** — the anomaly
    /// `lineage::LineageSelection` reports as `Exhaustion::NoBase` and
    /// `tail_capture::lineage_shape` reports as `None`: "not producible by the
    /// shipped write paths — suspect manual pruning of the base or partial
    /// checkpoint-store loss."
    ///
    /// When set, [`prune`](Self::prune) and [`orphan_tails`](Self::orphan_tails)
    /// are both **empty**: see [`plan_retention`]'s base-less guard.
    ///
    /// An **empty** listing does not set this. It carries no base either, but its
    /// operator meaning is routine lifecycle (never checkpointed, or retention
    /// took the last generation), and the two are traced apart everywhere else in
    /// this subsystem — folding them together here would put a store-loss WARN on
    /// every freshly created layer.
    pub(crate) base_less_listing: bool,
    /// How many handles in the listing carry a `seq()` **strictly above** the
    /// live `obj:hdr seq` this plan was built against (b.hka (b)).
    ///
    /// # The input is a fresh header read, never a producer-supplied number
    ///
    /// `live_seq` is a `HGET le:{object_id}:obj:hdr seq` the caller takes
    /// **immediately before this call**, against the live keyspace, and nothing
    /// else is mixed into it — not the sequence the tick's own write claimed,
    /// not the gate-time reading pass 1 took. That is a correctness rule, not a
    /// freshness preference, and the re-base producer is why (b.hka review
    /// round 2).
    ///
    /// Two of the three producers of a generation sequence do derive theirs
    /// from the live keyspace, and for them a producer-supplied value would
    /// have been sound:
    ///
    /// - the assembler pins `snap_seq` from the header's own `seq` at a
    ///   proven cut (atomic or fenced), so `snap_seq <= seq`;
    /// - a tail's `to_seq` is the last commit it captured off the live stream,
    ///   which is `<=` the header's `seq` for the same reason.
    ///
    /// The third does not. [`rebase_layer_lineage`](crate::checkpoint::rebase)
    /// holds no Valkey client at all: its `RebaseDecision::Rebased.to_seq` is
    /// `lineage.head_seq`, minted from the **stored listing**. In precisely the
    /// scenario this guard exists for — a listing that belongs to a different
    /// incarnation of the `object_id` — that number is the foreign lineage's own
    /// head, so folding it in (`live_seq.max(sequence)`, the round-1 shape)
    /// would import the untrusted evidence into the trust check, read
    /// `generations_above_live_seq == 0`, and prune the other incarnation's
    /// generations. Hence: one fresh read, no `max`.
    ///
    /// A DR restore is the remaining way `seq` moves discontinuously, and it
    /// moves it the safe way: the restore seeds the writer cursor **above** the
    /// snapshot it restored (`restore_floor + band`, SDD §22.6) — above *that*
    /// snapshot, note, not above every generation on the stem.
    ///
    /// So a generation above the live `seq` usually means the stem is being
    /// addressed by an `object_id` that is not the one that wrote it: object-id
    /// reuse after a keyspace loss, or a rewound keyspace restored from an older
    /// backup. In that state the listing describes a **different incarnation**
    /// of the id, and the quota arithmetic is being asked to choose which of
    /// another layer's generations to destroy.
    ///
    /// # The third producer: a restore that stepped down
    ///
    /// Since b.hka (d) there is a benign, *same*-incarnation way to reach a
    /// non-zero count, and an operator reading the guard's WARN has to be able
    /// to tell it apart from the two above. A restore that refuses the newest
    /// generation and steps down to an older one — the typed
    /// [`MissingMetaCompanion`](crate::checkpoint::store::MissingMetaCompanion)
    /// torn put, outcome label `stepped_down_missing_meta` — seeds
    /// `older_snap_seq + band`. The refused generation is still durable on the
    /// stem, and when its `seq` is above that seeded value this guard fires on
    /// every tick until the live incarnation commits past it.
    ///
    /// Consequence: that generation is never pruned while the condition holds
    /// (the refusal empties both delete sets, and the reconciliation
    /// dead-object phase does not reclaim it either — the object is very much
    /// alive). Reclaiming it is tracked on b.474 (b).
    ///
    /// When non-zero, [`prune`](Self::prune) and
    /// [`orphan_tails`](Self::orphan_tails) are both **empty** — the same
    /// refusal shape as [`base_less_listing`](Self::base_less_listing), and for
    /// the same reason: deletion is irreversible and refusing costs only
    /// deferred reclamation. It is also **self-clearing**, which the base-less
    /// anomaly is not: once the live incarnation commits past the stale
    /// generations' sequences — or the reconciliation dead-object phase reclaims
    /// them — the condition disappears on its own with no operator action.
    pub(crate) generations_above_live_seq: usize,
    /// Tails that tie in sequence with the nearest `Base` above them in the
    /// listing. **Diagnostic only** — no arm of this planner branches on it.
    ///
    /// # What it is, and why it is not a conservatism trigger
    ///
    /// This is the listing-visible footprint of the zero-width tail described in
    /// the module header — and it is *also*, exactly and indistinguishably, the
    /// footprint of the routine SR-4.1 re-base (`base@S` + the surviving head
    /// `tail@S`, rooted at an older base). From `(kind, sequence)` alone the two
    /// are congruent: `handle.seq()` cannot tell "my base is the one directly
    /// above me" from "my base is further down". Nothing short of the envelope
    /// header separates them, and fetching it is what the whole module refuses.
    ///
    /// A release-conservative arm — "do not prune a unit holding a sequence-tied
    /// tail" — was considered and **rejected**, because that predicate is true of
    /// the superseded unit on *every* layer after *every* chain collapse. It
    /// would suspend retention permanently on exactly the layers retention exists
    /// for, trading a bounded storage residual for an unbounded one.
    ///
    /// It is safe to not act, because a mis-grouped zero-width tail is inert:
    /// - it covers `(S, S]` — the empty commit range, so deleting it loses no
    ///   coverage and retaining it adds none;
    /// - it is never a link-walk member. A successor tail at `from_seq == S`
    ///   whose lineage root is also `S` resolves its predecessor's kind as `Base`
    ///   (`lineage::verify_links` picks `Base` iff `want_seq == lineage_root`),
    ///   so the walk goes straight to `base@S` and steps over it;
    /// - it is a tail, so it can neither close a unit nor consume a `retain` slot.
    ///
    /// The count is kept so the executor can log it and so a future change to the
    /// link rules — which is what would make the shape matter — has the number in
    /// hand instead of a silent mis-grouping.
    pub(crate) base_tied_tails: usize,
}

/// Group a `list_layer_checkpoints` result into lineage units and decide which
/// to delete. Pure: no `await`, no store, no meta, no blob.
///
/// `handles` MUST be the listing as the adapter returned it — newest-first,
/// `Base` before `Tail` at an equal sequence. A `debug_assert!` checks that
/// rather than repairing it: re-sorting here would fork the ordering rule that
/// `adapters::handle_generation_sort_key` exists to hold in one place, and a
/// caller handing over a shuffled listing has a bug that must surface as a
/// failing test, not as a quietly re-derived answer.
///
/// # The four rules
///
/// 1. **Grouping (D1).** Newest-first scan; a `Base` closes the accumulated
///    tails into a unit rooted at it; what is still accumulated at the end is the
///    orphan bucket.
/// 2. **Counting (D2).** Units count against `retain`; the orphan bucket never
///    does. See [`RetentionPlan::retained_units`] for the unit-count ==
///    base-count identity.
/// 3. **Order (D3).** `prune` is oldest-unit-first; each unit's `tails` stay
///    newest-first, so the executor deletes tails newest-first and the base last
///    — dependency-safe under a crash at any point.
/// 4. **Base-less guard (D4).** A non-empty listing with no `Base` at all prunes
///    **nothing**: empty `prune`, empty `orphan_tails`,
///    [`base_less_listing`](RetentionPlan::base_less_listing) set.
/// 5. **Prune-ordering guard (b.hka (b)).** A listing holding ANY generation
///    whose `seq()` exceeds `live_seq` prunes **nothing**, in the same D4 shape,
///    with [`generations_above_live_seq`](RetentionPlan::generations_above_live_seq)
///    set. It fires BEFORE the quota is spent — the whole point is that the
///    quota must not choose a victim from a listing it cannot trust — and it
///    leaves the D1 grouping and the D2/D3 `split_off` untouched, composing with
///    them rather than re-deriving the retention split.
///
/// # Why the prune-ordering guard keys on the live header sequence
///
/// `live_seq` is a **fresh `HGET obj:hdr seq` the caller takes immediately
/// before calling this**, and nothing else may be folded into it — see
/// [`RetentionPlan::generations_above_live_seq`] for why a producer-supplied
/// sequence (specifically the re-base's, which is minted from the stored
/// listing) is disqualified. Every live-keyspace producer of a generation
/// sequence keeps it at or below that value, so a listing that violates it is
/// not describing the live incarnation of this `object_id` at all — object-id
/// reuse after keyspace loss, or a keyspace rewound to an older backup. Spending
/// a retention quota on that listing means picking which of a *different*
/// layer's durable generations to destroy, irreversibly, on evidence that is
/// known to be inconsistent. Refusing defers reclamation and nothing more; the
/// condition self-clears as soon as the live incarnation commits past those
/// sequences, and the §22.17.5 dead-object reconcile phase is the reclamation
/// path in the meantime.
///
/// A `live_seq` the caller could not read is NOT a licence to prune, and in
/// **neither** unreadable case is this function called at all:
///
/// - an **absent** header means the object died mid-tick, so the caller skips
///   the prune before the planner runs. Passing `0` would be behaviourally
///   identical (a tick that reached the prune has just written a generation, so
///   every handle is above `0` and rule 5 refuses the whole listing), but it
///   would report the refusal as the prune-ordering guard's — "suspect
///   object-id reuse after keyspace loss" — which is the wrong diagnosis for an
///   ordinary concurrent `DeleteLayer`. The caller emits its own accurate WARN
///   instead, with the dead-object reconcile as the reclamation path;
/// - a header read that **errored** answers nothing at all, so the caller skips
///   the prune rather than calling this with a substitute value.
///
/// # Why the base-less guard refuses rather than reaps
///
/// That listing is `lineage_shape`'s / `recovery::select_restore_source`'s
/// documented "manual pruning or partial store loss" anomaly. Erasing it is
/// irreversible and reclaims nothing restorable — the tails may yet be the only
/// evidence of what the layer held, and a base restored out-of-band would make
/// them replayable again.
///
/// **The full consequence, recorded as accepted rather than left unstated:** a
/// genuinely orphaned tails-only store is then reaped by **nobody**. The
/// §22.17.5 reconciliation sweep registers only *incomplete* stems (its
/// "Torn-tail residuals" arm: an orphaned `.tail.meta.json` with no envelope); a
/// **complete** tail — envelope plus meta — is never a candidate there. So those
/// artifacts leak permanently, with no reclamation path and no signal beyond a
/// per-tick WARN at the call site. That is an accepted trade, since the shape
/// requires manual pruning or partial store loss (both out-of-band events), but
/// it does raise the Task-5 observability input for this arm from cosmetic to
/// operational.
///
/// **What an operator would do with a signal, stated so Task 5 can adjudicate on
/// facts rather than re-derive them.** A non-zero rate on this arm means *"a
/// layer's bases are gone — investigate partial checkpoint-store loss and
/// reclaim by hand"*. That is an alert-worthy condition, not a debug line, and
/// today it is reachable only by grepping WARNs. The same finding is what
/// motivates the reclamation-path Issue this Task's PM review defers; neither is
/// minted here (this Task adds no metric family, label value, or env knob).
pub(crate) fn plan_retention(
    handles: &[CheckpointHandle],
    retain: usize,
    live_seq: u64,
) -> RetentionPlan {
    debug_assert!(
        handles.is_sorted_by_key(crate::checkpoint::adapters::handle_generation_sort_key),
        "plan_retention requires the listing in `handle_generation_sort_key` order \
         (newest-first, Base before Tail at an equal sequence); the adjacency grouping is \
         unsound on any other order and deliberately does not re-derive it"
    );

    // Rule 5, evaluated over the raw listing and BEFORE the quota. It is
    // deliberately not folded into the grouping loop: the grouping is the D1
    // rule and stays a pure restatement of the listing, while this is a
    // whole-listing trust question whose answer is "do not use this listing at
    // all". Counting rather than short-circuiting so the refusal can say how
    // many generations were implicated.
    let generations_above_live_seq = handles.iter().filter(|h| h.seq() > live_seq).count();

    let mut units: Vec<LineageUnit> = Vec::new();
    let mut pending_tails: Vec<CheckpointHandle> = Vec::new();
    let mut base_tied_tails = 0usize;
    let mut newest_base_seq: Option<u64> = None;

    for handle in handles {
        match handle.kind {
            CheckpointKind::Tail => {
                if newest_base_seq == Some(handle.seq()) {
                    // Tied in sequence with the base above it: the routine
                    // re-base tie, and — indistinguishably — the zero-width-tail
                    // footprint. Diagnostic only; see `base_tied_tails`.
                    base_tied_tails += 1;
                }
                pending_tails.push(handle.clone());
            }
            CheckpointKind::Base => {
                newest_base_seq = Some(handle.seq());
                units.push(LineageUnit {
                    base: handle.clone(),
                    tails: std::mem::take(&mut pending_tails),
                });
            }
        }
    }

    // D4. `units.is_empty()` is "no Base in the listing"; the extra emptiness
    // test is what keeps a never-checkpointed layer out of the anomaly arm.
    if units.is_empty() {
        return RetentionPlan {
            retained_units: 0,
            prune: Vec::new(),
            orphan_tails: Vec::new(),
            base_less_listing: !handles.is_empty(),
            generations_above_live_seq,
            base_tied_tails,
        };
    }

    // Rule 5's refusal, in the D4 shape. Placed after the grouping so
    // `retained_units` still reports what the quota WOULD have kept (an honest
    // diagnostic for the WARN at the call site) and before the `split_off` so
    // the quota is never actually spent on a listing that cannot be trusted.
    if generations_above_live_seq > 0 {
        return RetentionPlan {
            retained_units: units.len().min(retain),
            prune: Vec::new(),
            orphan_tails: Vec::new(),
            base_less_listing: false,
            generations_above_live_seq,
            base_tied_tails,
        };
    }

    // D2 + D3. `units` is newest-first; the quota is spent on its head and the
    // remainder is reversed into oldest-unit-first delete order.
    let retained_units = units.len().min(retain);
    let mut prune = units.split_off(retained_units);
    prune.reverse();

    RetentionPlan {
        retained_units,
        prune,
        orphan_tails: pending_tails,
        base_less_listing: false,
        generations_above_live_seq,
        base_tied_tails,
    }
}

#[cfg(test)]
mod tests {
    //! Two families of cases share this module.
    //!
    //! **Shapes.** The five cases immediately below pin the grouping's own
    //! shapes: the re-base tie, the interim residue's orphan bucket, the
    //! within-unit delete order, the base-less guard, and the zero-width tied
    //! tail.
    //!
    //! **Counting.** The named fixtures after those are the SR-5.1 arithmetic:
    //! `retain_generations` counts LINEAGE UNITS, and a lone base is a unit of
    //! chain length 0. Each fixture states its listing, the partition the
    //! adjacency scan owes it, and the delete set the executor is handed.
    //!
    //! | Fixture | Listing, newest-first | Partition | `retain` | Delete set |
    //! |---|---|---|---|---|
    //! | `three_lone_bases` | `b30, b20, b10` | `{b30}` `{b20}` `{b10}` | 2 | `b10` |
    //! | `mixed_arithmetic` | `b20, t13, t12, t11, b10` | `{b20}` `{b10 + t13,t12,t11}` | 1 | `t13, t12, t11, b10` |
    //! | `restore_slot_pressure` | `b2014, b1012, t12, t11, b10` | `{b2014}` `{b1012}` `{b10 + t12,t11}` | 2 | `t12, t11, b10` |
    //! | `rebase_tie` | `b40, t40, t31, b20` | `{b40}` `{b20 + t40,t31}` | 2 | nothing |
    //! | `interim_residue` | `t33, b30, t25, b20, t15, t12` | `{b30 + t33}` `{b20 + t25}`, orphans `t15, t12` | 2 | nothing |
    //!
    //! The unit count equals the base count on every row — a recorded finding
    //! rather than a coincidence (see [`RetentionPlan::retained_units`]), and
    //! the reason these cases assert PARTITIONS instead of survivor lists: a
    //! base-count implementation agrees with a unit-count one on every *count*
    //! in the table and diverges only on delete-set MEMBERSHIP.
    //!
    //! ## Recorded mutation verification
    //!
    //! The counting rule is easy to assert vacuously — a base-only listing
    //! answers the same under every candidate implementation — so the table was
    //! mutation-checked and the results recorded here. The mutation was
    //! reverted and is not committed.
    //!
    //! **Mutation: the counting unit flipped back from lineages to BASES.**
    //! Three lines in [`plan_retention`], immediately after the `split_off`,
    //! clearing each pruned unit's tails (`for unit in &mut prune {
    //! unit.tails.clear(); }`) — i.e. the pre-Epic rule that pruned bases and
    //! left every tail behind. `retained_units` is untouched by it, which is
    //! the point: the two implementations differ only in what they DELETE.
    //!
    //! Result: 10 of the 15 cases in this module go red (`nextest --lib
    //! 'test(checkpoint::retention::)' --no-fail-fast`, 5 passed / 10 failed).
    //! The named counting cases and their values:
    //!
    //! - `a_whole_lineage_is_pruned_for_one_slot_where_a_base_count_would_strand_its_tails`
    //!   — partition `left: [(20, []), (10, [])]`,
    //!   `right: [(20, []), (10, [13, 12, 11])]`.
    //! - `restore_slot_pressure` — `left: [(2014, []), (1012, []), (10, [])]`,
    //!   `right: [(2014, []), (1012, []), (10, [12, 11])]`.
    //! - `the_rebase_tie_delete_set_names_the_tied_tail_and_never_the_new_base`
    //!   — delete set `left: [(20, Base)]`,
    //!   `right: [(40, Tail), (31, Tail), (20, Base)]`: the mutated prune
    //!   deletes the superseded base and strands both its tails.
    //! - `retain_of_one_keeps_exactly_the_newest_lineage_on_every_fixture`
    //!   (row `mixed_arithmetic`) — `left: [(10, Base)]`,
    //!   `right: [(13, Tail), (12, Tail), (11, Tail), (10, Base)]`. This is the
    //!   base-count-vs-unit-count divergence in one line: one artifact deleted
    //!   instead of four, three tails left with no base to replay them onto.
    //! - `retain_zero_is_a_partition_probe_and_not_a_supported_configuration`
    //!   (row `mixed_arithmetic`) — totality `left: [(10, Base), (20, Base)]`,
    //!   `right: [(10, Base), (11, Tail), (12, Tail), (13, Tail), (20, Base)]`.
    //! - `the_counting_table_partitions_into_the_stated_lineage_units` and
    //!   `the_rebase_tie_is_two_lineages_and_both_survive_at_retain_two` go red
    //!   on the same partition values; three of the shape cases above
    //!   (`rebase_shape_…`, `prune_is_oldest_unit_first_…`,
    //!   `a_sequence_tied_tail_…`) go red too.
    //!
    //! **The two counting cases that stay GREEN are findings, not gaps.**
    //! `the_retained_unit_count_equals_the_base_count_on_every_fixture` is
    //! green because the mutation does not change any count — that identity is
    //! precisely why a count cannot evidence the rule. And
    //! `three_lone_bases_partition_into_three_chain_zero_lineages` is green
    //! because a listing with no tails has none to strand, which is why the
    //! base-only fixture the pre-Epic suite already owned could not have caught
    //! this and the mixed fixtures had to be written.

    use super::*;

    fn base(seq: u64) -> CheckpointHandle {
        CheckpointHandle::base(7, seq, format!("7.{seq}"))
    }

    fn tail(seq: u64) -> CheckpointHandle {
        CheckpointHandle::tail(7, seq, format!("7.{seq}.tail"))
    }

    fn seqs(handles: &[CheckpointHandle]) -> Vec<u64> {
        handles.iter().map(CheckpointHandle::seq).collect()
    }

    /// AC shape 1 — the SR-4.1 re-base artifact. `base@S` ties with the
    /// surviving head `tail@S` and sorts first, so the scan closes an empty
    /// accumulation at the new chain-0 base and hands `tail@S` down to the
    /// superseded unit rooted at `base@B`.
    #[test]
    fn rebase_shape_groups_the_surviving_head_tail_into_the_superseded_unit() {
        let listing = [base(40), tail(40), tail(31), base(20)];

        let plan = plan_retention(&listing, 1, u64::MAX);

        assert_eq!(
            plan.retained_units, 1,
            "the chain-0 base@40 unit is retained"
        );
        assert!(!plan.base_less_listing);
        assert!(plan.orphan_tails.is_empty());
        assert_eq!(plan.prune.len(), 1);
        let pruned = &plan.prune[0];
        assert_eq!(pruned.base, base(20));
        assert_eq!(
            seqs(&pruned.tails),
            vec![40, 31],
            "the unit's tails stay newest-first, so the executor deletes tail@40, tail@31, \
             then base@20 last"
        );
        assert_eq!(pruned.generation_count(), 3);
    }

    /// AC shape 2 — the Epic-3 interim residue: two units plus an orphan bucket
    /// of tails below the oldest surviving base (`base@10` already gone).
    #[test]
    fn interim_residue_yields_two_units_plus_the_orphan_bucket() {
        let listing = [tail(33), base(30), tail(25), base(20), tail(15), tail(12)];

        let plan = plan_retention(&listing, 2, u64::MAX);

        assert_eq!(plan.retained_units, 2);
        assert!(plan.prune.is_empty(), "both units fit inside retain = 2");
        assert_eq!(
            seqs(&plan.orphan_tails),
            vec![15, 12],
            "tails below the OLDEST base are the orphan bucket, and they never count \
             against retain"
        );
        assert!(!plan.base_less_listing);
    }

    /// AC shape 3 — D3's ordering, on the same listing at a quota that actually
    /// bites: oldest unit first, tails newest-first inside each.
    #[test]
    fn prune_is_oldest_unit_first_with_tails_newest_first() {
        let listing = [tail(33), base(30), tail(25), base(20), tail(15), tail(12)];

        let plan = plan_retention(&listing, 0, u64::MAX);

        assert_eq!(plan.retained_units, 0);
        assert_eq!(
            plan.prune
                .iter()
                .map(|unit| unit.base.seq())
                .collect::<Vec<_>>(),
            vec![20, 30],
            "oldest unit first"
        );
        assert_eq!(seqs(&plan.prune[0].tails), vec![25]);
        assert_eq!(seqs(&plan.prune[1].tails), vec![33]);
        assert_eq!(seqs(&plan.orphan_tails), vec![15, 12]);
    }

    /// AC shape 4 — D4. A tails-only listing prunes nothing at all; an empty one
    /// is routine lifecycle and does not raise the anomaly flag.
    #[test]
    fn a_base_less_listing_prunes_nothing_and_an_empty_one_is_not_an_anomaly() {
        let tails_only = [tail(9), tail(4)];

        let plan = plan_retention(&tails_only, 1, u64::MAX);

        assert!(plan.base_less_listing);
        assert!(plan.prune.is_empty());
        assert!(
            plan.orphan_tails.is_empty(),
            "D4 refuses to reap the anomaly: erasing it is irreversible and reclaims nothing \
             restorable"
        );
        assert_eq!(plan.retained_units, 0);

        let empty = plan_retention(&[], 1, u64::MAX);
        assert!(
            !empty.base_less_listing,
            "an empty listing is routine lifecycle, not suspected store loss"
        );
        assert!(empty.prune.is_empty() && empty.orphan_tails.is_empty());
    }

    /// The zero-width guard. A zero-width tail (`base_snap_seq == from_seq ==
    /// to_seq == S`) is listing-congruent with the re-base tie, so it is counted
    /// and NOT acted on. Pinned so a future writer that starts emitting one finds
    /// the recorded behaviour instead of a silent mis-grouping.
    #[test]
    fn a_sequence_tied_tail_is_counted_and_grouped_below_its_tied_base() {
        let listing = [base(40), tail(40), base(20)];

        let plan = plan_retention(&listing, 1, u64::MAX);

        assert_eq!(plan.base_tied_tails, 1);
        assert_eq!(plan.retained_units, 1);
        assert_eq!(plan.prune.len(), 1);
        assert_eq!(plan.prune[0].base, base(20));
        assert_eq!(
            seqs(&plan.prune[0].tails),
            vec![40],
            "the tied tail groups with the NEXT-OLDER base, which is right for the re-base \
             shape and inert for the zero-width shape"
        );

        let untied = plan_retention(&[base(40), tail(31), base(20)], 1, u64::MAX);
        assert_eq!(untied.base_tied_tails, 0);
    }

    // ---- The SR-5.1 counting tables: units, not artifacts, not bases ----

    /// `(sequence, kind)` — a delete key minus the object id, which every
    /// fixture in this module holds constant. Same rendering
    /// `checkpointer::tests::listing()` uses, so the plan-level tables here and
    /// the store-level tables there read alike.
    fn marks(handles: &[CheckpointHandle]) -> Vec<(u64, CheckpointKind)> {
        handles
            .iter()
            .map(|handle| (handle.seq(), handle.kind))
            .collect()
    }

    /// A unit as the table states it: `(base sequence, tail sequences
    /// newest-first)`.
    fn shapes(units: &[LineageUnit]) -> Vec<(u64, Vec<u64>)> {
        units
            .iter()
            .map(|unit| (unit.base.seq(), seqs(&unit.tails)))
            .collect()
    }

    /// The FULL partition of a listing — the retained units included — as
    /// `(units newest-first, orphan tails)`.
    ///
    /// [`RetentionPlan`] names only the units it is about to delete, so the
    /// retained half of the partition is not directly readable. `retain = 0`
    /// makes it readable: every unit lands in `prune`, and reversing that
    /// oldest-first vector recovers the scan's own newest-first order. The
    /// probe is exact rather than approximate — [`plan_retention`] forms its
    /// units *before* the quota is applied and the quota is a `split_off`, so
    /// the units a `retain = 0` call reports are the same values any other
    /// `retain` would have formed over the same listing.
    ///
    /// `retain = 0` is a PROBE here and not a supported configuration:
    /// `Config::from_env` rejects it (`src/config.rs:1410-1418`) with
    /// *"invalid value for LIVE_EDIT_CHECKPOINT_RETAIN_GENERATIONS: must be
    /// greater than 0 (retention of 0 would prune every just-written checkpoint
    /// and leave the object with no recoverable blob)"*. The ordering property
    /// this helper leans on is pinned by
    /// [`retain_zero_is_a_partition_probe_and_not_a_supported_configuration`],
    /// so the probe is evidence rather than an assumption.
    fn partition(listing: &[CheckpointHandle]) -> (Vec<LineageUnit>, Vec<CheckpointHandle>) {
        let plan = plan_retention(listing, 0, u64::MAX);
        let mut units = plan.prune;
        units.reverse();
        (units, plan.orphan_tails)
    }

    /// The artifacts `checkpointer::prune_old_generations` is handed, in the
    /// order its step 3 issues them: pruned units oldest-first, within a unit
    /// the tails newest-first and the base last, then the orphan bucket.
    ///
    /// The executor-level pin of that order — and of its crash residuals — is
    /// `t3.dfh.be.mk.ah`'s. Rendering it here is what lets each counting case
    /// state its delete set as one value instead of three.
    fn delete_set(plan: &RetentionPlan) -> Vec<(u64, CheckpointKind)> {
        let mut out = Vec::new();
        for unit in &plan.prune {
            out.extend(marks(&unit.tails));
            out.push((unit.base.seq(), unit.base.kind));
        }
        out.extend(marks(&plan.orphan_tails));
        out
    }

    /// One row of the counting table: a listing, and the partition the
    /// adjacency scan owes it, **stated as a literal**.
    struct CountingCase {
        name: &'static str,
        /// Newest-first, as the adapters return it (`handle_generation_sort_key`
        /// order — the `debug_assert!` in [`plan_retention`] fires on any
        /// other).
        listing: Vec<CheckpointHandle>,
        /// Units newest-first, as `(base sequence, tail sequences newest-first)`.
        units: Vec<(u64, Vec<u64>)>,
        /// Orphan-bucket tail sequences — below the OLDEST base — newest-first.
        orphans: Vec<u64>,
    }

    impl CountingCase {
        /// The delete set this row states at `retain`, composed the way the
        /// executor consumes it: units below the quota oldest-first, tails then
        /// base inside each, orphan bucket last.
        ///
        /// Built from the row's LITERALS, never from a plan, so a table-wide
        /// property using it is an independent statement about the grouping
        /// rather than a restatement of whatever the grouping just did.
        fn expected_delete_set(&self, retain: usize) -> Vec<(u64, CheckpointKind)> {
            let mut out = Vec::new();
            for (base_seq, tail_seqs) in self.units.iter().skip(retain).rev() {
                out.extend(tail_seqs.iter().map(|seq| (*seq, CheckpointKind::Tail)));
                out.push((*base_seq, CheckpointKind::Base));
            }
            out.extend(self.orphans.iter().map(|seq| (*seq, CheckpointKind::Tail)));
            out
        }
    }

    /// The named fixtures of the counting table.
    ///
    /// The four table-wide properties below (the stated partition, the
    /// unit-count identity, the `retain = 1` floor, the `retain = 0` probe) run
    /// over every row, so a fixture added here is automatically held to all
    /// four.
    ///
    /// The expected partitions are written out rather than read back off
    /// [`plan_retention`] on purpose: a property whose expected value came from
    /// the function under test agrees with ANY implementation, including one
    /// that has stopped grouping altogether — which is exactly what the
    /// mutation recorded in this module's header produces.
    fn counting_table() -> Vec<CountingCase> {
        vec![
            CountingCase {
                name: "three_lone_bases",
                listing: vec![base(30), base(20), base(10)],
                units: vec![(30, vec![]), (20, vec![]), (10, vec![])],
                orphans: vec![],
            },
            CountingCase {
                name: "mixed_arithmetic",
                listing: vec![base(20), tail(13), tail(12), tail(11), base(10)],
                units: vec![(20, vec![]), (10, vec![13, 12, 11])],
                orphans: vec![],
            },
            CountingCase {
                name: "restore_slot_pressure",
                listing: vec![base(2014), base(1012), tail(12), tail(11), base(10)],
                units: vec![(2014, vec![]), (1012, vec![]), (10, vec![12, 11])],
                orphans: vec![],
            },
            CountingCase {
                name: "rebase_tie",
                listing: vec![base(40), tail(40), tail(31), base(20)],
                units: vec![(40, vec![]), (20, vec![40, 31])],
                orphans: vec![],
            },
            CountingCase {
                name: "interim_residue",
                listing: vec![tail(33), base(30), tail(25), base(20), tail(15), tail(12)],
                units: vec![(30, vec![33]), (20, vec![25])],
                orphans: vec![15, 12],
            },
        ]
    }

    /// Bases in a listing — the quantity the pre-Epic prune counted, kept here
    /// so the identity assertion states it rather than assuming it.
    fn base_count(listing: &[CheckpointHandle]) -> usize {
        listing
            .iter()
            .filter(|handle| handle.kind == CheckpointKind::Base)
            .count()
    }

    /// A lone base is a chain-0 lineage: three of them at `retain = 2` prune
    /// exactly the oldest.
    ///
    /// `checkpointer::tests::prune_over_a_base_only_listing_is_unchanged` pins
    /// that OUTCOME end-to-end and is kept verbatim across the rework. What it
    /// cannot show is why the answer is right — a base-only listing is the one
    /// shape on which every candidate implementation (artifact count, base
    /// count, unit count) agrees. This case asserts the PARTITION, so an
    /// implementation that reached the same delete set without ever forming a
    /// chain-0 unit fails here instead of passing by accident.
    #[test]
    fn three_lone_bases_partition_into_three_chain_zero_lineages() {
        let listing = [base(30), base(20), base(10)];

        let (units, orphans) = partition(&listing);

        assert_eq!(
            shapes(&units),
            vec![(30, vec![]), (20, vec![]), (10, vec![])],
            "each lone base closes an EMPTY accumulation, so it is a lineage unit of chain \
             length 0 — not a bare artifact that the quota happens to count"
        );
        assert!(orphans.is_empty(), "a base-only listing has no stray tails");

        let plan = plan_retention(&listing, 2, u64::MAX);

        assert_eq!(plan.retained_units, 2);
        assert_eq!(
            delete_set(&plan),
            vec![(10, CheckpointKind::Base)],
            "three chain-0 lineages at retain = 2 retire exactly the oldest lineage, which on \
             this shape is exactly one artifact"
        );
        assert_eq!(plan.base_tied_tails, 0);
        assert!(!plan.base_less_listing);
    }

    /// The sharp case: four artifacts spent on one retain slot, and the point
    /// at which a base-count implementation and a unit-count one part company.
    ///
    /// Lineage A is `base@10` + `(10,11]`, `(11,12]`, `(12,13]`; lineage B is
    /// the lone `base@20`. At `retain = 1` B is the newest unit and survives;
    /// A is pruned WHOLE.
    ///
    /// Both implementations agree on the COUNT here — two bases, two units, one
    /// slot spent, one unit retired — which is exactly why
    /// `checkpointer::tests::prune_counts_lineage_units_not_artifacts_when_deciding_whether_to_prune_at_all`
    /// cannot separate them and hands the discriminator to this table. They
    /// diverge on delete-set MEMBERSHIP: pruning bases only deletes `base@10`
    /// and spares its three tails, leaving artifacts whose root is gone — the
    /// unrestorable class FR-6 forbids, and precisely the Epic-3 interim hazard
    /// ("tails are deleted by nothing") this Task discharges. The unit rule
    /// deletes all four.
    #[test]
    fn a_whole_lineage_is_pruned_for_one_slot_where_a_base_count_would_strand_its_tails() {
        let listing = [base(20), tail(13), tail(12), tail(11), base(10)];

        let (units, orphans) = partition(&listing);

        assert_eq!(
            shapes(&units),
            vec![(20, vec![]), (10, vec![13, 12, 11])],
            "the scan closes a chain-0 unit at base@20 and hands the three tails down to \
             base@10, the base each of them is rooted at"
        );
        assert!(orphans.is_empty(), "every tail here sits above a base");

        let plan = plan_retention(&listing, 1, u64::MAX);

        assert_eq!(
            plan.retained_units, 1,
            "lineage B is the newest and survives"
        );
        assert_eq!(plan.prune.len(), 1);
        assert_eq!(
            plan.prune[0].generation_count(),
            4,
            "one retain slot retires FOUR artifacts — the unit is the accounting unit, the \
             artifact is not"
        );
        assert_eq!(
            delete_set(&plan),
            vec![
                (13, CheckpointKind::Tail),
                (12, CheckpointKind::Tail),
                (11, CheckpointKind::Tail),
                (10, CheckpointKind::Base),
            ],
            "a base-count prune answers `[(10, Base)]` here and calls itself done, stranding \
             (10,11], (11,12] and (12,13] with no base to replay them onto. The whole-lineage \
             delete is the discriminator between the two implementations, and this listing is \
             where it bites"
        );
    }

    /// The third base producer, and the case that discharges the Epic's
    /// 2026-08-31 traceability deferral by evidence.
    ///
    /// Lineage A is the layer's only captured coverage: `base@10` + `(10,11]`,
    /// `(11,12]`. A DR restore then mints a lone `base@1012` (SR-3.6's
    /// post-restore re-base, at `seeded_seq = restore_floor + band` — the head
    /// at 12 plus the default 1000 guard band), and a second restore a few
    /// commits later mints a lone `base@2014` the same way. Both are chain-0
    /// units, and each consumes one retain slot. At `retain = 2` lineage A is
    /// therefore pruned WHOLE by two bases that carry no captured tail between
    /// them.
    ///
    /// **The accepted consequence, quoted from the module header's *The third
    /// base producer* section rather than paraphrased:** *"At the default
    /// `retain_generations = 2`, a DR restore immediately demotes the
    /// pre-restore lineage to the last retained slot: the store goes from
    /// `[L1, L2]` to `[R, L1]`, and the next post-write prune deletes `L2`
    /// whole."* A second restore repeats the demotion, which is this fixture:
    /// two restores inside one retain window retire the pre-restore lineage
    /// outright. That is sound because a restore definitionally supersedes the
    /// prior timeline — the re-base writes a reset OMPF of exactly the state
    /// `apply_restore` committed, i.e. the composed state of the newest
    /// restorable lineage, so it carries everything the lineage it displaces
    /// carried and the lineage pushed out of the window holds only strictly
    /// older coverage. What does fall is the number of distinct HISTORICAL
    /// recovery points, and the header states where that preference belongs:
    /// *"An operator who needs `N` pre-restore points to survive a restore sets
    /// `retain_generations` to `N + 1`; the knob is where that preference
    /// belongs."*
    ///
    /// The safety floor is asserted too, because "sound" is the load-bearing
    /// half: what survives is restorable on its own.
    #[test]
    fn restore_slot_pressure() {
        let listing = [base(2014), base(1012), tail(12), tail(11), base(10)];

        let (units, orphans) = partition(&listing);

        assert_eq!(
            shapes(&units),
            vec![(2014, vec![]), (1012, vec![]), (10, vec![12, 11])],
            "each restore-minted base is a chain-0 unit in its own right; only the pre-restore \
             lineage carries captured tails"
        );
        assert!(orphans.is_empty());

        let plan = plan_retention(&listing, 2, u64::MAX);

        assert_eq!(
            plan.retained_units, 2,
            "the two restore-minted lone bases fill the window on their own"
        );
        assert_eq!(plan.prune.len(), 1);
        assert_eq!(
            delete_set(&plan),
            vec![
                (12, CheckpointKind::Tail),
                (11, CheckpointKind::Tail),
                (10, CheckpointKind::Base),
            ],
            "lineage A is retired WHOLE by two bases that hold no captured tail between them — \
             the accepted consequence in this test's doc-comment, asserted rather than assumed"
        );

        // The safety floor. Two independent halves: nothing at or above the
        // first restore is touched, and what remains newest is a base, which is
        // restorable standalone (`CheckpointKind::Base`: "restorable on its own,
        // with no lineage") — so the store never lands on a tail whose root the
        // prune has just deleted.
        assert!(
            delete_set(&plan).iter().all(|(seq, _)| *seq < 1012),
            "no restore-minted artifact may be pruned while it is inside the window"
        );
        assert_eq!(
            shapes(&units[..plan.retained_units]),
            vec![(2014, vec![]), (1012, vec![])],
            "the newest surviving unit is the newest restore-minted base, standing alone: the \
             recovery point after this prune is a full-state snapshot needing no tail"
        );
    }

    /// The SR-4.1 re-base artifact shape, at the quota where both lineages fit.
    ///
    /// After a chain collapse the listing holds `base@S` tying in sequence with
    /// the surviving head `tail@S`. They land in DIFFERENT units, and the
    /// reason is adjacency, not a link: the listing is newest-first with `Base`
    /// pinned before `Tail` at an equal sequence
    /// (`adapters::handle_generation_sort_key`, `src/checkpoint/adapters/mod.rs:214-220`,
    /// whose rustdoc names SR-4.1's re-base as the live tie case), so the scan
    /// reaches `base@S` with an EMPTY accumulation and closes a chain-0 unit
    /// there, and `tail@S` then accumulates toward the next base BELOW.
    /// Independent corroboration, not the mechanism: `tail@S`'s own header
    /// records `base_snap_seq == S0`, the older base — so the partition
    /// adjacency produces is the one the artifacts' own headers assert.
    ///
    /// At `retain = 2` both units survive intact and the superseded lineage
    /// stays complete and restorable. A grouping that merged the tie into one
    /// unit would show up here as a single unit and one spare retain slot —
    /// i.e. it would quietly prune a lineage a listing further on.
    ///
    /// `rebase_shape_groups_the_surviving_head_tail_into_the_superseded_unit`
    /// above drives the same listing at `retain = 1`, where the old unit is
    /// retired; this is the counting direction of the same shape.
    #[test]
    fn the_rebase_tie_is_two_lineages_and_both_survive_at_retain_two() {
        let listing = [base(40), tail(40), tail(31), base(20)];

        let (units, orphans) = partition(&listing);

        assert_eq!(
            shapes(&units),
            vec![(40, vec![]), (20, vec![40, 31])],
            "base@40 is a NEW chain-0 unit and tail@40 heads the OLD unit rooted at base@20; a \
             merged grouping would report ONE unit here and buy itself a retain slot it has not \
             earned"
        );
        assert!(orphans.is_empty());

        let plan = plan_retention(&listing, 2, u64::MAX);

        assert_eq!(
            plan.retained_units, 2,
            "two units, two slots: the collapse consumes one slot and the superseded lineage \
             keeps the other"
        );
        assert!(
            delete_set(&plan).is_empty(),
            "nothing is pruned at retain = 2, so the superseded lineage is still complete — \
             base@20 with both its tails — and therefore still restorable"
        );
        assert_eq!(
            plan.base_tied_tails, 1,
            "the tie is recorded as a diagnostic; no arm branches on it"
        );
    }

    /// The keying hazard at the tie, stated at the plan level: the delete set
    /// names `tail@40` and never `base@40`.
    ///
    /// `base@S` and `tail@S` share a sequence, so a `(object_id, sequence)`
    /// delete of the superseded unit's head tail would take the freshly minted
    /// re-base with it — the one artifact that must survive the collapse.
    /// Lineage deletion is keyed `(object_id, sequence, KIND)`, and that starts
    /// here: the plan hands the executor a HANDLE per artifact, kind included.
    /// The store-level counterpart —
    /// `checkpointer::tests::prune_at_an_equal_sequence_deletes_the_base_without_touching_the_tail_beside_it`
    /// — proves the adapters honour the kind; this proves the plan supplies it.
    #[test]
    fn the_rebase_tie_delete_set_names_the_tied_tail_and_never_the_new_base() {
        let listing = [base(40), tail(40), tail(31), base(20)];

        let plan = plan_retention(&listing, 1, u64::MAX);

        assert_eq!(
            delete_set(&plan),
            vec![
                (40, CheckpointKind::Tail),
                (31, CheckpointKind::Tail),
                (20, CheckpointKind::Base),
            ],
            "the superseded unit goes whole: its head tail at the tied sequence 40, then \
             tail@31, then base@20 last"
        );
        assert!(
            !delete_set(&plan).contains(&(40, CheckpointKind::Base)),
            "the newly minted re-base at the SAME sequence 40 must never appear in the delete \
             set; a sequence-keyed prune is what puts it there"
        );
    }

    /// Every row of the table partitions into the lineage units the table
    /// states — the claim the per-fixture cases below all rest on, made once
    /// over the whole table and against written-out expectations.
    #[test]
    fn the_counting_table_partitions_into_the_stated_lineage_units() {
        for case in counting_table() {
            let (units, orphans) = partition(&case.listing);

            assert_eq!(
                shapes(&units),
                case.units,
                "{}: the adjacency scan groups this listing into the units the table states — \
                 each Base closing the tails accumulated above it",
                case.name
            );
            assert_eq!(
                seqs(&orphans),
                case.orphans,
                "{}: the orphan bucket is exactly the tails below the OLDEST base",
                case.name
            );
        }
    }

    /// The recorded identity, pinned rather than left as prose: the unit count
    /// equals the base count on every fixture, at every quota.
    ///
    /// This is why the counting rule cannot be evidenced by a count. A
    /// base-count implementation and a unit-count implementation answer
    /// `retained_units` identically everywhere in this table — the divergence
    /// is delete-set membership, which
    /// `a_whole_lineage_is_pruned_for_one_slot_where_a_base_count_would_strand_its_tails`
    /// and [`restore_slot_pressure`] own. This case is therefore expected to
    /// stay GREEN under the counting mutation recorded in the module header;
    /// that is the finding, not a gap.
    #[test]
    fn the_retained_unit_count_equals_the_base_count_on_every_fixture() {
        for case in counting_table() {
            assert_eq!(
                case.units.len(),
                base_count(&case.listing),
                "{}: units and bases are in bijection — every unit is closed by exactly one base",
                case.name
            );

            for retain in 0..=4 {
                let plan = plan_retention(&case.listing, retain, u64::MAX);
                assert_eq!(
                    plan.retained_units,
                    case.units.len().min(retain),
                    "{} at retain = {retain}: retained_units is min(units, retain), and units == \
                     bases, so the COUNT half of retention is unchanged by the lineage rework",
                    case.name
                );
            }
        }
    }

    /// The SR-5.2 floor, over the whole table: at `retain = 1` exactly the
    /// newest lineage survives, every older one is pruned whole — base AND
    /// tails — oldest unit first, and the orphan bucket goes too because it
    /// never counted against the quota in the first place.
    #[test]
    fn retain_of_one_keeps_exactly_the_newest_lineage_on_every_fixture() {
        for case in counting_table() {
            let plan = plan_retention(&case.listing, 1, u64::MAX);

            assert_eq!(
                plan.retained_units, 1,
                "{}: the newest unit always survives",
                case.name
            );
            assert_eq!(
                delete_set(&plan),
                case.expected_delete_set(1),
                "{}: everything below the newest lineage goes, whole units at a time — a prune \
                 that spared the superseded units' tails would strand them",
                case.name
            );
        }
    }

    /// `retain = 0` is asserted here for its ORDERING property only, and is not
    /// a supported configuration: `Config::from_env` rejects it
    /// (`src/config.rs:1410-1418`) with *"invalid value for
    /// LIVE_EDIT_CHECKPOINT_RETAIN_GENERATIONS: must be greater than 0
    /// (retention of 0 would prune every just-written checkpoint and leave the
    /// object with no recoverable blob)"*, so no deployed service reaches this
    /// arm.
    ///
    /// The property matters anyway, because [`partition`] leans on it: at
    /// `retain = 0` every unit lands in `prune`, oldest-first, and units plus
    /// orphans account for the listing EXACTLY — no artifact invented, none
    /// dropped, none counted twice. That totality is what makes a partition
    /// assertion elsewhere in this module a statement about the whole listing
    /// rather than about the pruned end of it.
    ///
    /// `prune_is_oldest_unit_first_with_tails_newest_first` above pins the same
    /// quota for the within-unit order; this pins the across-listing totality.
    #[test]
    fn retain_zero_is_a_partition_probe_and_not_a_supported_configuration() {
        for case in counting_table() {
            let plan = plan_retention(&case.listing, 0, u64::MAX);

            assert_eq!(
                plan.retained_units, 0,
                "{}: nothing is retained at retain = 0",
                case.name
            );

            let pruned_bases: Vec<u64> = plan.prune.iter().map(|unit| unit.base.seq()).collect();
            let mut ascending = pruned_bases.clone();
            ascending.sort_unstable();
            assert_eq!(
                pruned_bases, ascending,
                "{}: prune is oldest-unit-first, which is what lets `partition` recover the \
                 scan's newest-first order by reversing it",
                case.name
            );

            let mut planned = delete_set(&plan);
            let mut whole_listing = marks(&case.listing);
            let sort_key = |(seq, kind): &(u64, CheckpointKind)| {
                (*seq, u8::from(*kind == CheckpointKind::Tail))
            };
            planned.sort_unstable_by_key(sort_key);
            whole_listing.sort_unstable_by_key(sort_key);
            assert_eq!(
                planned, whole_listing,
                "{}: units + orphan bucket account for the listing EXACTLY — no artifact \
                 invented, none dropped, none counted twice. The grouping partitions the \
                 listing; it does not merely sample it",
                case.name
            );
        }
    }

    /// The two base-less shapes, separated, and held at every quota.
    ///
    /// `a_base_less_listing_prunes_nothing_and_an_empty_one_is_not_an_anomaly`
    /// above pins the flag and the empty delete sets at `retain = 1`. What this
    /// adds is quota-independence: no value of `retain` makes the anomaly
    /// prunable. A tails-only listing has no unit to retire, so retention is
    /// not entitled to reap the tails as orphans either — a base may simply be
    /// mid-publish, and the orphan bucket is defined as "below the OLDEST
    /// base", which a listing with no base at all does not have. This is the
    /// `no_base` residual, not a retention decision.
    #[test]
    fn a_tails_only_listing_deletes_nothing_at_any_retain_and_an_empty_one_stays_routine() {
        let tails_only = [tail(9), tail(6), tail(4)];

        for retain in 0..=4 {
            let plan = plan_retention(&tails_only, retain, u64::MAX);

            assert!(
                plan.base_less_listing,
                "retain = {retain}: a non-empty listing with no Base is the store-loss anomaly \
                 whatever the quota is"
            );
            assert!(
                delete_set(&plan).is_empty(),
                "retain = {retain}: the D4 guard deletes NOTHING — neither as a unit (there is \
                 none) nor as orphans (erasing them is irreversible and reclaims nothing \
                 restorable; a base restored out-of-band would make them replayable again)"
            );
            assert_eq!(plan.retained_units, 0);

            let empty = plan_retention(&[], retain, u64::MAX);
            assert!(
                !empty.base_less_listing,
                "retain = {retain}: an empty listing is routine lifecycle — never checkpointed, \
                 or retention took the last generation — and must not raise the anomaly a \
                 freshly created layer would then carry on every tick"
            );
            assert!(delete_set(&empty).is_empty());
            assert_eq!(empty.retained_units, 0);
        }
    }

    // ── The prune-ordering guard (b.hka (b)) ────────────────────────────────
    //
    // Rule 5: a listing holding ANY generation whose `seq()` is strictly above
    // the live `obj:hdr seq` prunes NOTHING, in the D4 shape. Three claims,
    // separated: the refusal itself, its boundary (at-or-below is not a
    // refusal), and — the one that keeps the guard from being a silent
    // behaviour change — that on a trustworthy listing the plan is identical to
    // the pre-guard one.
    //
    // WHERE `live_seq` COMES FROM IS NOT PINNED HERE, and cannot be: this is a
    // pure function and every case below hands it a literal. The planner's
    // contract is that the caller supplies a FRESH `HGET obj:hdr seq` with
    // nothing folded into it — no producer-supplied sequence, and specifically
    // not the round-1 `live_seq.max(sequence)`, whose `sequence` at the re-base
    // call site is minted from the stored listing and so imports the untrusted
    // evidence into the trust check (b.hka review round 2; the argument is on
    // `RetentionPlan::generations_above_live_seq`). That obligation is the
    // CALLER's and is pinned at the call site, in `checkpointer.rs`'s
    // `tail_tick_tests`: the pair
    // `a_stale_gate_seq_does_not_refuse_a_prune_the_live_header_admits` /
    // `the_prune_ordering_guard_reads_the_live_header_not_the_gate_value` makes
    // the two inputs disagree in both directions, and the two
    // `prune_*_when_the_*header*` cases cover the read's other two answers.

    /// One generation above `live_seq` empties BOTH delete sets and reports the
    /// count.
    ///
    /// Both sets, not just `prune`: an implementation that refused the lineage
    /// units but still reaped the orphan bucket would destroy exactly the
    /// generations whose provenance is least certain, which is the opposite of
    /// the guard's purpose. `base_less_listing` must stay false so the two
    /// anomalies remain distinguishable at the WARN site — they have different
    /// operator stories, and only this one is self-clearing.
    #[test]
    fn a_generation_above_the_live_seq_refuses_the_whole_prune() {
        // `tail(15)` sits below the oldest base, so a pruning plan would put it
        // in the orphan bucket — the set most at risk from a partial refusal.
        let listing = [base(40), base(20), tail(15)];

        let plan = plan_retention(&listing, 1, 30);

        assert_eq!(
            plan.generations_above_live_seq, 1,
            "exactly base@40 is above the live seq of 30"
        );
        assert!(
            plan.prune.is_empty(),
            "the quota must not be spent on a listing that is not describing the live \
             incarnation of this object_id: at retain = 1 it would otherwise destroy base@20"
        );
        assert!(
            plan.orphan_tails.is_empty(),
            "the orphan bucket is refused too — a partial refusal would reap precisely the \
             generations whose provenance is least certain"
        );
        assert!(
            !plan.base_less_listing,
            "this listing HAS bases; conflating the two anomalies would report a self-clearing \
             condition as suspected store loss"
        );
        assert_eq!(
            plan.retained_units, 1,
            "the refusal still reports what the quota WOULD have kept — `units.len().min(retain)`, \
             here 1 of the 2 units — which is the honest diagnostic the call-site WARN carries \
             (`would_have_retained`). Zeroing it would make the WARN read as if the stem held \
             nothing worth keeping"
        );
    }

    /// Every implicated generation counts, not just the newest — the field is a
    /// count rather than a boolean precisely so the WARN can say how much of the
    /// stem is foreign.
    #[test]
    fn every_generation_above_the_live_seq_is_counted() {
        let listing = [base(40), tail(35), base(20), tail(15)];

        let plan = plan_retention(&listing, 1, 18);

        assert_eq!(
            plan.generations_above_live_seq, 3,
            "base@40, tail@35 and base@20 are all above 18 — the count is the difference \
             between one stray artifact and a wholly foreign stem"
        );
        assert!(plan.prune.is_empty() && plan.orphan_tails.is_empty());
    }

    /// The boundary: `seq() == live_seq` is NOT above it.
    ///
    /// This is the ordinary steady state — the newest generation was cut at the
    /// header's own sequence — so an off-by-one here would refuse every prune on
    /// every layer forever, and the only symptom would be unbounded generation
    /// growth behind a WARN.
    #[test]
    fn a_generation_exactly_at_the_live_seq_does_not_trip_the_guard() {
        let listing = [base(40), base(20)];

        let plan = plan_retention(&listing, 1, 40);

        assert_eq!(plan.generations_above_live_seq, 0);
        assert_eq!(
            delete_set(&plan),
            vec![(20, CheckpointKind::Base)],
            "at the boundary the plan is the ordinary one: base@20 is superseded and pruned"
        );
    }

    /// The regression fence for every pre-guard case in this module: over the
    /// whole counting table and every quota, a listing whose generations all sit
    /// at or below `live_seq` produces a plan EQUAL to the guard-disabled one.
    ///
    /// The sibling cases above all pass `live_seq = u64::MAX`, which is exactly
    /// "the guard is off" — so on their own they cannot tell a guard that is
    /// inert on a trustworthy listing from one that is inert only at
    /// `u64::MAX`. This case re-runs each of them at the tightest `live_seq`
    /// that should still admit the whole listing (its own newest sequence) and
    /// requires `RetentionPlan` equality, which covers delete-set membership,
    /// delete ORDER, the orphan bucket, both flags and the tied-tail count in
    /// one comparison.
    #[test]
    fn a_trustworthy_listing_plans_exactly_as_it_did_before_the_guard() {
        for case in counting_table() {
            let newest = case
                .listing
                .iter()
                .map(CheckpointHandle::seq)
                .max()
                .expect("every counting-table fixture is non-empty");

            for retain in 0..=4 {
                assert_eq!(
                    plan_retention(&case.listing, retain, newest),
                    plan_retention(&case.listing, retain, u64::MAX),
                    "{}: at retain {retain}, a live_seq equal to the listing's own newest \
                     sequence must produce the identical plan to the guard-disabled one — the \
                     guard is a refusal on an untrustworthy listing, never a change of plan on \
                     a trustworthy one",
                    case.name
                );
            }
        }
    }

    /// And the same table one sequence LOWER is refused wholesale — the
    /// anti-vacuity partner to the case above.
    ///
    /// Without this, a `plan_retention` that ignored `live_seq` entirely would
    /// satisfy the equality above on every row.
    #[test]
    fn the_same_listings_one_below_their_newest_are_all_refused() {
        for case in counting_table() {
            let newest = case
                .listing
                .iter()
                .map(CheckpointHandle::seq)
                .max()
                .expect("every counting-table fixture is non-empty");

            let plan = plan_retention(&case.listing, 0, newest - 1);
            assert!(
                plan.generations_above_live_seq >= 1,
                "{}: the newest generation is above live_seq {newest}-1 by construction",
                case.name
            );
            assert!(
                delete_set(&plan).is_empty(),
                "{}: and at retain = 0 — where the guard-disabled plan deletes EVERYTHING — the \
                 refusal must delete nothing at all",
                case.name
            );
        }
    }

    /// An empty listing is not an anomaly under this guard either.
    ///
    /// `live_seq = 0` is the sharpest input: with no handles there is nothing to
    /// compare, so a guard that keyed on the header rather than on the listing
    /// would raise here on every never-checkpointed layer.
    #[test]
    fn an_empty_listing_never_trips_the_prune_ordering_guard() {
        let plan = plan_retention(&[], 1, 0);

        assert_eq!(plan.generations_above_live_seq, 0);
        assert!(!plan.base_less_listing);
        assert!(delete_set(&plan).is_empty());
        assert_eq!(plan.retained_units, 0);
    }

    /// The two refusals compose: a base-less listing that ALSO holds a
    /// generation above `live_seq` reports both signals and still deletes
    /// nothing.
    ///
    /// The base-less guard returns first, so this pins that its early return
    /// carries the prune-ordering count rather than zeroing it — an operator
    /// reading the base-less WARN on a stem that is also mis-addressed would
    /// otherwise be told only half of what is wrong.
    #[test]
    fn a_base_less_listing_above_the_live_seq_reports_both_anomalies() {
        let listing = [tail(9), tail(4)];

        let plan = plan_retention(&listing, 1, 5);

        assert!(plan.base_less_listing);
        assert_eq!(
            plan.generations_above_live_seq, 1,
            "tail@9 is above live_seq 5, and the base-less early return must not drop that \
             count on the floor"
        );
        assert!(delete_set(&plan).is_empty());
    }
}
