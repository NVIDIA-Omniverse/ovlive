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

// Delta application engine.
//
// apply_delta_into(state, Option<OwnedDelta>, &mut MutationSink, &mut ApplyDeltaStats)
//   apply_delta(state, Option<OwnedDelta>)             → Vec<StorageMutation>
//   apply_delta_with_stats(state, delta, &mut stats)   → Vec<StorageMutation>
//
// apply_delta_into_mode(state, delta, sink, stats, ApplyMode)
//   — apply_delta_into is its `ApplyMode::Commit` wrapper; stream replay
//     (`domain::replay`) passes `ApplyMode::Replay`; the checkpoint base
//     decode (`checkpoint::apply`) passes `ApplyMode::RestoreBase`.
//
// apply_delta_bounded(state, delta, MutationAllowance, &mut dyn RewriteBudget, stats)
//   → BoundedPlan
//   — collects while the plan's rewrite beyond the delta's own elements
//     (`RewriteCount`) fits an allowance, asking its `RewriteBudget` for more
//     as the rewrite passes it — at an emission, or at a build site before an
//     object-scaled structure is allocated; once the budget refuses, aborts
//     the plan and reports the exact need (`BoundedPlan::Overflow`).
//
// Every write the engine makes to the state goes through `domain::delta_write`,
// which reports it to the plan's `EngineHooks`: `NoHooks` for every public
// entry point here but one, and the structural cache's persistent plan
// indexes (`domain::plan_index`) for its plan in place
// (`apply_delta_bounded_indexed`, with a pre-image journal,
// `domain::plan_journal`) and its stream repair
// (`domain::replay::apply_decoded_record_indexed`, behind
// `replay_stream_record_indexed`). A guard test pins it
// (`no_direct_table_or_header_mutation_outside_the_writer_module`).
//
// One engine, one set of semantics. The two wrappers pass a collecting sink;
// the callers that discard the mutation list pass `MutationSink::Discard`
// and build none of it (b.ot3 — see the `MutationSink` doc-comment for the
// exhaustive list of those callers); the bounded entry point passes its own
// private sink.
//
// Per-delta planning (b.ot3), all of it stack-local in `ChildPlan`:
//
//   * children-list writes are coalesced to ONE per dirty parent, emitted by
//     `finalize_children_lists` at the close of the structural phase;
//   * `(parent_id, name)` conflict lookups: the first scans `state.sections`
//     with no allocation, the second builds an index used from then on, with
//     a lowest-`section_id` tie-break either way (`find_child_by_name`);
//   * a bounded plan's build sites project the object-scaled structures they
//     are about to allocate — a parent's listing, the name index, the
//     adoption index's ids, a reset's, a cascade's or an adoption's rows —
//     into its rewrite count before allocating them, so its budget is asked
//     first (`BoundedMeter`);
//   * a cascade's field / time-sample erasure is deferred to one purge per
//     delta (`purge_cascaded_rows`), strictly before the delta's own
//     `SetField`s; on the compact form its field half probes the purged
//     sections' rows by interned name when that is cheaper than a scan.
//
// Ordered-command semantics are unchanged: the state is still mutated
// command-by-command in wire order. Residual quadratic cases R1–R5 survive by
// design and are pinned by the `ApplyDeltaStats` operation counters — SDD §9.4.
//
// Pure logic — zero I/O, zero async. See SDD §9.

use ordered_float::OrderedFloat;
use thiserror::Error;

use oodf_ffi::{
    CommandCreateSection, CommandDiffChangeSection, CommandMoveSection, CommandReorderChildren,
    OwnedDelta, SetField as OodfSetField, StructCommand, TimeSample as OodfTimeSample,
};

use crate::domain::alloc_model::{allocation_bytes, table_growth_bytes, vec_growth_bytes};
use crate::domain::delta_write::{self as write, EngineHooks, NoHooks};
use crate::domain::pass_timing::{self, Pass};
use crate::domain::types::{
    Field, FieldValue, ObjectStateOf, Section, SectionKey, TimeSample, ValueRows,
};

// ── Errors ────────────────────────────────────────────────────────────────────

// b.fd9: the pre-fix enum carried an `Oodf(#[from] OodfError)` variant that
// was never constructed by domain code. Every `parse_delta` caller maps
// `OodfError` itself — the gRPC handlers (see `src/grpc/layer.rs`) straight to
// `ServiceError::from(OodfError)`, the checkpoint decoder
// (`src/checkpoint/apply.rs`) to `CheckpointError::Permanent`, checkpoint
// compose (`src/checkpoint/compose.rs`) to `ComposeError::OodfParse`, and
// stream replay (`domain::replay`) to its own `ReplayError` — and the
// `From<DeltaError>` arm for the old
// `Oodf` variant just forwarded to that same impl, so the variant existed
// purely as dead plumbing. Removing it tightens the contract: `DeltaError`
// now represents only the apply-time invariant violations that the domain
// layer actually emits.
#[derive(Debug, Error)]
pub enum DeltaError {
    #[error("inconsistent state: section {section_id} missing during {operation}")]
    InconsistentState {
        section_id: u64,
        operation: &'static str,
    },

    #[error("invalid external-value hash length: got {got} bytes, expected {expected}")]
    InvalidExternalHashLength { got: usize, expected: usize },

    #[error("duplicate section_id {section_id} in CreateSection")]
    DuplicateSection { section_id: u64 },

    #[error(
        "set_order overflow: next_set_order={next} + delta_set_order={delta} exceeds the 2^53 - 1 ceiling"
    )]
    SetOrderOverflow { next: u64, delta: u64 },

    #[error("sequence_num overflow: the next sequence_num would exceed the 2^53 - 1 ceiling")]
    SequenceOverflow,

    #[error("delta bounds exceeded: {field} limit={limit}, actual={actual}")]
    DeltaBoundsExceeded {
        field: &'static str,
        limit: u64,
        actual: u64,
    },

    /// The delta carries more entries — struct commands + set fields + time
    /// samples, summed by [`own_element_count`] — than one delta may hold
    /// ([`MAX_DELTA_ENTRIES`](crate::domain::limits::MAX_DELTA_ENTRIES)).
    /// A delta from the parser never reaches this check over the limit: the
    /// parser refuses it first, with the same counts
    /// (`oodf_ffi::OodfError::TooManyTables`).
    #[error(
        "delta carries {} entries ({struct_commands} struct commands, {set_fields} set fields, \
         {time_samples} time samples); one delta may carry at most {max_entries} entries in total",
        struct_commands.saturating_add(*set_fields).saturating_add(*time_samples)
    )]
    TooManyEntries {
        struct_commands: u64,
        set_fields: u64,
        time_samples: u64,
        max_entries: u64,
    },

    /// A `ReorderChildren` command names more children than one list may
    /// hold ([`MAX_CHILDREN_LIST_LEN`](crate::domain::limits::MAX_CHILDREN_LIST_LEN)).
    /// Its own variant, so the refusal's reason travels as a type rather than
    /// as a field name; its text is the per-field bound's.
    #[error(
        "delta bounds exceeded: ReorderChildren.children_list.len limit={limit}, actual={actual}"
    )]
    ChildrenListTooLong { limit: u64, actual: u64 },

    /// b.39x medium: a delta name field (`section_name`, `new_name`,
    /// `key_name`) contained NUL or an ASCII control character (0x01..=0x1F).
    /// These flow into Valkey hash keys, structured tracing spans, and
    /// per-commit notification payloads — same threat model as the
    /// `validate_client_id` / `validate_base_path` policy. Caught at
    /// `validate_delta_bounds` alongside the existing byte-length check.
    #[error("delta name {field} contains NUL or control character")]
    NameContainsControlChar { field: &'static str },

    /// b.39x H: a `TimeSample` carried a non-finite `time` key (NaN, +∞,
    /// −∞). `OrderedFloat<f64>` wraps these without round-trip semantics —
    /// the materialized HashMap key is unstable across re-parses, and the
    /// Lua fast path coerces NaN to the literal string `"nan"`. Reject at
    /// validation time so neither backend sees a non-finite key.
    #[error("time sample at index {index} has non-finite time value: {time}")]
    TimeSampleNonFiniteTime { index: usize, time: f64 },

    /// A `TimeSample` carried the time `-0.0`. The parser canonicalizes every
    /// time to `+0.0` (`oodf_ffi::canonical_sample_time`), so no client delta
    /// reaches this: it fails closed on a delta built inside the service that
    /// bypassed the parser, which would otherwise store a `-0.0` row — a
    /// second stored key for the USD time code `0.0`. Caught at
    /// `validate_delta_content` beside the non-finite check, so on every
    /// apply, a checkpoint base decode (`ApplyMode::RestoreBase`) included.
    #[error(
        "time sample at index {index} has time -0.0; time codes must be canonical (0.0 for zero)"
    )]
    TimeSampleNegativeZeroTime { index: usize },

    /// b.39x medium: a structural command set `parent_id == section_id`,
    /// which would wire a section as its own parent. Self-parenting
    /// produces cycles in `state.children` that break tree walks
    /// (cascade-delete, diff reconstruction) with stack overflow or
    /// infinite loops. Caught at `validate_delta_bounds` before
    /// `apply_delta` touches the tree.
    #[error("{command} on section {section_id} sets parent_id == section_id (self-parenting)")]
    SelfParenting {
        command: &'static str,
        section_id: u64,
    },

    /// b.dmu item 2: `DiffChangeSection` would rename a section onto an
    /// existing sibling's `(parent_id, name)` slot. Mirrors `MoveSection`'s
    /// edge 4 / 5 conflict-detection **on the cross-parent move path only**:
    /// `MoveSection`'s same-parent rename (`old_parent_id ==
    /// new_parent_id`) performs no conflict check at all, by design (PRD
    /// §2.3.4, SDD §9.3.1), so it remains the one command shape that can
    /// produce duplicate `(parent_id, name)` siblings. When it does, the
    /// `conflicting_id` reported here is the lowest such `section_id` —
    /// see [`find_child_by_name`]. Without this guard the diff path
    /// silently stores two siblings with the same name, and the corruption
    /// surfaces on the *write* path: the engine's own `(parent_id, name)`
    /// conflict lookup — the one deciding which sibling a later
    /// `CreateSection`, `DiffChangeSection` or `MoveSection`
    /// cascade-deletes — then has to pick between the two (pre-b.ot3
    /// arbitrarily, now the lowest `section_id`), and that pick destroys a
    /// subtree in place of its namesake. Read-path diff reconstruction does
    /// not key by name (SDD §10.1). Rejected before the rename mutates
    /// state.
    #[error(
        "DiffChangeSection rename of section {section_id} under parent {parent_id} \
         conflicts with existing sibling section {conflicting_id} named {name:?}"
    )]
    DuplicateSiblingName {
        section_id: u64,
        parent_id: u64,
        conflicting_id: u64,
        name: String,
    },

    /// b.dmu item 3: the materialized section tree would exceed
    /// `MAX_SECTION_TREE_DEPTH` after this delta. The read-path
    /// `DiffEngine::sections_in_dfs_order` enforces the same cap, but
    /// catching it on the write path is the only path that closes the
    /// "unrecoverable stuck layer" scenario — once deep state is stored,
    /// `cascade_delete_section` can't walk it to clean up.
    #[error(
        "section-tree depth {depth} exceeds cap {cap}; \
         delta would leave the materialized tree unrecoverable on the read path"
    )]
    SectionTreeTooDeep { depth: usize, cap: usize },
}

// ── StorageMutation ───────────────────────────────────────────────────────────

/// All possible Valkey operations produced by apply_delta.
///
/// Variants carry pure domain types — no Valkey/fred types.
/// The storage layer translates these into actual Valkey commands.
///
/// **Section pre-images.** `SetSection::previous` and `DeleteSection`'s
/// `(parent_id, name)` describe the section row as it stood **immediately
/// before this mutation**, per mutation and in emission order: the in-state
/// row the mutation replaces or removes, which may itself have been written
/// earlier in the same delta (by an earlier command, or an earlier mutation
/// of the same command). `parent_id` is always the row's *stored* parent,
/// never a client-supplied `old_parent_id`.
///
/// No structural record carries these fields and neither commit path writes
/// anything from them; only the structural commit's plan fingerprint reads
/// them, as plan identity. The structural commit's section digests come from
/// the pre-plan state, never from these fields — a per-mutation pre-image can
/// name a row an earlier mutation of the same delta wrote, not the stored one.
///
/// The rule is checkable from the mutation list alone: replaying the
/// `SetSection` / `DeleteSection` pre-images in order against the pre-delta
/// set of `(parent_id, name, section_id)` rows lands exactly on the
/// post-delta set.
#[derive(Debug, Clone, PartialEq)]
pub enum StorageMutation {
    /// Upsert a section (HSET le:{oid}:obj:sec:{section_id} …).
    SetSection {
        section: Section,
        /// Key of the row this upsert replaces: `None` exactly when no row
        /// with `section.section_id` existed immediately before this
        /// mutation; `Some` whenever one did, even when its
        /// `(parent_id, name)` equals `section`'s — what an unchanged key
        /// means is the consumer's decision.
        previous: Option<SectionKey>,
    },

    /// Remove a section from the live index and delete its hash
    /// (SREM le:{oid}:obj:secs, DEL le:{oid}:obj:sec:{section_id}).
    ///
    /// `parent_id` and `name` are the removed row's stored values
    /// immediately before this mutation.
    DeleteSection {
        section_id: u64,
        parent_id: u64,
        name: String,
    },

    /// Create a tombstone for diff reconstruction
    /// (HSET le:{oid}:obj:del:{section_id} first_ver … last_ver …,
    ///  SADD le:{oid}:obj:dels {section_id}).
    CreateTombstone {
        section_id: u64,
        first_version: u64,
        last_version: u64,
    },

    /// Delete a section's children list (DEL le:{oid}:obj:ch:{section_id}).
    DeleteChildrenList { section_id: u64 },

    /// Upsert the children list for a section
    /// (replace le:{oid}:obj:ch:{section_id} with the given order).
    SetChildren { section_id: u64, children: Vec<u64> },

    /// Upsert a field value.
    SetField { field: Field },

    /// Delete a field (DEL le:{oid}:obj:fld:{section_id}:{key_name}, remove from sorted set).
    DeleteField { section_id: u64, key_name: String },

    /// Upsert a time sample.
    SetTimeSample { sample: TimeSample },

    /// Delete a time sample.
    DeleteTimeSample {
        section_id: u64,
        time: OrderedFloat<f64>,
    },

    /// Update the object header (sequence_num, first_version, source_format).
    SetObjectHeader {
        object_id: u64,
        sequence_num: u64,
        first_version: u64,
        source_format: Option<String>,
    },
}

impl StorageMutation {
    /// The section whose stored children list this mutation deletes: a
    /// `DeleteChildrenList`'s, or an empty `SetChildren`'s (the storage
    /// layer's `DEL` without `RPUSH`); `None` for every other mutation.
    ///
    /// The one definition the structural cache's dangling-parent upkeep reads
    /// a commit's list deletes by (`storage::structural_cache_repair::
    /// advance_to_successor`), whether from a collected mutation list or
    /// through [`MutationSink::ListDeletes`], which keeps exactly these ids.
    pub fn deleted_list(&self) -> Option<u64> {
        match self {
            StorageMutation::DeleteChildrenList { section_id } => Some(*section_id),
            StorageMutation::SetChildren {
                section_id,
                children,
            } if children.is_empty() => Some(*section_id),
            _ => None,
        }
    }
}

// ── apply_delta ───────────────────────────────────────────────────────────────

/// Observability stats collected during a single `apply_delta` run (b.39x
/// item 7). Returned alongside the mutations so the calling layer can
/// surface counters / log lines that the pure-domain engine itself cannot
/// reach (no access to `Metrics` from `domain`). Empty struct by default —
/// add fields here whenever a new "silent skip" path needs to be made
/// visible to operators without breaking the apply-engine API.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct ApplyDeltaStats {
    /// Count of `CommandCreateSection` commands that were silently
    /// skipped because `parent_id != 0 && state.sections` did not contain
    /// the parent (SDD §9.1.4c). Combined with the H4 "base_version
    /// unvalidated" footgun, an operator inspecting an unexpectedly-empty
    /// layer needs a way to see whether deltas are being silently dropped
    /// at this rung. The caller is expected to bump
    /// `delta_create_section_skipped_missing_parent_total` by this value.
    pub create_section_skipped_missing_parent: u64,

    /// Number of `u64` elements shifted inside a `state.children` vector
    /// (b.ot3 part F). `Vec::insert(p)` into a list of length `L` shifts
    /// `L - p` elements; `Vec::remove(p)` shifts `L - p - 1`. Bumped at
    /// exactly four sites, so a test can derive the expected value:
    ///
    /// * [`insert_child_into_parent`], once per insertion that actually
    ///   happens — `L - insert_pos`, and nothing at all when the child is
    ///   already listed (residual **R1**);
    /// * [`remove_child_from_parent`], once per removal of a child that was
    ///   present in a list free of duplicate ids — `L - pos - 1` (residual
    ///   **R2**). The legacy duplicate-id `retain` path bumps nothing;
    /// * [`rebuild_children_list`]'s sorted arm, **twice per call that
    ///   actually re-places an element** — once for the removal shift and
    ///   once for the reinsertion shift (residual **R3**). Three of its
    ///   exits bump nothing: the unsorted arm, because `sort_by_key` does
    ///   not report element moves; the legacy duplicate-id list, which takes
    ///   that same `sort_by_key` arm; and the early return taken when the
    ///   child is not listed under this parent at all (a legacy row its
    ///   parent's list lost).
    ///
    /// R1–R3 remain Θ(N²) element moves for Θ(N) input commands, by
    /// design — coalescing the *emitted* mutations does not turn a `Vec`
    /// into an order-statistics tree. See SDD §9.4.
    pub children_element_moves: u64,

    /// Number of children-list elements examined by a **linear** scan over a
    /// `state.children` vector or its cached `orders` (b.ot3 part F). Bumped
    /// at exactly four sites:
    ///
    /// * [`insert_child_into_parent`]'s unsorted-fallback insertion-point
    ///   search (residual **R4**) — the elements the `position` predicate
    ///   examined. The sorted arm's `partition_point` binary search bumps
    ///   nothing;
    /// * [`append_order`]'s unsorted-fallback largest-key search, the R4
    ///   counterpart on the append side — every cached order, because a
    ///   maximum has to examine them all. The sorted arm reads the last
    ///   cached order and bumps nothing. Both unsorted arms are reachable
    ///   only from the stored shapes [`ParentCache::sorted`] lists;
    /// * [`remove_child_from_parent`]'s `position` search for the child's
    ///   index — `pos + 1` — on every removal of a present child from a
    ///   duplicate-free list, which includes every cascade delete and every
    ///   cross-parent move, not only explicit removals;
    /// * [`rebuild_children_list`]'s sorted-arm `position` search for the
    ///   re-placed child's index — `pos + 1` — on every call that finds the
    ///   child listed in a sorted, duplicate-free list. The early return
    ///   for a legacy row its parent's list lost, taken when the `position`
    ///   search runs off the end, bumps nothing; neither does the `sort_by_key`
    ///   arm, which never runs the search and which takes both the unsorted
    ///   and the legacy duplicate-id list.
    ///
    /// Because removals contribute, `children_scan_steps == 0` does **not**
    /// mean "every insertion took the binary-search arm": a delta that only
    /// deletes sections has zero insertions and a non-zero count. The
    /// binary-search assertion has to be read off a delta with no removal
    /// and no `rebuild_children_list` call — i.e. insertions only — under a
    /// parent whose loaded list was sorted.
    pub children_scan_steps: u64,

    /// Number of `(parent_id, name)` conflict lookups performed — one per
    /// [`find_child_by_name`] call, which each command reaches only past its
    /// own earlier gates:
    ///
    /// * `DiffChangeSection`: always one (the probe precedes every branch).
    /// * `CreateSection`: one, unless the command returned first — a
    ///   `section_id` that already has a row (`DuplicateSection`) or a
    ///   non-root `parent_id` with no row (silently skipped, SDD §9.1.4c)
    ///   both probe **zero** times.
    /// * `MoveSection`: one, only when the command reaches the
    ///   destination-conflict check at edges 4/5 — that is, the section
    ///   exists, `old_parent_id != new_parent_id` (a same-parent rename
    ///   returns first), the new parent exists (edge 2), and the move is not
    ///   a cycle (edge 3). Each of those earlier exits probes **zero**
    ///   times.
    /// * `DeleteSection`, `ReorderChildren`, `DiffDeleteSection`: never.
    pub name_index_probes: u64,

    /// Number of times the per-delta `(parent_id, name)` index was built
    /// from `state.sections`. At most 1 per delta (the first probe scans
    /// `state.sections` instead, the second builds the index, and it is
    /// maintained incrementally afterwards), so `name_index_builds <= 1` is
    /// the assertion that the pre-b.ot3 per-command O(S) rescan is gone; a
    /// delta that probes once reads 0. Always 0 for a delta whose hooks keep
    /// a persistent index (the structural cache's, `domain::plan_index`): its
    /// probes are answered there, with no scan and no build.
    pub name_index_builds: u64,

    /// Number of times the per-delta [`AdoptionIndex`] was built from
    /// `state.sections`. At most 1 per delta (the index is built lazily by
    /// the delta's first birth in [`adopt_orphans`] and maintained
    /// incrementally afterwards), and 0 for a delta that births no section,
    /// so `adoption_index_builds <= 1` is the assertion that adoption does not
    /// pay an O(S) rescan per birth. Always 0 for a delta whose hooks keep a
    /// persistent index (`domain::plan_index`), which answers its adoptions.
    pub adoption_index_builds: u64,

    /// Number of children-list persistence mutations emitted
    /// (`SetChildren` + `DeleteChildrenList`, inline **and** from the
    /// per-delta finalization pass). Creating N siblings under one parent
    /// emits exactly 1, versus N before b.ot3.
    ///
    /// Counted even when the sink is [`MutationSink::Discard`] — the
    /// counters describe the *plan*, not what a particular caller chose to
    /// keep, so a `Discard` run and a `Collect` run of the same delta
    /// produce identical stats.
    pub children_writes_emitted: u64,

    /// Number of deferred cascade field/time-sample purges run (0 or 1 per
    /// delta — see [`purge_cascaded_rows`]), whichever arm the compact field
    /// purge took (the stat is the same on both forms). The reset path's
    /// own erasure is *not* counted here: it is a separate drain of each map
    /// that predates b.ot3 and is unchanged.
    pub field_purge_passes: u64,

    /// Number of times a parent's [`ParentCache`] was derived from the live
    /// maps — one per [`ChildPlan::build_cache`] call, which is the only
    /// constructor. A build over an empty children list counts: the point of
    /// the counter is "how often did a reader have to re-derive", and an
    /// empty build still means the previous cache was dropped.
    ///
    /// A build count on its own cannot distinguish re-deriving an empty list
    /// from re-deriving a million-entry one, which is why it is always read
    /// next to [`ApplyDeltaStats::parent_cache_entries_built`].
    pub parent_cache_builds: u64,

    /// Number of listed children read across all of those derivations —
    /// `state.children[parent].len()` summed over every
    /// [`ChildPlan::build_cache`] call. This is the quantity that actually
    /// scales: one build over a list of length `C` reads `C` entries, does
    /// `C` `state.sections` lookups, and allocates a `C`-element membership
    /// set and a `C`-element order vector.
    ///
    /// Creating N ascending-ordered siblings under one parent must keep this
    /// linear in N. A design that drops and re-derives the growing list on
    /// every insertion makes it N(N−1)/2 — 31,996,000 at N = 8,000 — while the
    /// six counters above read identically on the defective and the fixed
    /// engine: `children_element_moves` / `children_scan_steps` /
    /// `name_index_probes` / `name_index_builds` / `children_writes_emitted` /
    /// `field_purge_passes` are `0 / 0 / n / 1 / 1 / 0` either way, so not one
    /// of them can discriminate the defect. That is the blind spot this counter
    /// was added to close.
    pub parent_cache_entries_built: u64,

    /// Number of listing-parent entries inspected by
    /// [`ChildPlan::invalidate_order_listeners`]: one per parent the index
    /// still names for the written child, `0` when it names none. Counted
    /// whether or not that parent's cache is actually dropped — and whether or
    /// not it still has one — because the inspection is the work.
    ///
    /// Deliberately scoped to the invalidation walk rather than to index
    /// bookkeeping in general. De-registering a child on the list side is one
    /// O(1) map lookup per membership change, already bounded one-for-one by
    /// the `children_*` counters; the walk is the part whose length depends
    /// on how many caches name the child, so it is the part that would grow
    /// again if anyone replaced the index with a scan over every touched
    /// parent.
    ///
    /// The bound is one step per parent the **index still names**, which is
    /// one in steady state. It rises above one for either of two reasons, and
    /// a reader diagnosing a large value has to consider both. Stored state
    /// can carry a stale or ghost listing, putting the child in two parents'
    /// caches at once. Or a cascade left pairs behind: dropping a parent's
    /// cache wholesale does not de-register its children, deliberately — see
    /// [`ChildPlan::listed_in`], where removes are required only for the cost
    /// bound — so a child that was cached under a since-cascaded parent, and
    /// is later re-registered under a live one, is named by both.
    pub cache_dependency_steps: u64,

    /// High-water mark of the `(child, parent)` pairs stored in
    /// [`ChildPlan::listed_in`] — the most the dependency index ever held at
    /// one instant during the delta, not its size when the delta ended. The
    /// index shrinks as well as grows (every invalidation walk takes its
    /// child's whole entry out, and every membership change that leaves the
    /// parent's cache populated de-registers one pair), so a final-size
    /// reading would miss the moment that sets the memory bill. The peak is
    /// therefore folded in at each add rather than measured at the end.
    ///
    /// **Pairs, not distinct children.** The unit is stored entries, because
    /// that is what the map costs: a child two populated caches list is two
    /// pairs and a [`CacheListers::Many`] vector, and a pair a cascade left
    /// behind still counts for as long as it is stored. Counting children
    /// instead would report the same figure for a one-lister and a two-lister
    /// child, hiding exactly the cases where the index outgrows one entry per
    /// child.
    ///
    /// A bound on this counter is derived from the delta's own input — the sum
    /// over touched parents of the initial `state.children[P]` length, plus one
    /// per insertion, plus whatever extra listings stored state carries — never
    /// from a number read back off a run, which would re-assert whatever the
    /// implementation currently does.
    ///
    /// This is the one counter that describes state held *at once*.
    /// [`ApplyDeltaStats::parent_cache_entries_built`] measures work re-done
    /// and the two diverge in both directions: re-deriving a growing list on
    /// every command inflates that one while leaving this one flat, and an
    /// index that never dropped a pair would inflate this one while leaving
    /// that one alone.
    pub cache_index_peak_pairs: u64,
}

// ── MutationSink ──────────────────────────────────────────────────────────────

/// Where [`apply_delta_into`] sends the persistence mutations it plans.
///
/// b.ot3: most `apply_delta` call sites in the service would drop the
/// returned `Vec<StorageMutation>` on the floor. Building it anyway is the
/// dominant allocation on those paths — after children-list coalescing the
/// biggest remaining item is the `FieldValue::Inline(Vec<u8>)` payload
/// cloned into every `SetField`. [`MutationSink::push_with`] takes a
/// *closure* rather than a built mutation precisely so the [`Discard`] arm
/// never runs those clones: no `Section::clone`, no inline-payload clone, no
/// children `Vec::clone`.
///
/// [`Discard`]: MutationSink::Discard
///
/// Deliberately **no** `Default` and **no** `Clone`. A `Default` impl would
/// let a caller acquire a silently-empty sink by accident — the same
/// silent-empty-output footgun that the snapshot reader's children-list
/// `unwrap_or_default()` was before b.zvd. Picking `Discard` must always be
/// an explicit, reviewed act at the call site.
///
/// # The only legitimate `Discard` callers
///
/// 1. `handle_create_layer`'s empty-content branch (`src/grpc/layer.rs`) —
///    CreateLayer materializes the whole object from `ObjectState` via
///    `queue_object_state`, never from the delta's mutation list.
/// 2. `handle_create_layer`'s parsed-delta branch (same file, same reason);
///    this is the path the 8,000-sibling blowup in b.ot3 was measured on.
/// 3. `decode_checkpoint_ompf` (`src/checkpoint/apply.rs`) — the DR decoder
///    rebuilds state from checkpoint bytes and writes it out itself; it runs
///    in [`ApplyMode::RestoreBase`].
/// 4. `compose_lineage`'s record-replay loop (`src/checkpoint/compose.rs`) —
///    replay exists to reproduce an `ObjectState`, not to re-commit history.
/// 5. `replay_stream_record` (`src/domain/replay.rs`) — re-applies a
///    committed layer-stream entry to a decoded state, for the same reason
///    as (4); it also runs in [`ApplyMode::Replay`]. Its sibling
///    `replay_stream_record_into` runs the same step with the caller's sink,
///    so its `Discard` callers are listed on their own (6).
/// 6. `replay_chunk` (`src/storage/structural_cache_repair.rs`) — the
///    structural cache's stream repair replays through `Discard` while the
///    entry's `dangling_parents` is empty, for the same reason as (4) and
///    (5); while it is non-empty it replays through
///    [`MutationSink::ListDeletes`], to read off the deletes of dangling
///    children lists.
///
/// Every other caller — in particular `LayerStore::update_layer`, the only
/// consumer of the mutation list — goes through [`apply_delta`] or
/// [`apply_delta_with_stats`], which collect, or [`apply_delta_bounded`],
/// which collects within an allowance through a sink of its own.
#[derive(Debug)]
pub enum MutationSink {
    /// Collect every planned mutation, in emission order.
    Collect(Vec<StorageMutation>),
    /// Plan the delta but build no mutations at all.
    Discard,
    /// Keep only the ids of the children lists the plan deletes
    /// ([`StorageMutation::deleted_list`]: a `DeleteChildrenList`, or an
    /// empty `SetChildren`), in emission order; every other mutation is
    /// dropped as soon as it is built.
    ///
    /// For the structural cache's stream repair on an entry with dangling
    /// parents (`storage::structural_cache_repair`): those deletes are all
    /// it reads off a replayed record, and a collecting sink would hold a
    /// copy of every row the record writes or a cascade deletes — up to
    /// three mutations per deleted section plus one per purged row — beside
    /// the entry. This one holds a `u64` per deleted list and the one
    /// mutation being built (`domain::delta::list_delete_sink_bytes`).
    ListDeletes(Vec<u64>),
}

impl MutationSink {
    /// A fresh collecting sink.
    pub fn collecting() -> Self {
        MutationSink::Collect(Vec::new())
    }

    /// A fresh sink keeping only the deleted lists' ids
    /// ([`MutationSink::ListDeletes`]).
    pub fn list_deletes() -> Self {
        MutationSink::ListDeletes(Vec::new())
    }

    /// Emit one mutation, building it only if this sink collects or keeps
    /// list deletes.
    ///
    /// `make` must be side-effect-free: on the [`MutationSink::Discard`] arm
    /// it is never called. State mutation belongs at the call site, not
    /// inside the closure.
    fn push_with(&mut self, make: impl FnOnce() -> StorageMutation) {
        match self {
            MutationSink::Collect(mutations) => mutations.push(make()),
            MutationSink::Discard => {}
            MutationSink::ListDeletes(ids) => {
                // Built, read and dropped: one mutation alive at a time.
                if let Some(id) = make().deleted_list() {
                    ids.push(id);
                }
            }
        }
    }
}

/// What the engine's helpers emit into: a caller's [`MutationSink`], or the
/// [`BoundedSink`] of [`apply_delta_bounded`]. Private, so the public sink
/// stays the enum its callers match on, without a bounded variant.
trait PlanSink {
    /// Emit one mutation; see [`MutationSink::push_with`] for the contract on
    /// `make`.
    fn push_with(&mut self, make: impl FnOnce() -> StorageMutation);

    /// Whether this sink keeps a value-row delete it is given. A sink that
    /// keeps none — [`MutationSink::Discard`], and [`MutationSink::ListDeletes`],
    /// which keeps only children-list deletes — lets a plan skip work whose
    /// only output is such a delete ([`ZeroTimeDeletes`]).
    fn keeps_row_deletes(&self) -> bool {
        true
    }
}

impl PlanSink for MutationSink {
    fn push_with(&mut self, make: impl FnOnce() -> StorageMutation) {
        MutationSink::push_with(self, make);
    }

    fn keeps_row_deletes(&self) -> bool {
        matches!(self, MutationSink::Collect(_))
    }
}

/// The mutations a plan makes beyond the delta's own elements, the part of a
/// structural `UpdateLayer`'s memory that grows with the object it edits
/// rather than with the delta (SDD §5.4).
///
/// Counted over the plan's emitted [`StorageMutation`]s and, in a bounded
/// plan ([`apply_delta_bounded`]), over what its build sites project before
/// they allocate:
///
/// - `rows`: the mutations emitted, every kind counted once, the header
///   included, less the delta's own elements ([`own_element_count`], which
///   the upload's element charge already covers), saturating at zero. A
///   reorder that re-keys every child of a parent counts one row per re-keyed
///   child however short its list; a cascade or a reset counts the rows it
///   deletes. A reset projects its drain, a cascade its subtree, and a birth
///   each orphan it adopts, as rows before it builds the scratch that finds
///   them; each emitted row retires one projected row;
/// - `list_entries`: the child ids of every children list the plan rewrites
///   or lists, per parent the larger of its listing's projection (the
///   parent's whole list, projected before the plan builds its listing) and
///   the length of the [`StorageMutation::SetChildren`] list it emits. A
///   parent projected and never emitted keeps its projection;
/// - `indexed_sections`: the sections the plan's `(parent_id, name)` index
///   holds — the object's sections plus the entries the delta can add to it
///   ([`StructuralInserts::name_index_inserts`]: its births and moves),
///   projected when the plan builds the index at its second name probe, 0
///   for a plan that builds none. Plan scratch only: the mutation list holds
///   none of it;
/// - `adoption_entries`: the ids the plan's adoption index ([`AdoptionIndex`])
///   records — every row naming an id the delta can birth, live or not —
///   each projected before it is recorded, by the index's build and by the
///   name hook that keeps it, 0 for a plan that builds none. Plan scratch
///   only, never retired: the index lives until the plan ends;
/// - `walked_sections`: the sections a hooked plan's end-of-delta depth check
///   walks (`domain::plan_index`): the incremental check projects each hop
///   of its walks up before it records it, and each children list's length
///   before it pushes the list onto its walk down, and a full walk projects
///   every section before it starts; 0 for a plan without hooks. Plan
///   scratch only, never retired before the plan ends;
/// - `rollback_rebuild_bytes`: bytes, not a count, priced 1:1 — what a
///   journalled plan's rollback could rebuild, an upper bound, each rebuild
///   priced exactly (`domain::plan_journal`, "Rollback without growth"):
///   - the tables it may have to rebuild at their pre-plan length, because
///     the plan removed more of their keys than they have room to
///     re-create, each projected once, before the write that can make it
///     necessary, at `table_bytes(pre-plan length)`;
///   - the name-index vectors the plan drops with their parent keys, each
///     projected as it is dropped, at its capacity: the most restoring that
///     parent's vector allocates. A later drop of the same parent's key
///     projects again, an over-count.
///
///   No headroom; 0 for a plan without a journal. Never retired before the
///   plan ends.
///
/// What a bounded plan asks its [`RewriteBudget`] for, and its
/// [`BoundedPlan::Overflow`] need, are the kept count plus the projections
/// not yet retired (the name index, the adoption index, the depth walk and
/// the rollback rebuilds included); [`BoundedPlan::Within`] reports the kept
/// count alone, with `indexed_sections`, `adoption_entries`,
/// `walked_sections` and `rollback_rebuild_bytes` 0, since only the mutation
/// list outlives the plan (the cached route's CR2 holds the rebuilds the
/// plan's rollback could still need beside it, in its growth term).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RewriteCount {
    pub rows: u64,
    pub list_entries: u64,
    pub indexed_sections: u64,
    pub adoption_entries: u64,
    pub walked_sections: u64,
    pub rollback_rebuild_bytes: u64,
}

impl RewriteCount {
    /// `row_bytes × rows + list_entry_bytes × list_entries +
    /// index_entry_bytes × indexed_sections + adoption_entry_bytes ×
    /// adoption_entries + walk_entry_bytes × walked_sections +
    /// rollback_rebuild_bytes`, saturating.
    pub fn bytes(
        self,
        row_bytes: u64,
        list_entry_bytes: u64,
        index_entry_bytes: u64,
        adoption_entry_bytes: u64,
        walk_entry_bytes: u64,
    ) -> u64 {
        row_bytes
            .saturating_mul(self.rows)
            .saturating_add(list_entry_bytes.saturating_mul(self.list_entries))
            .saturating_add(index_entry_bytes.saturating_mul(self.indexed_sections))
            .saturating_add(adoption_entry_bytes.saturating_mul(self.adoption_entries))
            .saturating_add(walk_entry_bytes.saturating_mul(self.walked_sections))
            .saturating_add(self.rollback_rebuild_bytes)
    }

    /// Each count the larger of `self`'s and `other`'s.
    fn max(self, other: Self) -> Self {
        Self {
            rows: self.rows.max(other.rows),
            list_entries: self.list_entries.max(other.list_entries),
            indexed_sections: self.indexed_sections.max(other.indexed_sections),
            adoption_entries: self.adoption_entries.max(other.adoption_entries),
            walked_sections: self.walked_sections.max(other.walked_sections),
            rollback_rebuild_bytes: self
                .rollback_rebuild_bytes
                .max(other.rollback_rebuild_bytes),
        }
    }
}

/// A delta's own elements: struct commands + set fields + time samples,
/// saturating. The one definition of the count: the per-delta limit
/// ([`MAX_DELTA_ENTRIES`](crate::domain::limits::MAX_DELTA_ENTRIES), checked
/// by `validate_delta_bounds`) bounds it, the upload's element charge
/// (`grpc::upload_admission::DeltaFootprint`) charges it, and
/// [`RewriteCount::rows`] excludes it for that reason, so they must agree.
pub fn own_element_count(delta: &OwnedDelta) -> u64 {
    let count = |n: usize| u64::try_from(n).unwrap_or(u64::MAX);
    count(delta.struct_commands.len())
        .saturating_add(count(delta.set_fields.len()))
        .saturating_add(count(delta.time_samples.len()))
}

/// What a delta's structural commands can add to the keyed tables a plan
/// inserts into, counted from the delta alone: the one definition of each
/// bound, read both by the engine, which sizes its `(parent_id, name)` index
/// from it (`find_child_by_name`), and by the cached route, which charges
/// the tables its plan can grow in place by it
/// (`storage::structural_cache::PlannedInserts`). Each bound counts the
/// `insert` calls the plan's commands can make on that table, at most one
/// per count. An insert call that overwrites an existing key counts too: a
/// std `HashMap::insert` reserves its slot before it looks the key up, so an
/// overwrite on a table with no room left grows it as a new key would.
/// Removing a key makes no insert call. So a table whose length plus its
/// bound stays within its capacity never grows inside the plan. The plan's
/// rollback, if it has one, is not bounded here: it re-creates only the keys
/// the plan removed, into a table rebuilt at its pre-plan length when it
/// lacks the room, which the plan charges as it goes
/// ([`RewriteCount::rollback_rebuild_bytes`]).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct StructuralInserts {
    /// `CreateSection` and `DiffChangeSection` commands — the commands that
    /// can birth a section. A `DiffChangeSection` of a live row births none,
    /// so this is an upper bound.
    pub births: u64,
    /// `MoveSection` commands.
    pub moves: u64,
    /// `ReorderChildren` commands.
    pub reorders: u64,
}

impl StructuralInserts {
    /// The structural inserts `delta` can make.
    pub fn of(delta: &OwnedDelta) -> Self {
        let mut inserts = Self::default();
        for command in &delta.struct_commands {
            match command {
                StructCommand::CreateSection(_) | StructCommand::DiffChangeSection(_) => {
                    inserts.births = inserts.births.saturating_add(1);
                }
                StructCommand::MoveSection(_) => {
                    inserts.moves = inserts.moves.saturating_add(1);
                }
                StructCommand::ReorderChildren(_) => {
                    inserts.reorders = inserts.reorders.saturating_add(1);
                }
                StructCommand::DeleteSection(_) | StructCommand::DiffDeleteSection(_) => {}
            }
        }
        inserts
    }

    /// `state.sections` insert calls: one per birth — a `CreateSection`'s
    /// new row, or a `DiffChangeSection`'s row, new or rewritten
    /// (`write::insert_section`). A move, a reorder and an adoption edit rows
    /// in place (`write::update_section`, no insert call), and a delete and a
    /// reset only remove them.
    pub fn sections(self) -> u64 {
        self.births
    }

    /// `state.children` insert calls: two per birth, one per move and one per
    /// reorder. A list key is created only by `insert_child_into_parent`
    /// (`write::ensure_list`), for a live parent, or the root, with no list
    /// yet, and its callers reach it at most this often per command: a
    /// `CreateSection`, or a `DiffChangeSection` of a new row, lists the row
    /// under its parent and then adopts the rows naming it into a list keyed
    /// by the newborn (`adopt_orphans`) — two keys; a `DiffChangeSection`
    /// that reparents a live row lists it under the new parent — one; a
    /// `MoveSection` lists the section under its destination on its normal
    /// and rename-in-place arms — one. A `ReorderChildren` installs its
    /// parent's re-derived list over the existing key (`write::set_list`) —
    /// one overwrite. A delete and a reset make none.
    pub fn children_list_keys(self) -> u64 {
        self.births
            .saturating_mul(2)
            .saturating_add(self.moves)
            .saturating_add(self.reorders)
    }

    /// Insert calls on the parent-keyed map of the structural cache's
    /// persistent name index (`domain::plan_index::PlanIndexes`): two per
    /// birth, one per move and one per reorder. Every row write takes the
    /// row's entry out of its parent's vector and puts it back (under the new
    /// parent, if it moved), and the map gains a key — a new parent's, or the
    /// same parent's back after its only entry went out — only on that put
    /// back, at most once per row write. A command writes at most this many
    /// rows whose parent can have had no other entry: a birth its own row and
    /// at most one adopted orphan (the newborn's orphans all name it, so only
    /// a lone one empties its vector); a move its own row; a reorder at most
    /// one re-keyed row (its re-keyed rows all name its parent, so only a
    /// lone one does). A cascade only removes rows.
    pub fn name_key_inserts(self) -> u64 {
        self.births
            .saturating_mul(2)
            .saturating_add(self.moves)
            .saturating_add(self.reorders)
    }

    /// Entries the plan's per-delta `(parent_id, name)` index (a plan without
    /// persistent indexes) can gain after it is built: one per birth and one
    /// per move. `ChildPlan::name_inserted`
    /// is called at most once by each `CreateSection` and `DiffChangeSection`
    /// (a birth, a rename or a reparent; none for a create skipped for a
    /// missing parent or a command that fails) and at most once by each
    /// `MoveSection` (whichever of its rename and move arms runs), and by
    /// nothing else.
    pub fn name_index_inserts(self) -> u64 {
        self.births.saturating_add(self.moves)
    }
}

/// The bytes a bounded plan may spend on its rewrite ([`RewriteCount`]), and
/// what one row, one list entry, one indexed section, one adoption-index id
/// and one section its depth check walks cost against them. Pure numbers:
/// the caller supplies the costs, so this module carries no charge constants.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MutationAllowance {
    /// The most [`RewriteCount::bytes`] the plan may reach before it asks its
    /// [`RewriteBudget`] for more.
    pub bytes: u64,
    /// Cost of one rewritten row.
    pub row_bytes: u64,
    /// Cost of one rewritten or listed children-list entry.
    pub list_entry_bytes: u64,
    /// Cost of one section in the plan's `(parent_id, name)` index.
    pub index_entry_bytes: u64,
    /// Cost of one id the plan's adoption index records
    /// ([`RewriteCount::adoption_entries`]).
    pub adoption_entry_bytes: u64,
    /// Cost of one section the plan's depth check walks
    /// ([`RewriteCount::walked_sections`]).
    pub walk_entry_bytes: u64,
}

impl MutationAllowance {
    /// An allowance of `u64::MAX` bytes at no price: a plan that is never
    /// bounded (a test oracle's, or a commit outside any upload).
    pub const UNBOUNDED: Self = Self {
        bytes: u64::MAX,
        row_bytes: 0,
        list_entry_bytes: 0,
        index_entry_bytes: 0,
        adoption_entry_bytes: 0,
        walk_entry_bytes: 0,
    };

    /// What `count` costs at this allowance's prices ([`RewriteCount::bytes`]).
    fn cost(&self, count: RewriteCount) -> u64 {
        count.bytes(
            self.row_bytes,
            self.list_entry_bytes,
            self.index_entry_bytes,
            self.adoption_entry_bytes,
            self.walk_entry_bytes,
        )
    }
}

/// Where a bounded plan ([`apply_delta_bounded`]) asks for more allowance as
/// its rewrite grows past what it holds: the plan's memory is charged as it
/// is planned rather than after a first plan overflowed (SDD §5.4).
///
/// Synchronous and with no I/O, like the rest of this module: the plan calls
/// it from inside the engine, while its caller holds whatever it planned
/// under, so an implementation must answer at once and never wait. The
/// caller supplies the implementation — `storage` grows a structural
/// `UpdateLayer`'s commit charge through it — so this module carries no
/// charge constants and no notion of what is charged.
///
/// Implemented for every `FnMut(RewriteCount) -> Option<u64>`, so a caller
/// with nothing to grow passes a closure that refuses every extension,
/// `&mut |_: RewriteCount| None::<u64>`, which bounds the plan at its first
/// allowance.
pub trait RewriteBudget {
    /// The plan's rewrite has reached `needed`, which costs more than its
    /// allowance: answer the new allowance in bytes — what the plan may now
    /// spend on its whole rewrite, at the allowance's row, list-entry,
    /// index-entry, adoption-entry and walk-entry costs — or `None` to
    /// refuse, and the plan overflows ([`BoundedPlan::Overflow`]). `needed`
    /// is the plan's kept rewrite so far plus every projection not yet
    /// retired ([`RewriteCount`]): asked at an emission, or at a build site
    /// before the structure it projects is allocated (a parent's listing, the
    /// name index, the adoption index's ids, a reset's, a cascade's or an
    /// adoption's rows, a hooked plan's depth-check walk — each hop of its
    /// walks up, each list of its walks down, or every section before a full
    /// walk). Asked again each time the running cost passes
    /// the allowance last answered, and never again once it refused. An
    /// answer that still does not cover `needed` is taken as a refusal.
    fn extend(&mut self, needed: RewriteCount) -> Option<u64>;
}

impl<F: FnMut(RewriteCount) -> Option<u64>> RewriteBudget for F {
    fn extend(&mut self, needed: RewriteCount) -> Option<u64> {
        self(needed)
    }
}

/// How [`apply_delta_bounded`] ended a plan that applied without error.
#[derive(Debug)]
pub enum BoundedPlan {
    /// The rewrite fit the allowance, as its budget extended it: every
    /// mutation, in emission order (as [`apply_delta_with_stats`] returns
    /// them), and the rewrite they make.
    Within {
        mutations: Vec<StorageMutation>,
        rewrite: RewriteCount,
    },
    /// The rewrite passed the allowance and the budget refused to extend it.
    /// The plan is aborted: every mutation it had collected was dropped when
    /// it passed, none after was kept, and `state` is left planned to the
    /// delta's end: the caller must discard it, or roll it back to its
    /// pre-plan cut — the structural cache's plan in place does, through its
    /// journal (`storage::structural_cache::CheckedOut::put_back_held`).
    /// `needed` is exact: each count the largest its kept rewrite plus its
    /// unretired projections reached over the whole plan, which ran to its
    /// end counting — what an allowance must cover for the same delta on the
    /// same state to plan [`BoundedPlan::Within`] with no extension. The plan
    /// keeps counting after the refusal, by design, so `needed` is exact; the
    /// scratch it builds from there to its end runs above what is held, by
    /// at most that need's cost less what was held, until it returns.
    Overflow { needed: RewriteCount },
}

/// What a bounded plan's build site asks for before it allocates a structure
/// whose size follows the object rather than the delta (SDD §5.4): the
/// counts it projects into the plan's [`RewriteCount`].
#[derive(Debug, Clone, Copy)]
enum Projection {
    /// `entries` more list entries for `parent_id`: its listing, projected at
    /// the parent's whole list before [`ChildPlan::build_cache`] builds it.
    List { parent_id: u64, entries: u64 },
    /// Rows a reset's drain, a cascade's subtree or a birth's adopted orphans
    /// are about to emit, projected before the scratch that finds them grows.
    /// Each emitted row retires one.
    Rows(u64),
    /// The sections the plan's `(parent_id, name)` index holds, projected
    /// before [`NameIndex::build`] allocates it at that capacity. Never
    /// retired: the index lives until the plan ends.
    IndexedSections(u64),
    /// Ids the plan's adoption index ([`AdoptionIndex`]) is about to record,
    /// projected before each push into it — by its build and by the name
    /// hook that keeps it. Never retired: the index lives until the plan
    /// ends.
    AdoptionEntries(u64),
    /// Sections a hooked plan's depth check is about to walk, projected
    /// before its walk grows by them. Never retired before the plan ends.
    WalkedSections(u64),
    /// Bytes a journalled plan's rollback could rebuild, projected by its
    /// hooks ([`RebuildProjector`]): one table's rebuild at its pre-plan
    /// length, before the write that can make it necessary, or one
    /// name-index vector the plan drops with its parent key, at its
    /// capacity. Never retired before the plan ends.
    RollbackRebuild(u64),
}

/// Where a journalled plan's hooks (`domain::plan_index::IndexedHooks`)
/// project what its rollback could rebuild — a table at its pre-plan length,
/// or a name-index vector the plan dropped with its key: the bounded plan's
/// meter, which asks its [`RewriteBudget`] at once
/// ([`RewriteCount::rollback_rebuild_bytes`]).
pub(crate) trait RebuildProjector {
    /// Add `bytes` to the plan's rollback rebuilds and, when its cost passes
    /// the allowance, ask the budget, exactly as any projection does.
    fn project_rebuild(&self, bytes: u64);
}

/// Where a plan's build sites ask before they allocate: the bounded plan's
/// [`BoundedMeter`], shared by its [`BoundedSink`] and its [`ChildPlan`]. A
/// plan into a [`MutationSink`] has none, and its build sites ask nothing.
trait PlanMeter {
    /// Add `projection` to the plan's count and, when its cost passes the
    /// allowance, ask the [`RewriteBudget`] for more, exactly as an emission
    /// does.
    fn project(&self, projection: Projection);
}

/// The accounting of [`apply_delta_bounded`]: the plan's [`RewriteCount`],
/// its allowance and its budget, and the mutations it collects. Its sink
/// ([`BoundedSink`]) reports every emission and its [`ChildPlan`] every
/// projection, so the budget is asked before the structure a projection
/// covers is allocated, not after it.
///
/// - **Kept**: the rows emitted beyond the delta's own elements, and per
///   emitted children list the larger of its length and its parent's
///   listing projection — what the mutation list holds.
/// - **Pending**: projections not yet retired — a projected parent's listing
///   until its list is emitted (a parent whose list is deleted instead keeps
///   it), projected rows until as many rows are emitted, and the name index's
///   sections, the adoption index's ids, the depth walk's sections and the
///   rollback rebuilds a journalled plan's hooks project
///   ([`RebuildProjector`]) for the rest of the plan.
///
/// Its cost is kept + pending at the allowance's prices; the budget is asked
/// when that passes the allowance. [`BoundedPlan::Within`] reports the kept
/// count (`indexed_sections`, `adoption_entries`, `walked_sections` and
/// `rollback_rebuild_bytes` 0);
/// [`BoundedPlan::Overflow`] the largest kept + pending the plan reached,
/// count by count.
///
/// Counting needs each mutation's kind and, for a children list, its length,
/// so past the allowance each mutation is still built — one at a time, and
/// dropped at once — rather than skipped as [`MutationSink::Discard`] skips
/// it. The transient is at most one row or one list, both already in the
/// planned state.
struct BoundedMeter<'b> {
    allowance: MutationAllowance,
    /// Where the allowance grows from ([`RewriteBudget`]); asked only while
    /// the plan has not overflowed.
    budget: &'b mut dyn RewriteBudget,
    /// The delta's own elements, which the rewrite's `rows` excludes.
    own_elements: u64,
    /// Mutations emitted so far, of every kind.
    emitted: u64,
    /// Kept list entries: per emitted `SetChildren`, the larger of its length
    /// and its parent's projection.
    list_kept: u64,
    /// Each projected parent's listing projection, until its list is emitted.
    list_projected: std::collections::HashMap<u64, u64>,
    /// The sum of `list_projected`.
    list_pending: u64,
    /// Projected rows not yet retired by an emission.
    rows_pending: u64,
    /// The name index's projected sections (0 until it is built).
    indexed_sections: u64,
    /// The adoption index's projected ids (0 until it records one).
    adoption_entries: u64,
    /// The depth walk's projected sections (0 until it walks).
    walked_sections: u64,
    /// The projected rollback rebuilds, in bytes (0 until one is projected).
    rollback_rebuild_bytes: u64,
    /// The largest kept + pending the plan has reached, count by count.
    peak: RewriteCount,
    /// The collected mutations; emptied for good once the budget refuses.
    mutations: Vec<StorageMutation>,
    overflowed: bool,
}

impl<'b> BoundedMeter<'b> {
    fn new(
        allowance: MutationAllowance,
        own_elements: u64,
        budget: &'b mut dyn RewriteBudget,
    ) -> Self {
        Self {
            allowance,
            budget,
            own_elements,
            emitted: 0,
            list_kept: 0,
            list_projected: std::collections::HashMap::new(),
            list_pending: 0,
            rows_pending: 0,
            indexed_sections: 0,
            adoption_entries: 0,
            walked_sections: 0,
            rollback_rebuild_bytes: 0,
            peak: RewriteCount::default(),
            mutations: Vec::new(),
            overflowed: false,
        }
    }

    /// The kept rewrite: what the mutation list holds (no index, adoption,
    /// walk or rollback-rebuild entries: they are plan scratch).
    fn kept(&self) -> RewriteCount {
        RewriteCount {
            rows: self.emitted.saturating_sub(self.own_elements),
            list_entries: self.list_kept,
            indexed_sections: 0,
            adoption_entries: 0,
            walked_sections: 0,
            rollback_rebuild_bytes: 0,
        }
    }

    /// Kept + pending: what the plan holds right now.
    fn running(&self) -> RewriteCount {
        let kept = self.kept();
        RewriteCount {
            rows: kept.rows.saturating_add(self.rows_pending),
            list_entries: kept.list_entries.saturating_add(self.list_pending),
            indexed_sections: self.indexed_sections,
            adoption_entries: self.adoption_entries,
            walked_sections: self.walked_sections,
            rollback_rebuild_bytes: self.rollback_rebuild_bytes,
        }
    }

    /// Carry the peak, and ask the budget when the running cost passes the
    /// allowance. A refusal overflows the plan for good: its mutations are
    /// dropped, and it counts on.
    fn settle(&mut self) {
        let running = self.running();
        self.peak = self.peak.max(running);
        if self.overflowed {
            return;
        }
        let cost = self.allowance.cost(running);
        if cost > self.allowance.bytes {
            match self.budget.extend(running) {
                Some(bytes) if cost <= bytes => self.allowance.bytes = bytes,
                _ => {
                    self.overflowed = true;
                    self.mutations = Vec::new();
                }
            }
        }
    }

    fn project(&mut self, projection: Projection) {
        match projection {
            Projection::List { parent_id, entries } => {
                let projected = self.list_projected.entry(parent_id).or_default();
                *projected = projected.saturating_add(entries);
                self.list_pending = self.list_pending.saturating_add(entries);
            }
            Projection::Rows(rows) => self.rows_pending = self.rows_pending.saturating_add(rows),
            Projection::IndexedSections(sections) => {
                self.indexed_sections = self.indexed_sections.max(sections);
            }
            Projection::AdoptionEntries(ids) => {
                self.adoption_entries = self.adoption_entries.saturating_add(ids);
            }
            Projection::WalkedSections(sections) => {
                self.walked_sections = self.walked_sections.saturating_add(sections);
            }
            Projection::RollbackRebuild(bytes) => {
                self.rollback_rebuild_bytes = self.rollback_rebuild_bytes.saturating_add(bytes);
            }
        }
        self.settle();
    }

    fn push(&mut self, mutation: StorageMutation) {
        self.emitted = self.emitted.saturating_add(1);
        self.rows_pending = self.rows_pending.saturating_sub(1);
        if let StorageMutation::SetChildren {
            section_id,
            children,
        } = &mutation
        {
            let len = u64::try_from(children.len()).unwrap_or(u64::MAX);
            let projected = self.list_projected.remove(section_id).unwrap_or(0);
            self.list_pending = self.list_pending.saturating_sub(projected);
            self.list_kept = self.list_kept.saturating_add(len.max(projected));
        }
        self.settle();
        if !self.overflowed {
            self.mutations.push(mutation);
        }
    }

    fn finish(self) -> BoundedPlan {
        if self.overflowed {
            BoundedPlan::Overflow {
                needed: self.peak.max(self.running()),
            }
        } else {
            BoundedPlan::Within {
                rewrite: self.kept(),
                mutations: self.mutations,
            }
        }
    }
}

impl PlanMeter for std::cell::RefCell<BoundedMeter<'_>> {
    fn project(&self, projection: Projection) {
        self.borrow_mut().project(projection);
    }
}

impl RebuildProjector for std::cell::RefCell<BoundedMeter<'_>> {
    fn project_rebuild(&self, bytes: u64) {
        self.borrow_mut()
            .project(Projection::RollbackRebuild(bytes));
    }
}

/// The sink of [`apply_delta_bounded`]: collects like
/// [`MutationSink::Collect`] through its [`BoundedMeter`] while the rewrite
/// stays within its allowance. When the rewrite's cost passes the allowance
/// the meter asks its [`RewriteBudget`] for more and, given it, collects on
/// under the new allowance; once the budget refuses, it drops what it
/// collected and keeps none, counting on.
struct BoundedSink<'m, 'b> {
    meter: &'m std::cell::RefCell<BoundedMeter<'b>>,
}

impl PlanSink for BoundedSink<'_, '_> {
    fn push_with(&mut self, make: impl FnOnce() -> StorageMutation) {
        // Built before the meter is borrowed: `make` reads the planned state
        // only, never the meter.
        let mutation = make();
        self.meter.borrow_mut().push(mutation);
    }
}

// ── Per-delta planning state ──────────────────────────────────────────────────

/// Per-parent children-list bookkeeping, built lazily on the first time a
/// delta touches that parent's list (b.ot3 part C).
struct ParentCache {
    /// Membership of `state.children[parent_id]`, so the "already present"
    /// and "not present" answers are O(1) instead of O(C).
    members: std::collections::HashSet<u64>,
    /// `section_order` of each entry of `state.children[parent_id]`, in the
    /// same positions. Removes the per-element `state.sections` hash lookup
    /// from the insertion-point search and the append-key search.
    orders: Vec<u32>,
    /// `orders` is non-decreasing. Starting from cross-map-consistent state
    /// with non-decreasing lists, every command keeps it so, whatever order
    /// concurrent clients' commands arrive in: an append takes a key one above
    /// the largest, computed after the section has left every list that held
    /// it ([`append_order`], `apply_move_section`), a move onto the section's
    /// current parent keeps its key and place, explicit-order inserts land at
    /// their sorted position, and [`apply_reorder_children`] re-keys the
    /// children it places so the list it leaves is sorted. A *loaded* list
    /// need not be — a reorder stored before reorders re-keyed their children
    /// kept the client's order under the old keys, and a stale or ghost
    /// listing (also stored state only) carries a key nothing re-places — so
    /// the flag picks between the binary-search / last-element arms and the
    /// linear arms that reproduce the pre-b.ot3 answer exactly. Computed over the list as
    /// stored, duplicates included — so `sorted` and `duplicated` are
    /// independent, and a reader whose incremental arm assumes each id
    /// appears once must test **both**.
    sorted: bool,
    /// `state.children[parent_id]` contains the same id more than once.
    /// Unreachable through this engine (insertion de-dups, and
    /// `ReorderChildren` keeps one occurrence of each id) but
    /// reachable from state stored before those fixes, which the snapshot
    /// load `LRANGE`s back verbatim. All three list-side maintainers state
    /// their behaviour on this shape, and each keeps pre-b.ot3 semantics on
    /// it; the two whose edit can touch an occurrence of the repeated id
    /// re-derive through [`ChildPlan::build_cache`] afterwards rather than
    /// patching positions:
    ///
    /// * [`insert_child_into_parent`] — the membership dedup-skip makes an
    ///   already-listed id a no-op regardless of how many times it is
    ///   listed, so it neither adds a duplicate nor needs this flag; an id
    ///   that is *not* listed is patched in positionally exactly as on any
    ///   other list, because that edit touches no occurrence of the repeated
    ///   id (see [`ParentEntry::cache`]);
    /// * [`remove_child_from_parent`] — falls back to `retain` on this arm
    ///   so it keeps deleting *every* occurrence, as it did before b.ot3;
    /// * [`rebuild_children_list`] — falls back to the whole-list
    ///   `sort_by_key`, because a remove-and-reinsert would re-place only
    ///   the first occurrence.
    duplicated: bool,
}

/// The `section_order` a [`ParentCache`] mirrors for a listed child.
///
/// An id listed under a parent with no `state.sections` row of its own is a
/// *ghost*, and encodes as `0` — the value the pre-b.ot3 insertion-point scan
/// produced for it, so keeping it makes the cached predicate and the scan
/// agree. Having one definition of the quantity is what lets the invalidation
/// decision be *computed* (does this write change the child's effective
/// order?) instead of hand-enumerated per call site: a site that writes a row
/// without changing this value provably falsifies no cache, and one that
/// changes it provably falsifies every cache listing the child.
fn effective_order(sections: &std::collections::HashMap<u64, Section>, section_id: u64) -> u32 {
    sections
        .get(&section_id)
        .map(|s| s.section_order)
        .unwrap_or(0)
}

impl ParentCache {
    /// Derive a parent's cache from the live maps.
    ///
    /// Private to [`ChildPlan::build_cache`], which is the only caller: every
    /// build must also register the listed children in
    /// [`ChildPlan::listed_in`], and routing all builds through one wrapper is
    /// what makes that registration hold by construction rather than by
    /// remembering to do it at each of the five build sites.
    fn build(
        children: &std::collections::HashMap<u64, Vec<u64>>,
        sections: &std::collections::HashMap<u64, Section>,
        parent_id: u64,
    ) -> Self {
        let list = children.get(&parent_id).map(Vec::as_slice).unwrap_or(&[]);
        let members: std::collections::HashSet<u64> = list.iter().copied().collect();
        let orders: Vec<u32> = list
            .iter()
            .map(|cid| effective_order(sections, *cid))
            .collect();
        let sorted = orders.windows(2).all(|w| w[0] <= w[1]);
        ParentCache {
            duplicated: members.len() != list.len(),
            members,
            orders,
            sorted,
        }
    }
}

/// The parents listing one child in [`ChildPlan::listed_in`].
///
/// A bare `Vec<u64>` would allocate once per distinct child on every cache
/// build — precisely the per-child allocation class this index was introduced
/// to remove — so the single-lister steady-state case is stored inline.
///
/// `Many` is reached by two routes, and both are rare but real. Stored state
/// can carry a stale or ghost listing (SDD-tolerated, read back verbatim by the
/// snapshot load) that puts the child under two parents whose caches are both
/// populated. Or a cascade can leave a pair behind — it drops the parent's
/// cache without de-registering its children, deliberately, per
/// [`ChildPlan::listed_in`] — and the child is later registered under a live
/// parent as well; that route needs no stale stored state at all and reaches
/// `Many` from ordinary commands. Either way the extra entries cost a wasted
/// rebuild and never a wrong answer, but they must be representable.
enum CacheListers {
    One(u64),
    Many(Vec<u64>),
}

impl CacheListers {
    /// Idempotent: a parent already present is not added twice, so the
    /// `Many` vector length is the number of *distinct* listing parents.
    ///
    /// Returns whether a pair was actually stored, which is what
    /// [`ChildPlan::listed_pairs`] must move on — an idempotent re-insert
    /// stores nothing and must not be counted.
    fn insert(&mut self, parent_id: u64) -> bool {
        match self {
            CacheListers::One(existing) => {
                if *existing == parent_id {
                    return false;
                }
                *self = CacheListers::Many(vec![*existing, parent_id]);
                true
            }
            CacheListers::Many(parents) => {
                if parents.contains(&parent_id) {
                    return false;
                }
                parents.push(parent_id);
                true
            }
        }
    }

    /// Returns `(removed, empty)`: whether `parent_id` was actually stored here
    /// — the pair count moves only then, and de-registration is called on
    /// paths that cannot know in advance — and whether no lister remains, which
    /// is the caller's signal to drop the map entry rather than keep an empty
    /// one.
    fn remove(&mut self, parent_id: u64) -> (bool, bool) {
        match self {
            CacheListers::One(existing) => {
                let hit = *existing == parent_id;
                (hit, hit)
            }
            CacheListers::Many(parents) => {
                let before = parents.len();
                parents.retain(|p| *p != parent_id);
                let removed = parents.len() != before;
                if let [only] = parents[..] {
                    *self = CacheListers::One(only);
                    return (removed, false);
                }
                (removed, parents.is_empty())
            }
        }
    }

    /// Borrowing the single-lister case as a one-element slice keeps the walk
    /// in [`ChildPlan::invalidate_order_listeners`] allocation-free on the
    /// case it takes almost always.
    fn as_slice(&self) -> &[u64] {
        match self {
            CacheListers::One(parent_id) => std::slice::from_ref(parent_id),
            CacheListers::Many(parents) => parents.as_slice(),
        }
    }
}

/// One parent's entry in the [`ChildPlan`].
#[derive(Default)]
struct ParentEntry {
    /// This delta changed (or, matching pre-b.ot3 emission, re-wrote) the
    /// parent's children list, so the finalization pass owes it one write.
    dirty: bool,
    /// `None` until a children-list operation needs it.
    ///
    /// # Cache invariant
    ///
    /// While `Some(cache)`, for every position `i` of
    /// `state.children[parent_id]`:
    ///
    /// ```text
    /// cache.members  == the set of ids in state.children[parent_id]
    /// cache.orders[i] == effective_order(state.sections, children[i])
    /// cache.sorted   == cache.orders is non-decreasing
    /// ```
    ///
    /// Both the binary-search arm and the linear-scan arm of
    /// [`insert_child_into_parent`], and [`rebuild_children_list`]'s choice
    /// of arm, read the cache instead of `state.sections`, so a cache that
    /// drifts from that equality silently returns a different insertion
    /// index than the pre-b.ot3 scan did.
    ///
    /// # Everything that maintains it
    ///
    /// The list side (`state.children[parent_id]`):
    ///
    /// * [`insert_child_into_parent`] / [`remove_child_from_parent`] /
    ///   [`rebuild_children_list`] patch `members` and `orders` in place
    ///   alongside their own edit to the list. On a list that holds the same
    ///   id more than once, positional patching is unsound only for an edit
    ///   that *touches an occurrence of the repeated id* — so the two that
    ///   rewrite such a list wholesale (`remove_child_from_parent`'s
    ///   `retain`, `rebuild_children_list`'s `sort_by_key`) re-derive the
    ///   whole cache through [`ChildPlan::build_cache`] instead.
    ///   `insert_child_into_parent` is not one of them: it returns outright
    ///   when the incoming id is already a member, and otherwise inserts an
    ///   id that occurs nowhere in the list, so it reaches its patch
    ///   normally on a duplicated list and that patch is correct there —
    ///   `orders` stays positionally parallel because the two insertions
    ///   share one index, `members` gains exactly the new id, `sorted` stays
    ///   accurate (a non-decreasing list stays non-decreasing because the
    ///   index comes from `partition_point`, and a list with an inversion
    ///   keeps that inversion), and `duplicated` is unchanged because no id
    ///   gained an occurrence it did not already have. See
    ///   [`ParentCache::duplicated`] for that column;
    /// * [`apply_reorder_children`] rebuilds the cache from the list it just
    ///   installed and the keys it just re-assigned, through one
    ///   [`ChildPlan::build_cache`], rather than patching it, and
    ///   de-registers the ids that list dropped (entries that were not the
    ///   parent's children);
    /// * [`ChildPlan::invalidate_cache`] resets it to `None` when
    ///   `state.children[parent_id]` is removed wholesale (cascade delete),
    ///   and [`ChildPlan::invalidate_all_caches`] when the whole map is
    ///   cleared (reset), so a later re-insert cannot read a stale cache.
    ///
    /// Every one of those that leaves this cache *populated* also updates
    /// [`ChildPlan::listed_in`], because a populated cache the index does not
    /// name is exactly the drift the order side below is protecting against.
    ///
    /// The **order** side (`state.sections[child].section_order` of a child
    /// listed here) is the leg that made the round-1 b.ot3 caches drift: a
    /// child listed under *this* parent can have its row rewritten through a
    /// **different** parent, and nothing in the list-side maintenance above
    /// notices. That leg is closed by [`ChildPlan::listed_in`], the inverse of
    /// `members`: a write that changes a child's [`effective_order`] calls
    /// [`ChildPlan::invalidate_order_listeners`], which drops the cache of
    /// every parent the index names for that child. So a cache that survives
    /// an order-side write is one that provably does not list the written
    /// child — or the one parent whose falsified entries the caller repairs
    /// before anything reads them, which is the `keep` argument on
    /// [`ChildPlan::invalidate_order_listeners`].
    cache: Option<ParentCache>,
    /// The list entries this parent's listing has projected so far
    /// ([`Projection::List`]): its list's length the first time
    /// [`ChildPlan::build_cache`] listed it. A rebuild after an invalidation
    /// projects only a list grown beyond it, so a parent is never projected
    /// twice for the same entries.
    projected: u64,
}

/// `(parent_id, name)` → ascending section ids sharing that pair (b.ot3
/// part D).
///
/// Buckets are keyed by `(parent_id, hash(name))` rather than by an owned
/// `String`, so building the index costs no per-section allocation — a
/// one-command delta against a million-section layer would otherwise pay a
/// million `String` clones to answer one lookup. Bucket entries are
/// *candidates*: [`find_child_by_name`] re-verifies `parent_id` and `name`
/// against `state.sections` before returning, so a hash collision is a
/// correctness non-event.
///
/// Multimap-valued because duplicate `(parent_id, name)` siblings are
/// producible from legal commands today: `apply_move_section`'s same-parent
/// rename performs no conflict check (SDD §9.3.1, PRD §2.3.4) and is
/// deliberately left that way — rejecting it would be a client-visible wire
/// contract change. What b.ot3 fixes is that the *resolution* of such a
/// duplicate is deterministic; see [`find_child_by_name`].
#[derive(Default)]
struct NameIndex {
    buckets: std::collections::HashMap<(u64, u64), Vec<u64>>,
}

/// Deterministic hash of a section name. Only ever used to bucket
/// candidates inside a single [`NameIndex`]; the value never reaches
/// storage, the wire, or a comparison between two indexes.
fn name_hash(name: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    name.hash(&mut hasher);
    hasher.finish()
}

impl NameIndex {
    /// The index of `sections`, at a capacity of `capacity` sections: the
    /// object's sections plus every entry the delta can add to the index
    /// ([`StructuralInserts::name_index_inserts`]), so the plan's later
    /// inserts never grow its table ([`find_child_by_name`] projects that size
    /// first).
    fn build(sections: &std::collections::HashMap<u64, Section>, capacity: u64) -> Self {
        #[cfg(test)]
        test_counters::NAME_INDEX_BUILDS.with(|builds| builds.set(builds.get() + 1));
        let capacity = usize::try_from(capacity)
            .unwrap_or(usize::MAX)
            .max(sections.len());
        let mut index = NameIndex {
            buckets: std::collections::HashMap::with_capacity(capacity),
        };
        for section in sections.values() {
            index.insert(section.parent_id, &section.name, section.section_id);
        }
        index
    }

    fn insert(&mut self, parent_id: u64, name: &str, section_id: u64) {
        let bucket = self
            .buckets
            .entry((parent_id, name_hash(name)))
            .or_default();
        if let Err(pos) = bucket.binary_search(&section_id) {
            bucket.insert(pos, section_id);
        }
    }

    fn remove(&mut self, parent_id: u64, name: &str, section_id: u64) {
        let key = (parent_id, name_hash(name));
        if let Some(bucket) = self.buckets.get_mut(&key) {
            if let Ok(pos) = bucket.binary_search(&section_id) {
                bucket.remove(pos);
            }
            if bucket.is_empty() {
                self.buckets.remove(&key);
            }
        }
    }
}

/// The rows that name an id this delta can give a row to: the lookup
/// [`adopt_orphans`] reads to find the rows that already named a section that
/// has just been born.
///
/// # Scope
///
/// A row is born only by a `CreateSection` or by a `DiffChangeSection` whose
/// id has no row, and both name the id they birth in the command itself. So
/// the ids that can be born in a delta are known from the delta alone:
/// [`Self::candidates`] holds every `section_id` its `CreateSection` and
/// `DiffChangeSection` commands name — a superset, since a skipped create or
/// a diff-change of a live row births nothing — less the root `0`, which
/// adopts nothing. Only rows naming a candidate are indexed, so the build is
/// one pass over the rows with a binary search of the candidates per row,
/// and no per-row insert unless the row names one. It is built lazily, by
/// the delta's first birth, and at most once per delta; a delta that births
/// nothing builds nothing. Each build is counted in
/// [`ApplyDeltaStats::adoption_index_builds`], whose `<= 1` is what pins
/// that bound.
///
/// # Superset
///
/// [`Self::rows`] must name every row that names a candidate when
/// [`adopt_orphans`] reads it. Entries are candidates, not facts: the
/// adoption re-checks each against `state.sections`, so an entry whose row
/// was since deleted or reparented costs a lookup, never a wrong listing.
/// What has to be argued is only that no row naming a candidate is missing.
///
/// The build reads `state.sections` by `parent_id`, listed or not, so every
/// row naming a candidate at that moment is in. That includes a legacy row a
/// live candidate's list lost: a cascade of the candidate collects its
/// subtree from the lists, so such a row survives the cascade with its row
/// and `parent_id` intact, and the build — before or after that cascade —
/// reads it from the row.
///
/// After the build, every path that sets or changes a row's `parent_id` is
/// one of these:
///
/// * `apply_create_section` writes the new row;
/// * `apply_diff_change_section` writes a new row, or rewrites a live row
///   (a reparent, or the same parent with a new name or key);
/// * `apply_move_section` rewrites the row in its normal move and in both
///   rename arms (the rename arms keep the parent).
///
/// Each calls [`ChildPlan::name_inserted`] with the row's new `parent_id`,
/// and that hook records the row here whenever the parent is a candidate.
/// That covers a row created or reparented under a missing candidate (the
/// rows `insert_child_into_parent`'s missing-parent return leaves unlisted),
/// an orphan created earlier in the same delta, and a row moved or
/// diff-changed under a live candidate. The other two paths that write rows
/// never assign `parent_id`, so they call no hook and need none:
/// [`adopt_orphans`] rewrites only `last_version`, and
/// [`apply_reorder_children`] only `section_order` and `last_version`. Any
/// new path that assigns `parent_id` must call [`ChildPlan::name_inserted`]
/// for the same reason. The remaining paths only
/// remove rows — a cascade (`DeleteSection`, a conflict, or a move edge that
/// deletes) and a reset — and a removed row is no adoption candidate. A reset
/// also runs before any command, so it always precedes the build, and it
/// drops the index anyway.
///
/// # Charge
///
/// The index records rows naming *every* candidate, live or not, so its size
/// follows the object, not the delta: a `DiffChangeSection` of a live section
/// in a delta with any birth records all of that section's children. Live
/// candidates cannot be skipped. The newborn itself is live by the time the
/// index is built — its row is written before [`adopt_orphans`] runs — and a
/// legacy row a live candidate's list lost survives that candidate's cascade
/// and must be adopted when the delta births the id again, as it would be
/// with one command per delta. So in a bounded plan each recorded id is
/// charged instead: [`Self::row_named`] projects it
/// ([`Projection::AdoptionEntries`], counted in
/// [`RewriteCount::adoption_entries`]) before every push, by the build and
/// by [`ChildPlan::name_inserted`] alike. That count is plan scratch, never
/// retired. It is separate from the `Rows` projection [`adopt_orphans`]
/// makes per orphan it collects, which covers its `orphans` list and the
/// adopted row's `SetSection`.
#[derive(Default)]
struct AdoptionIndex {
    /// The ids this delta can birth, sorted and deduplicated, root excluded.
    candidates: Vec<u64>,
    /// Candidate id → the ids of the rows recorded naming it, in no order and
    /// possibly repeated; [`adopt_orphans`] sorts, deduplicates and
    /// re-checks them.
    rows: std::collections::HashMap<u64, Vec<u64>>,
}

impl AdoptionIndex {
    /// The index of `sections` for the ids `commands` can birth, each
    /// recorded row projected first through `meter`, if any
    /// ([`Self::row_named`]).
    fn build(
        sections: &std::collections::HashMap<u64, Section>,
        commands: &[StructCommand],
        meter: Option<&dyn PlanMeter>,
    ) -> Self {
        let mut candidates: Vec<u64> = commands
            .iter()
            .filter_map(|cmd| match cmd {
                StructCommand::CreateSection(c) => Some(c.section_id),
                StructCommand::DiffChangeSection(c) => Some(c.section_id),
                StructCommand::DeleteSection(_)
                | StructCommand::MoveSection(_)
                | StructCommand::ReorderChildren(_)
                | StructCommand::DiffDeleteSection(_) => None,
            })
            .filter(|&id| id != 0)
            .collect();
        candidates.sort_unstable();
        candidates.dedup();
        let mut index = AdoptionIndex {
            candidates,
            rows: std::collections::HashMap::new(),
        };
        for section in sections.values() {
            index.row_named(section.parent_id, section.section_id, meter);
        }
        index
    }

    /// Record that `section_id` now names `parent_id`, if `parent_id` is a
    /// candidate — in a bounded plan, after projecting the id through
    /// `meter` ([`Projection::AdoptionEntries`]), so the budget is asked
    /// before the push that can grow the candidate's list.
    fn row_named(&mut self, parent_id: u64, section_id: u64, meter: Option<&dyn PlanMeter>) {
        if self.candidates.binary_search(&parent_id).is_ok() {
            if let Some(meter) = meter {
                meter.project(Projection::AdoptionEntries(1));
            }
            #[cfg(test)]
            test_counters::ADOPTION_INDEX_RECORDS.with(|records| records.set(records.get() + 1));
            self.rows.entry(parent_id).or_default().push(section_id);
        }
    }
}

/// Test-only counts of the plan's object-scaled builds on this thread, so a
/// test's [`RewriteBudget`] can read, at each ask, whether the structure the
/// ask covers was already built: [`ParentCache::build`] calls (through
/// [`ChildPlan::build_cache`]), [`NameIndex::build`] calls, and ids pushed
/// into an [`AdoptionIndex`]. Compiled into unit tests only.
#[cfg(test)]
pub(crate) mod test_counters {
    thread_local! {
        /// [`super::ParentCache::build`] calls on this thread.
        pub(crate) static PARENT_CACHE_BUILDS: std::cell::Cell<u64> =
            const { std::cell::Cell::new(0) };
        /// [`super::NameIndex::build`] calls on this thread.
        pub(crate) static NAME_INDEX_BUILDS: std::cell::Cell<u64> =
            const { std::cell::Cell::new(0) };
        /// Ids pushed into [`super::AdoptionIndex`]'s rows on this thread, by
        /// its build and by the name hook.
        pub(crate) static ADOPTION_INDEX_RECORDS: std::cell::Cell<u64> =
            const { std::cell::Cell::new(0) };
    }
}

/// A test-support failpoint inside the engine, for the differential tests of
/// a plan made in place: armed with a step count, the engine fails with
/// [`DeltaError::InconsistentState`] (operation
/// [`failpoint::INJECTED`](crate::domain::delta::failpoint::INJECTED)) at
/// that step of its next deltas on this thread. A step is each structural
/// command, the close of the structural phase (children finalization, then
/// the cascade purge), each set field, each time sample, and the depth check,
/// in that order. Compiled out of every other build, where a step is a no-op.
///
/// Its panic mode ([`failpoint::arm_rollback_panic`]) instead panics at the
/// rebuild step of the next key-by-key rollback of a plan made in place on
/// this thread (`domain::plan_journal::PlanJournal::rollback`), so a test
/// can drive a rollback that unwinds part way.
#[cfg(any(test, feature = "test-support"))]
pub mod failpoint {
    use std::cell::Cell;

    use super::DeltaError;

    /// The `operation` an injected failure names, and the start of an
    /// injected panic's message.
    pub const INJECTED: &str = "test failpoint";

    thread_local! {
        static COUNTDOWN: Cell<Option<u64>> = const { Cell::new(None) };
        static STEPS: Cell<u64> = const { Cell::new(0) };
        static ROLLBACK_PANIC: Cell<bool> = const { Cell::new(false) };
    }

    /// Fail at step `step` (0-based) from now on this thread, once.
    pub fn arm(step: u64) {
        COUNTDOWN.with(|c| c.set(Some(step)));
    }

    /// Panic at the rebuild step of the next key-by-key rollback on this
    /// thread, once: after the rollback has restored the plan indexes' name
    /// index and external-hash counts, and before it restores any of the
    /// state's tables or decides whether one needs its rebuild — a step
    /// every rollback but a reset's swap or a torn journal reaches, whether
    /// or not it then rebuilds a table. The entry is left partly restored,
    /// as a panic in a rebuild's allocation would leave it.
    pub fn arm_rollback_panic() {
        ROLLBACK_PANIC.with(|p| p.set(true));
    }

    /// Disarm the failpoint on this thread, its panic mode included.
    pub fn disarm() {
        COUNTDOWN.with(|c| c.set(None));
        ROLLBACK_PANIC.with(|p| p.set(false));
    }

    /// The steps the engine has taken on this thread since the last
    /// [`reset_steps`]: the number of failure points a delta offers.
    pub fn steps() -> u64 {
        STEPS.with(Cell::get)
    }

    /// Zero the step count on this thread.
    pub fn reset_steps() {
        STEPS.with(|s| s.set(0));
    }

    /// Run `f` with the failpoint disarmed, its panic mode included, and its
    /// step count untouched, then restore all three: for a test-support
    /// check that plans a second time beside the plan under test.
    pub fn paused<T>(f: impl FnOnce() -> T) -> T {
        let armed = COUNTDOWN.with(|c| c.replace(None));
        let panic_armed = ROLLBACK_PANIC.with(|p| p.replace(false));
        let steps = STEPS.with(Cell::get);
        let out = f();
        COUNTDOWN.with(|c| c.set(armed));
        ROLLBACK_PANIC.with(|p| p.set(panic_armed));
        STEPS.with(|s| s.set(steps));
        out
    }

    /// The rollback's rebuild step ([`arm_rollback_panic`]): panics once
    /// when armed.
    pub(crate) fn rollback_rebuild_step() {
        if ROLLBACK_PANIC.with(|p| p.replace(false)) {
            panic!("{INJECTED}: a panic at the rollback's rebuild step");
        }
    }

    pub(super) fn step() -> Result<(), DeltaError> {
        STEPS.with(|s| s.set(s.get().saturating_add(1)));
        COUNTDOWN.with(|c| match c.get() {
            Some(0) => {
                c.set(None);
                Err(DeltaError::InconsistentState {
                    section_id: 0,
                    operation: INJECTED,
                })
            }
            Some(n) => {
                c.set(Some(n - 1));
                Ok(())
            }
            None => Ok(()),
        })
    }
}

/// One failpoint step ([`failpoint`]); a no-op outside test-support builds.
#[inline(always)]
fn failpoint_step() -> Result<(), DeltaError> {
    #[cfg(any(test, feature = "test-support"))]
    return failpoint::step();
    #[cfg(not(any(test, feature = "test-support")))]
    Ok(())
}

/// The failpoint's rollback rebuild step ([`failpoint::arm_rollback_panic`]);
/// a no-op outside test-support builds.
#[inline(always)]
pub(crate) fn rollback_rebuild_failpoint() {
    #[cfg(any(test, feature = "test-support"))]
    failpoint::rollback_rebuild_step();
}

/// Everything a single `apply_delta_into` run needs to coalesce children
/// writes, answer name lookups, and defer the cascade purge (b.ot3 parts
/// B–E).
///
/// Pure per-call stack-local state: created at `apply_delta_into` entry,
/// dropped with the frame on every exit path (including each `?`). Nothing
/// here is persisted or configured, and nothing leaves the frame but the
/// aggregate counters the fields below fold into [`ApplyDeltaStats`] —
/// no structure held here is ever handed to a caller.
///
/// In a bounded plan ([`apply_delta_bounded`]) it holds the plan's
/// [`PlanMeter`], which its build sites ask before they allocate a structure
/// whose size follows the object: a parent's listing
/// ([`ChildPlan::build_cache`]), the name index ([`find_child_by_name`]), the
/// adoption index's ids ([`AdoptionIndex::row_named`]), a reset's drain
/// ([`apply_reset`]), a cascade's subtree ([`collect_subtree`]) and a birth's
/// orphans ([`adopt_orphans`]).
///
/// It also carries the plan's [`EngineHooks`], which every write reaches
/// through `domain::delta_write`: [`NoHooks`] for every caller but the
/// structural cache.
struct ChildPlan<'m, H> {
    /// The bounded plan's meter ([`PlanMeter`]); `None` for a plan into a
    /// [`MutationSink`], whose build sites ask nothing.
    meter: Option<&'m dyn PlanMeter>,
    /// What every write is reported to (`domain::delta_write`).
    hooks: H,
    /// The entries the delta can add to the name index once it is built
    /// ([`StructuralInserts::name_index_inserts`]: its births and moves). The
    /// index is built at the object's sections plus these, so it never grows.
    index_inserts: u64,
    /// Whether the plan has answered a `(parent_id, name)` probe by a scan
    /// of `state.sections` ([`find_child_by_name`]): the first probe scans,
    /// and the second builds the index.
    name_scanned: bool,
    /// Parents in first-touched order — the order the finalization pass
    /// emits in, so mutation order stays deterministic.
    order: Vec<u64>,
    parents: std::collections::HashMap<u64, ParentEntry>,
    /// Parents for which a `DeleteChildrenList` was already emitted inline
    /// (cascade delete, reset). Keeps the finalizer from resurrecting a
    /// deleted key *and* from emitting a second delete for it.
    deleted_lists: std::collections::HashSet<u64>,
    /// Sections cascade-deleted by this delta that had a live
    /// `state.sections` row. Read by [`purge_cascaded_rows`] at the close of
    /// the structural phase, which erases their fields and time samples; the
    /// set itself is left in place and dies with the plan. Its hash order
    /// never orders a purge run: the compact field purge's probe arm sorts
    /// the ids first, and the scan arms only test membership. Ghost ids are
    /// deliberately absent — their fields were not purged before b.ot3
    /// either.
    purge_ids: std::collections::HashSet<u64>,
    /// Child id → the parents whose **populated** [`ParentCache`] lists that
    /// child. The inverse of `ParentCache::members`, and the structure that
    /// answers "which caches mirror this child's `section_order`?" in O(1)
    /// instead of by a scan.
    ///
    /// # Index invariant
    ///
    /// For every parent `P` whose [`ParentEntry::cache`] is `Some(c)` and
    /// every `X` in `c.members`, `listed_in[X]` names `P`.
    ///
    /// The asymmetry is deliberate and load-bearing. **Adds are mandatory for
    /// soundness**: a populated cache that the index does not name survives an
    /// order-side write it mirrors, and silently returns a different insertion
    /// index than the uncached scan would. **Removes are mandatory only for
    /// the cost bound**: a stale pair names a parent whose cache no longer
    /// lists the child, and dropping an already-absent or already-correct
    /// cache is a no-op — it costs one wasted rebuild, never a wrong answer.
    /// That is why the cascade paths, which drop a parent's cache outright,
    /// are allowed to leave their pairs behind, while
    /// [`remove_child_from_parent`] and [`apply_reorder_children`]'s
    /// dropped-ids pass de-register eagerly: each leaves the parent's cache
    /// *populated* across the membership change it makes, so a lingering pair
    /// would buy a spurious rebuild of a live list.
    ///
    /// # Memory
    ///
    /// At most one entry per distinct `(child, parent)` pair whose cache this
    /// delta built or patched — the same order as the `members` sets already
    /// held, and exactly N entries for N siblings under one parent.
    /// [`CacheListers`] keeps the common single-lister case free of a
    /// per-child `Vec` allocation, which matters because allocating once per
    /// distinct child on every build is the cost class this index exists to
    /// remove.
    ///
    /// That bound is observed rather than only argued:
    /// [`ApplyDeltaStats::cache_index_peak_pairs`] carries the high-water mark
    /// of `listed_pairs` below.
    listed_in: std::collections::HashMap<u64, CacheListers>,
    /// How many `(child, parent)` pairs [`ChildPlan::listed_in`] stores right
    /// now — the sum of the listing-parent counts over its entries, so a child
    /// with a [`CacheListers::Many`] of two parents contributes two.
    ///
    /// Maintained incrementally at each site that stores or drops a pair
    /// because the peak is a property of the *trajectory*: the map is a
    /// `HashMap` of enums with no running size of its own, and re-summing it
    /// per add would cost O(entries) per insertion — the quadratic shape this
    /// index was introduced to remove. `debug_assert_cache_index_superset`
    /// re-derives the sum once per delta and fails on drift.
    listed_pairs: u64,
    /// Built on the second `(parent_id, name)` probe (the first scans
    /// `state.sections` instead), then maintained incrementally. `None` means
    /// "not built yet" — the maintenance hooks are no-ops in that state,
    /// because a later build reads the current `state.sections`.
    names: Option<NameIndex>,
    /// Built by the delta's first birth ([`adopt_orphans`]), then kept by
    /// [`ChildPlan::name_inserted`]. `None` means "not built yet", with the
    /// same no-op hooks as `names`. See [`AdoptionIndex`] for its scope, why
    /// it names every row naming a candidate, and how a bounded plan charges
    /// each id it records.
    adoption: Option<AdoptionIndex>,
    /// Reused visited-set for [`is_descendant_of`], which is called up to
    /// twice per `MoveSection`. Reuse removes the per-call allocation; the
    /// O(depth) walk itself (residual R5) stays.
    descendant_scratch: std::collections::HashSet<u64>,
}

impl Default for ChildPlan<'_, NoHooks> {
    /// An unmetered plan with no hooks, for a delta that can make no inserts.
    fn default() -> Self {
        Self::new(None, StructuralInserts::default(), NoHooks)
    }
}

impl<'m, H> ChildPlan<'m, H> {
    /// A plan whose build sites ask `meter`, if any, for a delta that can
    /// make `inserts`, reporting its writes to `hooks`.
    fn new(meter: Option<&'m dyn PlanMeter>, inserts: StructuralInserts, hooks: H) -> Self {
        Self {
            meter,
            hooks,
            index_inserts: inserts.name_index_inserts(),
            name_scanned: false,
            order: Vec::new(),
            parents: std::collections::HashMap::new(),
            deleted_lists: std::collections::HashSet::new(),
            purge_ids: std::collections::HashSet::new(),
            listed_in: std::collections::HashMap::new(),
            listed_pairs: 0,
            names: None,
            adoption: None,
            descendant_scratch: std::collections::HashSet::new(),
        }
    }

    /// Ask the plan's meter for `projection` before allocating what it
    /// covers; nothing without one.
    fn project(&self, projection: Projection) {
        if let Some(meter) = self.meter {
            meter.project(projection);
        }
    }

    /// Register `parent_id` (preserving first-touch order) and return its
    /// entry. Does not build the cache and does not mark the parent dirty.
    fn entry(&mut self, parent_id: u64) -> &mut ParentEntry {
        if !self.parents.contains_key(&parent_id) {
            self.order.push(parent_id);
            self.parents.insert(parent_id, ParentEntry::default());
        }
        self.parents
            .get_mut(&parent_id)
            .expect("entry inserted immediately above")
    }

    /// Register `parent_id` and mark it as owing a final children write.
    fn mark_dirty(&mut self, parent_id: u64) {
        self.entry(parent_id).dirty = true;
    }

    /// Register `parent_id`, make sure its [`ParentCache`] is populated and
    /// describes the current `state.children[parent_id]` *and* the current
    /// `section_order`s, and hand back the cache together with the entry's
    /// `dirty` flag.
    ///
    /// Get-or-build: a populated cache is always current, because the only
    /// thing that can falsify one without the list-side maintenance noticing
    /// is an order-side write, and every such write drops the caches that
    /// mirror it through [`ChildPlan::invalidate_order_listeners`] before it
    /// returns. A build re-reads both maps, so it re-derives `sorted` as well
    /// — a raised order can turn a non-decreasing list non-monotone, and the
    /// `partition_point` arms on an unpartitioned sequence return an index the
    /// uncached scan never would.
    ///
    /// Returning the cache itself rather than the `ParentEntry` is what keeps
    /// the callers free of `expect("the cache is populated")` — the whole
    /// point of this method is that the cache *is* populated on return, and a
    /// caller that has to restate that is a caller that can get it wrong. The
    /// `dirty` flag rides along because [`insert_child_into_parent`] needs it
    /// next to the cache, and splitting them would cost a second hash lookup;
    /// [`append_order`], which only reads the cache, ignores it. (Insertion
    /// of an already-present child must NOT set `dirty`, so the flag is
    /// deliberately handed over rather than set here.)
    fn entry_cached(
        &mut self,
        children: &std::collections::HashMap<u64, Vec<u64>>,
        sections: &std::collections::HashMap<u64, Section>,
        parent_id: u64,
        stats: &mut ApplyDeltaStats,
    ) -> (&mut ParentCache, &mut bool) {
        // Built before the entry is borrowed because `build_cache` needs
        // `&mut self` for the index registration.
        let built = match self.parents.get(&parent_id) {
            Some(entry) if entry.cache.is_some() => None,
            _ => Some(self.build_cache(children, sections, parent_id, stats)),
        };
        let ParentEntry { dirty, cache, .. } = self.entry(parent_id);
        if let Some(built) = built {
            *cache = Some(built);
        }
        match cache.as_mut() {
            Some(cache) => (cache, dirty),
            // The `match` above builds into the slot on exactly the branch
            // where it was empty, so an empty slot here is impossible.
            None => unreachable!("entry_cached built into the slot above when it was empty"),
        }
    }

    /// Detach `parent_id`'s cache (building it first if the slot is empty), so
    /// the caller can edit it side by side with the children list itself.
    ///
    /// `ParentCache::orders` is positionally parallel to
    /// `state.children[parent_id]`, so the two have to be edited together —
    /// but they live in different owners, and the `&state.children` that
    /// [`ChildPlan::entry_cached`] needs conflicts with the `&mut` on the
    /// list. Handing the cache over by value is what lets both edits sit in
    /// one block, instead of a second lookup that has to assert the cache is
    /// still there.
    ///
    /// Pair with [`ChildPlan::restore_cache`]. **Dropping the detached cache
    /// instead is always safe** — the slot stays empty and the next
    /// [`ChildPlan::entry_cached`] rebuilds it from the live maps — so an
    /// early return needs no unwind. A path that leaves the list in a shape
    /// the incremental edit does not describe re-derives through
    /// [`ChildPlan::build_cache`] and restores that instead — three such paths
    /// today: [`rebuild_children_list`]'s wholesale re-sort of an unsorted
    /// list, the same function's re-sort of a legacy duplicate-id list, and
    /// [`remove_child_from_parent`]'s legacy duplicate-id `retain`. That is
    /// not for correctness but so the *next* command under the same parent
    /// still finds a cache to take the sorted arm with.
    fn detach_cache(
        &mut self,
        children: &std::collections::HashMap<u64, Vec<u64>>,
        sections: &std::collections::HashMap<u64, Section>,
        parent_id: u64,
        stats: &mut ApplyDeltaStats,
    ) -> ParentCache {
        match self.entry(parent_id).cache.take() {
            Some(cache) => cache,
            None => self.build_cache(children, sections, parent_id, stats),
        }
    }

    /// Put a cache detached by [`ChildPlan::detach_cache`] back, or install
    /// one freshly derived by [`ChildPlan::build_cache`].
    fn restore_cache(&mut self, parent_id: u64, cache: ParentCache) {
        self.entry(parent_id).cache = Some(cache);
    }

    /// Derive `parent_id`'s cache and register every child it lists in
    /// [`ChildPlan::listed_in`]. **The only caller of [`ParentCache::build`]**,
    /// which is what makes the index invariant hold by construction: a cache
    /// cannot come into existence without its members being indexed, so no
    /// populated cache can escape an order-side write that falsifies it.
    ///
    /// Counted into [`ApplyDeltaStats::parent_cache_builds`] and
    /// [`ApplyDeltaStats::parent_cache_entries_built`]. The second is the one
    /// that reveals a design re-deriving a growing list on every command: a
    /// build count alone cannot tell re-deriving an empty list from
    /// re-deriving a million-entry one.
    ///
    /// In a bounded plan it first projects the listing ([`Projection::List`]):
    /// the parent's whole list, the first time the plan lists the parent, or
    /// only what the list has grown by since ([`ParentEntry::projected`]) —
    /// before [`ParentCache::build`] and the index registration allocate it.
    fn build_cache(
        &mut self,
        children: &std::collections::HashMap<u64, Vec<u64>>,
        sections: &std::collections::HashMap<u64, Section>,
        parent_id: u64,
        stats: &mut ApplyDeltaStats,
    ) -> ParentCache {
        let listed = children
            .get(&parent_id)
            .map_or(0, |list| u64::try_from(list.len()).unwrap_or(u64::MAX));
        let projected = self.entry(parent_id).projected;
        if listed > projected {
            self.project(Projection::List {
                parent_id,
                entries: listed - projected,
            });
            self.entry(parent_id).projected = listed;
        }
        #[cfg(test)]
        test_counters::PARENT_CACHE_BUILDS.with(|builds| builds.set(builds.get() + 1));
        let cache = ParentCache::build(children, sections, parent_id);
        stats.parent_cache_builds = stats.parent_cache_builds.saturating_add(1);
        stats.parent_cache_entries_built = stats
            .parent_cache_entries_built
            .saturating_add(cache.orders.len() as u64);
        // Read off `children` rather than `cache.members` so the `&mut self`
        // the registration needs does not collide with a borrow of the cache.
        for &child_id in children.get(&parent_id).map(Vec::as_slice).unwrap_or(&[]) {
            self.register_listing(child_id, parent_id, stats);
        }
        cache
    }

    /// Record that `parent_id`'s populated cache lists `child_id`.
    fn register_listing(&mut self, child_id: u64, parent_id: u64, stats: &mut ApplyDeltaStats) {
        let stored = match self.listed_in.entry(child_id) {
            std::collections::hash_map::Entry::Occupied(mut entry) => {
                entry.get_mut().insert(parent_id)
            }
            std::collections::hash_map::Entry::Vacant(slot) => {
                slot.insert(CacheListers::One(parent_id));
                true
            }
        };
        if stored {
            self.note_pair_stored(stats);
        }
    }

    /// One more pair is stored; carry
    /// [`ApplyDeltaStats::cache_index_peak_pairs`] to the running maximum.
    ///
    /// Folding the maximum here, at the only two sites that can raise the live
    /// count, is what makes the counter a peak rather than a final size — the
    /// index shrinks again, and by the end of the delta the largest it ever was
    /// is no longer readable from it.
    fn note_pair_stored(&mut self, stats: &mut ApplyDeltaStats) {
        self.listed_pairs = self.listed_pairs.saturating_add(1);
        stats.cache_index_peak_pairs = stats.cache_index_peak_pairs.max(self.listed_pairs);
    }

    /// Record that `parent_id`'s cache no longer lists `child_id`.
    ///
    /// Optional for soundness — the index is allowed to name a parent whose
    /// cache does not list the child, and dropping such a cache is a no-op.
    /// It is called anyway wherever the parent's cache stays **populated**
    /// across the membership change, because there a lingering pair would buy
    /// a spurious rebuild of a live list on the next order-side write.
    fn deregister_listing(&mut self, child_id: u64, parent_id: u64) {
        if let std::collections::hash_map::Entry::Occupied(mut entry) =
            self.listed_in.entry(child_id)
        {
            let (removed, empty) = entry.get_mut().remove(parent_id);
            if empty {
                entry.remove();
            }
            if removed {
                self.listed_pairs = self.listed_pairs.saturating_sub(1);
            }
        }
    }

    /// An order-side write on `child_id` just landed: drop the cache of every
    /// parent whose populated cache mirrors that child's order, except `keep`.
    ///
    /// An order-side write is any change to [`effective_order`] of a listed
    /// child — including a row appearing (ghost `0` → live order) and a row
    /// disappearing (live order → ghost `0`). `ParentCache::orders` mirrors
    /// that quantity per listed child, and a child's row can be rewritten
    /// through a parent that is **not** the one listing it — a stale listing
    /// in stored state, which this engine does not produce but must tolerate
    /// (see [`insert_child_into_parent`]) — so no amount of list-side
    /// maintenance catches it. The index is what turns
    /// "which caches mirror this child?" from a scan into a lookup.
    ///
    /// # Why not a walk over the touched parents
    ///
    /// An earlier b.ot3 shape tested *every* parent the delta had touched on
    /// every order-side write: Θ(N²) for a delta spread over N parents, and
    /// `MAX_DELTA_ENTRIES` admits 99,999 of them in one synchronous block.
    /// This walk is over the parents the index still names for the child — one
    /// in steady state, more only after a cascade left pairs behind or on a
    /// stale/ghost stored listing, never the whole touched set — and
    /// [`ApplyDeltaStats::cache_dependency_steps`] counts each step, so a
    /// reintroduction of the all-parents walk shows up as a counter that grows
    /// with the number of touched parents instead of silently costing time.
    ///
    /// # The `keep` rule
    ///
    /// `keep` names the child's **previous** parent — the parent its row named
    /// *before* the command that is writing it — and that cache survives the
    /// write with one falsified entry still in it. It is sound because the
    /// caller repairs exactly that parent's entry for exactly this child —
    /// in the first two cases below, in the very next statement — and nothing
    /// reads an order-dependent field of that cache in between:
    ///
    /// * same-parent order change → [`rebuild_children_list`] removes the
    ///   child's stale `orders` entry and reinserts it at the new order;
    /// * cross-parent move → [`remove_child_from_parent`] deletes that entry
    ///   outright. It reads only `members`, `duplicated` and list positions to
    ///   get there. The destination parent is never `keep`, so the following
    ///   [`insert_child_into_parent`] — which reads `sorted` / `orders` past
    ///   its membership early-return — cannot read a stale cache;
    /// * reorder re-key → [`apply_reorder_children`] keeps the reordered
    ///   parent across each child it re-keys and, once the last one is
    ///   written, replaces that parent's cache wholesale through
    ///   [`ChildPlan::build_cache`]. The re-keys in between read no cache.
    ///
    /// The first two repairs self-dispose on a legacy duplicate-id list, where
    /// the write falsifies *both* of the child's entries and a positional
    /// repair would be wrong: those arms route to `retain` / `sort_by_key`,
    /// which re-derive the whole cache through [`ChildPlan::build_cache`] and
    /// discard the kept one entirely. The third re-derives on every list.
    ///
    /// Keeping it is worth the argument because a rebuilt cache reads as
    /// *unsorted* precisely when the order change broke monotonicity, which
    /// would force [`rebuild_children_list`] onto its whole-list `sort_by_key`
    /// arm instead of the remove-and-reinsert one (residual R3).
    fn invalidate_order_listeners(
        &mut self,
        child_id: u64,
        keep: Option<u64>,
        stats: &mut ApplyDeltaStats,
    ) {
        // Taken by value so the walk can hold the listing parents while
        // mutating `self.parents`.
        let Some(listers) = self.listed_in.remove(&child_id) else {
            return;
        };
        self.listed_pairs = self
            .listed_pairs
            .saturating_sub(listers.as_slice().len() as u64);
        let mut keep_was_listed = false;
        for &parent_id in listers.as_slice() {
            stats.cache_dependency_steps = stats.cache_dependency_steps.saturating_add(1);
            if Some(parent_id) == keep {
                keep_was_listed = true;
                continue;
            }
            if let Some(entry) = self.parents.get_mut(&parent_id) {
                entry.cache = None;
            }
        }
        // The kept cache survives still listing the child, so its pair must
        // survive with it — dropping the pair would let a *later* order-side
        // write on the same child miss that cache.
        if keep_was_listed {
            if let Some(parent_id) = keep {
                self.listed_in
                    .insert(child_id, CacheListers::One(parent_id));
                self.note_pair_stored(stats);
            }
        }
    }

    /// Drop a parent's cache — `state.children[parent_id]` was removed
    /// wholesale, so anything cached about it is stale.
    ///
    /// The parent's pairs in [`ChildPlan::listed_in`] are deliberately left
    /// behind: the slot is empty, so a later walk that names this parent drops
    /// an already-absent cache, and a re-materialized list is rebuilt through
    /// [`ChildPlan::build_cache`], which re-registers whatever it then holds.
    fn invalidate_cache(&mut self, parent_id: u64) {
        if let Some(entry) = self.parents.get_mut(&parent_id) {
            entry.cache = None;
        }
    }

    /// Drop every cache — `state.children` was cleared wholesale (reset).
    fn invalidate_all_caches(&mut self) {
        for entry in self.parents.values_mut() {
            entry.cache = None;
        }
    }

    /// Record that `section_id` now sits at `(parent_id, name)`. No-op
    /// until the name index has been built.
    ///
    /// Every write that *assigns* a row's `parent_id` calls this — a new row,
    /// a reparent, and the two `MoveSection` rename arms, which assign the
    /// same parent back — which also makes it the hook that keeps the
    /// [`AdoptionIndex`] a superset: the row is recorded there when
    /// `parent_id` is a candidate. Writes that leave `parent_id` unassigned
    /// ([`adopt_orphans`]'s `last_version` rewrite and
    /// [`apply_reorder_children`]'s re-key) do not call it, and do not need
    /// to. In a bounded plan the adoption index's record is projected first
    /// ([`Projection::AdoptionEntries`]), as its build's are.
    fn name_inserted(&mut self, parent_id: u64, name: &str, section_id: u64) {
        if let Some(index) = self.names.as_mut() {
            index.insert(parent_id, name, section_id);
        }
        let meter = self.meter;
        if let Some(adoption) = self.adoption.as_mut() {
            adoption.row_named(parent_id, section_id, meter);
        }
    }

    /// Record that `section_id` no longer sits at `(parent_id, name)`.
    fn name_removed(&mut self, parent_id: u64, name: &str, section_id: u64) {
        if let Some(index) = self.names.as_mut() {
            index.remove(parent_id, name, section_id);
        }
    }
}

/// Apply a parsed OODF delta to `state`, mutate it in place, and return the
/// corresponding [`StorageMutation`]s to commit to Valkey.
///
/// Thin wrapper around [`apply_delta_with_stats`] for callers that don't
/// care about per-delta observability stats (e.g. tests).
///
/// Every entry point is generic over the state's row form
/// ([`ObjectStateOf`]): the full `ObjectState` every reader and
/// `CreateLayer` builds, or the compact `CompactState` the structural cache
/// holds and every structural plan runs on. Only the value-row sites differ
/// (the reset's drain, the cascade purge and the two inserts), and they go
/// through [`FieldRows`] / [`SampleRows`], so both forms plan the same
/// mutations: the plan's `SetField` / `SetTimeSample` carry the delta's own
/// values, never a held row's. A compact store keys a time sample by its
/// stored bits, so it can hold a section's time 0 at `0.0`, at `-0.0` or at
/// both, where the full form's `OrderedFloat` key holds one row; rule E keeps
/// the plans equal anyway: a reset's drain or a cascade's purge that deletes
/// a section's time-0 row deletes its time 0 at both stored spellings, once
/// each, the one the state did not hold added at the end of the same run
/// ([`ZeroTimeDeletes`]). "The same" holds only up to order within one drain
/// or purge run: the `DeleteField` / `DeleteTimeSample`s a reset's drain or a
/// cascade's purge emits follow the row store's hash order — except a compact
/// field purge on its probe arm, which emits by ascending section id, then
/// ascending interned name id (`domain::types::PURGE_PROBE_RATIO`), and rule
/// E's added deletes, which close the run in ascending section id — and the
/// hash order is not fixed across states (two states of one form can emit
/// one run in different orders); a state's clone, which keeps its table
/// layouts and interned ids, emits the same run in the same order. The runs
/// themselves, and every mutation outside them, sit at the same positions.
///
/// `delta = None` preserves SDD §8.4 semantics: empty delta = no content
/// changes, but sequence_num still increments and the header mutation is
/// emitted. Callers that receive raw bytes should go through
/// `oodf_ffi::parse_delta` first and pass `Some(delta)` on success.
///
/// `delta.base_version` is intentionally *not* consulted here — optimistic
/// concurrency control lives entirely in the storage layer (see
/// `storage::layer::update_layer_watch_txn` / the Lua fast path). Validating
/// `base_version` inside this pure-logic engine would couple domain logic to
/// OCC semantics and make it impossible to reuse `apply_delta` for replay
/// during diff reconstruction, which deliberately applies deltas without
/// OCC checks.
///
/// # Partial-mutation contract (b.dmu item 5)
///
/// **On `Err(_)`, `state` may have been partially mutated** — `apply_delta`
/// applies structural commands sequentially against the in-place `state`,
/// and a helper that errors mid-delta leaves all prior commands' effects
/// committed to the maps (`state.sections`, `state.children`,
/// `state.fields`, `state.time_samples`). The caller MUST discard the
/// partially-mutated state and reload a fresh snapshot before retrying.
/// The storage layer (`LayerStore::update_layer` /
/// `LayerStore::create_layer`) already does this — `state` is rebuilt from
/// Valkey on every attempt — but new callers must honour the same
/// discard-on-error rule.
///
/// `state.sequence_num` is the **only** field guaranteed unchanged on
/// `Err(_)` (b.dmu item 6 — pre-validation pass). The bump is deferred
/// until every helper has succeeded so a delta where command N is invalid
/// but commands 1..N-1 are valid does not advance `sequence_num`. This is
/// the only field with that guarantee; everything else in `state` may
/// reflect partial application.
///
/// Per SDD §9.1.
pub fn apply_delta<R: ValueRows>(
    state: &mut ObjectStateOf<R>,
    delta: Option<OwnedDelta>,
) -> Result<Vec<StorageMutation>, DeltaError> {
    let mut stats = ApplyDeltaStats::default();
    apply_delta_with_stats(state, delta, &mut stats)
}

/// Like [`apply_delta`], but reports per-delta observability stats through
/// an out-parameter (b.39x item 7). The hot-path callers (`LayerStore::
/// update_layer`, `handle_create_layer`) use this variant so they can bump
/// `delta_create_section_skipped_missing_parent_total` from the metrics
/// registry — domain code has no access to `Metrics` and cannot bump
/// counters directly without violating the "zero I/O" rule.
///
/// See [`apply_delta`] for the partial-mutation contract on `Err(_)`.
pub fn apply_delta_with_stats<R: ValueRows>(
    state: &mut ObjectStateOf<R>,
    delta: Option<OwnedDelta>,
    stats: &mut ApplyDeltaStats,
) -> Result<Vec<StorageMutation>, DeltaError> {
    let mut sink = MutationSink::collecting();
    apply_delta_into(state, delta, &mut sink, stats)?;
    let MutationSink::Collect(mutations) = sink else {
        unreachable!("sink was constructed as MutationSink::Collect immediately above");
    };
    Ok(mutations)
}

/// [`apply_delta_with_stats`] with the plan's rewrite bounded by
/// `allowance`, grown through `budget`: the mutations beyond the delta's own
/// elements ([`RewriteCount`]) may cost at most `allowance.bytes`, and each
/// time their running cost passes it the plan asks `budget` for more
/// ([`RewriteBudget::extend`]) and plans on under what it answers. Its build
/// sites ask first, through the same count: before the plan lists a parent
/// it projects the parent's whole list, before it builds its
/// `(parent_id, name)` index (at its second name probe; the first scans) the
/// object's sections plus the entries the delta can add to it
/// ([`StructuralInserts::name_index_inserts`]), before its adoption index
/// records an id, that id ([`RewriteCount::adoption_entries`]), before a
/// reset or a cascade grows the scratch that finds its rows, three rows per
/// section it can delete, and before a birth collects the orphans it adopts,
/// one row per orphan ([`BoundedMeter`]). A plan with persistent indexes
/// ([`apply_delta_bounded_indexed`]) builds neither index, and instead
/// projects its end-of-delta depth check's walk
/// ([`RewriteCount::walked_sections`]): each hop of its walks up and each
/// list of its walks down before the walk records it, or every section
/// before a full walk. So the budget covers each of these structures before
/// it is allocated, not once its rows are emitted.
///
/// Within the allowance as extended, the result is exactly
/// [`apply_delta_with_stats`]'s — the same `state`, `stats` and mutations —
/// together with the kept rewrite they make (a projected parent's list
/// counted at the larger of its projection and its emitted length; no index,
/// adoption or walk entries). Once `budget` refuses, the plan is aborted and
/// [`BoundedPlan::Overflow`] reports the exact need: the largest kept
/// rewrite plus unretired projections the plan reached. The plan still runs
/// to the end so that count is exact, but it collects nothing more, so its
/// mutation list never holds more than the allowance covers; its scratch
/// from the refusal to its end is not covered, by design. A
/// caller with nothing to grow passes a budget that refuses
/// ([`RewriteBudget`]'s closure form), or an unbounded allowance, which never
/// asks.
///
/// On [`BoundedPlan::Overflow`] the caller must discard `state`, as on
/// `Err(_)` (see [`apply_delta`] for the partial-mutation contract, which
/// applies here unchanged): plan on a copy you can drop. The structural
/// cache plans in place instead, through [`apply_delta_bounded_indexed`],
/// and rolls an unfinished plan back through its journal.
pub fn apply_delta_bounded<R: ValueRows>(
    state: &mut ObjectStateOf<R>,
    delta: Option<OwnedDelta>,
    allowance: MutationAllowance,
    budget: &mut dyn RewriteBudget,
    stats: &mut ApplyDeltaStats,
) -> Result<BoundedPlan, DeltaError> {
    apply_delta_bounded_with(state, delta, allowance, budget, stats, NoHooks)
}

/// [`apply_delta_bounded`] on the structural cache's held entry itself, in
/// place: every write keeps `indexes` (the entry's persistent plan indexes,
/// `domain::plan_index`) in step, is recorded in `touches`, and — with a
/// `journal` — has its key's pre-plan value recorded on first touch
/// (`domain::plan_journal`). Lookups, adoptions and the depth check read
/// `indexes`, so no per-delta index is built from the whole state.
///
/// The same result as [`apply_delta_bounded`] on a copy of `state`. On
/// `Err(_)` or [`BoundedPlan::Overflow`] `state` and `indexes` are partly
/// planned, as [`apply_delta`]'s partial-mutation contract says: the caller
/// rolls them back through `journal` ([`PlanJournal::rollback`]) — or, with
/// no journal, drops them.
///
/// [`PlanJournal::rollback`]: crate::domain::plan_journal::PlanJournal::rollback
#[allow(clippy::too_many_arguments)]
pub fn apply_delta_bounded_indexed(
    state: &mut crate::domain::types::CompactState,
    delta: Option<OwnedDelta>,
    allowance: MutationAllowance,
    budget: &mut dyn RewriteBudget,
    stats: &mut ApplyDeltaStats,
    indexes: &mut crate::domain::plan_index::PlanIndexes,
    touches: &mut crate::domain::plan_index::PlanTouches,
    journal: Option<&mut crate::domain::plan_journal::PlanJournal>,
) -> Result<BoundedPlan, DeltaError> {
    let own_elements = delta.as_ref().map_or(0, own_element_count);
    let meter = std::cell::RefCell::new(BoundedMeter::new(allowance, own_elements, budget));
    {
        // The hooks project a journalled plan's rollback rebuilds into the
        // same meter as its build sites, so they borrow it until the plan
        // ends.
        let hooks = crate::domain::plan_index::IndexedHooks::new(indexes, touches);
        let hooks = match journal {
            Some(journal) => hooks.journalled(journal, &meter),
            None => hooks,
        };
        run_bounded(state, delta, &meter, stats, hooks)?;
    }
    Ok(meter.into_inner().finish())
}

/// [`apply_delta_bounded`] reporting every write to `hooks`.
pub(crate) fn apply_delta_bounded_with<R: ValueRows, H: EngineHooks<R>>(
    state: &mut ObjectStateOf<R>,
    delta: Option<OwnedDelta>,
    allowance: MutationAllowance,
    budget: &mut dyn RewriteBudget,
    stats: &mut ApplyDeltaStats,
    hooks: H,
) -> Result<BoundedPlan, DeltaError> {
    let own_elements = delta.as_ref().map_or(0, own_element_count);
    let meter = std::cell::RefCell::new(BoundedMeter::new(allowance, own_elements, budget));
    run_bounded(state, delta, &meter, stats, hooks)?;
    Ok(meter.into_inner().finish())
}

/// A bounded plan's run on `meter`, which its sink and its build sites share.
fn run_bounded<R: ValueRows, H: EngineHooks<R>>(
    state: &mut ObjectStateOf<R>,
    delta: Option<OwnedDelta>,
    meter: &std::cell::RefCell<BoundedMeter<'_>>,
    stats: &mut ApplyDeltaStats,
    hooks: H,
) -> Result<(), DeltaError> {
    let mut sink = BoundedSink { meter };
    apply_delta_engine(
        state,
        delta,
        &mut sink,
        Some(meter),
        stats,
        ApplyMode::Commit,
        hooks,
    )
}

/// The delta-application engine itself: one implementation of OODF delta
/// semantics, parameterized over what happens to the persistence mutations
/// it plans (b.ot3 part A).
///
/// [`apply_delta`] and [`apply_delta_with_stats`] are thin wrappers that
/// pass a collecting sink. Callers that discard the mutation list — see the
/// ones named in [`MutationSink`]'s doc-comment — pass
/// [`MutationSink::Discard`] and pay for none of it: `state` ends in exactly
/// the same shape, `stats` carries exactly the same counters, and no
/// `Section`, children `Vec`, or inline field payload is ever cloned into a
/// list nobody reads.
///
/// See [`apply_delta`] for the partial-mutation contract on `Err(_)`, which
/// applies here unchanged.
///
/// Runs in [`ApplyMode::Commit`]; see [`apply_delta_into_mode`] for the
/// replay variant.
pub fn apply_delta_into<R: ValueRows>(
    state: &mut ObjectStateOf<R>,
    delta: Option<OwnedDelta>,
    sink: &mut MutationSink,
    stats: &mut ApplyDeltaStats,
) -> Result<(), DeltaError> {
    apply_delta_into_mode(state, delta, sink, stats, ApplyMode::Commit)
}

/// Whether [`apply_delta_into_mode`] is planning a new commit, re-applying
/// one that already committed, or rebuilding a layer from a checkpoint base.
///
/// Each non-`Commit` mode changes exactly one thing relative to `Commit`:
/// `Replay` whether the post-mutation `check_section_tree_depth` walk runs
/// for a delta that cannot change the section tree, `RestoreBase` whether the
/// per-delta entry-count cap is checked. Every other step — the per-item
/// rules of bounds validation, sequence ceiling, command semantics, the sink,
/// the stats — is identical in all modes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ApplyMode {
    /// Planning a commit: the depth walk always runs. Every caller except
    /// stream replay and the checkpoint base decode uses this mode, through
    /// [`apply_delta`], [`apply_delta_with_stats`] or [`apply_delta_into`].
    Commit,
    /// Rebuilding a whole layer from a checkpoint base
    /// (`checkpoint::apply::decode_checkpoint_ompf`, its only caller). Skips
    /// the [`MAX_DELTA_ENTRIES`](crate::domain::limits::MAX_DELTA_ENTRIES)
    /// check of [`validate_delta_bounds`](crate::domain::validation::validate_delta_bounds)
    /// and nothing else: every per-item rule
    /// ([`validate_delta_content`](crate::domain::validation::validate_delta_content))
    /// and the depth walk still run, exactly as in [`ApplyMode::Commit`].
    ///
    /// Why only here: the entry cap bounds one delta, but a base describes a
    /// whole layer, and layers have no size cap. A base over the cap is
    /// written as several chunks (`checkpoint::chunked_base`) and decoded
    /// back into the one reset delta an unchunked base would have carried,
    /// which can legitimately exceed the cap. Rebuilding N entries costs O(N)
    /// however the base is framed, so skipping the cap here adds no work the
    /// layer's size did not already imply.
    RestoreBase,
    /// Re-applying a stream entry that already committed
    /// (`domain::replay::replay_stream_record`). The depth walk is skipped
    /// for an empty commit (`None`) and for a delta with no reset whose
    /// structural commands, if any, are all `DiffDeleteSection` (ignored on
    /// ingest, SDD §9.1.4f) — the class the Lua fast path commits
    /// (`storage::lua::is_fast_path`), without walking the tree. Such a delta
    /// cannot change `state.children`. A delta with a reset or any other
    /// structural command still walks, exactly as in [`ApplyMode::Commit`].
    Replay,
}

/// Whether `delta` can change the section tree: it carries a reset
/// (`is_diff` with `base_version == 0`) or a structural command other than
/// `DiffDeleteSection`, which step 3 ignores. An empty commit (`None`) cannot.
///
/// The complement is the class the Lua fast path commits
/// (`storage::lua::is_fast_path`): such a delta cannot change
/// `state.sections` or `state.children`. [`apply_delta_into_mode`] skips the
/// replay-mode depth walk for it, and in debug builds the pre-delta orphan
/// collection.
fn delta_can_change_tree(delta: Option<&OwnedDelta>) -> bool {
    delta.is_some_and(|d| {
        (d.is_diff && d.base_version == 0)
            || d.struct_commands
                .iter()
                .any(|c| !matches!(c, StructCommand::DiffDeleteSection(_)))
    })
}

/// [`apply_delta_into`] with an explicit [`ApplyMode`].
///
/// See [`apply_delta`] for the partial-mutation contract on `Err(_)`, which
/// applies here unchanged in every mode.
pub fn apply_delta_into_mode<R: ValueRows>(
    state: &mut ObjectStateOf<R>,
    delta: Option<OwnedDelta>,
    sink: &mut MutationSink,
    stats: &mut ApplyDeltaStats,
    mode: ApplyMode,
) -> Result<(), DeltaError> {
    apply_delta_into_mode_with(state, delta, sink, stats, mode, NoHooks)
}

/// [`apply_delta_into_mode`] reporting every write to `hooks`.
pub(crate) fn apply_delta_into_mode_with<R: ValueRows, H: EngineHooks<R>>(
    state: &mut ObjectStateOf<R>,
    delta: Option<OwnedDelta>,
    sink: &mut MutationSink,
    stats: &mut ApplyDeltaStats,
    mode: ApplyMode,
    hooks: H,
) -> Result<(), DeltaError> {
    apply_delta_engine(state, delta, sink, None, stats, mode, hooks)
}

/// The engine behind [`apply_delta_into_mode`] and [`apply_delta_bounded`],
/// over either sink. `meter` is the bounded plan's ([`PlanMeter`]), which its
/// build sites ask before they allocate; `None` for a [`MutationSink`].
/// Every write it makes goes through `domain::delta_write`, which reports it
/// to `hooks`.
fn apply_delta_engine<R: ValueRows, H: EngineHooks<R>>(
    state: &mut ObjectStateOf<R>,
    delta: Option<OwnedDelta>,
    sink: &mut impl PlanSink,
    meter: Option<&dyn PlanMeter>,
    stats: &mut ApplyDeltaStats,
    mode: ApplyMode,
    hooks: H,
) -> Result<(), DeltaError> {
    // Decided before `delta` is moved into the command loop below. In
    // `Commit` and `RestoreBase` mode this is always `false` (and `&&` never
    // evaluates the predicate), so those callers walk the tree on every
    // delta. The skip class is a delta that cannot change the tree
    // (`delta_can_change_tree`).
    let skip_depth_walk = mode == ApplyMode::Replay && !delta_can_change_tree(delta.as_ref());

    // Reject deltas whose primitive counts / name byte lengths / children-list
    // sizes exceed the caps in `domain::limits` before any mutation. The
    // OODF FlatBuffers verifier rejects malformed wire bytes and the 100 MB
    // OMPF cap bounds total size, but nothing else constrains per-primitive
    // shape — see [`crate::domain::validation::validate_delta_bounds`].
    // `RestoreBase` skips only the per-delta entry-count cap (see
    // `ApplyMode::RestoreBase`).
    if let Some(d) = delta.as_ref() {
        if mode == ApplyMode::RestoreBase {
            crate::domain::validation::validate_delta_content(d)?;
        } else {
            crate::domain::validation::validate_delta_bounds(d)?;
        }
    }

    // Step 1 — pre-compute the new version without bumping `state.sequence_num`.
    // b.k1r H-stor-9: the version is capped at the 2^53 - 1 Lua f64
    // ceiling so the slow path and the fast path fail at the same point.
    // The Lua fast-path script has no choice — Lua 5.1 `number` is f64
    // and `current_seq + 1` silently stops advancing past 2^53. Failing
    // the slow path at the same ceiling keeps the two backends
    // byte-identical at the boundary (an object on the slow path cannot
    // escape beyond 2^53 - 1 only to then be rejected if the backend
    // flips to Lua later).
    //
    // The `checked_add` u64-overflow branch below is unreachable: the
    // `>= SEQUENCE_CEILING` check returns first for every `sequence_num`
    // that could overflow, so the addition never sees a value near
    // `u64::MAX`. It stays as defence in depth — a silent wrap-around
    // would reset all version-indexed filters in the diff engine to 0.
    //
    // b.dmu item 6: the *commit* of `state.sequence_num = new_version`
    // is deferred to the end of the function so a mid-delta helper
    // failure leaves `state.sequence_num` unchanged. The other state
    // maps may still be partially mutated — see the partial-mutation
    // contract on `apply_delta`.
    const SEQUENCE_CEILING: u64 = (1u64 << 53) - 1;
    if state.sequence_num >= SEQUENCE_CEILING {
        return Err(DeltaError::SequenceOverflow);
    }
    let new_version = state
        .sequence_num
        .checked_add(1)
        .ok_or(DeltaError::SequenceOverflow)?;

    // b.ot3: per-delta planning state. Children-list writes are coalesced
    // into one write per dirty parent, `(parent_id, name)` lookups scan once
    // and then go through an index built at the second, and a cascade's
    // field / time-sample purge is deferred to one purge per delta. All of
    // it is stack-local and dies with this frame on every exit path,
    // including each `?` below. A bounded plan's build sites ask `meter`
    // first.
    let inserts = delta
        .as_ref()
        .map_or_else(StructuralInserts::default, StructuralInserts::of);
    let mut plan = ChildPlan::new(meter, inserts, hooks);

    if let Some(delta) = delta {
        // Step 2 — reset path: isDiff=true AND baseVersion=0 (SDD §9.1.3).
        if delta.is_diff && delta.base_version == 0 {
            let source_format = if delta.source_format.is_empty() {
                None
            } else {
                Some(delta.source_format.as_str())
            };
            apply_reset(state, new_version, source_format, &mut plan, sink, stats);
        }
        // isDiff=false: proceed directly to commands (SDD §9.1.2 — do NOT reject).

        // Step 3 — structural commands (exhaustive match, no wildcard). The
        // two birth commands also get the whole command list, from which a
        // birth's adoption derives the ids the delta can birth
        // (`AdoptionIndex`).
        let commands = delta.struct_commands.as_slice();
        for cmd in commands {
            failpoint_step()?;
            match cmd {
                StructCommand::CreateSection(c) => {
                    apply_create_section(state, new_version, c, commands, &mut plan, sink, stats)?;
                }
                StructCommand::DeleteSection(c) => {
                    apply_delete_section(state, new_version, c.section_id, &mut plan, sink, stats);
                }
                StructCommand::MoveSection(c) => {
                    apply_move_section(state, new_version, c, &mut plan, sink, stats)?;
                }
                StructCommand::ReorderChildren(c) => {
                    apply_reorder_children(state, new_version, c, &mut plan, sink, stats);
                }
                StructCommand::DiffChangeSection(c) => {
                    apply_diff_change_section(
                        state,
                        new_version,
                        c,
                        commands,
                        &mut plan,
                        sink,
                        stats,
                    )?;
                }
                StructCommand::DiffDeleteSection(_) => {
                    // Silently ignored on ingest path (SDD §9.1.4f).
                }
            }
        }

        // Step 3b — close of the structural phase (b.ot3 parts B and E).
        //
        // Placement is load-bearing in one direction. No structural-command
        // helper reads `state.fields` / `state.time_samples`, and no
        // children-list helper runs after this point, so running both passes
        // here is observationally identical to the pre-b.ot3 per-command
        // emission *inside* the delta. Running either one LATER would not
        // be: a `DeleteField` emitted after step 4's `SetField` for the same
        // row would invert the `(row_key, new_hash)` fold in
        // `ext_rows_written` (`src/storage/layer.rs`), releasing an
        // external-value reference the commit still holds.
        failpoint_step()?;
        finalize_children_lists(state, &plan, sink, stats);
        failpoint_step()?;
        purge_cascaded_rows(state, &mut plan, sink, stats);

        // Step 4 — field assignments.
        for sf in &delta.set_fields {
            failpoint_step()?;
            apply_set_field(state, new_version, sf, &mut plan.hooks, sink)?;
        }

        // Step 5 — time sample assignments.
        for ts in &delta.time_samples {
            failpoint_step()?;
            apply_time_sample(state, new_version, ts, &mut plan.hooks, sink)?;
        }
    }

    // b.dmu item 3: enforce `MAX_SECTION_TREE_DEPTH` on the WRITE path.
    // The pre-existing read-path enforcement in
    // `DiffEngine::sections_in_dfs_order` catches deep state at diff
    // time but cannot recover an already-stored deep tree —
    // `cascade_delete_section` walks `state.children` with no depth
    // bound, so once-stuck state is unrecoverable through normal client
    // RPCs. Checking the *post-mutation* tree here is the preferred
    // closure: stored state can never exceed the cap, so the read-side
    // `SectionTreeTooDeep` becomes unreachable on any object whose
    // history flowed exclusively through this engine. (Legacy state
    // that predates the check still trips the cap on the read side and
    // surfaces as gRPC INTERNAL; clients can recover by overwriting
    // the layer via `CreateLayer` rather than `UpdateLayer`.) Runs
    // before the sequence_num commit so a depth breach also does not
    // advance `state.sequence_num` under item 6.
    //
    // Skipped only in `ApplyMode::Replay` for a delta that cannot change
    // the tree (see `skip_depth_walk` above and `ApplyMode::Replay`).
    //
    // A plan whose hooks keep the structural cache's plan indexes gets its
    // verdict from them: incremental while the state is known to have passed
    // the walk, the same full walk otherwise (`domain::plan_index`). Its walk
    // projects the sections it is about to visit into a bounded plan's
    // count before it grows (`Projection::WalkedSections`).
    if !skip_depth_walk {
        failpoint_step()?;
        let meter = plan.meter;
        let mut walk = |sections: u64| {
            if let Some(meter) = meter {
                meter.project(Projection::WalkedSections(sections));
            }
        };
        pass_timing::time(Pass::DepthCheck, || {
            match plan.hooks.depth_verdict(state, &mut walk) {
                Some(verdict) => verdict,
                None => check_section_tree_depth(state),
            }
        })?;
    }

    // b.dmu item 6: commit the sequence_num bump only after every command
    // helper succeeded. A mid-delta error returned above leaves
    // `state.sequence_num` at its prior value. This is the single
    // field-level guarantee the partial-mutation contract makes — all
    // other state maps may be partially mutated and the caller must
    // discard.
    write::set_sequence_num(state, &mut plan.hooks, new_version);

    // Step 6 — emit header mutation for the updated sequence_num.
    sink.push_with(|| StorageMutation::SetObjectHeader {
        object_id: state.object_id,
        sequence_num: state.sequence_num,
        first_version: state.first_version,
        source_format: if state.source_format.is_empty() {
            None
        } else {
            Some(state.source_format.clone())
        },
    });

    // After a full delta has settled, every non-orphan section must live in
    // its parent's children list, every listed id must be a row naming that
    // parent, and no list may sit under an id with no row. Fires in debug
    // builds only.
    debug_assert_cross_map_consistent(state);
    debug_assert_cache_index_superset(&plan);

    Ok(())
}

// ── Per-delta finalization (b.ot3 parts B and E) ──────────────────────────────

/// Emit one children-list write per parent this delta dirtied, walking
/// `plan.order` — every parent the delta *touched*, in first-touched order —
/// and skipping the ones never marked dirty. (A parent can be registered
/// without being dirtied: `entry_cached` and `restore_cache` register one to
/// hang a cache off.)
///
/// Pre-b.ot3 every children-list helper emitted a `SetChildren` carrying a
/// clone of the whole list, so creating N siblings under one parent retained
/// N(N+1)/2 child ids. This pass emits the final list once instead.
///
/// The ordering argument is three lines: every *inline* `DeleteChildrenList`
/// (cascade delete, reset) is emitted during the structural phase, before any
/// of this pass's writes; within this pass a dirty parent takes exactly one of
/// the two arms below — `SetChildren` when its list still exists, its own
/// `DeleteChildrenList` when it does not — so a parent is never both deleted
/// and re-written here; and `SetChildren` is itself DEL-then-RPUSH at the
/// storage layer. So delete-then-recreate of one parent inside a single delta
/// lands as `DEL … DEL … RPUSH` with the same end state.
///
/// The **empty** vector is emitted, not dropped: an empty `SetChildren` is
/// the `DEL`-without-`RPUSH` plus `SREM le:{oid}:obj:chs` fork at the
/// storage layer, and skipping it would strand the parent in the children
/// index (SDD §7.7's "member present ⟺ children key exists").
fn finalize_children_lists<R: ValueRows, H>(
    state: &ObjectStateOf<R>,
    plan: &ChildPlan<'_, H>,
    sink: &mut impl PlanSink,
    stats: &mut ApplyDeltaStats,
) {
    for &parent_id in &plan.order {
        let Some(entry) = plan.parents.get(&parent_id) else {
            continue;
        };
        if !entry.dirty {
            continue;
        }
        match state.children.get(&parent_id) {
            Some(children) => {
                stats.children_writes_emitted = stats.children_writes_emitted.saturating_add(1);
                sink.push_with(|| StorageMutation::SetChildren {
                    section_id: parent_id,
                    children: children.clone(),
                });
            }
            None => {
                // The list is gone. If a cascade or the reset path already
                // emitted the delete inline, the finalizer must not emit a
                // second one — and must never "finalize" a deleted key back
                // into existence.
                if plan.deleted_lists.contains(&parent_id) {
                    continue;
                }
                // Reset fold-in: `apply_reset` erases `state.children`
                // wholesale but only emits `DeleteChildrenList` for ids that
                // had a `state.sections` row. The synthetic root `0` never
                // does, so before b.ot3 a reset left `le:{oid}:obj:ch:0`
                // populated with dead ids and `"0"` still in `obj:chs` —
                // which the snapshot read then re-read into the next
                // commit's `state.children[0]`, tripping the cross-map debug
                // assert and re-persisting ghosts in release builds.
                //
                // The root is this arm's net-new reach on a stored object,
                // plus one corrupt-index shape: an `obj:secs` member whose
                // section hash is missing loads a children list with no
                // `state.sections` row, so the per-section drain skips it and
                // this arm catches it. A children list under a *dangling*
                // parent (SDD §10.4) — one an earlier build stored, since
                // this engine keeps no list under a parent with no row — is
                // NOT reached: every snapshot read keeps only the lists under
                // the root and the live sections, so such a key is absent
                // from `state.children` and survives the reset with its index
                // member intact (SDD §7.7's ⟺ therefore still holds). The
                // structural commit path deletes it if a later commit creates
                // its parent; otherwise it is reaped at object retirement,
                // off the `obj:chs` UNLINK inventory.
                stats.children_writes_emitted = stats.children_writes_emitted.saturating_add(1);
                sink.push_with(|| StorageMutation::DeleteChildrenList {
                    section_id: parent_id,
                });
            }
        }
    }
}

/// Erase the fields and time samples of every section this delta
/// cascade-deleted, in one purge per delta.
///
/// Pre-b.ot3 `cascade_delete_section` ran a `retain` over both maps *per
/// deleted section*, approaching O(deleted sections × total fields). One
/// purge per delta bounds it, and only a delta that cascades pays it:
///
/// - on the full form (`ObjectState`: the WATCH path, readers' replay,
///   checkpoint compose), one `retain` over each map, O(F + T);
/// - on the compact form (`CompactState`: every structural plan on the
///   cached route and the repair's replays), the field purge costs
///   O(min(D × N, F)) for `D` purged sections and `N` interned-name id
///   slots, live and free — a probe of each purged section's rows by name,
///   or a scan when that is cheaper (`domain::types::PURGE_PROBE_RATIO`) —
///   and the time-sample purge O(T), nothing when the object holds no time
///   samples.
///
/// Such a delta carries a structural command, so it also pays the
/// end-of-delta depth check in every [`ApplyMode`]. A delta that replay
/// mode lets skip the check cascades nothing, and this purge returns at
/// once.
///
/// `apply_reset`'s own erasure is deliberately left where it is: it is
/// already a single drain of each map and moving it would reorder its
/// emissions against the structural phase for no gain.
fn purge_cascaded_rows<R: ValueRows, H: EngineHooks<R>>(
    state: &mut ObjectStateOf<R>,
    plan: &mut ChildPlan<'_, H>,
    sink: &mut impl PlanSink,
    stats: &mut ApplyDeltaStats,
) {
    if plan.purge_ids.is_empty() {
        return;
    }
    stats.field_purge_passes = stats.field_purge_passes.saturating_add(1);
    let ChildPlan {
        purge_ids,
        hooks,
        meter,
        ..
    } = plan;
    let meter = *meter;

    // One purge per row store — each removes every row of a purged section
    // (the compact field store by probe or by scan, see the function docs)
    // and hands each removed key to its closure, which emits the matching
    // mutation (SDD §9.2.6, §9.2.7), fields first. The key is lent,
    // so only the collecting arm clones it. A compact store keys a time
    // sample by its stored bits, so `0.0` and `-0.0` are two rows and both
    // are deleted. Both closures emit into the one sink, so it is shared
    // through a `RefCell`; they never run at the same time.
    //
    // Rule E: each time-0 row purged is noted, and once the purge is done the
    // other spelling of each such section's time 0 is deleted too, unless the
    // purge deleted it already ([`ZeroTimeDeletes`]) — still inside this
    // delete run, before anything else is emitted.
    let mut zero_time = sink.keeps_row_deletes().then(ZeroTimeDeletes::default);
    let sink = std::cell::RefCell::new(sink);
    pass_timing::time(Pass::CascadePurge, || {
        write::purge_value_rows(
            state,
            hooks,
            purge_ids,
            |section_id, key_name| {
                sink.borrow_mut()
                    .push_with(|| StorageMutation::DeleteField {
                        section_id,
                        key_name: key_name.to_owned(),
                    });
            },
            |section_id, time| {
                if let Some(zero_time) = zero_time.as_mut() {
                    zero_time.note(meter, section_id, time);
                }
                sink.borrow_mut()
                    .push_with(|| StorageMutation::DeleteTimeSample { section_id, time });
            },
        );
        if let Some(zero_time) = zero_time.take() {
            zero_time.emit_other_spellings(&mut **sink.borrow_mut());
        }
    });
}

/// Rule E's pairing for one delete run — a reset's sample drain or a
/// cascade's purge: **a run that deletes a section's time-sample row at time
/// 0, at either sign, deletes that section's time 0 at both stored spellings,
/// `+0.0` and `-0.0`.**
///
/// The store spells a time-sample row by its time's bits, so `0.0` and
/// `-0.0` are two keys, and the plan's state can hold fewer spellings of a
/// section's time 0 than the store does: the structural cache replays an
/// older build's `-0.0` commit at `+0.0` (the parser canonicalizes it), and
/// the reconciler's signed-zero merge rewrites rows without moving the
/// entry's cut. A delete names only the spellings the state holds, and the
/// structural commit checks each named row's existence
/// (`storage::structural::PreImageDigests`), so a spelling the state lacks
/// would survive the delete unseen. With both spellings named, the one the
/// state lacks carries an `absent` digest: if the store holds it after all,
/// the commit answers `CACHE_DIVERGED` and the call reloads the entry and
/// plans again, deleting both rows. If the store does not hold it, the
/// delete is a no-op (`DEL` and `ZREM` of absent keys, no `obj:ext` effect),
/// on the structural commit and the WATCH path alike.
///
/// Both row forms emit it: the full form holds time 0 at `+0.0`, so it adds
/// the `-0.0` delete exactly as a compact state holding only `+0.0` does, and
/// the two forms' plans stay equal up to the order within a run.
///
/// The run [`Self::note`]s every time-sample delete it emits; then, still
/// inside the run, [`Self::emit_other_spellings`] emits, in ascending section
/// id (never a hash set's order, so a state and its clone emit the same
/// list), one `DeleteTimeSample` at the other spelling for each section whose
/// time 0 the run deleted at one spelling only. A section deleted at both
/// gets nothing more. Only rows the run actually deleted are noted, so a
/// section with no time-0 row, or a ghost id, gets nothing. The added record
/// removes nothing from the state, so it reports nothing to the plan's hooks
/// or journal.
///
/// It is a mutation, not a delta: a `-0.0` `DeleteTimeSample` names a stored
/// key, and nothing may refuse it (the `-0.0` refusal applies to a delta's
/// times only, `validate_delta_content`).
///
/// A plan into a sink that keeps no row delete ([`PlanSink::keeps_row_deletes`])
/// skips the pairing, since its only output is a row delete.
#[derive(Debug, Default)]
struct ZeroTimeDeletes {
    /// `(section_id, negative)` for each time-0 delete the run emitted.
    deleted: Vec<(u64, bool)>,
}

impl ZeroTimeDeletes {
    /// Note one time-sample delete the run is emitting. Only a time-0 delete
    /// is kept. A bounded plan first projects one row on `meter`, so the
    /// budget is asked before the scratch grows; the meter retires a pending
    /// row on every emission, so the run's very next emission — the noted
    /// row's own delete, pushed right after this call — retires it. Nothing
    /// stays pending: there is no lasting over-projection, and the peak never
    /// exceeds the kept count by more than that one transient row. The
    /// scratch (16 B per noted row) is covered only by those rows' kept
    /// count; an added record is counted like any other emission and
    /// retires nothing of its own.
    fn note(&mut self, meter: Option<&dyn PlanMeter>, section_id: u64, time: OrderedFloat<f64>) {
        let time = time.into_inner();
        if time != 0.0 {
            return;
        }
        if let Some(meter) = meter {
            meter.project(Projection::Rows(1));
        }
        self.deleted.push((section_id, time.is_sign_negative()));
    }

    /// Emit, in ascending section id, a `DeleteTimeSample` at the other
    /// spelling of each section whose time 0 the run deleted at one spelling
    /// only.
    fn emit_other_spellings(mut self, sink: &mut impl PlanSink) {
        self.deleted.sort_unstable();
        for section in self.deleted.chunk_by(|a, b| a.0 == b.0) {
            let section_id = section[0].0;
            let negative = section.iter().any(|&(_, negative)| negative);
            let positive = section.iter().any(|&(_, negative)| !negative);
            if negative == positive {
                continue;
            }
            let other = if positive { -0.0 } else { 0.0 };
            sink.push_with(|| StorageMutation::DeleteTimeSample {
                section_id,
                time: OrderedFloat(other),
            });
        }
    }
}

/// b.dmu item 3: walk `state.children` and reject if the deepest reachable
/// chain exceeds [`MAX_SECTION_TREE_DEPTH`].
///
/// Runs on the WRITE path *after* the delta's structural commands have
/// been applied to `state` but *before* `state.sequence_num` is committed.
/// Walks from synthetic root `0` (the parent_id reserved for top-level
/// sections) so every section reachable from the root is examined.
/// Disconnected / orphan sections are out of scope here — the diff engine
/// likewise doesn't traverse them, so they cannot trigger an
/// unrecoverable read-path failure.
///
/// Cycles in stored state (which the cycle-tolerant `collect_subtree` and
/// `is_descendant_of` both guard against) are handled by a visited-set
/// dedup — a self-parenting section visited twice short-circuits without
/// looping forever. The dedup is applied **before** the depth check so a
/// corrupt cycle that loops back to a shallow node is silently deduped
/// rather than rejected, matching the read-side
/// `DiffEngine::sections_in_dfs_order` (see `src/domain/diff.rs`). This
/// preserves true write-side ⇔ read-side symmetry: any state the write
/// path admits, the read path can also serialize.
fn check_section_tree_depth<R: ValueRows>(state: &ObjectStateOf<R>) -> Result<(), DeltaError> {
    check_section_tree_depth_capped(state, crate::domain::limits::MAX_SECTION_TREE_DEPTH)
}

/// [`check_section_tree_depth`] at a cap of `cap` levels rather than
/// [`MAX_SECTION_TREE_DEPTH`](crate::domain::limits::MAX_SECTION_TREE_DEPTH):
/// the persistent plan indexes' incremental check (`domain::plan_index`)
/// falls back to it, and its equivalence tests run both at a small cap.
pub(crate) fn check_section_tree_depth_capped<R: ValueRows>(
    state: &ObjectStateOf<R>,
    cap: usize,
) -> Result<(), DeltaError> {
    // Stack carries `(section_id, depth)`. Synthetic root (`0`) is at
    // depth 0; its direct children are at depth 1. Visited set guards
    // against cycles in stored state (the cycle guard in
    // `cascade_delete_section`'s collect_subtree mirrors this).
    let mut stack: Vec<(u64, usize)> = state
        .children
        .get(&0)
        .map(|kids| kids.iter().map(|&cid| (cid, 1)).collect())
        .unwrap_or_default();
    let mut visited: std::collections::HashSet<u64> =
        std::collections::HashSet::with_capacity(state.sections.len());

    while let Some((sid, depth)) = stack.pop() {
        // Visited check FIRST — matches the read-side ordering in
        // `DiffEngine::sections_in_dfs_order`. A corrupt cycle that loops
        // back to an already-visited node short-circuits here without
        // tripping the depth cap, so write-side admits exactly the same
        // states the read-side can serialize.
        if !visited.insert(sid) {
            continue;
        }
        if depth > cap {
            return Err(DeltaError::SectionTreeTooDeep { depth, cap });
        }
        if let Some(kids) = state.children.get(&sid) {
            for &cid in kids {
                stack.push((cid, depth + 1));
            }
        }
    }
    Ok(())
}

// ── Reset path ────────────────────────────────────────────────────────────────

fn apply_reset<R: ValueRows, H: EngineHooks<R>>(
    state: &mut ObjectStateOf<R>,
    new_version: u64,
    source_format: Option<&str>,
    plan: &mut ChildPlan<'_, H>,
    sink: &mut impl PlanSink,
    stats: &mut ApplyDeltaStats,
) {
    // b.ot3 (P1) reset fold-in: register every key `state.children` holds
    // before the wholesale clear below, so the finalization pass emits the
    // `DeleteChildrenList`s the per-section drain loop cannot — chiefly the
    // synthetic root `0`, which is never a section.
    //
    // The reach is exactly what the snapshot read placed in
    // `state.children`: lists under the `obj:secs` members plus the root
    // (on an object carrying the `chs_complete` marker the read takes the
    // parents from `obj:chs` but keeps only those, so the reach is the same
    // for every state the object's own writes produce). A children list under
    // a *dangling* parent (SDD §10.4) is therefore out of reach here by
    // construction — it is not in the map to register. This engine stores no
    // such list (`insert_child_into_parent` keeps none under a parent with no
    // row), so only a list an earlier build stored can be left; it survives
    // the reset together with its `obj:chs` member, is deleted by the
    // structural commit path if a later commit creates its parent, and is
    // otherwise reaped at object retirement.
    //
    // A bounded plan first projects the drain's rows, before the key list and
    // the per-list registration below allocate: three per section (its
    // `DeleteSection`, `CreateTombstone` and `DeleteChildrenList`) and one per
    // list with no row (the finalizer's `DeleteChildrenList`, chiefly the
    // root's). Each emitted row retires one. Counted without allocating.
    let sections = u64::try_from(state.sections.len()).unwrap_or(u64::MAX);
    let lists_without_row = state
        .children
        .keys()
        .filter(|parent_id| !state.sections.contains_key(parent_id))
        .count();
    plan.project(Projection::Rows(sections.saturating_mul(3).saturating_add(
        u64::try_from(lists_without_row).unwrap_or(u64::MAX),
    )));
    // Sorted so the emitted order does not depend on HashMap layout.
    let mut children_keys: Vec<u64> = state.children.keys().copied().collect();
    children_keys.sort_unstable();
    for parent_id in children_keys {
        plan.mark_dirty(parent_id);
    }

    // Erase every row and list (`write::take_for_reset`). Without hooks this
    // is the in-place drain: the section and children maps are drained as
    // the deletes are emitted and handed back to `state` empty, with their
    // buckets (`write::finish_reset`), and the value rows are drained in
    // place. With hooks every table is only read here and then handed to the
    // hooks whole, in O(1). Both emit the same deletes, in the order a drain
    // hands the rows over.
    let mut taken = write::take_for_reset::<R, H>(state);

    // Generate tombstones and children-list deletes. Each tombstone reads the
    // row's own `first_version`; the pre-b.dmu `unwrap_or(0)` would silently
    // emit `first_version=0` if a re-fetch somehow missed.
    if H::OBSERVES {
        for (&sid, section) in &taken.sections {
            // The row is lent, so only the collecting arm clones its name.
            let (parent_id, first_version) = (section.parent_id, section.first_version);
            emit_reset_section(
                sink,
                stats,
                plan,
                new_version,
                sid,
                parent_id,
                first_version,
                || section.name.clone(),
            );
        }
    } else {
        for (sid, section) in taken.drain_sections() {
            // The drained row is owned, so its name moves into the mutation —
            // nothing is cloned, and the discarding arm drops it unbuilt.
            let (parent_id, first_version) = (section.parent_id, section.first_version);
            emit_reset_section(
                sink,
                stats,
                plan,
                new_version,
                sid,
                parent_id,
                first_version,
                || section.name,
            );
        }
        taken.clear_children();
    }
    plan.invalidate_all_caches();
    // No cache lists anything any more, so the whole dependency index is
    // vacuous. Clearing it keeps the per-delta memory bound tied to what is
    // still cached rather than to everything the delta ever cached, and the
    // live pair count has to fall with the map it describes or every later
    // peak reading would carry the pre-reset index along with it.
    plan.listed_in.clear();
    plan.listed_pairs = 0;
    // Every section row is gone; a later probe rebuilds from the (now
    // empty) map rather than reading through a stale index. The same holds
    // for the adoption index, which a reset (always the delta's first step)
    // precedes anyway.
    plan.names = None;
    plan.adoption = None;

    // Erase all fields. Each key is lent: the collecting arm clones it into
    // its mutation (a compact store holds no owned key to move), and the
    // discarding arm pays nothing at all.
    let mut on_field = |sid: u64, key: &str| {
        sink.push_with(|| StorageMutation::DeleteField {
            section_id: sid,
            key_name: key.to_owned(),
        });
    };
    write::reset_fields(state, &taken, &mut on_field);

    // Erase all time samples (in a compact store, `0.0` and `-0.0` are two
    // rows, and both go). Rule E: each time-0 row drained is noted, and once
    // the drain is done the other spelling of each such section's time 0 is
    // deleted too, unless the drain deleted it already ([`ZeroTimeDeletes`]) —
    // still inside this delete run, before the delta's own writes.
    let meter = plan.meter;
    let mut zero_time = sink.keeps_row_deletes().then(ZeroTimeDeletes::default);
    let mut on_sample = |sid: u64, time: OrderedFloat<f64>| {
        if let Some(zero_time) = zero_time.as_mut() {
            zero_time.note(meter, sid, time);
        }
        sink.push_with(|| StorageMutation::DeleteTimeSample {
            section_id: sid,
            time,
        });
    };
    write::reset_samples(state, &taken, &mut on_sample);
    if let Some(zero_time) = zero_time {
        zero_time.emit_other_spellings(sink);
    }
    write::finish_reset(state, &mut plan.hooks, taken);

    // Update version bookkeeping.
    write::set_first_version(state, &mut plan.hooks, new_version);

    // Store sourceFormat on the reset path (SDD §9.1.3c) — this is the first
    // version of a new object generation.
    if let Some(fmt) = source_format {
        write::set_source_format(state, &mut plan.hooks, fmt);
    }

    // Reset nextSetOrder (SDD §9.1.3d).
    write::set_next_set_order(state, &mut plan.hooks, 0);
}

/// One reset section's three deletes — its `DeleteSection`, `CreateTombstone`
/// and `DeleteChildrenList` — with `name` built only by a collecting sink:
/// moved from a drained row, or cloned from a lent one.
#[allow(clippy::too_many_arguments)]
fn emit_reset_section<H>(
    sink: &mut impl PlanSink,
    stats: &mut ApplyDeltaStats,
    plan: &mut ChildPlan<'_, H>,
    new_version: u64,
    section_id: u64,
    parent_id: u64,
    first_version: u64,
    name: impl FnOnce() -> String,
) {
    sink.push_with(|| StorageMutation::DeleteSection {
        section_id,
        parent_id,
        name: name(),
    });
    sink.push_with(|| StorageMutation::CreateTombstone {
        section_id,
        first_version,
        last_version: new_version,
    });
    stats.children_writes_emitted = stats.children_writes_emitted.saturating_add(1);
    sink.push_with(|| StorageMutation::DeleteChildrenList { section_id });
    plan.deleted_lists.insert(section_id);
}

// ── CommandDiffChangeSection ──────────────────────────────────────────────────

fn apply_diff_change_section<R: ValueRows, H: EngineHooks<R>>(
    state: &mut ObjectStateOf<R>,
    new_version: u64,
    cmd: &CommandDiffChangeSection,
    commands: &[StructCommand],
    plan: &mut ChildPlan<'_, H>,
    sink: &mut impl PlanSink,
    stats: &mut ApplyDeltaStats,
) -> Result<(), DeltaError> {
    let section_id = cmd.section_id;
    let parent_id = cmd.parent_id;
    let name = cmd.section_name.clone();
    let section_type = cmd.section_type;
    let section_order = cmd.section_order;

    // b.dmu item 2: detect `(parent_id, name)` conflict at the destination
    // before mutating. Mirrors `apply_move_section` edges 4 & 5 — without
    // this guard a rename can produce two siblings with the same
    // `(parent_id, name)`, which the diff path then keys on at every
    // child-by-name lookup. Cascade-delete the conflicting sibling on the
    // *new-section* branch (matching CreateSection's PRD §2.3.4 policy);
    // on the *existing-section* branch return the new
    // `DuplicateSiblingName` error variant so the client gets a clear
    // protocol-level signal rather than silent corruption. The
    // existing-section branch is the rename path where surfacing the
    // conflict is more useful than the silent cascade — clients can
    // disambiguate before retrying.
    let conflict_id = find_child_by_name(state, plan, stats, parent_id, &name, Some(section_id));

    if let Some(existing) = state.sections.get(&section_id).cloned() {
        if let Some(cid) = conflict_id {
            return Err(DeltaError::DuplicateSiblingName {
                section_id,
                parent_id,
                conflicting_id: cid,
                name,
            });
        }

        let old_parent = existing.parent_id;
        let old_order = existing.section_order;
        let order_before = effective_order(&state.sections, section_id);

        let updated = Section {
            section_id,
            parent_id,
            name,
            section_type,
            section_order,
            first_version: existing.first_version,
            last_version: new_version,
        };
        write::insert_section(state, &mut plan.hooks, updated.clone());
        // Keep the `(parent_id, name)` index in step with `state.sections`:
        // this row moved off `(existing.parent_id, existing.name)` onto
        // `(parent_id, updated.name)`.
        plan.name_removed(existing.parent_id, &existing.name, section_id);
        plan.name_inserted(parent_id, &updated.name, section_id);
        // `existing` is already an owned copy of the replaced row, so its
        // key moves into the mutation without a further clone.
        sink.push_with(|| StorageMutation::SetSection {
            section: updated,
            previous: Some(SectionKey {
                parent_id: existing.parent_id,
                name: existing.name,
            }),
        });

        // Every cached parent listing this child mirrors its `section_order`,
        // and stored state can list a child under a parent other than the one
        // its row names (a stale listing this engine does not produce but
        // tolerates; see `insert_child_into_parent`) — neither branch below
        // touches those parents' caches. Comparing the effective order across the write
        // is what decides it, so a future edit to the row shape inherits the
        // answer instead of inheriting an obligation to re-derive it.
        //
        // `old_parent` is kept on **both** arms because it is the parent whose
        // one falsified entry the next statement repairs: `rebuild_children_list`
        // re-places it on the same-parent arm, `remove_child_from_parent` deletes
        // it on the cross-parent arm. See `ChildPlan::invalidate_order_listeners`.
        if effective_order(&state.sections, section_id) != order_before {
            plan.invalidate_order_listeners(section_id, Some(old_parent), stats);
        }

        if old_parent != parent_id {
            remove_child_from_parent(state, plan, stats, old_parent, section_id);
            insert_child_into_parent(state, plan, stats, parent_id, section_id, section_order);
        } else if old_order != section_order {
            rebuild_children_list(state, plan, stats, parent_id, section_id);
        }
    } else {
        // New section. Mirror `apply_create_section`: cascade-delete the
        // conflicting sibling (if any) under SDD §9.1.4c / PRD §2.3.4.
        if let Some(cid) = conflict_id {
            cascade_delete_section(state, new_version, cid, Some(parent_id), plan, sink, stats);
        }

        // An id with no row is a *ghost* wherever it is still listed, and a
        // ghost's effective order is `0`. Giving it a live row at a non-zero
        // order therefore falsifies every cache that lists it, `parent_id`'s
        // own included — the insert below returns early when the child is
        // already listed, so it refreshes nothing by itself. The before/after
        // comparison says exactly that without special-casing the ghost
        // encoding; nothing is kept, because the row had no previous parent.
        let order_before = effective_order(&state.sections, section_id);
        let section = Section {
            section_id,
            parent_id,
            name,
            section_type,
            section_order,
            first_version: new_version,
            last_version: new_version,
        };
        write::insert_section(state, &mut plan.hooks, section.clone());
        plan.name_inserted(parent_id, &section.name, section_id);
        // No row with this id existed (a ghost id included), so there is no
        // pre-image.
        sink.push_with(|| StorageMutation::SetSection {
            section,
            previous: None,
        });
        if effective_order(&state.sections, section_id) != order_before {
            plan.invalidate_order_listeners(section_id, None, stats);
        }
        // Insert into the parent's children list. Under a missing parent no
        // list is kept and the new row stays an orphan until the parent is
        // born (SDD §10.4; `insert_child_into_parent`).
        insert_child_into_parent(state, plan, stats, parent_id, section_id, section_order);
        // The id now has a row, so rows that already named it join its list.
        adopt_orphans(state, new_version, section_id, commands, plan, sink, stats);
    }
    Ok(())
}

// ── CommandCreateSection ──────────────────────────────────────────────────────

/// Apply a `CommandCreateSection` (SDD §9.1.4c).
///
/// The schema's `CommandCreateSection` carries no `sectionOrder`, and every
/// client treats a received create as an **append**: the Kit client, the C++
/// client library and the legacy server all give the new child a key one
/// above the largest among its siblings, and a client sends a
/// `ReorderChildren` after a create only
/// when the order it wants differs from that append. The stored key is
/// computed the same way ([`append_order`]), so the child lands last and
/// later explicit-order inserts compare against the keys clients hold. A
/// late joiner rebuilding from the stored list therefore sees the order a
/// live subscriber built by appending.
///
/// Rows that already named the new id as their parent — orphans, listed
/// nowhere — join the new section's list before the next command runs
/// ([`adopt_orphans`]). A later command of the same delta then places its
/// children under the new section by the usual rules: a later
/// `CreateSection` or `MoveSection` appends after the adopted rows
/// ([`append_order`]), and a later `DiffChangeSection` inserts by its own
/// `section_order`, landing after them only when its key is at least the
/// largest of theirs.
fn apply_create_section<R: ValueRows, H: EngineHooks<R>>(
    state: &mut ObjectStateOf<R>,
    new_version: u64,
    cmd: &CommandCreateSection,
    commands: &[StructCommand],
    plan: &mut ChildPlan<'_, H>,
    sink: &mut impl PlanSink,
    stats: &mut ApplyDeltaStats,
) -> Result<(), DeltaError> {
    let parent_id = cmd.parent_id;
    let section_id = cmd.section_id;
    let name = cmd.section_name.clone();
    let section_type = cmd.section_type;

    // CreateSection must allocate a fresh section_id — silently overwriting
    // an existing section corrupts the tree (wrong parent linkage, wrong
    // first_version, orphaned fields/time-samples). Reject up front.
    if state.sections.contains_key(&section_id) {
        return Err(DeltaError::DuplicateSection { section_id });
    }

    // Ignore if parent is non-root and missing (SDD §9.1.4c). Pre-b.39x
    // this swallowed the command with no log/metric/error — combined with
    // H4 (base_version unvalidated) it was a silent footgun for clients
    // whose layer ended up unexpectedly empty. Emit a structured
    // `tracing::debug!` so the rate is observable in logs at debug level,
    // and bump a stats counter so the caller can surface the rate via
    // `delta_create_section_skipped_missing_parent_total` (b.39x item 7).
    // Domain stays pure: tracing emits a structured event (subscribers
    // bind separately, not an I/O call site) and the stats out-parameter
    // hands the counter increment off to the caller layer.
    if parent_id != 0 && !state.sections.contains_key(&parent_id) {
        tracing::debug!(
            target: "live_edit::domain::delta",
            section_id,
            parent_id,
            command = "CreateSection",
            "ignoring CreateSection with missing parent"
        );
        stats.create_section_skipped_missing_parent = stats
            .create_section_skipped_missing_parent
            .saturating_add(1);
        return Ok(());
    }

    // Name conflict: cascade-delete the conflicting sibling (SDD §9.1.4c / PRD §2.3.4).
    let conflicting_id = find_child_by_name(state, plan, stats, parent_id, &name, Some(section_id));
    if let Some(conflict_id) = conflicting_id {
        cascade_delete_section(
            state,
            new_version,
            conflict_id,
            Some(parent_id),
            plan,
            sink,
            stats,
        );
    }

    // Computed after the conflict cascade, so a deleted same-name sibling's
    // key no longer counts.
    let section_order = append_order(state, plan, stats, parent_id);

    // Same order-side guard every row-writing site runs. The id had no row
    // (`DuplicateSection` returned above), so its effective order was the
    // ghost `0`; it fires whenever the append key is non-zero, which matters
    // only when stored state still lists the id somewhere as a ghost — the
    // caches mirroring that listing hold `0` for it. Nothing is kept: the
    // row had no previous parent.
    let order_before = effective_order(&state.sections, section_id);
    let section = Section {
        section_id,
        parent_id,
        name,
        section_type,
        section_order,
        first_version: new_version,
        last_version: new_version,
    };
    write::insert_section(state, &mut plan.hooks, section.clone());
    plan.name_inserted(parent_id, &section.name, section_id);
    // `DuplicateSection` above guarantees no row with this id existed.
    sink.push_with(|| StorageMutation::SetSection {
        section,
        previous: None,
    });
    if effective_order(&state.sections, section_id) != order_before {
        plan.invalidate_order_listeners(section_id, None, stats);
    }
    insert_child_into_parent(state, plan, stats, parent_id, section_id, section_order);
    // The id now has a row, so rows that already named it join its list.
    adopt_orphans(state, new_version, section_id, commands, plan, sink, stats);
    Ok(())
}

// ── Adoption ──────────────────────────────────────────────────────────────────

/// List every row naming `section_id` as its parent under `section_id`, which
/// has just been given a row. Both places a row is born call it right after
/// listing the new row under its own parent: [`apply_create_section`] and
/// [`apply_diff_change_section`]'s new-row branch.
///
/// A row whose parent has no row is an *orphan*, and an orphan is listed
/// nowhere, because [`insert_child_into_parent`] keeps no list under a parent
/// with no row (SDD §10.4). Once the parent is born its orphans are its
/// children, so they join its list here, in `(section_order, section_id)`
/// order. That order comes from the rows alone, so the resulting list is the
/// same however a client split its commands into deltas. It runs before the
/// delta's next command, so the adopted rows are already listed when that
/// command runs: a later `CreateSection` or `MoveSection` under the new
/// section appends after them ([`append_order`]), a later `DiffChangeSection`
/// inserts by its own `section_order` (after them only when its key is at
/// least the largest of theirs), a `ReorderChildren` of the new section sees
/// them, and a cascade of it takes them along. The
/// adopted subtrees also come under the end-of-delta depth check.
///
/// Each adopted row is rewritten with `last_version = new_version` and gets a
/// `SetSection`. Without it a snapshot-mode reconnector whose `have_version`
/// lies between the orphaning and the adoption would never receive the row:
/// the coalesced diff emits only rows created or modified after the
/// reconnector's cut, plus the other children of a parent with such a row.
/// The parent born in this delta is itself new, but that touches only its own
/// parent's list; its adopted children, all older than the cut, may hold no
/// such row, so without the bump none of them would be sent.
/// The row's parent and key do not change, so no order cache is touched. A
/// row this delta already wrote carries `new_version`, and the delta's last
/// `SetSection` for it already holds the current row, so it emits nothing
/// more.
///
/// Candidates come from the [`AdoptionIndex`], which the delta's first birth
/// builds from `commands` (the delta's structural commands) and which names
/// every row naming an id the delta can birth. They are re-checked against
/// `state.sections`. The root `0` adopts nothing: its list always exists, so a
/// row naming it is listed already. (A section with id `0` shares the root's
/// list key, and adopting through it would only re-stamp every top-level
/// row.)
fn adopt_orphans<R: ValueRows, H: EngineHooks<R>>(
    state: &mut ObjectStateOf<R>,
    new_version: u64,
    section_id: u64,
    commands: &[StructCommand],
    plan: &mut ChildPlan<'_, H>,
    sink: &mut impl PlanSink,
    stats: &mut ApplyDeltaStats,
) {
    if section_id == 0 {
        return;
    }
    // A plan whose hooks keep a persistent index (the structural cache's)
    // reads every row naming the newborn from it and builds no adoption
    // index.
    let mut indexed: Vec<u64> = Vec::new();
    let hooked = plan.hooks.rows_naming(section_id, &mut indexed);
    if !hooked && plan.adoption.is_none() {
        stats.adoption_index_builds = stats.adoption_index_builds.saturating_add(1);
        // Each id the build records is charged at its record
        // (`Projection::AdoptionEntries`, see `AdoptionIndex`'s `# Charge`).
        plan.adoption = Some(pass_timing::time(Pass::AdoptionIndexBuild, || {
            AdoptionIndex::build(&state.sections, commands, plan.meter)
        }));
    }
    let candidates: &[u64] = if hooked {
        &indexed
    } else {
        match plan
            .adoption
            .as_ref()
            .and_then(|index| index.rows.get(&section_id))
        {
            Some(rows) => rows,
            None => return,
        }
    };
    // Each orphan is projected as the row its adoption emits (its
    // `SetSection`) before the list collecting it grows, so a bounded plan
    // asks its budget before this scratch, and the list growth the adoption
    // makes below, are allocated; the emission retires it. A repeated
    // candidate, or a row this delta already wrote (which emits nothing
    // more), over-projects, the safe direction. This is separate from the
    // index's own ids above, which were charged as `adoption_entries` when
    // recorded.
    let mut orphans: Vec<(u32, u64)> = Vec::new();
    for &cid in candidates {
        let Some(row) = state
            .sections
            .get(&cid)
            .filter(|row| row.parent_id == section_id && cid != section_id)
        else {
            continue;
        };
        plan.project(Projection::Rows(1));
        orphans.push((row.section_order, cid));
    }
    // The index can record one row more than once (each write naming the
    // candidate records it), so equal pairs are collapsed after the sort.
    orphans.sort_unstable();
    orphans.dedup();
    for (section_order, cid) in orphans {
        insert_child_into_parent(state, plan, stats, section_id, cid, section_order);
        if state
            .sections
            .get(&cid)
            .is_none_or(|row| row.last_version == new_version)
        {
            continue;
        }
        write::update_section(state, &mut plan.hooks, cid, |row| {
            row.last_version = new_version;
        });
        let Some(row) = state.sections.get(&cid) else {
            continue;
        };
        // A re-stamp of an existing row: its key is unchanged.
        sink.push_with(|| StorageMutation::SetSection {
            section: row.clone(),
            previous: Some(SectionKey {
                parent_id: row.parent_id,
                name: row.name.clone(),
            }),
        });
    }
}

// ── CommandDeleteSection ──────────────────────────────────────────────────────

fn apply_delete_section<R: ValueRows, H: EngineHooks<R>>(
    state: &mut ObjectStateOf<R>,
    new_version: u64,
    section_id: u64,
    plan: &mut ChildPlan<'_, H>,
    sink: &mut impl PlanSink,
    stats: &mut ApplyDeltaStats,
) {
    if !state.sections.contains_key(&section_id) {
        return;
    }
    let parent_id = state.sections[&section_id].parent_id;
    cascade_delete_section(
        state,
        new_version,
        section_id,
        Some(parent_id),
        plan,
        sink,
        stats,
    );
}

// ── CommandMoveSection ────────────────────────────────────────────────────────

fn apply_move_section<R: ValueRows, H: EngineHooks<R>>(
    state: &mut ObjectStateOf<R>,
    new_version: u64,
    cmd: &CommandMoveSection,
    plan: &mut ChildPlan<'_, H>,
    sink: &mut impl PlanSink,
    stats: &mut ApplyDeltaStats,
) -> Result<(), DeltaError> {
    let section_id = cmd.section_id;
    let old_parent_id = cmd.old_parent_id;
    let new_parent_id = cmd.new_parent_id;
    let new_name = cmd.new_name.clone();

    if !state.sections.contains_key(&section_id) {
        return Ok(());
    }

    // Past the rename arm, the section's *stored* parent decides every
    // outcome: it is the list the section actually sits in, and the parent a
    // client that has seen every earlier commit also holds it under.
    // `cmd.old_parent_id` is client-supplied and can be stale — two clients
    // moving the same section concurrently both carry the parent they saw —
    // so beyond picking the rename arm it is used only to clean its own list
    // as well (`unlink_carried_parent`), which keeps every list the carried
    // id names as clean as when the carried id alone drove the bookkeeping.
    //
    // The same stored `(parent_id, name)` pair is the `SetSection::previous`
    // pre-image on every arm that writes this row — edge 1, the rename in
    // place, and edge 6: nothing between here and any of those emissions
    // writes this row (edge 5's cascade removes only the non-ancestor
    // conflict's subtree, and a cascade that removed this row instead fails
    // the arm's re-read before emitting).
    let (stored_parent_id, stored_name) = state
        .sections
        .get(&section_id)
        .map(|s| (s.parent_id, s.name.clone()))
        .ok_or(DeltaError::InconsistentState {
            section_id,
            operation: "apply_move_section (read stored parent/name)",
        })?;

    // Edge case 1: same parent as the command carries it → rename only, under
    // the stored parent, with no list write (SDD §9.3.1).
    //
    // No conflict check here, deliberately. PRD §2.3.4 and SDD §9.3.1
    // describe rename as unconditional, and a previously-accepted delta
    // starting to return INVALID_ARGUMENT would be a client-visible wire
    // contract change. A rename onto an occupied `(parent_id, name)` still
    // succeeds and still produces duplicate siblings — what b.ot3 changes is
    // that the *resolution* of such a duplicate is deterministic, see
    // [`find_child_by_name`].
    if old_parent_id == new_parent_id {
        let order_before = effective_order(&state.sections, section_id);
        let updated = write::update_section(state, &mut plan.hooks, section_id, |sec| {
            sec.name = new_name;
            sec.last_version = new_version;
            sec.clone()
        })
        .ok_or(DeltaError::InconsistentState {
            section_id,
            operation: "apply_move_section (rename)",
        })?;
        plan.name_removed(stored_parent_id, &stored_name, section_id);
        plan.name_inserted(stored_parent_id, &updated.name, section_id);
        sink.push_with(|| StorageMutation::SetSection {
            section: updated,
            previous: Some(SectionKey {
                parent_id: stored_parent_id,
                name: stored_name,
            }),
        });
        // A rename writes `name` and `last_version` only, so this fires
        // nothing — by computation rather than by convention, like every
        // other row-writing site.
        if effective_order(&state.sections, section_id) != order_before {
            plan.invalidate_order_listeners(section_id, None, stats);
        }
        return Ok(());
    }

    // The carried parent, when it names a list other than the stored one.
    let carried_parent = (old_parent_id != stored_parent_id).then_some(old_parent_id);

    // Edges 2–4 delete the section. It is unlinked from its stored parent —
    // the list that actually holds it — and from a differing carried parent;
    // unlinking from the carried parent alone would leave the stored parent
    // listing a section with no row (a ghost listing).

    // Edge case 2: newParentId doesn't exist → delete the section (SDD §9.3.2).
    if new_parent_id != 0 && !state.sections.contains_key(&new_parent_id) {
        unlink_carried_parent(state, plan, stats, carried_parent, section_id);
        cascade_delete_section(
            state,
            new_version,
            section_id,
            Some(stored_parent_id),
            plan,
            sink,
            stats,
        );
        return Ok(());
    }

    // Edge case 3: cycle — newParentId is a descendant of sectionId → delete (SDD §9.3.3).
    if is_descendant_of(
        state,
        new_parent_id,
        section_id,
        &mut plan.descendant_scratch,
    ) {
        unlink_carried_parent(state, plan, stats, carried_parent, section_id);
        cascade_delete_section(
            state,
            new_version,
            section_id,
            Some(stored_parent_id),
            plan,
            sink,
            stats,
        );
        return Ok(());
    }

    // Edges 4 & 5: name conflict at destination.
    let conflict_id = find_child_by_name(
        state,
        plan,
        stats,
        new_parent_id,
        &new_name,
        Some(section_id),
    );
    if let Some(cid) = conflict_id {
        if is_descendant_of(state, section_id, cid, &mut plan.descendant_scratch) {
            // Edge case 4: conflicting section is an ancestor → delete the moved section.
            unlink_carried_parent(state, plan, stats, carried_parent, section_id);
            cascade_delete_section(
                state,
                new_version,
                section_id,
                Some(stored_parent_id),
                plan,
                sink,
                stats,
            );
            return Ok(());
        }
        // Edge case 5: non-ancestor conflict → delete the conflicting section, proceed.
        // The conflict was found by its own row's parent (`find_child_by_name`
        // re-verifies `parent_id`), so `new_parent_id` is its stored parent.
        cascade_delete_section(
            state,
            new_version,
            cid,
            Some(new_parent_id),
            plan,
            sink,
            stats,
        );
    }

    // Destination is already the stored parent — reachable only with a stale
    // carried parent, e.g. a second client repeating a move another client's
    // commit already made. Every client that applied that commit holds the
    // section under this parent already and leaves it in place, so it is a
    // rename: the key and the list position stay. The insert keeps the
    // section listed — a no-op when it is, and a re-listing by its existing
    // key for a legacy row its parent's list lost.
    if new_parent_id == stored_parent_id {
        unlink_carried_parent(state, plan, stats, carried_parent, section_id);
        let order_before = effective_order(&state.sections, section_id);
        let updated = write::update_section(state, &mut plan.hooks, section_id, |sec| {
            sec.name = new_name;
            sec.last_version = new_version;
            sec.clone()
        })
        .ok_or(DeltaError::InconsistentState {
            section_id,
            operation: "apply_move_section (rename in place)",
        })?;
        let section_order = updated.section_order;
        plan.name_removed(stored_parent_id, &stored_name, section_id);
        plan.name_inserted(stored_parent_id, &updated.name, section_id);
        sink.push_with(|| StorageMutation::SetSection {
            section: updated,
            previous: Some(SectionKey {
                parent_id: stored_parent_id,
                name: stored_name,
            }),
        });
        // Writes `name` and `last_version` only, so this fires nothing — by
        // computation, like every other row-writing site.
        if effective_order(&state.sections, section_id) != order_before {
            plan.invalidate_order_listeners(section_id, None, stats);
        }
        insert_child_into_parent(state, plan, stats, new_parent_id, section_id, section_order);
        return Ok(());
    }

    // Normal move (edge case 6): the parent changes. Clients append a section
    // moved under a new parent, exactly as they append a created one (see
    // `apply_create_section`), so the moved section takes the destination's
    // append key rather than carrying its old key across — the old key
    // compares against a different sibling set and would land it mid-list.
    //
    // Every list that can hold the section is cleared first: the stored
    // parent's, a differing carried parent's, and the destination's (a legacy
    // stale listing there would otherwise keep its old position and turn the
    // insert below into a no-op). So the key is computed over the
    // destination's other children only — after the edge-5 cascade, so the
    // deleted conflict does not count either — and the section lands last.
    remove_child_from_parent(state, plan, stats, stored_parent_id, section_id);
    unlink_carried_parent(state, plan, stats, carried_parent, section_id);
    remove_child_from_parent(state, plan, stats, new_parent_id, section_id);
    let section_order = append_order(state, plan, stats, new_parent_id);
    let order_before = effective_order(&state.sections, section_id);
    write::update_section(state, &mut plan.hooks, section_id, |sec| {
        sec.parent_id = new_parent_id;
        sec.name = new_name;
        sec.section_order = section_order;
        sec.last_version = new_version;
    })
    .ok_or(DeltaError::InconsistentState {
        section_id,
        operation: "apply_move_section (move)",
    })?;
    // Fires whenever the append key differs from the old key. The removals
    // above took the section out of every cache this command names, so a
    // cache that still lists it is a legacy stale listing under some other
    // parent, which mirrors the old key and is dropped. Nothing is kept, and
    // the destination's cache — which no longer lists the section — survives
    // for the insert below.
    if effective_order(&state.sections, section_id) != order_before {
        plan.invalidate_order_listeners(section_id, None, stats);
    }
    let updated = state
        .sections
        .get(&section_id)
        .ok_or(DeltaError::InconsistentState {
            section_id,
            operation: "apply_move_section (move: clone updated)",
        })?
        .clone();
    plan.name_removed(stored_parent_id, &stored_name, section_id);
    plan.name_inserted(new_parent_id, &updated.name, section_id);
    sink.push_with(|| StorageMutation::SetSection {
        section: updated,
        previous: Some(SectionKey {
            parent_id: stored_parent_id,
            name: stored_name,
        }),
    });
    insert_child_into_parent(state, plan, stats, new_parent_id, section_id, section_order);
    Ok(())
}

/// A `MoveSection`'s carried `old_parent_id` only cleans its own list: when it
/// differs from the section's stored parent — the parent the move otherwise
/// works from — the caller passes it as `Some`, and the section is taken out
/// of that list too. `None` (the carried parent names the stored one) needs
/// nothing here; the caller decides what happens to the stored parent's list.
fn unlink_carried_parent<R: ValueRows, H: EngineHooks<R>>(
    state: &mut ObjectStateOf<R>,
    plan: &mut ChildPlan<'_, H>,
    stats: &mut ApplyDeltaStats,
    carried_parent: Option<u64>,
    section_id: u64,
) {
    if let Some(parent_id) = carried_parent {
        remove_child_from_parent(state, plan, stats, parent_id, section_id);
    }
}

// ── CommandReorderChildren ────────────────────────────────────────────────────

/// Apply a `CommandReorderChildren` (SDD §9.1.4e) with USD's list-ordering
/// rule, `SdfApplyListOrdering` (OpenUSD `pxr/usd/sdf/listOp.cpp:674-726`,
/// `:897-921` @ `ee47c679`), applied **per child list**.
///
/// A parent's one stored list interleaves up to three of USD's child lists
/// (a prim's child prims, its properties, its variant sets; see
/// [`child_list`]). A client reorders one of them at a time — Kit names only
/// one child kind per command — so each child list the command names is
/// reordered among the positions its members occupy, and every other child
/// list keeps its order and positions. Within a child list, the named
/// members come out in the command's order, each followed by the unnamed
/// members that followed it, and unnamed members before the first named one
/// stay in front ([`apply_list_ordering`]). A command naming every member of
/// a child list leaves that list exactly in the command's order.
///
/// The command never fails. Ids that are not this parent's listed children
/// are **ignored**, not rejected: an id with no row (unknown, or deleted by a
/// peer's commit), a row naming another parent (moved away, or never a
/// child), a legacy row naming this parent that its list lost (it stays
/// unlisted), and every repeat after the first. Ignoring them is what USD
/// does with an ordering that names absent items, and it is what lets a
/// reorder authored against a slightly older view still land. None of them
/// can enter a list, so the hazards a rejection used to guard stay closed: no
/// ghost id is listed, no section is listed under two parents, and no id is
/// stored twice. A command that names nothing usable — an empty list, or one
/// whose every id is ignored — changes nothing, as USD returns the list
/// unchanged.
///
/// Keys: when the parent's stored keys strictly increase along its list (the
/// normal case; see [`reorder_layout`]), each position keeps the key it held,
/// so a child takes the key of the position it lands in and a child that did
/// not move keeps its own; the parent's key set, and its next append key, are
/// unchanged, and for a command naming a whole child list that is the key
/// Kit assigns. Otherwise (equal keys, or keys out of step with the list)
/// [`reorder_layout`] re-keys the whole list `0..n-1`, which changes the key
/// set and can rewrite children that did not move, in any child list. Each
/// child whose key changes — in the first case exactly the children that
/// moved — has its row rewritten with `last_version` bumped and a
/// `SetSection` emitted. That is what makes a reorder visible to every reader
/// that does not replay this delta: a snapshot-mode reconnector, whose
/// coalesced diff re-emits rows modified since its cut, and any reader that
/// rebuilds a list from the stored keys (checkpoint restore, a rollback-mode
/// cold start). The layout and keys come from [`reorder_layout`].
fn apply_reorder_children<R: ValueRows, H: EngineHooks<R>>(
    state: &mut ObjectStateOf<R>,
    new_version: u64,
    cmd: &CommandReorderChildren,
    plan: &mut ChildPlan<'_, H>,
    sink: &mut impl PlanSink,
    stats: &mut ApplyDeltaStats,
) {
    let parent_id = cmd.section_id;
    // childrenListId is present in the schema but ignored by the server (SDD
    // §9.1.4e): the child list each id belongs to is derived from the section
    // types instead (`child_list`).
    let named: &[u64] = &cmd.children_list;
    if named.is_empty() {
        return;
    }

    // The working list is the parent's own children, each once: an entry
    // with no row (a ghost), a stale listing of a section whose row names
    // another parent, and a repeated occurrence all leave the list. All three
    // are stored shapes this engine does not produce. Dropping them is what
    // lets a command naming every listed child of a child list leave that
    // list exactly in the command's order, keeps the ordering below from ever
    // seeing a repeat (USD's own lookup would keep the *last* occurrence), and
    // keeps the re-key below from ever writing a row another parent owns.
    let current = state
        .children
        .get(&parent_id)
        .map(Vec::as_slice)
        .unwrap_or(&[]);
    let mut kept = std::collections::HashSet::<u64>::with_capacity(current.len());
    let working: Vec<u64> = current
        .iter()
        .copied()
        .filter(|cid| {
            state
                .sections
                .get(cid)
                .is_some_and(|s| s.parent_id == parent_id)
                && kept.insert(*cid)
        })
        .collect();

    // The effective order: the first occurrence of each named id that is in
    // the working list. Everything else the command names is ignored (see the
    // fn doc). When nothing is left the command is a no-op and touches no
    // state — not even the clean-up below.
    let mut ordered = std::collections::HashSet::<u64>::with_capacity(named.len());
    let order: Vec<u64> = named
        .iter()
        .copied()
        .filter(|cid| kept.contains(cid) && ordered.insert(*cid))
        .collect();
    if order.is_empty() {
        return;
    }

    // The parent's cache is rebuilt *populated* below, so an id that left the
    // list must lose its `(child, parent)` pair: a lingering pair would spend
    // a rebuild of this live list on the next order-side write to an id that
    // is no longer in it. (An id that only lost a repeat is still listed and
    // keeps its pair.) `state` and `plan` are disjoint `&mut` parameters, so
    // the borrow driving the loop does not conflict with the de-registration.
    for &cid in current.iter().filter(|cid| !kept.contains(*cid)) {
        plan.deregister_listing(cid, parent_id);
    }
    // `None` for the root: no row, no type. (A parent with no row keeps no
    // children list, so a reorder naming one returned above with an empty
    // working list and changes nothing.)
    let parent_type = state.sections.get(&parent_id).map(|s| s.section_type);
    let layout = reorder_layout(&working, &state.sections, parent_type, &order);

    // Re-key before installing the list. Every entry is one of this parent's
    // children with a row of its own, so the `else` arm is unreachable and
    // only keeps the loop total.
    for &(cid, key) in &layout {
        if state
            .sections
            .get(&cid)
            .is_none_or(|sec| sec.section_order == key)
        {
            continue;
        }
        write::update_section(state, &mut plan.hooks, cid, |sec| {
            sec.section_order = key;
            sec.last_version = new_version;
        });
        let Some(sec) = state.sections.get(&cid) else {
            continue;
        };
        // A re-key writes `section_order` and `last_version` only, so the
        // row's `(parent_id, name)` — its pre-image — is what it holds now.
        sink.push_with(|| StorageMutation::SetSection {
            section: sec.clone(),
            previous: Some(SectionKey {
                parent_id: sec.parent_id,
                name: sec.name.clone(),
            }),
        });
        // Every cache mirroring the old key is now false. This parent's is
        // kept only because the build below replaces it before anything
        // reads it; any other parent's (a legacy stale listing elsewhere) is
        // dropped.
        plan.invalidate_order_listeners(cid, Some(parent_id), stats);
    }

    // Re-derived through the one builder rather than patched, since the
    // positions, the membership and the keys may all have moved; the build
    // re-registers every member. Its per-element lookups are the same O(C)
    // the filter above already paid.
    write::set_list(
        state,
        &mut plan.hooks,
        parent_id,
        layout.into_iter().map(|(cid, _)| cid).collect(),
    );
    let cache = plan.build_cache(&state.children, &state.sections, parent_id, stats);
    plan.restore_cache(parent_id, cache);
    plan.mark_dirty(parent_id);
}

/// The list a `ReorderChildren` leaves under its parent, each entry paired
/// with the stored key it must carry.
///
/// `working` is the parent's list as [`apply_reorder_children`] filtered it —
/// the parent's own children, each with a row, each once — `parent_type` the
/// parent row's `section_type` (`None` for the root, which has no row), and
/// `order` the command's effective order: ids of `working`, each once.
///
/// **Placement.** `working` is split into child lists by [`child_list`]. For
/// each child list holding at least one id of `order`, its members (in
/// `working` order) are reordered by `order` restricted to that child list
/// ([`apply_list_ordering`]), and the result is written back into the
/// positions the child list occupies. A child list `order` does not touch is
/// left exactly as it was, and no child list's positions change, so a reorder
/// of one child list never moves another's members.
///
/// **Keys.** When `working` is strictly increasing in stored key — every list
/// this engine builds, unless a client sent duplicate explicit orders or a
/// parent's key space saturated at `u32::MAX` — each position keeps its key:
/// the entry placed at position `i` takes the key position `i` held. An entry
/// that did not move keeps its own key, the parent's key set is unchanged (so
/// is its next append key), and a command naming a whole child list hands
/// that child list's sorted keys out in the command's order, which is the key
/// Kit assigns. The list stays strictly increasing.
///
/// Otherwise — equal keys, or keys out of step with the list (a reorder
/// stored before reorders re-keyed) — position keys would carry the equal or
/// out-of-step keys over, so the **whole** result is re-keyed `0..n-1` in list
/// order, which leaves it strictly increasing. Every entry is one of the parent's own
/// children, so either arm writes no row another parent owns.
fn reorder_layout(
    working: &[u64],
    sections: &std::collections::HashMap<u64, Section>,
    parent_type: Option<u8>,
    order: &[u64],
) -> Vec<(u64, u32)> {
    let keys: Vec<u32> = working
        .iter()
        .map(|&cid| effective_order(sections, cid))
        .collect();
    let increasing = keys.windows(2).all(|w| w[0] < w[1]);
    let list_of = |cid: u64| {
        let child_type = sections.get(&cid).map_or(SPEC_UNKNOWN, |s| s.section_type);
        child_list(parent_type, child_type)
    };

    // Positions of each child list in `working`, and `order` restricted to
    // each. A `BTreeMap` keeps the walk deterministic; a parent has at most
    // three child lists plus the fallback, so its size is a constant.
    let mut positions: std::collections::BTreeMap<Option<u8>, Vec<usize>> =
        std::collections::BTreeMap::new();
    for (position, &cid) in working.iter().enumerate() {
        positions.entry(list_of(cid)).or_default().push(position);
    }
    let mut orders: std::collections::BTreeMap<Option<u8>, Vec<u64>> =
        std::collections::BTreeMap::new();
    for &cid in order {
        orders.entry(list_of(cid)).or_default().push(cid);
    }

    let mut placed = working.to_vec();
    for (list, list_order) in &orders {
        let Some(list_positions) = positions.get(list) else {
            continue;
        };
        let members: Vec<u64> = list_positions.iter().map(|&p| working[p]).collect();
        let reordered = apply_list_ordering(&members, list_order);
        for (&position, cid) in list_positions.iter().zip(reordered) {
            placed[position] = cid;
        }
    }

    placed
        .into_iter()
        .enumerate()
        .map(|(position, cid)| {
            let key = if increasing {
                keys[position]
            } else {
                u32::try_from(position).unwrap_or(u32::MAX)
            };
            (cid, key)
        })
        .collect()
}

// `SdfSpecType` values, as clients write them into `section_type`: the Kit
// client sends the spec type cast to a byte, and the C++ client library reads
// the byte back as an `SdfSpecType`. OpenUSD `pxr/usd/sdf/types.h:71-89` @
// `ee47c679`.
const SPEC_UNKNOWN: u8 = 0;
const SPEC_ATTRIBUTE: u8 = 1;
const SPEC_CONNECTION: u8 = 2;
const SPEC_EXPRESSION: u8 = 3;
const SPEC_MAPPER: u8 = 4;
const SPEC_MAPPER_ARG: u8 = 5;
const SPEC_PRIM: u8 = 6;
const SPEC_PSEUDO_ROOT: u8 = 7;
const SPEC_RELATIONSHIP: u8 = 8;
const SPEC_RELATIONSHIP_TARGET: u8 = 9;
const SPEC_VARIANT: u8 = 10;
const SPEC_VARIANT_SET: u8 = 11;

/// Which of its parent's child lists a child belongs to, from the two
/// sections' `SdfSpecType`s (`section_type`). `None` is the **fallback** list.
///
/// A USD spec keeps its children in separate lists — a prim's child prims
/// (0), its properties (1) and its variant sets (2) — and a client reorders
/// one list at a time. The table is the C++ client library's child-list
/// table; the Kit client's child-index mapping agrees on every pair except
/// Variant → VariantSet, which the Kit client treats as invalid and never
/// produces (both tables at the revisions recorded in
/// `docs-internal/cross-repo-debugging.md` § Sibling order and move rules).
///
/// Every other pair — the root parent (no row, so `parent_type` is `None`),
/// an Unknown type, or a pair USD does not allow — falls into
/// the one fallback list. Data that carries no meaningful types therefore
/// behaves as a single list.
fn child_list(parent_type: Option<u8>, child_type: u8) -> Option<u8> {
    let parent_type = parent_type?;
    match (parent_type, child_type) {
        (SPEC_PSEUDO_ROOT | SPEC_PRIM | SPEC_VARIANT, SPEC_PRIM) => Some(0),
        (SPEC_PRIM | SPEC_VARIANT, SPEC_ATTRIBUTE | SPEC_RELATIONSHIP) => Some(1),
        (SPEC_PRIM | SPEC_VARIANT, SPEC_VARIANT_SET) => Some(2),
        (SPEC_VARIANT_SET, SPEC_VARIANT) => Some(0),
        (SPEC_ATTRIBUTE, SPEC_CONNECTION) => Some(0),
        (SPEC_ATTRIBUTE, SPEC_EXPRESSION) => Some(1),
        (SPEC_ATTRIBUTE, SPEC_MAPPER) => Some(2),
        (SPEC_RELATIONSHIP, SPEC_RELATIONSHIP_TARGET) => Some(0),
        (SPEC_RELATIONSHIP, SPEC_EXPRESSION) => Some(1),
        (SPEC_RELATIONSHIP_TARGET, SPEC_ATTRIBUTE) => Some(0),
        (SPEC_MAPPER, SPEC_MAPPER_ARG) => Some(0),
        _ => None,
    }
}

/// `SdfApplyListOrdering(list, order)` (OpenUSD `pxr/usd/sdf/listOp.cpp`
/// `:674-726` and `:897-921` @ `ee47c679`), in linear time.
///
/// USD walks `order`; for each id present in the list it splices out that
/// id together with the run of following items that `order` does not name,
/// up to the next named item still in the list, and appends the run to the
/// result. Items left over — those before the first named item — go in
/// front, in their order. Ids of `order` that are not in the list, and
/// repeats in `order`, are skipped.
///
/// Here the runs are cut from the original list in one pass instead: the
/// front group is the unnamed items before the first named one; each named
/// item's run is itself plus the unnamed items after it, up to the next named
/// item; the output is the front group, then the runs in `order`. That equals
/// the splice walk on a list without repeats, because a run spliced out
/// earlier is a contiguous block ending right before the next named item
/// still present — so when a later walk crosses its old position, the next
/// item it meets is that named item, and it picks up nothing extra.
/// [`apply_reorder_children`] guarantees `list` holds no repeat.
///
/// For example, `[0, 2, 4, 1, 5, 3]` ordered by `[1, 2, 3]` gives
/// `[0, 1, 5, 2, 4, 3]`, and `[A, B, C, D]` ordered by `[D, B]` gives
/// `[A, D, B, C]`.
fn apply_list_ordering(list: &[u64], order: &[u64]) -> Vec<u64> {
    let named: std::collections::HashSet<u64> = order.iter().copied().collect();
    let mut front = Vec::new();
    let mut runs: std::collections::HashMap<u64, Vec<u64>> = std::collections::HashMap::new();
    let mut owner: Option<u64> = None;
    for &id in list {
        if named.contains(&id) {
            owner = Some(id);
        }
        match owner {
            Some(head) => runs.entry(head).or_default().push(id),
            None => front.push(id),
        }
    }
    let mut out = front;
    for id in order {
        if let Some(run) = runs.remove(id) {
            out.extend(run);
        }
    }
    out
}

// ── Field application ─────────────────────────────────────────────────────────

fn apply_set_field<R: ValueRows, H: EngineHooks<R>>(
    state: &mut ObjectStateOf<R>,
    new_version: u64,
    sf: &OodfSetField,
    hooks: &mut H,
    sink: &mut impl PlanSink,
) -> Result<(), DeltaError> {
    let section_id = sf.section_id;
    let key_name = sf.key_name.clone();
    let delta_set_order = sf.set_order;

    // b.qw6 item 2: `set_order` is capped at the Lua f64 integer-precision
    // ceiling (2^53 - 1). The Lua fast-path script reads and writes the
    // running `next_set_order` cursor through Lua `number` (f64), which
    // silently rounds integers past 2^53. To keep both backends failing
    // at the same boundary — AC #2 of b.qw6 — we enforce the same
    // ceiling here in pure Rust. Pre-fix this check used
    // `checked_add(...)` on u64 and only tripped at u64::MAX, so a
    // `next_set_order` sequence between 2^53 and u64::MAX committed
    // cleanly on the slow path while the fast path silently corrupted.
    //
    // The matching ceiling lives in `update_layer_fast.lua` and is
    // additionally bounded at delta-validation time
    // (`MAX_DELTA_SET_ORDER` in `validate_delta_bounds`).
    //
    // Both `stored_set_order` and `stored_set_order + 1` (used to advance
    // `next_set_order` per SDD §9.1.5b) must fit under 2^53 - 1. With
    // `delta_set_order` and `next_set_order` each individually bounded
    // at 2^53 - 1, the sum is bounded by 2^54 - 2 and never overflows
    // u64 — so no checked-arithmetic guard is needed beyond the
    // ceiling test.
    const SET_ORDER_CEILING: u64 = (1u64 << 53) - 1;
    if state.next_set_order > SET_ORDER_CEILING - delta_set_order.min(SET_ORDER_CEILING)
        || delta_set_order > SET_ORDER_CEILING
        || state.next_set_order > SET_ORDER_CEILING
    {
        return Err(DeltaError::SetOrderOverflow {
            next: state.next_set_order,
            delta: delta_set_order,
        });
    }

    // Stored setOrder = nextSetOrder + delta.setOrder (SDD §9.1.5a).
    // After the ceiling check above, the sum is guaranteed <= 2^53 - 1
    // and the +1 below is guaranteed <= 2^53, both well within u64.
    let stored_set_order = state.next_set_order + delta_set_order;
    // Belt-and-braces: re-verify stored_set_order is at the ceiling, so
    // `stored_set_order + 1` is at most 2^53. Treat the boundary case
    // as overflow because the NEXT call would not be able to advance.
    if stored_set_order >= SET_ORDER_CEILING {
        return Err(DeltaError::SetOrderOverflow {
            next: state.next_set_order,
            delta: delta_set_order,
        });
    }

    // nextSetOrder = max(nextSetOrder, stored + 1) (SDD §9.1.5b).
    let advanced = stored_set_order + 1;
    let next_set_order = state.next_set_order.max(advanced);
    write::set_next_set_order(state, hooks, next_set_order);

    let value =
        field_value_from_oodf(&sf.value_or_ext_hash, sf.ext_value_size, sf.ext_value_index)?;

    let field = Field {
        section_id,
        key_name,
        value,
        set_order: stored_set_order,
        authoring_set_order: delta_set_order,
        version: new_version,
    };
    // b.ot3 part A: the clone is INVERTED into the sink closure. The
    // collecting arm still pays exactly two copies of the field (one for
    // `state.fields`, one for the mutation), as it did before; the
    // discarding arm never clones the `FieldValue::Inline(Vec<u8>)` payload
    // at all, which is the dominant discarded allocation on the CreateLayer
    // and checkpoint paths once children writes are coalesced. A compact
    // store keeps only the row's key and external hash of the field it is
    // handed.
    sink.push_with(|| StorageMutation::SetField {
        field: field.clone(),
    });
    write::insert_field(state, hooks, field);
    Ok(())
}

// ── Time sample application ───────────────────────────────────────────────────

fn apply_time_sample<R: ValueRows, H: EngineHooks<R>>(
    state: &mut ObjectStateOf<R>,
    new_version: u64,
    ts: &OodfTimeSample,
    hooks: &mut H,
    sink: &mut impl PlanSink,
) -> Result<(), DeltaError> {
    let section_id = ts.section_id;
    let time = OrderedFloat(ts.time);
    let value =
        field_value_from_oodf(&ts.value_or_ext_hash, ts.ext_value_size, ts.ext_value_index)?;

    let sample = TimeSample {
        section_id,
        time,
        value,
        version: new_version,
    };
    // Clone inverted into the sink closure — see `apply_set_field`.
    sink.push_with(|| StorageMutation::SetTimeSample {
        sample: sample.clone(),
    });
    write::insert_sample(state, hooks, sample);
    Ok(())
}

// ── Cascade delete ────────────────────────────────────────────────────────────

/// Delete `section_id` and all its descendants (DFS), generating tombstones.
///
/// `explicit_parent` is the parent to remove the section from — applies to
/// the root of the deletion; descendants are handled via children-list deletions.
/// Every caller passes the root's *stored* parent (its row's `parent_id`), the
/// list that actually holds it; passing any other id would leave that list
/// holding a ghost.
///
/// Ghost ids (entries in `state.children` with no matching `state.sections`
/// row) are still pruned from the children map but do *not* emit tombstones —
/// there's no section to delete and no `first_version` to record. Since
/// `apply_reorder_children` lists only rows parented to its parent, and with
/// the strict H-dom-3 `find_child_by_name`, the post-write invariant holds that
/// `state.children[p] ⊆ state.sections`, so this branch should only fire for
/// stored state that predates the bundled fixes. That invariant is about
/// cross-map *membership* only — it does not make `(parent_id, name)`
/// unique, which same-parent `MoveSection` rename can still violate; which
/// duplicate a cascade targets is settled by [`find_child_by_name`].
///
/// b.ot3 part E: the deleted sections' fields and time samples are **not**
/// erased here. Their ids are accumulated on the plan and
/// [`purge_cascaded_rows`] erases them in one purge per delta at the close
/// of the structural phase, instead of one `retain` over both maps per
/// deleted section. Observationally identical inside the delta — no structural
/// command reads `state.fields` or `state.time_samples` — and strictly
/// before step 4's `SetField`s, which the `ext_rows_written` fold requires.
fn cascade_delete_section<R: ValueRows, H: EngineHooks<R>>(
    state: &mut ObjectStateOf<R>,
    new_version: u64,
    section_id: u64,
    explicit_parent: Option<u64>,
    plan: &mut ChildPlan<'_, H>,
    sink: &mut impl PlanSink,
    stats: &mut ApplyDeltaStats,
) {
    let ids_to_delete = collect_subtree(state, section_id, plan);

    // Remove the root from its parent's children list.
    if let Some(parent_id) = explicit_parent {
        remove_child_from_parent(state, plan, stats, parent_id, section_id);
    }

    for sid in ids_to_delete {
        // Taking the row out up front hands over the owned value, so its
        // `(parent_id, name)` moves into the `DeleteSection` without a clone.
        // Everything below reads the row only through the owned `section`,
        // except the closing `effective_order` check, which relies on the row
        // already being gone (the ghost order `0`) — so the removal must stay
        // ahead of that check.
        let Some(section) = write::remove_section(state, &mut plan.hooks, sid) else {
            // Ghost id — clean stale children entry but skip mutations. Its
            // fields are deliberately NOT accumulated for the purge: there
            // is no section row, and pre-b.ot3 they were not purged either.
            write::remove_list(state, &mut plan.hooks, sid);
            plan.invalidate_cache(sid);
            continue;
        };
        let first_ver = section.first_version;
        let section_order = section.section_order;
        plan.name_removed(section.parent_id, &section.name, sid);

        sink.push_with(|| StorageMutation::DeleteSection {
            section_id: sid,
            parent_id: section.parent_id,
            name: section.name,
        });
        sink.push_with(|| StorageMutation::CreateTombstone {
            section_id: sid,
            first_version: first_ver,
            last_version: new_version,
        });
        stats.children_writes_emitted = stats.children_writes_emitted.saturating_add(1);
        sink.push_with(|| StorageMutation::DeleteChildrenList { section_id: sid });
        plan.deleted_lists.insert(sid);

        // Fields (SDD §9.2.6) and time samples (SDD §9.2.7) for this section
        // are erased by the deferred purge, one per delta — see the fn doc
        // above.
        plan.purge_ids.insert(sid);

        let order_before = section_order;
        write::remove_list(state, &mut plan.hooks, sid);
        // `invalidate_cache` clears `sid`-as-a-parent. The row this loop just
        // dropped is the other side of the coin: every parent still listing
        // `sid` cached its real `section_order`, and from here on `sid` is a
        // ghost, whose effective order is `0`. (The parent this cascade
        // removed the root from no longer lists it, so it does not match; a
        // *stale* listing under some other parent does.) Nothing is kept — the
        // row is gone, so no following statement repairs any parent's entry
        // for it.
        plan.invalidate_cache(sid);
        if effective_order(&state.sections, sid) != order_before {
            plan.invalidate_order_listeners(sid, None, stats);
        }
    }
}

/// Collect `root` and all descendants via DFS on the children map.
///
/// In a bounded plan each id is projected as three rows (the cascade's
/// `DeleteSection`, `CreateTombstone` and `DeleteChildrenList`) before the
/// walk's collections grow to hold it: the root before the walk starts, and
/// each list's children before they are pushed. A ghost id, or an id reached
/// twice, emits nothing and over-projects, the safe direction; the cascade's
/// emitted rows retire the projection.
fn collect_subtree<R: ValueRows, H>(
    state: &ObjectStateOf<R>,
    root: u64,
    plan: &ChildPlan<'_, H>,
) -> Vec<u64> {
    // Visited set: the delta contract does not guarantee acyclic children maps
    // (CommandCreateSection can re-parent an existing section_id without
    // cleaning the old parent's children list), so guard against infinite DFS.
    plan.project(Projection::Rows(3));
    let mut result = Vec::new();
    let mut visited = std::collections::HashSet::new();
    let mut stack = vec![root];
    while let Some(sid) = stack.pop() {
        if !visited.insert(sid) {
            continue;
        }
        result.push(sid);
        if let Some(children) = state.children.get(&sid) {
            let ids = u64::try_from(children.len()).unwrap_or(u64::MAX);
            plan.project(Projection::Rows(ids.saturating_mul(3)));
            stack.extend_from_slice(children);
        }
    }
    result
}

// ── Cascade, reset and sink scratch models ────────────────────────────────────
//
// What a replayed record's cascades, reset and list-delete sink allocate
// beside the state, priced before the record applies by the structural
// cache's per-record repair charge (RP2,
// `storage::structural_cache::estimate_replay_record_growth`). Each term is
// one collection the engine builds, priced by the shared allocation model
// (`domain::alloc_model`) at an insert or push count that bounds the
// engine's own. Every collection modelled here starts empty in its plan (or
// its record's sink) and is only inserted into or pushed onto, so its
// tables carry no tombstones, and the model's 16 trailing control bytes
// cover hashbrown's group width of 8 or 16.

/// The most frames [`subtree_pushes`] holds: the levels below its root it
/// walks before it answers `None`. A fixed array on the walk's own stack, so
/// the walk allocates nothing on the heap — it runs before the charge that
/// prices it holds anything.
pub const SUBTREE_WALK_FRAMES: usize = 64;

/// An upper bound on the ids `collect_subtree` pushes onto its stack when
/// a cascade deletes `root` from `state`, counted without allocating; `None`
/// past `cap` pushes or past [`SUBTREE_WALK_FRAMES`] levels below `root`.
///
/// The count is the root plus the length of the children list of every id
/// the walk reaches, as the engine's walk starts its stack at the root
/// (delta.rs:4562) and extends it by each list it visits (delta.rs:4571).
/// Unlike the engine's, this walk keeps no visited set: an id
/// reached twice (a stale listing, a DAG) is counted, and its list walked,
/// every time, and a ghost (a listed id with no row) is walked as the engine
/// walks it. So the count is at least the engine's pushes, which are at
/// least the ids it visits and the sections it deletes. A cycle never ends,
/// so it answers `None` at the cap or the depth limit.
///
/// Its frames — the unwalked rest of each list on the path from `root` —
/// live in a fixed array of [`SUBTREE_WALK_FRAMES`] slices on the walk's
/// own stack. One `children` lookup per id reached.
pub fn subtree_pushes<R: ValueRows>(state: &ObjectStateOf<R>, root: u64, cap: u64) -> Option<u64> {
    let mut pushes: u64 = 1;
    if pushes > cap {
        return None;
    }
    let mut frames: [&[u64]; SUBTREE_WALK_FRAMES] = [&[]; SUBTREE_WALK_FRAMES];
    let mut depth = 0;
    let mut id = root;
    loop {
        // `id` is reached: the engine pushes its whole list at once.
        let children = state.children.get(&id).map_or(&[][..], Vec::as_slice);
        if !children.is_empty() {
            pushes = pushes.saturating_add(u64::try_from(children.len()).unwrap_or(u64::MAX));
            if pushes > cap || depth == SUBTREE_WALK_FRAMES {
                return None;
            }
            frames[depth] = children;
            depth += 1;
        }
        // The next id: the first unwalked one of the deepest list with any
        // left; none left anywhere ends the walk.
        loop {
            let Some(top) = depth.checked_sub(1) else {
                return Some(pushes);
            };
            match frames[top].split_first() {
                Some((&first, rest)) => {
                    frames[top] = rest;
                    id = first;
                    break;
                }
                None => depth = top,
            }
        }
    }
}

/// What a replayed record's cascades can allocate beside the state at once,
/// for `pushes`, at least every id their subtree walks push
/// ([`subtree_pushes`] over the roots the record can delete, plus its
/// births): the sum of the collections `cascade_delete_section`,
/// `collect_subtree` and the deferred purge (`purge_cascaded_rows`)
/// build. Each is modelled at `pushes` inserts or pushes from empty, an upper
/// bound on its own count (each section a cascade deletes or purges was
/// pushed once at least), and the phases are summed rather than maxed — the
/// walk's collections are freed before the purge builds its own — so the
/// model stays monotone in its inputs and independent of the engine's
/// ordering, at about three times the real peak.
///
/// - **The walk's stack** (`collect_subtree`'s `stack`, delta.rs:4562,
///   extended by each list at delta.rs:4571): it grows to at most twice the
///   ids ever on it, and while it grows its previous buffer is alive too —
///   `allocation_bytes(16 P)` plus `allocation_bytes(8 P)`.
/// - **The walk's result** (`result`, delta.rs:4560, one push per visited id
///   at delta.rs:4567) — `vec_growth_bytes(0, 0, P, 8)`.
/// - **The walk's visited set** (`visited`, delta.rs:4561, one insert per
///   popped id at delta.rs:4564) — `table_growth_bytes(0, 0, P, 8)`.
/// - **The plan's `deleted_lists` and `purge_ids`** (one insert per deleted
///   section each, `cascade_delete_section` at delta.rs:4519 and
///   delta.rs:4524) — `table_growth_bytes(0, 0, P, 8)` each.
/// - **With `hooked`** (a replay through the entry's plan indexes), the
///   record's `PlanTouches::lists` (`IndexedHooks::list_before`,
///   plan_index.rs:1316): one insert per list a cascade removes and per other
///   list the record writes (`touched_lists`) —
///   `table_growth_bytes(0, 0, P + touched_lists, 8)`.
/// - **The compact field purge's probe arm**
///   (`CompactFields::purge_sections_with`'s `sections`, types.rs:1217, the
///   purged ids collected and sorted) — `allocation_bytes(8 P)`.
/// - **The name interner's free list** (`NameInterner::release`, types.rs:614,
///   pushes each id whose last row the purge removes) — at most every live
///   name of `names = (live, free_len, free_cap)` pushed onto a free list of
///   that length and capacity, `vec_growth_bytes(free_len, free_cap, live,
///   4)`. The purge runs at the close of the structural phase, before the
///   record's own set fields intern anything, so the live names are the
///   state's.
///
/// 0 for no pushes: a record that cannot cascade allocates none of it.
pub fn cascade_scratch_bytes(
    pushes: u64,
    touched_lists: u64,
    hooked: bool,
    names: (usize, usize, usize),
) -> u64 {
    if pushes == 0 {
        return 0;
    }
    let id_bytes = std::mem::size_of::<u64>() as u64;
    let set = |inserts: u64| table_growth_bytes(0, 0, inserts, id_bytes);
    let stack = allocation_bytes(pushes.saturating_mul(2 * id_bytes))
        .saturating_add(allocation_bytes(pushes.saturating_mul(id_bytes)));
    let result = vec_growth_bytes(0, 0, pushes, id_bytes);
    let visited = set(pushes);
    let deleted_lists = set(pushes);
    let purge_ids = set(pushes);
    let touches = if hooked {
        set(pushes.saturating_add(touched_lists))
    } else {
        0
    };
    let probe = allocation_bytes(pushes.saturating_mul(id_bytes));
    let (live, free_len, free_cap) = names;
    let free = vec_growth_bytes(
        free_len,
        free_cap,
        u64::try_from(live).unwrap_or(u64::MAX),
        std::mem::size_of::<crate::domain::types::NameId>() as u64,
    );
    stack
        .saturating_add(result)
        .saturating_add(visited)
        .saturating_add(deleted_lists)
        .saturating_add(purge_ids)
        .saturating_add(touches)
        .saturating_add(probe)
        .saturating_add(free)
}

/// What a replayed reset (`apply_reset`) can allocate beside the state, on
/// a state of `sections` section rows and `lists` children lists, modelled
/// as [`cascade_scratch_bytes`] is:
///
/// - **`children_keys`** (delta.rs:3170), every list's key collected (exact
///   length) to be sorted — `allocation_bytes(8 L)`;
/// - **the plan's `parents` and `order`**, one registration per list marked
///   dirty (`ChildPlan::entry`, delta.rs:2036 and delta.rs:2035) —
///   `table_growth_bytes(0, 0, L, size_of::<(u64, ParentEntry)>())` and
///   `vec_growth_bytes(0, 0, L, 8)`;
/// - **the plan's `deleted_lists`**, one insert per section's
///   `DeleteChildrenList` (`emit_reset_section`, delta.rs:3297) —
///   `table_growth_bytes(0, 0, S, 8)`;
/// - **with `hooked`**, the record's `PlanTouches::lists`, extended by every
///   list key the reset takes (`IndexedHooks::reset`, plan_index.rs:1384) —
///   `table_growth_bytes(0, 0, L, 8)`.
///
/// The reset is a record's first step and its plan's sets start empty, but
/// the same sets keep growing through the record's later commands. A caller
/// pricing a reset that can also cascade passes the sets' whole insert counts
/// — `sections` plus the cascades' pushes, `lists` plus the record's other
/// touched lists — since two tables priced apart can come to less than the
/// one table they are.
pub fn reset_scratch_bytes(sections: u64, lists: u64, hooked: bool) -> u64 {
    let id_bytes = std::mem::size_of::<u64>() as u64;
    let parent_slot = std::mem::size_of::<(u64, ParentEntry)>() as u64;
    let keys = allocation_bytes(lists.saturating_mul(id_bytes));
    let parents = table_growth_bytes(0, 0, lists, parent_slot);
    let order = vec_growth_bytes(0, 0, lists, id_bytes);
    let deleted_lists = table_growth_bytes(0, 0, sections, id_bytes);
    let touches = if hooked {
        table_growth_bytes(0, 0, lists, id_bytes)
    } else {
        0
    };
    keys.saturating_add(parents)
        .saturating_add(order)
        .saturating_add(deleted_lists)
        .saturating_add(touches)
}

/// What a [`MutationSink::ListDeletes`] holds while a record replays through
/// it, for at most `deleted_lists` lists deleted inline (a cascade's
/// `DeleteChildrenList` per deleted section, a reset's per section) and at
/// most `touched_lists` parents whose final write is a delete
/// (`finalize_children_lists`): its id vector (pushed at delta.rs:624),
/// `vec_growth_bytes(0, 0, deleted_lists + touched_lists, 8)`, plus the one
/// mutation being built at a time. A mutation built from the state carries
/// at most two names — a `SetSection`'s row and its pre-image's key (as at
/// delta.rs:3364) — each at most
/// [`MAX_NAME_BYTES`](crate::domain::limits::MAX_NAME_BYTES) (limits.rs:51),
/// the limit every stored name was admitted under; what a mutation copies
/// from the record's own elements (a set field's value) is the record's
/// delta bytes, not counted here. A `SetChildren` copies its parent's whole
/// list (delta.rs:2951): the caller prices each touched parent's copy at its
/// list's length.
pub fn list_delete_sink_bytes(deleted_lists: u64, touched_lists: u64) -> u64 {
    let name = allocation_bytes(crate::domain::limits::MAX_NAME_BYTES as u64);
    vec_growth_bytes(
        0,
        0,
        deleted_lists.saturating_add(touched_lists),
        std::mem::size_of::<u64>() as u64,
    )
    .saturating_add(name.saturating_mul(2))
}

// ── Children list helpers ─────────────────────────────────────────────────────

/// The stored key an appending command — `CreateSection`, or `MoveSection`
/// to a different parent — gives a new child of `parent_id`: the largest key
/// among the children currently listed there + 1, or `0` under an empty
/// list. That is the key Kit and the legacy server assign on append, so the
/// server's keys match the ones an early client holds. `apply_move_section`
/// takes the section out of the destination's list before asking, so a
/// moved section never counts toward its own key.
///
/// The `+ 1` saturates at `u32::MAX` rather than rejecting the command, so a
/// parent whose key space is exhausted gives every further appended child
/// `u32::MAX`. The child still lands last: [`insert_child_into_parent`] puts
/// an incoming key after every equal one on both of its arms, so ties at the
/// ceiling keep arrival order, deterministically.
///
/// Reads the parent's [`ParentCache`] and writes nothing. On a sorted list the
/// largest key is the last cached one; an unsorted list (loaded stored state
/// only — see [`ParentCache::sorted`]) takes a linear maximum, counted in
/// [`ApplyDeltaStats::children_scan_steps`].
fn append_order<R: ValueRows, H>(
    state: &ObjectStateOf<R>,
    plan: &mut ChildPlan<'_, H>,
    stats: &mut ApplyDeltaStats,
    parent_id: u64,
) -> u32 {
    let (cache, _) = plan.entry_cached(&state.children, &state.sections, parent_id, stats);
    let largest = if cache.sorted {
        cache.orders.last().copied()
    } else {
        stats.children_scan_steps = stats
            .children_scan_steps
            .saturating_add(cache.orders.len() as u64);
        cache.orders.iter().copied().max()
    };
    largest.map_or(0, |order| order.saturating_add(1))
}

/// Insert `child_id` into `parent_id`'s children list, sorted by section_order
/// with insertion-order tie-breaking (SDD §7.7): the child lands after every
/// sibling whose key is less than **or equal to** its own.
///
/// The key is the caller's. `apply_create_section` and `apply_move_section`'s
/// parent-changing arm pass [`append_order`]'s key, which no listed key
/// exceeds, so the child lands last on both arms below — including at the
/// `u32::MAX` saturation tie. `apply_diff_change_section` passes the client's
/// explicit `sectionOrder`, which can land anywhere. `apply_move_section`'s
/// rename-in-place arm passes the section's existing key, to keep it listed.
/// [`adopt_orphans`] passes the adopted row's existing stored key, in
/// `(key, id)` order, into the list of a section born in this command —
/// which is new, so the de-dup and scrub described next never apply to it.
///
/// If `child_id` is already present in the children list, this is a silent
/// no-op (the parent is not even marked dirty). Stored state can list an id
/// under a parent ahead of it arriving there — a ghost listing with no row,
/// or a stale listing of a section whose row names another parent. Those are
/// shapes this engine does not produce, but a snapshot read reads them
/// back. `apply_move_section`'s parent-changing arm scrubs the destination
/// first, so the moved section always lands last; the create and
/// `apply_diff_change_section` paths do not, and there the skip leaves the
/// listed position in place. Silent de-dup stops this engine ever
/// *introducing* a duplicate id, without breaking those callers.
/// It does not make the list duplicate-free: a list loaded from state stored
/// before b.dmu item 4 can already hold one, which the skip leaves exactly as
/// it found it — see [`ParentCache::duplicated`]. An id that is *not* already
/// a member is inserted into such a list exactly as into any other: the
/// positional patch below touches no occurrence of the repeated id, so the
/// cache stays faithful to the maps (see [`ParentEntry::cache`]). The stricter
/// "duplicate section_id" case — a `CreateSection` that would overwrite an
/// existing section — is rejected earlier in `apply_create_section`.
///
/// b.ot3 parts B and C: no mutation is emitted here. The parent is marked
/// dirty and [`finalize_children_lists`] writes the settled list once at the
/// close of the structural phase. The insertion point comes from the cached
/// parallel `section_order` vector rather than a `state.sections` hash
/// lookup per element — by binary search when the list is sorted, by the
/// pre-b.ot3 linear scan when a loaded list is not (see
/// [`ParentCache::sorted`] for the stored shapes that are not).
///
/// Both arms return the same index the pre-b.ot3 scan did **because the
/// cache holds the invariant documented on [`ParentEntry::cache`]**: the
/// cached orders are the listed children's live [`effective_order`]s and
/// `sorted` reflects that vector. The cache is patched in place by this
/// function and its two siblings, and dropped by
/// [`ChildPlan::invalidate_order_listeners`] whenever a site rewrites or
/// removes a listed child's row — including through a *different* parent,
/// which no amount of list-side maintenance would notice. Without that leg
/// the binary-search arm searches an unpartitioned sequence and the linear arm
/// compares against orders the map no longer holds, and the resulting index
/// depends on how a client batched its commands.
///
/// The residual Θ(N²) element shift (R1) is unchanged and counted in
/// [`ApplyDeltaStats::children_element_moves`].
///
/// **No list under a missing parent.** A children list exists only under the
/// root `0` or a section with a row (SDD §10.4). When `parent_id` is neither,
/// this creates no list, touches no cache and marks nothing dirty: the child
/// stays an unlisted row — an *orphan* — until its parent is born, when
/// [`adopt_orphans`] lists it.
///
/// Three routes arrive here with a missing parent, and each leaves exactly
/// the orphan that rule describes:
///
/// * `apply_diff_change_section`, which does not check that its parent
///   exists (SDD §10.4) — a new row, or a reparent, under an id with no row;
/// * `apply_create_section`, whose parent check runs *before* its
///   name-conflict cascade. When the conflicting sibling's listed subtree
///   loops back to the parent (a cycle in the lists), the cascade deletes the
///   parent too, and the new row is left naming an id that no longer has a
///   row;
/// * `apply_move_section`'s normal (parent-changing) move, whose destination
///   check likewise runs before the edge-5 conflict cascade that can delete
///   the destination the same way. (Its rename-in-place arm cannot, on
///   consistent state: there the destination is the section's own stored
///   parent, so a destination inside the conflicting sibling's subtree makes
///   that sibling an ancestor of the section, and edge 4 deletes the section
///   first.)
///
/// In all three the row's `parent_id` names an id with no row, so keeping no
/// list for it is the rule above, and the row is listed if and when that id
/// is born again. [`adopt_orphans`] itself never arrives with a missing parent:
/// it inserts under the section just born.
fn insert_child_into_parent<R: ValueRows, H: EngineHooks<R>>(
    state: &mut ObjectStateOf<R>,
    plan: &mut ChildPlan<'_, H>,
    stats: &mut ApplyDeltaStats,
    parent_id: u64,
    child_id: u64,
    section_order: u32,
) {
    if parent_id != 0 && !state.sections.contains_key(&parent_id) {
        return;
    }
    // The list is materialized on the first insert under a live parent or the
    // root, as the pre-b.ot3 `entry(parent_id).or_default()` did.
    write::ensure_list(state, &mut plan.hooks, parent_id);

    let (cache, dirty) = plan.entry_cached(&state.children, &state.sections, parent_id, stats);

    if cache.members.contains(&child_id) {
        return;
    }

    // Insertion point: first position whose existing child's section_order
    // is strictly greater than the incoming one. Ties append
    // (insertion-order tie-breaking, SDD §7.7).
    let insert_pos = if cache.sorted {
        // On a non-decreasing list, "first element whose order > x" is
        // exactly `partition_point(|o| o <= x)`.
        cache
            .orders
            .partition_point(|&order| order <= section_order)
    } else {
        // A loaded list is not necessarily non-decreasing (a reorder stored
        // with the client's order under the old keys, a stale or ghost
        // listing), and a binary search would land somewhere else than the
        // pre-b.ot3 scan did. Scan — over the cached orders, which is still
        // one hash lookup per element cheaper. No cached order exceeds an
        // append key, so it runs off the end and lands last here too.
        let pos = cache
            .orders
            .iter()
            .position(|&order| order > section_order)
            .unwrap_or(cache.orders.len());
        // A `position` that finds a hit examined `pos + 1` elements; one
        // that runs off the end examined all of them.
        let steps = (pos + 1).min(cache.orders.len());
        stats.children_scan_steps = stats.children_scan_steps.saturating_add(steps as u64);
        pos
    };

    stats.children_element_moves = stats
        .children_element_moves
        .saturating_add((cache.orders.len() - insert_pos) as u64);

    cache.orders.insert(insert_pos, section_order);
    cache.members.insert(child_id);
    *dirty = true;

    // This parent's populated cache now lists `child_id`, so the index must
    // name it — an add is what the index invariant requires for soundness.
    plan.register_listing(child_id, parent_id, stats);

    write::insert_listed_child(state, &mut plan.hooks, parent_id, insert_pos, child_id);
}

/// Remove `child_id` from `parent_id`'s children list.
///
/// b.ot3: emits nothing — the parent is marked dirty for
/// [`finalize_children_lists`]. It is marked dirty whenever the parent's
/// list *exists*, even if `child_id` was not in it, because that is exactly
/// when the pre-b.ot3 code emitted a `SetChildren`. The membership set makes
/// the absent case O(1); the present case is a `position` + `Vec::remove`
/// (residual R2 — same asymptote as the old `retain`, one memmove instead of
/// a per-element closure).
fn remove_child_from_parent<R: ValueRows, H: EngineHooks<R>>(
    state: &mut ObjectStateOf<R>,
    plan: &mut ChildPlan<'_, H>,
    stats: &mut ApplyDeltaStats,
    parent_id: u64,
    child_id: u64,
) {
    if !state.children.contains_key(&parent_id) {
        return;
    }

    // Detached for the duration so the list edit below and the matching
    // `orders` edit sit in one block — see [`ChildPlan::detach_cache`].
    let mut cache = plan.detach_cache(&state.children, &state.sections, parent_id, stats);
    plan.mark_dirty(parent_id);
    let present = cache.members.remove(&child_id);

    if !present {
        plan.restore_cache(parent_id, cache);
        return;
    }

    // The single point at which this parent's cache stops listing the child,
    // covering both the positional arm and the legacy `retain` arm below. The
    // cache is restored populated on every path from here, so the pair has to
    // go or it buys a spurious rebuild of a live list later.
    plan.deregister_listing(child_id, parent_id);

    let Some(children) = state.children.get(&parent_id) else {
        return;
    };

    if cache.duplicated {
        // Legacy stored state can hold the same id twice in one list (only
        // reachable from before b.dmu item 4 / the de-dup above). Keep the
        // pre-b.ot3 `retain` semantics — remove EVERY occurrence — and
        // re-derive the cache from the repaired list rather than tracking
        // positions.
        write::update_list(state, &mut plan.hooks, parent_id, |children| {
            children.retain(|&c| c != child_id);
        });
        let cache = plan.build_cache(&state.children, &state.sections, parent_id, stats);
        plan.restore_cache(parent_id, cache);
        return;
    }

    let Some(pos) = children.iter().position(|&c| c == child_id) else {
        return;
    };
    stats.children_scan_steps = stats.children_scan_steps.saturating_add((pos + 1) as u64);
    stats.children_element_moves = stats
        .children_element_moves
        .saturating_add((children.len() - pos - 1) as u64);
    write::update_list(state, &mut plan.hooks, parent_id, |children| {
        children.remove(pos);
    });
    cache.orders.remove(pos);
    plan.restore_cache(parent_id, cache);
}

/// Re-place `child_id` inside `parent_id`'s children list after its
/// `section_order` changed, keeping the list sorted by `section_order` with
/// existing insertion order preserved within ties.
///
/// b.ot3 (residual R3): on a sorted, duplicate-free list this is a
/// remove-and-reinsert of the single element whose order changed, dropping
/// the `log C` factor of the pre-b.ot3 `sort_by_key` while reproducing its
/// output exactly — *given the cache invariant documented on
/// [`ParentEntry::cache`]*. Both the `sorted` flag this function branches on
/// and the `orders` it partitions are cached, so this equivalence lasts
/// exactly as long as every site that rewrites or removes a listed child's
/// row keeps them true; see [`ChildPlan::invalidate_order_listeners`], which
/// every such site reaches by comparing [`effective_order`] across its own
/// write.
///
/// A list holding `child_id` more than once (legacy stored state — see
/// [`ParentCache::duplicated`]) takes the `sort_by_key` arm too, so parity
/// there holds *by construction* rather than by argument: that arm is the
/// pre-b.ot3 function, same sort and same key closure. A remove-and-reinsert
/// cannot serve that shape — it re-places only the first occurrence, leaving
/// the list in an order the stable sort never produces and the cache
/// describing orders the maps no longer hold at the other occurrences.
///
/// The one place the cache is deliberately *behind* the maps is this
/// function's own input: the sole caller
/// ([`apply_diff_change_section`]'s same-parent arm) has already written the
/// new `section_order` into `state.sections` and named this parent as the one
/// cache to keep across the invalidation, so a cache that was already
/// populated still holds the child's **old** order at its position. That is
/// what makes the arm valid —
/// removing that position leaves the other listed children's orders
/// untouched and therefore still non-decreasing, which is exactly the
/// sequence the `partition_point` calls below need. When no cache was
/// populated, the detach re-derives one from the post-write maps instead; the
/// same removal argument holds (the list is sorted *including* the new order,
/// so the rest is still non-decreasing and the clamp returns the element's
/// own index), and where the new order broke monotonicity the re-derived
/// cache reads unsorted and the `sort_by_key` arm repairs the list. Exactly
/// means exactly: a *stable* sort keeps equal-order siblings that preceded
/// the moved element in front of it and those that followed it behind it, so
/// the reinsertion index is the element's own former index **clamped** into
/// the equal-order block `[first index with order >= new, first index with
/// order > new)` — not the end of that block, which is where a plain
/// "insert after equals" would wrongly put an element that started in front
/// of its equals.
///
/// On an **unsorted** list (a loaded shape — see [`ParentCache::sorted`] — or
/// a cache re-derived after this very write broke monotonicity) the stable
/// sort is kept, because that sort is what *repairs* a scrambled list;
/// remove-and-reinsert there would silently stop repairing it. A **duplicated** list takes the
/// same arm, for the reason above — and that also disposes of the one case
/// where the kept cache's soundness argument in
/// [`ChildPlan::invalidate_order_listeners`] does not hold, because the sort
/// arm re-reads `state.sections` and discards the kept cache wholesale.
fn rebuild_children_list<R: ValueRows, H: EngineHooks<R>>(
    state: &mut ObjectStateOf<R>,
    plan: &mut ChildPlan<'_, H>,
    stats: &mut ApplyDeltaStats,
    parent_id: u64,
    child_id: u64,
) {
    if !state.children.contains_key(&parent_id) {
        return;
    }

    // Detached for the duration so the list edit and the matching `orders`
    // edit sit in one block — see [`ChildPlan::detach_cache`].
    let mut cache = plan.detach_cache(&state.children, &state.sections, parent_id, stats);
    plan.mark_dirty(parent_id);

    if !cache.sorted || cache.duplicated {
        let sorted = write::update_list_with_sections(
            state,
            &mut plan.hooks,
            parent_id,
            |children, sections| {
                children
                    .sort_by_key(|&cid| sections.get(&cid).map(|s| s.section_order).unwrap_or(0));
            },
        );
        if sorted.is_none() {
            return;
        }
        // The sort read `state.sections` directly, so the detached cache
        // describes neither map any more — drop it and re-derive. Re-deriving
        // here rather than leaving the slot empty is what lets the *next*
        // order change under this parent take the sorted remove-and-reinsert
        // arm: a lazily-rebuilt cache is always built after that change has
        // already landed in `state.sections`, which is exactly the shape that
        // reads as unsorted.
        let cache = plan.build_cache(&state.children, &state.sections, parent_id, stats);
        plan.restore_cache(parent_id, cache);
        return;
    }

    let new_order = state
        .sections
        .get(&child_id)
        .map(|s| s.section_order)
        .unwrap_or(0);

    let Some(children) = state.children.get(&parent_id) else {
        return;
    };
    let Some(pos) = children.iter().position(|&c| c == child_id) else {
        // Not listed under this parent — a legacy row its parent's list
        // lost (an orphan never gets here: its parent keeps no list, so the
        // early return above took it) — nothing to re-place. Pre-b.ot3 this
        // still re-emitted the unchanged list; the parent is already marked
        // dirty above, so the finalizer does.
        plan.restore_cache(parent_id, cache);
        return;
    };
    stats.children_scan_steps = stats.children_scan_steps.saturating_add((pos + 1) as u64);

    stats.children_element_moves = stats
        .children_element_moves
        .saturating_add((children.len() - pos - 1) as u64);
    cache.orders.remove(pos);

    // `[lo, hi)` is the block of siblings whose order equals the new one.
    let lo = cache.orders.partition_point(|&order| order < new_order);
    let hi = cache.orders.partition_point(|&order| order <= new_order);
    let insert_pos = pos.clamp(lo, hi);

    stats.children_element_moves = stats
        .children_element_moves
        .saturating_add((cache.orders.len() - insert_pos) as u64);
    // One list edit: out of `pos`, back in at `insert_pos` (an index into the
    // list without it, as `orders` computed it).
    write::update_list(state, &mut plan.hooks, parent_id, |children| {
        children.remove(pos);
        children.insert(insert_pos, child_id);
    });
    cache.orders.insert(insert_pos, new_order);
    plan.restore_cache(parent_id, cache);
}

/// Find the section whose `(parent_id, name)` matches the arguments,
/// excluding `except_id`.
///
/// Indexes `state.sections` (authoritative membership) rather than the
/// `state.children[parent_id]` list (ordering-only). Stored state can hold a
/// row its parent's list does not reach — a partial reorder stored before
/// reorders kept the children they did not name dropped them from the list
/// — and the snapshot load reads it back. If such a row shares a name with a new sibling, a
/// children-list-only scan misses it and `apply_create_section` happily
/// admits a duplicate `(parent_id, name)` pair (see b.55p H-dom-3).
///
/// b.ot3 part D: the pre-b.ot3 implementation scanned `sections.values()` on
/// **every** command, which is O(S) per command and unbounded by
/// `MAX_DELTA_ENTRIES` (the cap bounds the delta, not the stored layer).
/// The per-command scan is gone: a plan's first probe scans `state.sections`
/// once, allocating nothing ([`scan_child_by_name`]), and its second builds a
/// [`NameIndex`], maintained incrementally afterwards — so a delta pays at
/// most one scan and one build, and a one-probe delta (most interactive
/// edits) builds no index at all. The build is sized at the object's sections
/// plus every entry the delta can add to the index
/// ([`StructuralInserts::name_index_inserts`]: its births, renames and
/// reparents), so it never grows, and in a bounded plan it is projected at
/// that size before it is allocated ([`Projection::IndexedSections`]). Both
/// answer by the same rule (below).
///
/// # Tie-break: the lowest `section_id` wins (b.ot3 P2)
///
/// Two or more siblings can legally share `(parent_id, name)` today:
/// `apply_move_section`'s rename arm (carried `old_parent_id ==
/// new_parent_id`) performs no conflict check, and that stays true. When they do, **the lowest `section_id` is the conflict
/// target.** That is not a cosmetic choice about which id an error names:
///
/// - For `apply_create_section` and `apply_diff_change_section`'s
///   new-section branch it decides **which entire subtree is
///   cascade-deleted** — its descendants, tombstones, fields and time
///   samples.
/// - For `apply_move_section` it additionally decides **which of edge 4 and
///   edge 5 applies**, i.e. whether the moving section's subtree is
///   destroyed or the conflicting sibling's.
/// - For `apply_diff_change_section`'s existing-section branch it fixes
///   `DuplicateSiblingName::conflicting_id`.
///
/// Resolution removes one name-holder; it does not converge the parent to a
/// single holder, so a second conflicting command picks the next-lowest id.
///
/// The pre-b.ot3 `HashMap::values().find()` answered this from the hash
/// table's layout, which depends on a process-local seed — two replicas
/// replaying the same delta bytes, or the live path and
/// `checkpoint::compose`'s replay of the same history, could destroy
/// different subtrees. Any principal already authorized to write this layer
/// could always cause a name-holder's subtree to be cascade-deleted with a
/// legal delta, subject to that same write permission; what changes here is
/// only that the *selection* is now reproducible.
fn find_child_by_name<R: ValueRows, H: EngineHooks<R>>(
    state: &ObjectStateOf<R>,
    plan: &mut ChildPlan<'_, H>,
    stats: &mut ApplyDeltaStats,
    parent_id: u64,
    name: &str,
    except_id: Option<u64>,
) -> Option<u64> {
    stats.name_index_probes = stats.name_index_probes.saturating_add(1);
    // A plan whose hooks keep a persistent index (the structural cache's)
    // answers from it, by the same rule, and builds nothing.
    if let Some(answer) = plan
        .hooks
        .child_by_name(&state.sections, parent_id, name, except_id)
    {
        return answer;
    }
    if plan.names.is_none() {
        if !plan.name_scanned {
            // The plan's first probe: answered by a scan, which allocates
            // nothing, so a one-command edit never builds the index.
            plan.name_scanned = true;
            return pass_timing::time(Pass::NameScan, || {
                scan_child_by_name(&state.sections, parent_id, name, except_id)
            });
        }
        // The second: build the index at the object's sections plus every
        // entry the delta can add to it (its births and moves), projected
        // before it is allocated.
        let indexed = u64::try_from(state.sections.len())
            .unwrap_or(u64::MAX)
            .saturating_add(plan.index_inserts);
        plan.project(Projection::IndexedSections(indexed));
        stats.name_index_builds = stats.name_index_builds.saturating_add(1);
        plan.names = Some(pass_timing::time(Pass::NameIndexBuild, || {
            NameIndex::build(&state.sections, indexed)
        }));
    }
    let index = plan
        .names
        .as_ref()
        .expect("name index built immediately above");

    let bucket = index.buckets.get(&(parent_id, name_hash(name)))?;
    // Bucket entries are ascending candidates: re-verify each against
    // `state.sections` (a hash collision, or an id the index has not yet
    // been told about, must never be reported as a conflict) and return the
    // lowest surviving id.
    bucket.iter().copied().find(|&section_id| {
        except_id != Some(section_id)
            && state
                .sections
                .get(&section_id)
                .is_some_and(|s| s.parent_id == parent_id && s.name == name)
    })
}

/// [`find_child_by_name`]'s answer without the index, by one pass over
/// `sections` that allocates nothing: the same rule — the lowest
/// `section_id` among the rows naming `(parent_id, name)`, re-verified by
/// that id's own row as the index's candidates are, other than `except_id`.
/// The plan's first probe takes it.
fn scan_child_by_name(
    sections: &std::collections::HashMap<u64, Section>,
    parent_id: u64,
    name: &str,
    except_id: Option<u64>,
) -> Option<u64> {
    sections
        .values()
        .filter(|s| s.parent_id == parent_id && s.name == name)
        .map(|s| s.section_id)
        .filter(|&section_id| {
            except_id != Some(section_id)
                && sections
                    .get(&section_id)
                    .is_some_and(|s| s.parent_id == parent_id && s.name == name)
        })
        .min()
}

/// Debug-only cross-map consistency check (debug builds only — release
/// builds skip the walk entirely).
///
/// Verifies three invariants at a stable post-mutation boundary:
///
/// 1. **Forward:** every non-orphan section in `state.sections` is listed in
///    its parent's `state.children[parent_id]`. (A section whose parent
///    isn't in `state.sections` is a legitimate orphan per SDD §10.4, and is
///    listed nowhere.)
///
/// 2. **Reverse:** every id in `state.children[p]` (for every `p`) exists in
///    `state.sections` and reports `parent_id == p`. The reverse direction
///    is the b.55p PM-rework strengthening: a pre-H-dom-2 `ReorderChildren`
///    that listed a foreign section under a new parent used to leave the
///    child on two parents simultaneously, with its own `parent_id` still
///    pointing at the original parent — cross-map drift that breaks move
///    and delete cascades that read `parent_id` to pick the live children
///    list.
///
/// 3. **Lists under live parents only:** every key of `state.children` is the
///    root `0` or a section with a row. The reverse direction cannot see a
///    list under a missing parent, because every entry such a list could hold
///    names that parent. Keeping no such list is what lets the forward
///    direction exempt orphans without hiding a missed listing: an orphan's
///    parent is born only through [`adopt_orphans`], which lists it.
///
/// All three are meant to hold after every command applied to a state
/// that already satisfied them, whatever order concurrent clients' commands
/// arrive in. A command that takes a section out of a list deletes its row
/// or lists it under its new parent (or, under a missing parent, nowhere);
/// a section's birth lists the rows already naming it, and a cascade or a
/// reset that removes a row removes its list with it;
/// [`apply_reorder_children`] keeps every child it does not name (installing
/// a partial list as the whole list would drop them, breaking the forward
/// direction from legal input) and ignores every id that is not already one
/// of its parent's listed children (listing one would break the reverse
/// direction); and `apply_move_section` works from the section's *stored*
/// parent, removing it from the carried `old_parent_id`'s list as well, so a
/// stale carried parent leaves no ghost or stale listing behind (working from
/// the carried parent alone would break the reverse direction from legal
/// input). A firing
/// therefore means an input state that was already inconsistent — for
/// instance a snapshot read torn by an unfenced concurrent delete, or legacy
/// stored state — or an engine defect.
///
/// Called only from stable boundaries (end of `apply_delta`) because
/// sections can be transiently absent from children maps mid-mutation (e.g.
/// between `remove_child_from_parent` and the matching
/// `insert_child_into_parent` inside `apply_move_section`).
#[cfg(debug_assertions)]
#[inline]
fn debug_assert_cross_map_consistent<R: ValueRows>(state: &ObjectStateOf<R>) {
    // Forward: section → children-list membership.
    for section in state.sections.values() {
        let parent_id = section.parent_id;
        // Root (0) is not a section, but its list is checked like any other.
        // A section parented to a missing section is an orphan (SDD §10.4),
        // listed nowhere until its parent is born; the third check below
        // keeps that exemption from hiding anything.
        if parent_id != 0 && !state.sections.contains_key(&parent_id) {
            continue;
        }
        let children = state.children.get(&parent_id);
        debug_assert!(
            children.is_some_and(|cs| cs.contains(&section.section_id)),
            "cross-map drift: section {} (parent {}) missing from children list",
            section.section_id,
            parent_id,
        );
    }

    // Reverse: children-list entry → section existence + parent_id match.
    for (parent_id, children) in &state.children {
        for child_id in children {
            let Some(child) = state.sections.get(child_id) else {
                debug_assert!(
                    false,
                    "cross-map drift: state.children[{}] lists section_id {} \
                     which is not in state.sections",
                    parent_id, child_id,
                );
                continue;
            };
            debug_assert!(
                child.parent_id == *parent_id,
                "cross-map drift: state.children[{}] lists section_id {} whose \
                 own parent_id is {} (listed under the wrong parent)",
                parent_id,
                child_id,
                child.parent_id,
            );
        }
    }

    // Lists under live parents only: no list is kept under an id with no row.
    for parent_id in state.children.keys() {
        debug_assert!(
            *parent_id == 0 || state.sections.contains_key(parent_id),
            "cross-map drift: state.children holds a list under {} which is neither the \
             root nor a section with a row",
            parent_id,
        );
    }
}

#[cfg(not(debug_assertions))]
#[inline]
fn debug_assert_cross_map_consistent<R: ValueRows>(_state: &ObjectStateOf<R>) {}

/// Audit [`ChildPlan::listed_in`] against the caches it indexes: every child a
/// populated [`ParentCache`] lists must have its parent named in the index.
///
/// This is the soundness half of the index invariant — the half a missing
/// registration breaks, and the only half that produces a wrong answer. Extra
/// index entries are permitted by design (they cost a rebuild, never a wrong
/// insertion index), so the assertion is a superset check, not an equality.
///
/// Also re-derives [`ChildPlan::listed_pairs`] from the map and checks it
/// against the incrementally maintained value — the one place anything
/// compares the two, and the reason the peak counter can be trusted to have
/// seen every stored pair.
///
/// Run once per delta, at the same boundary as its neighbour, over data
/// already in memory: O(total cached entries). For a delta that walks the
/// tree, that sits inside the O(S) `check_section_tree_depth` walk. A delta
/// that [`ApplyMode::Replay`] lets skip the walk touches no children list,
/// builds no cache, and makes this audit a no-op.
/// Deliberately **not** a re-derivation on every cache use — that would make
/// debug builds quadratic again, which is the cost this design exists to
/// remove. Not reached on an error return: it follows the last `?`, and the
/// partial-mutation contract already tells callers to discard that state.
#[cfg(debug_assertions)]
#[inline]
fn debug_assert_cache_index_superset<H>(plan: &ChildPlan<'_, H>) {
    for (parent_id, entry) in &plan.parents {
        let Some(cache) = entry.cache.as_ref() else {
            continue;
        };
        for child_id in &cache.members {
            debug_assert!(
                plan.listed_in
                    .get(child_id)
                    .is_some_and(|listers| listers.as_slice().contains(parent_id)),
                "cache index drift: parent {} caches child {} but the dependency \
                 index does not name it — an order-side write on {} would leave \
                 that cache stale",
                parent_id,
                child_id,
                child_id,
            );
        }
    }

    // The live pair count is maintained incrementally at every add and drop,
    // so nothing re-reads the map to check it during the delta. Re-deriving it
    // once here is what turns "the increments and decrements balance" from an
    // argument into a checked fact. A drifted count misreports
    // `cache_index_peak_pairs` in whichever direction it drifted, and a
    // memory-bound counter reading low is a growth the bound's own tests would
    // then fail to see.
    let stored: usize = plan
        .listed_in
        .values()
        .map(|listers| listers.as_slice().len())
        .sum();
    debug_assert_eq!(
        plan.listed_pairs, stored as u64,
        "cache index pair-count drift: the incrementally maintained count and the \
         pairs actually stored disagree",
    );
}

#[cfg(not(debug_assertions))]
#[inline]
fn debug_assert_cache_index_superset<H>(_plan: &ChildPlan<'_, H>) {}

/// Check whether `candidate` is a descendant of `ancestor` (or equal).
/// Used for cycle detection and ancestor-conflict detection (SDD §9.3).
///
/// b.dmu item 1: the prior implementation guarded the parent-walk arm with
/// `sec.parent_id != candidate` and fell through to `_ => return false` on
/// match. That silently reported "not a descendant" for a self-parenting
/// section, which let `apply_move_section`'s cycle check pass for a target
/// that is its own parent. With the guard removed, a self-parent (i.e.
/// `sec.parent_id == candidate`) walks to itself on the next iteration and
/// is caught by the `visited.insert(candidate)` check at the loop head —
/// returning `false` only because the cycle was detected, never because
/// the function silently absorbed the self-parent edge.
///
/// b.ot3: the visited set is supplied by the caller (`ChildPlan::
/// descendant_scratch`) and cleared here, so a delta full of `MoveSection`
/// commands allocates one set instead of one per call. The O(depth) walk
/// itself (residual R5) is unchanged.
fn is_descendant_of<R: ValueRows>(
    state: &ObjectStateOf<R>,
    mut candidate: u64,
    ancestor: u64,
    visited: &mut std::collections::HashSet<u64>,
) -> bool {
    visited.clear();
    loop {
        if candidate == ancestor {
            return true;
        }
        if !visited.insert(candidate) {
            // Cycle guard — protects against corrupt stored state and
            // self-parenting (b.dmu item 1).
            return false;
        }
        match state.sections.get(&candidate) {
            Some(sec) => candidate = sec.parent_id,
            None => return false,
        }
    }
}

// ── Value helpers ─────────────────────────────────────────────────────────────

fn field_value_from_oodf(
    value_or_ext_hash: &[u8],
    ext_value_size: u64,
    ext_value_index: u32,
) -> Result<FieldValue, DeltaError> {
    if ext_value_size > 0 {
        // External value — the content hash (a SHA-256 digest) is always
        // exactly 32 bytes by protocol (SDD §11.1). Reject anything else
        // instead of silently padding or truncating — a mismatched length
        // signals a protocol violation (parser bug, schema drift, or
        // malicious client).
        // `validate_delta_bounds` refuses such a delta first on every path that
        // reaches here; this check keeps the conversion total on its own.
        if value_or_ext_hash.len() != 32 {
            return Err(DeltaError::InvalidExternalHashLength {
                got: value_or_ext_hash.len(),
                expected: 32,
            });
        }
        let mut hash = [0u8; 32];
        hash.copy_from_slice(value_or_ext_hash);
        Ok(FieldValue::External {
            hash,
            size: ext_value_size,
            ext_value_index,
        })
    } else {
        Ok(FieldValue::Inline(value_or_ext_hash.to_vec()))
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::types::{ObjectState, SampleRows};

    /// The direct writes to an object's state that only
    /// `domain::delta_write` may make: a table mutator on `sections` or
    /// `children`, a row-store mutator on `fields` or `time_samples`, a
    /// header-scalar assignment, or a wholesale swap of the state.
    const DIRECT_WRITE_NEEDLES: &[&str] = &[
        ".sections.insert(",
        ".sections.remove(",
        ".sections.get_mut(",
        ".sections.entry(",
        ".sections.drain(",
        ".sections.clear(",
        ".sections.retain(",
        ".sections.values_mut(",
        ".sections.iter_mut(",
        ".children.insert(",
        ".children.remove(",
        ".children.get_mut(",
        ".children.entry(",
        ".children.drain(",
        ".children.clear(",
        ".children.retain(",
        ".children.values_mut(",
        ".children.iter_mut(",
        ".insert_row(",
        ".purge_sections(",
        ".purge_sections_with(",
        ".drain_rows(",
        ".take_all(",
        ".sequence_num =",
        ".first_version =",
        ".source_format =",
        ".next_set_order =",
        "mem::take(&mut state",
        "mem::replace(&mut state",
        "mem::swap(&mut state",
        ".shrink_to_fit(",
        ".zero_ext_indexes(",
        // `extend` and `append` on the state's tables only: the engine calls
        // both on its own scratch (`RewriteBudget::extend`, a local `Vec`).
        ".sections.extend(",
        ".children.extend(",
        ".fields.extend(",
        ".time_samples.extend(",
        ".children.append(",
        "&mut state.sections",
        "&mut state.children",
        "&mut state.fields",
        "&mut state.time_samples",
        "*state =",
    ];

    /// A struct pattern (or literal) of the state's type, under each of its
    /// names. A destructure binds the state's tables by name, after which a
    /// mutator on a binding carries no `.sections` or `.fields` prefix for
    /// [`DIRECT_WRITE_NEEDLES`] to match. Not matched after `->`, where the
    /// type is a return type before a function body.
    const STATE_PATTERN_NEEDLES: &[&str] = &[
        "ObjectStateOf{",
        "ObjectStateOf::<",
        "ObjectState{",
        "CompactState{",
    ];

    /// `source` with its comments, the contents of its string and character
    /// literals and all its whitespace removed — so a needle matches however
    /// rustfmt splits a chain or an assignment across lines, and never inside
    /// a comment or a literal — and, beside it, the 1-based source line of
    /// each byte kept.
    fn strip_for_scan(source: &str) -> (Vec<u8>, Vec<usize>) {
        let bytes = source.as_bytes();
        let is_ident = |b: u8| b.is_ascii_alphanumeric() || b == b'_';
        let mut code = Vec::with_capacity(bytes.len());
        let mut lines = Vec::with_capacity(bytes.len());
        let mut line = 1;
        let mut i = 0;
        // Skip `bytes[from..]` up to and including `close`, counting newlines;
        // `escapes` honours a backslash escape. Answers the index after it.
        let skip_to = |from: usize, close: &[u8], escapes: bool, line: &mut usize| {
            let mut j = from;
            while j < bytes.len() {
                if escapes && bytes[j] == b'\\' {
                    if bytes.get(j + 1) == Some(&b'\n') {
                        *line += 1;
                    }
                    j += 2;
                    continue;
                }
                if bytes[j..].starts_with(close) {
                    return j + close.len();
                }
                if bytes[j] == b'\n' {
                    *line += 1;
                }
                j += 1;
            }
            bytes.len()
        };
        // The identifier characters just before `at`: a raw string's `r`
        // follows none, or a lone `b` (a raw byte string's).
        let ident_before = |at: usize| {
            let start = bytes[..at]
                .iter()
                .rposition(|&c| !is_ident(c))
                .map_or(0, |p| p + 1);
            &bytes[start..at]
        };
        // The `#`s after a raw string's `r`, when a `"` follows them.
        let raw_hashes = |at: usize| {
            let hashes = bytes[at + 1..].iter().take_while(|&&c| c == b'#').count();
            (bytes.get(at + 1 + hashes) == Some(&b'"')).then_some(hashes)
        };
        while i < bytes.len() {
            let b = bytes[i];
            if b == b'/' && bytes.get(i + 1) == Some(&b'/') {
                // A line comment: up to, not including, its newline.
                while i < bytes.len() && bytes[i] != b'\n' {
                    i += 1;
                }
            } else if b == b'/' && bytes.get(i + 1) == Some(&b'*') {
                // A block comment, nested ones included.
                let mut depth = 0usize;
                while i < bytes.len() {
                    if bytes[i..].starts_with(b"/*") {
                        depth += 1;
                        i += 2;
                    } else if bytes[i..].starts_with(b"*/") {
                        depth -= 1;
                        i += 2;
                        if depth == 0 {
                            break;
                        }
                    } else {
                        if bytes[i] == b'\n' {
                            line += 1;
                        }
                        i += 1;
                    }
                }
            } else if let Some(hashes) = (b == b'r' && matches!(ident_before(i), b"" | b"b"))
                .then(|| raw_hashes(i))
                .flatten()
            {
                // A raw string, `r"…"` or `r#"…"#`, a byte string's too.
                let mut close = vec![b'"'];
                close.extend(vec![b'#'; hashes]);
                i = skip_to(i + 2 + hashes, &close, false, &mut line);
                code.extend_from_slice(b"\"\"");
                lines.extend([line, line]);
            } else if b == b'"' {
                i = skip_to(i + 1, b"\"", true, &mut line);
                code.extend_from_slice(b"\"\"");
                lines.extend([line, line]);
            } else if b == b'\'' {
                // A character literal, or a lifetime (kept).
                let rest = &source[i + 1..];
                let width = rest.chars().next().map_or(0, char::len_utf8);
                if rest.starts_with('\\') {
                    i = skip_to(i + 1, b"'", true, &mut line);
                } else if width > 0 && bytes.get(i + 1 + width) == Some(&b'\'') {
                    i += 2 + width;
                } else {
                    code.push(b);
                    lines.push(line);
                    i += 1;
                }
            } else if b == b'\n' {
                line += 1;
                i += 1;
            } else if b.is_ascii_whitespace() {
                i += 1;
            } else {
                code.push(b);
                lines.push(line);
                i += 1;
            }
        }
        (code, lines)
    }

    /// The start of every occurrence of `needle` in `hay`.
    fn occurrences<'a>(hay: &'a [u8], needle: &'a [u8]) -> impl Iterator<Item = usize> + 'a {
        hay.windows(needle.len())
            .enumerate()
            .filter(move |(_, window)| *window == needle)
            .map(|(at, _)| at)
    }

    /// Each line of `source` before its test module where a
    /// [`DIRECT_WRITE_NEEDLES`] or [`STATE_PATTERN_NEEDLES`] entry starts,
    /// matched on the source stripped of comments, literals and whitespace
    /// ([`strip_for_scan`]), as `(line number, line)`.
    fn direct_writes(source: &str) -> Vec<(usize, String)> {
        let code_end = source
            .find("#[cfg(test)]\nmod tests {")
            .unwrap_or(source.len());
        let source = &source[..code_end];
        let (code, lines) = strip_for_scan(source);
        let mut hits = std::collections::BTreeSet::new();
        for needle in DIRECT_WRITE_NEEDLES {
            let needle: Vec<u8> = needle
                .bytes()
                .filter(|b| !b.is_ascii_whitespace())
                .collect();
            hits.extend(occurrences(&code, &needle).map(|at| lines[at]));
        }
        for needle in STATE_PATTERN_NEEDLES {
            hits.extend(
                occurrences(&code, needle.as_bytes())
                    .filter(|&at| !code[..at].ends_with(b"->"))
                    .map(|at| lines[at]),
            );
        }
        let source_lines: Vec<&str> = source.lines().collect();
        hits.into_iter()
            .map(|line| (line, source_lines[line - 1].trim().to_owned()))
            .collect()
    }

    /// Every write the engine makes to an object's state goes through
    /// `domain::delta_write`, whose hooks keep the structural cache's
    /// persistent plan indexes and its rollback journal. A direct write here
    /// would change the state behind those hooks: an index that no longer
    /// matches its entry, or a rollback that leaves the write in place.
    #[test]
    fn no_direct_table_or_header_mutation_outside_the_writer_module() {
        // The scanner itself: it flags a direct write and ignores one that is
        // commented out or that sits in the test module.
        let probe =
            "fn f() {\n    state.sections.insert(1, row);\n    // state.children.clear();\n    \
                     write::insert_section(state, hooks, row);\n}\n#[cfg(test)]\nmod tests {\n    \
                     state.sequence_num = 3;\n}\n";
        assert_eq!(
            direct_writes(probe),
            vec![(2, "state.sections.insert(1, row);".to_owned())],
            "the scanner must flag exactly the one live direct write in its probe"
        );
        // Across lines, as rustfmt splits a chain or an assignment, and
        // through a destructure; never inside a literal or a block comment,
        // nor for a return type.
        let multi_line_probe = "fn g(state: &mut CompactState) {\n    state\n        .sections\n        \
                                .insert(1, row);\n    let ObjectStateOf { children, .. } = state;\n    \
                                let note = \"state.fields.take_all()\"; /* state.children.clear() */\n    \
                                let quote = '\"'; state.next_set_order\n        = 2;\n}\n\
                                fn h() -> ObjectState {\n    ObjectState::default()\n}\n";
        assert_eq!(
            direct_writes(multi_line_probe),
            vec![
                (3, ".sections".to_owned()),
                (5, "let ObjectStateOf { children, .. } = state;".to_owned()),
                (7, "let quote = '\"'; state.next_set_order".to_owned()),
            ],
            "the scanner must flag the split chain, the destructure and the split assignment, \
             and nothing else, in its multi-line probe"
        );

        let found = direct_writes(include_str!("delta.rs"));
        assert!(
            found.is_empty(),
            "direct writes to the object state in src/domain/delta.rs outside \
             domain::delta_write (route each through a writer there): {found:#?}"
        );
    }

    fn make_state() -> ObjectState {
        ObjectState {
            object_id: 1,
            ..Default::default()
        }
    }

    fn make_section(id: u64, parent: u64, name: &str) -> Section {
        Section {
            section_id: id,
            parent_id: parent,
            name: name.to_string(),
            section_type: 0,
            section_order: 0,
            first_version: 1,
            last_version: 1,
        }
    }

    // ── Empty-delta / sequence-number invariants (SDD §8.4) ──────────────────

    #[test]
    fn apply_delta_none_always_increments_sequence_num() {
        let mut state = make_state();
        assert_eq!(state.sequence_num, 0);
        apply_delta(&mut state, None).unwrap();
        assert_eq!(state.sequence_num, 1);
        apply_delta(&mut state, None).unwrap();
        assert_eq!(state.sequence_num, 2);
    }

    #[test]
    fn apply_delta_none_returns_header_mutation_only() {
        // SDD §8.4: empty delta (None) = no content changes, but still emits
        // the SetObjectHeader mutation reflecting the incremented sequence_num.
        let mut state = make_state();
        let mutations = apply_delta(&mut state, None).unwrap();
        assert_eq!(mutations.len(), 1);
        assert!(matches!(
            mutations[0],
            StorageMutation::SetObjectHeader { .. }
        ));
    }

    #[test]
    fn apply_delta_some_default_owned_delta_only_emits_header_mutation() {
        // A fully-default OwnedDelta (is_diff=false, no commands/fields/samples)
        // is semantically equivalent to None for mutation output.
        let mut state = make_state();
        let delta = OwnedDelta::default();
        let mutations = apply_delta(&mut state, Some(delta)).unwrap();
        assert_eq!(mutations.len(), 1);
        assert!(matches!(
            mutations[0],
            StorageMutation::SetObjectHeader { .. }
        ));
    }

    // ── Reset path (SDD §9.1.3) ──────────────────────────────────────────────

    #[test]
    fn apply_delta_reset_erases_all_state_and_stores_source_format() {
        let mut state = make_state();
        state.sections.insert(10, make_section(10, 0, "Root"));
        state.children.insert(0, vec![10]);
        state.fields.insert(
            (10, "k".to_string()),
            Field {
                section_id: 10,
                key_name: "k".to_string(),
                value: FieldValue::Inline(vec![1]),
                set_order: 0,
                authoring_set_order: 0,
                version: 1,
            },
        );
        state.time_samples.insert(
            (10, OrderedFloat(1.0)),
            TimeSample {
                section_id: 10,
                time: OrderedFloat(1.0),
                value: FieldValue::Inline(vec![2]),
                version: 1,
            },
        );

        let delta = OwnedDelta {
            is_diff: true,
            base_version: 0,
            source_format: "usda".to_string(),
            ..Default::default()
        };
        apply_delta(&mut state, Some(delta)).unwrap();

        assert!(state.sections.is_empty());
        assert!(state.fields.is_empty());
        assert!(state.time_samples.is_empty());
        assert_eq!(state.first_version, 1);
        assert_eq!(state.next_set_order, 0);
        assert_eq!(state.source_format, "usda");
    }

    // ── Signed-zero sample times on the compact form (b.mr6 PD1) ────────────

    /// The external hash both signed-zero rows of
    /// [`signed_zero_compact_state`] hold.
    const SIGNED_ZERO_HASH: [u8; 32] = [0x5A; 32];
    /// The section holding the two signed-zero rows, under [`SIGNED_ZERO_PARENT`].
    const SIGNED_ZERO_SECTION: u64 = 11;
    const SIGNED_ZERO_PARENT: u64 = 10;

    /// A compact state holding the two stored rows `(11, 0.0)` and
    /// `(11, -0.0)` — two stored keys, both of one external hash — on section
    /// 11 under section 10 under the root. The full form cannot hold both: its
    /// `OrderedFloat` key conflates them.
    fn signed_zero_compact_state() -> crate::domain::types::CompactState {
        let mut state = crate::domain::types::CompactState {
            object_id: 1,
            sequence_num: 4,
            first_version: 1,
            ..Default::default()
        };
        state.sections.insert(
            SIGNED_ZERO_PARENT,
            make_section(SIGNED_ZERO_PARENT, 0, "Parent"),
        );
        state.sections.insert(
            SIGNED_ZERO_SECTION,
            make_section(SIGNED_ZERO_SECTION, SIGNED_ZERO_PARENT, "Child"),
        );
        state.children.insert(0, vec![SIGNED_ZERO_PARENT]);
        state
            .children
            .insert(SIGNED_ZERO_PARENT, vec![SIGNED_ZERO_SECTION]);
        for time in [0.0f64, -0.0] {
            state
                .time_samples
                .insert(SIGNED_ZERO_SECTION, time.to_bits(), Some(SIGNED_ZERO_HASH));
        }
        assert_eq!(SampleRows::len(&state.time_samples), 2, "fixture: two rows");
        state
    }

    /// The `(section_id, time bits)` of every `DeleteTimeSample` in
    /// `mutations`, sorted.
    fn deleted_sample_bits(mutations: &[StorageMutation]) -> Vec<(u64, u64)> {
        let mut out: Vec<(u64, u64)> = mutations
            .iter()
            .filter_map(|m| match m {
                StorageMutation::DeleteTimeSample { section_id, time } => {
                    Some((*section_id, time.into_inner().to_bits()))
                }
                _ => None,
            })
            .collect();
        out.sort_unstable();
        out
    }

    /// Both signed-zero rows, as `(section_id, time bits)`, sorted.
    fn both_signed_zero_rows() -> Vec<(u64, u64)> {
        let mut rows = vec![
            (SIGNED_ZERO_SECTION, 0.0f64.to_bits()),
            (SIGNED_ZERO_SECTION, (-0.0f64).to_bits()),
        ];
        rows.sort_unstable();
        rows
    }

    /// A reset on a compact state holding `(s, 0.0)` and `(s, -0.0)` emits a
    /// `DeleteTimeSample` for each stored row, by its own bit pattern.
    #[test]
    fn apply_delta_reset_on_a_compact_state_deletes_both_signed_zero_rows() {
        let mut state = signed_zero_compact_state();
        let reset = OwnedDelta {
            is_diff: true,
            base_version: 0,
            ..Default::default()
        };

        let mutations = apply_delta(&mut state, Some(reset)).expect("the reset applies");

        assert_eq!(deleted_sample_bits(&mutations), both_signed_zero_rows());
        assert!(state.time_samples.is_empty());
    }

    /// A cascade over a section holding `(s, 0.0)` and `(s, -0.0)` purges
    /// both rows, each by its own bit pattern.
    #[test]
    fn apply_delta_cascade_on_a_compact_state_purges_both_signed_zero_rows() {
        let mut state = signed_zero_compact_state();
        let delete = OwnedDelta {
            struct_commands: vec![StructCommand::DeleteSection(
                oodf_ffi::CommandDeleteSection {
                    parent_id: 0,
                    section_id: SIGNED_ZERO_PARENT,
                },
            )],
            ..Default::default()
        };

        let mutations = apply_delta(&mut state, Some(delete)).expect("the delete applies");

        assert_eq!(deleted_sample_bits(&mutations), both_signed_zero_rows());
        assert!(state.time_samples.is_empty());
    }

    /// A delta setting one sample of [`SIGNED_ZERO_SECTION`] at `time`,
    /// external of [`SIGNED_ZERO_HASH`].
    fn set_signed_zero_sample(time: f64) -> OwnedDelta {
        OwnedDelta {
            time_samples: vec![OodfTimeSample {
                section_id: SIGNED_ZERO_SECTION,
                time,
                value_or_ext_hash: SIGNED_ZERO_HASH.to_vec(),
                ext_value_size: 64,
                ext_value_index: 0,
            }],
            ..Default::default()
        }
    }

    /// A full state holding [`SIGNED_ZERO_SECTION`] under the root, with no
    /// row.
    fn full_state_with_signed_zero_section() -> ObjectState {
        let mut state = make_state();
        state.sections.insert(
            SIGNED_ZERO_SECTION,
            make_section(SIGNED_ZERO_SECTION, 0, "Child"),
        );
        state.children.insert(0, vec![SIGNED_ZERO_SECTION]);
        state
    }

    /// A delta carrying a `-0.0` time — one built in the service that
    /// bypassed the canonicalizing parser — is refused by the engine's
    /// backstop on both forms with `TimeSampleNegativeZeroTime` at the
    /// sample's index, and plans nothing: no `-0.0` row is ever made.
    #[test]
    fn apply_delta_negative_zero_time_is_refused_by_the_backstop_on_both_forms() {
        let mut compact_state = signed_zero_compact_state();
        compact_state.time_samples = Default::default();
        let mut full_state = full_state_with_signed_zero_section();

        let compact_err = apply_delta(&mut compact_state, Some(set_signed_zero_sample(-0.0)))
            .expect_err("compact: a -0.0 time is refused");
        let full_err = apply_delta(&mut full_state, Some(set_signed_zero_sample(-0.0)))
            .expect_err("full: a -0.0 time is refused");

        for (form, err) in [("compact", compact_err), ("full", full_err)] {
            assert!(
                matches!(err, DeltaError::TimeSampleNegativeZeroTime { index: 0 }),
                "{form}: expected TimeSampleNegativeZeroTime at index 0, got {err:?}"
            );
        }
        assert_eq!(SampleRows::len(&compact_state.time_samples), 0);
        assert_eq!(SampleRows::len(&full_state.time_samples), 0);
    }

    /// Setting a `0.0` sample on a compact state holding a legacy `-0.0` row
    /// of the same section makes a second compact row, as it makes a second
    /// stored key, and leaves the legacy row; the full form, which a reader
    /// decodes the legacy row into at `+0.0`, holds one row for both.
    #[test]
    fn apply_delta_zero_sample_beside_a_legacy_negative_zero_row_makes_a_second_compact_row() {
        let mut compact_state = signed_zero_compact_state();
        compact_state.time_samples = Default::default();
        compact_state.time_samples.insert(
            SIGNED_ZERO_SECTION,
            (-0.0f64).to_bits(),
            Some(SIGNED_ZERO_HASH),
        );
        let mut full_state = full_state_with_signed_zero_section();
        apply_delta(&mut full_state, Some(set_signed_zero_sample(0.0)))
            .expect("full: the decoded legacy row, at +0.0");

        apply_delta(&mut compact_state, Some(set_signed_zero_sample(0.0))).expect("compact set");
        apply_delta(&mut full_state, Some(set_signed_zero_sample(0.0))).expect("full set");

        let ts = &compact_state.time_samples;
        assert_eq!(SampleRows::len(ts), 2, "two stored keys, two compact rows");
        assert!(ts.contains(SIGNED_ZERO_SECTION, 0.0f64.to_bits()));
        assert!(ts.contains(SIGNED_ZERO_SECTION, (-0.0f64).to_bits()));
        assert_eq!(
            SampleRows::len(&full_state.time_samples),
            1,
            "the full form holds one row for both"
        );
    }

    // ── Rule E: a reset or cascade deletes time 0 at both spellings (b.8jc) ─

    /// The parent every rule-E fixture section is listed under.
    const RULE_E_PARENT: u64 = 20;

    /// A full state with `sections` listed under [`RULE_E_PARENT`] (under the
    /// root), each holding one time-sample row at each of its `times`.
    fn rule_e_full_state(sections: &[(u64, &[f64])]) -> ObjectState {
        let mut state = make_state();
        state.sequence_num = 3;
        state.first_version = 1;
        seed_section(&mut state, RULE_E_PARENT, 0, "P", 0);
        state.children.insert(0, vec![RULE_E_PARENT]);
        let mut listed = Vec::new();
        for &(section_id, times) in sections {
            seed_section(
                &mut state,
                section_id,
                RULE_E_PARENT,
                &format!("S{section_id}"),
                0,
            );
            listed.push(section_id);
            for &time in times {
                seed_time_sample(&mut state, section_id, time);
            }
        }
        state.children.insert(RULE_E_PARENT, listed);
        state
    }

    /// The two deltas that delete every row of [`RULE_E_PARENT`]'s sections:
    /// a reset's drain, and the cascade of deleting the parent.
    fn rule_e_deletions() -> [(&'static str, OwnedDelta); 2] {
        [
            (
                "reset",
                OwnedDelta {
                    is_diff: true,
                    base_version: 0,
                    ..Default::default()
                },
            ),
            ("cascade", batch(vec![delete_cmd(RULE_E_PARENT)])),
        ]
    }

    /// `(section_id, time bits)` for every listed `(section_id, time)`,
    /// sorted.
    fn sample_bits(rows: &[(u64, f64)]) -> Vec<(u64, u64)> {
        let mut out: Vec<(u64, u64)> = rows.iter().map(|&(s, t)| (s, t.to_bits())).collect();
        out.sort_unstable();
        out
    }

    /// Each mutation's `Debug` text, sorted: a plan's mutations as a
    /// multiset, for comparing two plans up to the order within a run.
    fn mutation_multiset(mutations: &[StorageMutation]) -> Vec<String> {
        let mut out: Vec<String> = mutations.iter().map(|m| format!("{m:?}")).collect();
        out.sort_unstable();
        out
    }

    /// A section holding time 0 only at `+0.0` — on the full form, which
    /// always holds time 0 there, and on its compact projection — is deleted
    /// by a reset and by a cascade at both `+0.0` and `-0.0`, once each; a
    /// section holding no time-0 row (only `1.0`) gets no added record; and
    /// the full and compact plans are the same mutations up to the order
    /// within a run.
    #[test]
    fn apply_delta_reset_and_cascade_delete_a_positive_zero_row_at_both_spellings_on_both_forms() {
        let full = rule_e_full_state(&[(21, &[0.0, 1.0]), (22, &[1.0])]);
        let expected = sample_bits(&[(21, 0.0), (21, -0.0), (21, 1.0), (22, 1.0)]);
        for (what, delta) in rule_e_deletions() {
            let mut full_state = full.clone();
            let mut compact_state = crate::domain::types::compact(&full);

            let full_plan = apply_delta(&mut full_state, Some(delta.clone()))
                .unwrap_or_else(|e| panic!("full {what}: {e:?}"));
            let compact_plan = apply_delta(&mut compact_state, Some(delta))
                .unwrap_or_else(|e| panic!("compact {what}: {e:?}"));

            assert_eq!(deleted_sample_bits(&full_plan), expected, "full {what}");
            assert_eq!(
                deleted_sample_bits(&compact_plan),
                expected,
                "compact {what}"
            );
            assert_eq!(
                mutation_multiset(&full_plan),
                mutation_multiset(&compact_plan),
                "{what}: the two forms plan the same mutations up to run order"
            );
            assert_eq!(SampleRows::len(&compact_state.time_samples), 0, "{what}");
        }
    }

    /// A compact section holding time 0 only at `-0.0` — a legacy row — is
    /// deleted by a reset and by a cascade at both spellings, once each.
    #[test]
    fn apply_delta_reset_and_cascade_delete_a_legacy_negative_zero_row_at_both_spellings() {
        let full = rule_e_full_state(&[(21, &[])]);
        for (what, delta) in rule_e_deletions() {
            let mut state = crate::domain::types::compact(&full);
            state.time_samples.insert(21, (-0.0f64).to_bits(), None);

            let plan =
                apply_delta(&mut state, Some(delta)).unwrap_or_else(|e| panic!("{what}: {e:?}"));

            assert_eq!(
                deleted_sample_bits(&plan),
                sample_bits(&[(21, 0.0), (21, -0.0)]),
                "{what}"
            );
        }
    }

    /// A compact section holding time 0 at both spellings gets exactly its
    /// two deletes from a reset and from a cascade: no added record, no
    /// duplicate. A section with no time-0 row beside it gets none.
    #[test]
    fn apply_delta_reset_and_cascade_over_both_spellings_add_no_record() {
        let full = rule_e_full_state(&[(21, &[]), (22, &[2.5])]);
        for (what, delta) in rule_e_deletions() {
            let mut state = crate::domain::types::compact(&full);
            for time in [0.0f64, -0.0] {
                state.time_samples.insert(21, time.to_bits(), None);
            }

            let plan =
                apply_delta(&mut state, Some(delta)).unwrap_or_else(|e| panic!("{what}: {e:?}"));

            assert_eq!(
                deleted_sample_bits(&plan),
                sample_bits(&[(21, 0.0), (21, -0.0), (22, 2.5)]),
                "{what}"
            );
        }
    }

    /// Constraint 1: rule E's added record is emitted inside the delete run,
    /// never after the delta's own writes. A compact state holds time 0 of
    /// section 21 only at `-0.0`, so the added record is a `+0.0` delete; a
    /// reset (or a cascade) that re-creates 21 and sets a `+0.0` sample in
    /// the same delta must emit that delete before the `SetTimeSample`, or
    /// the commit would delete the new sample. The new sample is the state's
    /// only row of 21 afterwards.
    #[test]
    fn apply_delta_rule_e_added_delete_precedes_a_re_set_of_the_same_zero_row() {
        let full = rule_e_full_state(&[(21, &[])]);
        let re_set = OodfTimeSample {
            section_id: 21,
            time: 0.0,
            value_or_ext_hash: vec![9],
            ext_value_size: 0,
            ext_value_index: 0,
        };
        let deltas = [
            (
                "reset",
                OwnedDelta {
                    is_diff: true,
                    base_version: 0,
                    struct_commands: vec![create_cmd(0, 21, "S21")],
                    time_samples: vec![re_set.clone()],
                    ..Default::default()
                },
            ),
            (
                "cascade",
                OwnedDelta {
                    struct_commands: vec![delete_cmd(RULE_E_PARENT), create_cmd(0, 21, "S21")],
                    time_samples: vec![re_set],
                    ..Default::default()
                },
            ),
        ];
        for (what, delta) in deltas {
            let mut state = crate::domain::types::compact(&full);
            state.time_samples.insert(21, (-0.0f64).to_bits(), None);

            let plan =
                apply_delta(&mut state, Some(delta)).unwrap_or_else(|e| panic!("{what}: {e:?}"));

            let position = |is: &dyn Fn(&StorageMutation) -> bool, name: &str| {
                plan.iter()
                    .position(is)
                    .unwrap_or_else(|| panic!("{what}: no {name} in {plan:#?}"))
            };
            let added = position(
                &|m| {
                    matches!(m, StorageMutation::DeleteTimeSample { section_id: 21, time }
                        if time.into_inner().to_bits() == 0)
                },
                "+0.0 delete",
            );
            let set = position(
                &|m| matches!(m, StorageMutation::SetTimeSample { sample } if sample.section_id == 21),
                "SetTimeSample",
            );
            assert!(
                added < set,
                "{what}: the added +0.0 delete (index {added}) must precede the SetTimeSample \
                 (index {set})"
            );
            assert_eq!(SampleRows::len(&state.time_samples), 1, "{what}");
            assert!(state.time_samples.contains(21, 0.0f64.to_bits()), "{what}");
        }
    }

    /// Constraint 2: a state and its clones emit rule E's added records in
    /// the same order. A cascade (and a reset) over 16 sections, each holding
    /// time 0 only at `+0.0`, planned on a state and on four of its clones,
    /// gives five identical mutation lists — on the compact form, which the
    /// shadow check compares, and on the full form. Emitting the added
    /// records in a per-plan hash set's order (the cascade's `purge_ids`)
    /// would differ between plans.
    #[test]
    fn apply_delta_rule_e_added_records_are_in_the_same_order_on_a_state_and_its_clones() {
        let sections: Vec<(u64, &[f64])> = (30..46).map(|id| (id, &[0.0][..])).collect();
        let full = rule_e_full_state(&sections);
        let compact = crate::domain::types::compact(&full);
        for (what, delta) in rule_e_deletions() {
            let full_plans: Vec<Vec<StorageMutation>> = (0..5)
                .map(|_| apply_delta(&mut full.clone(), Some(delta.clone())).expect("full plan"))
                .collect();
            let compact_plans: Vec<Vec<StorageMutation>> = (0..5)
                .map(|_| {
                    apply_delta(&mut compact.clone(), Some(delta.clone())).expect("compact plan")
                })
                .collect();
            for (form, plans) in [("full", &full_plans), ("compact", &compact_plans)] {
                assert_eq!(
                    deleted_sample_bits(&plans[0]).len(),
                    32,
                    "{form} {what}: each section's time 0 at both spellings"
                );
                for (i, plan) in plans.iter().enumerate().skip(1) {
                    assert_eq!(plan, &plans[0], "{form} {what}: clone {i}'s plan");
                }
            }
        }
    }

    /// Constraint 5: rule E's added record is counted against a bounded
    /// plan's rewrite like any emitted row. The same reset and cascade over a
    /// section whose only row sits at time 0 keep exactly one more row than
    /// over the same section with its row at time 1.0.
    #[test]
    fn apply_delta_bounded_counts_rule_e_added_record_as_a_rewrite_row() {
        let kept_rows = |time: f64, delta: OwnedDelta| -> u64 {
            let mut state = crate::domain::types::compact(&rule_e_full_state(&[(21, &[time])]));
            let mut stats = ApplyDeltaStats::default();
            match apply_delta_bounded(
                &mut state,
                Some(delta),
                MutationAllowance::UNBOUNDED,
                &mut |_: RewriteCount| None::<u64>,
                &mut stats,
            )
            .expect("the plan applies")
            {
                BoundedPlan::Within { rewrite, .. } => rewrite.rows,
                BoundedPlan::Overflow { needed } => panic!("unbounded plan overflowed: {needed:?}"),
            }
        };
        for (what, delta) in rule_e_deletions() {
            assert_eq!(
                kept_rows(0.0, delta.clone()),
                kept_rows(1.0, delta) + 1,
                "{what}: the added -0.0 delete is one more kept row"
            );
        }
    }

    // ── StructCommand variants ───────────────────────────────────────────────

    #[test]
    fn apply_delta_create_section_creates_section_under_root() {
        let mut state = make_state();
        let delta = OwnedDelta {
            is_diff: false,
            base_version: 0,
            struct_commands: vec![StructCommand::CreateSection(
                oodf_ffi::CommandCreateSection {
                    parent_id: 0,
                    section_id: 200,
                    section_name: "Child".to_string(),
                    section_type: 3,
                },
            )],
            ..Default::default()
        };
        apply_delta(&mut state, Some(delta)).unwrap();

        let sec = state.sections.get(&200).expect("section 200 must exist");
        assert_eq!(sec.name, "Child");
        assert_eq!(sec.section_type, 3);
        assert_eq!(sec.parent_id, 0);
    }

    #[test]
    fn apply_delta_delete_section_removes_and_emits_tombstone() {
        let mut state = make_state();
        state.sections.insert(77, make_section(77, 0, "ToDelete"));
        state.children.insert(0, vec![77]);

        let delta = OwnedDelta {
            is_diff: false,
            struct_commands: vec![StructCommand::DeleteSection(
                oodf_ffi::CommandDeleteSection {
                    parent_id: 0,
                    section_id: 77,
                },
            )],
            ..Default::default()
        };
        let mutations = apply_delta(&mut state, Some(delta)).unwrap();
        assert!(!state.sections.contains_key(&77));
        assert!(mutations
            .iter()
            .any(|m| matches!(m, StorageMutation::CreateTombstone { section_id: 77, .. })));
    }

    #[test]
    fn apply_delta_move_section_rename_same_parent() {
        let mut state = make_state();
        state.sections.insert(20, make_section(20, 0, "OldName"));
        state.children.insert(0, vec![20]);

        let delta = OwnedDelta {
            is_diff: false,
            struct_commands: vec![StructCommand::MoveSection(oodf_ffi::CommandMoveSection {
                old_parent_id: 0,
                new_parent_id: 0,
                section_id: 20,
                new_name: "NewName".to_string(),
            })],
            ..Default::default()
        };
        apply_delta(&mut state, Some(delta)).unwrap();
        assert_eq!(state.sections[&20].name, "NewName");
        assert_eq!(state.sections[&20].parent_id, 0);
    }

    #[test]
    fn apply_delta_reorder_children_replaces_list_atomically() {
        let mut state = make_state();
        // A reorder ignores any id that is not one of the parent's listed
        // children, so 1/2/3 are real sections listed under parent 5 for the
        // reorder to take effect. Naming all three — a complete list — pins
        // that a complete list installs exactly as sent.
        state.sections.insert(5, make_section(5, 0, "Parent"));
        state.sections.insert(1, make_section(1, 5, "A"));
        state.sections.insert(2, make_section(2, 5, "B"));
        state.sections.insert(3, make_section(3, 5, "C"));
        state.children.insert(0, vec![5]);
        state.children.insert(5, vec![1, 2, 3]);

        let delta = OwnedDelta {
            is_diff: false,
            struct_commands: vec![StructCommand::ReorderChildren(
                oodf_ffi::CommandReorderChildren {
                    section_id: 5,
                    children_list_id: 99, // ignored by server (SDD §9.1.4e)
                    children_list: vec![3, 1, 2],
                },
            )],
            ..Default::default()
        };
        apply_delta(&mut state, Some(delta)).unwrap();
        assert_eq!(state.children[&5], vec![3u64, 1, 2]);
    }

    #[test]
    fn apply_delta_diff_change_section_creates_new_section() {
        let mut state = make_state();

        let delta = OwnedDelta {
            is_diff: true,
            base_version: 10,
            struct_commands: vec![StructCommand::DiffChangeSection(
                oodf_ffi::CommandDiffChangeSection {
                    parent_id: 0,
                    section_id: 100,
                    section_name: "PrimA".to_string(),
                    section_type: 1,
                    section_order: 5,
                },
            )],
            ..Default::default()
        };
        apply_delta(&mut state, Some(delta)).unwrap();
        let sec = state.sections.get(&100).expect("section 100 must exist");
        assert_eq!(sec.name, "PrimA");
        assert_eq!(sec.section_type, 1);
        assert_eq!(sec.section_order, 5);
    }

    #[test]
    fn apply_delta_diff_delete_section_silently_ignored() {
        // SDD §9.1.4f: CommandDiffDeleteSection is silently ignored on ingest.
        let mut state = make_state();
        state.sections.insert(88, make_section(88, 0, "Keep"));
        state.children.insert(0, vec![88]);

        let delta = OwnedDelta {
            is_diff: true,
            base_version: 1,
            struct_commands: vec![StructCommand::DiffDeleteSection(
                oodf_ffi::CommandDiffDeleteSection { section_id: 88 },
            )],
            ..Default::default()
        };
        apply_delta(&mut state, Some(delta)).unwrap();
        assert!(
            state.sections.contains_key(&88),
            "DiffDeleteSection must be silently ignored on ingest"
        );
    }

    // ── SetField (field assignment) ──────────────────────────────────────────

    #[test]
    fn apply_delta_set_field_stores_inline_value_with_correct_set_order() {
        let mut state = make_state();
        state.next_set_order = 10;

        let delta = OwnedDelta {
            is_diff: false,
            set_fields: vec![oodf_ffi::SetField {
                section_id: 1,
                key_name: "color".to_string(),
                value_or_ext_hash: vec![0xFF, 0x00],
                ext_value_size: 0,
                ext_value_index: 0,
                set_order: 3,
            }],
            ..Default::default()
        };
        apply_delta(&mut state, Some(delta)).unwrap();

        let field = state
            .fields
            .get(&(1, "color".to_string()))
            .expect("field must exist");
        assert_eq!(field.set_order, 13);
        // The NEW authoring rank must be the delta's raw set_order (3),
        // provably distinct from the cumulative cursor value (13). This is
        // the cursor-divergence pin: `authoring_set_order` carries the true
        // per-delta authoring rank independent of `state.next_set_order`.
        assert_eq!(
            field.authoring_set_order, 3,
            "authoring_set_order must capture the raw delta set_order (3), not the cumulative cursor (13)"
        );
        assert_ne!(
            field.authoring_set_order, field.set_order,
            "authoring rank and cumulative set_order must be distinct when next_set_order was non-zero"
        );
        assert_eq!(field.version, 1);
        assert_eq!(state.next_set_order, 14);
        assert!(matches!(field.value, FieldValue::Inline(ref v) if v == &[0xFF, 0x00]));
    }

    #[test]
    fn apply_delta_set_field_burst_stores_raw_per_field_authoring_ranks() {
        // A single delta carrying three SetFields with distinct raw
        // `set_order` values (0/5/2). The cumulative `set_order` cursor
        // advances monotonically across the burst (0/6/9), but the NEW
        // `authoring_set_order` must preserve each field's raw per-delta
        // rank verbatim (0/5/2) — contrast with
        // `apply_delta_two_fields_in_same_delta_get_monotonic_set_orders`,
        // which pins the monotonic cumulative sequence.
        let mut state = make_state();

        let delta = OwnedDelta {
            is_diff: false,
            set_fields: vec![
                oodf_ffi::SetField {
                    section_id: 1,
                    key_name: "a".to_string(),
                    value_or_ext_hash: vec![0x0A],
                    ext_value_size: 0,
                    ext_value_index: 0,
                    set_order: 0,
                },
                oodf_ffi::SetField {
                    section_id: 1,
                    key_name: "b".to_string(),
                    value_or_ext_hash: vec![0x0B],
                    ext_value_size: 0,
                    ext_value_index: 0,
                    set_order: 5,
                },
                oodf_ffi::SetField {
                    section_id: 1,
                    key_name: "c".to_string(),
                    value_or_ext_hash: vec![0x0C],
                    ext_value_size: 0,
                    ext_value_index: 0,
                    set_order: 2,
                },
            ],
            ..Default::default()
        };
        apply_delta(&mut state, Some(delta)).unwrap();

        let field_a = state.fields.get(&(1, "a".to_string())).unwrap();
        let field_b = state.fields.get(&(1, "b".to_string())).unwrap();
        let field_c = state.fields.get(&(1, "c".to_string())).unwrap();

        // Authoring ranks are the raw per-field set_orders, verbatim.
        assert_eq!(field_a.authoring_set_order, 0);
        assert_eq!(field_b.authoring_set_order, 5);
        assert_eq!(field_c.authoring_set_order, 2);

        // Cumulative cursor advances monotonically across the burst:
        //   a: next=0 + delta=0 -> 0 (next -> 1)
        //   b: next=1 + delta=5 -> 6 (next -> 7)
        //   c: next=7 + delta=2 -> 9 (next -> 10)
        assert_eq!(field_a.set_order, 0);
        assert_eq!(field_b.set_order, 6);
        assert_eq!(field_c.set_order, 9);
        assert_eq!(state.next_set_order, 10);

        // The authoring ranks are NOT the monotonic cumulative sequence.
        assert_ne!(field_b.authoring_set_order, field_b.set_order);
        assert_ne!(field_c.authoring_set_order, field_c.set_order);
    }

    #[test]
    fn apply_delta_set_field_lww_overwrite_retains_latest_authoring_rank() {
        // Last-writer-wins: two separate deltas write the same
        // (section_id, key_name) with different raw `set_order` values. The
        // surviving Field must carry the LATEST delta's authoring rank, not
        // the first writer's and not the cumulative cursor.
        let mut state = make_state();

        let first = OwnedDelta {
            is_diff: false,
            set_fields: vec![oodf_ffi::SetField {
                section_id: 3,
                key_name: "weight".to_string(),
                value_or_ext_hash: vec![0x01],
                ext_value_size: 0,
                ext_value_index: 0,
                set_order: 7,
            }],
            ..Default::default()
        };
        apply_delta(&mut state, Some(first)).unwrap();
        let after_first = state.fields.get(&(3, "weight".to_string())).unwrap();
        assert_eq!(after_first.authoring_set_order, 7);

        let second = OwnedDelta {
            is_diff: false,
            set_fields: vec![oodf_ffi::SetField {
                section_id: 3,
                key_name: "weight".to_string(),
                value_or_ext_hash: vec![0x02],
                ext_value_size: 0,
                ext_value_index: 0,
                set_order: 2,
            }],
            ..Default::default()
        };
        apply_delta(&mut state, Some(second)).unwrap();

        let field = state.fields.get(&(3, "weight".to_string())).unwrap();
        assert_eq!(
            field.authoring_set_order, 2,
            "surviving field must carry the LATEST writer's authoring rank (2), not the first's (7)"
        );
        assert!(
            matches!(field.value, FieldValue::Inline(ref v) if v == &[0x02]),
            "surviving field must carry the latest writer's value"
        );
    }

    #[test]
    fn apply_delta_set_field_stored_authoring_zero_is_preserved() {
        // A raw `set_order` of 0 must be stored as a genuine 0 in the
        // authoring field, distinguishable from a defaulted/absent value.
        // Pre-set `next_set_order` so the cumulative cursor is non-zero,
        // proving the stored 0 is the delta's real authoring rank and not
        // an artifact of the cursor also being 0.
        let mut state = make_state();
        state.next_set_order = 42;

        let delta = OwnedDelta {
            is_diff: false,
            set_fields: vec![oodf_ffi::SetField {
                section_id: 9,
                key_name: "flag".to_string(),
                value_or_ext_hash: vec![0x00],
                ext_value_size: 0,
                ext_value_index: 0,
                set_order: 0,
            }],
            ..Default::default()
        };
        apply_delta(&mut state, Some(delta)).unwrap();

        let field = state.fields.get(&(9, "flag".to_string())).unwrap();
        assert_eq!(
            field.authoring_set_order, 0,
            "a raw set_order of 0 must be stored as a genuine authoring 0"
        );
        // The cumulative cursor is non-zero (42), so the stored authoring 0
        // cannot be a coincidence of the cursor also being 0.
        assert_eq!(field.set_order, 42);
        assert_ne!(field.authoring_set_order, field.set_order);
    }

    // ── TimeSample application ───────────────────────────────────────────────

    #[test]
    fn apply_delta_time_sample_stores_value_with_version() {
        let mut state = make_state();

        let delta = OwnedDelta {
            is_diff: false,
            time_samples: vec![oodf_ffi::TimeSample {
                section_id: 7,
                time: 2.5,
                value_or_ext_hash: vec![0xAB, 0xCD],
                ext_value_size: 0,
                ext_value_index: 0,
            }],
            ..Default::default()
        };
        apply_delta(&mut state, Some(delta)).unwrap();

        let sample = state
            .time_samples
            .get(&(7, OrderedFloat(2.5)))
            .expect("time sample must exist");
        assert_eq!(sample.version, 1);
        assert!(matches!(sample.value, FieldValue::Inline(ref v) if v == &[0xAB, 0xCD]));
    }

    // ── Error-formatting regression ──────────────────────────────────────────

    #[test]
    fn delta_error_inconsistent_state_formats_correctly() {
        let err = DeltaError::InconsistentState {
            section_id: 42,
            operation: "apply_move_section (rename)",
        };
        let msg = err.to_string();
        assert!(msg.contains("42"));
        assert!(msg.contains("apply_move_section (rename)"));
    }

    // ── Ported inline tests (OwnedDelta fixtures) ────────────────────────────
    // These tests exercise every StructCommand variant, SetField, TimeSample,
    // cascade-delete paths, and the MoveSection edge cases.

    use oodf_ffi::{
        CommandCreateSection as OodfCreateSection, CommandDeleteSection as OodfDeleteSection,
        CommandDiffChangeSection as OodfDiffChangeSection, CommandMoveSection as OodfMoveSection,
        CommandReorderChildren as OodfReorderChildren,
    };

    fn make_section_versioned(id: u64, parent: u64, name: &str, first_version: u64) -> Section {
        Section {
            section_id: id,
            parent_id: parent,
            name: name.to_string(),
            section_type: 0,
            section_order: 0,
            first_version,
            last_version: first_version,
        }
    }

    fn make_three_level_tree() -> ObjectState {
        let mut state = make_state();
        state
            .sections
            .insert(1, make_section_versioned(1, 0, "Root", 1));
        state
            .sections
            .insert(2, make_section_versioned(2, 1, "Child", 2));
        state
            .sections
            .insert(3, make_section_versioned(3, 2, "Grandchild", 3));
        state.children.insert(0, vec![1]);
        state.children.insert(1, vec![2]);
        state.children.insert(2, vec![3]);
        state
    }

    fn delta_create_section(parent_id: u64, section_id: u64, name: &str) -> OwnedDelta {
        OwnedDelta {
            is_diff: false,
            struct_commands: vec![StructCommand::CreateSection(OodfCreateSection {
                parent_id,
                section_id,
                section_name: name.to_string(),
                section_type: 0,
            })],
            ..Default::default()
        }
    }

    fn delta_delete_section(section_id: u64) -> OwnedDelta {
        OwnedDelta {
            is_diff: false,
            struct_commands: vec![StructCommand::DeleteSection(OodfDeleteSection {
                parent_id: 0,
                section_id,
            })],
            ..Default::default()
        }
    }

    fn delta_move_section(
        section_id: u64,
        old_parent_id: u64,
        new_parent_id: u64,
        new_name: &str,
    ) -> OwnedDelta {
        OwnedDelta {
            is_diff: false,
            struct_commands: vec![StructCommand::MoveSection(OodfMoveSection {
                old_parent_id,
                new_parent_id,
                section_id,
                new_name: new_name.to_string(),
            })],
            ..Default::default()
        }
    }

    fn delta_diff_change_section(
        parent_id: u64,
        section_id: u64,
        name: &str,
        section_order: u32,
    ) -> OwnedDelta {
        OwnedDelta {
            is_diff: true,
            base_version: 1,
            struct_commands: vec![StructCommand::DiffChangeSection(OodfDiffChangeSection {
                parent_id,
                section_id,
                section_name: name.to_string(),
                section_type: 0,
                section_order,
            })],
            ..Default::default()
        }
    }

    fn children_of(state: &ObjectState, parent_id: u64) -> Vec<u64> {
        state.children.get(&parent_id).cloned().unwrap_or_default()
    }

    // ── SDD §8.4 semantics (explicit test) ────────────────────────────────────

    /// SDD §8.4: empty delta (None) = no content changes, but sequence_num
    /// still increments and only the SetObjectHeader mutation is emitted.
    #[test]
    fn apply_delta_none_preserves_sdd_8_3_empty_semantics() {
        let mut state = make_state();
        state.sections.insert(1, make_section(1, 0, "Keep"));
        state.children.insert(0, vec![1]);
        state.fields.insert(
            (1, "k".to_string()),
            Field {
                section_id: 1,
                key_name: "k".to_string(),
                value: FieldValue::Inline(vec![1]),
                set_order: 0,
                authoring_set_order: 0,
                version: 1,
            },
        );
        state.time_samples.insert(
            (1, OrderedFloat(0.5)),
            TimeSample {
                section_id: 1,
                time: OrderedFloat(0.5),
                value: FieldValue::Inline(vec![2]),
                version: 1,
            },
        );
        let before_seq = state.sequence_num;

        let mutations = apply_delta(&mut state, None).unwrap();

        assert_eq!(state.sequence_num, before_seq + 1);
        assert!(state.sections.contains_key(&1), "section tree preserved");
        assert!(state.fields.contains_key(&(1, "k".to_string())));
        assert!(state.time_samples.contains_key(&(1, OrderedFloat(0.5))));
        assert_eq!(mutations.len(), 1);
        assert!(matches!(
            mutations[0],
            StorageMutation::SetObjectHeader { .. }
        ));
    }

    // ── isDiff=false passthrough and seq increments ──────────────────────────

    #[test]
    fn apply_delta_is_diff_false_accepted_and_processed() {
        let mut state = make_state();
        let delta = OwnedDelta {
            is_diff: false,
            base_version: 0,
            ..Default::default()
        };
        let mutations = apply_delta(&mut state, Some(delta)).unwrap();
        assert_eq!(state.sequence_num, 1);
        assert!(!mutations.is_empty());
    }

    #[test]
    fn apply_delta_seq_increments_fields_only_delta() {
        let mut state = make_state();
        assert_eq!(state.sequence_num, 0);

        let delta = OwnedDelta {
            is_diff: false,
            set_fields: vec![oodf_ffi::SetField {
                section_id: 1,
                key_name: "x".to_string(),
                value_or_ext_hash: vec![1u8],
                ext_value_size: 0,
                ext_value_index: 0,
                set_order: 0,
            }],
            ..Default::default()
        };
        apply_delta(&mut state, Some(delta)).unwrap();
        assert_eq!(
            state.sequence_num, 1,
            "seq must increment on fields-only delta"
        );
    }

    #[test]
    fn apply_delta_seq_increments_commands_only_delta() {
        let mut state = make_state();
        assert_eq!(state.sequence_num, 0);
        apply_delta(&mut state, Some(delta_create_section(0, 300, "PrimA"))).unwrap();
        assert_eq!(
            state.sequence_num, 1,
            "seq must increment on commands-only delta"
        );
    }

    #[test]
    fn apply_delta_is_diff_false_does_not_store_source_format() {
        let mut state = make_state();
        let delta = OwnedDelta {
            is_diff: false,
            base_version: 0,
            source_format: "usda".to_string(),
            ..Default::default()
        };
        apply_delta(&mut state, Some(delta)).unwrap();
        assert_eq!(
            state.source_format, "",
            "isDiff=false must not store sourceFormat"
        );
    }

    #[test]
    fn apply_delta_source_format_stored_only_on_reset_not_on_incremental() {
        let mut state = make_state();

        // Reset with sourceFormat=usda.
        let reset = OwnedDelta {
            is_diff: true,
            base_version: 0,
            source_format: "usda".to_string(),
            ..Default::default()
        };
        apply_delta(&mut state, Some(reset)).unwrap();
        assert_eq!(
            state.source_format, "usda",
            "sourceFormat must be stored on reset"
        );

        // Incremental (is_diff=true, base_version=1) with a different sourceFormat.
        let incr = OwnedDelta {
            is_diff: true,
            base_version: 1,
            source_format: "usdz".to_string(),
            ..Default::default()
        };
        apply_delta(&mut state, Some(incr)).unwrap();
        assert_eq!(
            state.source_format, "usda",
            "incremental delta must not overwrite sourceFormat"
        );
    }

    // ── Reset path (legacy variant — reset with empty source_format) ─────────

    #[test]
    fn apply_delta_reset_erases_all_state_when_no_source_format() {
        let mut state = make_state();
        state.sections.insert(10, make_section(10, 0, "Root"));
        state.children.insert(0, vec![10]);
        state.fields.insert(
            (10, "k".to_string()),
            Field {
                section_id: 10,
                key_name: "k".to_string(),
                value: FieldValue::Inline(vec![1]),
                set_order: 0,
                authoring_set_order: 0,
                version: 1,
            },
        );
        state.time_samples.insert(
            (10, OrderedFloat(1.0)),
            TimeSample {
                section_id: 10,
                time: OrderedFloat(1.0),
                value: FieldValue::Inline(vec![2]),
                version: 1,
            },
        );

        let reset = OwnedDelta {
            is_diff: true,
            base_version: 0,
            ..Default::default()
        };
        apply_delta(&mut state, Some(reset)).unwrap();

        assert!(state.sections.is_empty(), "sections erased on reset");
        assert!(state.fields.is_empty(), "fields erased on reset");
        assert!(
            state.time_samples.is_empty(),
            "time_samples erased on reset"
        );
        assert_eq!(state.first_version, 1, "first_version = new sequence_num");
        assert_eq!(state.next_set_order, 0, "nextSetOrder reset to 0");
    }

    // ── CommandDiffChangeSection ─────────────────────────────────────────────

    #[test]
    fn apply_delta_diff_change_section_updates_existing_section() {
        let mut state = make_state();
        state.sections.insert(100, make_section(100, 0, "OldName"));
        state.children.insert(0, vec![100]);

        let delta = OwnedDelta {
            is_diff: true,
            base_version: 1,
            struct_commands: vec![StructCommand::DiffChangeSection(OodfDiffChangeSection {
                parent_id: 0,
                section_id: 100,
                section_name: "NewName".to_string(),
                section_type: 2,
                section_order: 10,
            })],
            ..Default::default()
        };
        apply_delta(&mut state, Some(delta)).unwrap();

        let sec = state.sections.get(&100).expect("section 100 must exist");
        assert_eq!(sec.name, "NewName");
        assert_eq!(sec.section_type, 2);
        assert_eq!(
            sec.first_version, 1,
            "first_version must not change on update"
        );
    }

    // ── CommandCreateSection ─────────────────────────────────────────────────

    #[test]
    fn apply_delta_create_section_ignored_if_parent_missing() {
        let mut state = make_state();
        apply_delta(&mut state, Some(delta_create_section(999, 201, "Orphan"))).unwrap();
        assert!(
            !state.sections.contains_key(&201),
            "section with missing parent must be ignored"
        );
    }

    #[test]
    fn apply_delta_create_section_cascades_delete_on_name_conflict() {
        let mut state = make_state();
        state.sections.insert(50, make_section(50, 0, "Conflict"));
        state.children.insert(0, vec![50]);
        apply_delta(&mut state, Some(delta_create_section(0, 51, "Conflict"))).unwrap();

        assert!(
            !state.sections.contains_key(&50),
            "conflicting section must be deleted"
        );
        assert!(
            state.sections.contains_key(&51),
            "new section must be created"
        );
    }

    // ── CommandDeleteSection ─────────────────────────────────────────────────

    #[test]
    fn apply_delta_delete_section_removes_section_and_generates_tombstone() {
        let mut state = make_state();
        state.sections.insert(77, make_section(77, 0, "ToDelete"));
        state.children.insert(0, vec![77]);

        let mutations = apply_delta(&mut state, Some(delta_delete_section(77))).unwrap();

        assert!(!state.sections.contains_key(&77), "section must be deleted");
        let has_tombstone = mutations
            .iter()
            .any(|m| matches!(m, StorageMutation::CreateTombstone { section_id: 77, .. }));
        assert!(has_tombstone, "tombstone mutation must be emitted");
    }

    #[test]
    fn apply_delta_delete_section_cascades_to_descendants() {
        let mut state = make_state();
        state.sections.insert(10, make_section(10, 0, "Parent"));
        state.sections.insert(11, make_section(11, 10, "Child"));
        state.children.insert(0, vec![10]);
        state.children.insert(10, vec![11]);

        apply_delta(&mut state, Some(delta_delete_section(10))).unwrap();

        assert!(!state.sections.contains_key(&10), "parent must be deleted");
        assert!(
            !state.sections.contains_key(&11),
            "child must be cascade-deleted"
        );
    }

    // ── CommandMoveSection ───────────────────────────────────────────────────

    #[test]
    fn apply_delta_move_section_missing_new_parent_deletes_section() {
        let mut state = make_state();
        state.sections.insert(30, make_section(30, 0, "MoveSect"));
        state.children.insert(0, vec![30]);

        apply_delta(&mut state, Some(delta_move_section(30, 0, 999, "MoveSect"))).unwrap();

        assert!(
            !state.sections.contains_key(&30),
            "section must be deleted when newParent missing"
        );
    }

    // ── CommandReorderChildren ───────────────────────────────────────────────

    #[test]
    fn apply_delta_reorder_children_ignores_children_list_id() {
        // Server ignores the incoming children_list_id (SDD §9.1.4e).
        let mut state = make_state();
        // The ids must be parent 10's listed children for the reorder to take
        // effect (any other id is ignored); naming both, a complete list,
        // installs exactly as sent.
        state.sections.insert(10, make_section(10, 0, "Parent"));
        state.sections.insert(1, make_section(1, 10, "A"));
        state.sections.insert(2, make_section(2, 10, "B"));
        state.children.insert(0, vec![10]);
        state.children.insert(10, vec![1, 2]);

        let delta = OwnedDelta {
            is_diff: false,
            struct_commands: vec![StructCommand::ReorderChildren(OodfReorderChildren {
                section_id: 10,
                children_list_id: 42, // non-zero — must be ignored
                children_list: vec![2u64, 1],
            })],
            ..Default::default()
        };
        let mutations = apply_delta(&mut state, Some(delta)).unwrap();

        let set_children = mutations
            .iter()
            .find(|m| matches!(m, StorageMutation::SetChildren { section_id: 10, .. }));
        assert!(
            set_children.is_some(),
            "SetChildren mutation must be emitted"
        );
        // The incoming children_list_id (42) is ignored: the reorder still
        // applies from the command's children_list, emitting [2, 1].
        if let Some(StorageMutation::SetChildren { children, .. }) = set_children {
            assert_eq!(
                *children,
                vec![2u64, 1],
                "reorder must apply the command's children_list, ignoring children_list_id"
            );
        }
    }

    // ── Field application ────────────────────────────────────────────────────

    #[test]
    fn apply_delta_set_field_stores_external_value() {
        let mut state = make_state();
        let hash_bytes: Vec<u8> = (0u8..32).collect();

        let delta = OwnedDelta {
            is_diff: false,
            set_fields: vec![oodf_ffi::SetField {
                section_id: 2,
                key_name: "texture".to_string(),
                value_or_ext_hash: hash_bytes,
                ext_value_size: 1_048_576,
                ext_value_index: 1,
                set_order: 0,
            }],
            ..Default::default()
        };
        apply_delta(&mut state, Some(delta)).unwrap();

        let field = state
            .fields
            .get(&(2, "texture".to_string()))
            .expect("field must exist");
        assert!(
            matches!(
                field.value,
                FieldValue::External {
                    size: 1_048_576,
                    ..
                }
            ),
            "field value must be External"
        );
    }

    #[test]
    fn apply_delta_two_fields_in_same_delta_get_monotonic_set_orders() {
        let mut state = make_state();

        let delta = OwnedDelta {
            is_diff: false,
            set_fields: vec![
                oodf_ffi::SetField {
                    section_id: 1,
                    key_name: "alpha".to_string(),
                    value_or_ext_hash: vec![1u8],
                    ext_value_size: 0,
                    ext_value_index: 0,
                    set_order: 0,
                },
                oodf_ffi::SetField {
                    section_id: 1,
                    key_name: "beta".to_string(),
                    value_or_ext_hash: vec![2u8],
                    ext_value_size: 0,
                    ext_value_index: 0,
                    set_order: 0,
                },
            ],
            ..Default::default()
        };
        apply_delta(&mut state, Some(delta)).unwrap();

        let field_a = state.fields.get(&(1, "alpha".to_string())).unwrap();
        let field_b = state.fields.get(&(1, "beta".to_string())).unwrap();
        assert_eq!(field_a.set_order, 0, "first field set_order must be 0");
        assert_eq!(field_b.set_order, 1, "second field set_order must be 1");
        assert!(
            field_b.set_order > field_a.set_order,
            "set_order must be monotonically increasing within a delta"
        );
        assert_eq!(state.next_set_order, 2);
    }

    #[test]
    fn apply_delta_set_field_overwrites_existing_field_same_key() {
        let mut state = make_state();
        state.fields.insert(
            (5, "opacity".to_string()),
            Field {
                section_id: 5,
                key_name: "opacity".to_string(),
                value: FieldValue::Inline(vec![0x3F]),
                set_order: 0,
                authoring_set_order: 0,
                version: 1,
            },
        );
        state.sequence_num = 1;

        let delta = OwnedDelta {
            is_diff: true,
            base_version: 1,
            set_fields: vec![oodf_ffi::SetField {
                section_id: 5,
                key_name: "opacity".to_string(),
                value_or_ext_hash: vec![0x7F],
                ext_value_size: 0,
                ext_value_index: 0,
                set_order: 0,
            }],
            ..Default::default()
        };
        apply_delta(&mut state, Some(delta)).unwrap();

        let field = state.fields.get(&(5, "opacity".to_string())).unwrap();
        assert!(
            matches!(field.value, FieldValue::Inline(ref v) if v == &[0x7F]),
            "field value must be updated"
        );
        assert_eq!(
            field.version, 2,
            "version must be updated to current sequence_num"
        );
    }

    #[test]
    fn apply_delta_field_version_equals_sequence_num_after_apply() {
        let mut state = make_state();
        state.sequence_num = 5;

        let delta = OwnedDelta {
            is_diff: false,
            set_fields: vec![oodf_ffi::SetField {
                section_id: 10,
                key_name: "myField".to_string(),
                value_or_ext_hash: vec![0xAA],
                ext_value_size: 0,
                ext_value_index: 0,
                set_order: 0,
            }],
            ..Default::default()
        };
        apply_delta(&mut state, Some(delta)).unwrap();

        let field = state.fields.get(&(10, "myField".to_string())).unwrap();
        assert_eq!(
            field.version, state.sequence_num,
            "field.version must equal state.sequence_num after apply"
        );
    }

    #[test]
    fn apply_delta_next_set_order_computed_across_multiple_deltas() {
        let mut state = make_state();

        // First delta: setOrder=5 → stored=5, nextSetOrder=6.
        let d1 = OwnedDelta {
            is_diff: false,
            set_fields: vec![oodf_ffi::SetField {
                section_id: 1,
                key_name: "a".to_string(),
                value_or_ext_hash: vec![1],
                ext_value_size: 0,
                ext_value_index: 0,
                set_order: 5,
            }],
            ..Default::default()
        };
        apply_delta(&mut state, Some(d1)).unwrap();
        assert_eq!(state.next_set_order, 6);

        // Second delta: setOrder=2 → stored = 6+2 = 8, nextSetOrder = max(6, 9) = 9.
        let d2 = OwnedDelta {
            is_diff: false,
            set_fields: vec![oodf_ffi::SetField {
                section_id: 1,
                key_name: "b".to_string(),
                value_or_ext_hash: vec![2],
                ext_value_size: 0,
                ext_value_index: 0,
                set_order: 2,
            }],
            ..Default::default()
        };
        apply_delta(&mut state, Some(d2)).unwrap();
        assert_eq!(state.next_set_order, 9);
    }

    // ── Time sample application ──────────────────────────────────────────────

    #[test]
    fn apply_delta_time_sample_upsert_by_section_id_and_time() {
        let mut state = make_state();
        state.sequence_num = 1;

        state.time_samples.insert(
            (1, OrderedFloat(1.0)),
            TimeSample {
                section_id: 1,
                time: OrderedFloat(1.0),
                value: FieldValue::Inline(vec![0x01]),
                version: 1,
            },
        );

        let delta = OwnedDelta {
            is_diff: true,
            base_version: 1,
            time_samples: vec![oodf_ffi::TimeSample {
                section_id: 1,
                time: 1.0,
                value_or_ext_hash: vec![0x02],
                ext_value_size: 0,
                ext_value_index: 0,
            }],
            ..Default::default()
        };
        apply_delta(&mut state, Some(delta)).unwrap();

        let sample = state.time_samples.get(&(1, OrderedFloat(1.0))).unwrap();
        assert_eq!(sample.version, 2, "version must update to new sequence_num");
        assert!(matches!(sample.value, FieldValue::Inline(ref v) if v == &[0x02]));
    }

    // ── Cascade delete tests (t3.1it.9v.nu.kq) ───────────────────────────────

    #[test]
    fn cascade_delete_single_section_emits_all_required_mutation_types() {
        let mut state = make_state();
        state.sections.insert(10, make_section(10, 0, "Leaf"));
        state.children.insert(0, vec![10]);
        state.fields.insert(
            (10, "color".to_string()),
            Field {
                section_id: 10,
                key_name: "color".to_string(),
                value: FieldValue::Inline(vec![0xFF]),
                set_order: 0,
                authoring_set_order: 0,
                version: 1,
            },
        );
        state.time_samples.insert(
            (10, OrderedFloat(1.0)),
            TimeSample {
                section_id: 10,
                time: OrderedFloat(1.0),
                value: FieldValue::Inline(vec![0xAB]),
                version: 1,
            },
        );

        let mutations = apply_delta(&mut state, Some(delta_delete_section(10))).unwrap();

        let has_delete_section = mutations
            .iter()
            .any(|m| matches!(m, StorageMutation::DeleteSection { section_id: 10, .. }));
        let has_tombstone = mutations
            .iter()
            .any(|m| matches!(m, StorageMutation::CreateTombstone { section_id: 10, .. }));
        let has_delete_children = mutations
            .iter()
            .any(|m| matches!(m, StorageMutation::DeleteChildrenList { section_id: 10 }));
        let has_delete_field = mutations.iter().any(|m| {
            matches!(
                m,
                StorageMutation::DeleteField {
                    section_id: 10,
                    key_name,
                } if key_name == "color"
            )
        });
        let has_delete_ts = mutations.iter().any(|m| {
            matches!(
                m,
                StorageMutation::DeleteTimeSample {
                    section_id: 10,
                    time,
                } if *time == OrderedFloat(1.0)
            )
        });

        assert!(has_delete_section);
        assert!(has_tombstone);
        assert!(has_delete_children);
        assert!(has_delete_field);
        assert!(has_delete_ts);
    }

    #[test]
    fn cascade_delete_multi_level_tree_every_descendant_appears_exactly_once() {
        let mut state = make_three_level_tree();
        let mutations = apply_delta(&mut state, Some(delta_delete_section(1))).unwrap();

        for deleted_id in [1u64, 2, 3] {
            let delete_count = mutations
                .iter()
                .filter(|m| matches!(m, StorageMutation::DeleteSection { section_id, .. } if *section_id == deleted_id))
                .count();
            let tombstone_count = mutations
                .iter()
                .filter(|m| matches!(m, StorageMutation::CreateTombstone { section_id, .. } if *section_id == deleted_id))
                .count();
            assert_eq!(delete_count, 1);
            assert_eq!(tombstone_count, 1);
        }
    }

    #[test]
    fn cascade_delete_multi_level_tree_no_sections_remain_in_live_or_tombstone_maps() {
        let mut state = make_three_level_tree();
        apply_delta(&mut state, Some(delta_delete_section(1))).unwrap();

        for deleted_id in [1u64, 2, 3] {
            assert!(!state.sections.contains_key(&deleted_id));
            assert!(!state.children.contains_key(&deleted_id));
        }
    }

    #[test]
    fn cascade_delete_tombstone_first_ver_equals_section_original_first_version() {
        let mut state = make_state();
        state
            .sections
            .insert(10, make_section_versioned(10, 0, "Old", 5));
        state.children.insert(0, vec![10]);
        state.sequence_num = 7;

        let mutations = apply_delta(&mut state, Some(delta_delete_section(10))).unwrap();

        let tombstone = mutations
            .iter()
            .find(|m| matches!(m, StorageMutation::CreateTombstone { section_id: 10, .. }))
            .expect("tombstone must be emitted");
        if let StorageMutation::CreateTombstone {
            first_version,
            last_version,
            ..
        } = tombstone
        {
            assert_eq!(*first_version, 5);
            assert_eq!(*last_version, 8);
        }
    }

    #[test]
    fn cascade_delete_fields_and_time_samples_of_deleted_section_included_in_mutations() {
        let mut state = make_state();
        state.sections.insert(20, make_section(20, 0, "WithData"));
        state.children.insert(0, vec![20]);
        state.fields.insert(
            (20, "k1".to_string()),
            Field {
                section_id: 20,
                key_name: "k1".to_string(),
                value: FieldValue::Inline(vec![1]),
                set_order: 0,
                authoring_set_order: 0,
                version: 1,
            },
        );
        state.fields.insert(
            (20, "k2".to_string()),
            Field {
                section_id: 20,
                key_name: "k2".to_string(),
                value: FieldValue::Inline(vec![2]),
                set_order: 1,
                authoring_set_order: 1,
                version: 1,
            },
        );
        state.time_samples.insert(
            (20, OrderedFloat(0.5)),
            TimeSample {
                section_id: 20,
                time: OrderedFloat(0.5),
                value: FieldValue::Inline(vec![3]),
                version: 1,
            },
        );
        state.time_samples.insert(
            (20, OrderedFloat(1.5)),
            TimeSample {
                section_id: 20,
                time: OrderedFloat(1.5),
                value: FieldValue::Inline(vec![4]),
                version: 1,
            },
        );

        let mutations = apply_delta(&mut state, Some(delta_delete_section(20))).unwrap();

        let deleted_fields: Vec<&str> = mutations
            .iter()
            .filter_map(|m| {
                if let StorageMutation::DeleteField {
                    section_id: 20,
                    key_name,
                } = m
                {
                    Some(key_name.as_str())
                } else {
                    None
                }
            })
            .collect();
        assert_eq!(deleted_fields.len(), 2);
        assert!(deleted_fields.contains(&"k1"));
        assert!(deleted_fields.contains(&"k2"));

        let deleted_ts_times: Vec<OrderedFloat<f64>> = mutations
            .iter()
            .filter_map(|m| {
                if let StorageMutation::DeleteTimeSample {
                    section_id: 20,
                    time,
                } = m
                {
                    Some(*time)
                } else {
                    None
                }
            })
            .collect();
        assert_eq!(deleted_ts_times.len(), 2);
        assert!(deleted_ts_times.contains(&OrderedFloat(0.5)));
        assert!(deleted_ts_times.contains(&OrderedFloat(1.5)));
    }

    #[test]
    fn cascade_delete_section_with_no_fields_or_samples_still_produces_tombstone() {
        let mut state = make_state();
        state.sections.insert(30, make_section(30, 0, "Empty"));
        state.children.insert(0, vec![30]);

        let mutations = apply_delta(&mut state, Some(delta_delete_section(30))).unwrap();

        let has_tombstone = mutations
            .iter()
            .any(|m| matches!(m, StorageMutation::CreateTombstone { section_id: 30, .. }));
        assert!(has_tombstone);
        assert!(!state.sections.contains_key(&30));
    }

    // Regression for b.2w7: cyclic children map must not infinite-loop.
    #[test]
    fn cascade_delete_on_cyclic_children_terminates() {
        let mut state = make_state();
        state.sections.insert(2, make_section(2, 0, "A"));
        state.sections.insert(3, make_section(3, 2, "B"));
        state.children.insert(0, vec![2]);
        state.children.insert(2, vec![3]);
        state.children.insert(3, vec![2]);

        let result = apply_delta(&mut state, Some(delta_delete_section(2))).unwrap();

        let has_tombstone_2 = result
            .iter()
            .any(|m| matches!(m, StorageMutation::CreateTombstone { section_id: 2, .. }));
        let has_tombstone_3 = result
            .iter()
            .any(|m| matches!(m, StorageMutation::CreateTombstone { section_id: 3, .. }));
        assert!(has_tombstone_2);
        assert!(has_tombstone_3);
        assert!(!state.sections.contains_key(&2));
        assert!(!state.sections.contains_key(&3));
    }

    // ── CommandMoveSection edge cases (t3.1it.9v.nu.85) ──────────────────────

    #[test]
    fn move_section_case1_rename_same_parent_updates_name_only() {
        let mut state = make_state();
        state
            .sections
            .insert(10, make_section_versioned(10, 5, "OldName", 1));
        state.sections.insert(5, make_section(5, 0, "Parent"));
        state.children.insert(0, vec![5]);
        state.children.insert(5, vec![10]);

        apply_delta(&mut state, Some(delta_move_section(10, 5, 5, "NewName"))).unwrap();

        let sec = &state.sections[&10];
        assert_eq!(sec.name, "NewName");
        assert_eq!(sec.parent_id, 5);
        assert!(state.sections.contains_key(&10));
    }

    #[test]
    fn move_section_case2_missing_new_parent_cascade_deletes_section() {
        let mut state = make_state();
        state.sections.insert(10, make_section(10, 0, "ToMove"));
        state.children.insert(0, vec![10]);

        apply_delta(&mut state, Some(delta_move_section(10, 0, 999, "ToMove"))).unwrap();

        assert!(!state.sections.contains_key(&10));
    }

    #[test]
    fn move_section_case2_missing_new_parent_emits_tombstone_for_moved_section() {
        let mut state = make_state();
        state
            .sections
            .insert(10, make_section_versioned(10, 0, "ToMove", 3));
        state.children.insert(0, vec![10]);
        state.sequence_num = 4;

        let mutations =
            apply_delta(&mut state, Some(delta_move_section(10, 0, 999, "ToMove"))).unwrap();

        let tombstone = mutations
            .iter()
            .find(|m| matches!(m, StorageMutation::CreateTombstone { section_id: 10, .. }));
        assert!(tombstone.is_some());
    }

    #[test]
    fn move_section_case3_cycle_detection_deletes_moved_section() {
        let mut state = make_state();
        state.sections.insert(1, make_section(1, 0, "A"));
        state.sections.insert(2, make_section(2, 1, "B"));
        state.sections.insert(3, make_section(3, 2, "C"));
        state.children.insert(0, vec![1]);
        state.children.insert(1, vec![2]);
        state.children.insert(2, vec![3]);

        apply_delta(&mut state, Some(delta_move_section(1, 0, 3, "A"))).unwrap();

        assert!(!state.sections.contains_key(&1));
    }

    #[test]
    fn move_section_case3_cycle_detection_also_deletes_descendants_of_moved_section() {
        let mut state = make_state();
        state.sections.insert(1, make_section(1, 0, "A"));
        state.sections.insert(2, make_section(2, 1, "B"));
        state.sections.insert(3, make_section(3, 2, "C"));
        state.children.insert(0, vec![1]);
        state.children.insert(1, vec![2]);
        state.children.insert(2, vec![3]);

        apply_delta(&mut state, Some(delta_move_section(1, 0, 3, "A"))).unwrap();

        assert!(!state.sections.contains_key(&1));
        assert!(!state.sections.contains_key(&2));
        assert!(!state.sections.contains_key(&3));
    }

    #[test]
    fn move_section_case4_ancestor_conflict_deletes_moved_section() {
        let mut state = make_state();
        state
            .sections
            .insert(1, make_section_versioned(1, 0, "GrandP", 1));
        state
            .sections
            .insert(2, make_section_versioned(2, 1, "Parent", 1));
        state.children.insert(0, vec![1]);
        state.children.insert(1, vec![2]);

        apply_delta(&mut state, Some(delta_move_section(2, 1, 0, "GrandP"))).unwrap();

        assert!(!state.sections.contains_key(&2));
        assert!(state.sections.contains_key(&1));
    }

    #[test]
    fn move_section_case5_sibling_conflict_deletes_conflicting_and_moves_section() {
        let mut state = make_state();
        state.sections.insert(1, make_section(1, 0, "Target"));
        state
            .sections
            .insert(3, make_section(3, 0, "ParentOfMover"));
        state.sections.insert(2, make_section(2, 3, "Mover"));
        state.children.insert(0, vec![1, 3]);
        state.children.insert(3, vec![2]);

        apply_delta(&mut state, Some(delta_move_section(2, 3, 0, "Target"))).unwrap();

        assert!(!state.sections.contains_key(&1));
        assert!(state.sections.contains_key(&2));
        assert_eq!(state.sections[&2].parent_id, 0);
        assert_eq!(state.sections[&2].name, "Target");
    }

    #[test]
    fn move_section_case5_sibling_conflict_tombstone_emitted_for_conflicting_section() {
        let mut state = make_state();
        state
            .sections
            .insert(1, make_section_versioned(1, 0, "Target", 2));
        state
            .sections
            .insert(3, make_section(3, 0, "ParentOfMover"));
        state.sections.insert(2, make_section(2, 3, "Mover"));
        state.children.insert(0, vec![1, 3]);
        state.children.insert(3, vec![2]);

        let mutations =
            apply_delta(&mut state, Some(delta_move_section(2, 3, 0, "Target"))).unwrap();

        let tombstone = mutations
            .iter()
            .find(|m| matches!(m, StorageMutation::CreateTombstone { section_id: 1, .. }));
        assert!(tombstone.is_some());
    }

    #[test]
    fn move_section_case6_clean_move_updates_parent_and_children_lists() {
        let mut state = make_state();
        state.sections.insert(1, make_section(1, 0, "Parent1"));
        state.sections.insert(2, make_section(2, 0, "Parent2"));
        state.sections.insert(3, make_section(3, 1, "Child"));
        state.children.insert(0, vec![1, 2]);
        state.children.insert(1, vec![3]);
        state.children.insert(2, vec![]);

        apply_delta(&mut state, Some(delta_move_section(3, 1, 2, "Child"))).unwrap();

        assert_eq!(state.sections[&3].parent_id, 2);
        assert!(!state
            .children
            .get(&1)
            .map(|v| v.contains(&3))
            .unwrap_or(false));
        assert!(state
            .children
            .get(&2)
            .map(|v| v.contains(&3))
            .unwrap_or(false));
    }

    #[test]
    fn move_section_case6_clean_move_emits_set_section_mutation() {
        let mut state = make_state();
        state.sections.insert(1, make_section(1, 0, "Parent1"));
        state.sections.insert(2, make_section(2, 0, "Parent2"));
        state.sections.insert(3, make_section(3, 1, "Child"));
        state.children.insert(0, vec![1, 2]);
        state.children.insert(1, vec![3]);
        state.children.insert(2, vec![]);

        let mutations = apply_delta(
            &mut state,
            Some(delta_move_section(3, 1, 2, "RenamedChild")),
        )
        .unwrap();

        let set_section = mutations.iter().find(
            |m| matches!(m, StorageMutation::SetSection { section, .. } if section.section_id == 3),
        );
        assert!(set_section.is_some());
        if let Some(StorageMutation::SetSection { section, .. }) = set_section {
            assert_eq!(section.parent_id, 2);
            assert_eq!(section.name, "RenamedChild");
        }
    }

    #[test]
    fn move_section_nonexistent_section_is_silently_ignored() {
        let mut state = make_state();
        let mutations =
            apply_delta(&mut state, Some(delta_move_section(99, 0, 1, "Name"))).unwrap();

        assert_eq!(mutations.len(), 1);
        assert!(matches!(
            mutations[0],
            StorageMutation::SetObjectHeader { .. }
        ));
    }

    // ── CommandCreateSection name conflict cascade (t3.1it.9v.nu.dg) ─────────

    #[test]
    fn create_section_normal_creates_section_under_parent() {
        let mut state = make_state();
        state.sections.insert(1, make_section(1, 0, "Parent"));
        state.children.insert(0, vec![1]);

        apply_delta(&mut state, Some(delta_create_section(1, 10, "NewChild"))).unwrap();

        assert!(state.sections.contains_key(&10));
        assert_eq!(state.sections[&10].name, "NewChild");
        assert_eq!(state.sections[&10].parent_id, 1);
        assert!(state
            .children
            .get(&1)
            .map(|v| v.contains(&10))
            .unwrap_or(false));
    }

    #[test]
    fn create_section_normal_fresh_first_version_equals_current_sequence_num() {
        let mut state = make_state();
        state.sequence_num = 7;
        state.sections.insert(1, make_section(1, 0, "Parent"));
        state.children.insert(0, vec![1]);

        apply_delta(&mut state, Some(delta_create_section(1, 10, "NewChild"))).unwrap();

        assert_eq!(state.sections[&10].first_version, 8);
    }

    #[test]
    fn create_section_missing_parent_silently_ignored() {
        let mut state = make_state();
        apply_delta(&mut state, Some(delta_create_section(999, 50, "Orphan"))).unwrap();
        assert!(!state.sections.contains_key(&50));
    }

    #[test]
    fn create_section_name_conflict_cascade_deletes_existing_sibling() {
        let mut state = make_state();
        state
            .sections
            .insert(10, make_section_versioned(10, 0, "Dupe", 1));
        state.children.insert(0, vec![10]);

        apply_delta(&mut state, Some(delta_create_section(0, 11, "Dupe"))).unwrap();

        assert!(!state.sections.contains_key(&10));
        assert!(state.sections.contains_key(&11));
    }

    #[test]
    fn create_section_name_conflict_tombstone_emitted_for_deleted_sibling() {
        let mut state = make_state();
        state
            .sections
            .insert(10, make_section_versioned(10, 0, "Dupe", 3));
        state.children.insert(0, vec![10]);

        let mutations = apply_delta(&mut state, Some(delta_create_section(0, 11, "Dupe"))).unwrap();

        let tombstone = mutations
            .iter()
            .find(|m| matches!(m, StorageMutation::CreateTombstone { section_id: 10, .. }));
        assert!(tombstone.is_some());
        if let Some(StorageMutation::CreateTombstone {
            first_version,
            section_id: 10,
            ..
        }) = tombstone
        {
            assert_eq!(*first_version, 3);
        }
    }

    #[test]
    fn create_section_name_conflict_cascades_to_descendants_of_conflicting_section() {
        let mut state = make_state();
        state
            .sections
            .insert(10, make_section_versioned(10, 0, "Dupe", 1));
        state
            .sections
            .insert(20, make_section_versioned(20, 10, "ChildOfDupe", 1));
        state.children.insert(0, vec![10]);
        state.children.insert(10, vec![20]);

        let mutations = apply_delta(&mut state, Some(delta_create_section(0, 11, "Dupe"))).unwrap();

        assert!(!state.sections.contains_key(&10));
        assert!(!state.sections.contains_key(&20));
        assert!(state.sections.contains_key(&11));

        let tombstone_child = mutations
            .iter()
            .find(|m| matches!(m, StorageMutation::CreateTombstone { section_id: 20, .. }));
        assert!(tombstone_child.is_some());
    }

    #[test]
    fn create_section_name_conflict_new_section_gets_fresh_first_version() {
        let mut state = make_state();
        state
            .sections
            .insert(10, make_section_versioned(10, 0, "Dupe", 5));
        state.children.insert(0, vec![10]);
        state.sequence_num = 9;

        apply_delta(&mut state, Some(delta_create_section(0, 11, "Dupe"))).unwrap();

        assert_eq!(state.sections[&11].first_version, 10);
    }

    #[test]
    fn create_section_at_root_parent_zero_always_allowed() {
        let mut state = make_state();
        apply_delta(&mut state, Some(delta_create_section(0, 5, "TopLevel"))).unwrap();
        assert!(state.sections.contains_key(&5));
    }

    // ── Children list sort deterministic tests (t3.1it.9v.2h.4y) ─────────────

    #[test]
    fn rebuild_children_list_empty_siblings_yields_empty_result() {
        let mut state = make_state();
        state.sections.insert(1, make_section(1, 0, "Parent"));
        state.children.insert(0, vec![1]);

        let list = children_of(&state, 1);
        assert!(list.is_empty());
    }

    #[test]
    fn rebuild_children_list_distinct_section_orders_sorted_ascending() {
        let mut state = make_state();
        state.sections.insert(1, make_section(1, 0, "Parent"));
        state.children.insert(0, vec![1]);

        for (id, name, order) in [(10u64, "C", 30u32), (11, "B", 20), (12, "A", 10)] {
            apply_delta(
                &mut state,
                Some(delta_diff_change_section(1, id, name, order)),
            )
            .unwrap();
        }

        let list = children_of(&state, 1);
        assert_eq!(list.len(), 3);
        let orders: Vec<u32> = list
            .iter()
            .map(|&cid| state.sections[&cid].section_order)
            .collect();
        assert_eq!(orders, vec![10, 20, 30]);
    }

    #[test]
    fn rebuild_children_list_equal_section_orders_preserved_insertion_order() {
        let mut state = make_state();
        state.sections.insert(1, make_section(1, 0, "Parent"));
        state.children.insert(0, vec![1]);

        let ids = [10u64, 11, 12];
        for id in ids {
            let name = format!("Child{id}");
            apply_delta(&mut state, Some(delta_diff_change_section(1, id, &name, 0))).unwrap();
        }

        let list = children_of(&state, 1);
        assert_eq!(list.len(), 3);
        assert_eq!(list, vec![10u64, 11, 12]);
    }

    #[test]
    fn rebuild_children_list_large_list_100_children_sorted_correctly() {
        let mut state = make_state();
        state.sections.insert(1, make_section(1, 0, "Parent"));
        state.children.insert(0, vec![1]);

        // Section ids 101..=200 so the loop never overlaps with `parent_id =
        // 1`. Pre-b.39x the loop walked 100..=1 inclusive, which made the
        // last iteration self-parent section_id=1 onto itself (parent_id =
        // section_id = 1). The b.39x SelfParenting validator now rejects
        // that shape — bump the section_id range so we still exercise 100
        // children without tripping the new rule.
        for i in 0u32..100 {
            let section_id = u64::from(200 - i);
            let order = 99 - i;
            let name = format!("S{section_id}");
            apply_delta(
                &mut state,
                Some(delta_diff_change_section(1, section_id, &name, order)),
            )
            .unwrap();
        }

        let list = children_of(&state, 1);
        assert_eq!(list.len(), 100);

        let orders: Vec<u32> = list
            .iter()
            .map(|&cid| state.sections[&cid].section_order)
            .collect();
        let mut sorted = orders.clone();
        sorted.sort_unstable();
        assert_eq!(orders, sorted);
        assert_eq!(orders[0], 0);
        assert_eq!(orders[99], 99);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // b.y2t — Input validation regression tests
    // ═══════════════════════════════════════════════════════════════════════

    // ── (2) external-value hash-length validation ───────────────────────────
    //
    // An external value's content hash (a SHA-256 digest; SDD §11.1) is always
    // 32 bytes. The previous implementation silently
    // zero-padded shorter inputs and truncated longer inputs via
    // `copy_from_slice(&input[..min(len, 32)])`. Any non-32-byte input must
    // now produce DeltaError::InvalidExternalHashLength. Exercised via
    // apply_delta with ext_value_size > 0: the rejection comes from
    // `validate_delta_bounds` at the top of `apply_delta_into`, ahead of
    // `field_value_from_oodf`'s own check.

    /// Build an OwnedDelta with a single SetField that carries an external
    /// value whose hash bytes have the given length.
    fn delta_set_field_external(hash_bytes: Vec<u8>) -> OwnedDelta {
        OwnedDelta {
            is_diff: false,
            set_fields: vec![oodf_ffi::SetField {
                section_id: 1,
                key_name: "k".to_string(),
                value_or_ext_hash: hash_bytes,
                ext_value_size: 1024, // > 0 → external branch
                ext_value_index: 0,
                set_order: 0,
            }],
            ..Default::default()
        }
    }

    #[test]
    fn apply_delta_set_field_external_hash_zero_bytes_returns_invalid_hash_length() {
        let mut state = make_state();
        let err = apply_delta(&mut state, Some(delta_set_field_external(vec![]))).unwrap_err();
        assert!(
            matches!(
                err,
                DeltaError::InvalidExternalHashLength {
                    got: 0,
                    expected: 32,
                }
            ),
            "expected InvalidExternalHashLength(got=0, expected=32), got: {err:?}"
        );
    }

    #[test]
    fn apply_delta_set_field_external_hash_16_bytes_returns_invalid_hash_length() {
        let mut state = make_state();
        let err =
            apply_delta(&mut state, Some(delta_set_field_external(vec![0xAA; 16]))).unwrap_err();
        assert!(
            matches!(
                err,
                DeltaError::InvalidExternalHashLength {
                    got: 16,
                    expected: 32,
                }
            ),
            "expected InvalidExternalHashLength(got=16, expected=32), got: {err:?}"
        );
    }

    #[test]
    fn apply_delta_set_field_external_hash_31_bytes_returns_invalid_hash_length() {
        let mut state = make_state();
        let err =
            apply_delta(&mut state, Some(delta_set_field_external(vec![0xAA; 31]))).unwrap_err();
        assert!(
            matches!(
                err,
                DeltaError::InvalidExternalHashLength {
                    got: 31,
                    expected: 32,
                }
            ),
            "expected InvalidExternalHashLength(got=31, expected=32), got: {err:?}"
        );
    }

    #[test]
    fn apply_delta_set_field_external_hash_33_bytes_returns_invalid_hash_length() {
        let mut state = make_state();
        let err =
            apply_delta(&mut state, Some(delta_set_field_external(vec![0xAA; 33]))).unwrap_err();
        assert!(
            matches!(
                err,
                DeltaError::InvalidExternalHashLength {
                    got: 33,
                    expected: 32,
                }
            ),
            "expected InvalidExternalHashLength(got=33, expected=32), got: {err:?}"
        );
    }

    #[test]
    fn apply_delta_set_field_external_hash_64_bytes_returns_invalid_hash_length() {
        let mut state = make_state();
        let err =
            apply_delta(&mut state, Some(delta_set_field_external(vec![0xAA; 64]))).unwrap_err();
        assert!(
            matches!(
                err,
                DeltaError::InvalidExternalHashLength {
                    got: 64,
                    expected: 32,
                }
            ),
            "expected InvalidExternalHashLength(got=64, expected=32), got: {err:?}"
        );
    }

    #[test]
    fn apply_delta_set_field_external_hash_32_bytes_succeeds_sanity() {
        // Sanity check: a 32-byte hash — the valid content-hash length —
        // continues to succeed. Protects against an over-eager fix that rejects
        // all hashes.
        let mut state = make_state();
        let hash_bytes: Vec<u8> = (0u8..32).collect();
        apply_delta(
            &mut state,
            Some(delta_set_field_external(hash_bytes.clone())),
        )
        .expect(
            "a 32-byte external hash (the valid content-hash length) must continue to be accepted",
        );

        let field = state
            .fields
            .get(&(1, "k".to_string()))
            .expect("field must exist");
        match &field.value {
            FieldValue::External { hash, size, .. } => {
                assert_eq!(hash.as_slice(), hash_bytes.as_slice());
                assert_eq!(*size, 1024);
            }
            other => panic!("expected FieldValue::External, got {other:?}"),
        }
    }

    /// Sibling check: the same validation must run for TimeSample external
    /// hashes. `validate_delta_bounds` checks `time_samples` in a loop of its
    /// own, so this arm is what proves the TimeSample side is gated too.
    #[test]
    fn apply_delta_time_sample_external_hash_wrong_length_returns_invalid_hash_length() {
        let mut state = make_state();
        let delta = OwnedDelta {
            is_diff: false,
            time_samples: vec![oodf_ffi::TimeSample {
                section_id: 7,
                time: 1.0,
                value_or_ext_hash: vec![0xAA; 16],
                ext_value_size: 1024,
                ext_value_index: 0,
            }],
            ..Default::default()
        };
        let err = apply_delta(&mut state, Some(delta)).unwrap_err();
        assert!(
            matches!(
                err,
                DeltaError::InvalidExternalHashLength {
                    got: 16,
                    expected: 32,
                }
            ),
            "expected InvalidExternalHashLength(got=16, expected=32), got: {err:?}"
        );
    }

    /// Inline (non-external) values must NOT trigger the length check even
    /// when their byte length is not 32 — InvalidExternalHashLength is gated
    /// on ext_value_size > 0.
    #[test]
    fn apply_delta_set_field_inline_odd_length_unaffected_by_hash_check() {
        let mut state = make_state();
        let delta = OwnedDelta {
            is_diff: false,
            set_fields: vec![oodf_ffi::SetField {
                section_id: 1,
                key_name: "inline".to_string(),
                value_or_ext_hash: vec![0xAA; 17], // arbitrary non-32 length
                ext_value_size: 0,                 // inline branch
                ext_value_index: 0,
                set_order: 0,
            }],
            ..Default::default()
        };
        apply_delta(&mut state, Some(delta))
            .expect("inline value must not trigger the external-hash length check");
    }

    // ── (3a) apply_create_section duplicate section_id ──────────────────────
    //
    // Previously, CreateSection with a section_id already present silently
    // overwrote the existing section's record. Must now return
    // DeltaError::DuplicateSection.

    #[test]
    fn apply_delta_create_section_duplicate_section_id_returns_duplicate_section() {
        // Arrange: section 42 already exists under root.
        let mut state = make_state();
        state.sections.insert(42, make_section(42, 0, "Existing"));
        state.children.insert(0, vec![42]);

        // Act: try to create another section with the same section_id (42)
        // under root, with a different name so the failure is not just the
        // name-conflict / cascade-delete path.
        let err = apply_delta(&mut state, Some(delta_create_section(0, 42, "Reused"))).unwrap_err();

        // Assert: DuplicateSection with the offending section_id.
        assert!(
            matches!(err, DeltaError::DuplicateSection { section_id: 42 }),
            "expected DuplicateSection {{ section_id: 42 }}, got: {err:?}"
        );

        // State must be unchanged — the original section survives untouched.
        let existing = state
            .sections
            .get(&42)
            .expect("pre-existing section must survive");
        assert_eq!(existing.name, "Existing", "name must not be overwritten");
    }

    /// When the name collides with an existing *sibling* (different section_id),
    /// the cascade-delete-on-name-conflict path must still work — the
    /// duplicate-section check is about section_id, not name. This test
    /// protects against a fix that confuses the two.
    #[test]
    fn apply_delta_create_section_distinct_id_same_name_still_cascades() {
        let mut state = make_state();
        state.sections.insert(50, make_section(50, 0, "Collide"));
        state.children.insert(0, vec![50]);

        // section_id=51 is fresh, but name "Collide" matches section 50 under
        // the same parent. Legacy cascade-delete behavior must still apply.
        apply_delta(&mut state, Some(delta_create_section(0, 51, "Collide")))
            .expect("distinct section_id with name collision must still succeed (cascade)");

        assert!(
            !state.sections.contains_key(&50),
            "name-colliding sibling must be cascade-deleted"
        );
        assert!(
            state.sections.contains_key(&51),
            "new section must be created"
        );
    }

    // ── (3b) insert_child_into_parent duplicate child — silent de-dup ───────
    //
    // Design choice: the Engineer kept duplicate-child-insertion a silent
    // no-op rather than an error. Stored state can list an id under a parent
    // ahead of it arriving there (a ghost or stale listing the snapshot load
    // reads back), and the create and DiffChangeSection paths insert without
    // scrubbing the destination first, so silent de-dup avoids introducing a
    // duplicate id on those paths. The stricter case — a CreateSection that
    // would overwrite an existing section record — is caught earlier by the
    // DuplicateSection check (see the test above).
    //
    // This test documents and locks in the silent-dedup invariant by
    // exercising insert_child_into_parent directly.

    #[test]
    fn insert_child_into_parent_existing_child_is_silent_no_op() {
        // Directly exercise insert_child_into_parent: section 5 has a
        // children list that already contains child_id 100. Calling
        // insert_child_into_parent again with child_id 100 must not
        // duplicate the entry and must not emit a mutation.
        let mut state = make_state();
        state.children.insert(5, vec![100]);
        let mut plan = ChildPlan::default();
        let mut stats = ApplyDeltaStats::default();

        insert_child_into_parent(&mut state, &mut plan, &mut stats, 5, 100, 0);

        assert_eq!(
            state.children.get(&5),
            Some(&vec![100u64]),
            "child list must not grow on duplicate insert"
        );
        // b.ot3: children writes are coalesced into one finalization pass,
        // so "emitted no mutation" is now "did not mark the parent dirty" —
        // asserted by running the finalizer and getting nothing.
        let mut sink = MutationSink::collecting();
        finalize_children_lists(&state, &plan, &mut sink, &mut stats);
        let MutationSink::Collect(mutations) = sink else {
            unreachable!("collecting sink");
        };
        assert!(
            mutations.is_empty(),
            "duplicate insert must not emit a SetChildren mutation, got: {mutations:?}"
        );
    }

    /// End-to-end sibling check: a MoveSection call whose destination
    /// children list already contains the moved child_id must succeed
    /// without duplicating the entry. A parent-changing move scrubs the
    /// destination before it appends (see
    /// `apply_delta_move_onto_a_legacy_stale_listing_at_the_destination_lands_last`
    /// for where the section lands), so the list holds the id once.
    #[test]
    fn apply_delta_move_section_destination_already_contains_child_no_duplicate() {
        let mut state = make_state();
        // Section 1 is the current parent.
        state.sections.insert(1, make_section(1, 0, "OldParent"));
        // Section 2 is the destination parent.
        state.sections.insert(2, make_section(2, 0, "NewParent"));
        // Section 100 currently lives under section 1.
        state.sections.insert(100, make_section(100, 1, "ToMove"));
        state.children.insert(0, vec![1, 2]);
        state.children.insert(1, vec![100]);
        // Destination parent 2 ALREADY lists 100: a stale listing in stored
        // state, which this engine does not produce but the snapshot load
        // reads back.
        state.children.insert(2, vec![100]);

        apply_delta(&mut state, Some(delta_move_section(100, 1, 2, "MovedName")))
            .expect("move must succeed even if destination already contains the child");

        let dest = state.children.get(&2).cloned().unwrap_or_default();
        let count = dest.iter().filter(|&&c| c == 100).count();
        assert_eq!(
            count, 1,
            "destination children list must contain child 100 exactly once, got: {dest:?}"
        );
    }

    // ── (4) apply_set_field set_order overflow ──────────────────────────────
    //
    // Previously, state.next_set_order + delta_set_order and the +1 that
    // bumps next_set_order both used unchecked u64 arithmetic, silently
    // wrapping on overflow and breaking monotonic ordering. Must now
    // return DeltaError::SetOrderOverflow.

    fn delta_set_field_with_order(delta_set_order: u64) -> OwnedDelta {
        OwnedDelta {
            is_diff: false,
            set_fields: vec![oodf_ffi::SetField {
                section_id: 1,
                key_name: "k".to_string(),
                value_or_ext_hash: vec![0x01],
                ext_value_size: 0,
                ext_value_index: 0,
                set_order: delta_set_order,
            }],
            ..Default::default()
        }
    }

    #[test]
    fn apply_delta_set_field_next_plus_delta_overflow_returns_set_order_overflow() {
        // b.qw6 item 2: the ceiling moved from u64::MAX to 2^53 - 1 for
        // parity with the Lua fast path's f64 integer precision. With
        // `state.next_set_order = 2^53 - 1` and `delta_set_order = 1`,
        // `stored = next + delta = 2^53` crosses the ceiling and must
        // be rejected. `apply_delta` (this test) bypasses
        // `validate_delta_bounds` so we exercise the runtime ceiling
        // check directly — that's the surviving guard for objects whose
        // `next_set_order` cursor accumulated to the ceiling across many
        // legitimate writes (a path the validator can't see).
        let ceiling = (1u64 << 53) - 1;
        let mut state = make_state();
        state.next_set_order = ceiling;

        let err = apply_delta(&mut state, Some(delta_set_field_with_order(1))).unwrap_err();
        assert!(
            matches!(
                err,
                DeltaError::SetOrderOverflow {
                    next,
                    delta: 1,
                } if next == ceiling
            ),
            "expected SetOrderOverflow(next=2^53-1, delta=1), got: {err:?}"
        );
    }

    #[test]
    fn apply_delta_set_field_stored_plus_one_overflow_returns_set_order_overflow() {
        // b.qw6 item 2: `state.next_set_order = 2^53 - 2`, `delta = 1`
        // produces `stored = 2^53 - 1` (the ceiling itself) and the
        // NEXT call would require `stored + 1 = 2^53` which crosses
        // the line. The ceiling check fires at the boundary —
        // `stored_set_order >= SET_ORDER_CEILING` — to keep the cursor
        // strictly below 2^53 - 1 so subsequent writes have headroom.
        let ceiling = (1u64 << 53) - 1;
        let mut state = make_state();
        state.next_set_order = ceiling - 1;

        let err = apply_delta(&mut state, Some(delta_set_field_with_order(1))).unwrap_err();
        assert!(
            matches!(err, DeltaError::SetOrderOverflow { .. }),
            "expected SetOrderOverflow at the stored=2^53-1 boundary, got: {err:?}"
        );
    }

    #[test]
    fn apply_delta_set_field_large_delta_with_small_next_overflow_returns_set_order_overflow() {
        // b.qw6 item 2: a `delta_set_order` larger than the ceiling itself
        // is rejected at the apply layer even with `next_set_order = 0`.
        // (Note: `validate_delta_bounds` ALSO catches this earlier from
        // the gRPC entry point, but `apply_delta` is called by
        // CreateLayer and replay paths that don't always flow through
        // the validator — defense in depth.)
        let ceiling = (1u64 << 53) - 1;
        let mut state = make_state();
        state.next_set_order = 10;

        let err = apply_delta(&mut state, Some(delta_set_field_with_order(ceiling))).unwrap_err();
        assert!(
            matches!(
                err,
                DeltaError::SetOrderOverflow {
                    next: 10,
                    delta,
                } if delta == ceiling
            ),
            "expected SetOrderOverflow(next=10, delta=2^53-1), got: {err:?}"
        );
    }

    #[test]
    fn apply_delta_set_field_far_below_u64_max_succeeds_sanity() {
        // Sanity check: nowhere near overflow — continues to succeed.
        // Protects against an over-eager checked_add that rejects normal input.
        let mut state = make_state();
        state.next_set_order = 100;

        apply_delta(&mut state, Some(delta_set_field_with_order(5)))
            .expect("normal set_order values must continue to be accepted");

        let field = state
            .fields
            .get(&(1, "k".to_string()))
            .expect("field must exist");
        assert_eq!(field.set_order, 105, "stored = 100 + 5 = 105");
        assert_eq!(
            state.next_set_order, 106,
            "next_set_order = max(100, 106) = 106"
        );
    }

    // ── b.55p H-dom-2: a reorder never lists a ghost or another parent's child ─
    //
    // Before b.55p, `apply_reorder_children` overwrote
    // `state.children[section_id]` verbatim, so a listed id with no row reached
    // the DiffEngine DFS, which marked it "visited" and silently omitted a
    // legit orphan sharing that id from the reconstructed diff. b.55p rejected
    // such a reorder; b.xje ignores the id instead (USD's
    // `SdfApplyListOrdering` skips an ordered item the list does not hold).
    // Either way the id never enters a list, which is the hazard these pin.
    // The full set of ignored-id cases is in the b.xje block at the end.

    /// A reorder naming an id with no row (case (a) of the ignored ids) commits
    /// as if the id were absent: the ghost is never listed.
    ///
    /// Discriminated by: an early return in `apply_reorder_children` whenever
    /// a named id is not one of the parent's children — the shape of the old
    /// rejection — which leaves P's list `[A, B, C]`.
    #[test]
    fn apply_reorder_children_with_an_unknown_section_id_ignores_it() {
        const P: u64 = 5;
        const GHOST: u64 = 42;
        let (state, [a, b, c]) = ignored_id_fixture();

        let after = assert_reorder_ignores("unknown id", &state, P, vec![c, GHOST, a], vec![c, a]);
        assert_eq!(after.children[&P], vec![c, a, b]);
        assert!(!after.sections.contains_key(&GHOST));
    }

    /// A reorder naming a section whose row names another parent (case (c))
    /// ignores it: the section is not listed under two parents, its row gets
    /// no `SetSection`, and its own parent's list is untouched.
    ///
    /// Discriminated by: the same early return as the test above.
    #[test]
    fn apply_reorder_children_naming_another_parents_child_ignores_it() {
        const P: u64 = 5;
        const Q: u64 = 9;
        const X: u64 = 7;
        let (mut state, [a, b, c]) = ignored_id_fixture();
        seed_section(&mut state, Q, 0, "Q", 1);
        seed_section(&mut state, X, Q, "X", 0);
        state.children.insert(0, vec![P, Q]);
        state.children.insert(Q, vec![X]);

        let after = assert_reorder_ignores(
            "another parent's child",
            &state,
            P,
            vec![c, X, a],
            vec![c, a],
        );
        assert_eq!(after.children[&P], vec![c, a, b]);
        assert_eq!(after.children[&Q], vec![X], "Q's list is untouched");
        assert_eq!(
            after.sections[&X], state.sections[&X],
            "X's row is untouched"
        );
    }

    #[test]
    fn apply_reorder_children_real_ids_only_still_succeeds() {
        // Sanity: a reorder naming only the parent's listed children, every
        // one of them — a complete list — installs exactly as sent. (Ids that
        // are not listed children are ignored; the tests above pin that.)
        let mut state = make_state();
        state.sections.insert(5, make_section(5, 0, "Parent"));
        state.sections.insert(1, make_section(1, 5, "A"));
        state.sections.insert(2, make_section(2, 5, "B"));
        state.children.insert(0, vec![5]);
        state.children.insert(5, vec![1, 2]);

        let delta = OwnedDelta {
            is_diff: false,
            struct_commands: vec![StructCommand::ReorderChildren(OodfReorderChildren {
                section_id: 5,
                children_list_id: 0,
                children_list: vec![2, 1],
            })],
            ..Default::default()
        };
        apply_delta(&mut state, Some(delta)).expect("real-id reorder must succeed");
        assert_eq!(state.children.get(&5), Some(&vec![2u64, 1]));
    }

    // ── b.55p H-dom-3: duplicate-name slip past CreateSection ───────────────
    //
    // Pre-fix, `find_child_by_name` iterated `state.children[parent_id]` only.
    // An orphan section (present in `state.sections` but absent from its
    // parent's children list — stored state the snapshot load can read back,
    // e.g. a partial reorder stored before reorders kept the children they
    // did not name) was invisible to the conflict check, letting
    // `apply_create_section` create a second sibling with the same
    // `(parent_id, name)`. Post-fix, `find_child_by_name` scans
    // `state.sections.values()`, so the orphan is detected and the conflict
    // cascade fires.

    #[test]
    fn apply_create_section_orphan_with_same_name_triggers_conflict_cascade() {
        let mut state = make_state();
        // Section 200 lives in state.sections with parent 0 and name "Target",
        // but is NOT present in state.children[0] — the orphan shape stored
        // state can hold (see the block comment above).
        state.sections.insert(200, make_section(200, 0, "Target"));
        // state.children[0] deliberately omits 200 — THAT is the orphan shape.
        state.children.insert(0, Vec::new());

        // Attempt to create a second sibling named "Target" under the same
        // parent. Pre-fix this admitted a duplicate name; post-fix the
        // orphan is detected, cascade-deleted, then the new section is
        // admitted with a unique (parent=0, name="Target") pair.
        let delta = delta_create_section(0, 201, "Target");
        apply_delta(&mut state, Some(delta)).expect("create with conflict must cascade");

        // The orphan (200) must be gone; the new section (201) must remain
        // and hold the name uniquely.
        assert!(
            !state.sections.contains_key(&200),
            "conflicting orphan (200) must be cascade-deleted"
        );
        let new_sec = state
            .sections
            .get(&201)
            .expect("new section (201) must be created");
        assert_eq!(new_sec.name, "Target");
        assert_eq!(new_sec.parent_id, 0);

        // Post-state cross-map invariant: no two sections share (parent, name).
        let count_named_target = state
            .sections
            .values()
            .filter(|s| s.parent_id == 0 && s.name == "Target")
            .count();
        assert_eq!(
            count_named_target, 1,
            "duplicate-name slip regression: exactly one section named 'Target' must remain"
        );
    }

    // ── b.55p: sequence_num overflow rejection ──────────────────────────────
    //
    // b.k1r H-stor-9: the ceiling moved from `u64::MAX` to `2^53 - 1` so
    // the slow path fails at the same point as the Lua fast path (Lua 5.1
    // `number` is f64 and integer precision stops at 2^53). The
    // pre-b.k1r `u64::MAX` case is still covered by `checked_add` — it's
    // just unreachable because the 2^53 ceiling fires first.
    const SEQUENCE_CEILING: u64 = (1u64 << 53) - 1;

    #[test]
    fn apply_delta_at_sequence_ceiling_returns_sequence_overflow() {
        // Seed the state one below the ceiling and step once to reach the
        // ceiling, then attempt a second step which must overflow.
        let mut state = make_state();
        state.sequence_num = SEQUENCE_CEILING - 1;

        apply_delta(&mut state, None).expect("step to 2^53 - 1 must still succeed");
        assert_eq!(state.sequence_num, SEQUENCE_CEILING);

        let err = apply_delta(&mut state, None).unwrap_err();
        assert!(
            matches!(err, DeltaError::SequenceOverflow),
            "expected SequenceOverflow at 2^53 - 1 + 1, got {err:?}"
        );
        // State must remain pinned at the ceiling — the guard fails
        // before mutating.
        assert_eq!(state.sequence_num, SEQUENCE_CEILING);
    }

    // ── b.55p medium: cascade_delete_section ghost-id handling ──────────────
    //
    // Invariant (post-fix): a ghost id in `state.children[p]` — one present
    // in a children list but missing from `state.sections` — must not emit
    // a bogus `CreateTombstone { first_version: 0, .. }` during cascade
    // delete. It must still prune the stale children-map entry.

    #[test]
    fn cascade_delete_section_ghost_id_emits_no_tombstone_and_clears_children_entry() {
        let mut state = make_state();
        // Real parent + real child.
        state.sections.insert(10, make_section(10, 0, "Parent"));
        state.sections.insert(11, make_section(11, 10, "Child"));
        state.children.insert(0, vec![10]);
        // Parent's children list contains real child 11 AND a ghost 99.
        state.children.insert(10, vec![11, 99]);
        // Also seed a children entry for the ghost so we can assert it is
        // cleared during cascade.
        state.children.insert(99, Vec::new());

        let mutations = apply_delta(&mut state, Some(delta_delete_section(10))).unwrap();

        // No tombstone for 99 — the ghost-id branch must skip.
        let tombstones_for_ghost: Vec<_> = mutations
            .iter()
            .filter_map(|m| match m {
                StorageMutation::CreateTombstone {
                    section_id: 99,
                    first_version,
                    ..
                } => Some(*first_version),
                _ => None,
            })
            .collect();
        assert!(
            tombstones_for_ghost.is_empty(),
            "ghost-id 99 must not emit a tombstone (got first_version list {tombstones_for_ghost:?})"
        );

        // Real child 11 and parent 10 both got tombstones.
        let tombstoned_ids: Vec<u64> = mutations
            .iter()
            .filter_map(|m| match m {
                StorageMutation::CreateTombstone { section_id, .. } => Some(*section_id),
                _ => None,
            })
            .collect();
        assert!(tombstoned_ids.contains(&10), "parent 10 must be tombstoned");
        assert!(
            tombstoned_ids.contains(&11),
            "real child 11 must be tombstoned"
        );

        // The ghost entry in state.children must be gone.
        assert!(
            !state.children.contains_key(&99),
            "ghost entry in state.children must be cleared"
        );
    }

    // ── b.55p: cascade delete emits DeleteField for external-value fields ───
    //
    // Cascade delete must emit a `DeleteField` mutation for every field
    // under the deleted subtree, regardless of whether the field stores an
    // inline value or an external content-addressed value. Storage uses the
    // mutation to decrement external-value reference counts and enqueue GC
    // (see storage::ext_value). Missing the mutation on the external branch
    // would silently leak external blobs.

    #[test]
    fn cascade_delete_section_emits_delete_field_for_external_value_field() {
        let mut state = make_state();
        state.sections.insert(10, make_section(10, 0, "Parent"));
        state.sections.insert(11, make_section(11, 10, "Child"));
        state.children.insert(0, vec![10]);
        state.children.insert(10, vec![11]);

        // Field on the child section backed by an external value.
        let ext_hash = [0xAB_u8; 32];
        state.fields.insert(
            (11, "tex".to_string()),
            Field {
                section_id: 11,
                key_name: "tex".to_string(),
                value: FieldValue::External {
                    hash: ext_hash,
                    size: 1_048_576,
                    ext_value_index: 1,
                },
                set_order: 0,
                authoring_set_order: 0,
                version: 1,
            },
        );

        let mutations = apply_delta(&mut state, Some(delta_delete_section(10))).unwrap();

        let has_delete_field = mutations.iter().any(|m| {
            matches!(
                m,
                StorageMutation::DeleteField {
                    section_id: 11,
                    key_name,
                } if key_name == "tex"
            )
        });
        assert!(
            has_delete_field,
            "cascade must emit DeleteField for external-value field (got {mutations:?})"
        );

        // Sanity: the field is also gone from state.
        assert!(
            !state.fields.contains_key(&(11, "tex".to_string())),
            "external-value field must be removed from state.fields after cascade"
        );
    }

    // ═══════════════════════════════════════════════════════════════════════
    // b.39x — Validation surface coverage
    //
    // Each block pins one of the b.39x acceptance criteria at the apply-engine
    // boundary so a regression in either `validate_delta_bounds` or the
    // `apply_delta` wiring is caught by a targeted unit test rather than
    // surfacing only in integration runs. The validator-level tests live in
    // `domain::validation::tests`; these are the apply-engine smoke tests
    // that prove the rule fires through the public API.
    // ═══════════════════════════════════════════════════════════════════════

    // ── b.39x item 3: delta name with NUL / control char (NameContainsControlChar) ──

    #[test]
    fn apply_delta_create_section_name_with_null_byte_returns_name_contains_control_char() {
        let mut state = make_state();
        let err =
            apply_delta(&mut state, Some(delta_create_section(0, 10, "bad\0name"))).unwrap_err();
        match err {
            DeltaError::NameContainsControlChar { field } => {
                assert_eq!(field, "CreateSection.section_name");
            }
            other => panic!("expected NameContainsControlChar, got {other:?}"),
        }
        assert!(
            !state.sections.contains_key(&10),
            "rejected delta must not mutate state"
        );
    }

    #[test]
    fn apply_delta_create_section_name_with_ansi_escape_returns_name_contains_control_char() {
        // ESC (0x1B) is the ANSI-escape introducer. A name echoed into
        // structured tracing or stream payloads could overwrite operator
        // terminal state — same threat model as `validate_client_id`.
        let mut state = make_state();
        let err = apply_delta(
            &mut state,
            Some(delta_create_section(0, 11, "red\x1b[31mtext")),
        )
        .unwrap_err();
        assert!(
            matches!(err, DeltaError::NameContainsControlChar { field } if field == "CreateSection.section_name"),
            "expected NameContainsControlChar(CreateSection.section_name), got {err:?}"
        );
    }

    #[test]
    fn apply_delta_move_section_new_name_with_control_char_returns_name_contains_control_char() {
        let mut state = make_state();
        state.sections.insert(20, make_section(20, 0, "ToMove"));
        state.children.insert(0, vec![20]);
        // Bell (0x07) — clearly a control char, not whitespace, not UTF-8.
        let err = apply_delta(
            &mut state,
            Some(delta_move_section(20, 0, 0, "bell\x07name")),
        )
        .unwrap_err();
        assert!(
            matches!(err, DeltaError::NameContainsControlChar { field } if field == "MoveSection.new_name"),
            "expected NameContainsControlChar(MoveSection.new_name), got {err:?}"
        );
    }

    #[test]
    fn apply_delta_diff_change_section_name_with_newline_returns_name_contains_control_char() {
        let mut state = make_state();
        let err = apply_delta(
            &mut state,
            Some(delta_diff_change_section(0, 30, "line1\nline2", 0)),
        )
        .unwrap_err();
        assert!(
            matches!(err, DeltaError::NameContainsControlChar { field } if field == "DiffChangeSection.section_name"),
            "expected NameContainsControlChar(DiffChangeSection.section_name), got {err:?}"
        );
    }

    #[test]
    fn apply_delta_set_field_key_name_with_null_byte_returns_name_contains_control_char() {
        let mut state = make_state();
        let delta = OwnedDelta {
            is_diff: false,
            set_fields: vec![oodf_ffi::SetField {
                section_id: 1,
                key_name: "key\0bad".to_string(),
                value_or_ext_hash: vec![0u8],
                ext_value_size: 0,
                ext_value_index: 0,
                set_order: 0,
            }],
            ..Default::default()
        };
        let err = apply_delta(&mut state, Some(delta)).unwrap_err();
        assert!(
            matches!(err, DeltaError::NameContainsControlChar { field } if field == "SetField.key_name"),
            "expected NameContainsControlChar(SetField.key_name), got {err:?}"
        );
    }

    // Positive case: ordinary names round-trip through validation. Pins the
    // rule scope — only control bytes 0x00..=0x1F are rejected; printable
    // ASCII and multi-byte UTF-8 must pass.
    #[test]
    fn apply_delta_create_section_name_with_printable_ascii_and_utf8_succeeds() {
        let mut state = make_state();
        // ASCII space (0x20) is the boundary — must pass.
        apply_delta(&mut state, Some(delta_create_section(0, 100, "has space")))
            .expect("space must be accepted");
        // Multi-byte UTF-8 must pass too.
        apply_delta(&mut state, Some(delta_create_section(0, 101, "セクション")))
            .expect("UTF-8 multibyte must be accepted");
        assert!(state.sections.contains_key(&100));
        assert!(state.sections.contains_key(&101));
    }

    // ── b.39x item 4: NaN / +∞ / −∞ time samples (TimeSampleNonFiniteTime) ────

    fn delta_one_time_sample(time: f64) -> OwnedDelta {
        OwnedDelta {
            is_diff: false,
            time_samples: vec![oodf_ffi::TimeSample {
                section_id: 1,
                time,
                value_or_ext_hash: vec![0u8],
                ext_value_size: 0,
                ext_value_index: 0,
            }],
            ..Default::default()
        }
    }

    #[test]
    fn apply_delta_time_sample_nan_returns_non_finite_time() {
        let mut state = make_state();
        state.sections.insert(1, make_section(1, 0, "S"));
        let err = apply_delta(&mut state, Some(delta_one_time_sample(f64::NAN))).unwrap_err();
        match err {
            DeltaError::TimeSampleNonFiniteTime { index, time } => {
                assert_eq!(index, 0);
                assert!(time.is_nan(), "rejected time must be NaN: got {time}");
            }
            other => panic!("expected TimeSampleNonFiniteTime, got {other:?}"),
        }
    }

    #[test]
    fn apply_delta_time_sample_positive_infinity_returns_non_finite_time() {
        let mut state = make_state();
        let err = apply_delta(&mut state, Some(delta_one_time_sample(f64::INFINITY))).unwrap_err();
        match err {
            DeltaError::TimeSampleNonFiniteTime { index, time } => {
                assert_eq!(index, 0);
                assert_eq!(time, f64::INFINITY);
            }
            other => panic!("expected TimeSampleNonFiniteTime, got {other:?}"),
        }
    }

    #[test]
    fn apply_delta_time_sample_negative_infinity_returns_non_finite_time() {
        let mut state = make_state();
        let err =
            apply_delta(&mut state, Some(delta_one_time_sample(f64::NEG_INFINITY))).unwrap_err();
        match err {
            DeltaError::TimeSampleNonFiniteTime { index, time } => {
                assert_eq!(index, 0);
                assert_eq!(time, f64::NEG_INFINITY);
            }
            other => panic!("expected TimeSampleNonFiniteTime, got {other:?}"),
        }
    }

    #[test]
    fn apply_delta_time_sample_non_finite_index_reports_offending_position() {
        // Build a delta with three time samples; the middle one is NaN. The
        // index in the error must point at that exact sample so operators
        // triaging the failure don't have to scan a multi-sample payload.
        let mut state = make_state();
        let delta = OwnedDelta {
            is_diff: false,
            time_samples: vec![
                oodf_ffi::TimeSample {
                    section_id: 1,
                    time: 0.0,
                    value_or_ext_hash: vec![0u8],
                    ext_value_size: 0,
                    ext_value_index: 0,
                },
                oodf_ffi::TimeSample {
                    section_id: 1,
                    time: f64::NAN,
                    value_or_ext_hash: vec![0u8],
                    ext_value_size: 0,
                    ext_value_index: 0,
                },
                oodf_ffi::TimeSample {
                    section_id: 1,
                    time: 1.0,
                    value_or_ext_hash: vec![0u8],
                    ext_value_size: 0,
                    ext_value_index: 0,
                },
            ],
            ..Default::default()
        };
        let err = apply_delta(&mut state, Some(delta)).unwrap_err();
        assert!(
            matches!(err, DeltaError::TimeSampleNonFiniteTime { index: 1, .. }),
            "expected TimeSampleNonFiniteTime at index 1, got {err:?}"
        );
    }

    // Positive case: finite times (including 0.0, negative, large magnitudes,
    // subnormal) all round-trip. Pins the rule scope — only NaN/Inf rejected.
    #[test]
    fn apply_delta_time_sample_finite_values_succeed() {
        let mut state = make_state();
        // Create section 1 via apply_delta so cross-map invariants
        // (state.children entry for parent 0) stay consistent — the
        // `debug_assert_cross_map_consistent` at the end of
        // `apply_delta_with_stats` would fire on a bare
        // `state.sections.insert`.
        apply_delta(&mut state, Some(delta_create_section(0, 1, "S")))
            .expect("setup CreateSection must succeed");
        // Sanity check `f64::is_finite` against the values we accept.
        for time in [0.0_f64, -1.0, 1.0, f64::MIN_POSITIVE, f64::MAX, f64::MIN] {
            assert!(time.is_finite(), "test bug: {time} must be finite");
            apply_delta(&mut state, Some(delta_one_time_sample(time)))
                .unwrap_or_else(|e| panic!("finite time {time} must be accepted, got {e:?}"));
        }
    }

    // ── b.39x item 5: self-parenting (SelfParenting) ─────────────────────────

    #[test]
    fn apply_delta_create_section_self_parenting_returns_self_parenting_error() {
        let mut state = make_state();
        // parent_id = section_id = 7 (both non-zero) → SelfParenting.
        let err = apply_delta(&mut state, Some(delta_create_section(7, 7, "Cycle"))).unwrap_err();
        match err {
            DeltaError::SelfParenting {
                command,
                section_id,
            } => {
                assert_eq!(command, "CreateSection");
                assert_eq!(section_id, 7);
            }
            other => panic!("expected SelfParenting(CreateSection), got {other:?}"),
        }
        assert!(
            !state.sections.contains_key(&7),
            "rejected delta must not mutate state"
        );
    }

    #[test]
    fn apply_delta_move_section_self_parenting_returns_self_parenting_error() {
        let mut state = make_state();
        state.sections.insert(8, make_section(8, 0, "Movable"));
        state.children.insert(0, vec![8]);
        // new_parent_id == section_id == 8.
        let err =
            apply_delta(&mut state, Some(delta_move_section(8, 0, 8, "Movable"))).unwrap_err();
        assert!(
            matches!(
                err,
                DeltaError::SelfParenting {
                    command: "MoveSection",
                    section_id: 8,
                }
            ),
            "expected SelfParenting(MoveSection, section_id=8), got {err:?}"
        );
        // State must be unchanged after a rejected delta.
        assert_eq!(state.sections[&8].parent_id, 0);
    }

    #[test]
    fn apply_delta_move_section_old_parent_self_parenting_returns_self_parenting_error() {
        // b.39x PM follow-up: parallel to the new_parent_id-based test above,
        // pin the apply-engine behaviour when `old_parent_id == section_id`.
        // The validator branch added in
        // `validate_delta_bounds_move_section_old_parent_self_parenting_returns_self_parenting`
        // must propagate through the public `apply_delta` API and leave
        // materialized state untouched.
        let mut state = make_state();
        state.sections.insert(9, make_section(9, 0, "Movable"));
        state.children.insert(0, vec![9]);
        // old_parent_id == section_id == 9; new_parent_id = 12 (distinct).
        let err = apply_delta(
            &mut state,
            Some(delta_move_section(9, 9, 12, "MovedOutOfSelf")),
        )
        .unwrap_err();
        assert!(
            matches!(
                err,
                DeltaError::SelfParenting {
                    command: "MoveSection",
                    section_id: 9,
                }
            ),
            "expected SelfParenting(MoveSection, section_id=9) on old_parent_id == section_id, got {err:?}"
        );
        // State must be unchanged after a rejected delta — the section is
        // still parented at the root, its name is unchanged, and no spurious
        // children-table entry exists for section_id 9.
        assert_eq!(state.sections[&9].parent_id, 0);
        assert_eq!(state.sections[&9].name, "Movable");
        assert!(!state.children.contains_key(&9));
    }

    #[test]
    fn apply_delta_diff_change_section_self_parenting_returns_self_parenting_error() {
        let mut state = make_state();
        let err = apply_delta(
            &mut state,
            Some(delta_diff_change_section(9, 9, "Cycle", 0)),
        )
        .unwrap_err();
        assert!(
            matches!(
                err,
                DeltaError::SelfParenting {
                    command: "DiffChangeSection",
                    section_id: 9,
                }
            ),
            "expected SelfParenting(DiffChangeSection, section_id=9), got {err:?}"
        );
    }

    // Positive case: parent_id == 0 (root marker) is NOT self-parenting even
    // when section_id == 0. Pins the carve-out documented on the validator —
    // the rule only fires when parent_id != 0 AND parent_id == section_id.
    #[test]
    fn apply_delta_create_section_root_marker_does_not_trigger_self_parenting() {
        let mut state = make_state();
        // parent_id = 0 (root), section_id = 5 — must succeed.
        apply_delta(&mut state, Some(delta_create_section(0, 5, "Child")))
            .expect("root-parented CreateSection must succeed");
        assert!(state.sections.contains_key(&5));
    }

    #[test]
    fn apply_delta_move_section_to_root_does_not_trigger_self_parenting() {
        // Edge case: moving a section to the root (new_parent_id = 0) when
        // section_id happens to share a value with the *old* parent must not
        // false-positive on the self-parenting rule — the rule examines the
        // NEW parent only.
        let mut state = make_state();
        state.sections.insert(12, make_section(12, 0, "Parent"));
        state.sections.insert(13, make_section(13, 12, "Child"));
        state.children.insert(0, vec![12]);
        state.children.insert(12, vec![13]);
        apply_delta(&mut state, Some(delta_move_section(13, 12, 0, "Child")))
            .expect("move to root must succeed");
        assert_eq!(state.sections[&13].parent_id, 0);
    }

    // ── b.39x item 7: ApplyDeltaStats counter for silent-skip path ──────────
    //
    // The CLAUDE.md "Mandatory Test Coverage" rule requires that the new
    // counter `delta_create_section_skipped_missing_parent_total` be exercised
    // by a test. The counter is bumped from `inc_delta_create_section_skipped_missing_parent`
    // by `LayerStore::update_layer` and `handle_create_layer` using the
    // `ApplyDeltaStats` returned from `apply_delta_with_stats`. The unit-test
    // surface that pins the contract is the stats out-parameter — if
    // `apply_delta_with_stats` stops incrementing on the silent-skip path the
    // caller would silently produce a zero counter even on real skips.
    //
    // Two cases:
    // 1. A delta whose CreateSection names a missing non-root parent must
    //    bump the counter by exactly 1 and leave state unmutated for that
    //    section.
    // 2. A delta with no silent-skip path must leave the counter at 0.

    #[test]
    fn apply_delta_with_stats_missing_parent_increments_skipped_counter() {
        let mut state = make_state();
        let mut stats = ApplyDeltaStats::default();
        // parent_id = 999 (non-root, not in state.sections) → silent skip.
        apply_delta_with_stats(
            &mut state,
            Some(delta_create_section(999, 50, "Orphan")),
            &mut stats,
        )
        .expect("silent skip must not error");
        assert_eq!(
            stats.create_section_skipped_missing_parent, 1,
            "stats counter must increment when CreateSection silently skipped"
        );
        assert!(
            !state.sections.contains_key(&50),
            "silently-skipped CreateSection must not mutate state.sections"
        );
    }

    #[test]
    fn apply_delta_with_stats_successful_create_does_not_increment_skipped_counter() {
        let mut state = make_state();
        let mut stats = ApplyDeltaStats::default();
        // Root-parented CreateSection succeeds — no skip.
        apply_delta_with_stats(
            &mut state,
            Some(delta_create_section(0, 60, "Real")),
            &mut stats,
        )
        .expect("root-parented CreateSection must succeed");
        assert_eq!(
            stats.create_section_skipped_missing_parent, 0,
            "stats counter must stay at 0 when nothing was skipped"
        );
        assert!(state.sections.contains_key(&60));
    }

    #[test]
    fn apply_delta_with_stats_multiple_missing_parents_increment_by_count() {
        // Two CreateSection commands in a single delta, both with missing
        // non-root parents. The counter must accumulate — not coalesce to 1
        // or saturate to anything else — so operators reading the metric can
        // tell apart "many client bugs" from "one client bug".
        let mut state = make_state();
        let mut stats = ApplyDeltaStats::default();
        let delta = OwnedDelta {
            is_diff: false,
            struct_commands: vec![
                StructCommand::CreateSection(OodfCreateSection {
                    parent_id: 998,
                    section_id: 70,
                    section_name: "Orphan1".to_string(),
                    section_type: 0,
                }),
                StructCommand::CreateSection(OodfCreateSection {
                    parent_id: 997,
                    section_id: 71,
                    section_name: "Orphan2".to_string(),
                    section_type: 0,
                }),
            ],
            ..Default::default()
        };
        apply_delta_with_stats(&mut state, Some(delta), &mut stats)
            .expect("silent skips must not error");
        assert_eq!(
            stats.create_section_skipped_missing_parent, 2,
            "counter must accumulate across multiple silent-skip paths in one delta"
        );
    }

    // ── cascade-delete on self-parented section: collect_subtree visited-set ─
    //
    // Regression pin against a future refactor that drops the visited-set in
    // `collect_subtree`. A self-parenting section in stored state (legacy data
    // from before the b.39x SelfParenting validator landed) creates a
    // single-node cycle in `state.children`; `cascade_delete_section` walks
    // that subtree via `collect_subtree`, and without the visited-set guard
    // the walk would loop forever. Termination is the load-bearing property
    // here — a regression would hang the test rather than fail an assertion,
    // and the per-test time budget is the safety net. The `is_descendant_of`
    // cycle-check coverage lives in
    // `apply_delta_move_section_cycle_check_via_is_descendant_of_terminates_on_multi_id_cycle`
    // further below.

    #[test]
    fn cascade_delete_on_self_parented_section_terminates_via_collect_subtree_visited_set() {
        // Build a state where section 7 reports itself as its own parent.
        let mut state = make_state();
        state.sections.insert(
            7,
            Section {
                section_id: 7,
                parent_id: 7, // self-parent
                name: "SelfParent".to_string(),
                section_type: 0,
                section_order: 0,
                first_version: 1,
                last_version: 1,
            },
        );
        state.children.insert(7, vec![7]); // self-cycle in children map too
        let result = apply_delta(&mut state, Some(delta_delete_section(7)));
        // Must terminate with a successful Ok — the visited-set inside
        // `collect_subtree` short-circuits the cycle.
        assert!(
            result.is_ok(),
            "cascade-delete on a self-parented section must terminate, got {result:?}"
        );
        assert!(
            !state.sections.contains_key(&7),
            "self-parented section must be removed via cascade-delete"
        );
    }

    // ── b.dmu item 2: apply_diff_change_section name-uniqueness ──────────────

    #[test]
    fn diff_change_section_rename_onto_occupied_slot_returns_duplicate_sibling_name() {
        // Set up two sections under root: 10 named "A", 11 named "B". Rename
        // 11 to "A" via DiffChangeSection — must fail with DuplicateSiblingName
        // pointing at the conflicting sibling 10.
        let mut state = make_state();
        state.sections.insert(10, make_section(10, 0, "A"));
        state.sections.insert(11, make_section(11, 0, "B"));
        state.children.insert(0, vec![10, 11]);

        let err =
            apply_delta(&mut state, Some(delta_diff_change_section(0, 11, "A", 0))).unwrap_err();

        assert!(
            matches!(
                err,
                DeltaError::DuplicateSiblingName {
                    section_id: 11,
                    parent_id: 0,
                    conflicting_id: 10,
                    ..
                }
            ),
            "expected DuplicateSiblingName(section_id=11, conflicting_id=10), got {err:?}"
        );
        // State must be unchanged — both sections retain their original
        // names and the delta did not bump sequence_num (item 6 contract).
        assert_eq!(state.sections[&11].name, "B");
        assert_eq!(state.sections[&10].name, "A");
        assert_eq!(state.sequence_num, 0);
    }

    #[test]
    fn diff_change_section_new_section_with_name_conflict_cascade_deletes_sibling() {
        // The new-section branch of DiffChangeSection mirrors CreateSection's
        // cascade-delete policy under PRD §2.3.4. A DiffChangeSection that
        // would create a new section under a parent where a sibling already
        // owns the name must cascade-delete that sibling and admit the new
        // section.
        let mut state = make_state();
        state.sections.insert(10, make_section(10, 0, "Target"));
        state.children.insert(0, vec![10]);

        apply_delta(
            &mut state,
            Some(delta_diff_change_section(0, 99, "Target", 0)),
        )
        .expect("new-section diff-change with cascade must succeed");

        // Section 10 must be cascade-deleted; section 99 must own the slot.
        assert!(
            !state.sections.contains_key(&10),
            "conflicting sibling must be cascade-deleted"
        );
        let new_sec = state
            .sections
            .get(&99)
            .expect("new section must be created");
        assert_eq!(new_sec.name, "Target");
        assert_eq!(new_sec.parent_id, 0);
    }

    #[test]
    fn diff_change_section_rename_to_unique_name_succeeds() {
        // Positive case: a rename that doesn't collide with any existing
        // sibling succeeds and updates the section. Pins the rule scope —
        // only collisions are rejected.
        let mut state = make_state();
        state.sections.insert(10, make_section(10, 0, "OldName"));
        state.children.insert(0, vec![10]);

        apply_delta(
            &mut state,
            Some(delta_diff_change_section(0, 10, "NewName", 0)),
        )
        .expect("unique-name rename must succeed");
        assert_eq!(state.sections[&10].name, "NewName");
    }

    // ── b.dmu item 4: a repeated id in a reorder is stored once ──────────────

    /// A reorder repeating an id (case (d)) counts its first occurrence:
    /// `[B, A, B]` behaves exactly as `[B, A]`, and the id is stored once —
    /// the hazard b.dmu used to reject repeats for.
    ///
    /// Discriminated by: the early return described at
    /// `apply_reorder_children_with_an_unknown_section_id_ignores_it`.
    #[test]
    fn reorder_children_duplicate_id_counts_its_first_occurrence() {
        const P: u64 = 5;
        let (state, [a, b, c]) = ignored_id_fixture();

        let after = assert_reorder_ignores("repeated id", &state, P, vec![b, a, b], vec![b, a]);
        // USD: B carries C (unnamed, after it); A carries nothing.
        assert_eq!(after.children[&P], vec![b, c, a]);
    }

    #[test]
    fn reorder_children_unique_list_still_succeeds() {
        // Positive case: a reorder list with no duplicates round-trips.
        let mut state = make_state();
        state.sections.insert(5, make_section(5, 0, "Parent"));
        state.sections.insert(1, make_section(1, 5, "A"));
        state.sections.insert(2, make_section(2, 5, "B"));
        state.children.insert(0, vec![5]);
        state.children.insert(5, vec![1, 2]);

        let delta = OwnedDelta {
            is_diff: false,
            struct_commands: vec![StructCommand::ReorderChildren(OodfReorderChildren {
                section_id: 5,
                children_list_id: 0,
                children_list: vec![2, 1],
            })],
            ..Default::default()
        };
        apply_delta(&mut state, Some(delta)).expect("unique reorder must succeed");
        assert_eq!(state.children.get(&5), Some(&vec![2u64, 1]));
    }

    // ── b.dmu item 3: SectionTreeTooDeep write-path enforcement ──────────────

    #[test]
    fn apply_delta_post_mutation_tree_too_deep_returns_section_tree_too_deep() {
        // Build a linear chain of `MAX_SECTION_TREE_DEPTH + 1` sections in
        // state.sections + state.children, then apply an empty delta. The
        // post-mutation depth check must reject because the resulting tree
        // (which is the same as the starting tree, the delta is empty) is
        // one too deep.
        use crate::domain::limits::MAX_SECTION_TREE_DEPTH;

        let mut state = make_state();
        // Build chain 1 → 2 → 3 → ... → cap+1 rooted at synthetic parent 0.
        let chain_len = MAX_SECTION_TREE_DEPTH + 1;
        for i in 1..=chain_len {
            let id = i as u64;
            let parent = if i == 1 { 0 } else { (i - 1) as u64 };
            state.sections.insert(
                id,
                Section {
                    section_id: id,
                    parent_id: parent,
                    name: format!("S{id}"),
                    section_type: 0,
                    section_order: 0,
                    first_version: 1,
                    last_version: 1,
                },
            );
            state.children.entry(parent).or_default().push(id);
        }

        let err = apply_delta(&mut state, None).unwrap_err();
        assert!(
            matches!(
                err,
                DeltaError::SectionTreeTooDeep { depth, cap }
                    if depth == MAX_SECTION_TREE_DEPTH + 1 && cap == MAX_SECTION_TREE_DEPTH
            ),
            "expected SectionTreeTooDeep(depth=cap+1), got {err:?}"
        );
        // sequence_num must not advance (item 6 + the depth check fires
        // before the commit).
        assert_eq!(state.sequence_num, 0);
    }

    #[test]
    fn apply_delta_tree_at_exact_cap_succeeds() {
        // Boundary case: a tree exactly at MAX_SECTION_TREE_DEPTH must be
        // accepted. Pins the off-by-one.
        use crate::domain::limits::MAX_SECTION_TREE_DEPTH;

        let mut state = make_state();
        let chain_len = MAX_SECTION_TREE_DEPTH;
        for i in 1..=chain_len {
            let id = i as u64;
            let parent = if i == 1 { 0 } else { (i - 1) as u64 };
            state.sections.insert(
                id,
                Section {
                    section_id: id,
                    parent_id: parent,
                    name: format!("S{id}"),
                    section_type: 0,
                    section_order: 0,
                    first_version: 1,
                    last_version: 1,
                },
            );
            state.children.entry(parent).or_default().push(id);
        }
        apply_delta(&mut state, None).expect("depth == cap must be accepted");
        // sequence_num must advance because the depth check passes.
        assert_eq!(state.sequence_num, 1);
    }

    #[test]
    fn apply_delta_create_section_that_would_breach_cap_rejected_before_commit() {
        // Build a tree at the cap, then attempt to add one more leaf at the
        // bottom. The post-mutation check must reject. Pins AC3 (the
        // write-path closes the unrecoverable-stuck-layer scenario by
        // ensuring no apply_delta call can grow stored state past the cap).
        use crate::domain::limits::MAX_SECTION_TREE_DEPTH;

        let mut state = make_state();
        let chain_len = MAX_SECTION_TREE_DEPTH;
        for i in 1..=chain_len {
            let id = i as u64;
            let parent = if i == 1 { 0 } else { (i - 1) as u64 };
            state.sections.insert(
                id,
                Section {
                    section_id: id,
                    parent_id: parent,
                    name: format!("S{id}"),
                    section_type: 0,
                    section_order: 0,
                    first_version: 1,
                    last_version: 1,
                },
            );
            state.children.entry(parent).or_default().push(id);
        }
        // Append a leaf under the deepest section so the post-mutation depth
        // is cap + 1.
        let leaf_parent = chain_len as u64;
        let leaf_id = (chain_len + 1) as u64;
        let err = apply_delta(
            &mut state,
            Some(delta_create_section(leaf_parent, leaf_id, "OneTooDeep")),
        )
        .unwrap_err();
        assert!(
            matches!(err, DeltaError::SectionTreeTooDeep { .. }),
            "expected SectionTreeTooDeep, got {err:?}"
        );
        // sequence_num must not advance (item 6).
        assert_eq!(state.sequence_num, 0);
    }

    #[test]
    fn check_section_tree_depth_corrupt_cycle_silently_deduped_matching_read_side() {
        // Symmetry pin: the write-side depth check must apply `visited.insert`
        // BEFORE the depth check, matching `DiffEngine::sections_in_dfs_order`.
        // For a corrupt cycle state where a cycle loops back to a node already
        // visited at a shallow depth, both sides should silently dedupe and
        // accept the state — not reject with SectionTreeTooDeep. Without the
        // visited-first ordering, the write side would happily walk the cycle
        // until depth exceeded MAX_SECTION_TREE_DEPTH and reject, while the
        // read side would dedupe and serve a clean diff — breaking the
        // documented "stored state can never exceed the cap, so the read-side
        // SectionTreeTooDeep becomes unreachable" contract on the
        // function-level doc-comment in `apply_delta`'s b.dmu item 3 block.
        //
        // Build a self-cycle: section 1 with parent_id = 1, listed under both
        // state.children[0] (reachable from root) and state.children[1] (the
        // cycle edge). This state is *deliberately corrupt* — it violates
        // `debug_assert_cross_map_consistent`, which is exactly the legacy-
        // state shape this symmetry contract protects against. We therefore
        // call `check_section_tree_depth` directly rather than going through
        // `apply_delta`, which runs the debug-assert in debug builds. Release
        // builds skip the debug-assert entirely, so this is the realistic
        // path corrupt state would take through production code.
        let mut state = make_state();
        state.sections.insert(
            1,
            Section {
                section_id: 1,
                parent_id: 1, // deliberately corrupt (self-parented)
                name: "Cycle".to_string(),
                section_type: 0,
                section_order: 0,
                first_version: 1,
                last_version: 1,
            },
        );
        // Synthetic root sees 1 as a top-level child (otherwise the DFS walk
        // wouldn't reach it at all — unreachable cycles aren't a concern).
        state.children.insert(0, vec![1]);
        // …and 1 is its own child, forming the cycle. A DFS that checks
        // depth before dedup would push (1, 1) → pop+expand, push (1, 2) →
        // pop+expand, push (1, 3) → … until depth > MAX_SECTION_TREE_DEPTH
        // and return SectionTreeTooDeep. With visited-first the second
        // visit to 1 short-circuits and the walk terminates cleanly.
        state.children.insert(1, vec![1]);

        check_section_tree_depth(&state)
            .expect("write-side must silently dedupe corrupt cycles, matching read-side semantics");
    }

    // ── b.dmu item 5: apply_delta partial-mutation contract pin ──────────────
    //
    // The function-level doc-comment promises:
    //   "On Err(_), `state` may have been partially mutated — caller must
    //    discard the partially-mutated state and reload a fresh snapshot."
    //
    // This test pins that contract end-to-end: build a delta with one
    // valid CreateSection followed by one invalid command (a second
    // CreateSection reusing the id the first one just made live, which fails
    // at apply time). After the call returns Err, observe that the valid
    // CreateSection's effect IS present in `state.sections` (proving the
    // partial-mutation behaviour) while `state.sequence_num` has NOT advanced
    // (proving the item 6 single-field guarantee).

    #[test]
    fn apply_delta_partial_mutation_contract_pins_documented_behaviour() {
        let mut state = make_state();
        let pre_seq = state.sequence_num;

        // First command: valid CreateSection under root, section_id=300. Will
        // succeed and mutate `state.sections` + `state.children[0]`.
        // Second command: a CreateSection reusing the now-live id 300 — fails
        // with DuplicateSection, a "command N is invalid" case from item 6
        // that only the state the first command left behind can reject.
        let delta = OwnedDelta {
            is_diff: false,
            struct_commands: vec![
                StructCommand::CreateSection(OodfCreateSection {
                    parent_id: 0,
                    section_id: 300,
                    section_name: "Doomed".to_string(),
                    section_type: 0,
                }),
                StructCommand::CreateSection(OodfCreateSection {
                    parent_id: 0,
                    section_id: 300,
                    section_name: "Reused".to_string(),
                    section_type: 0,
                }),
            ],
            ..Default::default()
        };
        let err = apply_delta(&mut state, Some(delta)).unwrap_err();
        assert!(
            matches!(err, DeltaError::DuplicateSection { section_id: 300 }),
            "expected DuplicateSection(300), got {err:?}"
        );

        // Contract clause 1 (partial mutation): the valid CreateSection's
        // effect is observable in `state.sections`. The doc-comment promises
        // this — callers MUST discard the state on Err. This test pins the
        // promise so any future "snapshot-and-swap" refactor (which would
        // also satisfy the contract, just more strongly) is a deliberate
        // public-API change rather than a silent strengthening.
        assert!(
            state.sections.contains_key(&300),
            "valid command's effect is observable on the partially-mutated state \
             (documented contract — callers must discard, not rely on rollback)"
        );

        // Contract clause 2 (item 6 — sequence_num guarantee): the bump is
        // deferred until after all helpers succeed, so a mid-delta error
        // leaves `state.sequence_num` untouched.
        assert_eq!(
            state.sequence_num, pre_seq,
            "sequence_num must NOT advance on mid-delta failure (b.dmu item 6 / AC6)"
        );
    }

    #[test]
    fn apply_delta_error_does_not_advance_sequence_num_on_validate_bounds_failure() {
        // Even the simplest validate_delta_bounds error must leave
        // sequence_num untouched. This was already true pre-b.dmu (the bounds
        // check ran before the bump) but pin it so item 6's contract covers
        // every error path uniformly.
        let mut state = make_state();
        // Self-parenting CreateSection — caught by validate_delta_bounds.
        let err = apply_delta(&mut state, Some(delta_create_section(7, 7, "Cycle"))).unwrap_err();
        assert!(matches!(err, DeltaError::SelfParenting { .. }));
        assert_eq!(state.sequence_num, 0);
    }

    #[test]
    fn apply_delta_error_does_not_advance_sequence_num_on_helper_failure() {
        // A helper-level failure (DuplicateSection from apply_create_section)
        // must also leave sequence_num untouched — this is the rung item 6
        // was specifically called out for.
        let mut state = make_state();
        state.sections.insert(42, make_section(42, 0, "Existing"));
        state.children.insert(0, vec![42]);
        let pre_seq = state.sequence_num;

        let err = apply_delta(&mut state, Some(delta_create_section(0, 42, "Reused"))).unwrap_err();
        assert!(matches!(
            err,
            DeltaError::DuplicateSection { section_id: 42 }
        ));
        assert_eq!(
            state.sequence_num, pre_seq,
            "DuplicateSection (post-validate-delta-bounds) must not advance sequence_num (b.dmu AC6)"
        );
    }

    // ── b.dmu item 1: is_descendant_of via MoveSection cycle-check ───────────
    //
    // The "Special focus" rung: `is_descendant_of` is reached on the WRITE
    // path via `apply_move_section`'s cycle-check (delta.rs:737 — the
    // `is_descendant_of(state, new_parent_id, section_id)` branch).
    // The sibling test `is_descendant_of_self_parent_is_not_descendant_via_
    // visited_set` exercises the visited-set guard in
    // `cascade_delete_section::collect_subtree`, NOT in `is_descendant_of`
    // itself (cascade-delete reaches `collect_subtree`, which has its own
    // visited set that predates b.dmu). This test closes the gap by routing
    // a longer corrupt parent chain through `is_descendant_of` exactly.
    //
    // The fix comment on `is_descendant_of` says: "the VISITED-SET path
    // means a malicious or corrupt parent chain that loops through other
    // ids (not just a single self-parent) is also detected — the helper
    // now treats every cycle uniformly." Pre-fix had a `parent_id !=
    // candidate` guard that only short-circuited the single-step self-loop
    // case; multi-id cycles A→B→A still relied on the visited set. The
    // post-fix code removes the special case, making the walk handle every
    // cycle uniformly through the visited set.
    //
    // Construction: a multi-id parent cycle A=7 → B=8 → A=7 in stored
    // state (legacy pre-validator data — the b.39x SelfParenting validator
    // rejects single self-parent shapes at ingress, but cannot detect
    // longer cycles since the validator does not walk stored state).
    // `MoveSection(section_id=5, new_parent_id=7)` invokes
    // `is_descendant_of(state, 7, 5)` whose walk is `7 → 8 → 7` →
    // visited-set catches the cycle and returns false in finite time.
    //
    // Pre-fix and post-fix both terminate here via the visited-set (the
    // multi-id cycle case is the path the visited set was designed for —
    // only the single-step self-parent had the redundant explicit guard).
    // The value of this test is pinning end-to-end TERMINATION through
    // `is_descendant_of` so a future refactor that re-introduces the
    // silent-absorb arm (or accidentally removes the visited-set) is
    // caught — both regressions would hang the test rather than fail an
    // assertion, but the `cargo test` time budget per case is the safety
    // net.
    #[test]
    fn apply_delta_move_section_cycle_check_via_is_descendant_of_terminates_on_multi_id_cycle() {
        let mut state = make_state();
        // Section 7 and section 8 form a 2-cycle in stored state:
        // 7's parent is 8, 8's parent is 7. This is the multi-id case the
        // fix-comment specifically calls out.
        state.sections.insert(
            7,
            Section {
                section_id: 7,
                parent_id: 8,
                name: "CycleNodeA".to_string(),
                section_type: 0,
                section_order: 0,
                first_version: 1,
                last_version: 1,
            },
        );
        state.sections.insert(
            8,
            Section {
                section_id: 8,
                parent_id: 7,
                name: "CycleNodeB".to_string(),
                section_type: 0,
                section_order: 0,
                first_version: 1,
                last_version: 1,
            },
        );
        // Mirror in state.children so the debug-build cross-map check
        // doesn't complain.
        state.children.insert(8, vec![7]);
        state.children.insert(7, vec![8]);

        // Section 5: well-formed under root.
        state.sections.insert(5, make_section(5, 0, "Movable"));
        state.children.insert(0, vec![5]);

        // MoveSection(section_id=5, old_parent=0, new_parent=7) — the
        // cycle-check `is_descendant_of(state, candidate=7, ancestor=5)`
        // walks candidate=7 → state.sections[7].parent_id=8 → candidate=8
        // → state.sections[8].parent_id=7 → candidate=7 → visited-set
        // catches the cycle → returns false. Termination is the load-
        // bearing property; the cycle is not "5 is an ancestor of 7" so
        // is_descendant_of correctly returns false and the move proceeds.
        let result = apply_delta(
            &mut state,
            Some(delta_move_section(5, 0, 7, "MovedNearCycle")),
        );
        assert!(
            result.is_ok(),
            "MoveSection cycle-check through is_descendant_of on multi-id cycle \
             must terminate, got {result:?}"
        );
        // Section 5 reached new_parent_id=7 normally — the cycle is in 7/8,
        // not "below" 5, so the move is legitimate. The test's value is
        // pinning termination, not the move-result semantics.
        assert_eq!(
            state.sections.get(&5).map(|s| s.parent_id),
            Some(7),
            "section 5 must complete its move to new_parent_id=7 (cycle is not \
             below section 5, so is_descendant_of correctly returns false)"
        );
    }

    // ── SR-5.6: abandoned-prim pruning stays O(live prims) ───────────────────
    //
    // The faithful catch-up builder (`DiffEngine::plan_faithful_snapshot`)
    // derives per-prim liveness at BUILD TIME via a reachable-from-root DFS
    // walk (`live_prim_ids`, the net-live set) — there is NO stored per-prim
    // tombstone/liveness field on `ObjectState`/`Section`/`Field`. The design is
    // pure prune-on-consolidate, so the SR-5.6 bound must hold by construction:
    // after any number of create→abandon (scratch) cycles the per-object maps
    // (`sections` / `children` / `fields` / `time_samples`) hold only the LIVE
    // prims — O(live), never O(cycle-count). These tests apply a long
    // deterministic apply-engine sequence and assert exactly that, so a
    // regression to a one-tombstone-per-cycle design (any per-object structure
    // that grew one entry per abandoned prim) would fail them.
    //
    // Naming: the net-live liveness concept exercised here is `live_prim_ids`
    // (reachable-from-root); it is deliberately DISTINCT from the object-level
    // `superseded` CreateLayer-overwrite marker, which these tests never touch.

    /// SetField delta targeting an arbitrary section, so each scratch prim
    /// carries a field whose removal on delete proves the `fields` map is not a
    /// one-entry-per-cycle accumulator either.
    fn delta_set_field_on(section_id: u64, key: &str, byte: u8) -> OwnedDelta {
        OwnedDelta {
            is_diff: false,
            set_fields: vec![oodf_ffi::SetField {
                section_id,
                key_name: key.to_string(),
                value_or_ext_hash: vec![byte],
                ext_value_size: 0,
                ext_value_index: 0,
                set_order: 0,
            }],
            ..Default::default()
        }
    }

    /// N ≥ 1000 create-then-delete scratch cycles under one persistent live
    /// prim. The per-object maps that track prims must end bounded to the LIVE
    /// prim count (the single root R), independent of the 1000 cycles — the
    /// direct observable of the "no stored per-prim tombstone" design.
    #[test]
    fn faithful_pruning_state_bounded_over_1000_create_delete_scratch_cycles() {
        const N: u64 = 1000;
        let mut state = make_state();

        // One persistent, net-live prim R (id=1) directly under root (parent 0),
        // carrying a field. This is the entire live set for the whole run.
        apply_delta(&mut state, Some(delta_create_section(0, 1, "Root"))).unwrap();
        apply_delta(&mut state, Some(delta_set_field_on(1, "k", 0x11))).unwrap();

        // 1000 scratch cycles: create a fresh scratch prim under R (each with a
        // field), then immediately delete it. Deterministic, no randomness.
        for i in 0..N {
            let scratch_id = 1000 + i;
            apply_delta(
                &mut state,
                Some(delta_create_section(1, scratch_id, "scratch")),
            )
            .unwrap();
            apply_delta(&mut state, Some(delta_set_field_on(scratch_id, "k", 0x22))).unwrap();
            apply_delta(&mut state, Some(delta_delete_section(scratch_id))).unwrap();
        }

        // BOUND (SR-5.6): `sections` holds exactly the 1 live prim, NOT 1 + N.
        // Under a one-tombstone-per-cycle design this map (or a sibling liveness
        // map) would hold ~1001 entries.
        assert_eq!(
            state.sections.len(),
            1,
            "sections map must hold only the 1 live prim after {N} scratch cycles, \
             not accumulate one entry per abandoned prim"
        );
        assert!(state.sections.contains_key(&1), "live prim R must survive");
        for i in 0..N {
            assert!(
                !state.sections.contains_key(&(1000 + i)),
                "no residual scratch prim entry may remain"
            );
        }

        // `children` holds only the two parents that ever gained a child: the
        // root sentinel (0 → [1]) and R (1 → []). Constant, independent of N.
        assert_eq!(
            state.children.len(),
            2,
            "children map keys must be bounded to live parents (root + R), not O(N)"
        );
        assert_eq!(children_of(&state, 0), vec![1]);
        assert!(
            children_of(&state, 1).is_empty(),
            "R's children list must be empty after all scratch prims are deleted"
        );

        // `fields` / `time_samples` likewise bounded to the live prim's content.
        assert_eq!(
            state.fields.len(),
            1,
            "only R's field may survive; scratch fields must not accumulate"
        );
        assert!(state.fields.contains_key(&(1, "k".to_string())));
        assert!(
            state.time_samples.is_empty(),
            "no scratch time samples were added; map stays empty"
        );

        // Ties the bound back to correct output: the consolidated faithful
        // snapshot over the post-churn state emits exactly the net-live prim
        // (via `live_prim_ids`), no dropped live state and no residual scratch.
        match crate::domain::diff::DiffEngine::plan_faithful_snapshot(&state).unwrap() {
            crate::domain::diff::DiffPlan::FullContent {
                sections,
                fields,
                time_samples,
            } => {
                let ids: Vec<u64> = sections.iter().map(|s| s.section_id).collect();
                assert_eq!(
                    ids,
                    vec![1],
                    "faithful snapshot must emit exactly the net-live prim R"
                );
                assert_eq!(fields.len(), 1, "only R's field in the snapshot");
                assert_eq!(fields[0].section_id, 1);
                assert!(time_samples.is_empty());
            }
            other => panic!("expected FullContent, got {other:?}"),
        }
    }

    /// Create-then-reparent variant. Each of N cycles creates a fresh scratch
    /// prim under A, reparents it A→B, then deletes it; on top of that a single
    /// persistent prim P is reparented back and forth every cycle. The reparent
    /// churn must leave no stale duplicate children entries and no duplicated
    /// section rows: only the net-live parentage survives, bounded to O(live).
    #[test]
    fn faithful_pruning_state_bounded_over_1000_create_reparent_cycles() {
        const N: u64 = 1000;
        let mut state = make_state();

        // Two persistent live containers A(1), B(2) under root, plus a
        // persistent prim P(3) created under A that we reparent every cycle.
        apply_delta(&mut state, Some(delta_create_section(0, 1, "A"))).unwrap();
        apply_delta(&mut state, Some(delta_create_section(0, 2, "B"))).unwrap();
        apply_delta(&mut state, Some(delta_create_section(1, 3, "P"))).unwrap();

        for i in 0..N {
            let scratch_id = 1000 + i;
            // create scratch under A, reparent A→B, then delete it.
            apply_delta(
                &mut state,
                Some(delta_create_section(1, scratch_id, "scratch")),
            )
            .unwrap();
            apply_delta(
                &mut state,
                Some(delta_move_section(scratch_id, 1, 2, "scratch")),
            )
            .unwrap();
            apply_delta(&mut state, Some(delta_delete_section(scratch_id))).unwrap();

            // reparent the persistent P: alternate its parent A(1) ↔ B(2).
            let cur_parent = state.sections[&3].parent_id;
            let new_parent = if cur_parent == 1 { 2 } else { 1 };
            apply_delta(
                &mut state,
                Some(delta_move_section(3, cur_parent, new_parent, "P")),
            )
            .unwrap();
        }

        // BOUND: only the 3 net-live prims (A, B, P) survive — never O(N).
        assert_eq!(
            state.sections.len(),
            3,
            "only the 3 live prims (A, B, P) survive {N} reparent cycles"
        );
        for id in [1u64, 2, 3] {
            assert!(state.sections.contains_key(&id));
        }
        for i in 0..N {
            assert!(!state.sections.contains_key(&(1000 + i)));
        }

        // `children` keys bounded to the live parents (root, A, B); P leaves no
        // stale entry under whichever container it is NOT currently under.
        assert_eq!(
            state.children.len(),
            3,
            "children map keys bounded to live parents (root, A, B), not O(N)"
        );
        // P appears in exactly one parent's children list — no stale duplicate.
        let p_parent = state.sections[&3].parent_id;
        assert_eq!(
            p_parent, 1,
            "after {N} (even) alternating reparents P returns under A(1)"
        );
        let p_occurrences = children_of(&state, 1)
            .iter()
            .chain(children_of(&state, 2).iter())
            .filter(|&&c| c == 3)
            .count();
        assert_eq!(
            p_occurrences, 1,
            "reparented prim P must be linked under exactly one parent, no stale entry"
        );

        // Consolidated snapshot: exactly the net-live prims, P under its final
        // (net-live) parent exactly once, no residual scratch parentage.
        match crate::domain::diff::DiffEngine::plan_faithful_snapshot(&state).unwrap() {
            crate::domain::diff::DiffPlan::FullContent { sections, .. } => {
                let ids: Vec<u64> = sections.iter().map(|s| s.section_id).collect();
                let mut sorted = ids.clone();
                sorted.sort_unstable();
                assert_eq!(
                    sorted,
                    vec![1, 2, 3],
                    "faithful snapshot must emit exactly the 3 net-live prims"
                );
                let p = sections.iter().filter(|s| s.section_id == 3).count();
                assert_eq!(p, 1, "P must appear exactly once, never duplicated");
                let sec_p = sections.iter().find(|s| s.section_id == 3).unwrap();
                assert_eq!(
                    sec_p.parent_id, p_parent,
                    "P must sit under its net-live parent, not a stale one"
                );
                let pos = |id: u64| ids.iter().position(|&x| x == id).unwrap();
                assert!(
                    pos(p_parent) < pos(3),
                    "P must follow its final parent in DFS order, got {ids:?}"
                );
            }
            other => panic!("expected FullContent, got {other:?}"),
        }
    }

    /// Boundary / oscillation test: interleave create/abandon so the live prim
    /// count rises to a fixed peak and falls back to baseline over many rounds.
    /// The high-water mark of the tracked per-object state must equal the peak
    /// LIVE set size, NOT the cumulative churn across rounds.
    #[test]
    fn faithful_pruning_high_water_tracks_live_set_not_cumulative_churn() {
        const ROUNDS: u64 = 200;
        const PEAK_SCRATCH: u64 = 5;
        let mut state = make_state();

        // Persistent live root R(1).
        apply_delta(&mut state, Some(delta_create_section(0, 1, "Root"))).unwrap();

        let mut high_water = state.sections.len();
        for r in 0..ROUNDS {
            // Grow: add PEAK_SCRATCH concurrently-live scratch prims under R.
            // Names must be unique within the round — CreateSection cascade-
            // deletes a same-(parent,name) sibling, which would otherwise keep
            // only one scratch prim live at a time.
            for s in 0..PEAK_SCRATCH {
                let scratch_id = 10_000 + r * PEAK_SCRATCH + s;
                apply_delta(
                    &mut state,
                    Some(delta_create_section(1, scratch_id, &format!("scratch{s}"))),
                )
                .unwrap();
            }
            high_water = high_water.max(state.sections.len());

            // Shrink: delete them all — live count returns to baseline.
            for s in 0..PEAK_SCRATCH {
                let scratch_id = 10_000 + r * PEAK_SCRATCH + s;
                apply_delta(&mut state, Some(delta_delete_section(scratch_id))).unwrap();
            }
        }

        // High-water == 1 live root + PEAK_SCRATCH concurrently-live scratch,
        // independent of ROUNDS. A design retaining a record per abandoned prim
        // would push this to O(ROUNDS × PEAK_SCRATCH).
        assert_eq!(
            high_water,
            (1 + PEAK_SCRATCH) as usize,
            "sections high-water must track the peak live set ({}), not cumulative churn ({})",
            1 + PEAK_SCRATCH,
            1 + ROUNDS * PEAK_SCRATCH
        );

        // Back at baseline after the final round: only the live root remains.
        assert_eq!(
            state.sections.len(),
            1,
            "state returns to O(live) baseline after oscillation completes"
        );
        assert_eq!(state.children.len(), 2, "children keys bounded to root + R");

        // Consolidated snapshot over the settled state: exactly the net-live R.
        match crate::domain::diff::DiffEngine::plan_faithful_snapshot(&state).unwrap() {
            crate::domain::diff::DiffPlan::FullContent { sections, .. } => {
                let ids: Vec<u64> = sections.iter().map(|s| s.section_id).collect();
                assert_eq!(ids, vec![1], "only the net-live root R in the snapshot");
            }
            other => panic!("expected FullContent, got {other:?}"),
        }
    }

    // ═════════════════════════════════════════════════════════════════════════
    // b.ot3 — bounded structural delta planning
    //
    // Three families live below, and they assert different things:
    //
    // * **Counts** — the emitted child-id payload, the number of children-list
    //   writes, and the four `ApplyDeltaStats` operation counters. These are
    //   the acceptance criteria: "N siblings retain N child ids, not
    //   N(N+1)/2", "one final rebuild, not N". Deterministic integers, never
    //   wall-clock: SDD §19 and the test-writing guide both put timing in
    //   `benches/`, and a loaded runner must not be able to block a release.
    // * **Ordering** — where the coalesced children writes and the deferred
    //   cascade purge sit relative to the mutations around them. Coalescing
    //   moved two emissions later in the list, and "later" is only safe up to
    //   a point that the storage layer's folds depend on.
    // * **Cache parity** — the per-delta `ParentCache` / `NameIndex` must
    //   answer exactly what the pre-b.ot3 per-command rescan answered. The
    //   oracle for that is [`apply_one_command_per_delta`]: one command per
    //   delta is one command per *plan*, so every cache is built from the live
    //   maps and cannot be stale. Several shapes below diverged between the
    //   batched and the one-per-delta answer in intermediate revisions of the
    //   fix; each is pinned with the discriminating value spelled out.
    // ═════════════════════════════════════════════════════════════════════════

    // ── Shared b.ot3 helpers ─────────────────────────────────────────────────

    /// Apply `delta` and hand back both the mutation list and the stats.
    fn apply_counted(
        state: &mut ObjectState,
        delta: OwnedDelta,
    ) -> (Vec<StorageMutation>, ApplyDeltaStats) {
        let mut stats = ApplyDeltaStats::default();
        let mutations =
            apply_delta_with_stats(state, Some(delta), &mut stats).expect("delta must apply");
        (mutations, stats)
    }

    /// The one-command-per-delta oracle.
    ///
    /// Replays `delta` as a sequence of single-primitive deltas: the reset (if
    /// the delta carries one) alone, then one delta per structural command,
    /// then one per `SetField` and per `TimeSample` — the same phase order
    /// `apply_delta_into` uses internally.
    ///
    /// Why this is the right oracle for the b.ot3 caches: every per-delta
    /// planning structure (`ChildPlan`, its `ParentCache`s, the `NameIndex`)
    /// is created at `apply_delta_into` entry and dropped with the frame, so a
    /// one-command delta always derives its caches from the live maps at the
    /// moment of use. That is by construction the pre-b.ot3 behaviour, where
    /// every command re-read `state.sections` / `state.children` directly. Any
    /// divergence between a batched delta and this replay is therefore a cache
    /// that went stale under a write it did not notice.
    ///
    /// Version stamps are **not** comparable across the two runs (N deltas
    /// advance `sequence_num` N times where one advances it once, and
    /// `last_version` / `first_version` / `Field::set_order` follow), which is
    /// why comparisons go through [`structural_shape`] rather than
    /// `assert_eq!` on the whole `ObjectState`.
    fn apply_one_command_per_delta(state: &mut ObjectState, delta: &OwnedDelta) {
        if delta.is_diff && delta.base_version == 0 {
            apply_delta(
                state,
                Some(OwnedDelta {
                    is_diff: true,
                    base_version: 0,
                    source_format: delta.source_format.clone(),
                    ..Default::default()
                }),
            )
            .expect("reset-only reference delta must apply");
        }
        for cmd in &delta.struct_commands {
            apply_delta(
                state,
                Some(OwnedDelta {
                    is_diff: false,
                    struct_commands: vec![cmd.clone()],
                    ..Default::default()
                }),
            )
            .expect("single-command reference delta must apply");
        }
        for sf in &delta.set_fields {
            apply_delta(
                state,
                Some(OwnedDelta {
                    is_diff: false,
                    set_fields: vec![sf.clone()],
                    ..Default::default()
                }),
            )
            .expect("single-field reference delta must apply");
        }
        for ts in &delta.time_samples {
            apply_delta(
                state,
                Some(OwnedDelta {
                    is_diff: false,
                    time_samples: vec![ts.clone()],
                    ..Default::default()
                }),
            )
            .expect("single-sample reference delta must apply");
        }
    }

    /// Version-stripped projection of an `ObjectState`, for comparing a
    /// batched delta against its [`apply_one_command_per_delta`] replay.
    ///
    /// Carries everything the b.ot3 change can affect — section rows (id,
    /// parent, name, type, **order**), the children lists **in order** (the
    /// whole point), and which field / time-sample rows survive — and nothing
    /// that legitimately differs between one delta and N (`sequence_num`,
    /// `first_version`, `last_version`, the `set_order` cursor).
    #[derive(Debug, PartialEq, Eq)]
    struct StructuralShape {
        sections: Vec<(u64, u64, String, u8, u32)>,
        children: Vec<(u64, Vec<u64>)>,
        field_keys: Vec<(u64, String)>,
        time_sample_keys: Vec<(u64, OrderedFloat<f64>)>,
    }

    fn structural_shape(state: &ObjectState) -> StructuralShape {
        let mut sections: Vec<(u64, u64, String, u8, u32)> = state
            .sections
            .values()
            .map(|s| {
                (
                    s.section_id,
                    s.parent_id,
                    s.name.clone(),
                    s.section_type,
                    s.section_order,
                )
            })
            .collect();
        sections.sort();
        let mut children: Vec<(u64, Vec<u64>)> = state
            .children
            .iter()
            .map(|(p, c)| (*p, c.clone()))
            .collect();
        children.sort();
        let mut field_keys: Vec<(u64, String)> = state.fields.keys().cloned().collect();
        field_keys.sort();
        let mut time_sample_keys: Vec<(u64, OrderedFloat<f64>)> =
            state.time_samples.keys().copied().collect();
        time_sample_keys.sort();
        StructuralShape {
            sections,
            children,
            field_keys,
            time_sample_keys,
        }
    }

    /// Every `SetChildren` in emission order, as `(parent_id, children)`.
    fn set_children_writes(mutations: &[StorageMutation]) -> Vec<(u64, Vec<u64>)> {
        mutations
            .iter()
            .filter_map(|m| match m {
                StorageMutation::SetChildren {
                    section_id,
                    children,
                } => Some((*section_id, children.clone())),
                _ => None,
            })
            .collect()
    }

    /// Every `DeleteChildrenList` in emission order.
    fn deleted_children_lists(mutations: &[StorageMutation]) -> Vec<u64> {
        mutations
            .iter()
            .filter_map(|m| match m {
                StorageMutation::DeleteChildrenList { section_id } => Some(*section_id),
                _ => None,
            })
            .collect()
    }

    /// Total child ids retained across every emitted `SetChildren` — the
    /// quantity the ticket measured at 32,004,000 for 8,000 siblings.
    fn retained_child_ids(mutations: &[StorageMutation]) -> usize {
        mutations
            .iter()
            .map(|m| match m {
                StorageMutation::SetChildren { children, .. } => children.len(),
                _ => 0,
            })
            .sum()
    }

    fn create_cmd(parent_id: u64, section_id: u64, name: &str) -> StructCommand {
        StructCommand::CreateSection(OodfCreateSection {
            parent_id,
            section_id,
            section_name: name.to_string(),
            section_type: 1,
        })
    }

    fn change_cmd(
        parent_id: u64,
        section_id: u64,
        name: &str,
        section_order: u32,
    ) -> StructCommand {
        StructCommand::DiffChangeSection(OodfDiffChangeSection {
            parent_id,
            section_id,
            section_name: name.to_string(),
            section_type: 1,
            section_order,
        })
    }

    fn delete_cmd(section_id: u64) -> StructCommand {
        StructCommand::DeleteSection(OodfDeleteSection {
            parent_id: 0,
            section_id,
        })
    }

    fn move_cmd(
        section_id: u64,
        old_parent_id: u64,
        new_parent_id: u64,
        new_name: &str,
    ) -> StructCommand {
        StructCommand::MoveSection(OodfMoveSection {
            old_parent_id,
            new_parent_id,
            section_id,
            new_name: new_name.to_string(),
        })
    }

    fn reorder_cmd(parent_id: u64, children: Vec<u64>) -> StructCommand {
        StructCommand::ReorderChildren(OodfReorderChildren {
            section_id: parent_id,
            children_list_id: 0,
            children_list: children,
        })
    }

    /// A non-reset delta carrying `cmds` in order.
    fn batch(cmds: Vec<StructCommand>) -> OwnedDelta {
        OwnedDelta {
            is_diff: false,
            struct_commands: cmds,
            ..Default::default()
        }
    }

    /// Install a section row (does not list it under its parent), bypassing
    /// the command engine; callers arrange `state.children` themselves.
    ///
    /// Several cache-drift shapes below are only reachable from state that was
    /// *stored* before a later fix (a children list holding the same id twice,
    /// an id listed under a parent that is not its own) — a snapshot read
    /// `LRANGE`s such a list back verbatim, so the engine must handle it, but no
    /// sequence of commands through this engine produces it. Building the
    /// `ObjectState` directly is the only way to arrange them.
    fn seed_section(state: &mut ObjectState, id: u64, parent: u64, name: &str, order: u32) {
        state.sections.insert(
            id,
            Section {
                section_id: id,
                parent_id: parent,
                name: name.to_string(),
                section_type: 1,
                section_order: order,
                first_version: 1,
                last_version: 1,
            },
        );
    }

    /// Install a field row on `section_id`, bypassing the command engine.
    fn seed_field(state: &mut ObjectState, section_id: u64, key_name: &str) {
        state.fields.insert(
            (section_id, key_name.to_string()),
            Field {
                section_id,
                key_name: key_name.to_string(),
                value: FieldValue::Inline(vec![1]),
                set_order: 0,
                authoring_set_order: 0,
                version: 1,
            },
        );
    }

    /// Install a time-sample row on `section_id`, bypassing the command engine.
    fn seed_time_sample(state: &mut ObjectState, section_id: u64, time: f64) {
        state.time_samples.insert(
            (section_id, OrderedFloat(time)),
            TimeSample {
                section_id,
                time: OrderedFloat(time),
                value: FieldValue::Inline(vec![2]),
                version: 1,
            },
        );
    }

    // ── Allocation and operation counts (acceptance criteria 1) ──────────────

    /// The headline acceptance criterion: creating N siblings under a
    /// previously empty root retains **N** child ids across the emitted
    /// `SetChildren`s, not N(N+1)/2, and rebuilds the list **once**, not N
    /// times.
    ///
    /// The ticket's measured pre-fix figures are the comparison — 8,000
    /// siblings retained 32,004,000 ids / 256,032,000 bytes of `u64` child
    /// vectors from a 296,088-byte delta. The target it states is 8,000 ids /
    /// 64,000 bytes / one rebuild, and that is what is asserted here, at all
    /// four sibling counts so the growth is visibly linear rather than
    /// quadratic at a single point.
    ///
    /// `children_element_moves == 0` is the second half of the story:
    /// `CommandCreateSection` appends — each sibling takes the key one above
    /// the largest already listed (`0, 1, 2, …` here) and lands last. No
    /// element is ever shifted, and on a sorted list the append key reads the
    /// last cached order and the insertion point is a binary search, so
    /// `children_scan_steps == 0` too. (Descending insertion is the shape that
    /// *does* pay element moves — see the test below it.)
    #[test]
    fn apply_delta_n_siblings_under_empty_root_retain_linear_child_ids_and_one_write() {
        for n in [1000_u64, 2000, 4000, 8000] {
            let mut state = make_state();
            let delta = batch(
                (1..=n)
                    .map(|i| create_cmd(0, i, &format!("s{i}")))
                    .collect(),
            );
            let (mutations, stats) = apply_counted(&mut state, delta);

            let writes = set_children_writes(&mutations);
            assert_eq!(
                writes.len(),
                1,
                "n={n}: exactly one final children rebuild, not one per sibling"
            );
            assert_eq!(writes[0].0, 0, "n={n}: the rebuilt list is the root's");
            assert_eq!(
                writes[0].1.len() as u64,
                n,
                "n={n}: the single write carries the final list"
            );
            assert_eq!(
                retained_child_ids(&mutations) as u64,
                n,
                "n={n}: retained child ids must be N, not N(N+1)/2 = {}",
                n * (n + 1) / 2
            );
            assert_eq!(
                stats.children_writes_emitted, 1,
                "n={n}: one children-list persistence mutation"
            );
            assert_eq!(
                stats.name_index_builds, 1,
                "n={n}: the (parent, name) index is built once per delta, not rescanned per command"
            );
            assert_eq!(
                stats.name_index_probes, n,
                "n={n}: CreateSection probes exactly once past its own gates"
            );
            assert_eq!(
                stats.children_element_moves, 0,
                "n={n}: appended siblings land last; no element is ever shifted"
            );
            assert_eq!(
                stats.children_scan_steps, 0,
                "n={n}: insertions only, sorted list — neither the append key nor the \
                 binary-search arm scans anything"
            );
            assert_eq!(
                state.children[&0].len() as u64,
                n,
                "n={n}: state carries every sibling"
            );
        }
    }

    /// The ticket states the 8,000-sibling target in bytes as well as ids:
    /// 64,000 bytes (~62.5 KiB) of `u64` child vector, against the measured
    /// 256,032,000. Pinned separately from the loop above so a regression
    /// reads in the unit the ticket and the pod memory budget are written in.
    #[test]
    fn apply_delta_8000_siblings_retain_64000_bytes_of_child_vectors() {
        const N: u64 = 8000;
        const U64_BYTES: usize = std::mem::size_of::<u64>();

        let mut state = make_state();
        let delta = batch(
            (1..=N)
                .map(|i| create_cmd(0, i, &format!("s{i}")))
                .collect(),
        );
        let (mutations, _) = apply_counted(&mut state, delta);

        let child_vector_bytes = retained_child_ids(&mutations) * U64_BYTES;
        assert_eq!(
            child_vector_bytes, 64_000,
            "8k siblings must retain 64,000 bytes of child vectors (pre-fix: 256,032,000)"
        );
    }

    /// The acceptance criteria ask for the expected count "when modifying
    /// preexisting children" too: the one final write carries the *whole*
    /// settled list, so N new siblings under a parent that already held M
    /// retain M + N ids — still one write, still linear, and still no element
    /// moves because every `CreateSection` appends past the largest key.
    #[test]
    fn apply_delta_siblings_under_non_empty_parent_retain_existing_plus_new_ids() {
        const M: u64 = 250;
        const N: u64 = 1000;

        let mut state = make_state();
        apply_delta(
            &mut state,
            Some(batch(
                (1..=M)
                    .map(|i| create_cmd(0, i, &format!("old{i}")))
                    .collect(),
            )),
        )
        .expect("arrangement delta must apply");

        let (mutations, stats) = apply_counted(
            &mut state,
            batch(
                (M + 1..=M + N)
                    .map(|i| create_cmd(0, i, &format!("new{i}")))
                    .collect(),
            ),
        );

        let writes = set_children_writes(&mutations);
        assert_eq!(writes.len(), 1, "still exactly one final rebuild");
        assert_eq!(
            writes[0].1.len() as u64,
            M + N,
            "the final write carries the pre-existing children plus the new ones"
        );
        assert_eq!(retained_child_ids(&mutations) as u64, M + N);
        assert_eq!(stats.children_writes_emitted, 1);
        assert_eq!(stats.children_element_moves, 0);
        assert_eq!(stats.name_index_builds, 1);
    }

    /// Descending-order insertion is the shape that pays the documented
    /// residual **R1**: each new row sorts ahead of every sibling already in
    /// the list, so `Vec::insert(0)` shifts the whole list, N(N-1)/2 element
    /// moves for N commands.
    ///
    /// This test **pins** that residual rather than claiming it away.
    /// Coalescing the emitted mutations does not turn a `Vec` into an
    /// order-statistics tree (see `ApplyDeltaStats::children_element_moves`),
    /// and the fix deliberately does not pretend otherwise — what it removes
    /// is the retained N(N+1)/2 *child ids* and the N *writes*, both of which
    /// are asserted here alongside the residual. If a later change does make
    /// insertion sub-linear, this test fails loudly and is the right place to
    /// re-state the new bound.
    ///
    /// b.sao adds the cache counters to the same picture, and the pairing is
    /// the point: an element-shift residual is *not* a licence to re-derive the
    /// list. Descending insertion shifts N(N-1)/2 elements while deriving the
    /// parent's cache exactly once, over the empty list it starts from — the
    /// shifts are a genuine `Vec` cost, the re-derivations were not. Asserting
    /// a linear total-runtime bound on this shape would be false; asserting a
    /// constant cache-derivation bound on it is not.
    #[test]
    fn apply_delta_descending_order_insertions_pin_quadratic_element_moves_and_one_write() {
        for n in [16_u64, 64, 256] {
            let mut state = make_state();
            // section_order strictly descending: N, N-1, … 1. All non-zero, so
            // every command is an order-side write.
            let delta = batch(
                (0..n)
                    .map(|i| change_cmd(0, i + 1, &format!("d{i}"), (n - i) as u32))
                    .collect(),
            );
            let (mutations, stats) = apply_counted(&mut state, delta);

            assert_eq!(
                stats.children_element_moves,
                n * (n - 1) / 2,
                "n={n}: residual R1 — each insertion at position 0 shifts the whole list"
            );
            assert_eq!(
                stats.children_writes_emitted, 1,
                "n={n}: one children write regardless of how many insertions happened"
            );
            assert_eq!(stats.name_index_builds, 1, "n={n}: one index build");
            assert_eq!(
                stats.children_scan_steps, 0,
                "n={n}: the list stays sorted, so every insertion point is a binary search"
            );
            assert_eq!(
                stats.parent_cache_builds, 1,
                "n={n}: the shifts are a residual; re-deriving the cache is not. One \
                 derivation for the whole delta, however many elements moved"
            );
            assert_eq!(
                stats.parent_cache_entries_built, 0,
                "n={n}: that derivation reads the root's list while it is still empty"
            );
            assert_eq!(
                stats.cache_dependency_steps, 0,
                "n={n}: each written child is brand new, so no cache lists it"
            );
            assert_eq!(
                stats.cache_index_peak_pairs, n,
                "n={n}: one pair per insertion — shifting elements does not store \
                 anything extra in the index"
            );
            assert_eq!(
                set_children_writes(&mutations).len(),
                1,
                "n={n}: one SetChildren"
            );
            assert_eq!(
                retained_child_ids(&mutations) as u64,
                n,
                "n={n}: N retained ids, not N(N+1)/2"
            );

            // Ascending by section_order: the reverse of the command order.
            let expected: Vec<u64> = (0..n).rev().map(|i| i + 1).collect();
            assert_eq!(state.children[&0], expected, "n={n}: final order");
        }
    }

    /// Repeated sibling removal. Deleting the whole list front-first costs one
    /// `position` step per removal (the child is always at index 0) and shifts
    /// the remainder each time — residual **R2**, N(N-1)/2 element moves — and
    /// emits exactly one children write per deleted section (its own
    /// `DeleteChildrenList`) plus the one final rebuild of the parent.
    ///
    /// `name_index_builds == 0` is the second contract here: `DeleteSection`
    /// never probes `(parent_id, name)`, so a delete-only delta must not pay
    /// for the index at all — the lazy build is what makes that true.
    #[test]
    fn apply_delta_repeated_sibling_removal_costs_one_scan_step_per_removal() {
        const N: u64 = 64;

        let mut state = make_state();
        apply_delta(
            &mut state,
            Some(batch(
                (1..=N)
                    .map(|i| create_cmd(0, i, &format!("s{i}")))
                    .collect(),
            )),
        )
        .expect("arrangement delta must apply");

        let (mutations, stats) =
            apply_counted(&mut state, batch((1..=N).map(delete_cmd).collect()));

        assert_eq!(
            stats.children_scan_steps, N,
            "front-first removal finds the child at index 0 every time: one step each"
        );
        assert_eq!(
            stats.children_element_moves,
            N * (N - 1) / 2,
            "residual R2 — each removal shifts the remainder of the list"
        );
        assert_eq!(
            stats.children_writes_emitted,
            N + 1,
            "one DeleteChildrenList per deleted section, plus the root's single rebuild"
        );
        assert_eq!(
            stats.name_index_builds, 0,
            "a delete-only delta never probes (parent, name), so the index is never built"
        );
        assert_eq!(stats.name_index_probes, 0);

        // The root's list emptied. The empty vector is EMITTED, not dropped:
        // at the storage layer an empty `SetChildren` is the DEL-without-RPUSH
        // plus `SREM obj:chs` fork, and dropping it would strand the root in
        // the children index (SDD §7.7).
        assert_eq!(
            set_children_writes(&mutations),
            vec![(0, Vec::new())],
            "exactly one children write for the root, carrying the empty final list"
        );
        assert_eq!(
            deleted_children_lists(&mutations).len() as u64,
            N,
            "one children-list delete per cascade-deleted section"
        );
    }

    /// Removing a child that is not in the parent's list costs **zero** scan
    /// steps and zero element moves — the membership set answers it in O(1).
    ///
    /// The shape is a section whose row says `parent_id = 0` while the root's
    /// list does not mention it: a stale listing the engine tolerates (SDD
    /// §10.4) and a snapshot read can hand back. Deleting it must not
    /// walk the parent's list looking for it.
    #[test]
    fn apply_delta_removing_an_unlisted_child_costs_no_scan_steps() {
        let mut state = make_state();
        seed_section(&mut state, 1, 0, "A", 0);
        seed_section(&mut state, 2, 0, "B", 0);
        // Section 3 claims the root as its parent but is absent from its list.
        seed_section(&mut state, 3, 0, "Unlisted", 0);
        state.children.insert(0, vec![1, 2]);

        let (_, stats) = apply_counted(&mut state, batch(vec![delete_cmd(3)]));

        assert_eq!(
            stats.children_scan_steps, 0,
            "the absent case is answered by the membership set, not a list walk"
        );
        assert_eq!(stats.children_element_moves, 0, "nothing was shifted");
        assert_eq!(
            state.children[&0],
            vec![1, 2],
            "the parent's list is untouched"
        );
        assert!(!state.sections.contains_key(&3), "the row is still deleted");
    }

    /// A list stored before reorders re-keyed their children is not sorted by
    /// `section_order` — a `ReorderChildren` then stored the client's order
    /// with the old keys — and the binary-search insertion arm would land
    /// somewhere the pre-b.ot3 linear scan did not. The cache's `sorted` flag
    /// routes such a list to the linear arm instead; `children_scan_steps` is
    /// how that choice is observable. No command now leaves a list unsorted, so
    /// the list is seeded as the legacy state a snapshot read reads back.
    ///
    /// The value being pinned is the resulting order, against the
    /// one-command-per-delta oracle.
    ///
    /// b.sao adds the cache-cost side of the same shape: the first insertion
    /// derives the parent's cache once, over the three seeded entries, and the
    /// second reads that one derivation rather than forcing another — a
    /// scrambled list is not a reason to re-derive, only a reason to take the
    /// linear insertion arm.
    #[test]
    fn apply_delta_insertion_into_a_legacy_unsorted_list_matches_one_per_delta() {
        fn arrange() -> ObjectState {
            let mut state = make_state();
            seed_section(&mut state, 1, 0, "A", 1);
            seed_section(&mut state, 2, 0, "B", 2);
            seed_section(&mut state, 3, 0, "C", 3);
            // Legacy: the list's orders read [3, 1, 2].
            state.children.insert(0, vec![3, 1, 2]);
            state
        }

        let delta = batch(vec![
            // Order 2 — the linear scan stops at the leading order-3 entry.
            change_cmd(0, 4, "D", 2),
            // Order 5 — greater than every entry, so the scan runs off the end.
            change_cmd(0, 5, "E", 5),
        ]);

        let mut batched = arrange();
        let (mutations, stats) = apply_counted(&mut batched, delta.clone());

        let mut reference = arrange();
        apply_one_command_per_delta(&mut reference, &delta);

        assert_eq!(
            structural_shape(&batched),
            structural_shape(&reference),
            "batched application must equal the one-command-per-delta replay"
        );
        assert_eq!(
            batched.children[&0],
            vec![4, 3, 1, 2, 5],
            "insertion points come from the verbatim (unsorted) order"
        );
        assert_eq!(
            stats.children_scan_steps, 5,
            "one linear scan stopping at index 0 (1 step) plus one running off a 4-element list"
        );
        assert_eq!(
            stats.children_element_moves, 3,
            "inserting at index 0 of a 3-element list shifts 3; the append shifts none"
        );
        assert_eq!(
            stats.children_writes_emitted, 1,
            "two insertions under one parent are still one write"
        );
        assert_eq!(
            stats.parent_cache_builds, 1,
            "the first insertion's derivation is the only one: the second insertion \
             into the unsorted list reads that cache, not a fresh one"
        );
        assert_eq!(
            stats.parent_cache_entries_built, 3,
            "the one derivation reads the three seeded entries"
        );
        assert_eq!(
            stats.cache_dependency_steps, 0,
            "both insertions write a brand-new row, so no cache lists the written child"
        );
        assert_eq!(set_children_writes(&mutations).len(), 1);
    }

    /// A mixed structural sequence — reset, create, order change, reorder,
    /// create after the reorder, cross-parent move, same-parent rename,
    /// cascade delete — in one delta, against the one-command-per-delta
    /// replay.
    ///
    /// This is the broad equivalence net. The per-shape tests below it pin
    /// specific cache-staleness bugs with a spelled-out discriminating value;
    /// this one exists to catch the shapes nobody thought to enumerate.
    ///
    /// The children-write accounting is asserted as an identity rather than a
    /// bare number: `children_writes_emitted` counts the inline
    /// `DeleteChildrenList`s (reset, cascade) **and** the finalizer's writes,
    /// so it must equal the total number of children mutations in the list —
    /// and the finalizer must contribute exactly one `SetChildren` per dirty
    /// parent whose list survived.
    #[test]
    fn apply_delta_mixed_structural_sequence_matches_one_per_delta_and_writes_once_per_parent() {
        fn arrange() -> ObjectState {
            let mut state = make_state();
            seed_section(&mut state, 1, 0, "P1", 0);
            seed_section(&mut state, 2, 0, "P2", 0);
            seed_section(&mut state, 3, 1, "C", 0);
            state.children.insert(0, vec![1, 2]);
            state.children.insert(1, vec![3]);
            state
        }

        let delta = OwnedDelta {
            is_diff: true,
            base_version: 0,
            source_format: "usda".to_string(),
            struct_commands: vec![
                create_cmd(0, 1, "A"),
                create_cmd(0, 2, "B"),
                create_cmd(1, 3, "A1"),
                create_cmd(1, 4, "A2"),
                // Order change within one parent → rebuild_children_list.
                change_cmd(1, 4, "A2", 5),
                // Reorder parent 1's complete list.
                reorder_cmd(1, vec![4, 3]),
                // Create after the reorder.
                create_cmd(1, 5, "A3"),
                // Cross-parent move.
                move_cmd(3, 1, 2, "Moved"),
                // Same-parent rename (no conflict check, SDD §9.3.1).
                move_cmd(5, 1, 1, "A3renamed"),
                // Cascade delete of a leaf.
                delete_cmd(4),
            ],
            ..Default::default()
        };

        let mut batched = arrange();
        let (mutations, stats) = apply_counted(&mut batched, delta.clone());

        let mut reference = arrange();
        apply_one_command_per_delta(&mut reference, &delta);

        assert_eq!(
            structural_shape(&batched),
            structural_shape(&reference),
            "batched application must equal the one-command-per-delta replay"
        );

        let writes = set_children_writes(&mutations);
        let mut write_parents: Vec<u64> = writes.iter().map(|(p, _)| *p).collect();
        write_parents.sort();
        assert_eq!(
            write_parents,
            vec![0, 1, 2],
            "exactly one SetChildren per dirty parent whose list survived the delta"
        );
        assert_eq!(
            stats.children_writes_emitted as usize,
            writes.len() + deleted_children_lists(&mutations).len(),
            "the counter accounts for every children mutation, inline deletes included"
        );
    }

    /// Growth over **distinct** parents must be linear.
    ///
    /// N `DiffChangeSection` new rows, each under its own parent at a non-zero
    /// `section_order`, is the shape an intermediate revision of this fix made
    /// quadratic: it invalidated caches by walking every parent the delta had
    /// touched on every order-side write, which is Θ(N²) for N distinct
    /// parents — and `MAX_DELTA_ENTRIES` admits 99,999 commands in one
    /// synchronous block. The shipped design walks only the parents that
    /// actually list the written child, via [`ChildPlan::listed_in`].
    ///
    /// **Scope.** It pins *output* linearity, exhaustively: every field of
    /// `ApplyDeltaStats` and the emitted child-id payload grow at most one
    /// step per command as N grows.
    ///
    /// It now also detects a return to the touched-parent walk, which it could
    /// not before [`ApplyDeltaStats::cache_dependency_steps`] existed: the
    /// counter is asserted at `0` here, because each command's written child is
    /// brand new and therefore in no cache, while a walk over every touched
    /// parent would reach N(N−1)/2 inspections. `parent_cache_builds: n` and
    /// `parent_cache_entries_built: 0` pin the other half — one build per fresh
    /// parent, each over an empty list, so no child entry is ever re-read.
    /// A cost trend over N distinct parents, if one is ever wanted, belongs in
    /// `benches/`.
    #[test]
    fn apply_delta_new_rows_under_distinct_parents_stay_linear_in_every_counter() {
        for n in [64_u64, 256, 1024] {
            let mut state = make_state();
            // N live parents, disjoint from the new rows' ids, each a
            // top-level section with no list of its own yet. They must have
            // rows: under a parent with no row the engine keeps no list (SDD
            // §10.4), so the new rows would be orphans and write nothing.
            // The root's list is seeded directly and no command touches it,
            // so it adds no cache build and no write.
            let parents: Vec<u64> = (0..n).map(|i| 1_000_000 + i).collect();
            for &p in &parents {
                seed_section(&mut state, p, 0, &format!("p{p}"), 0);
            }
            state.children.insert(0, parents);
            let delta = batch(
                (0..n)
                    .map(|i| change_cmd(1_000_000 + i, i + 1, &format!("s{i}"), 1))
                    .collect(),
            );
            let (mutations, stats) = apply_counted(&mut state, delta);

            assert_eq!(
                set_children_writes(&mutations).len() as u64,
                n,
                "n={n}: one SetChildren each"
            );
            assert_eq!(
                retained_child_ids(&mutations) as u64,
                n,
                "n={n}: one retained child id per parent — no quadratic term"
            );
            // Exhaustive by construction: a counter added to `ApplyDeltaStats`
            // does not compile here until it is given an expected value, which
            // is what keeps this test's name true as the struct grows.
            assert_eq!(
                stats,
                ApplyDeltaStats {
                    // Exactly one children write per distinct parent.
                    children_writes_emitted: n,
                    // Each insertion lands in an empty list.
                    children_element_moves: 0,
                    // No list is ever scanned.
                    children_scan_steps: 0,
                    // Still one index build for the whole delta.
                    name_index_builds: 1,
                    // Every command births a row, and the adoption index is
                    // still built once, by the first of them.
                    adoption_index_builds: 1,
                    // One conflict probe per command.
                    name_index_probes: n,
                    // No cascade, so no deferred purge pass.
                    field_purge_passes: 0,
                    // `DiffChangeSection` never takes `CreateSection`'s
                    // missing-parent skip arm.
                    create_section_skipped_missing_parent: 0,
                    // One cache build per distinct parent — each parent is
                    // reached exactly once, and its cache is never dropped
                    // because no later command writes a child it lists.
                    parent_cache_builds: n,
                    // Every one of those builds reads an empty list: the
                    // parent is fresh, so there is nothing to re-derive.
                    parent_cache_entries_built: 0,
                    // Each command's written child is brand new, so no cache
                    // lists it and the invalidation walk inspects nothing. A
                    // walk over every *touched* parent instead would reach
                    // n(n-1)/2 here, which is what this field discriminates.
                    cache_dependency_steps: 0,
                    // One pair per command and none ever dropped: every parent
                    // is fresh (its build reads an empty list), every child is
                    // distinct, and nothing invalidates. The index therefore
                    // ends at its peak, n — one entry per live insertion, which
                    // is the linear term this shape is allowed to hold.
                    cache_index_peak_pairs: n,
                },
                "n={n}: every counter grows at most one step per command"
            );
        }
    }

    /// `parent_cache_builds` and `parent_cache_entries_built` mean exactly one
    /// derivation and exactly the number of children that derivation read.
    ///
    /// Pinned at the bump site rather than through a workload, because the two
    /// counters are only useful if their units are unambiguous: one delta, one
    /// parent whose cache is built once, over a list whose length the test
    /// chose. `c = 0` is included deliberately — a build over an empty list
    /// still counts as a build, which is what lets a reader subtract builds
    /// from entries and get "children re-read", and what stops an
    /// implementation from hiding re-derivations behind empty lists.
    ///
    /// The single command is a new row at order `0` under an existing parent:
    /// its effective order does not change (no row ⇒ ghost `0`, then a live
    /// `0`), so nothing is invalidated and the one build is the lazy one
    /// `insert_child_into_parent` needs.
    #[test]
    fn parent_cache_counters_report_one_build_and_the_listed_children_it_read() {
        const P: u64 = 1;
        const NEW: u64 = 900;

        for c in [0_u64, 1, 5] {
            let mut state = make_state();
            seed_section(&mut state, P, 0, "P", 0);
            state.children.insert(0, vec![P]);
            let existing: Vec<u64> = (1..=c).map(|i| 100 + i).collect();
            for &cid in &existing {
                seed_section(&mut state, cid, P, &format!("c{cid}"), 0);
            }
            state.children.insert(P, existing);

            let (_mutations, stats) =
                apply_counted(&mut state, batch(vec![change_cmd(P, NEW, "new", 0)]));

            assert_eq!(
                stats.parent_cache_builds, 1,
                "c={c}: exactly one parent's cache is derived, exactly once"
            );
            assert_eq!(
                stats.parent_cache_entries_built, c,
                "c={c}: the build read every listed child and no more — this is the \
                 quantity that goes quadratic when a growing list is re-derived per command"
            );
            assert_eq!(
                stats.cache_dependency_steps, 0,
                "c={c}: a new row at order 0 changes no effective order, so no \
                 invalidation walk runs"
            );
        }
    }

    /// `cache_dependency_steps` counts one step per parent the index still
    /// names for the written child — not per touched parent, and not per cache
    /// actually dropped.
    ///
    /// This test reaches `k > 1` through the stale-listing route. The other
    /// route is a cascade that left pairs behind (see the counter's own
    /// doc-comment), which no case here exercises —
    /// `cache_dependency_steps_counts_a_pair_a_cascade_left_behind` below is
    /// that one, and it matters because it needs no stale stored state at all.
    /// Read together, the two are what make the counter's bound "one step per
    /// parent the **index** names" rather than the narrower "one per listing
    /// parent": in this test's arrangement every indexed parent does list the
    /// child, and in that one the cascaded parent lists nothing.
    ///
    /// `k` listing parents are arranged from *stored* state: `C`'s row names
    /// only `P[k-1]`, but every `P[i]` lists it — a stale listing (see
    /// `insert_child_into_parent`), which a snapshot read `LRANGE`s
    /// back verbatim and which no sequence of commands through this engine
    /// produces. The first `k` commands populate each parent's cache; the
    /// order change on `C` then has to find all `k` of them.
    ///
    /// `k = 1` is the cross-map-consistent case and pins the steady-state cost
    /// at one step per order-side write. `k = 2, 3` are what separate "one step
    /// per listing parent" from "one step, always", and they also exercise
    /// `CacheListers::Many`.
    ///
    /// The kept parent (`C`'s previous parent, which is also the one the
    /// command names) is counted as an inspected step even though its cache
    /// survives: the inspection is the work the counter describes.
    ///
    /// The trailing moves exist only to leave the delta cross-map consistent,
    /// so `debug_assert_cross_map_consistent` is satisfied at the boundary.
    /// `C`'s row already names `home`, so each takes `apply_move_section`'s
    /// rename-in-place arm and unlinks `C` from the carried parent's list. None
    /// writes `section_order`, so none of them adds a step.
    #[test]
    fn cache_dependency_steps_counts_one_step_per_listing_parent() {
        const C: u64 = 500;

        for k in [1_u64, 2, 3] {
            let parents: Vec<u64> = (1..=k).collect();
            let home = parents[parents.len() - 1];

            let mut state = make_state();
            for &p in &parents {
                seed_section(&mut state, p, 0, &format!("P{p}"), 0);
            }
            // C's row names `home`, but every parent lists it.
            seed_section(&mut state, C, home, "C", 0);
            state.children.insert(0, parents.clone());
            for &p in &parents {
                state.children.insert(p, vec![C]);
            }

            let mut cmds: Vec<StructCommand> = parents
                .iter()
                .map(|&p| create_cmd(p, 800 + p, &format!("fill{p}")))
                .collect();
            cmds.push(change_cmd(home, C, "C", 50));
            for &p in &parents {
                if p != home {
                    cmds.push(move_cmd(C, p, home, "C"));
                }
            }

            let (_mutations, stats) = apply_counted(&mut state, batch(cmds));

            assert_eq!(
                stats.cache_dependency_steps, k,
                "k={k}: the order write on C inspects exactly the parents whose caches \
                 list it"
            );
            assert_eq!(
                state.children[&home],
                vec![800 + home, C],
                "k={k}: non-vacuity — the order write really landed, so C (order 50) \
                 sorts behind the filler that was appended after it while C still read 0"
            );
        }
    }

    /// The **other** route to `cache_dependency_steps > 1`: a cascade that left
    /// a pair behind, reached from ordinary commands against cross-map
    /// **consistent** stored state.
    ///
    /// `cache_dependency_steps_counts_one_step_per_listing_parent` above reaches
    /// `k > 1` only through a stale stored listing — state no sequence of
    /// commands through this engine produces. That left the counter's documented
    /// bound ("one step per parent the index still names; one in steady state;
    /// above one after a cascade left pairs behind **or** on a stale/ghost
    /// stored listing") half-unpinned, and the half without a test is the one a
    /// reader is most likely to dismiss as unreachable. It is not.
    ///
    /// The mechanism, in the order the four commands exercise it:
    ///
    /// 1. `change_cmd(P, FILL, 0)` builds `P`'s cache over `[X]`, which
    ///    registers the pair `X → P`. The filler is an explicit order-0 row
    ///    rather than a `CreateSection`: a create appends at the next key (1
    ///    here), and the cascade below would then delete a row with a live
    ///    non-zero order — an order-side write on `FILL` that adds a third
    ///    step this test does not count.
    /// 2. `delete_cmd(P)` cascades `P` away. `invalidate_cache` empties `P`'s
    ///    cache slot but deliberately does **not** de-register its children —
    ///    see [`ChildPlan::listed_in`], where removes are required only for the
    ///    cost bound — so `X → P` survives, now naming a parent with no cache.
    ///    `X`'s own row is at order `0`, so its removal is not an order-side
    ///    write and nothing clears the pair on that route either.
    /// 3. `change_cmd(Q, X, 0)` gives the now-rowless `X` a fresh row under the
    ///    live parent `Q` and lists it there, registering `X → Q`. The index
    ///    entry is now `CacheListers::Many([P, Q])` — from ordinary commands,
    ///    with no stale or ghost stored listing anywhere.
    /// 4. `change_cmd(Q, X, 7)` is the order-side write that has to walk it:
    ///    **two** steps, one of which drops an already-absent cache.
    ///
    /// The fifth command is the correctness half: an explicit order-0 insert
    /// that reads `Q`'s kept cache and lands at the position `X`'s **new**
    /// order implies. (A `CreateSection` would append past every key, stale
    /// or fresh, and land last either way.)
    ///
    /// Discriminated by: this is the shape that decides whether the counter's
    /// bound may be written as the superseded "one step per *listing* parent".
    /// It cannot — `P` lists nothing here. A design that de-registered a
    /// cascaded parent's children (removing the leftover pair) would report `1`;
    /// the shipped one reports `2` and pays one wasted no-op drop for it, which
    /// is the trade [`ChildPlan::listed_in`] documents.
    #[test]
    fn cache_dependency_steps_counts_a_pair_a_cascade_left_behind() {
        const P: u64 = 10; // cascaded parent — cached X, then went away
        const Q: u64 = 40; // live parent — X's home afterwards
        const X: u64 = 30; // the child both parents come to name
        const FILL: u64 = 99;
        const LATER: u64 = 77;

        // Fully cross-map consistent: every row names the parent listing it.
        let mut state = make_state();
        seed_section(&mut state, P, 0, "P", 0);
        seed_section(&mut state, Q, 0, "Q", 1);
        seed_section(&mut state, X, P, "X", 0);
        state.children.insert(0, vec![P, Q]);
        state.children.insert(P, vec![X]);
        state.children.insert(Q, Vec::new());

        let (_mutations, stats) = apply_counted(
            &mut state,
            batch(vec![
                change_cmd(P, FILL, "fill", 0),
                delete_cmd(P),
                change_cmd(Q, X, "X", 0),
                change_cmd(Q, X, "X", 7),
                change_cmd(Q, LATER, "later", 0),
            ]),
        );

        assert_eq!(
            stats.cache_dependency_steps, 2,
            "the order write on X inspects both the live parent Q and the leftover \
             pair naming the cascaded parent P — reachable without any stale or ghost \
             stored listing, which is what makes the counter's bound 'one per parent \
             the INDEX names', not 'one per listing parent'"
        );
        assert_eq!(
            stats.cache_index_peak_pairs, 4,
            "the peak is reached INSIDE delete_cmd(P), not at the end of the delta: \
             remove_child_from_parent(0, P) detaches the root's cache over [P, Q], \
             registering P->0 and Q->0 on top of the X->P and FILL->P the first \
             command stored. Four pairs at that instant, before P->0 is de-registered \
             a statement later. Measured by replaying command prefixes: the peak is 2 \
             after command 1 and 4 after command 2, and the rest of the delta reaches \
             4 again without exceeding it"
        );
        assert_eq!(
            state.children[&Q],
            vec![LATER, X],
            "correctness half: the last command reads Q's kept cache and must see X at \
             its new order 7, so the order-0 sibling sorts ahead of it (a Q cache still \
             believing X's order 0 would append and give [X, LATER])"
        );
        assert!(
            !state.sections.contains_key(&P) && !state.sections.contains_key(&FILL),
            "non-vacuity — the cascade really removed P and its filler child"
        );
        assert_eq!(
            state.sections[&X].parent_id, Q,
            "X really was re-homed under the live parent"
        );
    }

    /// `cache_index_peak_pairs` is the largest number of `(child, parent)`
    /// pairs [`ChildPlan::listed_in`] held at any one instant, counted as
    /// stored entries and not as distinct children.
    ///
    /// Pinned at the bump site in two arrangements, because the counter is only
    /// useful if its unit is unambiguous. Both expected peaks are derived from
    /// the delta's own input — initial list lengths plus insertions, minus what
    /// was dropped before the peak instant — rather than read back off a run.
    ///
    /// **Part 1, the plain shape.** One parent whose stored list holds `c`
    /// children, one new child inserted at order `0`: the build stores `c`
    /// pairs and the insertion one more, nothing is dropped, so the peak is
    /// `c + 1`. The root's list is never touched, so it contributes nothing.
    /// `c = 0` is included because a build over an empty list stores no pair
    /// while the insertion still must register one.
    ///
    /// **Part 2, leftovers and two-lister children**, reached from ordinary
    /// commands against cross-map consistent stored state:
    ///
    /// `P` sits under a top-level section `S` and `Q` directly under the root,
    /// so the one list `delete_cmd(P)` removes `P` from is `S`'s. (With `Q`
    /// under the root beside `P`, that removal would build the root's cache
    /// over `[P, Q]` and store a `Q` pair the count below does not include.)
    ///
    /// 1. `change_cmd(P, FILL, 0)` builds `P`'s cache over `[X, Y]` (two pairs)
    ///    and inserts `FILL` (a third).
    /// 2. `delete_cmd(P)` first removes `P` from `S`'s list, which builds
    ///    `S`'s cache over `[P]` (a fourth pair) and immediately
    ///    de-registers it again — a transient the peak sees. The cascade then
    ///    empties `P`'s cache slot without de-registering its children, per
    ///    [`ChildPlan::listed_in`], so all three of `P`'s pairs are left
    ///    behind naming a parent that no longer has a cache. Every row involved
    ///    sits at order `0`, so no removal is an order-side write and no
    ///    invalidation walk clears them.
    /// 3. `change_cmd(Q, X, 0)` and `change_cmd(Q, Y, 0)` give the now-rowless
    ///    `X` and `Y` fresh rows under the live parent `Q` and list them there.
    ///    Each index entry becomes a `CacheListers::Many` of two parents, and
    ///    the fifth pair takes the index to its peak.
    ///
    /// Discriminated by: the two definitional choices the counter's own
    /// doc-comment makes. Counting distinct **children** instead of pairs peaks
    /// at 4 — the two `Many` entries would collapse to one apiece — and so does
    /// a design that de-registered a cascaded parent's children instead of
    /// leaving the pairs behind, which would empty the index at step 2 and
    /// reach only 4 across the whole delta. The shipped definition reports 5.
    #[test]
    fn cache_index_peak_pairs_counts_stored_pairs_at_their_high_water_mark() {
        // Part 1.
        const PARENT: u64 = 1;
        const NEW: u64 = 900;

        for c in [0_u64, 1, 5] {
            let mut state = make_state();
            seed_section(&mut state, PARENT, 0, "P", 0);
            state.children.insert(0, vec![PARENT]);
            let existing: Vec<u64> = (1..=c).map(|i| 100 + i).collect();
            for &cid in &existing {
                seed_section(&mut state, cid, PARENT, &format!("c{cid}"), 0);
            }
            state.children.insert(PARENT, existing);

            let (_mutations, stats) =
                apply_counted(&mut state, batch(vec![change_cmd(PARENT, NEW, "new", 0)]));

            assert_eq!(
                stats.cache_index_peak_pairs,
                c + 1,
                "c={c}: the one build stores a pair per listed child and the insertion \
                 stores one more; nothing here drops a pair"
            );
        }

        // Part 2.
        const S: u64 = 5; // P's parent — the list the delete takes P out of
        const P: u64 = 10; // cascaded parent — cached X and Y, then went away
        const Q: u64 = 40; // live parent afterwards, listed under the root
        const X: u64 = 30;
        const Y: u64 = 31;
        const FILL: u64 = 99;

        let mut state = make_state();
        seed_section(&mut state, S, 0, "S", 0);
        seed_section(&mut state, Q, 0, "Q", 1);
        seed_section(&mut state, P, S, "P", 0);
        seed_section(&mut state, X, P, "X", 0);
        seed_section(&mut state, Y, P, "Y", 0);
        state.children.insert(0, vec![S, Q]);
        state.children.insert(S, vec![P]);
        state.children.insert(P, vec![X, Y]);

        let (_mutations, stats) = apply_counted(
            &mut state,
            batch(vec![
                change_cmd(P, FILL, "fill", 0),
                delete_cmd(P),
                change_cmd(Q, X, "X", 0),
                change_cmd(Q, Y, "Y", 0),
            ]),
        );

        assert_eq!(
            stats.cache_index_peak_pairs, 5,
            "at the peak the index stores X->P, Y->P, FILL->P (the cascade's leftovers), \
             X->Q and Y->Q (the two live listings) — five stored pairs over three \
             distinct children"
        );
        assert_eq!(
            state.children[&Q],
            vec![X, Y],
            "non-vacuity — both children really were re-homed under the live parent"
        );
        assert!(
            !state.sections.contains_key(&P) && !state.sections.contains_key(&FILL),
            "non-vacuity — the cascade really removed P and its filler child"
        );
    }

    /// The `CacheListers::Many` arm, asserted on **both** listing parents'
    /// lists: an order write on a child two populated caches mirror must leave
    /// each of them answering from the child's new order.
    ///
    /// The two caches take different routes to that answer, which is why both
    /// are checked. `P2` is the row's previous parent, so it is `keep` — its
    /// cache survives with one falsified entry and `rebuild_children_list`
    /// re-places the child. `P1` is a third parent that merely lists the child;
    /// nothing in the command names it, so only the dependency walk reaches it,
    /// and its cache is dropped and re-derived from the live maps.
    ///
    /// Two populated caches naming one child needs a stale stored listing (see
    /// `insert_child_into_parent`; the *other* route, a cascade leftover,
    /// leaves the cascaded parent's cache empty and so cannot show two
    /// re-sorts). The arrangement therefore starts cross-map INCONSISTENT and
    /// the last command repairs it, exactly as the three staleness tests
    /// further down do — which also rules out `apply_one_command_per_delta` as
    /// an oracle here, so both expected orders are spelled out with the
    /// drifted answer named.
    ///
    /// The three insertions are explicit order-0 rows. A `CreateSection` would
    /// append past every key, stale or fresh, and land last either way; the
    /// order-0 insert is what lands differently in front of a stale entry.
    ///
    /// Discriminated by: making `invalidate_order_listeners` drop only the
    /// *last* lister's cache (still counting every lister). Measured: `P1`
    /// keeps its stale `orders == [0, 0]`, reads it as sorted, appends the
    /// order-0 child, and the `P1` assertion below fails with `[N1, N3]`.
    /// Dropping only the *first* lister instead is caught earlier, by the
    /// delta-boundary index audit (`cache index drift`): the walk then stops
    /// before reaching `keep`, whose pair is lost.
    #[test]
    fn apply_delta_order_write_on_a_child_two_caches_list_re_sorts_both_lists() {
        const P1: u64 = 1; // stale lister — reached only by the dependency walk
        const P2: u64 = 2; // the row's own parent, and therefore `keep`
        const C: u64 = 5;
        const N1: u64 = 11;
        const N2: u64 = 12;
        const N3: u64 = 13;

        let mut state = make_state();
        seed_section(&mut state, P1, 0, "P1", 0);
        seed_section(&mut state, P2, 0, "P2", 1);
        // C's row names P2; P1 lists it too — a stale listing.
        seed_section(&mut state, C, P2, "C", 0);
        state.children.insert(0, vec![P1, P2]);
        state.children.insert(P1, vec![C]);
        state.children.insert(P2, vec![C]);

        let (_mutations, stats) = apply_counted(
            &mut state,
            batch(vec![
                // Populates P1's cache with C at order 0.
                change_cmd(P1, N1, "a", 0),
                // Populates P2's cache with C at order 0 — the index entry for
                // C is now `Many([P1, P2])`.
                change_cmd(P2, N2, "b", 0),
                // The order-side write. Same-parent arm, so `keep` is P2.
                change_cmd(P2, C, "C", 50),
                // Reads P1's cache, which the walk above must have dropped.
                change_cmd(P1, N3, "c", 0),
                // Repair: C's row already names P2, so the move renames it in
                // place and unlinks it from the carried parent P1, clearing the
                // stale listing so the delta ends cross-map consistent.
                move_cmd(C, P1, P2, "C"),
            ]),
        );

        assert_eq!(
            stats.cache_dependency_steps, 2,
            "the walk inspects both parents whose caches list C"
        );
        assert_eq!(
            state.children[&P1],
            vec![N3, N1],
            "P1's cache was dropped and re-derived, so it saw C at order 50 and put the \
             order-0 child ahead of it (a surviving stale cache reads orders [0, 0] as \
             sorted, appends, and gives [N1, N3])"
        );
        assert_eq!(
            state.children[&P2],
            vec![N2, C],
            "P2's kept cache had exactly one falsified entry, and rebuild_children_list \
             re-placed C behind the order-0 sibling"
        );
    }

    /// A reset inside the same delta leaves the dependency index describing
    /// nothing, so structure built after it pays the ordinary bounded cost.
    ///
    /// **Scope, stated rather than overclaimed.** `apply_reset` runs before the
    /// structural loop (`apply_delta_into` step 2 precedes step 3), so
    /// `ChildPlan::listed_in` is *always* empty when its `clear()` executes.
    /// That call is therefore a no-op on every reachable path today, and no test
    /// can discriminate it — it is defence against a future edit that gives some
    /// structural command a reset, which would otherwise carry pre-reset pairs
    /// into a wholly new tree. What this test does pin is the observable
    /// contract around it: after a reset wipes `state.children`, the rebuilt
    /// structure derives each parent's cache once, over the empty list the reset
    /// left, and no stale pair makes a later order write inspect a parent that
    /// no longer exists.
    #[test]
    fn apply_delta_reset_then_rebuild_pays_one_cache_build_per_parent() {
        const P: u64 = 1;
        const C1: u64 = 2;
        const C2: u64 = 3;

        // Pre-reset state the delta erases wholesale.
        let mut state = make_state();
        seed_section(&mut state, P, 0, "Old", 0);
        seed_section(&mut state, C1, P, "OldChild", 5);
        state.children.insert(0, vec![P]);
        state.children.insert(P, vec![C1]);

        let (_mutations, stats) = apply_counted(
            &mut state,
            OwnedDelta {
                is_diff: true,
                base_version: 0,
                source_format: "usda".to_string(),
                struct_commands: vec![
                    change_cmd(0, P, "P", 1),
                    change_cmd(P, C1, "C1", 1),
                    change_cmd(P, C2, "C2", 2),
                    // An order change on a child of the rebuilt tree: one step,
                    // naming the one parent whose cache lists it. A pair
                    // surviving the reset would make this inspect more.
                    change_cmd(P, C1, "C1", 9),
                ],
                ..Default::default()
            },
        );

        assert_eq!(
            stats.parent_cache_builds, 2,
            "one derivation for the root and one for P — both over the empty lists the \
             reset left behind"
        );
        assert_eq!(
            stats.parent_cache_entries_built, 0,
            "every derivation reads an empty list: the reset cleared state.children"
        );
        assert_eq!(
            stats.cache_dependency_steps, 1,
            "the one order change on an already-listed child inspects exactly one \
             parent — the steady-state cost"
        );
        assert_eq!(
            state.children[&P],
            vec![C2, C1],
            "non-vacuity — the rebuilt tree really re-sorted C1 behind C2 at order 9"
        );
        assert!(
            !state.sections.values().any(|s| s.name == "OldChild"),
            "non-vacuity — the reset really erased the pre-reset tree"
        );
    }

    /// A delta that populates caches and the dependency index and *then* fails
    /// must return `Err` with `sequence_num` unchanged.
    ///
    /// The point of driving real order-side writes first is that the failure
    /// lands on a plan that is genuinely carrying state — populated caches and a
    /// non-empty `ChildPlan::listed_in` — rather than on the empty plan the
    /// pre-existing partial-mutation tests fail with. What is pinned is the two
    /// clauses the contract promises on that path: `sequence_num` does not
    /// advance, and the earlier commands' effects are observable (callers are
    /// obliged to discard them).
    ///
    /// **Scope — no test can discriminate the audit's placement, and this one
    /// does not claim to.** `debug_assert_cache_index_superset` sits after the
    /// last `?` in `apply_delta_into`, and an earlier revision of this comment
    /// asserted that moving it above the structural loop's `?` would make this
    /// test red. That is false, and the reason is worth recording so the claim
    /// is not reinvented: `apply_create_section` rejects the reused id at its
    /// **first check**, before it touches any list, cache or index entry. At
    /// the failure point `P`'s cache still holds exactly `{C1, C2}` and the
    /// index still names `P` for both, so the audit
    /// would pass wherever it ran. Reaching a genuinely index-inconsistent error
    /// return would need a fallible statement *between* a membership change and
    /// its matching registration, and no such statement exists — the engine's
    /// `?`s are all in validation, never mid-maintenance. The audit's own
    /// discrimination is covered instead by
    /// `debug_assert_cache_index_superset_panics_on_a_cache_the_index_does_not_name`
    /// below, which calls it directly. Same posture as
    /// `apply_delta_reset_then_rebuild_pays_one_cache_build_per_parent`'s note
    /// about the reset clear: state the gap rather than imply coverage.
    #[test]
    fn apply_delta_error_after_order_writes_leaves_sequence_num_unchanged() {
        const P: u64 = 1;
        const C1: u64 = 2;
        const C2: u64 = 3;

        let mut state = make_state();
        seed_section(&mut state, P, 0, "P", 0);
        seed_section(&mut state, C1, P, "C1", 1);
        seed_section(&mut state, C2, P, "C2", 2);
        state.children.insert(0, vec![P]);
        state.children.insert(P, vec![C1, C2]);
        let pre_seq = state.sequence_num;

        let err = apply_delta(
            &mut state,
            Some(batch(vec![
                // Real order-side writes: P's cache is built and C1/C2 are
                // registered in the dependency index before anything fails.
                change_cmd(P, C1, "C1", 50),
                change_cmd(P, C2, "C2", 60),
                // Fails mid-delta, leaving the plan abandoned: C1 is live.
                create_cmd(P, C1, "again"),
            ])),
        )
        .expect_err("a create reusing a live id must fail the delta");

        assert!(
            matches!(err, DeltaError::DuplicateSection { section_id: C1 }),
            "expected DuplicateSection(C1), got {err:?}"
        );
        assert_eq!(
            state.sequence_num, pre_seq,
            "a mid-delta failure must not advance sequence_num, however much cache \
             maintenance already ran"
        );
        assert_eq!(
            state.sections[&C1].section_order, 50,
            "partial-mutation contract: the earlier commands' effects are observable, \
             and the caller is the one obliged to discard them"
        );
    }

    /// `apply_reorder_children`'s de-registration pass, with its loop body
    /// actually executing.
    ///
    /// A reorder over a list holding only the parent's own children drops
    /// nothing, so the loop body never runs — the pass is reached but not
    /// exercised. Dropping an id needs stored state the command engine cannot
    /// produce, because no command lists an id that lacks a row naming the
    /// parent (the reorder itself ignores such ids rather than listing them):
    /// a listed entry that is not the parent's child — here a ghost — has to be
    /// seeded. The reorder below names the whole working list, `[C1, FIRST]`,
    /// so it is not a no-op and the clean-up runs.
    ///
    /// The shape:
    ///
    /// 1. `create_cmd(P, FIRST)` builds `P`'s cache over `[C1, GHOST]`, which
    ///    registers both — a build indexes every listed child, ghosts included,
    ///    because a ghost's effective order (`0`) is a value the cache mirrors
    ///    and an order-side write can change.
    /// 2. `reorder_cmd(P, [C1, FIRST])` omits `GHOST`. `P`'s cache stays
    ///    **populated** across the swap, so the stale pair has to go with it.
    /// 3. `change_cmd(R, GHOST, 50)` gives the ghost a live row at a non-zero
    ///    order under a different parent — an order-side write on exactly the id
    ///    the reorder dropped.
    /// 4. `create_cmd(P, LAST)` reads `P`'s cache again.
    ///
    /// Discriminated by: deleting the loop body at `apply_reorder_children`'s
    /// de-registration pass. The pair `GHOST → P` then survives step 2, so
    /// step 3's walk inspects it and drops `P`'s live cache, which step 4 must
    /// re-derive over `[C1, FIRST]`. **Measured** on that perturbation:
    /// `cache_dependency_steps` 0 → 1, `parent_cache_builds` 3 → 4,
    /// `parent_cache_entries_built` 4 → 6. All three are pure waste: `P` no
    /// longer lists the ghost, so the answer is unchanged either way — which is
    /// precisely why only a cost assertion can see it.
    #[test]
    fn apply_delta_reorder_dropping_a_ghost_id_de_registers_its_index_pair() {
        const P: u64 = 1;
        const R: u64 = 2; // the ghost's eventual home
        const C1: u64 = 10;
        const GHOST: u64 = 77;
        const FIRST: u64 = 20;
        const LAST: u64 = 30;

        let mut state = make_state();
        seed_section(&mut state, P, 0, "P", 0);
        seed_section(&mut state, R, 0, "R", 1);
        seed_section(&mut state, C1, P, "C1", 0);
        state.children.insert(0, vec![P, R]);
        // GHOST is listed under P with no row of its own — stored state the
        // snapshot load `LRANGE`s back verbatim (SDD §10.4).
        state.children.insert(P, vec![C1, GHOST]);
        state.children.insert(R, Vec::new());

        let (_mutations, stats) = apply_counted(
            &mut state,
            batch(vec![
                create_cmd(P, FIRST, "first"),
                reorder_cmd(P, vec![C1, FIRST]),
                change_cmd(R, GHOST, "ghost", 50),
                create_cmd(P, LAST, "last"),
            ]),
        );

        assert_eq!(
            stats.cache_dependency_steps, 0,
            "the reorder dropped GHOST from P's list, so the order write that gives it \
             a row must find no cache naming it — a surviving pair costs one inspection \
             and one wasted rebuild of a list the ghost is no longer in"
        );
        assert_eq!(
            stats.parent_cache_builds, 3,
            "three derivations, all accounted for: P's lazy one for the first \
             insertion, the reorder's wholesale re-derivation of P, and R's when the \
             ghost's new row is listed there. Step 4 re-reads P's surviving cache and \
             adds none"
        );
        assert_eq!(
            stats.parent_cache_entries_built, 4,
            "two entries read by P's first build ([C1, GHOST]) and two by the reorder's \
             re-derivation ([C1, FIRST]); R's build reads its empty list"
        );
        assert_eq!(
            state.children[&P],
            vec![C1, FIRST, LAST],
            "non-vacuity — the reorder really dropped the ghost and the later insert \
             really appended"
        );
        assert_eq!(
            state.children[&R],
            vec![GHOST],
            "non-vacuity — the ghost really gained a live row under R"
        );
    }

    /// The index audit itself, driven directly, so that deleting it turns
    /// something red.
    ///
    /// `debug_assert_cache_index_superset` is the **sole** guard for the unsound
    /// half of the index invariant: the hook-removal table in
    /// `tests/proptests.rs` records that a missing `register_listing` in
    /// `insert_child_into_parent` is caught by nothing else at 256 cases and
    /// nothing else at 20,000 either. An assertion that is the last line of
    /// defence and has no test of its own can be deleted in a refactor without a
    /// single red run, which is the gap this closes — same
    /// `#[cfg(debug_assertions)]` + `#[should_panic]` shape the repo already
    /// uses for `debug_assert!`-guarded invariants elsewhere.
    ///
    /// The plan is built through the production constructor and then has its
    /// index emptied, which is exactly the state a missing registration
    /// produces: a populated cache listing a child no index entry names.
    ///
    /// Discriminated by: deleting the `debug_assert!` inside the audit's member
    /// loop, or the whole function.
    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "cache index drift")]
    fn debug_assert_cache_index_superset_panics_on_a_cache_the_index_does_not_name() {
        const P: u64 = 1;
        const C: u64 = 2;

        let mut plan = ChildPlan::default();
        let mut stats = ApplyDeltaStats::default();
        let mut children = std::collections::HashMap::new();
        children.insert(P, vec![C]);
        let sections = std::collections::HashMap::new();

        let cache = plan.build_cache(&children, &sections, P, &mut stats);
        plan.restore_cache(P, cache);
        // The unsound half, simulated: the cache is populated and lists C, but
        // nothing in the index names P for it.
        plan.listed_in.clear();
        plan.listed_pairs = 0;

        debug_assert_cache_index_superset(&plan);
    }

    /// The audit's other half: the incrementally maintained live pair count must
    /// agree with the pairs actually stored.
    ///
    /// `ChildPlan::listed_pairs` is maintained by hand at every add and drop,
    /// and nothing else in the engine ever re-reads the map to check it —
    /// `ApplyDeltaStats::cache_index_peak_pairs`, the memory-bound counter the
    /// b.sao acceptance criteria are written against, is folded from that value
    /// alone. A count that drifts low reports a peak lower than the truth, and
    /// the bound tests would then pass over a growth they exist to catch. The
    /// `debug_assert_eq!` is the only thing standing between those two facts, so
    /// it gets its own test for the same reason the superset half does.
    ///
    /// Discriminated by: deleting the `debug_assert_eq!` at the end of the
    /// audit.
    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "pair-count drift")]
    fn debug_assert_cache_index_superset_panics_on_a_drifted_live_pair_count() {
        const P: u64 = 1;
        const C: u64 = 2;

        let mut plan = ChildPlan::default();
        let mut stats = ApplyDeltaStats::default();
        let mut children = std::collections::HashMap::new();
        children.insert(P, vec![C]);
        let sections = std::collections::HashMap::new();

        let cache = plan.build_cache(&children, &sections, P, &mut stats);
        plan.restore_cache(P, cache);
        // The index and the caches agree, so the superset half passes; only the
        // hand-maintained count is wrong, which is the drift an unbalanced
        // increment or decrement would produce.
        plan.listed_pairs += 1;

        debug_assert_cache_index_superset(&plan);
    }

    /// `CacheListers::insert`'s dedup guard on the `Many` arm, reached through
    /// ordinary commands.
    ///
    /// A child already in a `Many` entry is re-registered whenever a parent that
    /// lists it re-derives its cache, and `ReorderChildren` is the reachable
    /// route: its `build_cache` re-registers every id in the list it leaves,
    /// while its de-registration pass leaves ids that list keeps alone. If
    /// the guard stopped rejecting the duplicate, the entry would grow a second
    /// copy of the same parent — silently inflating both counters that read the
    /// index's size.
    ///
    /// The `Many` entry is reached the cascade-leftover way (no stale stored
    /// listing needed): `P` caches `X`, `P` is cascaded away leaving the pair
    /// behind, and `X` is then listed under the live parent `Q`. The filler and
    /// the closing probe are explicit order-0 rows for the reasons
    /// `cache_dependency_steps_counts_a_pair_a_cascade_left_behind` gives.
    ///
    /// Discriminated by: replacing the `Many` arm's `if parents.contains(…)
    /// { return false }` with an unconditional `push`. `X`'s entry becomes
    /// `[P, Q, Q]`, so the final order-side write inspects three parents instead
    /// of two (`cache_dependency_steps` 2 → 3) and the index stores one pair
    /// more than it holds children-under-parents (`cache_index_peak_pairs`
    /// 4 → 5). The answer is unaffected either way — dropping `Q`'s cache twice
    /// is idempotent — so, once again, only the counters see it.
    #[test]
    fn apply_delta_re_registering_a_parent_already_in_a_many_entry_stores_no_second_pair() {
        const P: u64 = 10; // cascaded parent — caches X, then goes away
        const Q: u64 = 40; // live parent, X's home afterwards
        const X: u64 = 30;
        const FILL: u64 = 99;
        const LATER: u64 = 77;

        let mut state = make_state();
        seed_section(&mut state, P, 0, "P", 0);
        seed_section(&mut state, Q, 0, "Q", 1);
        seed_section(&mut state, X, P, "X", 0);
        state.children.insert(0, vec![P, Q]);
        state.children.insert(P, vec![X]);
        state.children.insert(Q, Vec::new());

        let (_mutations, stats) = apply_counted(
            &mut state,
            batch(vec![
                // Registers X -> P (and FILL -> P).
                change_cmd(P, FILL, "fill", 0),
                // Cascades P; both pairs are left behind by design.
                delete_cmd(P),
                // Lists the now-rowless X under Q: the entry becomes Many([P, Q]).
                change_cmd(Q, X, "X", 0),
                // Re-derives Q's cache over a list that still holds X, so
                // `register_listing(X, Q)` runs against an entry already naming
                // Q — the dedup the guard has to reject.
                reorder_cmd(Q, vec![X]),
                // The order-side write whose walk length reveals the entry.
                change_cmd(Q, X, "X", 7),
                // Reads Q's surviving cache, so the run is not vacuous.
                change_cmd(Q, LATER, "later", 0),
            ]),
        );

        assert_eq!(
            stats.cache_dependency_steps, 2,
            "X's index entry names P and Q — each exactly once, however many times a \
             cache listing X is re-derived"
        );
        assert_eq!(
            stats.cache_index_peak_pairs, 4,
            "the peak is the same root-build transient inside delete_cmd(P) that \
             cache_dependency_steps_counts_a_pair_a_cascade_left_behind pins — X->P, \
             FILL->P, P->0 and Q->0 held at once — and the point here is that the \
             reorder's re-registration of X under Q, two commands later, does not \
             push past it. Without the Many-arm guard it would: measured at 5 on that \
             perturbation, the duplicate storing a fifth pair"
        );
        assert_eq!(
            state.children[&Q],
            vec![LATER, X],
            "non-vacuity — the order write landed and the later order-0 child sorts \
             ahead of X at order 7"
        );
    }

    // ── b.sao cache-maintenance cost (acceptance criteria) ───────────────────
    //
    // The family below bounds the work the per-delta `ParentCache`s cost, which
    // none of the six pre-b.sao counters could see: b.ot3's epoch design left
    // `children_scan_steps`, `children_element_moves`, `children_writes_emitted`
    // and `name_index_builds` at 0 / 0 / 1 / 1 on a workload that re-derived a
    // growing list on every single command. `parent_cache_builds`,
    // `parent_cache_entries_built` and `cache_dependency_steps` are what make
    // that work observable, and these tests are what hold it to a bound.
    //
    // Every cost assertion here is an **exact equality**, not a loose upper
    // bound. A `<=` would pass a design that merely improved the constant, and
    // the defect being fenced out is asymptotic: the whole point is that
    // `parent_cache_entries_built` must not depend on how many siblings were
    // already inserted when a command runs.

    /// **The b.sao headline.** N fresh siblings created under one parent in one
    /// delta, at ascending non-zero `section_order`s, must derive that parent's
    /// cache **once** and re-read **no** already-listed child — at every N.
    ///
    /// This is an ordinary, entirely valid `DiffChangeSection` workload:
    /// no descending order, no unsorted list, no duplicate name, no deletion,
    /// no ghost, no corrupt stored state. The list only ever grows at its end.
    ///
    /// Discriminated by: the b.ot3 epoch implementation, where each non-zero
    /// order bumped a per-delta epoch that stamped *every* populated cache
    /// stale, so the next `insert_child_into_parent` re-derived the parent's
    /// whole growing list. Measured on that engine (debug, rustc 1.94.1
    /// aarch64): `parent_cache_builds == 8000` and
    /// `parent_cache_entries_built == 31,996,000` at N = 8000 — against the
    /// 1 / 0 asserted here — while every assertion in the second block below
    /// stayed byte-identical, which is precisely why the defect survived b.ot3's
    /// review. Re-checked during this fix by forcing `entry_cached` to rebuild
    /// unconditionally, which reproduces the same shape.
    ///
    /// The second block is carried alongside deliberately: it is b.ot3's own
    /// acceptance criterion, and a cache design that bought its bound by
    /// emitting more mutations or shifting more elements would have to break it.
    #[test]
    fn apply_delta_ascending_ordered_siblings_build_one_cache_and_re_read_no_entries() {
        for n in [1000_u64, 2000, 4000, 8000] {
            let mut state = make_state();
            // Ascending, all non-zero: every command is an order-side write,
            // and every one of them appends.
            let delta = batch(
                (1..=n)
                    .map(|i| change_cmd(0, i, &format!("s{i}"), i as u32))
                    .collect(),
            );
            let (mutations, stats) = apply_counted(&mut state, delta);

            assert_eq!(
                stats.parent_cache_builds, 1,
                "n={n}: the root's cache is derived once for the whole delta, not once \
                 per command ({n} builds is the pre-fix figure)"
            );
            assert_eq!(
                stats.parent_cache_entries_built,
                0,
                "n={n}: the one build reads the root's list while it is still EMPTY, so \
                 no already-inserted sibling is ever re-read (pre-fix: n(n-1)/2 = {})",
                n * (n - 1) / 2
            );
            assert_eq!(
                stats.cache_dependency_steps, 0,
                "n={n}: each written child is brand new, so no cache lists it and the \
                 invalidation walk inspects nothing"
            );
            // Memory bound, derived from the input: the root's stored list is
            // empty (0 pairs at its build) and the delta inserts n children,
            // one pair each; nothing here drops a pair, so the peak is the sum
            // and the bound is exact.
            assert_eq!(
                stats.cache_index_peak_pairs, n,
                "n={n}: the dependency index peaks at one pair per inserted child — \
                 0 initial listings + {n} insertions"
            );

            // b.ot3's own bound, unchanged by this fix.
            assert_eq!(
                stats.children_writes_emitted, 1,
                "n={n}: still one coalesced children write"
            );
            assert_eq!(
                stats.children_element_moves, 0,
                "n={n}: ascending orders append; no element is shifted"
            );
            assert_eq!(
                stats.children_scan_steps, 0,
                "n={n}: the list stays sorted, so every insertion point is a binary search"
            );
            assert_eq!(
                retained_child_ids(&mutations) as u64,
                n,
                "n={n}: N retained child ids, not N(N+1)/2"
            );

            // Non-vacuity: the delta really built the list it claims to have.
            assert_eq!(
                state.children[&0],
                (1..=n).collect::<Vec<u64>>(),
                "n={n}: the siblings land in ascending order"
            );
        }
    }

    /// The same bound when the commands **alternate between parents**.
    ///
    /// The acceptance criteria ask for a bound in "initial children plus
    /// inserted children", and this is the shape that separates it from a fix
    /// that merely rescues *consecutive* insertions under one parent: every
    /// command here revisits a parent some other command touched in between.
    ///
    /// The parents start **non-empty**, which is what makes
    /// `parent_cache_entries_built` non-zero and therefore meaningful: the bound
    /// asserted is exactly Σ (initial list lengths), i.e. each parent's existing
    /// children are read once, by the one build that parent ever pays for, and
    /// never again — not once per revisit.
    ///
    /// Discriminated by: any design that drops a parent's cache on an unrelated
    /// parent's order-side write (b.ot3's global epoch did exactly that), which
    /// re-derives each revisited list and gives
    /// `parent_cache_entries_built ≈ PARENTS × NEW_PER_PARENT × M`.
    #[test]
    fn apply_delta_ascending_siblings_interleaved_across_parents_build_one_cache_each() {
        const PARENTS: u64 = 3;
        const M: u64 = 200; // initial children per parent
        const NEW_PER_PARENT: u64 = 400;

        // Disjoint id spaces so no parent's children can collide with another's.
        let initial_id = |p: u64, i: u64| p * 10_000 + i;
        let new_id = |p: u64, j: u64| 1_000_000 + p * 10_000 + j;

        let mut state = make_state();
        state.children.insert(0, (1..=PARENTS).collect());
        for p in 1..=PARENTS {
            seed_section(&mut state, p, 0, &format!("P{p}"), 0);
            let mut list = Vec::new();
            for i in 1..=M {
                let cid = initial_id(p, i);
                seed_section(&mut state, cid, p, &format!("p{p}c{i}"), i as u32);
                list.push(cid);
            }
            state.children.insert(p, list);
        }

        // Round-robin: parent 1, parent 2, parent 3, parent 1, … each new
        // sibling ordered above every child its parent already holds.
        let mut cmds = Vec::new();
        for j in 0..NEW_PER_PARENT {
            for p in 1..=PARENTS {
                cmds.push(change_cmd(
                    p,
                    new_id(p, j),
                    &format!("p{p}n{j}"),
                    (M + 1 + j) as u32,
                ));
            }
        }
        let (_mutations, stats) = apply_counted(&mut state, batch(cmds));

        assert_eq!(
            stats.parent_cache_builds, PARENTS,
            "one cache derivation per touched parent for the whole delta, however \
             many times the commands alternate between them"
        );
        assert_eq!(
            stats.parent_cache_entries_built,
            PARENTS * M,
            "each parent's pre-existing children are read by its single build and \
             never re-read: the bound is Σ initial list lengths, independent of how \
             many siblings the delta goes on to insert"
        );
        assert_eq!(
            stats.cache_dependency_steps, 0,
            "every written child is brand new, so no cache lists it"
        );
        // Memory bound, derived from the input and exact: each touched parent's
        // build stores one pair per child its stored list already held, and
        // each insertion stores one more. No order-side write lands on an
        // already-listed child, so no pair is ever dropped and the peak is the
        // running total. The root is never inserted into, so its cache is never
        // built and it contributes nothing.
        assert_eq!(
            stats.cache_index_peak_pairs,
            PARENTS * M + PARENTS * NEW_PER_PARENT,
            "the index peaks at Σ initial listings + insertions over the touched \
             parents, and at no point holds a pair for a parent the delta never \
             reached"
        );
        assert_eq!(
            stats.children_writes_emitted, PARENTS,
            "one coalesced children write per dirty parent"
        );
        assert_eq!(
            stats.children_element_moves, 0,
            "every new sibling outranks its parent's whole list and appends"
        );

        for p in 1..=PARENTS {
            assert_eq!(
                state.children[&p].len() as u64,
                M + NEW_PER_PARENT,
                "parent {p} holds its initial children plus its new ones"
            );
            assert_eq!(
                state.children[&p][M as usize..],
                (0..NEW_PER_PARENT)
                    .map(|j| new_id(p, j))
                    .collect::<Vec<u64>>(),
                "parent {p}: the new siblings appended in command order"
            );
        }
    }

    /// The controls that keep the ascending case honest **as a comparison**.
    ///
    /// Neither of these workloads invalidates a cache. A `DiffChangeSection` at
    /// order `0` writes the value a rowless id already reads as, so it writes
    /// no order at all. A `CreateSection` does write one — it appends at the
    /// next key, `0, 1, 2, …` — but always to a brand-new child that no cache
    /// lists yet, so the invalidation walk finds nobody. Both pinned 1 / 0 / 0
    /// on the epoch engine too, when `CreateSection` still stored order `0`.
    ///
    /// That is exactly what makes them load-bearing: the ticket's measurement
    /// used them as the baseline, and without them "the ascending case now costs
    /// one build" would be compatible with an engine that had stopped building
    /// caches altogether. Here the ordered and the order-free workloads meet at
    /// the same numbers, from opposite directions.
    #[test]
    fn apply_delta_order_free_sibling_workloads_build_one_empty_cache() {
        const N: u64 = 2000;

        let cases: Vec<(&str, Vec<StructCommand>)> = vec![
            (
                "CreateSection (appends at the next key, on a child no cache lists)",
                (1..=N)
                    .map(|i| create_cmd(0, i, &format!("s{i}")))
                    .collect(),
            ),
            (
                "DiffChangeSection at order 0 (the ghost encoding, so no order changes)",
                (1..=N)
                    .map(|i| change_cmd(0, i, &format!("s{i}"), 0))
                    .collect(),
            ),
        ];

        for (label, cmds) in cases {
            let mut state = make_state();
            let (_mutations, stats) = apply_counted(&mut state, batch(cmds));

            assert_eq!(
                stats.parent_cache_builds, 1,
                "{label}: one derivation of the root's cache"
            );
            assert_eq!(
                stats.parent_cache_entries_built, 0,
                "{label}: that derivation read an empty list"
            );
            assert_eq!(
                stats.cache_dependency_steps, 0,
                "{label}: no command changes the effective order of a child any cache \
                 lists, so the invalidation walk inspects nothing"
            );
            assert_eq!(
                stats.cache_index_peak_pairs, N,
                "{label}: one pair per insertion — the same memory bound the ordered \
                 workload pays, which is what makes the comparison a control on the \
                 index as well as on the derivations"
            );
            assert_eq!(
                state.children[&0].len() as u64,
                N,
                "{label}: non-vacuity — every sibling landed"
            );
        }
    }

    /// Residual **R4**, re-stated in the new counters: `ReorderChildren`
    /// re-derives the parent's cache wholesale, so `parent_cache_entries_built`
    /// grows by the list length **per reorder**.
    ///
    /// This is pinned rather than claimed away. The re-derivation is not
    /// incidental: the new list is an arbitrary permutation the command
    /// supplies, so there is no incremental edit that could produce the new
    /// `orders` vector and `sorted` flag from the old one. A caller issuing R
    /// reorders of a C-element list pays R × C — linear in the commands, which
    /// is the bound b.sao owes, and *not* linear in a single command's input.
    ///
    /// Every reorder here names the whole list (a rotation), so each one moves
    /// all C children and re-keys all C of them: the keys `1..=C` stay with
    /// their positions. Each re-key is an order-side write, and from the second
    /// reorder on the parent's cache (built by the reorder before) lists the
    /// re-keyed child, so each costs one dependency step: C × (R − 1).
    ///
    /// Discriminated by: a future change that re-derives more than one list per
    /// reorder (the count would exceed R × C), or one that skips the
    /// re-derivation and hands the next insertion a cache describing the
    /// pre-reorder positions. Removing the re-key's
    /// `invalidate_order_listeners` call drops the dependency steps to 0.
    #[test]
    fn apply_delta_repeated_reorders_re_derive_exactly_one_list_each() {
        const C: u64 = 8;

        for r in [1_u64, 4, 16] {
            let mut state = make_state();
            state.children.insert(0, (1..=C).collect());
            for i in 1..=C {
                seed_section(&mut state, i, 0, &format!("s{i}"), i as u32);
            }

            // Rotate the list by one on each reorder: always a permutation of
            // the same C ids, so no id enters or leaves the parent.
            let cmds: Vec<StructCommand> = (0..r)
                .map(|k| {
                    let split = (k as usize % C as usize) + 1;
                    let mut list: Vec<u64> = (1..=C).collect();
                    list.rotate_left(split);
                    reorder_cmd(0, list)
                })
                .collect();
            let (_mutations, stats) = apply_counted(&mut state, batch(cmds));

            assert_eq!(
                stats.parent_cache_builds, r,
                "r={r}: one re-derivation per reorder — the reordered list is not \
                 derivable from the cache it replaces"
            );
            assert_eq!(
                stats.parent_cache_entries_built,
                r * C,
                "r={r}: each re-derivation reads the whole reordered list and no more"
            );
            assert_eq!(
                stats.cache_dependency_steps,
                C * (r - 1),
                "r={r}: every rotation re-keys all C children; the first reorder finds \
                 no cache listing them, and each later one finds exactly the parent's"
            );
            // Memory bound, derived from the input and exact. It is also one of
            // the places the `One`-arm dedup guard in `CacheListers::insert` is
            // observable — not the only one; removing that guard turns seven
            // tests in this module red, this among them. Every reorder
            // re-registers all C children, and every re-registration after the
            // first is a duplicate the guard must reject.
            //
            // With the `One`-arm early return removed, each build after the
            // first converts every child's `One(0)` into `Many([0, 0])` — two
            // stored pairs per child — and the next reorder's re-key
            // invalidation drops it back to `One(0)` before its build doubles
            // it again; the `Many` arm's own `contains` guard, left intact,
            // stops it there. So the peak is 2C (16 at C = 8), not r * C:
            // bounded at one extra pair per child, but enough to fail this
            // assertion. Re-measured since reorders began re-keying, the test
            // now goes red one assertion earlier: every re-key walks both
            // copies, so `cache_dependency_steps` reads 40 at r = 4, not 24.
            assert_eq!(
                stats.cache_index_peak_pairs, C,
                "r={r}: one pair per listed child no matter how many times the \
                 re-derivations re-register the same (child, parent)"
            );
            assert_eq!(
                stats.children_writes_emitted, 1,
                "r={r}: R reorders of one parent still coalesce to one write"
            );
        }
    }

    /// Residual **R3**, re-stated in the new counters: repeated same-parent
    /// order changes cost **one dependency step each** and a number of cache
    /// derivations that does not grow with the number of commands.
    ///
    /// This is the shape `invalidate_order_listeners`' `keep` argument exists
    /// for. Each command rewrites one child's `section_order` under the parent
    /// that already lists it; keeping that parent's cache across the
    /// invalidation is what lets `rebuild_children_list` take its
    /// remove-and-reinsert arm instead of re-deriving the list.
    ///
    /// The two derivations are both paid by the **first** command and neither
    /// depends on `r`: the lazy build at its `detach_cache`, plus the one the
    /// `sort_by_key` arm makes because that first write raised an order past its
    /// successors and the freshly-derived cache therefore reads as unsorted.
    /// From the second command on, the kept cache is sorted and carries every
    /// remaining command without a single re-read. The first command's step is
    /// likewise absent because nothing is cached yet when its invalidation runs.
    ///
    /// Discriminated by: forcing `keep` to `None` in
    /// `apply_diff_change_section`'s existing-row branch, which drops the
    /// parent's own cache on every command and makes both counters grow with
    /// `r` — `parent_cache_builds == 2r`, since each rebuilt cache reads as
    /// unsorted and routes to the sort arm's second derivation.
    #[test]
    fn apply_delta_repeated_same_parent_order_changes_cost_one_dependency_step_each() {
        const P: u64 = 1;
        const C: u64 = 4; // children under P
        const MOVED: u64 = 11; // the child whose order is rewritten

        for r in [2_u64, 8, 64] {
            let mut state = make_state();
            seed_section(&mut state, P, 0, "P", 0);
            state.children.insert(0, vec![P]);
            let list: Vec<u64> = (0..C).map(|i| 11 + i).collect();
            for (i, &cid) in list.iter().enumerate() {
                seed_section(&mut state, cid, P, &format!("c{cid}"), i as u32 + 1);
            }
            state.children.insert(P, list);

            let cmds: Vec<StructCommand> = (0..r)
                .map(|k| change_cmd(P, MOVED, "c11", 100 + k as u32))
                .collect();
            let (_mutations, stats) = apply_counted(&mut state, batch(cmds));

            assert_eq!(
                stats.cache_dependency_steps,
                r - 1,
                "r={r}: one step per order-side write, except the first — nothing is \
                 cached yet when its invalidation runs, so the index names no parent"
            );
            assert_eq!(
                stats.parent_cache_builds, 2,
                "r={r}: constant in r. The first command pays the lazy build and the \
                 sort arm's re-derivation; the kept cache carries every command after it"
            );
            assert_eq!(
                stats.parent_cache_entries_built,
                2 * C,
                "r={r}: both of those derivations read the same {C}-element list, and \
                 nothing re-reads it afterwards"
            );
            // Memory bound, derived from the input and exact, and constant in
            // r: the parent's stored list holds C children and the delta
            // inserts none, so the index cannot exceed C whatever the order
            // changes do. Each write takes the child's entry out and puts it
            // back under `keep`, which is a dip and a recovery, never a growth.
            assert_eq!(
                stats.cache_index_peak_pairs, C,
                "r={r}: {C} stored listings + 0 insertions, independent of how many \
                 order changes walk the index"
            );
            assert_eq!(
                stats.children_writes_emitted, 1,
                "r={r}: r order changes under one parent are one coalesced write"
            );
            assert_eq!(
                state.children[&P],
                vec![12, 13, 14, MOVED],
                "r={r}: non-vacuity — the repeatedly-raised child really ends last"
            );
        }
    }

    /// One of the two workloads the b.sao sweep added to this Issue: repeated
    /// cascade-delete of siblings whose rows carry **non-zero** orders.
    ///
    /// `apply_delta_repeated_sibling_removal_costs_one_scan_step_per_removal`
    /// does not pin this. It was written when `CreateSection` stored order `0`,
    /// so its deletions never fired `cascade_delete_section`'s order-side hook;
    /// its rows now carry append keys `0..N`, but it asserts none of the cache
    /// counters. Here the rows are seeded at non-zero orders, so every deletion
    /// is an order-side write (a live order → the ghost `0`), which is the
    /// second command family the epoch design was quadratic on.
    ///
    /// The bound: one derivation of the parent's cache for the whole delta,
    /// reading the initial list once. `cache_dependency_steps == 0` is the
    /// subtler half — `remove_child_from_parent` de-registers the child from the
    /// parent *before* the row is dropped, so by the time the order-side
    /// invalidation runs the index names nobody. That ordering is what keeps the
    /// walk from re-inspecting a parent the same command already handled.
    ///
    /// Discriminated by: the epoch implementation, measured at 2000 ordered
    /// cascade-deletes → `parent_cache_builds == 2000`,
    /// `parent_cache_entries_built == 2,001,000` (debug, rustc 1.94.1 aarch64),
    /// against the 1 / N asserted here.
    ///
    /// Residuals R2 (the `Vec` shift per removal) and the one scan step per
    /// removal are pinned alongside, unchanged.
    #[test]
    fn apply_delta_repeated_cascade_delete_of_ordered_siblings_builds_one_cache() {
        for n in [250_u64, 500, 1000] {
            let mut state = make_state();
            state.children.insert(0, (1..=n).collect());
            for i in 1..=n {
                // Non-zero, ascending: a live order the parent's cache mirrors,
                // and which the delete turns into the ghost 0.
                seed_section(&mut state, i, 0, &format!("s{i}"), i as u32);
            }

            let (_mutations, stats) =
                apply_counted(&mut state, batch((1..=n).map(delete_cmd).collect()));

            assert_eq!(
                stats.parent_cache_builds, 1,
                "n={n}: the root's cache is derived once for the whole delta \
                 (pre-fix: once per deletion)"
            );
            assert_eq!(
                stats.parent_cache_entries_built,
                n,
                "n={n}: that one derivation reads the initial list once; pre-fix each \
                 deletion re-read the shrinking remainder, n(n+1)/2 = {} entries",
                n * (n + 1) / 2
            );
            assert_eq!(
                stats.cache_dependency_steps, 0,
                "n={n}: the child is de-registered by its removal from the list, which \
                 runs before the row is dropped, so the order-side walk finds no lister"
            );
            // Memory bound, derived from the input and exact: the root's single
            // build stores one pair per stored child, and the delta inserts
            // nothing, so the peak is the initial list length. Every deletion
            // then drops a pair, so the index only shrinks from there — which is
            // exactly why a final-size reading would report 0 and a peak
            // reading reports the truth.
            assert_eq!(
                stats.cache_index_peak_pairs, n,
                "n={n}: the index peaks at the parent's initial list length ({n} stored \
                 listings + 0 insertions) and falls to empty as the deletions land"
            );
            assert_eq!(
                stats.children_scan_steps, n,
                "n={n}: residual — front-first removal finds the child at index 0 every \
                 time"
            );
            assert_eq!(
                stats.children_element_moves,
                n * (n - 1) / 2,
                "n={n}: residual R2 — each removal shifts the remainder"
            );
            assert!(
                state.children[&0].is_empty(),
                "n={n}: non-vacuity — every sibling really was deleted"
            );
        }
    }

    /// The sweep's other workload: repeated cross-parent moves that also change
    /// the child's order, out of one parent and into another.
    ///
    /// Each command is `apply_diff_change_section`'s existing-row branch with
    /// `old_parent != parent_id` *and* a changed `section_order` — so it runs
    /// the order-side invalidation, then removes the child from the source and
    /// inserts it into the destination. b.ot3 kept a cache across that
    /// invalidation only on the same-parent arm; generalizing the kept parent to
    /// "the row's previous parent, on both arms" is what makes this linear,
    /// because the source parent's cache is exactly the one
    /// `remove_child_from_parent` is about to repair.
    ///
    /// The bound: **one derivation per parent** — the source's over its initial
    /// list, the destination's over its empty one — for the whole delta.
    /// `cache_dependency_steps` is one per command bar the first (nothing is
    /// cached when its invalidation runs), which is the steady-state cost the
    /// counter's doc-comment states.
    ///
    /// Discriminated by: forcing `keep` to `None` on the cross-parent arm, which
    /// drops the source's cache on every command and makes
    /// `remove_child_from_parent` re-derive the shrinking source list —
    /// `parent_cache_builds` grows to n + 1 and `parent_cache_entries_built` to
    /// roughly n²/2.
    #[test]
    fn apply_delta_repeated_cross_parent_order_changes_build_one_cache_per_parent() {
        const SRC: u64 = 1;
        const DST: u64 = 2;

        for n in [250_u64, 500, 1000] {
            let mut state = make_state();
            seed_section(&mut state, SRC, 0, "SRC", 0);
            seed_section(&mut state, DST, 0, "DST", 1);
            state.children.insert(0, vec![SRC, DST]);
            state.children.insert(DST, Vec::new());

            let ids: Vec<u64> = (1..=n).map(|i| 1000 + i).collect();
            for (i, &cid) in ids.iter().enumerate() {
                seed_section(&mut state, cid, SRC, &format!("c{cid}"), i as u32 + 1);
            }
            state.children.insert(SRC, ids.clone());

            // Each command moves one child SRC → DST and raises its order above
            // everything already in DST, so the destination insert appends.
            let cmds: Vec<StructCommand> = ids
                .iter()
                .enumerate()
                .map(|(i, &cid)| change_cmd(DST, cid, &format!("c{cid}"), 10_000 + i as u32))
                .collect();
            let (_mutations, stats) = apply_counted(&mut state, batch(cmds));

            assert_eq!(
                stats.parent_cache_builds, 2,
                "n={n}: one derivation for the source and one for the destination, for \
                 the whole delta — not one per command"
            );
            assert_eq!(
                stats.parent_cache_entries_built, n,
                "n={n}: the source's single derivation reads its initial list once and \
                 the destination's reads an empty one; pre-fix the source was \
                 re-derived per command"
            );
            assert_eq!(
                stats.cache_dependency_steps,
                n - 1,
                "n={n}: one step per order-side write, except the first — nothing is \
                 cached yet when its invalidation runs"
            );
            // Memory bound. Two statements, and the difference between them is
            // the point:
            //
            // * The bound the INPUT alone gives is an inequality — Σ initial
            //   listings over the touched parents (SRC holds n, DST holds 0)
            //   plus one per insertion (n into DST) — because nothing in the
            //   command list says the removals and the insertions interleave.
            // * They do interleave, one for one: each command de-registers the
            //   child from SRC before listing it under DST, so the live count
            //   returns to where it started after every command and the peak is
            //   the source's initial length. That is still derived from the
            //   shape of the input rather than read off a run, and it is the
            //   tighter claim, so both are asserted.
            assert!(
                stats.cache_index_peak_pairs <= 2 * n,
                "n={n}: the index can never hold more than Σ initial listings + \
                 insertions, got {}",
                stats.cache_index_peak_pairs
            );
            assert_eq!(
                stats.cache_index_peak_pairs, n,
                "n={n}: every insertion into DST is preceded by the same child's \
                 de-registration from SRC, so the index never grows past the source's \
                 initial list length — a move does not double-count its child"
            );
            assert_eq!(
                stats.children_writes_emitted, 2,
                "n={n}: one coalesced write per dirty parent"
            );
            assert_eq!(
                state.children[&DST], ids,
                "n={n}: non-vacuity — every child really moved, in command order"
            );
            assert!(state.children[&SRC].is_empty(), "n={n}: the source emptied");
        }
    }

    /// `MutationSink::Discard` is the same engine with nothing built: the
    /// resulting `ObjectState` and the resulting `ApplyDeltaStats` must be
    /// **identical** to a collecting run of the same delta.
    ///
    /// Both halves matter. The state equality is what makes `Discard` legal at
    /// the four call sites that materialize their object from `ObjectState`
    /// (CreateLayer's two branches, the DR decoder, lineage replay); the stats
    /// equality is what makes the counters describe the *plan* rather than
    /// what a particular caller chose to keep — which is what lets the
    /// count-based tests above stand for the CreateLayer path too.
    ///
    /// The comparison is the whole `ObjectState`, not [`structural_shape`]:
    /// both runs apply the *same* delta to the *same* start state, so — unlike
    /// the one-command-per-delta oracle — nothing here may legitimately differ,
    /// version stamps and `source_format` included.
    ///
    /// The fourth tuple element is a per-case non-vacuity guard, run against
    /// the collecting result: an equality between two runs is satisfied by two
    /// equally-broken runs, so each case states what its delta must actually
    /// have done.
    #[test]
    fn apply_delta_into_discard_leaves_identical_state_and_identical_stats() {
        type NonVacuityGuard = fn(&ObjectState, &ApplyDeltaStats);
        let deltas: Vec<(&str, ObjectState, OwnedDelta, NonVacuityGuard)> = vec![
            (
                "many siblings",
                make_state(),
                batch(
                    (1..=500_u64)
                        .map(|i| create_cmd(0, i, &format!("s{i}")))
                        .collect(),
                ),
                |state, stats| {
                    assert_eq!(state.sections.len(), 500, "every sibling landed");
                    assert_eq!(stats.children_writes_emitted, 1, "coalesced to one write");
                },
            ),
            (
                // b.sao: the ascending-order workload, so the four cache
                // counters are covered by the parity guarantee rather than only
                // by the collecting-mode cost tests. Without a case that
                // actually derives a cache and walks the dependency index, all
                // four would be zero on both sides and the parity claim over
                // them would be vacuous.
                "ascending-ordered siblings then repeated order changes",
                make_state(),
                batch(
                    (1..=200_u64)
                        .map(|i| change_cmd(0, i, &format!("s{i}"), i as u32))
                        .chain(
                            // Order changes on already-listed children: these
                            // are what make `cache_dependency_steps` non-zero.
                            (1..=20_u64)
                                .map(|i| change_cmd(0, i, &format!("s{i}"), 500 + i as u32)),
                        )
                        .collect(),
                ),
                |state, stats| {
                    assert_eq!(state.children[&0].len(), 200, "every sibling landed");
                    assert_eq!(
                        stats.parent_cache_builds, 1,
                        "the b.sao bound holds on the collecting run too"
                    );
                    assert!(
                        stats.cache_dependency_steps > 0,
                        "non-vacuity: the order changes really walked the dependency \
                         index, so the parity claim over that counter means something"
                    );
                    assert!(
                        stats.cache_index_peak_pairs > 0,
                        "non-vacuity: the index really stored pairs, so the parity \
                         claim over the peak counter means something too. The struct's \
                         `PartialEq` makes the Collect/Discard comparison exhaustive \
                         over every field, but an exhaustive comparison of two zeros \
                         proves nothing about the field that is zero"
                    );
                },
            ),
            (
                "reset plus mixed structure plus values",
                {
                    let mut state = make_state();
                    seed_section(&mut state, 1, 0, "P", 0);
                    seed_section(&mut state, 2, 1, "C", 0);
                    state.children.insert(0, vec![1]);
                    state.children.insert(1, vec![2]);
                    state
                },
                OwnedDelta {
                    is_diff: true,
                    base_version: 0,
                    source_format: "usda".to_string(),
                    struct_commands: vec![
                        create_cmd(0, 1, "A"),
                        create_cmd(1, 2, "B"),
                        change_cmd(1, 3, "C", 7),
                        reorder_cmd(1, vec![3, 2]),
                        move_cmd(2, 1, 0, "Moved"),
                        delete_cmd(3),
                    ],
                    set_fields: vec![OodfSetField {
                        section_id: 1,
                        key_name: "k".to_string(),
                        value_or_ext_hash: vec![7; 64],
                        ext_value_size: 0,
                        ext_value_index: 0,
                        set_order: 3,
                    }],
                    time_samples: vec![OodfTimeSample {
                        section_id: 1,
                        time: 1.5,
                        value_or_ext_hash: vec![9; 32],
                        ext_value_size: 0,
                        ext_value_index: 0,
                    }],
                },
                |state, _| {
                    assert_eq!(
                        state.source_format, "usda",
                        "the reset stamped its source format"
                    );
                    assert_eq!(state.fields.len(), 1, "the delta's field row landed");
                    assert_eq!(
                        state.time_samples.len(),
                        1,
                        "the delta's time-sample row landed"
                    );
                },
            ),
            (
                // Rows on BOTH sides of the cascade: `purge_cascaded_rows`'
                // `DeleteField` / `DeleteTimeSample` arms are the most fragile
                // place in the `push_with` contract — removal and emission are
                // entangled in one `retain` predicate — so the case that
                // exercises them must actually have rows to purge, and rows
                // that must survive the same pass.
                "cascade delete with doomed and surviving rows",
                {
                    let mut state = make_state();
                    seed_section(&mut state, 1, 0, "P", 0);
                    seed_section(&mut state, 2, 1, "C", 0);
                    seed_section(&mut state, 3, 0, "Keep", 1);
                    state.children.insert(0, vec![1, 3]);
                    state.children.insert(1, vec![2]);
                    // Doomed: one row on the cascade root, one on its child.
                    seed_field(&mut state, 1, "doomed_p");
                    seed_field(&mut state, 2, "doomed_c");
                    seed_time_sample(&mut state, 2, 1.5);
                    // Surviving: same maps, untouched by the cascade.
                    seed_field(&mut state, 3, "kept");
                    seed_time_sample(&mut state, 3, 2.5);
                    state
                },
                batch(vec![delete_cmd(1)]),
                |state, stats| {
                    assert_eq!(
                        stats.field_purge_passes, 1,
                        "the cascade really ran its deferred purge pass"
                    );
                    assert_eq!(
                        state.fields.keys().collect::<Vec<_>>(),
                        vec![&(3, "kept".to_string())],
                        "the doomed field rows were purged and the survivor kept"
                    );
                    assert_eq!(
                        state.time_samples.keys().collect::<Vec<_>>(),
                        vec![&(3, OrderedFloat(2.5))],
                        "the doomed time sample was purged and the survivor kept. \
                         Asserted as the exact surviving KEY, not a length: fields and \
                         time samples are separate `retain` predicates in the same pass, \
                         so a predicate that purged (3, 2.5) and kept (2, 1.5) would also \
                         leave length 1 — and the field assertion above says nothing about \
                         the time-sample half"
                    );
                },
            ),
        ];

        for (label, initial, delta, assert_non_vacuous) in deltas {
            let mut collecting = initial.clone();
            let mut collect_stats = ApplyDeltaStats::default();
            apply_delta_with_stats(&mut collecting, Some(delta.clone()), &mut collect_stats)
                .unwrap_or_else(|e| panic!("{label}: collecting run must apply: {e}"));
            assert_non_vacuous(&collecting, &collect_stats);

            let mut discarding = initial;
            let mut discard_stats = ApplyDeltaStats::default();
            let mut sink = MutationSink::Discard;
            apply_delta_into(&mut discarding, Some(delta), &mut sink, &mut discard_stats)
                .unwrap_or_else(|e| panic!("{label}: discarding run must apply: {e}"));

            assert_eq!(
                discarding, collecting,
                "{label}: Discard must leave the same ObjectState, version stamps included"
            );
            assert_eq!(
                discard_stats, collect_stats,
                "{label}: the counters describe the plan, not what the caller kept"
            );
        }
    }

    /// A cascade delete of a large subtree filters `state.fields` and
    /// `state.time_samples` **once** per delta, not once per deleted section.
    ///
    /// Pre-b.ot3 `cascade_delete_section` ran a `retain` over both maps for
    /// every deleted section, approaching O(deleted sections × total rows);
    /// `field_purge_passes` is the counter that makes the single deferred pass
    /// observable. The unrelated rows must survive it untouched — a single
    /// pass over a shared map is exactly where an over-broad predicate would
    /// take them out.
    #[test]
    fn apply_delta_large_subtree_delete_runs_exactly_one_field_purge_pass() {
        const SUBTREE: u64 = 200;
        const UNRELATED: u64 = 200;

        let mut state = make_state();
        // Doomed subtree: section 1 with SUBTREE children, each carrying a
        // field and a time sample. Survivor: section 2, same shape.
        let mut cmds = vec![create_cmd(0, 1, "Doomed"), create_cmd(0, 2, "Keep")];
        for i in 0..SUBTREE {
            cmds.push(create_cmd(1, 100 + i, &format!("d{i}")));
        }
        for i in 0..UNRELATED {
            cmds.push(create_cmd(2, 10_000 + i, &format!("k{i}")));
        }
        let mut set_fields = Vec::new();
        let mut time_samples = Vec::new();
        for i in 0..SUBTREE {
            set_fields.push(OodfSetField {
                section_id: 100 + i,
                key_name: "f".to_string(),
                value_or_ext_hash: vec![1],
                ext_value_size: 0,
                ext_value_index: 0,
                set_order: 0,
            });
            time_samples.push(OodfTimeSample {
                section_id: 100 + i,
                time: 1.0,
                value_or_ext_hash: vec![1],
                ext_value_size: 0,
                ext_value_index: 0,
            });
        }
        for i in 0..UNRELATED {
            set_fields.push(OodfSetField {
                section_id: 10_000 + i,
                key_name: "f".to_string(),
                value_or_ext_hash: vec![2],
                ext_value_size: 0,
                ext_value_index: 0,
                set_order: 0,
            });
            time_samples.push(OodfTimeSample {
                section_id: 10_000 + i,
                time: 1.0,
                value_or_ext_hash: vec![2],
                ext_value_size: 0,
                ext_value_index: 0,
            });
        }
        apply_delta(
            &mut state,
            Some(OwnedDelta {
                is_diff: false,
                struct_commands: cmds,
                set_fields,
                time_samples,
                ..Default::default()
            }),
        )
        .expect("arrangement delta must apply");
        assert_eq!(state.fields.len() as u64, SUBTREE + UNRELATED);
        assert_eq!(state.time_samples.len() as u64, SUBTREE + UNRELATED);

        let (mutations, stats) = apply_counted(&mut state, batch(vec![delete_cmd(1)]));

        assert_eq!(
            stats.field_purge_passes, 1,
            "one deferred pass over each map for the whole delta, not one per deleted section"
        );
        let deleted_fields = mutations
            .iter()
            .filter(|m| matches!(m, StorageMutation::DeleteField { .. }))
            .count();
        assert_eq!(
            deleted_fields as u64, SUBTREE,
            "exactly the doomed subtree's fields are erased"
        );
        assert_eq!(
            state.fields.len() as u64,
            UNRELATED,
            "every unrelated field survives the single pass"
        );
        assert_eq!(
            state.time_samples.len() as u64,
            UNRELATED,
            "every unrelated time sample survives the single pass"
        );
        assert!(
            state.fields.keys().all(|(sid, _)| *sid >= 10_000),
            "no doomed row survived"
        );
    }

    /// A delta with no cascade runs **no** purge pass at all.
    ///
    /// The counterpart to the test above: `field_purge_passes` must
    /// discriminate, or "exactly one pass" is not evidence of anything.
    #[test]
    fn apply_delta_without_a_cascade_runs_no_field_purge_pass() {
        let mut state = make_state();
        let (_, stats) = apply_counted(
            &mut state,
            batch(vec![create_cmd(0, 1, "A"), create_cmd(0, 2, "B")]),
        );
        assert_eq!(stats.field_purge_passes, 0);
    }

    // ── The compact purge costs the edit, not the layer (b.k39) ──────────────

    /// The dev shape's field names, as this test builds its layers: a prim's
    /// `specifier` and `typeName`, an asset root's `references` (added
    /// below), an attribute's `custom`, `default`, `typeName` and
    /// `variability`.
    const DEV_PRIM_FIELDS: [&str; 2] = ["specifier", "typeName"];
    const DEV_ATTR_FIELDS: [&str; 4] = ["custom", "default", "typeName", "variability"];
    /// A dev-shape asset's attributes per prim and child prims.
    const DEV_ATTRS_PER_PRIM: u64 = 6;
    const DEV_CHILD_PRIMS: u64 = 4;
    /// Sections per dev-shape asset: its root prim, the root's attributes,
    /// its child prims and theirs.
    const DEV_ASSET_SECTIONS: u64 = (1 + DEV_CHILD_PRIMS) * (1 + DEV_ATTRS_PER_PRIM);
    /// Assets per group under the root, as the harness places them.
    const DEV_ASSETS_PER_GROUP: u64 = 100;

    /// A compact layer of `assets` dev-shape assets (`DEV_ASSET_SECTIONS`
    /// sections each: a root prim with `references`, six attributes and four
    /// child prims of six attributes each), `DEV_ASSETS_PER_GROUP` to a
    /// group prim under the root, and the root id of the first asset. Built
    /// straight into the tables: a layer this size through the engine would
    /// cost the debug build seconds per test.
    fn dev_shape_layer(assets: u64) -> (crate::domain::types::CompactState, u64) {
        let mut state = crate::domain::types::CompactState {
            object_id: 1,
            sequence_num: 5,
            first_version: 1,
            ..Default::default()
        };
        let mut next_id = 1u64;
        let mut id = || {
            let id = next_id;
            next_id += 1;
            id
        };
        let add = |state: &mut crate::domain::types::CompactState,
                   section_id: u64,
                   parent_id: u64,
                   name: String,
                   section_type: u8,
                   fields: &[&str]| {
            state.sections.insert(
                section_id,
                Section {
                    section_id,
                    parent_id,
                    name,
                    section_type,
                    section_order: 0,
                    first_version: 1,
                    last_version: 1,
                },
            );
            state
                .children
                .entry(parent_id)
                .or_default()
                .push(section_id);
            for key_name in fields {
                state.fields.insert(section_id, key_name, None);
            }
        };
        let mut first_asset = None;
        let mut group = 0;
        for k in 0..assets {
            if k % DEV_ASSETS_PER_GROUP == 0 {
                group = id();
                add(
                    &mut state,
                    group,
                    0,
                    format!("Group_{k}"),
                    6,
                    &DEV_PRIM_FIELDS,
                );
            }
            let root = id();
            first_asset.get_or_insert(root);
            let mut root_fields = DEV_PRIM_FIELDS.to_vec();
            root_fields.push("references");
            add(
                &mut state,
                root,
                group,
                format!("Asset_{k:05}"),
                6,
                &root_fields,
            );
            let mut prims = vec![root];
            for c in 0..DEV_CHILD_PRIMS {
                let child = id();
                add(
                    &mut state,
                    child,
                    root,
                    format!("Child_{c}"),
                    6,
                    &DEV_PRIM_FIELDS,
                );
                prims.push(child);
            }
            for prim in prims {
                for a in 0..DEV_ATTRS_PER_PRIM {
                    let attr = id();
                    add(
                        &mut state,
                        attr,
                        prim,
                        format!("attr_{a}"),
                        1,
                        &DEV_ATTR_FIELDS,
                    );
                }
            }
        }
        (state, first_asset.expect("at least one asset"))
    }

    /// I1 regression (b.k39): on the compact form, deleting one dev-shape
    /// asset (`DEV_ASSET_SECTIONS` sections, cascading their value rows)
    /// visits the same number of purge steps on a layer of 100 assets and on
    /// one of 2,000 — 20× the field rows — and at most `D × N` for `D`
    /// purged sections and `N` interned names: the purge costs the edit, not
    /// the layer. Before the probe arm, the purge was a `retain` over every
    /// field row (O(F)), which the seam counts as one step per row.
    ///
    /// Hard precondition: at both sizes the count-only predicate picks the
    /// probe arm, with at least `4 × D × N × R` rows at the small one, so the
    /// arm is chosen by the content, not forced, and stays chosen at either
    /// hash-group width and under any hasher seed.
    #[test]
    fn apply_delta_compact_asset_delete_visits_the_same_purge_steps_on_a_layer_20x_larger() {
        use crate::domain::types::{purge_arm, FieldRows, PURGE_PROBE_RATIO};

        let _natural = purge_arm::force(None);
        let mut visited = Vec::new();
        for assets in [100u64, 2_000] {
            let (mut state, doomed) = dev_shape_layer(assets);
            let rows = state.fields.len();
            let names = state.fields.names();
            let n = names.len();
            assert!(
                names.free_ids_sorted_for_test().is_empty(),
                "fixture: every id slot live, so N is the slot count"
            );
            let d = DEV_ASSET_SECTIONS as usize;
            assert!(
                state.fields.purges_by_probe(d),
                "fixture, {assets} assets: the probe arm by content ({d} × {n} × \
                 {PURGE_PROBE_RATIO} against {rows} rows)"
            );
            if assets == 100 {
                assert!(
                    rows >= 4 * d * n * PURGE_PROBE_RATIO,
                    "fixture: {rows} rows at least 4 × D × N × R past the boundary"
                );
            }
            let asset_rows = {
                let mut ids = vec![doomed];
                let mut at = 0;
                while at < ids.len() {
                    ids.extend(state.children.get(&ids[at]).into_iter().flatten());
                    at += 1;
                }
                assert_eq!(ids.len(), d, "fixture: the asset's sections");
                let mut count = 0;
                state.fields.for_each(|section_id, _, _| {
                    count += usize::from(ids.contains(&section_id));
                });
                count
            };

            purge_arm::reset();
            let mut stats = ApplyDeltaStats::default();
            apply_delta_with_stats(
                &mut state,
                Some(batch(vec![delete_cmd(doomed)])),
                &mut stats,
            )
            .expect("the delete applies");
            let counts = purge_arm::counts();

            assert_eq!(stats.field_purge_passes, 1, "{assets} assets: one purge");
            assert!(
                counts.steps_visited <= (d * n) as u64,
                "{assets} assets: {} steps, above D × N = {}",
                counts.steps_visited,
                d * n
            );
            assert_eq!(
                (counts.probe_purges, counts.scan_purges),
                (1, 0),
                "{assets} assets: the purge took the probe arm"
            );
            assert_eq!(
                counts.rows_removed as usize, asset_rows,
                "{assets} assets: the asset's field rows, and no other"
            );
            assert_eq!(state.fields.len(), rows - asset_rows);
            visited.push(counts.steps_visited);
        }
        assert_eq!(
            visited[0], visited[1],
            "the purge visits the same steps on a layer 20× larger"
        );
    }

    // ── Ordering contracts ───────────────────────────────────────────────────

    /// The deferred cascade purge must stay **strictly before** the field
    /// phase.
    ///
    /// b.ot3 moved the `DeleteField` / `DeleteTimeSample` emissions of a
    /// cascade out of `cascade_delete_section` and into one pass at the close
    /// of the structural phase. That is observationally identical to the
    /// pre-b.ot3 emission *only* while the pass stays ahead of step 4's
    /// `SetField`s: `src/storage/layer.rs`'s `ext_rows_written` folds the
    /// mutation list into `(row_key, new_hash)` in order, so a `DeleteField`
    /// landing after the `SetField` for the same row would invert that fold
    /// and release an external-value reference the commit still holds.
    ///
    /// The delta below is the tightest shape that can expose it: one delta
    /// cascade-deletes a section, re-creates the same `section_id`, and then
    /// writes the same field key and the same time-sample key back.
    ///
    /// The section also holds a time-0 sample, which the delta does not
    /// re-set, so the cascade deletes it at both `+0.0` and `-0.0` (rule E):
    /// the added `-0.0` record belongs to the same delete run and must also
    /// precede the delta's re-set `SetTimeSample`.
    #[test]
    fn apply_delta_cascade_purge_emissions_precede_set_field_for_the_same_row() {
        let mut state = make_state();
        apply_delta(
            &mut state,
            Some(OwnedDelta {
                is_diff: false,
                struct_commands: vec![create_cmd(0, 5, "Five")],
                set_fields: vec![OodfSetField {
                    section_id: 5,
                    key_name: "k".to_string(),
                    value_or_ext_hash: vec![1],
                    ext_value_size: 0,
                    ext_value_index: 0,
                    set_order: 0,
                }],
                time_samples: vec![
                    OodfTimeSample {
                        section_id: 5,
                        time: 1.0,
                        value_or_ext_hash: vec![1],
                        ext_value_size: 0,
                        ext_value_index: 0,
                    },
                    OodfTimeSample {
                        section_id: 5,
                        time: 0.0,
                        value_or_ext_hash: vec![1],
                        ext_value_size: 0,
                        ext_value_index: 0,
                    },
                ],
                ..Default::default()
            }),
        )
        .expect("arrangement delta must apply");

        let (mutations, stats) = apply_counted(
            &mut state,
            OwnedDelta {
                is_diff: false,
                struct_commands: vec![delete_cmd(5), change_cmd(0, 5, "Five", 0)],
                set_fields: vec![OodfSetField {
                    section_id: 5,
                    key_name: "k".to_string(),
                    value_or_ext_hash: vec![2],
                    ext_value_size: 0,
                    ext_value_index: 0,
                    set_order: 0,
                }],
                time_samples: vec![OodfTimeSample {
                    section_id: 5,
                    time: 1.0,
                    value_or_ext_hash: vec![2],
                    ext_value_size: 0,
                    ext_value_index: 0,
                }],
                ..Default::default()
            },
        );

        let delete_field = mutations
            .iter()
            .position(|m| {
                matches!(m, StorageMutation::DeleteField { section_id, key_name }
                         if *section_id == 5 && key_name == "k")
            })
            .expect("the cascade must erase the field of the deleted section");
        let set_field = mutations
            .iter()
            .position(|m| {
                matches!(m, StorageMutation::SetField { field }
                         if field.section_id == 5 && field.key_name == "k")
            })
            .expect("the delta's SetField must be emitted");
        assert!(
            delete_field < set_field,
            "DeleteField (index {delete_field}) must precede SetField (index {set_field}) \
             for the same row, or `ext_rows_written` releases a live reference"
        );

        let delete_sample = mutations
            .iter()
            .position(|m| {
                matches!(m, StorageMutation::DeleteTimeSample { section_id, .. }
                         if *section_id == 5)
            })
            .expect("the cascade must erase the time sample of the deleted section");
        let set_sample = mutations
            .iter()
            .position(|m| {
                matches!(m, StorageMutation::SetTimeSample { sample }
                         if sample.section_id == 5)
            })
            .expect("the delta's SetTimeSample must be emitted");
        assert!(
            delete_sample < set_sample,
            "DeleteTimeSample (index {delete_sample}) must precede SetTimeSample \
             (index {set_sample}) for the same row"
        );
        let added_negative_zero = mutations
            .iter()
            .position(|m| {
                matches!(m, StorageMutation::DeleteTimeSample { section_id, time }
                         if *section_id == 5 && time.into_inner() == 0.0
                             && time.into_inner().is_sign_negative())
            })
            .expect("rule E must delete the purged time 0 at -0.0 too");
        assert!(
            added_negative_zero < set_sample,
            "rule E's -0.0 DeleteTimeSample (index {added_negative_zero}) must precede the \
             re-set SetTimeSample (index {set_sample}): it belongs to the purge's delete run"
        );

        assert_eq!(stats.field_purge_passes, 1);
        // The re-write survives the purge: the row is live at the end, and
        // the time-0 sample the delta did not re-set is gone.
        assert!(state.fields.contains_key(&(5, "k".to_string())));
        assert!(state.time_samples.contains_key(&(5, OrderedFloat(1.0))));
        assert!(!state.time_samples.contains_key(&(5, OrderedFloat(0.0))));
    }

    /// Delete-and-recreate of the same parent inside one delta lands as
    /// `DeleteChildrenList` **then** `SetChildren` — `DEL … DEL … RPUSH` at
    /// the storage layer, same end state.
    ///
    /// This is the ordering argument the finalization pass rests on: every
    /// inline children-list delete (cascade, reset) is emitted during the
    /// structural phase, and the finalizer's writes all come after it, so a
    /// parent can never be "finalized" back into existence after its delete,
    /// nor deleted after its rebuild.
    ///
    /// The delta deletes section 1 (cascading to its child 2), creates section
    /// 1 again and then lists a new section under it — the way a children key
    /// gets re-materialized under an id whose list the same delta deleted.
    /// The re-creation is what makes the second list exist at all: under an
    /// id with no row the engine keeps no list (SDD §10.4).
    #[test]
    fn apply_delta_delete_then_recreate_parent_emits_delete_before_the_rebuild() {
        let mut state = make_state();
        seed_section(&mut state, 1, 0, "P", 0);
        seed_section(&mut state, 2, 1, "C", 0);
        state.children.insert(0, vec![1]);
        state.children.insert(1, vec![2]);

        let (mutations, stats) = apply_counted(
            &mut state,
            batch(vec![
                delete_cmd(1),
                create_cmd(0, 1, "P"),
                change_cmd(1, 9, "New", 0),
            ]),
        );

        let inline_delete = mutations
            .iter()
            .position(|m| matches!(m, StorageMutation::DeleteChildrenList { section_id: 1 }))
            .expect("the cascade must delete the children list of section 1");
        let rebuild = mutations
            .iter()
            .position(|m| matches!(m, StorageMutation::SetChildren { section_id: 1, .. }))
            .expect("the re-materialized list must be written back");
        assert!(
            inline_delete < rebuild,
            "DeleteChildrenList (index {inline_delete}) must precede the rebuild \
             (index {rebuild}) so the DEL cannot erase the RPUSH"
        );

        let writes = set_children_writes(&mutations);
        assert_eq!(
            writes,
            vec![(0, vec![1]), (1, vec![9])],
            "one write per dirty parent: the root, which lost and regained 1, and \
             the re-materialized parent 1"
        );
        assert_eq!(
            deleted_children_lists(&mutations),
            vec![1, 2],
            "the cascade's own deletes, emitted inline during the structural phase"
        );
        assert_eq!(
            stats.children_writes_emitted as usize,
            writes.len() + 2,
            "the counter accounts for both arms"
        );
        assert_eq!(state.children[&1], vec![9]);
    }

    /// A parent whose list is emptied still gets a write carrying the **empty
    /// vector**, rather than having the write skipped.
    ///
    /// The empty `SetChildren` is the `DEL`-without-`RPUSH` plus
    /// `SREM le:{oid}:obj:chs` fork at the storage layer; dropping it would
    /// leave the parent stranded in the children index while its key no longer
    /// exists, breaking SDD §7.7's "member present ⟺ children key exists".
    /// (The storage-side halves of that fork are pinned in
    /// `tests/integration/storage/b_9fn_tests.rs`; this is the domain-side
    /// half — that the mutation is emitted at all.)
    #[test]
    fn apply_delta_emptied_parent_list_still_emits_an_empty_set_children() {
        let mut state = make_state();
        seed_section(&mut state, 1, 0, "P", 0);
        seed_section(&mut state, 2, 1, "OnlyChild", 0);
        state.children.insert(0, vec![1]);
        state.children.insert(1, vec![2]);

        let (mutations, stats) = apply_counted(&mut state, batch(vec![delete_cmd(2)]));

        assert_eq!(
            set_children_writes(&mutations),
            vec![(1, Vec::new())],
            "the emptied parent's list is written as an empty vector, not dropped"
        );
        assert_eq!(state.children[&1], Vec::<u64>::new());
        assert_eq!(
            stats.children_writes_emitted, 2,
            "the empty rebuild plus the deleted child's own children-list delete"
        );
    }

    // ── Cache correctness: shapes only reachable from stored state ───────────
    //
    // The three tests below share an arrangement property that is worth
    // stating once: each starts from an `ObjectState` that is cross-map
    // INCONSISTENT — an id listed under a parent that is not its own, or
    // listed with no row at all. Both shapes are legitimate stored state (SDD
    // §10.4; a snapshot read `LRANGE`s a children list back verbatim),
    // and each delta below leaves the state consistent again by its final
    // command, which is why `debug_assert_cross_map_consistent` — which runs
    // only at the delta boundary — tolerates them.
    //
    // That same property is why these three cannot use
    // `apply_one_command_per_delta` as their oracle: splitting the delta puts
    // a boundary in the middle of the inconsistent window, and the reference
    // run would trip the debug assert rather than produce an answer. The
    // expected order is therefore spelled out literally, with the pre-fix
    // answer named in each case so a regression reads as "the cache went stale
    // again" rather than as an unexplained order change.

    /// Cross-parent order change: a child listed under parent 1 but stored
    /// under parent 2 has its `section_order` rewritten through parent 2, and
    /// parent 1's cache — which is mirroring that order — must notice.
    ///
    /// Nothing on the list-side maintenance path touches parent 1 here: the
    /// command names parent 2, and the child's position in parent 1's list
    /// does not change. Only the dependency index closes it — parent 1 is
    /// registered as a lister of section 3 when its cache is built, so the
    /// order write finds it (see `ChildPlan::invalidate_order_listeners`).
    /// With the invalidation, the third command rebuilds parent 1's cache,
    /// sees `orders == [50, 0]` (non-monotone), takes the linear arm and
    /// inserts its order-0 row at the front. Without it, the stale
    /// `orders == [0, 0]` reads as sorted and the insertion appends.
    ///
    /// Both insertions are explicit order-0 rows: a `CreateSection` appends
    /// past every key, stale or fresh, and lands last either way. (What a
    /// stale cache does to a create's *key* is pinned by
    /// `apply_delta_create_after_a_cross_parent_order_raise_stores_the_true_max_plus_one`.)
    ///
    /// Final `children[1]`: **`[10, 9]`** with the fix, `[9, 10]` before it.
    #[test]
    fn apply_delta_cross_parent_order_change_invalidates_the_listing_parents_cache() {
        let mut state = make_state();
        seed_section(&mut state, 1, 0, "P1", 0);
        seed_section(&mut state, 2, 0, "P2", 0);
        // Section 3's row says parent 2, but parent 1 is the one listing it —
        // a stale listing (see `insert_child_into_parent`).
        seed_section(&mut state, 3, 2, "C", 0);
        state.children.insert(0, vec![1, 2]);
        state.children.insert(1, vec![3]);
        state.children.insert(2, Vec::new());

        apply_counted(
            &mut state,
            batch(vec![
                // Populates parent 1's cache while section 3 still reads order 0.
                change_cmd(1, 9, "N9", 0),
                // Rewrites section 3's order THROUGH PARENT 2 — parent 1 is
                // not named anywhere in this command.
                change_cmd(2, 3, "C", 50),
                // Reads parent 1's cache again. The insertion point depends on
                // whether it still believes section 3 has order 0.
                change_cmd(1, 10, "N10", 0),
                // Repair: section 3's row already names parent 2, so the move
                // renames it in place, unlinks it from the carried parent 1 and
                // lists it under parent 2, leaving the delta cross-map
                // consistent.
                move_cmd(3, 1, 2, "C"),
            ]),
        );

        assert_eq!(
            state.children[&1],
            vec![10, 9],
            "the order-50 entry must be visible to parent 1's insertion point \
             (a stale cache appends the order-0 row and gives [9, 10])"
        );
        assert_eq!(
            state.children[&2],
            vec![3],
            "section 3 landed under its row's parent"
        );
    }

    /// Cascade-delete as the order-side write: a section with a non-zero order
    /// is deleted, and a parent that still lists it (and cached its real
    /// order) must stop believing that order — a ghost id caches as 0.
    ///
    /// Every delete path removes the section from the list its *row* names, so
    /// the only parent that can still list it afterwards is one holding a
    /// stale listing in stored state: here parent 2 lists section 5, whose row
    /// names parent 9 (which lists it too). The delete is truthful — it goes
    /// through parent 9 — and parent 2's listing survives it as a ghost. That
    /// is the shape `cascade_delete_section`'s order-side invalidation exists
    /// for, reachable today only from stored stale listings: the row's
    /// effective order drops from 50 to the ghost 0.
    ///
    /// The insertions are explicit order-0 rows, because that is the insert
    /// whose position depends on the cached entry for section 5 (a
    /// `CreateSection` appends past every key and would land last against a
    /// stale or a fresh cache alike).
    ///
    /// Final `children[2]`: **`[7, 5, 6]`** with the invalidation, `[7, 6, 5]`
    /// without it (the stale cache still reads section 5 at order 50, so the
    /// new order-0 child sorts ahead of it).
    #[test]
    fn apply_delta_cascade_delete_of_an_ordered_child_invalidates_a_stale_listing_cache() {
        let mut state = make_state();
        seed_section(&mut state, 2, 0, "P2", 0);
        seed_section(&mut state, 9, 0, "P9", 1);
        // Section 5's row names parent 9, which lists it; parent 2 lists it
        // too — a stale listing in stored state.
        seed_section(&mut state, 5, 9, "C5", 50);
        state.children.insert(0, vec![2, 9]);
        state.children.insert(9, vec![5]);
        state.children.insert(2, vec![5]);

        apply_counted(
            &mut state,
            batch(vec![
                // Populates parent 2's cache while section 5 still has order 50.
                change_cmd(2, 7, "C7", 0),
                // A truthful delete: the cascade removes section 5 from the
                // list its row names (parent 9). Parent 2's stale listing of
                // it survives, now as a ghost.
                delete_cmd(5),
                // Reads parent 2's cache again.
                change_cmd(2, 6, "C6", 0),
                // Repair: give the ghost id a row under the parent still
                // listing it, so the delta ends cross-map consistent.
                change_cmd(2, 5, "C5", 0),
            ]),
        );

        assert_eq!(
            state.children[&2],
            vec![7, 5, 6],
            "a deleted child reads as a ghost (order 0), so the new order-0 child lands \
             after it (a stale order-50 entry would push it to [7, 6, 5])"
        );
        assert!(
            state.children[&9].is_empty(),
            "non-vacuity — the delete went through the list the row named"
        );
    }

    /// Ghost-to-live as the order-side write: an id that is listed with no row
    /// gains one at a non-zero order, and the parent that cached it under the
    /// ghost encoding (`0`) must stop believing that.
    ///
    /// The insertions are explicit order-0 rows, for the reason the cascade
    /// test above gives.
    ///
    /// Final `children[1]`: **`[10, 77, 9]`** with the invalidation,
    /// `[77, 9, 10]` without it.
    #[test]
    fn apply_delta_ghost_id_gaining_an_ordered_row_invalidates_the_listing_parents_cache() {
        let mut state = make_state();
        seed_section(&mut state, 1, 0, "P", 0);
        state.children.insert(0, vec![1]);
        // Id 77 is listed under section 1 but has no row: a ghost.
        state.children.insert(1, vec![77]);

        apply_counted(
            &mut state,
            batch(vec![
                // Populates parent 1's cache while 77 is still a ghost (order 0).
                change_cmd(1, 9, "N9", 0),
                // 77 gains a row at order 50. The insert that follows inside
                // this command returns early (77 is already a member), so it
                // refreshes nothing by itself.
                change_cmd(1, 77, "G", 50),
                // Reads parent 1's cache again.
                change_cmd(1, 10, "N10", 0),
            ]),
        );

        assert_eq!(
            state.children[&1],
            vec![10, 77, 9],
            "the order-50 row must be visible to the insertion point \
             (the ghost encoding left cached gives [77, 9, 10])"
        );
    }

    /// A children list holding the same id twice — legacy stored state, from
    /// before `ReorderChildren` stopped storing a repeated id (b.dmu item 4;
    /// it now counts the first occurrence) and insert de-duplicated.
    ///
    /// All three list-side maintainers must keep their pre-b.ot3 answer on
    /// this shape, and `rebuild_children_list` is the one where it is subtle:
    /// a remove-and-reinsert re-places only the *first* occurrence and leaves
    /// an order the stable sort never produces, so a duplicated list must take
    /// the whole-list `sort_by_key` arm instead.
    ///
    /// The **two-command** shape is required. A single command against a fresh
    /// cache does not discriminate — the cache the detach builds is derived
    /// after the order write has landed, so it already reads unsorted and
    /// takes the sort arm for that reason. Populating the cache first (command
    /// one) and then raising an order (command two) is what routes the
    /// carried-forward cache into the arm choice, which is the case the
    /// `duplicated` flag exists for.
    ///
    /// Expected `[B, D, A, A]` — the stable `sort_by_key` answer over keys
    /// `[9, 9, 7, 7]`. A remove-and-reinsert would give `[A, B, D, A]`.
    ///
    /// b.sao restates the same argument in the `keep` vocabulary, and asserts
    /// the counters that make the **self-disposal** observable. The order write
    /// on `A` keeps `P`'s cache (one dependency step, and `P` is the row's own
    /// parent), which on an ordinary list would let the next statement repair
    /// one entry. Here it cannot: the write falsifies *both* of `A`'s entries,
    /// so `rebuild_children_list` routes to its whole-list `sort_by_key` arm,
    /// which re-derives the cache from `state.sections` and discards the kept
    /// one entirely. `parent_cache_builds == 2` is that second derivation — the
    /// kept cache's soundness argument does not have to hold on this shape,
    /// because nothing of the kept cache survives it.
    #[test]
    fn apply_delta_order_change_on_a_legacy_duplicate_id_list_keeps_the_stable_sort_answer() {
        const A: u64 = 10;
        const B: u64 = 20;
        const D: u64 = 30;
        const P: u64 = 1;

        fn arrange() -> ObjectState {
            let mut state = make_state();
            seed_section(&mut state, P, 0, "P", 0);
            seed_section(&mut state, A, P, "A", 5);
            seed_section(&mut state, B, P, "B", 7);
            state.children.insert(0, vec![P]);
            // Only reachable from state stored before the duplicate-id fixes.
            state.children.insert(P, vec![A, A, B]);
            state
        }

        let delta = batch(vec![
            // Populates parent P's cache, and lands D at the end of the
            // equal-order-7 tie group: [A, A, B, D].
            change_cmd(P, D, "D", 7),
            // Raises A to order 9 under the same parent → rebuild_children_list
            // over a list that holds A twice.
            change_cmd(P, A, "A", 9),
        ]);

        let mut batched = arrange();
        let (_mutations, stats) = apply_counted(&mut batched, delta.clone());

        // This arrangement IS cross-map consistent (a duplicate id in one list
        // is not drift), so the one-command-per-delta oracle applies here.
        let mut reference = arrange();
        apply_one_command_per_delta(&mut reference, &delta);

        assert_eq!(
            batched.children[&P],
            vec![B, D, A, A],
            "a duplicated list takes the stable whole-list sort, not a remove-and-reinsert"
        );
        assert_eq!(
            stats.cache_dependency_steps, 1,
            "the order write on A inspects the one parent whose cache lists it, and \
             keeps that cache — A's own parent"
        );
        assert_eq!(
            stats.parent_cache_builds, 2,
            "the kept cache self-disposes: the sort arm re-derives P's cache wholesale \
             rather than repairing either of A's two falsified entries"
        );
        assert_eq!(
            structural_shape(&batched),
            structural_shape(&reference),
            "batched application must equal the one-command-per-delta replay"
        );
    }

    /// Removing a child from a legacy duplicate-id list removes **every**
    /// occurrence — the `retain` fallback, which keeps the pre-b.ot3
    /// semantics.
    #[test]
    fn apply_delta_removing_a_child_from_a_duplicate_id_list_removes_every_occurrence() {
        const A: u64 = 10;
        const B: u64 = 20;
        const P: u64 = 1;

        let mut state = make_state();
        seed_section(&mut state, P, 0, "P", 0);
        seed_section(&mut state, A, P, "A", 5);
        seed_section(&mut state, B, P, "B", 7);
        state.children.insert(0, vec![P]);
        state.children.insert(P, vec![A, A, B]);

        apply_counted(&mut state, batch(vec![delete_cmd(A)]));

        assert_eq!(
            state.children[&P],
            vec![B],
            "both occurrences of the removed id are gone"
        );
        assert!(!state.sections.contains_key(&A));
    }

    /// Inserting a *new* child into a legacy duplicate-id list lands where the
    /// pre-b.ot3 scan put it.
    ///
    /// `insert_child_into_parent` deliberately does **not** fall back on this
    /// shape: the id it inserts occurs nowhere in the list, so the positional
    /// patch touches no occurrence of the repeated id and the cache stays
    /// faithful. The explicit order-0 child sorts ahead of the whole list (a
    /// `CreateSection` would append and land last, which no insertion point
    /// could get wrong).
    #[test]
    fn apply_delta_inserting_into_a_duplicate_id_list_uses_the_pre_fix_position() {
        const A: u64 = 10;
        const B: u64 = 20;
        const C: u64 = 30;
        const P: u64 = 1;

        let mut state = make_state();
        seed_section(&mut state, P, 0, "P", 0);
        seed_section(&mut state, A, P, "A", 5);
        seed_section(&mut state, B, P, "B", 7);
        state.children.insert(0, vec![P]);
        state.children.insert(P, vec![A, A, B]);

        apply_counted(&mut state, batch(vec![change_cmd(P, C, "C", 0)]));

        assert_eq!(
            state.children[&P],
            vec![C, A, A, B],
            "the order-0 insertion sorts ahead of the order-5 entries, duplicates untouched"
        );
    }

    // ── Duplicate-name determinism (b.ot3 P2) ────────────────────────────────
    //
    // Two siblings can legally share `(parent_id, name)`: `apply_move_section`'s
    // same-parent rename performs no conflict check, and that is deliberately
    // unchanged (rejecting it would be a client-visible wire contract change,
    // PRD §2.3.4 / SDD §9.3.1). What b.ot3 fixes is that the RESOLUTION of such
    // a duplicate is deterministic — the lowest `section_id` is the conflict
    // target — where the pre-fix `HashMap::values().find()` answered from a
    // process-local hash seed. The stakes are not cosmetic: the pick decides
    // which whole subtree is cascade-deleted, and for `MoveSection` whether
    // edge 4 or edge 5 applies.

    /// The unchanged half: a same-parent rename onto an occupied sibling name
    /// succeeds and produces the duplicate.
    ///
    /// Pinned explicitly so a future "tidy-up" that adds a conflict check here
    /// fails a test that says why it must not.
    #[test]
    fn apply_delta_same_parent_rename_onto_an_occupied_name_creates_a_duplicate() {
        let mut state = make_state();
        seed_section(&mut state, 1, 0, "P", 0);
        seed_section(&mut state, 10, 1, "A", 0);
        seed_section(&mut state, 20, 1, "B", 0);
        state.children.insert(0, vec![1]);
        state.children.insert(1, vec![10, 20]);

        apply_delta(&mut state, Some(batch(vec![move_cmd(20, 1, 1, "A")])))
            .expect("a same-parent rename is unconditional (SDD §9.3.1)");

        assert_eq!(state.sections[&10].name, "A");
        assert_eq!(state.sections[&20].name, "A");
        assert_eq!(
            state.children[&1],
            vec![10, 20],
            "a rename touches no children list"
        );
    }

    /// `CreateSection` onto a duplicated name cascade-deletes the **lowest**
    /// id's whole subtree, and leaves the other holder's subtree entirely
    /// alone.
    ///
    /// Both name holders carry a child section and a field, so the assertion
    /// covers what the pick actually decides: descendants, tombstones and
    /// field rows, not just which row disappears.
    #[test]
    fn apply_delta_create_section_name_conflict_cascades_the_lowest_id_subtree() {
        let mut state = make_state();
        seed_section(&mut state, 1, 0, "P", 0);
        seed_section(&mut state, 10, 1, "A", 0);
        seed_section(&mut state, 11, 10, "child-of-10", 0);
        seed_section(&mut state, 20, 1, "A", 0);
        seed_section(&mut state, 21, 20, "child-of-20", 0);
        state.children.insert(0, vec![1]);
        state.children.insert(1, vec![10, 20]);
        state.children.insert(10, vec![11]);
        state.children.insert(20, vec![21]);
        for sid in [10, 11, 20, 21] {
            state.fields.insert(
                (sid, "f".to_string()),
                Field {
                    section_id: sid,
                    key_name: "f".to_string(),
                    value: FieldValue::Inline(vec![1]),
                    set_order: 0,
                    authoring_set_order: 0,
                    version: 1,
                },
            );
            state.time_samples.insert(
                (sid, OrderedFloat(1.0)),
                TimeSample {
                    section_id: sid,
                    time: OrderedFloat(1.0),
                    value: FieldValue::Inline(vec![1]),
                    version: 1,
                },
            );
        }

        let (mutations, _) = apply_counted(&mut state, batch(vec![create_cmd(1, 30, "A")]));

        assert!(
            !state.sections.contains_key(&10) && !state.sections.contains_key(&11),
            "the lowest name holder and its descendants are cascade-deleted"
        );
        assert!(
            state.sections.contains_key(&20) && state.sections.contains_key(&21),
            "the other name holder's subtree is untouched"
        );
        assert!(!state.fields.contains_key(&(10, "f".to_string())));
        assert!(!state.fields.contains_key(&(11, "f".to_string())));
        assert!(state.fields.contains_key(&(20, "f".to_string())));
        assert!(state.fields.contains_key(&(21, "f".to_string())));
        assert!(!state.time_samples.contains_key(&(10, OrderedFloat(1.0))));
        assert!(state.time_samples.contains_key(&(21, OrderedFloat(1.0))));

        let mut tombstoned: Vec<u64> = mutations
            .iter()
            .filter_map(|m| match m {
                StorageMutation::CreateTombstone { section_id, .. } => Some(*section_id),
                _ => None,
            })
            .collect();
        tombstoned.sort();
        assert_eq!(
            tombstoned,
            vec![10, 11],
            "tombstones follow the same pick, for the whole doomed subtree"
        );
        assert_eq!(state.children[&1], vec![20, 30]);
    }

    /// Two same-named `CreateSection`s in **one** delta: the earlier-created
    /// sibling is the cascade target, exactly as the one-command-per-delta
    /// replay resolves it.
    ///
    /// This is the test that pins the `NameIndex`'s **incremental insert** hook
    /// in [`apply_create_section`]. The plan's first `(parent_id, name)` probe
    /// scans `state.sections` and builds nothing; its second builds the index
    /// from `state.sections` as they stand then. So a leading unrelated probe
    /// (command 0, under the root) makes command 1's own probe the second: it
    /// builds the index *before* command 1 inserts `(1, "A")`, and the parent
    /// is seeded with no `"A"` at all, so the build cannot supply the hit.
    /// Only the `plan.name_inserted(...)` performed while command 1 executes
    /// can make command 3 see a conflict. Drop that call and the delta
    /// silently admits two siblings sharing `(parent_id, name)`, where both
    /// this replay oracle and the pre-b.ot3 engine (which re-read
    /// `state.sections` at every lookup) cascade-delete the first — a semantic
    /// divergence no other test in the suite catches, because every other
    /// same-name case is a single-command delta against a pre-seeded holder
    /// and therefore exercises only the scan or `NameIndex::build`. Without
    /// command 0 the build would run at command 2's probe, after command 1's
    /// insert, and read `(1, "A")` straight off `state.sections`: the hook
    /// would go untested.
    ///
    /// **Insert is the only correctness-bearing direction.**
    /// [`find_child_by_name`] re-verifies every bucket candidate against
    /// `state.sections` before reporting it, so a *stale* entry is filtered at
    /// probe time and [`ChildPlan::name_removed`] / [`NameIndex::remove`] are
    /// pure bucket hygiene — removing them changes no observable resolution.
    #[test]
    fn apply_delta_same_name_creates_in_one_delta_cascade_the_earlier_sibling() {
        fn arrange() -> ObjectState {
            let mut state = make_state();
            seed_section(&mut state, 1, 0, "P", 0);
            state.children.insert(0, vec![1]);
            state
        }

        let delta = batch(vec![
            // An unrelated first probe, so command 1's probe is the second
            // and builds the index before command 1's insert.
            create_cmd(0, 90, "Probe"),
            // First holder of (1, "A") — created inside this delta, so the
            // name index only knows about it via the incremental hook.
            create_cmd(1, 10, "A"),
            // ... carrying a subtree, so the assertion covers the cascade and
            // not merely which row disappears.
            create_cmd(10, 11, "child-of-10"),
            // Second holder of the same name: must cascade-delete 10.
            create_cmd(1, 20, "A"),
        ]);

        let mut batched = arrange();
        let (mutations, _) = apply_counted(&mut batched, delta.clone());

        let mut reference = arrange();
        apply_one_command_per_delta(&mut reference, &delta);

        assert_eq!(
            structural_shape(&batched),
            structural_shape(&reference),
            "an in-delta same-name create must resolve exactly as the \
             one-command-per-delta replay does"
        );

        assert!(
            !batched.sections.contains_key(&10) && !batched.sections.contains_key(&11),
            "the earlier-created sibling and its whole subtree are cascade-deleted"
        );
        assert!(
            batched.sections.contains_key(&20),
            "the later-created sibling survives"
        );
        assert_eq!(
            batched.children[&1],
            vec![20],
            "the survivor is the only child the parent is left holding"
        );

        let mut tombstoned: Vec<u64> = mutations
            .iter()
            .filter_map(|m| match m {
                StorageMutation::CreateTombstone { section_id, .. } => Some(*section_id),
                _ => None,
            })
            .collect();
        tombstoned.sort();
        assert_eq!(
            tombstoned,
            vec![10, 11],
            "tombstones follow the same pick, for the whole doomed subtree"
        );
    }

    /// The `DiffChangeSection` twin of the test above: a row created by the
    /// diff path's **new-section** branch is visible to a same-named command
    /// later in the same delta.
    ///
    /// Pins the second incremental insert hook — the `plan.name_inserted(...)`
    /// in [`apply_diff_change_section`]'s new-row branch. Both shapes of
    /// follow-up command are driven, because they reach the conflict through
    /// different call sites: `CreateSection`'s probe, and `DiffChangeSection`'s
    /// own probe landing on its new-section branch. As above, a leading
    /// unrelated probe makes command 1's probe the plan's second, so the
    /// index is built before command 1 inserts its row, and the parent is
    /// seeded without the name, so a fresh `NameIndex::build` cannot answer
    /// the follow-up probe — only the incremental insert can.
    #[test]
    fn apply_delta_diff_change_new_row_then_same_name_command_cascades_the_new_row() {
        fn arrange() -> ObjectState {
            let mut state = make_state();
            seed_section(&mut state, 1, 0, "P", 0);
            state.children.insert(0, vec![1]);
            state
        }

        for (label, second_holder) in [
            ("CreateSection follow-up", create_cmd(1, 20, "A")),
            ("DiffChangeSection follow-up", change_cmd(1, 20, "A", 0)),
        ] {
            let delta = batch(vec![
                // An unrelated first probe, so the next probe builds the
                // index before the new row is inserted.
                create_cmd(0, 90, "Probe"),
                // New row via the diff path's new-section branch.
                change_cmd(1, 10, "A", 0),
                // ... with a subtree, likewise created by the diff path.
                change_cmd(10, 11, "child-of-10", 0),
                second_holder,
            ]);

            let mut batched = arrange();
            apply_counted(&mut batched, delta.clone());

            let mut reference = arrange();
            apply_one_command_per_delta(&mut reference, &delta);

            assert_eq!(
                structural_shape(&batched),
                structural_shape(&reference),
                "{label}: must resolve exactly as the one-command-per-delta replay does"
            );
            assert!(
                !batched.sections.contains_key(&10) && !batched.sections.contains_key(&11),
                "{label}: the earlier new row and its subtree are cascade-deleted"
            );
            assert!(
                batched.sections.contains_key(&20),
                "{label}: the later holder survives"
            );
            assert_eq!(
                batched.children[&1],
                vec![20],
                "{label}: the survivor is the only child the parent is left holding"
            );
        }
    }

    /// The `DiffChangeSection` **existing-row** twin: a row that acquires its
    /// `(parent, name)` by a rename — in place, or a rename plus reparent — is
    /// visible to a same-named command later in the same delta.
    ///
    /// Pins the `plan.name_inserted(...)` in [`apply_diff_change_section`]'s
    /// existing-row branch, which the new-row test above does not reach: that
    /// one exercises the sibling call in the same function's *new-section*
    /// branch. A leading unrelated probe makes the rename's own probe the
    /// plan's second, so `NameIndex::build` runs there, against
    /// `state.sections` before the rename lands, where row 10 is `(1, "B")`,
    /// and neither destination holds `"A"` before the delta: the build cannot
    /// answer the follow-up probe. Drop the incremental insert and the
    /// `CreateSection` finds no conflict and admits a second
    /// `(destination, "A")` holder instead of cascading, where both this
    /// replay oracle and the pre-b.ot3 engine cascade-delete the renamed row.
    /// (Without the leading probe the rename's probe would only scan, and the
    /// build at the `CreateSection` would read the renamed row straight off
    /// `state.sections`.)
    ///
    /// The rename has to land on a free name: the existing-row branch refuses
    /// a rename onto an occupied slot with `DuplicateSiblingName` rather than
    /// cascading, so an occupied destination would never reach the insert.
    #[test]
    fn apply_delta_diff_change_existing_row_rename_then_same_name_create_cascades_the_renamed_row()
    {
        fn arrange() -> ObjectState {
            let mut state = make_state();
            seed_section(&mut state, 1, 0, "P1", 0);
            seed_section(&mut state, 2, 0, "P2", 0);
            // The row that acquires `(destination, "A")` inside the delta. It
            // carries a child, so the assertion covers the cascade and not
            // merely which row disappears.
            seed_section(&mut state, 10, 1, "B", 0);
            seed_section(&mut state, 11, 10, "child-of-10", 0);
            state.children.insert(0, vec![1, 2]);
            state.children.insert(1, vec![10]);
            state.children.insert(10, vec![11]);
            state
        }

        // Both arms of the existing-row branch reach the same insert: the
        // same-parent rename, and the cross-parent arm that also re-links the
        // children lists.
        for (label, destination) in [("rename in place", 1u64), ("rename plus reparent", 2u64)] {
            let delta = batch(vec![
                // An unrelated first probe, so the rename's probe builds the
                // index before the rename lands.
                create_cmd(0, 90, "Probe"),
                // Existing row 10 acquires `(destination, "A")`.
                change_cmd(destination, 10, "A", 0),
                // ... which only the incremental insert can make this see.
                create_cmd(destination, 20, "A"),
            ]);

            let mut batched = arrange();
            apply_counted(&mut batched, delta.clone());

            let mut reference = arrange();
            apply_one_command_per_delta(&mut reference, &delta);

            assert_eq!(
                structural_shape(&batched),
                structural_shape(&reference),
                "{label}: must resolve exactly as the one-command-per-delta replay does"
            );
            assert!(
                !batched.sections.contains_key(&10) && !batched.sections.contains_key(&11),
                "{label}: the renamed row and its subtree are the cascade target"
            );
            assert!(
                batched.sections.contains_key(&20),
                "{label}: the later holder survives"
            );
            assert_eq!(
                batched.children[&destination],
                vec![20],
                "{label}: the survivor is the only child the destination is left holding"
            );
        }
    }

    /// `MoveSection` edge 5 against a *rename-created* duplicate targets the
    /// lowest id.
    ///
    /// The duplicate is produced inside the same delta by the rename in
    /// command 1, so the tie-break has to hold over a second holder that did
    /// not exist when the delta started.
    ///
    /// It does **not** discriminate the index's incremental insert. The pick
    /// is id 10, which `NameIndex::build` already supplies from the
    /// *pre-delta* `state.sections`; delete the `plan.name_inserted(...)` in
    /// [`apply_move_section`]'s same-parent rename arm and every assertion
    /// below is unchanged. The hooks that only the incremental insert can
    /// answer are pinned by the three
    /// `apply_delta_*_same_name_*` / `..._cascades_the_new_row` tests above
    /// and by `apply_delta_mixed_conflict_sequence_matches_one_per_delta`.
    #[test]
    fn apply_delta_move_section_edge5_conflict_targets_the_lowest_id() {
        let mut state = make_state();
        seed_section(&mut state, 1, 0, "P1", 0);
        seed_section(&mut state, 2, 0, "P2", 0);
        seed_section(&mut state, 10, 1, "A", 0);
        seed_section(&mut state, 20, 1, "B", 0);
        seed_section(&mut state, 5, 2, "Mover", 0);
        state.children.insert(0, vec![1, 2]);
        state.children.insert(1, vec![10, 20]);
        state.children.insert(2, vec![5]);

        apply_delta(
            &mut state,
            Some(batch(vec![
                // Two probes first — the plan's first probe scans, its second
                // builds — so the name index is built (and then maintained)
                // rather than built fresh at the conflicting command.
                create_cmd(1, 40, "Z"),
                create_cmd(1, 41, "Y"),
                // Rename 20 onto 10's name: two "A" siblings under parent 1.
                move_cmd(20, 1, 1, "A"),
                // Edge 5: the mover is not a descendant of the conflict target.
                move_cmd(5, 2, 1, "A"),
            ])),
        )
        .expect("delta must apply");

        assert!(
            !state.sections.contains_key(&10),
            "the lowest name holder is the one cascade-deleted"
        );
        assert!(
            state.sections.contains_key(&20),
            "the higher-id name holder survives"
        );
        assert_eq!(state.sections[&20].name, "A");
        assert_eq!(
            state.sections[&5].parent_id, 1,
            "the mover completed its move"
        );
        assert_eq!(state.sections[&5].name, "A");
    }

    /// The pick decides **which edge applies**, not merely which id an error
    /// names.
    ///
    /// Section 10 is an ancestor of the mover; section 20 is not; both are
    /// named "A" under parent 1. Picking 10 means edge 4 — the mover's own
    /// subtree is destroyed. Picking 20 would mean edge 5 — 20's subtree
    /// destroyed and the move completed. The two outcomes share no state, so
    /// this is the sharpest available assertion on determinism.
    #[test]
    fn apply_delta_move_section_lowest_id_pick_selects_edge4_over_edge5() {
        let mut state = make_state();
        seed_section(&mut state, 1, 0, "P", 0);
        seed_section(&mut state, 10, 1, "A", 0);
        seed_section(&mut state, 20, 1, "A", 0);
        seed_section(&mut state, 21, 20, "child-of-20", 0);
        // The mover sits under section 10, making 10 (and only 10) an ancestor.
        seed_section(&mut state, 30, 10, "Mover", 0);
        seed_section(&mut state, 31, 30, "child-of-mover", 0);
        state.children.insert(0, vec![1]);
        state.children.insert(1, vec![10, 20]);
        state.children.insert(10, vec![30]);
        state.children.insert(20, vec![21]);
        state.children.insert(30, vec![31]);

        apply_delta(&mut state, Some(batch(vec![move_cmd(30, 10, 1, "A")])))
            .expect("delta must apply");

        assert!(
            !state.sections.contains_key(&30) && !state.sections.contains_key(&31),
            "edge 4: the conflict target is an ancestor, so the MOVER's subtree is destroyed"
        );
        assert!(
            state.sections.contains_key(&20) && state.sections.contains_key(&21),
            "edge 5 did not fire: the non-ancestor holder is untouched"
        );
        assert!(state.sections.contains_key(&10));
        assert_eq!(state.children[&10], Vec::<u64>::new());
    }

    /// `DiffChangeSection`'s existing-row rename reports the lowest id in
    /// `DuplicateSiblingName::conflicting_id`.
    #[test]
    fn apply_delta_diff_change_rename_onto_a_duplicated_name_reports_the_lowest_id() {
        let mut state = make_state();
        seed_section(&mut state, 1, 0, "P", 0);
        seed_section(&mut state, 10, 1, "A", 0);
        seed_section(&mut state, 20, 1, "A", 0);
        seed_section(&mut state, 5, 1, "B", 0);
        state.children.insert(0, vec![1]);
        state.children.insert(1, vec![10, 20, 5]);

        let err = apply_delta(&mut state, Some(batch(vec![change_cmd(1, 5, "A", 0)])))
            .expect_err("a rename onto an occupied sibling name must be refused");

        match err {
            DeltaError::DuplicateSiblingName {
                section_id,
                parent_id,
                conflicting_id,
                ref name,
            } => {
                assert_eq!(section_id, 5);
                assert_eq!(parent_id, 1);
                assert_eq!(
                    conflicting_id, 10,
                    "the reported conflict is the lowest id sharing the name"
                );
                assert_eq!(name, "A");
            }
            other => panic!("expected DuplicateSiblingName, got {other:?}"),
        }
    }

    /// A mixed conflict sequence — two probes, rename-duplicate, create, move,
    /// reorder, delete, create — settles to one deterministic state.
    ///
    /// Resolution removes one name holder but does not converge the parent to
    /// a single holder, so a second conflicting command picks the next-lowest
    /// id; this sequence exercises exactly that succession. The
    /// `structural_shape` comparison against the one-command-per-delta replay
    /// is what pins the whole sequence rather than the individual picks: this
    /// arrangement is cross-map consistent throughout, so the oracle applies.
    ///
    /// Both of [`apply_move_section`]'s `NameIndex` incremental insert hooks
    /// are pinned here, and nowhere else:
    ///
    /// * The **same-parent rename** hook — the second conflict (command 5)
    ///   picks id 20, which is a `(1, "A")` holder only because command 3
    ///   renamed it there.
    /// * The **cross-parent move** hook — the final conflict picks id 40,
    ///   which acquired `(1, "A")` by the move in command 5. Pre-delta, 40
    ///   sits under parent 2 named "Mover".
    ///
    /// Drop either call and a later same-name command picks the next id up
    /// instead — 50 in both cases — leaving two live `(1, "A")` siblings and
    /// diverging from the replay.
    ///
    /// **Commands 1 and 2 are load-bearing, not decoration.** The plan's
    /// first `(parent_id, name)` probe scans `state.sections` and builds
    /// nothing; `NameIndex::build` runs lazily at its *second* probe, not at
    /// delta entry, and [`apply_move_section`]'s same-parent arm returns
    /// before it probes. With fewer than two earlier probes the build is
    /// therefore deferred until command 4 (or the scan answers it) and reads
    /// the already-renamed row straight off `state.sections`, so the
    /// same-parent hook supplies nothing the build did not — measured with one
    /// leading probe under the first-probe build: commenting out that hook
    /// left every assertion below green. The two probes pin the build to the
    /// pre-delta snapshot, where `(1, "A")` holds only id 10, and hand the
    /// hook its job back. They are aimed at the root rather than at parent 1
    /// so they perturb neither command 6's complete-list reorder nor the
    /// children assertion.
    ///
    /// The final create is likewise appended at the end rather than spliced in
    /// right after the move so that the reorder still names a live id; "after
    /// the move" is all the cross-parent hook's coverage requires.
    ///
    /// The persisted half of this sequence — `LRANGE` children and `SMEMBERS
    /// obj:chs` after a real commit — is asserted in
    /// `tests/integration/storage/b_ot3_tests.rs`.
    #[test]
    fn apply_delta_mixed_conflict_sequence_matches_one_per_delta() {
        fn arrange() -> ObjectState {
            let mut state = make_state();
            seed_section(&mut state, 1, 0, "P1", 0);
            seed_section(&mut state, 2, 0, "P2", 0);
            seed_section(&mut state, 10, 1, "A", 0);
            seed_section(&mut state, 20, 1, "B", 1);
            seed_section(&mut state, 30, 1, "C", 2);
            seed_section(&mut state, 40, 2, "Mover", 0);
            state.children.insert(0, vec![1, 2]);
            state.children.insert(1, vec![10, 20, 30]);
            state.children.insert(2, vec![40]);
            state
        }

        let delta = batch(vec![
            // Two probes under an unrelated parent: the first scans, the
            // second builds `NameIndex` here — against the *pre-delta*
            // `state.sections` — rather than at command 4. Without them the
            // build is deferred past the rename and reads the renamed row
            // straight off `state.sections`, which is what makes the
            // same-parent hook look redundant. See the doc-comment above.
            create_cmd(0, 70, "Z"),
            create_cmd(0, 71, "Y"),
            // Rename 20 onto "A": holders are now {10, 20}.
            move_cmd(20, 1, 1, "A"),
            // Create onto "A" → cascade-deletes the lowest holder, 10.
            create_cmd(1, 50, "A"),
            // Move onto "A" → the live holders are now {20, 50} and the
            // lowest is 20, which only the same-parent rename hook supplies.
            move_cmd(40, 2, 1, "A"),
            // A complete-list reorder of what is left under parent 1.
            reorder_cmd(1, vec![50, 30, 40]),
            // Delete one of the survivors.
            delete_cmd(30),
            // Create onto "A" once more. Live holders are {40, 50}; the
            // lowest is 40, which is a candidate only because the
            // cross-parent move above told the index so.
            create_cmd(1, 60, "A"),
        ]);

        let mut batched = arrange();
        let (mutations, _) = apply_counted(&mut batched, delta.clone());

        let mut reference = arrange();
        apply_one_command_per_delta(&mut reference, &delta);

        assert_eq!(
            structural_shape(&batched),
            structural_shape(&reference),
            "batched conflict resolution must equal the one-command-per-delta replay"
        );
        assert!(
            !batched.sections.contains_key(&10),
            "the first conflict took the lowest holder"
        );
        assert!(
            !batched.sections.contains_key(&20),
            "the second conflict took the next-lowest holder — id 20, which is a \
             (1, \"A\") holder only because the same-parent rename hook said so"
        );
        assert!(
            !batched.sections.contains_key(&40),
            "the third conflict took 40 — the row that acquired (1, \"A\") by a \
             cross-parent move inside this delta, which only the incremental \
             insert puts in the bucket"
        );
        assert!(
            batched.sections.contains_key(&50) && batched.sections.contains_key(&60),
            "the two higher-id holders survive"
        );
        assert_eq!(batched.children[&1], vec![50, 60]);

        let writes = set_children_writes(&mutations);
        let mut parents: Vec<u64> = writes.iter().map(|(p, _)| *p).collect();
        parents.sort();
        parents.dedup();
        assert_eq!(
            writes.len(),
            parents.len(),
            "no parent is written twice: {writes:?}"
        );
    }

    // ── Per-delta plan hooks with no other pinning test ──────────────────────
    //
    // Each test below is aimed at ONE hook of the b.ot3 plan machinery that a
    // hook-by-hook enumeration found unpinned: removing that single statement
    // leaves the rest of the suite green while changing observable behaviour.
    // The perturbation each one discriminates is named in its doc-comment, so
    // a future edit that makes the test redundant can be checked rather than
    // guessed at.

    /// An order change is the *only* thing a delta does to a parent: the
    /// parent still owes a `SetChildren`.
    ///
    /// `apply_diff_change_section`'s same-parent order-change arm registers
    /// the parent twice without dirtying it — `invalidate_order_listeners`
    /// keeps this parent's cache and `detach_cache` hangs one off the entry,
    /// neither of which sets `dirty`. The single dirtying statement on that arm is
    /// `rebuild_children_list`'s own `plan.mark_dirty(parent_id)`. Every other
    /// order-change test in this file dirties the parent through some *other*
    /// command in the same delta, so with that statement removed they all stay
    /// green while `state.children[P]` is reordered in memory and **no**
    /// children write is emitted — persisted order silently diverging from
    /// state, which survives until the next delta happens to rewrite the list.
    ///
    /// Discriminated by: deleting `plan.mark_dirty(parent_id)` from
    /// `rebuild_children_list`.
    #[test]
    fn apply_delta_lone_order_change_still_emits_the_reordered_children_write() {
        const P: u64 = 1;
        const C: u64 = 10;
        const SIB: u64 = 20;

        let mut state = make_state();
        seed_section(&mut state, P, 0, "P", 0);
        seed_section(&mut state, C, P, "C", 0);
        seed_section(&mut state, SIB, P, "S", 5);
        state.children.insert(0, vec![P]);
        state.children.insert(P, vec![C, SIB]);

        // The delta's only command, and it names P only through the order
        // change: no create, no delete, no move, no reorder.
        let (mutations, stats) = apply_counted(&mut state, batch(vec![change_cmd(P, C, "C", 9)]));

        assert_eq!(
            state.children[&P],
            vec![SIB, C],
            "the in-memory list is reordered by the raised section_order"
        );
        assert_eq!(
            set_children_writes(&mutations),
            vec![(P, vec![SIB, C])],
            "the reordered list must be persisted — without the write, storage keeps \
             the pre-delta order while state holds the new one"
        );
        assert_eq!(
            stats.children_writes_emitted, 1,
            "exactly one children-list write, for the one parent the delta touched"
        );
    }

    /// A parent dirtied earlier in the delta and then cascade-deleted gets
    /// **one** `DeleteChildrenList`, not two, and is never rebuilt.
    ///
    /// This is the only shape that makes `finalize_children_lists` reach its
    /// `plan.deleted_lists` guard: the finalizer walks the registered parents,
    /// finds one that is dirty and whose `state.children` entry is gone, and
    /// must emit nothing at all because the cascade already deleted the key
    /// inline. `apply_delta_delete_then_recreate_parent_emits_delete_before_
    /// the_rebuild` reads as if it covers this but does not — there the
    /// cascaded parent is re-materialized by a later command, so the finalizer
    /// takes the `Some(children)` arm and never reaches the guard.
    ///
    /// Discriminated by: deleting either the guard in
    /// `finalize_children_lists` or the `plan.deleted_lists.insert(sid)` that
    /// feeds it from `cascade_delete_section` — both produce a second
    /// `DeleteChildrenList` for P and a fourth counted children write.
    #[test]
    fn apply_delta_cascade_deleting_a_dirtied_parent_emits_exactly_one_children_delete() {
        const P: u64 = 1;
        const X: u64 = 9;

        let mut state = make_state();
        seed_section(&mut state, P, 0, "P", 0);
        state.children.insert(0, vec![P]);
        state.children.insert(P, Vec::new());

        let (mutations, stats) = apply_counted(
            &mut state,
            // Command 1 dirties P; command 2 cascade-deletes it.
            batch(vec![create_cmd(P, X, "x"), delete_cmd(P)]),
        );

        assert_eq!(
            deleted_children_lists(&mutations),
            vec![P, X],
            "P's list is deleted exactly once, by the cascade — the finalizer must not \
             add a second delete for the parent it still has registered as dirty"
        );
        assert_eq!(
            set_children_writes(&mutations),
            vec![(0, Vec::new())],
            "the finalizer must never write a deleted key back into existence; the only \
             rebuild is the emptied root"
        );
        assert_eq!(
            stats.children_writes_emitted, 3,
            "two inline deletes plus the root's rebuild — the skipped parent is not counted"
        );
    }

    /// The reset path's own `deleted_lists` bookkeeping: a parent whose
    /// children key `apply_reset` already deleted inline is skipped by the
    /// finalizer, while the synthetic root — which has no section row and so
    /// no inline delete — still gets one.
    ///
    /// The root's delete is the reset fold-in b.ot3 added (`le:{oid}:obj:ch:0`
    /// used to survive a reset and be read back into the next commit); the
    /// skip is what stops that fold-in from double-deleting every *section*
    /// parent alongside it. Asserting the mutation list rather than the final
    /// key state is what separates the two: both are absent at the end either
    /// way.
    ///
    /// The per-section drain order follows `HashMap` iteration, so only the
    /// multiset of deleted ids is deterministic.
    ///
    /// Discriminated by: deleting `plan.deleted_lists.insert(sid)` from
    /// `apply_reset`, which re-adds a delete for section 1.
    #[test]
    fn apply_delta_reset_emits_exactly_one_children_delete_per_registered_parent() {
        let mut state = make_state();
        seed_section(&mut state, 1, 0, "P", 0);
        seed_section(&mut state, 2, 1, "C", 0);
        state.children.insert(0, vec![1]);
        state.children.insert(1, vec![2]);

        let (mutations, stats) = apply_counted(
            &mut state,
            OwnedDelta {
                is_diff: true,
                base_version: 0,
                ..Default::default()
            },
        );

        let mut deletes = deleted_children_lists(&mutations);
        deletes.sort_unstable();
        assert_eq!(
            deletes,
            vec![0, 1, 2],
            "one delete for each of the two drained sections and one for the synthetic \
             root — section 1 must not be deleted twice for being registered as dirty"
        );
        assert!(
            set_children_writes(&mutations).is_empty(),
            "a reset clears every list; nothing may be finalized back into existence"
        );
        assert_eq!(
            stats.children_writes_emitted, 3,
            "the counter matches the emitted deletes exactly"
        );
    }

    /// A cascade-deleted parent whose list is re-materialized later in the
    /// same delta must be re-inserted into through a **fresh** cache.
    ///
    /// `cascade_delete_section` drops `state.children[sid]` wholesale, so the
    /// cache describing that list survives the removal describing nothing.
    /// The four-command shape is what reaches it: command 1 builds P's cache
    /// and leaves it holding one entry, command 2 removes the list under it,
    /// command 3 creates P again, and command 4 re-materializes the list
    /// *empty* and then asks the cache for an insertion point. (Command 3 is
    /// what lets a list exist under P again at all: under an id with no row
    /// the engine keeps none, SDD §10.4.) Without the invalidation the stale
    /// `orders == [0]` puts `partition_point` past the end of the fresh empty
    /// list and `Vec::insert` panics — the failure is a panic, not a wrong
    /// order, which is why the current code passing this shape is itself part
    /// of the assertion.
    ///
    /// `apply_delta_delete_then_recreate_parent_emits_delete_before_the_
    /// rebuild` does not reach this: it never builds parent 1's cache before
    /// the cascade, so the slot is already empty when the delete runs.
    ///
    /// Discriminated by: deleting the `plan.invalidate_cache(sid)` that
    /// follows `state.children.remove(&sid)` on `cascade_delete_section`'s
    /// live-row arm.
    #[test]
    fn apply_delta_cascaded_parents_re_materialized_list_uses_a_fresh_cache() {
        const P: u64 = 1;
        const C1: u64 = 2;
        const C2: u64 = 3;
        const NEW: u64 = 9;

        let mut state = make_state();
        seed_section(&mut state, P, 0, "P", 0);
        seed_section(&mut state, C1, P, "C1", 0);
        seed_section(&mut state, C2, P, "C2", 0);
        state.children.insert(0, vec![P]);
        state.children.insert(P, vec![C1, C2]);

        apply_counted(
            &mut state,
            batch(vec![
                // Builds P's cache and leaves it describing one remaining
                // child (C2) at order 0.
                delete_cmd(C1),
                // Cascades P itself: `state.children[P]` is removed, and the
                // cache describing it is now describing nothing.
                delete_cmd(P),
                // P has a row again; the rows that named it are gone, so it
                // adopts nothing and still has no list.
                create_cmd(0, P, "P"),
                // Re-materializes `state.children[P]` as an empty list, then
                // asks for an insertion point.
                change_cmd(P, NEW, "New", 0),
            ]),
        );

        assert_eq!(
            state.children[&P],
            vec![NEW],
            "the re-materialized list holds only the new row (with the stale cache, the \
             insertion point lands past the end of the empty list and Vec::insert panics)"
        );
    }

    /// Same hook, ghost arm: a cascade that walks over a **rowless** listed id
    /// carrying a children list of its own must drop that id's cache too.
    ///
    /// The ghost arm of `cascade_delete_section` removes `state.children[sid]`
    /// and `continue`s before any of the live-row bookkeeping, so it needs its
    /// own invalidation. Reaching it needs a ghost that is *also* a parent
    /// whose cache a command populated first. This engine keeps no list under
    /// an id with no row (SDD §10.4), so the ghost's list is one an earlier
    /// build stored, built here by hand: `X` and `Y`, rows naming the ghost.
    /// Moving `Y` out builds the ghost's cache through
    /// `remove_child_from_parent` and leaves it holding `X`.
    ///
    /// After the cascade, a `CreateSection` gives the ghost id a row, so the
    /// last command re-materializes its list (empty — the rows that named it
    /// went with the cascade, so it adopts nothing) and asks for an insertion
    /// point.
    ///
    /// Discriminated by: deleting the `plan.invalidate_cache(sid)` on
    /// `cascade_delete_section`'s ghost arm — same panic as the live-row case.
    #[test]
    fn apply_delta_cascaded_ghost_parents_re_materialized_list_uses_a_fresh_cache() {
        const P: u64 = 1;
        const GHOST: u64 = 2;
        const X: u64 = 6;
        const Y: u64 = 7;
        const SECOND: u64 = 9;

        let mut state = make_state();
        seed_section(&mut state, P, 0, "P", 0);
        seed_section(&mut state, X, GHOST, "X", 0);
        seed_section(&mut state, Y, GHOST, "Y", 0);
        state.children.insert(0, vec![P]);
        // GHOST is listed under P with no `state.sections` row of its own,
        // and carries the list an earlier build stored under it.
        state.children.insert(P, vec![GHOST]);
        state.children.insert(GHOST, vec![X, Y]);

        apply_counted(
            &mut state,
            batch(vec![
                // Builds GHOST's cache over [X, Y] and takes Y out, leaving
                // one entry at order 0.
                move_cmd(Y, GHOST, P, "Y"),
                // The cascade walks P → {GHOST, Y}, GHOST → X; GHOST takes the
                // ghost arm, which removes its list without any of the
                // live-row bookkeeping.
                delete_cmd(P),
                // GHOST gets a row of its own.
                create_cmd(0, GHOST, "G"),
                // Re-materializes GHOST's list empty and asks for an
                // insertion point.
                change_cmd(GHOST, SECOND, "S", 0),
            ]),
        );

        assert_eq!(
            state.children[&GHOST],
            vec![SECOND],
            "the ghost parent's re-materialized list holds only the new row (with the \
             stale cache, Vec::insert panics past the end of the empty list)"
        );
    }

    /// The **destination** parent of a cross-parent order change must not be
    /// the kept cache.
    ///
    /// `invalidate_order_listeners`'s `keep` spares exactly one parent's cache,
    /// and it is sound only because the statement following the call repairs
    /// the one entry the write falsified in *that* parent. The parent it names
    /// is the row's **previous** one: on the same-parent arm
    /// `rebuild_children_list` re-places the child there, and on the
    /// cross-parent arm `remove_child_from_parent` deletes the entry outright.
    /// The **destination** has no such repair — `insert_child_into_parent`
    /// finds the child already listed there and returns before refreshing
    /// anything — so keeping it would leave its cache asserting the child's
    /// **old** order for the rest of the delta. Keeping `old_parent` costs
    /// only the cheap arm and is green; keeping `parent_id` on both arms is
    /// the drift, and was green before this test existed.
    ///
    /// The arrangement starts cross-map INCONSISTENT — section C's row names
    /// parent A while parent D is the one listing it, a stale listing (see
    /// `insert_child_into_parent`) — and the delta's cross-parent order change
    /// leaves it consistent. As with the three cache tests above, that rules
    /// out `apply_one_command_per_delta` as an oracle (the replay would trip
    /// the cross-map debug assert mid-window), so the expected order is
    /// spelled out with the drifted answer named.
    ///
    /// Both insertions are explicit order-0 rows. A `CreateSection` would not
    /// discriminate: it appends at the largest key + 1, and a kept cache
    /// (`[50, 51]`) and a fresh one (`[0, 51]`) share the largest key, so both
    /// give the same append.
    ///
    /// Final `children[D]`: **`[FIRST, C, SECOND]`** with the fix,
    /// `[FIRST, SECOND, C]` if the destination's cache were kept (it still
    /// reads C at order 50, so the new order-0 child sorts ahead of it).
    ///
    /// Discriminated by: replacing `Some(old_parent)` with `Some(parent_id)`
    /// in `apply_diff_change_section`'s existing-row branch.
    #[test]
    fn apply_delta_cross_parent_order_change_does_not_keep_the_destination_parents_cache() {
        const D: u64 = 1; // destination parent — the one listing C
        const A: u64 = 2; // the parent C's row actually names
        const C: u64 = 3;
        const FIRST: u64 = 9;
        const SECOND: u64 = 10;

        let mut state = make_state();
        seed_section(&mut state, D, 0, "D", 0);
        seed_section(&mut state, A, 0, "A", 0);
        seed_section(&mut state, C, A, "C", 50);
        state.children.insert(0, vec![D, A]);
        state.children.insert(D, vec![C]);
        state.children.insert(A, Vec::new());

        apply_counted(
            &mut state,
            batch(vec![
                // Populates D's cache while C still reads order 50.
                change_cmd(D, FIRST, "F", 0),
                // Cross-parent order change: C's row moves A → D and its order
                // drops 50 → 0. The insert into D returns early (C is already
                // listed there), so nothing but the order-side invalidation
                // can refresh D's cache.
                change_cmd(D, C, "C", 0),
                // Reads D's cache again.
                change_cmd(D, SECOND, "S", 0),
            ]),
        );

        assert_eq!(
            state.children[&D],
            vec![FIRST, C, SECOND],
            "C's new order-0 must be visible to the next insertion point; keeping the \
             destination's cache across the cross-parent arm holds the stale order-50 \
             and gives [FIRST, SECOND, C]"
        );
    }

    /// A cascade that walks over a ghost id must not purge that id's field and
    /// time-sample rows.
    ///
    /// Ghost ids are deliberately kept out of `plan.purge_ids`: there is no
    /// section row to delete, and pre-b.ot3 the per-section `retain` never
    /// reached them either, so purging them now would be a silent behaviour
    /// change on exactly the legacy state the ghost arm exists for. The shape
    /// is reachable because `apply_set_field` requires no `state.sections`
    /// row, so a field on a listed-but-rowless id is ordinary stored state.
    ///
    /// The live parent's own field is asserted purged in the same run, so the
    /// test cannot pass by the purge pass simply not running.
    ///
    /// Discriminated by: adding `plan.purge_ids.insert(sid)` to
    /// `cascade_delete_section`'s ghost arm.
    #[test]
    fn apply_delta_cascade_over_a_ghost_id_leaves_its_field_rows_in_place() {
        const P: u64 = 1;
        const GHOST: u64 = 77;

        let mut state = make_state();
        seed_section(&mut state, P, 0, "P", 0);
        state.children.insert(0, vec![P]);
        state.children.insert(P, vec![GHOST]);
        seed_field(&mut state, P, "live");
        seed_field(&mut state, GHOST, "ghost");
        seed_time_sample(&mut state, GHOST, 1.0);

        let (mutations, stats) = apply_counted(&mut state, batch(vec![delete_cmd(P)]));

        assert_eq!(
            stats.field_purge_passes, 1,
            "the deferred purge pass ran — the ghost's survival below is not vacuous"
        );
        assert!(
            !state.fields.contains_key(&(P, "live".to_string())),
            "the cascaded row's own field is purged"
        );
        assert!(
            mutations.iter().any(
                |m| matches!(m, StorageMutation::DeleteField { section_id, key_name }
                    if *section_id == P && key_name == "live")
            ),
            "and its erasure is persisted"
        );

        assert!(
            !mutations.iter().any(
                |m| matches!(m, StorageMutation::DeleteField { section_id, .. }
                    if *section_id == GHOST)
            ),
            "no DeleteField may be emitted for a ghost id"
        );
        assert!(
            !mutations.iter().any(
                |m| matches!(m, StorageMutation::DeleteTimeSample { section_id, .. }
                    if *section_id == GHOST)
            ),
            "and no DeleteTimeSample either"
        );
        assert!(
            state.fields.contains_key(&(GHOST, "ghost".to_string())),
            "the ghost's field row survives the cascade, as it did pre-b.ot3"
        );
        assert!(
            state.time_samples.contains_key(&(GHOST, OrderedFloat(1.0))),
            "and so does its time sample"
        );
    }

    // ═════════════════════════════════════════════════════════════════════════
    // Section pre-images on `SetSection` / `DeleteSection`
    //
    // `SetSection::previous` and `DeleteSection`'s `(parent_id, name)` name the
    // row each mutation replaces or removes, as it stood immediately before
    // that mutation (see the `StorageMutation` doc-comment). That in-order
    // rule is the invariant these tests pin, so every assertion below spells
    // out the expected `SectionKey` / `(parent_id, name)` literally.
    //
    // Each shape is a fixture function returning `(pre-delta state, delta)`.
    // The replay test at the end of the section reads every fixture and checks
    // the index invariant itself; most fixtures also have a test of their own
    // pinning literal values, and the four multi-command chains marked "Only
    // the replay test reads it" have none.
    //
    // Several move fixtures start from a *misfiled* row (its `parent_id` names
    // one parent while a different parent lists it) so that the client's
    // `old_parent_id` — which names the listing parent — differs from the
    // stored `parent_id`. That difference is what tells "the stored key" apart
    // from "the key the command claims".
    // ═════════════════════════════════════════════════════════════════════════

    /// Every `SetSection` in emission order, as `(section, previous)`.
    fn set_section_writes(mutations: &[StorageMutation]) -> Vec<(Section, Option<SectionKey>)> {
        mutations
            .iter()
            .filter_map(|m| match m {
                StorageMutation::SetSection { section, previous } => {
                    Some((section.clone(), previous.clone()))
                }
                _ => None,
            })
            .collect()
    }

    /// Every `SetSection` in emission order, as `(section_id, previous)`.
    fn set_section_previous(mutations: &[StorageMutation]) -> Vec<(u64, Option<SectionKey>)> {
        set_section_writes(mutations)
            .into_iter()
            .map(|(section, previous)| (section.section_id, previous))
            .collect()
    }

    /// A `DeleteSection`'s `(section_id, parent_id, name)`.
    type DeletedKey = (u64, u64, String);

    /// Every `DeleteSection` in emission order.
    fn delete_section_writes(mutations: &[StorageMutation]) -> Vec<DeletedKey> {
        mutations
            .iter()
            .filter_map(|m| match m {
                StorageMutation::DeleteSection {
                    section_id,
                    parent_id,
                    name,
                } => Some((*section_id, *parent_id, name.clone())),
                _ => None,
            })
            .collect()
    }

    fn key(parent_id: u64, name: &str) -> SectionKey {
        SectionKey {
            parent_id,
            name: name.to_string(),
        }
    }

    fn deleted(section_id: u64, parent_id: u64, name: &str) -> DeletedKey {
        (section_id, parent_id, name.to_string())
    }

    // ── Pre-image fixtures ───────────────────────────────────────────────────

    type PreImageFixture = fn() -> (ObjectState, OwnedDelta);

    /// `P1` (1) and `P2` (2) under the root, and `X` (5) under `P1`, listed
    /// consistently.
    fn base_x_under_p1() -> ObjectState {
        let mut state = make_state();
        seed_section(&mut state, 1, 0, "P1", 0);
        seed_section(&mut state, 2, 0, "P2", 0);
        seed_section(&mut state, 5, 1, "X", 0);
        state.children.insert(0, vec![1, 2]);
        state.children.insert(1, vec![5]);
        state.children.insert(2, Vec::new());
        state
    }

    /// `P1` (1), `P2` (2) and `P3` (3) under the root, and `X` (5) whose row
    /// names `P1` while only `P3` lists it. A move command naming
    /// `old_parent_id = 3` is the one that cleans the listing up, so each
    /// delta below ends cross-map consistent — and that command's
    /// `old_parent_id` (3) differs from the stored `parent_id` (1).
    fn base_x_misfiled_under_p3() -> ObjectState {
        let mut state = make_state();
        seed_section(&mut state, 1, 0, "P1", 0);
        seed_section(&mut state, 2, 0, "P2", 0);
        seed_section(&mut state, 3, 0, "P3", 0);
        seed_section(&mut state, 5, 1, "X", 0);
        state.children.insert(0, vec![1, 2, 3]);
        state.children.insert(1, Vec::new());
        state.children.insert(2, Vec::new());
        state.children.insert(3, vec![5]);
        state
    }

    /// `A` (1) under the root with one child `C` (2): the conflict target of
    /// the two name-conflict fixtures.
    fn base_a_with_child() -> ObjectState {
        let mut state = make_state();
        seed_section(&mut state, 1, 0, "A", 0);
        seed_section(&mut state, 2, 1, "C", 0);
        state.children.insert(0, vec![1]);
        state.children.insert(1, vec![2]);
        state
    }

    fn fixture_create_plain() -> (ObjectState, OwnedDelta) {
        (make_state(), batch(vec![create_cmd(0, 1, "A")]))
    }

    fn fixture_create_name_conflict() -> (ObjectState, OwnedDelta) {
        (base_a_with_child(), batch(vec![create_cmd(0, 3, "A")]))
    }

    fn fixture_change_rename() -> (ObjectState, OwnedDelta) {
        (base_x_under_p1(), batch(vec![change_cmd(1, 5, "Y", 0)]))
    }

    fn fixture_change_reparent() -> (ObjectState, OwnedDelta) {
        (base_x_under_p1(), batch(vec![change_cmd(2, 5, "X", 0)]))
    }

    fn fixture_change_rename_and_reparent() -> (ObjectState, OwnedDelta) {
        (base_x_under_p1(), batch(vec![change_cmd(2, 5, "Y", 0)]))
    }

    /// Same `(parent_id, name)`; only the order changes.
    fn fixture_change_unchanged_key() -> (ObjectState, OwnedDelta) {
        (base_x_under_p1(), batch(vec![change_cmd(1, 5, "X", 7)]))
    }

    fn fixture_change_new_id() -> (ObjectState, OwnedDelta) {
        (base_x_under_p1(), batch(vec![change_cmd(2, 9, "N", 0)]))
    }

    /// Id 77 is listed under `P` with no row of its own.
    fn fixture_change_ghost_id() -> (ObjectState, OwnedDelta) {
        let mut state = make_state();
        seed_section(&mut state, 1, 0, "P", 0);
        state.children.insert(0, vec![1]);
        state.children.insert(1, vec![77]);
        (state, batch(vec![change_cmd(1, 77, "G", 3)]))
    }

    fn fixture_change_new_row_name_conflict() -> (ObjectState, OwnedDelta) {
        (base_a_with_child(), batch(vec![change_cmd(0, 9, "A", 0)]))
    }

    /// A three-deep chain `A` (1) → `B` (2) → `C` (3).
    fn fixture_delete_subtree() -> (ObjectState, OwnedDelta) {
        let mut state = make_state();
        seed_section(&mut state, 1, 0, "A", 0);
        seed_section(&mut state, 2, 1, "B", 0);
        seed_section(&mut state, 3, 2, "C", 0);
        state.children.insert(0, vec![1]);
        state.children.insert(1, vec![2]);
        state.children.insert(2, vec![3]);
        (state, batch(vec![delete_cmd(1)]))
    }

    /// `O` (7) names parent 99, which has no row; `C` (8) is its child.
    fn fixture_delete_orphan() -> (ObjectState, OwnedDelta) {
        let mut state = make_state();
        seed_section(&mut state, 7, 99, "O", 0);
        seed_section(&mut state, 8, 7, "C", 0);
        state.children.insert(7, vec![8]);
        (state, batch(vec![delete_cmd(7)]))
    }

    /// `X` (5) names `P2` (2) but is listed only under `P1` (1), so deleting
    /// `P1` reaches it through a list it does not name.
    fn fixture_delete_cross_listed() -> (ObjectState, OwnedDelta) {
        let mut state = make_state();
        seed_section(&mut state, 1, 0, "P1", 0);
        seed_section(&mut state, 2, 0, "P2", 0);
        seed_section(&mut state, 5, 2, "X", 0);
        state.children.insert(0, vec![1, 2]);
        state.children.insert(1, vec![5]);
        state.children.insert(2, Vec::new());
        (state, batch(vec![delete_cmd(1)]))
    }

    /// Edge 1: `old_parent_id == new_parent_id == 2`, but the row's stored
    /// parent is 1.
    fn fixture_move_edge1_rename_only() -> (ObjectState, OwnedDelta) {
        (base_x_under_p1(), batch(vec![move_cmd(5, 2, 2, "Y")]))
    }

    /// Edge 2: the new parent (888) does not exist.
    fn fixture_move_edge2_missing_new_parent() -> (ObjectState, OwnedDelta) {
        (
            base_x_misfiled_under_p3(),
            batch(vec![move_cmd(5, 3, 888, "X")]),
        )
    }

    /// Edge 3: the new parent `D` (6) is a descendant of the moved row.
    fn fixture_move_edge3_cycle() -> (ObjectState, OwnedDelta) {
        let mut state = base_x_misfiled_under_p3();
        seed_section(&mut state, 6, 5, "D", 0);
        state.children.insert(5, vec![6]);
        (state, batch(vec![move_cmd(5, 3, 6, "Z")]))
    }

    /// Edge 4: the destination name conflicts with `P1`, the moved row's own
    /// (stored) ancestor.
    fn fixture_move_edge4_ancestor_conflict() -> (ObjectState, OwnedDelta) {
        (
            base_x_misfiled_under_p3(),
            batch(vec![move_cmd(5, 3, 0, "P1")]),
        )
    }

    /// Edge 5: the destination name conflicts with `Y` (6) under `P2`, which
    /// is not an ancestor and has a child `Z` (7).
    fn fixture_move_edge5_non_ancestor_conflict() -> (ObjectState, OwnedDelta) {
        let mut state = base_x_misfiled_under_p3();
        seed_section(&mut state, 6, 2, "Y", 0);
        seed_section(&mut state, 7, 6, "Z", 0);
        state.children.insert(2, vec![6]);
        state.children.insert(6, vec![7]);
        (state, batch(vec![move_cmd(5, 3, 2, "Y")]))
    }

    /// Edge 6: a plain move to `P2` under a new name.
    fn fixture_move_edge6_normal() -> (ObjectState, OwnedDelta) {
        (
            base_x_misfiled_under_p3(),
            batch(vec![move_cmd(5, 3, 2, "W")]),
        )
    }

    /// `A` (1) → `B` (2), plus an orphan `O` (7) under the rowless 99.
    fn base_for_reset() -> ObjectState {
        let mut state = make_state();
        seed_section(&mut state, 1, 0, "A", 0);
        seed_section(&mut state, 2, 1, "B", 0);
        seed_section(&mut state, 7, 99, "O", 0);
        state.children.insert(0, vec![1]);
        state.children.insert(1, vec![2]);
        state
    }

    fn reset_delta(cmds: Vec<StructCommand>) -> OwnedDelta {
        OwnedDelta {
            is_diff: true,
            base_version: 0,
            struct_commands: cmds,
            ..Default::default()
        }
    }

    fn fixture_reset() -> (ObjectState, OwnedDelta) {
        (base_for_reset(), reset_delta(Vec::new()))
    }

    /// A reset followed by a create that reuses a drained id.
    fn fixture_reset_then_recreate() -> (ObjectState, OwnedDelta) {
        (base_for_reset(), reset_delta(vec![create_cmd(0, 1, "A")]))
    }

    fn fixture_create_then_rename() -> (ObjectState, OwnedDelta) {
        (
            make_state(),
            batch(vec![create_cmd(0, 5, "a"), move_cmd(5, 0, 0, "b")]),
        )
    }

    fn fixture_rename_then_move() -> (ObjectState, OwnedDelta) {
        let mut state = make_state();
        seed_section(&mut state, 1, 0, "P", 0);
        seed_section(&mut state, 5, 0, "a", 0);
        state.children.insert(0, vec![1, 5]);
        state.children.insert(1, Vec::new());
        (
            state,
            batch(vec![move_cmd(5, 0, 0, "b"), move_cmd(5, 0, 1, "c")]),
        )
    }

    fn fixture_delete_then_recreate() -> (ObjectState, OwnedDelta) {
        let mut state = make_state();
        seed_section(&mut state, 5, 0, "a", 0);
        state.children.insert(0, vec![5]);
        (state, batch(vec![delete_cmd(5), create_cmd(0, 5, "a")]))
    }

    /// `2^53 + 1`: the smallest id an `f64` cannot represent exactly.
    const ID_ABOVE_F64_EXACT: u64 = (1u64 << 53) + 1;

    /// `X` (`ID_ABOVE_F64_EXACT`) with an empty name, under a parent whose id
    /// is `u64::MAX`; the delta rewrites only its order, then deletes it.
    fn fixture_extreme_key() -> (ObjectState, OwnedDelta) {
        let mut state = make_state();
        seed_section(&mut state, u64::MAX, 0, "P", 0);
        seed_section(&mut state, ID_ABOVE_F64_EXACT, u64::MAX, "", 0);
        state.children.insert(0, vec![u64::MAX]);
        state.children.insert(u64::MAX, vec![ID_ABOVE_F64_EXACT]);
        (
            state,
            batch(vec![
                change_cmd(u64::MAX, ID_ABOVE_F64_EXACT, "", 4),
                delete_cmd(ID_ABOVE_F64_EXACT),
            ]),
        )
    }

    fn fixture_reorder_children() -> (ObjectState, OwnedDelta) {
        let mut state = make_state();
        seed_section(&mut state, 1, 0, "P", 0);
        seed_section(&mut state, 2, 1, "A", 0);
        seed_section(&mut state, 3, 1, "B", 0);
        state.children.insert(0, vec![1]);
        state.children.insert(1, vec![2, 3]);
        (state, batch(vec![reorder_cmd(1, vec![3, 2])]))
    }

    /// Several shapes chained in one delta, each command acting on a row an
    /// earlier command of the same delta wrote or removed. Only the replay
    /// test reads it.
    fn fixture_mixed_chain() -> (ObjectState, OwnedDelta) {
        (
            base_x_under_p1(),
            batch(vec![
                create_cmd(2, 10, "N"),
                change_cmd(2, 10, "N2", 3),
                // Edge 5 against the row the two commands above wrote.
                move_cmd(5, 1, 2, "N2"),
                // Cascades `P2` and the `X` just moved under it.
                delete_cmd(2),
                create_cmd(0, 2, "P2"),
                change_cmd(0, 5, "X", 0),
                move_cmd(5, 0, 2, "X"),
            ]),
        )
    }

    /// `X` is renamed and then its parent `P1` is deleted: the cascaded delete
    /// must carry the renamed key. Only the replay test reads it.
    fn fixture_rename_then_delete() -> (ObjectState, OwnedDelta) {
        (
            base_x_under_p1(),
            batch(vec![move_cmd(5, 1, 1, "Y"), delete_cmd(1)]),
        )
    }

    /// `X` moves under `P2`, then `P2` is deleted: the cascade reaches `X` with
    /// its in-delta parent 2. Only the replay test reads it.
    fn fixture_move_then_cascade() -> (ObjectState, OwnedDelta) {
        (
            base_x_under_p1(),
            batch(vec![move_cmd(5, 1, 2, "X"), delete_cmd(2)]),
        )
    }

    /// A row created in the delta is then the name conflict of a later
    /// create. Only the replay test reads it.
    fn fixture_create_then_conflict() -> (ObjectState, OwnedDelta) {
        (
            make_state(),
            batch(vec![create_cmd(0, 1, "A"), create_cmd(0, 2, "A")]),
        )
    }

    // ── Per-shape pre-images ─────────────────────────────────────────────────

    #[test]
    fn apply_delta_create_section_new_row_previous_is_none() {
        let (mut state, delta) = fixture_create_plain();

        let (mutations, _) = apply_counted(&mut state, delta);

        assert_eq!(set_section_previous(&mutations), vec![(1, None)]);
        assert!(delete_section_writes(&mutations).is_empty());
    }

    #[test]
    fn apply_delta_create_section_name_conflict_cascade_carries_the_conflicting_rows_keys() {
        let (mut state, delta) = fixture_create_name_conflict();

        let (mutations, _) = apply_counted(&mut state, delta);

        assert_eq!(
            delete_section_writes(&mutations),
            vec![deleted(1, 0, "A"), deleted(2, 1, "C")],
            "the conflicting sibling and its descendant, each with its own stored key"
        );
        assert_eq!(set_section_previous(&mutations), vec![(3, None)]);
    }

    #[test]
    fn apply_delta_diff_change_section_existing_row_previous_is_the_stored_key() {
        let cases: [(&str, PreImageFixture); 4] = [
            ("rename", fixture_change_rename),
            ("re-parent", fixture_change_reparent),
            ("rename and re-parent", fixture_change_rename_and_reparent),
            ("unchanged key", fixture_change_unchanged_key),
        ];
        for (label, fixture) in cases {
            let (mut state, delta) = fixture();

            let (mutations, _) = apply_counted(&mut state, delta);

            assert_eq!(
                set_section_previous(&mutations),
                vec![(5, Some(key(1, "X")))],
                "{label}: previous is the row's stored key, Some even when unchanged"
            );
            assert!(
                delete_section_writes(&mutations).is_empty(),
                "{label}: nothing is deleted"
            );
        }
    }

    #[test]
    fn apply_delta_diff_change_section_new_or_ghost_id_previous_is_none() {
        let cases: [(&str, PreImageFixture, u64); 2] = [
            ("new id", fixture_change_new_id, 9),
            ("ghost id", fixture_change_ghost_id, 77),
        ];
        for (label, fixture, id) in cases {
            let (mut state, delta) = fixture();

            let (mutations, _) = apply_counted(&mut state, delta);

            assert_eq!(
                set_section_previous(&mutations),
                vec![(id, None)],
                "{label}: no row with this id existed"
            );
        }
    }

    #[test]
    fn apply_delta_diff_change_section_new_row_name_conflict_cascade_carries_the_conflicting_rows_keys(
    ) {
        let (mut state, delta) = fixture_change_new_row_name_conflict();

        let (mutations, _) = apply_counted(&mut state, delta);

        assert_eq!(
            delete_section_writes(&mutations),
            vec![deleted(1, 0, "A"), deleted(2, 1, "C")]
        );
        assert_eq!(set_section_previous(&mutations), vec![(9, None)]);
    }

    #[test]
    fn apply_delta_delete_section_subtree_each_delete_carries_its_rows_stored_key() {
        let (mut state, delta) = fixture_delete_subtree();

        let (mutations, _) = apply_counted(&mut state, delta);

        assert_eq!(
            delete_section_writes(&mutations),
            vec![deleted(1, 0, "A"), deleted(2, 1, "B"), deleted(3, 2, "C")]
        );
    }

    #[test]
    fn apply_delta_delete_section_orphan_carries_its_missing_stored_parent() {
        let (mut state, delta) = fixture_delete_orphan();

        let (mutations, _) = apply_counted(&mut state, delta);

        assert_eq!(
            delete_section_writes(&mutations),
            vec![deleted(7, 99, "O"), deleted(8, 7, "C")],
            "the orphan's parent_id is the rowless 99 it stores"
        );
    }

    #[test]
    fn apply_delta_delete_section_cross_listed_row_carries_its_stored_parent_not_the_listing_parent(
    ) {
        let (mut state, delta) = fixture_delete_cross_listed();

        let (mutations, _) = apply_counted(&mut state, delta);

        assert_eq!(
            delete_section_writes(&mutations),
            vec![deleted(1, 0, "P1"), deleted(5, 2, "X")],
            "X was reached through P1's list but stores parent 2"
        );
    }

    #[test]
    fn apply_delta_move_section_rename_only_previous_is_the_stored_key_not_old_parent_id() {
        let (mut state, delta) = fixture_move_edge1_rename_only();

        let (mutations, _) = apply_counted(&mut state, delta);

        let writes = set_section_writes(&mutations);
        assert_eq!(writes.len(), 1);
        let (section, previous) = &writes[0];
        assert_eq!(
            previous,
            &Some(key(1, "X")),
            "stored parent 1, not the command's old_parent_id 2"
        );
        assert_eq!(
            (section.parent_id, section.name.as_str()),
            (1, "Y"),
            "a rename keeps the stored parent"
        );
    }

    #[test]
    fn apply_delta_move_section_deleting_edges_carry_the_stored_key_not_old_parent_id() {
        let cases: [(&str, PreImageFixture, Vec<DeletedKey>); 3] = [
            (
                "edge 2 (missing new parent)",
                fixture_move_edge2_missing_new_parent,
                vec![deleted(5, 1, "X")],
            ),
            (
                "edge 3 (cycle)",
                fixture_move_edge3_cycle,
                vec![deleted(5, 1, "X"), deleted(6, 5, "D")],
            ),
            (
                "edge 4 (ancestor conflict)",
                fixture_move_edge4_ancestor_conflict,
                vec![deleted(5, 1, "X")],
            ),
        ];
        for (label, fixture, expected) in cases {
            let (mut state, delta) = fixture();

            let (mutations, _) = apply_counted(&mut state, delta);

            assert_eq!(
                delete_section_writes(&mutations),
                expected,
                "{label}: the moved row carries stored parent 1, not old_parent_id 3"
            );
            assert!(
                set_section_writes(&mutations).is_empty(),
                "{label}: the moved row is deleted, not written"
            );
        }
    }

    #[test]
    fn apply_delta_move_section_non_ancestor_conflict_deletes_carry_their_keys_then_previous_is_the_stored_key(
    ) {
        let (mut state, delta) = fixture_move_edge5_non_ancestor_conflict();

        let (mutations, _) = apply_counted(&mut state, delta);

        assert_eq!(
            delete_section_writes(&mutations),
            vec![deleted(6, 2, "Y"), deleted(7, 6, "Z")],
            "the conflicting subtree, each row with its stored key"
        );
        let writes = set_section_writes(&mutations);
        assert_eq!(writes.len(), 1);
        let (section, previous) = &writes[0];
        assert_eq!(
            previous,
            &Some(key(1, "X")),
            "stored parent 1, not the command's old_parent_id 3"
        );
        assert_eq!((section.parent_id, section.name.as_str()), (2, "Y"));
    }

    #[test]
    fn apply_delta_move_section_normal_move_previous_is_the_stored_key_not_old_parent_id() {
        let (mut state, delta) = fixture_move_edge6_normal();

        let (mutations, _) = apply_counted(&mut state, delta);

        let writes = set_section_writes(&mutations);
        assert_eq!(writes.len(), 1);
        let (section, previous) = &writes[0];
        assert_eq!(
            previous,
            &Some(key(1, "X")),
            "stored parent 1, not the command's old_parent_id 3"
        );
        assert_eq!((section.parent_id, section.name.as_str()), (2, "W"));
        assert!(delete_section_writes(&mutations).is_empty());
    }

    #[test]
    fn apply_delta_reset_emits_one_delete_per_drained_row_with_its_key() {
        let (mut state, delta) = fixture_reset();

        let (mutations, _) = apply_counted(&mut state, delta);

        // The drain walks a HashMap, so emission order is unspecified here.
        let mut deletes = delete_section_writes(&mutations);
        deletes.sort();
        assert_eq!(
            deletes,
            vec![deleted(1, 0, "A"), deleted(2, 1, "B"), deleted(7, 99, "O")]
        );
        assert!(set_section_writes(&mutations).is_empty());
    }

    #[test]
    fn apply_delta_create_then_rename_second_previous_is_the_key_the_first_wrote() {
        let (mut state, delta) = fixture_create_then_rename();

        let (mutations, _) = apply_counted(&mut state, delta);

        assert_eq!(
            set_section_previous(&mutations),
            vec![(5, None), (5, Some(key(0, "a")))]
        );
    }

    #[test]
    fn apply_delta_rename_then_move_second_previous_is_the_key_the_first_wrote() {
        let (mut state, delta) = fixture_rename_then_move();

        let (mutations, _) = apply_counted(&mut state, delta);

        assert_eq!(
            set_section_previous(&mutations),
            vec![(5, Some(key(0, "a"))), (5, Some(key(0, "b")))],
            "the move replaces the renamed row (0, \"b\"), not the pre-delta (0, \"a\")"
        );
    }

    #[test]
    fn apply_delta_delete_then_recreate_emits_delete_then_set_with_previous_none() {
        let (mut state, delta) = fixture_delete_then_recreate();

        let (mutations, _) = apply_counted(&mut state, delta);

        let pre_images: Vec<_> = mutations
            .iter()
            .filter(|m| {
                matches!(
                    m,
                    StorageMutation::SetSection { .. } | StorageMutation::DeleteSection { .. }
                )
            })
            .cloned()
            .collect();
        assert_eq!(
            pre_images,
            vec![
                StorageMutation::DeleteSection {
                    section_id: 5,
                    parent_id: 0,
                    name: "a".to_string(),
                },
                StorageMutation::SetSection {
                    section: state.sections[&5].clone(),
                    previous: None,
                },
            ],
            "the recreate follows the delete that removed the row, so it has no pre-image"
        );
    }

    #[test]
    fn apply_delta_section_key_empty_name_and_ids_above_2_pow_53_survive_unchanged() {
        let (mut state, delta) = fixture_extreme_key();

        let (mutations, _) = apply_counted(&mut state, delta);

        assert_eq!(
            set_section_previous(&mutations),
            vec![(ID_ABOVE_F64_EXACT, Some(key(u64::MAX, "")))]
        );
        assert_eq!(
            delete_section_writes(&mutations),
            vec![deleted(ID_ABOVE_F64_EXACT, u64::MAX, "")]
        );
    }

    /// A reorder never changes a row's `(parent_id, name)`: the only
    /// `SetSection` it emits is a re-key (`section_order`, `last_version`),
    /// whose `previous` is the row's own unchanged key, and it emits no
    /// `DeleteSection`. The fixture's two children tie on key 0, so the list
    /// is re-keyed `0..n-1` in its new order `[B, A]`: B keeps 0 and is not
    /// rewritten, A moves to 1 and is.
    #[test]
    fn apply_delta_reorder_children_rewrites_a_rekeyed_row_under_its_own_key_and_deletes_none() {
        let (mut state, delta) = fixture_reorder_children();

        let (mutations, _) = apply_counted(&mut state, delta);

        assert_eq!(
            set_section_previous(&mutations),
            vec![(2, Some(key(1, "A")))],
            "only the re-keyed A is rewritten, with its own (parent, name) as the pre-image"
        );
        assert_eq!(
            (
                state.sections[&2].section_order,
                state.sections[&3].section_order
            ),
            (1, 0),
            "fixture: the tie is re-keyed in the new order [B, A]"
        );
        assert!(delete_section_writes(&mutations).is_empty());
    }

    // ── Replay invariant ─────────────────────────────────────────────────────

    use std::collections::BTreeSet;

    /// `(parent_id, name, section_id)` for every section row: the set the
    /// replay below folds the section pre-images onto.
    fn name_index_members(state: &ObjectState) -> BTreeSet<(u64, String, u64)> {
        state
            .sections
            .values()
            .map(|s| (s.parent_id, s.name.clone(), s.section_id))
            .collect()
    }

    /// Apply each `SetSection` / `DeleteSection` to `index` in emission order,
    /// using only what the mutation carries, never the state.
    ///
    /// Strict, so an inexact pre-image fails at the mutation that carries it
    /// rather than only in the final comparison: a `previous` or a delete key
    /// must name a member the index holds for that id, and `previous: None`
    /// must mean the index holds no member for that id.
    fn replay_name_index(
        fixture: &str,
        mut index: BTreeSet<(u64, String, u64)>,
        mutations: &[StorageMutation],
    ) -> BTreeSet<(u64, String, u64)> {
        for (pos, mutation) in mutations.iter().enumerate() {
            match mutation {
                StorageMutation::SetSection { section, previous } => {
                    let id = section.section_id;
                    match previous {
                        Some(k) => assert!(
                            index.remove(&(k.parent_id, k.name.clone(), id)),
                            "{fixture}: mutation {pos}: SetSection {id} names previous {k:?}, \
                             which is not the row the index holds: {index:?}"
                        ),
                        None => assert!(
                            !index.iter().any(|(_, _, sid)| *sid == id),
                            "{fixture}: mutation {pos}: SetSection {id} has previous None, but \
                             the index holds a row for it: {index:?}"
                        ),
                    }
                    index.insert((section.parent_id, section.name.clone(), id));
                }
                StorageMutation::DeleteSection {
                    section_id,
                    parent_id,
                    name,
                } => assert!(
                    index.remove(&(*parent_id, name.clone(), *section_id)),
                    "{fixture}: mutation {pos}: DeleteSection {section_id} names \
                     ({parent_id}, {name:?}), which is not the row the index holds: {index:?}"
                ),
                _ => {}
            }
        }
        index
    }

    /// The in-order rule of the section pre-images, checked by replay:
    /// replaying the emitted `SetSection` / `DeleteSection` list in order
    /// against the pre-delta state's `(parent_id, name, section_id)` set
    /// yields the post-delta state's, for every emitting command shape and
    /// for multi-command deltas whose later commands act on rows the earlier
    /// ones wrote.
    ///
    /// Discriminated by: filling edge 6's `previous` from `cmd.old_parent_id`
    /// (fails at `move edge 5`, the first fixture that reaches edge 6's
    /// `SetSection` emission with a client `old_parent_id` (3) differing from
    /// the stored parent (1) — `move edge 2` to `move edge 4` carry the same
    /// mismatch but cascade-delete and return before edge 6); and filling a
    /// cascade root's `DeleteSection::parent_id` from the caller's
    /// `explicit_parent` (fails at `move edge 2`, for the same reason).
    #[test]
    fn apply_delta_section_pre_images_replayed_in_order_keep_a_name_index_exact() {
        let fixtures: [(&str, PreImageFixture); 29] = [
            ("create", fixture_create_plain),
            ("create name conflict", fixture_create_name_conflict),
            ("change rename", fixture_change_rename),
            ("change re-parent", fixture_change_reparent),
            (
                "change rename and re-parent",
                fixture_change_rename_and_reparent,
            ),
            ("change unchanged key", fixture_change_unchanged_key),
            ("change new id", fixture_change_new_id),
            ("change ghost id", fixture_change_ghost_id),
            (
                "change new-row name conflict",
                fixture_change_new_row_name_conflict,
            ),
            ("delete subtree", fixture_delete_subtree),
            ("delete orphan", fixture_delete_orphan),
            ("delete cross-listed", fixture_delete_cross_listed),
            ("move edge 1", fixture_move_edge1_rename_only),
            ("move edge 2", fixture_move_edge2_missing_new_parent),
            ("move edge 3", fixture_move_edge3_cycle),
            ("move edge 4", fixture_move_edge4_ancestor_conflict),
            ("move edge 5", fixture_move_edge5_non_ancestor_conflict),
            ("move edge 6", fixture_move_edge6_normal),
            ("reset", fixture_reset),
            ("reset then recreate", fixture_reset_then_recreate),
            ("create then rename", fixture_create_then_rename),
            ("rename then move", fixture_rename_then_move),
            ("delete then recreate", fixture_delete_then_recreate),
            ("extreme key", fixture_extreme_key),
            ("reorder children", fixture_reorder_children),
            ("mixed chain", fixture_mixed_chain),
            // The same delete shapes once more after a rename in the same
            // delta, so the delete key must come from the in-delta row.
            ("rename then delete", fixture_rename_then_delete),
            ("move then cascade", fixture_move_then_cascade),
            ("create then conflict", fixture_create_then_conflict),
        ];

        let mut silent = Vec::new();
        for (label, fixture) in fixtures {
            let (mut state, delta) = fixture();
            let pre = name_index_members(&state);

            let (mutations, _) = apply_counted(&mut state, delta);

            let replayed = replay_name_index(label, pre, &mutations);
            assert_eq!(
                replayed,
                name_index_members(&state),
                "{label}: replaying the pre-images must land on the post-delta index"
            );
            if set_section_writes(&mutations).is_empty()
                && delete_section_writes(&mutations).is_empty()
            {
                silent.push(label);
            }
        }
        assert!(
            silent.is_empty(),
            "every fixture emits at least one pre-image, so no replay is vacuous; silent: \
             {silent:?}"
        );
    }

    // ── ApplyMode::Replay: the depth walk is skipped only for a delta that
    //    cannot change the section tree ───────────────────────────────────────

    /// A linear chain `1 → 2 → … → cap + 1` under the root: one level deeper
    /// than `MAX_SECTION_TREE_DEPTH`, so any apply that runs
    /// `check_section_tree_depth` over it — and does not shorten the chain —
    /// fails `SectionTreeTooDeep`. Whether a delta walked is therefore
    /// observable from its result alone.
    fn too_deep_chain_state() -> ObjectState {
        use crate::domain::limits::MAX_SECTION_TREE_DEPTH;

        let mut state = make_state();
        for id in 1..=(MAX_SECTION_TREE_DEPTH as u64 + 1) {
            let parent = id - 1;
            seed_section(&mut state, id, parent, &format!("S{id}"), 0);
            state.children.entry(parent).or_default().push(id);
        }
        state
    }

    fn apply_in_mode(
        state: &mut ObjectState,
        delta: Option<OwnedDelta>,
        mode: ApplyMode,
    ) -> Result<ApplyDeltaStats, DeltaError> {
        let mut stats = ApplyDeltaStats::default();
        apply_delta_into_mode(state, delta, &mut MutationSink::Discard, &mut stats, mode)?;
        Ok(stats)
    }

    fn inline_set_field(section_id: u64, key: &str, set_order: u64) -> OodfSetField {
        OodfSetField {
            section_id,
            key_name: key.to_string(),
            value_or_ext_hash: vec![3; 4],
            ext_value_size: 0,
            ext_value_index: 0,
            set_order,
        }
    }

    fn diff_delete_cmd(section_id: u64) -> StructCommand {
        StructCommand::DiffDeleteSection(oodf_ffi::CommandDiffDeleteSection { section_id })
    }

    /// Every delta replay mode lets skip the walk: an empty commit, and a
    /// delta with no reset whose structural commands, if any, are all the
    /// ignored `DiffDeleteSection`. On a too-deep stored tree each one applies
    /// in `Replay` and is refused in `Commit` — the `Commit` half is what shows
    /// the tree really is too deep, so the `Replay` success is the skip.
    #[test]
    fn apply_delta_into_mode_replay_delta_that_cannot_change_the_tree_skips_the_depth_walk() {
        let cases: Vec<(&str, Option<OwnedDelta>)> = vec![
            ("empty commit (None)", None),
            ("default delta", Some(OwnedDelta::default())),
            (
                "fields only",
                Some(OwnedDelta {
                    set_fields: vec![inline_set_field(1, "k", 1)],
                    ..Default::default()
                }),
            ),
            (
                "time samples only",
                Some(OwnedDelta {
                    time_samples: vec![OodfTimeSample {
                        section_id: 1,
                        time: 0.5,
                        value_or_ext_hash: vec![9; 4],
                        ext_value_size: 0,
                        ext_value_index: 0,
                    }],
                    ..Default::default()
                }),
            ),
            (
                "DiffDeleteSection only, with a field",
                Some(OwnedDelta {
                    struct_commands: vec![diff_delete_cmd(1), diff_delete_cmd(2)],
                    set_fields: vec![inline_set_field(2, "k", 1)],
                    ..Default::default()
                }),
            ),
            (
                // `is_diff` with a non-zero base is an incremental diff, not
                // a reset.
                "incremental diff with a field",
                Some(OwnedDelta {
                    is_diff: true,
                    base_version: 5,
                    set_fields: vec![inline_set_field(3, "k", 1)],
                    ..Default::default()
                }),
            ),
        ];

        for (label, delta) in cases {
            let mut replayed = too_deep_chain_state();
            apply_in_mode(&mut replayed, delta.clone(), ApplyMode::Replay)
                .unwrap_or_else(|e| panic!("{label}: Replay must skip the walk, got {e:?}"));
            assert_eq!(replayed.sequence_num, 1, "{label}: the commit advanced");

            let mut committed = too_deep_chain_state();
            let err = apply_in_mode(&mut committed, delta, ApplyMode::Commit).unwrap_err();
            assert!(
                matches!(err, DeltaError::SectionTreeTooDeep { .. }),
                "{label}: Commit mode must still walk and refuse the tree, got {err:?}"
            );
        }
    }

    /// Any structural command other than `DiffDeleteSection` — alone, or
    /// beside one — makes a `Replay` apply walk the tree, exactly as `Commit`
    /// does. No case shortens the chain, so each walk fails.
    #[test]
    fn apply_delta_into_mode_replay_structural_delta_still_walks_the_depth_check() {
        use crate::domain::limits::MAX_SECTION_TREE_DEPTH;

        let leaf = MAX_SECTION_TREE_DEPTH as u64 + 1;
        let cases: Vec<(&str, OwnedDelta)> = vec![
            ("CreateSection", batch(vec![create_cmd(0, 900_000, "New")])),
            (
                "DiffChangeSection",
                batch(vec![change_cmd(0, 900_000, "New", 0)]),
            ),
            (
                "MoveSection (rename only)",
                batch(vec![move_cmd(leaf, leaf - 1, leaf - 1, "Renamed")]),
            ),
            ("ReorderChildren", batch(vec![reorder_cmd(0, vec![1])])),
            (
                "DeleteSection of an absent id",
                batch(vec![delete_cmd(900_000)]),
            ),
            (
                "DiffDeleteSection beside a CreateSection",
                batch(vec![diff_delete_cmd(1), create_cmd(0, 900_000, "New")]),
            ),
        ];

        for (label, delta) in cases {
            let mut state = too_deep_chain_state();
            let err = apply_in_mode(&mut state, Some(delta), ApplyMode::Replay).unwrap_err();
            assert!(
                matches!(err, DeltaError::SectionTreeTooDeep { .. }),
                "{label}: a structural delta must walk in Replay mode, got {err:?}"
            );
            assert_eq!(state.sequence_num, 0, "{label}: the commit did not advance");
        }
    }

    /// A reset is outside the skip class. A reset clears the whole tree, so
    /// the walk can only fail on one whose own commands rebuild a too-deep
    /// tree; that delta must be refused in `Replay` exactly as in `Commit`.
    ///
    /// The rebuild needs structural commands, which walk on their own, so this
    /// pins the outcome ("replaying a reset never admits a too-deep tree")
    /// rather than discriminating the reset clause of the skip predicate: a
    /// reset with no structural command ends on an empty tree, where the walk
    /// cannot fail, so that clause has no observable effect.
    #[test]
    fn apply_delta_into_mode_replay_reset_rebuilding_a_too_deep_tree_is_refused() {
        use crate::domain::limits::MAX_SECTION_TREE_DEPTH;

        let rebuild: Vec<StructCommand> = (1..=(MAX_SECTION_TREE_DEPTH as u64 + 1))
            .map(|id| create_cmd(id - 1, id, &format!("S{id}")))
            .collect();
        let reset = OwnedDelta {
            is_diff: true,
            base_version: 0,
            struct_commands: rebuild,
            ..Default::default()
        };

        for mode in [ApplyMode::Replay, ApplyMode::Commit] {
            let mut state = make_state();
            let err = apply_in_mode(&mut state, Some(reset.clone()), mode).unwrap_err();
            assert!(
                matches!(err, DeltaError::SectionTreeTooDeep { .. }),
                "{mode:?}: the rebuilt chain is one level too deep, got {err:?}"
            );
        }
    }

    /// SR-4.1's engine-level pin: on an ordinary tree a delta that skips the
    /// walk in `Replay` leaves exactly the state and counters `Commit`, which
    /// walks, leaves — the walk is a check, never a mutation.
    #[test]
    fn apply_delta_into_mode_replay_and_commit_leave_identical_state_for_a_field_only_delta() {
        let delta = OwnedDelta {
            is_diff: true,
            base_version: 2,
            struct_commands: vec![diff_delete_cmd(2)],
            set_fields: vec![
                inline_set_field(2, "a", 1),
                OodfSetField {
                    section_id: 3,
                    key_name: "ext".to_string(),
                    value_or_ext_hash: vec![0xab; 32],
                    ext_value_size: 10,
                    ext_value_index: 1,
                    set_order: 2,
                },
            ],
            time_samples: vec![OodfTimeSample {
                section_id: 1,
                time: 2.5,
                value_or_ext_hash: vec![5; 2],
                ext_value_size: 0,
                ext_value_index: 0,
            }],
            ..Default::default()
        };

        let mut replayed = make_three_level_tree();
        let replay_stats = apply_in_mode(&mut replayed, Some(delta.clone()), ApplyMode::Replay)
            .expect("Replay apply");
        let mut committed = make_three_level_tree();
        let commit_stats =
            apply_in_mode(&mut committed, Some(delta), ApplyMode::Commit).expect("Commit apply");

        assert_eq!(replayed, committed);
        assert_eq!(replay_stats, commit_stats);
        assert_eq!(
            (replayed.fields.len(), replayed.time_samples.len()),
            (2, 1),
            "non-vacuity: the delta's rows landed"
        );
    }

    // ── ApplyMode::RestoreBase: the per-delta entry cap is skipped, and
    //    nothing else ─────────────────────────────────────────────────────────
    //
    // A checkpoint base describes a whole layer, and a chunked base decodes
    // into one reset delta that may carry more entries than one delta may.
    // Every fixture below carries `MAX_DELTA_ENTRIES + 1` entries, so a
    // `RestoreBase` apply that ran the entry cap would refuse it with
    // `TooManyEntries` instead of the outcome under test.

    /// A whole-layer reset delta (`is_diff`, `base_version = 0`): `commands`,
    /// then inline set fields `k000000`, `k000001`, … on section 1 with
    /// authoring rank `i % 3`, enough to bring the total to
    /// `MAX_DELTA_ENTRIES + 1`.
    fn over_cap_reset(commands: Vec<StructCommand>) -> OwnedDelta {
        use crate::domain::limits::MAX_DELTA_ENTRIES;

        let n_fields = MAX_DELTA_ENTRIES as usize + 1 - commands.len();
        let delta = OwnedDelta {
            is_diff: true,
            base_version: 0,
            source_format: "usda".to_string(),
            struct_commands: commands,
            set_fields: (0..n_fields)
                .map(|i| inline_set_field(1, &format!("k{i:06}"), (i % 3) as u64))
                .collect(),
            ..Default::default()
        };
        assert_eq!(
            own_element_count(&delta),
            MAX_DELTA_ENTRIES + 1,
            "the fixture is one entry over the cap"
        );
        delta
    }

    /// The same over-cap whole-layer reset: `Commit` and `Replay` refuse it
    /// for its entry count before mutating anything, exactly as before
    /// `RestoreBase` existed; `RestoreBase` applies all of it, as one reset —
    /// one version, one cursor over every field.
    #[test]
    fn apply_delta_into_mode_restore_base_applies_an_over_cap_reset_that_commit_and_replay_refuse()
    {
        use crate::domain::limits::MAX_DELTA_ENTRIES;

        let delta = over_cap_reset(vec![change_cmd(0, 1, "World", 0)]);

        for mode in [ApplyMode::Commit, ApplyMode::Replay] {
            let mut state = make_state();
            let err = apply_in_mode(&mut state, Some(delta.clone()), mode).unwrap_err();
            assert!(
                matches!(
                    err,
                    DeltaError::TooManyEntries {
                        struct_commands: 1,
                        set_fields,
                        time_samples: 0,
                        max_entries,
                    } if set_fields == MAX_DELTA_ENTRIES && max_entries == MAX_DELTA_ENTRIES
                ),
                "{mode:?} must refuse the over-cap delta for its entry count; got {err:?}"
            );
            assert_eq!(
                state,
                make_state(),
                "{mode:?}: the entry cap is checked before any mutation"
            );
        }

        let expected_cursor: u64 = delta.set_fields.iter().map(|f| f.set_order + 1).sum();
        let mut state = make_state();
        apply_in_mode(&mut state, Some(delta), ApplyMode::RestoreBase)
            .expect("RestoreBase must apply a whole-layer reset over the per-delta cap");

        assert_eq!(state.sections.len(), 1, "the section landed");
        assert_eq!(
            state.fields.len(),
            MAX_DELTA_ENTRIES as usize,
            "every set field landed"
        );
        assert_eq!(
            state.children.get(&0).map(Vec::as_slice),
            Some([1u64].as_slice()),
            "the section is listed under the root"
        );
        assert_eq!(
            state.sequence_num, 1,
            "a reset onto a fresh state stamps version 1, as in Commit"
        );
        assert!(
            state.fields.values().all(|f| f.version == 1),
            "every field carries the one reset's version"
        );
        assert_eq!(
            state.next_set_order, expected_cursor,
            "the set_order cursor accumulated over every field"
        );
    }

    /// Over the cap, `RestoreBase` still runs the per-item content rules and
    /// the section-tree depth walk: each defect is refused with its own
    /// content or depth error, never applied and never reported as
    /// `TooManyEntries`.
    #[test]
    fn apply_delta_into_mode_restore_base_still_refuses_content_and_depth_defects_over_the_cap() {
        use crate::domain::limits::MAX_SECTION_TREE_DEPTH;

        let depth = MAX_SECTION_TREE_DEPTH + 1;
        let mut bad_key = over_cap_reset(vec![change_cmd(0, 1, "World", 0)]);
        bad_key
            .set_fields
            .last_mut()
            .expect("the fixture carries fields")
            .key_name = "bad\u{1}key".to_string();

        /// A label, the over-cap delta, and whether an error is the one owed.
        type Case = (&'static str, OwnedDelta, fn(&DeltaError) -> bool);
        let cases: Vec<Case> = vec![
            (
                "control character in a section name",
                over_cap_reset(vec![change_cmd(0, 1, "bad\u{1}name", 0)]),
                |e| {
                    matches!(e, DeltaError::NameContainsControlChar { field }
                        if *field == "DiffChangeSection.section_name")
                },
            ),
            ("control character in a field key", bad_key, |e| {
                matches!(e, DeltaError::NameContainsControlChar { field }
                        if *field == "SetField.key_name")
            }),
            (
                "self-parented section",
                over_cap_reset(vec![change_cmd(0, 1, "World", 0), create_cmd(2, 2, "Loop")]),
                |e| {
                    matches!(
                        e,
                        DeltaError::SelfParenting {
                            command: "CreateSection",
                            section_id: 2,
                        }
                    )
                },
            ),
            (
                "section tree one level over the depth cap",
                over_cap_reset(
                    (1..=depth as u64)
                        .map(|id| change_cmd(id - 1, id, &format!("s{id}"), 0))
                        .collect(),
                ),
                |e| {
                    matches!(e, DeltaError::SectionTreeTooDeep { depth: d, cap }
                        if *d == MAX_SECTION_TREE_DEPTH + 1 && *cap == MAX_SECTION_TREE_DEPTH)
                },
            ),
        ];

        for (label, delta, is_expected) in cases {
            let mut state = make_state();
            let err = apply_in_mode(&mut state, Some(delta), ApplyMode::RestoreBase).unwrap_err();
            assert!(
                is_expected(&err),
                "{label}: RestoreBase skips only the entry cap; got {err:?}"
            );
        }
    }

    // ── Adoption and the cross-map check (b.a4k) ────────────────────────────
    //
    // `debug_assert_cross_map_consistent` exempts only orphans — sections
    // whose parent has no row — and no list is kept under such a parent. A
    // delta that makes an orphan's parent live adopts the orphan into the
    // parent's list (`adopt_orphans`), so the check holds after it, in both
    // apply modes. The tests below pin the adoption and the check's edges.

    /// The orphan's parent in the adoption fixtures: no section.
    const ADOPTED_PARENT: u64 = 77;
    /// A section under [`ADOPTED_PARENT`].
    const ADOPTED_ORPHAN: u64 = 4;

    /// A state holding [`ADOPTED_ORPHAN`] under the missing
    /// [`ADOPTED_PARENT`], with no list for that parent — the only legal
    /// shape: no list is kept under an id with no row.
    fn orphan_state() -> ObjectState {
        let mut state = make_state();
        state.sections.insert(
            ADOPTED_ORPHAN,
            make_section(ADOPTED_ORPHAN, ADOPTED_PARENT, "Orphan"),
        );
        state
    }

    fn structural_delta(cmds: Vec<StructCommand>) -> OwnedDelta {
        OwnedDelta {
            is_diff: false,
            struct_commands: cmds,
            ..Default::default()
        }
    }

    /// A delta creating the parent an orphan names: the orphan is adopted
    /// into the new parent's list, and neither a commit plan nor a stream
    /// replay of it trips the debug consistency check.
    #[test]
    fn apply_delta_creating_an_orphans_parent_adopts_the_orphan_into_its_list() {
        for mode in [ApplyMode::Commit, ApplyMode::Replay] {
            let mut state = orphan_state();

            apply_in_mode(
                &mut state,
                Some(structural_delta(vec![create_cmd(
                    0,
                    ADOPTED_PARENT,
                    "Adopter",
                )])),
                mode,
            )
            .unwrap_or_else(|e| panic!("{mode:?}: the creation applies: {e:?}"));

            assert!(
                state.sections.contains_key(&ADOPTED_PARENT),
                "{mode:?}: the parent is live"
            );
            assert_eq!(
                state.children.get(&ADOPTED_PARENT),
                Some(&vec![ADOPTED_ORPHAN]),
                "{mode:?}: the parent's list holds the adopted orphan"
            );
        }
    }

    /// The forward check exempts only an orphan: a section under a live,
    /// childless parent that is missing from that parent's (absent) list is
    /// drift, whatever the delta does, and fires.
    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "cross-map drift: section 2 (parent 1) missing from children list")]
    fn apply_delta_with_a_live_parents_first_child_unlisted_trips_the_cross_map_check() {
        let mut state = make_state();
        state.sections.insert(1, make_section(1, 0, "Parent"));
        state.children.insert(0, vec![1]);
        // The parent's first child, with no list for the parent.
        state.sections.insert(2, make_section(2, 1, "Child"));
        let _ = apply_delta(
            &mut state,
            Some(structural_delta(vec![create_cmd(0, 3, "Unrelated")])),
        );
    }

    /// One delta that makes an orphan's parent live **and** gives it a child
    /// lists both: the adopted orphan first (adoption runs at the parent's
    /// birth, before the next command), then the new child.
    #[test]
    fn apply_delta_creating_an_orphans_parent_and_a_child_under_it_in_one_delta_lists_the_orphan_then_the_child(
    ) {
        for mode in [ApplyMode::Commit, ApplyMode::Replay] {
            let mut state = orphan_state();
            apply_in_mode(
                &mut state,
                Some(structural_delta(vec![
                    create_cmd(0, ADOPTED_PARENT, "Adopter"),
                    create_cmd(ADOPTED_PARENT, 5, "Child"),
                ])),
                mode,
            )
            .unwrap_or_else(|e| panic!("{mode:?}: the delta applies: {e:?}"));
            assert_eq!(
                state.children.get(&ADOPTED_PARENT),
                Some(&vec![ADOPTED_ORPHAN, 5]),
                "{mode:?}: the orphan is listed first, then the child"
            );
        }
    }

    // ═════════════════════════════════════════════════════════════════════════
    // b.xje — sibling order: appends, stored-parent moves and USD reorders
    //
    // Every client appends a created section and a section moved under a new
    // parent, and orders a `ReorderChildren` by USD's `SdfApplyListOrdering`
    // within each child list. The tests below pin the server's stored list,
    // keys and emitted rows to those answers. Most of them also check the
    // property a client-library audit found broken, directly: a client that applied
    // every delta live ([`ClientModel`]) and a client that joins from a fresh
    // snapshot afterwards hold the same child order under every parent as the
    // server.
    // ═════════════════════════════════════════════════════════════════════════

    /// One section as a client holds it: its parent, its name, its
    /// `SdfSpecType` (`section_type`) and the order key it was placed by.
    #[derive(Clone, Debug)]
    struct ClientRow {
        parent: u64,
        name: String,
        ty: u8,
        key: u32,
    }

    /// Which client's reading of the wire a [`ClientModel`] follows.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum Reader {
        /// The Kit client's default v1 tree: shared per-parent keys, catch-up
        /// orders read as keys with displacement, live reorders slot-filled.
        KitV1,
        /// The C++ client library: catch-up orders read as indexes into the
        /// parent's sequence, then renumbered; live reorders by USD's rule
        /// per child list.
        Ovlive,
    }

    /// Both readers, for tests that run each.
    const READERS: [Reader; 2] = [Reader::KitV1, Reader::Ovlive];

    /// Independent models of a client's scene, built only from what the wire
    /// carries and written from the clients' published rules rather than from
    /// this engine's code. `tests/common/sibling_model.rs` is the twin for the
    /// gRPC and property tests, which cannot reach this module; keep the two
    /// in step.
    ///
    /// Each [`Reader`] is a **model**: a transliteration of one client's
    /// ordering rules, not the client's code. The client revisions and source
    /// anchors each was written from are recorded in
    /// `docs-internal/cross-repo-debugging.md` § Sibling order and move rules.
    ///
    /// Both readers keep each parent's children as one sequence across every
    /// child list, ordered by a per-parent key; a USD child list is that
    /// sequence filtered to the list's children
    /// ([`child_lists`](Self::child_lists)). Shared rules:
    ///
    /// * `CreateSection`, and a `MoveSection` onto a new parent, **append**:
    ///   the child takes the largest key among its new siblings + 1 and lands
    ///   last.
    /// * `MoveSection` classifies against the parent the client holds the
    ///   section under, never the carried `old_parent_id` — except that a
    ///   carried parent equal to the destination is a rename. A section already
    ///   under the destination keeps its key and position.
    /// * `DeleteSection` and a catch-up's `DiffDeleteSection` remove the
    ///   subtree; a name conflict on create or move cascades the lowest-id
    ///   holder, as on the server. A `DiffChangeSection` row cascades no
    ///   holder: neither client does.
    /// * A row whose parent has no row — an *orphan*, which a create or a move
    ///   whose name-conflict cascade deletes the parent it names makes (the
    ///   holder's subtree loops back to it) — is listed nowhere, and when its
    ///   parent is born by a create every row naming it joins its list in
    ///   `(key, id)` order, before the next command. No real client holds
    ///   orphan rows: this is the server's SDD §10.4 rule, modelled so the
    ///   parity checks extend to histories that hold them.
    ///
    /// Reader rules:
    ///
    /// * [`Reader::KitV1`]: a `DiffChangeSection` applies in wire order; a new
    ///   row, one moving to a new parent, or an orphan row the model holds, is
    ///   appended, then takes the
    ///   command's `sectionOrder` as its key, and a sibling already holding
    ///   that key (in any child list) moves to the parent's largest key + 1. A
    ///   change whose parent is not held waits for the parent, after the
    ///   delta's other commands. A live `ReorderChildren` slot-fills per child
    ///   list: the named children, each once, take their own sorted keys in
    ///   command order. Keys are never renumbered.
    /// * [`Reader::Ovlive`]: per delta, retractions are removed and every
    ///   `DiffChangeSection` row upserted with `sectionOrder` as its order;
    ///   then per parent with a restated row, its other listed children keep
    ///   their order, the restated rows (stable-sorted by `sectionOrder`) are
    ///   each inserted at `min(sectionOrder, len)`, and the sequence is
    ///   renumbered `0..n-1`. A live `ReorderChildren` applies USD's
    ///   `SdfApplyListOrdering` ([`usd_list_ordering`], transliterated from
    ///   OpenUSD rather than from this engine) separately to each child list
    ///   ([`client_child_list`]), each list keeping the positions it holds,
    ///   ignoring ids that are not the parent's children; the sequence is then
    ///   renumbered `0..n-1`. That is the service's rule too (SDD §7.7), so a
    ///   live model of this reader holds the server's order after any reorder.
    ///
    /// It holds no caches, no listing index and no stale entries: it is the
    /// answer the engine's bookkeeping must agree with, not a copy of it.
    #[derive(Clone, Debug)]
    struct ClientModel {
        reader: Reader,
        rows: std::collections::BTreeMap<u64, ClientRow>,
        /// Each parent's children in the reader's order (ascending key; keys
        /// are unique per parent between deltas).
        lists: std::collections::BTreeMap<u64, Vec<u64>>,
    }

    impl ClientModel {
        /// An empty scene read by `reader`.
        fn new(reader: Reader) -> Self {
            Self {
                reader,
                rows: std::collections::BTreeMap::new(),
                lists: std::collections::BTreeMap::new(),
            }
        }

        /// A from-empty joiner: the fresh snapshot's sections, each placed by
        /// the sibling position the snapshot carries as its order.
        fn joined_from_snapshot(reader: Reader, state: &ObjectState) -> Self {
            let mut model = Self::new(reader);
            model.apply(&snapshot_delta(state));
            model
        }

        /// Apply one delta the server accepted, each command by the reader's
        /// rule.
        fn apply(&mut self, delta: &OwnedDelta) {
            if delta.is_diff && delta.base_version == 0 {
                *self = Self::new(self.reader);
            }
            let mut deferred: Vec<(usize, OodfDiffChangeSection)> = Vec::new();
            let mut retracted_at: std::collections::HashMap<u64, usize> =
                std::collections::HashMap::new();
            let mut restated: Vec<(u64, u32, u64)> = Vec::new();
            for (index, cmd) in delta.struct_commands.iter().enumerate() {
                match cmd {
                    StructCommand::CreateSection(c) => {
                        self.create(c.parent_id, c.section_id, &c.section_name, c.section_type);
                    }
                    StructCommand::DiffChangeSection(c) => match self.reader {
                        Reader::KitV1 => {
                            if !self.kit_change(c) {
                                deferred.push((index, c.clone()));
                            }
                        }
                        Reader::Ovlive => {
                            self.ovlive_upsert(c);
                            restated.push((c.parent_id, c.section_order, c.section_id));
                        }
                    },
                    StructCommand::DeleteSection(c) => self.delete(c.section_id),
                    StructCommand::DiffDeleteSection(c) => {
                        retracted_at.insert(c.section_id, index);
                        self.delete(c.section_id);
                    }
                    StructCommand::MoveSection(c) => {
                        self.move_section(
                            c.section_id,
                            c.old_parent_id,
                            c.new_parent_id,
                            &c.new_name,
                        );
                    }
                    StructCommand::ReorderChildren(c) => {
                        self.reorder(c.section_id, &c.children_list);
                    }
                }
            }
            match self.reader {
                Reader::KitV1 => self.kit_apply_deferred(deferred, &retracted_at),
                Reader::Ovlive => self.ovlive_place(&restated),
            }
        }

        /// Every non-empty children sequence, by parent.
        fn lists(&self) -> std::collections::BTreeMap<u64, Vec<u64>> {
            self.lists
                .iter()
                .filter(|(_, l)| !l.is_empty())
                .map(|(p, l)| (*p, l.clone()))
                .collect()
        }

        /// Every non-empty USD child list, by `(parent, child list)`.
        fn child_lists(&self) -> std::collections::BTreeMap<(u64, Option<u8>), Vec<u64>> {
            split_child_lists(self.lists.iter().map(|(p, l)| (*p, l)), |id| {
                self.rows.get(&id).map(|r| r.ty)
            })
        }

        /// The lowest-id section named `name` under `parent`, other than
        /// `except`.
        fn holder(&self, parent: u64, name: &str, except: u64) -> Option<u64> {
            self.rows
                .iter()
                .find(|(id, r)| **id != except && r.parent == parent && r.name == name)
                .map(|(id, _)| *id)
        }

        /// `candidate` is `ancestor` or sits under it.
        fn is_descendant(&self, mut candidate: u64, ancestor: u64) -> bool {
            let mut seen = std::collections::HashSet::new();
            loop {
                if candidate == ancestor {
                    return true;
                }
                if !seen.insert(candidate) {
                    return false;
                }
                match self.rows.get(&candidate) {
                    Some(row) => candidate = row.parent,
                    None => return false,
                }
            }
        }

        fn append_key(&self, parent: u64) -> u32 {
            self.lists
                .get(&parent)
                .and_then(|l| l.iter().map(|id| self.rows[id].key).max())
                .map_or(0, |largest| largest.saturating_add(1))
        }

        fn unlink(&mut self, parent: u64, id: u64) {
            if let Some(list) = self.lists.get_mut(&parent) {
                list.retain(|c| *c != id);
            }
        }

        fn sort_by_key(&mut self, parent: u64) {
            let rows = &self.rows;
            if let Some(list) = self.lists.get_mut(&parent) {
                list.sort_by_key(|c| rows[c].key);
            }
        }

        /// List `id` under `parent` after every sibling whose key is ≤ its
        /// own — unless `parent` is neither the root nor a row, when `id` is
        /// an orphan and is listed nowhere.
        fn insert_by_key(&mut self, parent: u64, id: u64) {
            if parent != 0 && !self.rows.contains_key(&parent) {
                return;
            }
            let key = self.rows[&id].key;
            let rows = &self.rows;
            let list = self.lists.entry(parent).or_default();
            let at = list
                .iter()
                .position(|c| rows[c].key > key)
                .unwrap_or(list.len());
            list.insert(at, id);
        }

        /// `id` was just given a row by a create: every row already naming it
        /// joins its list, in `(key, id)` order.
        fn adopt(&mut self, id: u64) {
            if id == 0 {
                return;
            }
            let mut orphans: Vec<(u32, u64)> = self
                .rows
                .iter()
                .filter(|(cid, r)| r.parent == id && **cid != id)
                .map(|(cid, r)| (r.key, *cid))
                .collect();
            orphans.sort_unstable();
            for (_, cid) in orphans {
                self.insert_by_key(id, cid);
            }
        }

        fn create(&mut self, parent: u64, id: u64, name: &str, ty: u8) {
            if self.rows.contains_key(&id) || (parent != 0 && !self.rows.contains_key(&parent)) {
                return;
            }
            if let Some(holder) = self.holder(parent, name, id) {
                self.delete(holder);
            }
            let key = self.append_key(parent);
            self.rows.insert(
                id,
                ClientRow {
                    parent,
                    name: name.to_string(),
                    ty,
                    key,
                },
            );
            // The holder's cascade can take the parent with it (a holder
            // whose subtree loops back to the parent); the new row is then an
            // orphan.
            if parent == 0 || self.rows.contains_key(&parent) {
                self.lists.entry(parent).or_default().push(id);
            }
            self.adopt(id);
        }

        /// The Kit client's `DiffChangeSection`; `false`, applying nothing,
        /// when the parent is not held.
        fn kit_change(&mut self, c: &OodfDiffChangeSection) -> bool {
            let (parent, id) = (c.parent_id, c.section_id);
            if parent != 0 && !self.rows.contains_key(&parent) {
                return false;
            }
            match self.rows.get(&id).map(|r| r.parent) {
                None => {
                    let key = self.append_key(parent);
                    self.rows.insert(
                        id,
                        ClientRow {
                            parent,
                            name: c.section_name.clone(),
                            ty: c.section_type,
                            key,
                        },
                    );
                    self.lists.entry(parent).or_default().push(id);
                }
                Some(old_parent) => {
                    // An orphan row the model holds (the server's adoption
                    // rule; no Kit client holds one) is not listed under its
                    // parent, so it is linked like a row moving in.
                    let unlisted = self.lists.get(&parent).is_none_or(|l| !l.contains(&id));
                    if old_parent != parent || unlisted {
                        self.unlink(old_parent, id);
                        let key = self.append_key(parent);
                        let row = self.rows.get_mut(&id).expect("checked above");
                        row.parent = parent;
                        row.key = key;
                        self.lists.entry(parent).or_default().push(id);
                    }
                    let row = self.rows.get_mut(&id).expect("checked above");
                    row.name = c.section_name.clone();
                    row.ty = c.section_type;
                }
            }
            if self.rows[&id].key != c.section_order {
                // Displacement: a sibling holding the key, in any child list,
                // moves to the largest key + 1.
                let holder = self
                    .lists
                    .get(&parent)
                    .and_then(|l| {
                        l.iter()
                            .find(|s| **s != id && self.rows[s].key == c.section_order)
                    })
                    .copied();
                if let Some(holder) = holder {
                    let next = self.append_key(parent);
                    self.rows
                        .get_mut(&holder)
                        .expect("a listed child has a row")
                        .key = next;
                }
                self.rows.get_mut(&id).expect("linked above").key = c.section_order;
                self.sort_by_key(parent);
            }
            true
        }

        /// The Kit client's deferred changes, applied once their parent has
        /// arrived, unless a later retraction in the delta superseded them.
        fn kit_apply_deferred(
            &mut self,
            mut pending: Vec<(usize, OodfDiffChangeSection)>,
            retracted_at: &std::collections::HashMap<u64, usize>,
        ) {
            loop {
                let before = pending.len();
                let mut still = Vec::new();
                for (index, c) in pending {
                    if retracted_at
                        .get(&c.section_id)
                        .is_some_and(|&at| at > index)
                    {
                        continue;
                    }
                    if !self.kit_change(&c) {
                        still.push((index, c));
                    }
                }
                pending = still;
                if pending.is_empty() || pending.len() == before {
                    return;
                }
            }
        }

        /// The C++ client library's `DiffChangeSection` before placement.
        fn ovlive_upsert(&mut self, c: &OodfDiffChangeSection) {
            let id = c.section_id;
            if let Some(old_parent) = self.rows.get(&id).map(|r| r.parent) {
                self.unlink(old_parent, id);
            }
            self.rows.insert(
                id,
                ClientRow {
                    parent: c.parent_id,
                    name: c.section_name.clone(),
                    ty: c.section_type,
                    key: c.section_order,
                },
            );
        }

        /// The C++ client library's placement of a delta's restated rows, per
        /// parent (type doc).
        fn ovlive_place(&mut self, restated: &[(u64, u32, u64)]) {
            if restated.is_empty() {
                return;
            }
            let restated_ids: std::collections::HashSet<u64> =
                restated.iter().map(|(_, _, id)| *id).collect();
            let mut by_parent: std::collections::BTreeMap<u64, Vec<(u32, u64)>> =
                std::collections::BTreeMap::new();
            for &(parent, order, id) in restated {
                by_parent.entry(parent).or_default().push((order, id));
            }
            for (parent, mut entries) in by_parent {
                if parent != 0 && !self.rows.contains_key(&parent) {
                    continue;
                }
                let mut others: Vec<(u32, u64)> = self
                    .lists
                    .get(&parent)
                    .map(|l| {
                        l.iter()
                            .filter(|id| !restated_ids.contains(id))
                            .map(|id| (self.rows[id].key, *id))
                            .collect()
                    })
                    .unwrap_or_default();
                others.sort_unstable();
                let mut children: Vec<u64> = others.into_iter().map(|(_, id)| id).collect();
                entries.sort_by_key(|(order, _)| *order);
                for (order, id) in entries {
                    if self.rows.get(&id).is_none_or(|r| r.parent != parent)
                        || children.contains(&id)
                    {
                        continue;
                    }
                    let at = (order as usize).min(children.len());
                    children.insert(at, id);
                }
                for (position, id) in children.iter().enumerate() {
                    self.rows.get_mut(id).expect("a placed child has a row").key =
                        u32::try_from(position).expect("a test list is short");
                }
                self.lists.insert(parent, children);
            }
        }

        fn delete(&mut self, id: u64) {
            let Some(parent) = self.rows.get(&id).map(|r| r.parent) else {
                return;
            };
            self.unlink(parent, id);
            let mut stack = vec![id];
            let mut seen = std::collections::HashSet::new();
            while let Some(sid) = stack.pop() {
                if !seen.insert(sid) {
                    continue;
                }
                self.rows.remove(&sid);
                if let Some(children) = self.lists.remove(&sid) {
                    stack.extend(children);
                }
            }
        }

        fn move_section(&mut self, id: u64, carried: u64, dest: u64, name: &str) {
            let Some(current) = self.rows.get(&id).map(|r| r.parent) else {
                return;
            };
            if carried == dest {
                self.rows.get_mut(&id).expect("checked above").name = name.to_string();
                return;
            }
            if dest != 0 && !self.rows.contains_key(&dest) {
                self.delete(id);
                return;
            }
            if self.is_descendant(dest, id) {
                self.delete(id);
                return;
            }
            if let Some(holder) = self.holder(dest, name, id) {
                if self.is_descendant(id, holder) {
                    self.delete(id);
                    return;
                }
                self.delete(holder);
            }
            if dest == current {
                self.rows.get_mut(&id).expect("checked above").name = name.to_string();
                return;
            }
            self.unlink(current, id);
            let key = self.append_key(dest);
            let row = self.rows.get_mut(&id).expect("checked above");
            row.parent = dest;
            row.name = name.to_string();
            row.key = key;
            // The holder's cascade can take the destination with it (a
            // holder whose subtree loops back to the destination); the
            // section then sits under a parent with no row, an orphan listed
            // nowhere.
            if dest == 0 || self.rows.contains_key(&dest) {
                self.lists.entry(dest).or_default().push(id);
            }
        }

        fn reorder(&mut self, parent: u64, named: &[u64]) {
            let list = self.lists.get(&parent).cloned().unwrap_or_default();
            if !named.iter().any(|id| list.contains(id)) {
                return;
            }
            let parent_ty = self.rows.get(&parent).map(|r| r.ty);
            let list_of = |id: &u64| client_child_list(parent_ty, self.rows[id].ty);
            match self.reader {
                Reader::KitV1 => {
                    let mut seen = std::collections::HashSet::new();
                    let mut by_list: std::collections::BTreeMap<Option<u8>, Vec<u64>> =
                        std::collections::BTreeMap::new();
                    for id in named {
                        if list.contains(id) && seen.insert(*id) {
                            by_list.entry(list_of(id)).or_default().push(*id);
                        }
                    }
                    for members in by_list.values() {
                        let mut keys: Vec<u32> =
                            members.iter().map(|id| self.rows[id].key).collect();
                        keys.sort_unstable();
                        for (id, key) in members.iter().zip(keys) {
                            self.rows.get_mut(id).expect("a listed child has a row").key = key;
                        }
                    }
                    self.sort_by_key(parent);
                }
                Reader::Ovlive => {
                    let mut groups: std::collections::BTreeMap<Option<u8>, Vec<usize>> =
                        std::collections::BTreeMap::new();
                    for (position, id) in list.iter().enumerate() {
                        groups.entry(list_of(id)).or_default().push(position);
                    }
                    let mut placed = list.clone();
                    for positions in groups.values() {
                        let members: Vec<u64> = positions.iter().map(|&p| list[p]).collect();
                        for (&p, id) in positions.iter().zip(usd_list_ordering(&members, named)) {
                            placed[p] = id;
                        }
                    }
                    for (position, id) in placed.iter().enumerate() {
                        self.rows.get_mut(id).expect("a listed child has a row").key =
                            u32::try_from(position).expect("a test list is short");
                    }
                    self.lists.insert(parent, placed);
                }
            }
        }
    }

    /// `SdfApplyListOrdering(list, order)`, transliterated from OpenUSD's
    /// `SdfListOp<T>::_ReorderKeysHelper` (`pxr/usd/sdf/listOp.cpp:674-726`)
    /// and its entry point (`:897-921`) @ `ee47c679` — the model's own copy,
    /// deliberately not the engine's run-based formulation.
    ///
    /// `scratch` is USD's linked list with a tombstone where an item has been
    /// spliced out; `search` maps each item to its position (the last
    /// occurrence wins, as `search[*i] = i` does — the lists here hold no
    /// repeats). For each first occurrence in `order` of an item the list
    /// holds, the walk splices out that item and every following item `order`
    /// does not name, up to the next named item still in the list, and appends
    /// the run to `result`; whatever is left in `scratch` goes in front.
    fn usd_list_ordering(list: &[u64], order: &[u64]) -> Vec<u64> {
        if order.is_empty() || list.is_empty() {
            return list.to_vec();
        }
        let mut order_set = std::collections::HashSet::new();
        let unique_order: Vec<u64> = order
            .iter()
            .copied()
            .filter(|id| order_set.insert(*id))
            .collect();
        let mut scratch: Vec<Option<u64>> = list.iter().copied().map(Some).collect();
        let search: std::collections::HashMap<u64, usize> =
            list.iter().enumerate().map(|(i, id)| (*id, i)).collect();
        let mut result = Vec::new();
        for id in &unique_order {
            let Some(&j) = search.get(id) else {
                continue;
            };
            let mut e = j + 1;
            while e < scratch.len() && scratch[e].is_none_or(|x| !order_set.contains(&x)) {
                e += 1;
            }
            result.extend(scratch[j..e].iter_mut().filter_map(Option::take));
        }
        let mut out: Vec<u64> = scratch.into_iter().flatten().collect();
        out.extend(result);
        out
    }

    /// Which child list of its parent a child belongs to, written from the
    /// C++ client library's child-list table (the table at the revision
    /// recorded in `docs-internal/cross-repo-debugging.md` § Sibling order and
    /// move rules), keyed by `SdfSpecType` values (OpenUSD
    /// `pxr/usd/sdf/types.h:71-89`): Attribute 1, Connection 2, Expression 3,
    /// Mapper 4, MapperArg 5, Prim 6, PseudoRoot 7, Relationship 8,
    /// RelationshipTarget 9, Variant 10, VariantSet 11. `None` is the one
    /// fallback list: every other pair, the root (no parent row) and Unknown
    /// (0).
    fn client_child_list(parent_ty: Option<u8>, child_ty: u8) -> Option<u8> {
        match (parent_ty?, child_ty) {
            (7 | 6 | 10, 6) => Some(0),
            (6 | 10, 1 | 8) => Some(1),
            (6 | 10, 11) => Some(2),
            (11, 10) => Some(0),
            (1, 2) => Some(0),
            (1, 3) => Some(1),
            (1, 4) => Some(2),
            (8, 9) => Some(0),
            (8, 3) => Some(1),
            (9, 1) => Some(0),
            (4, 5) => Some(0),
            _ => None,
        }
    }

    /// Split each parent's sequence into its child lists, keeping the
    /// sequence's order within each; an entry with no row is skipped.
    fn split_child_lists<'a>(
        lists: impl IntoIterator<Item = (u64, &'a Vec<u64>)>,
        ty_of: impl Fn(u64) -> Option<u8>,
    ) -> std::collections::BTreeMap<(u64, Option<u8>), Vec<u64>> {
        let mut out: std::collections::BTreeMap<(u64, Option<u8>), Vec<u64>> =
            std::collections::BTreeMap::new();
        for (parent, list) in lists {
            let parent_ty = ty_of(parent);
            for &id in list {
                let Some(ty) = ty_of(id) else {
                    continue;
                };
                out.entry((parent, client_child_list(parent_ty, ty)))
                    .or_default()
                    .push(id);
            }
        }
        out
    }

    /// The server's stored order split into child lists: each stored list
    /// filtered to the parent's true children (first occurrence, row naming
    /// that parent), grouped by [`client_child_list`].
    fn server_child_lists(
        state: &ObjectState,
    ) -> std::collections::BTreeMap<(u64, Option<u8>), Vec<u64>> {
        let true_children: std::collections::BTreeMap<u64, Vec<u64>> = state
            .children
            .iter()
            .map(|(parent, list)| {
                let mut seen = std::collections::HashSet::new();
                let kept: Vec<u64> = list
                    .iter()
                    .copied()
                    .filter(|id| {
                        state
                            .sections
                            .get(id)
                            .is_some_and(|s| s.parent_id == *parent)
                            && seen.insert(*id)
                    })
                    .collect();
                (*parent, kept)
            })
            .collect();
        split_child_lists(true_children.iter().map(|(p, l)| (*p, l)), |id| {
            state.sections.get(&id).map(|s| s.section_type)
        })
    }

    /// The from-empty catch-up a joiner receives now: `plan_faithful_snapshot`
    /// as the reset delta of `DiffChangeSection`s the wire carries.
    fn snapshot_delta(state: &ObjectState) -> OwnedDelta {
        match crate::domain::diff::DiffEngine::plan_faithful_snapshot(state)
            .expect("the snapshot must plan")
        {
            crate::domain::diff::DiffPlan::FullContent { sections, .. } => OwnedDelta {
                is_diff: true,
                base_version: 0,
                struct_commands: sections.iter().map(section_as_change).collect(),
                ..Default::default()
            },
            other => panic!("a snapshot plans FullContent, got {other:?}"),
        }
    }

    /// The `DiffChangeSection` a catch-up or checkpoint base carries for `s`:
    /// its parent, name, type and (emitted) order.
    fn section_as_change(s: &Section) -> StructCommand {
        typed_change_cmd(
            s.parent_id,
            s.section_id,
            &s.name,
            s.section_type,
            s.section_order,
        )
    }

    /// [`change_cmd`] with an explicit `section_type`.
    fn typed_change_cmd(
        parent_id: u64,
        section_id: u64,
        name: &str,
        section_type: u8,
        section_order: u32,
    ) -> StructCommand {
        StructCommand::DiffChangeSection(OodfDiffChangeSection {
            parent_id,
            section_id,
            section_name: name.to_string(),
            section_type,
            section_order,
        })
    }

    /// The structural core of the checkpoint round trip, without the FFI
    /// encoding: `build_reset_ompf_from_state` plans
    /// `DiffEngine::plan(state, &[], 0, 0, None)` — full content, each row at
    /// its *stored* key — and `decode_checkpoint_ompf` replays that reset of
    /// `DiffChangeSection`s through `apply_delta`. The real pair is driven in
    /// `src/checkpoint/assembler.rs`'s b.xje tests.
    fn full_content_round_trip(state: &ObjectState) -> ObjectState {
        let sections = match crate::domain::diff::DiffEngine::plan(state, &[], 0, 0, None)
            .expect("full content must plan")
        {
            crate::domain::diff::DiffPlan::FullContent { sections, .. } => sections,
            other => panic!("have_version 0 plans FullContent, got {other:?}"),
        };
        let mut rebuilt = ObjectState {
            object_id: state.object_id,
            ..Default::default()
        };
        apply_delta(
            &mut rebuilt,
            Some(OwnedDelta {
                is_diff: true,
                base_version: 0,
                struct_commands: sections.iter().map(section_as_change).collect(),
                ..Default::default()
            }),
        )
        .expect("the full-content reset must apply");
        rebuilt
    }

    /// Every non-empty children list the server stores, by parent.
    fn server_lists(state: &ObjectState) -> std::collections::BTreeMap<u64, Vec<u64>> {
        state
            .children
            .iter()
            .filter(|(_, l)| !l.is_empty())
            .map(|(p, l)| (*p, l.clone()))
            .collect()
    }

    /// The rows a delta's `SetSection`s wrote, by id.
    fn set_section_rows(mutations: &[StorageMutation]) -> std::collections::BTreeMap<u64, Section> {
        mutations
            .iter()
            .filter_map(|m| match m {
                StorageMutation::SetSection { section, .. } => {
                    Some((section.section_id, section.clone()))
                }
                _ => None,
            })
            .collect()
    }

    /// Every stored list is non-decreasing in key.
    fn assert_lists_sorted(label: &str, state: &ObjectState) {
        for (parent, list) in &state.children {
            let keys: Vec<u32> = list
                .iter()
                .map(|id| effective_order(&state.sections, *id))
                .collect();
            assert!(
                keys.windows(2).all(|w| w[0] <= w[1]),
                "{label}: parent {parent}'s list {list:?} is not sorted by key: {keys:?}"
            );
        }
    }

    /// The client-visible property: a client that applied every delta live and
    /// a client of either reader joining from a fresh snapshot now hold the
    /// server's child order under every parent.
    fn assert_live_and_snapshot_parity(label: &str, live: &ClientModel, state: &ObjectState) {
        let stored = server_lists(state);
        assert_eq!(
            live.lists(),
            stored,
            "{label}: a client that applied every delta live must hold the server's child \
             order under every parent"
        );
        for reader in READERS {
            assert_eq!(
                ClientModel::joined_from_snapshot(reader, state).lists(),
                stored,
                "{label} / {reader:?}: a client joining from a fresh snapshot must hold the same \
                 child order"
            );
        }
    }

    /// Apply `delta` to the server state and to the live client model.
    fn apply_both(
        state: &mut ObjectState,
        live: &mut ClientModel,
        delta: OwnedDelta,
    ) -> Vec<StorageMutation> {
        live.apply(&delta);
        apply_delta(state, Some(delta)).expect("delta must apply")
    }

    // ── Appends ──────────────────────────────────────────────────────────────

    /// The client-library audit's worked example: siblings seeded with explicit orders
    /// A(0), B(1) by `DiffChangeSection`, then an ordinary `CreateSection` C.
    /// Every client appends C, so the server stores `[A, B, C]` with C at key 2
    /// and a fresh snapshot emits A0 B1 C2.
    ///
    /// Fails on the pre-change engine, which stored C at order 0 and inserted
    /// it after the order-0 sibling: `[A, C, B]`.
    #[test]
    fn apply_delta_create_section_after_seeded_siblings_appends_last() {
        const P: u64 = 1;
        const A: u64 = 10;
        const B: u64 = 11;
        const C: u64 = 12;

        let mut state = make_state();
        let mut live = ClientModel::new(Reader::Ovlive);
        apply_both(
            &mut state,
            &mut live,
            batch(vec![
                change_cmd(0, P, "P", 0),
                change_cmd(P, A, "A", 0),
                change_cmd(P, B, "B", 1),
            ]),
        );
        apply_both(&mut state, &mut live, batch(vec![create_cmd(P, C, "C")]));

        assert_eq!(state.children[&P], vec![A, B, C], "the create lands last");
        assert_eq!(
            state.sections[&C].section_order, 2,
            "the created child takes the largest sibling key + 1"
        );
        let emitted: Vec<(u64, u32)> = snapshot_delta(&state)
            .struct_commands
            .iter()
            .filter_map(|c| match c {
                StructCommand::DiffChangeSection(s) if s.parent_id == P => {
                    Some((s.section_id, s.section_order))
                }
                _ => None,
            })
            .collect();
        assert_eq!(
            emitted,
            vec![(A, 0), (B, 1), (C, 2)],
            "a fresh snapshot emits the appended child last"
        );
        assert_live_and_snapshot_parity("create after seeded siblings", &live, &state);
    }

    /// A cross-parent `MoveSection` into a parent whose children carry seeded
    /// keys X(0), Y(1): the moved section takes the destination's append key
    /// (2) and lands last, as every client places it.
    ///
    /// Fails on the pre-change engine, which carried Z's old key (0) across and
    /// inserted it after X: `[X, Z, Y]`.
    #[test]
    fn apply_delta_cross_parent_move_into_a_seeded_parent_appends_last() {
        const P: u64 = 1;
        const Q: u64 = 2;
        const X: u64 = 20;
        const Y: u64 = 21;
        const Z: u64 = 10;

        let mut state = make_state();
        let mut live = ClientModel::new(Reader::Ovlive);
        apply_both(
            &mut state,
            &mut live,
            batch(vec![
                change_cmd(0, P, "P", 0),
                change_cmd(0, Q, "Q", 1),
                change_cmd(P, Z, "Z", 0),
                change_cmd(Q, X, "X", 0),
                change_cmd(Q, Y, "Y", 1),
            ]),
        );
        let mutations = apply_both(&mut state, &mut live, batch(vec![move_cmd(Z, P, Q, "Z")]));

        assert_eq!(
            state.children[&Q],
            vec![X, Y, Z],
            "the moved section lands last"
        );
        assert_eq!(
            state.sections[&Z].section_order, 2,
            "the moved section takes the destination's append key, not its old key"
        );
        assert_eq!(
            set_section_rows(&mutations)[&Z].section_order,
            2,
            "the row written for the move carries the append key"
        );
        assert_live_and_snapshot_parity("cross-parent move", &live, &state);
    }

    /// A create under a parent whose listed child had its order raised
    /// through **another** parent stores the true largest key + 1.
    ///
    /// Parent 1 lists section 3 (a stale listing — section 3's row names
    /// parent 2), and the order raise to 50 is written through parent 2, so
    /// only the dependency index can tell parent 1's cache. The append key is
    /// read off that cache: a fresh one holds `[50, 1]` and gives 51, a stale
    /// one still holds `[0, 1]` and gives 2.
    #[test]
    fn apply_delta_create_after_a_cross_parent_order_raise_stores_the_true_max_plus_one() {
        let mut state = make_state();
        seed_section(&mut state, 1, 0, "P1", 0);
        seed_section(&mut state, 2, 0, "P2", 1);
        seed_section(&mut state, 3, 2, "C", 0);
        state.children.insert(0, vec![1, 2]);
        state.children.insert(1, vec![3]);
        state.children.insert(2, Vec::new());

        apply_counted(
            &mut state,
            batch(vec![
                // Populates parent 1's cache while section 3 still reads order
                // 0; section 9 appends at key 1.
                create_cmd(1, 9, "N9"),
                // Raises section 3's order THROUGH PARENT 2.
                change_cmd(2, 3, "C", 50),
                // Appends under parent 1, reading its cache for the key.
                create_cmd(1, 10, "N10"),
                // Repair: section 3's row names parent 2, so the move renames
                // it in place and unlinks it from the carried parent 1.
                move_cmd(3, 1, 2, "C"),
            ]),
        );

        assert_eq!(
            state.sections[&9].section_order, 1,
            "precondition: 9 appended at key 1"
        );
        assert_eq!(
            state.sections[&10].section_order, 51,
            "the append key is the true largest key (50) + 1; a stale cache gives 2"
        );
        assert_eq!(state.children[&1], vec![9, 10]);
        assert_eq!(state.children[&2], vec![3]);
    }

    // ── A create after a complete reorder ───────────────────────────────────

    /// A `ReorderChildren` naming every child installs exactly the client's
    /// list, with each position keeping its key — C0 A1 B2, a `SetSection`
    /// for exactly the three children it moved — and a `CreateSection` after
    /// it appends last at key 3, as every client places it.
    ///
    /// Fails on the pre-change engine, which stored the create at order 0 and,
    /// on the reordered list (keys `[2, 0, 1]`), placed it first:
    /// `[D, C, A, B]`.
    #[test]
    fn apply_delta_create_after_a_complete_reorder_appends_last() {
        const P: u64 = 1;
        const A: u64 = 10;
        const B: u64 = 11;
        const C: u64 = 12;
        const D: u64 = 13;

        let mut state = make_state();
        let mut live = ClientModel::new(Reader::Ovlive);
        apply_both(
            &mut state,
            &mut live,
            batch(vec![
                change_cmd(0, P, "P", 0),
                change_cmd(P, A, "A", 0),
                change_cmd(P, B, "B", 1),
                change_cmd(P, C, "C", 2),
            ]),
        );
        let mutations = apply_both(
            &mut state,
            &mut live,
            batch(vec![reorder_cmd(P, vec![C, A, B])]),
        );
        assert_eq!(
            state.children[&P],
            vec![C, A, B],
            "the client's complete list"
        );
        assert_eq!(
            listed_keys(&state, P),
            vec![(C, 0), (A, 1), (B, 2)],
            "each position keeps its key"
        );
        let rewritten = set_section_rows(&mutations);
        assert_eq!(
            rewritten.keys().copied().collect::<Vec<_>>(),
            vec![A, B, C],
            "a SetSection for exactly the re-keyed children — all three moved"
        );
        assert!(
            rewritten
                .values()
                .all(|s| s.last_version == state.sequence_num),
            "each re-keyed row carries the reorder's version"
        );

        apply_both(&mut state, &mut live, batch(vec![create_cmd(P, D, "D")]));
        assert_eq!(state.children[&P], vec![C, A, B, D], "the create appends");
        assert_eq!(
            state.sections[&D].section_order, 3,
            "at the largest key + 1"
        );
        assert_live_and_snapshot_parity("complete reorder then create", &live, &state);
    }

    // ── MoveSection with a stale carried parent ─────────────────────────────
    //
    // Two clients moving the same section concurrently both carry the parent
    // they saw. The second commit's `old_parent_id` is then stale: the section
    // already sits under the parent the first commit moved it to. Every client
    // that applied the first commit classifies the second against the parent
    // it holds the section under, so the server must too.

    const SC_P1: u64 = 1;
    const SC_P2: u64 = 2;
    const SC_P3: u64 = 3;
    const SC_X: u64 = 20;
    const SC_Y: u64 = 21;
    const SC_Z: u64 = 30;
    const SC_S: u64 = 40;
    const SC_K: u64 = 41;
    const SC_T: u64 = 50;

    /// P1, P2 and P3 under the root; P2 holds X(0) and Y(1), P3 holds Z(0), and
    /// S sits under P1 with a child K.
    fn stale_carried_parent_seed() -> OwnedDelta {
        batch(vec![
            change_cmd(0, SC_P1, "P1", 0),
            change_cmd(0, SC_P2, "P2", 1),
            change_cmd(0, SC_P3, "P3", 2),
            change_cmd(SC_P2, SC_X, "X", 0),
            change_cmd(SC_P2, SC_Y, "Y", 1),
            change_cmd(SC_P3, SC_Z, "Z", 0),
            change_cmd(SC_P1, SC_S, "S", 0),
            change_cmd(SC_S, SC_K, "K", 0),
        ])
    }

    /// The seed, then the first client's truthful move of S from P1 to P2.
    fn stale_carried_parent_after_the_first_move() -> (ObjectState, ClientModel) {
        let mut state = make_state();
        let mut live = ClientModel::new(Reader::Ovlive);
        apply_both(&mut state, &mut live, stale_carried_parent_seed());
        apply_both(
            &mut state,
            &mut live,
            batch(vec![move_cmd(SC_S, SC_P1, SC_P2, "S")]),
        );
        assert_eq!(
            state.children[&SC_P2],
            vec![SC_X, SC_Y, SC_S],
            "precondition: the first move appended S under P2"
        );
        (state, live)
    }

    /// A second client repeats the move S: P1 → P2 after the first client's
    /// identical move landed (with a create of T under P2 in between). S is
    /// already under P2, so it stays where it is: same position, same key — a
    /// rename at most. Variants: the second move renames S (renamed in place),
    /// and renames it onto a sibling's name (that sibling is cascade-deleted,
    /// edge 5, and S still keeps its place).
    ///
    /// Fails on the pre-change engine, which carried S's old key into the
    /// first move and placed it mid-list.
    #[test]
    fn apply_delta_repeated_move_with_a_stale_carried_parent_keeps_the_section_in_place() {
        for (label, rename_to, expected) in [
            ("same name", "S", vec![SC_X, SC_Y, SC_S, SC_T]),
            ("renamed", "S2", vec![SC_X, SC_Y, SC_S, SC_T]),
            ("renamed onto a sibling's name", "X", vec![SC_Y, SC_S, SC_T]),
        ] {
            let (mut state, mut live) = stale_carried_parent_after_the_first_move();
            let first_key = state.sections[&SC_S].section_order;
            apply_both(
                &mut state,
                &mut live,
                batch(vec![create_cmd(SC_P2, SC_T, "T")]),
            );
            let mutations = apply_both(
                &mut state,
                &mut live,
                batch(vec![move_cmd(SC_S, SC_P1, SC_P2, rename_to)]),
            );

            assert_eq!(
                state.children[&SC_P2], expected,
                "{label}: S keeps its place"
            );
            assert_eq!(
                state.sections[&SC_S].section_order, first_key,
                "{label}: the repeated move does not re-key S"
            );
            assert_eq!(
                state.sections[&SC_S].name, rename_to,
                "{label}: the name is written"
            );
            assert_lists_sorted(label, &state);
            assert_eq!(
                server_lists(&full_content_round_trip(&state)),
                server_lists(&state),
                "{label}: the checkpoint round trip rebuilds the live lists"
            );
            assert_live_and_snapshot_parity(label, &live, &state);
            if rename_to == "X" {
                assert!(
                    !state.sections.contains_key(&SC_X)
                        && mutations.iter().any(|m| matches!(
                            m,
                            StorageMutation::CreateTombstone {
                                section_id: SC_X,
                                ..
                            }
                        )),
                    "{label}: the sibling holding the name is cascade-deleted"
                );
            }
        }
    }

    /// The second client moves S from its stale P1 to a third parent P3. S
    /// leaves P2 — the list that holds it — and lands last under P3; no list
    /// keeps a stale listing of it, so a later `DeleteSection(P2)` leaves S's
    /// row alive and writes no tombstone for it.
    ///
    /// On the pre-change engine the move unlinked S from P1 only: P2 kept
    /// listing it, the reverse cross-map check panicked, and (in release) P2's
    /// delete would have tombstoned S through that listing.
    #[test]
    fn apply_delta_move_with_a_stale_carried_parent_to_another_parent_leaves_no_stale_listing() {
        let (mut state, mut live) = stale_carried_parent_after_the_first_move();

        apply_both(
            &mut state,
            &mut live,
            batch(vec![move_cmd(SC_S, SC_P1, SC_P3, "S")]),
        );

        assert_eq!(
            state.children[&SC_P2],
            vec![SC_X, SC_Y],
            "S left the list holding it"
        );
        assert!(
            !state
                .children
                .get(&SC_P1)
                .is_some_and(|l| l.contains(&SC_S)),
            "S is not listed under the carried parent either"
        );
        assert_eq!(
            state.children[&SC_P3],
            vec![SC_Z, SC_S],
            "S lands last under P3"
        );
        let emitted = snapshot_delta(&state)
            .struct_commands
            .iter()
            .find_map(|c| match c {
                StructCommand::DiffChangeSection(s) if s.section_id == SC_S => {
                    Some((s.parent_id, s.section_order))
                }
                _ => None,
            });
        assert_eq!(
            emitted,
            Some((SC_P3, 1)),
            "a fresh snapshot emits S under P3 at its last index"
        );
        assert_live_and_snapshot_parity("move to another parent", &live, &state);

        let mutations = apply_both(&mut state, &mut live, batch(vec![delete_cmd(SC_P2)]));
        assert!(
            state.sections.contains_key(&SC_S),
            "deleting P2 does not reach S"
        );
        assert!(
            !mutations.iter().any(|m| matches!(
                m,
                StorageMutation::CreateTombstone {
                    section_id: SC_S,
                    ..
                }
            )),
            "and writes no tombstone for it"
        );
        assert_live_and_snapshot_parity("then delete P2", &live, &state);
    }

    /// The deleting edges — 2 (missing destination), 3 (destination under the
    /// section) and 4 (an ancestor holds the name at the destination) — with a
    /// stale carried parent: S is deleted out of the list that holds it, so no
    /// ghost listing of S survives under P2.
    ///
    /// On the pre-change engine the cascade unlinked S from the carried P1
    /// only, leaving P2 listing a section with no row, and the reverse
    /// cross-map check panicked.
    #[test]
    fn apply_delta_deleting_move_edges_with_a_stale_carried_parent_leave_no_ghost_listing() {
        for (edge, final_move) in [
            ("edge 2", move_cmd(SC_S, SC_P1, 999, "S")),
            ("edge 3", move_cmd(SC_S, SC_P1, SC_K, "S")),
            // P2, S's parent, is the root's child named "P2".
            ("edge 4", move_cmd(SC_S, SC_P1, 0, "P2")),
        ] {
            let (mut state, mut live) = stale_carried_parent_after_the_first_move();
            let mutations = apply_both(&mut state, &mut live, batch(vec![final_move]));

            assert!(!state.sections.contains_key(&SC_S), "{edge}: S is deleted");
            assert_eq!(
                state.children[&SC_P2],
                vec![SC_X, SC_Y],
                "{edge}: no ghost listing of S survives under P2"
            );
            assert!(
                mutations.iter().any(|m| matches!(
                    m,
                    StorageMutation::CreateTombstone {
                        section_id: SC_S,
                        ..
                    }
                )),
                "{edge}: S is tombstoned"
            );
            assert_live_and_snapshot_parity(edge, &live, &state);
        }
    }

    /// Legacy stored state can list a section under a parent other than its
    /// row's — here S, stored under P, is also listed mid-list under D. A move
    /// of S from P to D clears that listing before it appends, so S is listed
    /// once, last, at D's largest key (excluding its own) + 1.
    ///
    /// The pre-change engine found S already listed under D and left it
    /// mid-list.
    #[test]
    fn apply_delta_move_onto_a_legacy_stale_listing_at_the_destination_lands_last() {
        const P: u64 = 1;
        const D: u64 = 2;
        const X: u64 = 20;
        const Y: u64 = 21;
        const S: u64 = 30;

        let mut state = make_state();
        seed_section(&mut state, P, 0, "P", 0);
        seed_section(&mut state, D, 0, "D", 1);
        seed_section(&mut state, X, D, "X", 0);
        seed_section(&mut state, Y, D, "Y", 1);
        seed_section(&mut state, S, P, "S", 5);
        state.children.insert(0, vec![P, D]);
        state.children.insert(P, vec![S]);
        // The stale listing: S's row names P.
        state.children.insert(D, vec![X, S, Y]);
        let mut live = ClientModel::joined_from_snapshot(Reader::Ovlive, &state);

        apply_both(&mut state, &mut live, batch(vec![move_cmd(S, P, D, "S")]));

        assert_eq!(state.children[&D], vec![X, Y, S], "S is listed once, last");
        assert_eq!(
            state.sections[&S].section_order, 2,
            "S takes D's largest key among its other children + 1"
        );
        assert!(state.children[&P].is_empty());
        assert_live_and_snapshot_parity("move onto a stale listing", &live, &state);
    }

    // ── ReorderChildren: USD's list ordering, per child list ───────────────
    //
    // A reorder places its parent's children with OpenUSD's
    // `SdfApplyListOrdering` (`pxr/usd/sdf/listOp.cpp:674-726`, `:897-921` @
    // `ee47c679`), applied separately to each child list; each position keeps
    // its key (or the list is re-keyed `0..n-1` when its keys do not strictly
    // increase); ids that are not the parent's listed children are ignored.
    // The expected values below are worked by hand from USD's splice walk,
    // and the live [`ClientModel`] reaches them through its own transliteration
    // of it ([`usd_list_ordering`]).

    // `SdfSpecType` values the tests type their sections with
    // (`pxr/usd/sdf/types.h:71-89`).
    const T_UNKNOWN: u8 = 0;
    const T_ATTRIBUTE: u8 = 1;
    const T_PRIM: u8 = 6;
    const T_RELATIONSHIP: u8 = 8;

    /// The keys `parent`'s listed children carry, in list order.
    fn listed_keys(state: &ObjectState, parent: u64) -> Vec<(u64, u32)> {
        state
            .children
            .get(&parent)
            .map(|l| {
                l.iter()
                    .map(|id| (*id, state.sections[id].section_order))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// The keys a delta's `SetSection`s wrote, by id.
    fn rewritten_keys(mutations: &[StorageMutation]) -> Vec<(u64, u32)> {
        set_section_rows(mutations)
            .into_iter()
            .map(|(id, s)| (id, s.section_order))
            .collect()
    }

    /// `parent` (typed `parent_ty`) under the root, listing `children` — each
    /// `(id, type)` — keyed by list position `0..n-1`. Every row is at version
    /// 1 and the state at sequence 5, so a commit's version (6) tells a
    /// rewritten row from an untouched one.
    fn typed_parent_state(parent: u64, parent_ty: u8, children: &[(u64, u8)]) -> ObjectState {
        let mut state = make_state();
        state.sequence_num = 5;
        seed_section(&mut state, parent, 0, "P", 0);
        state
            .sections
            .get_mut(&parent)
            .expect("seeded")
            .section_type = parent_ty;
        state.children.insert(0, vec![parent]);
        for (position, (id, ty)) in children.iter().enumerate() {
            let key = u32::try_from(position).expect("a short test list");
            seed_section(&mut state, *id, parent, &format!("s{id}"), key);
            state.sections.get_mut(id).expect("seeded").section_type = *ty;
        }
        state
            .children
            .insert(parent, children.iter().map(|(id, _)| *id).collect());
        state
    }

    /// The client team's examples, on a prim's child prims keyed `0..n-1`: an unnamed
    /// child before the first named one stays in front, and each named child
    /// carries the unnamed ones that followed it.
    ///
    /// * `[A, B, C]` by `[C, A]` gives `[C, A, B]`: C carries nothing, A
    ///   carries B. Keys C0 A1 B2, so all three are rewritten.
    /// * `[A, B, C, D]` by `[D, B]` gives `[A, D, B, C]`: A is in front, D
    ///   carries nothing, B carries C. A keeps key 0 and gets no `SetSection`.
    ///
    /// Fails on the pre-change engine, which installed the list as the whole
    /// list (`[C, A]`, `[D, B]`), and on revision 2's slot fill (`[C, B, A]`,
    /// `[A, D, C, B]`).
    #[test]
    fn apply_reorder_children_partial_list_follows_sdf_apply_list_ordering() {
        const P: u64 = 1;
        const A: u64 = 10;
        const B: u64 = 11;
        const C: u64 = 12;
        const D: u64 = 13;

        for (label, children, order, expected, rewritten) in [
            (
                "[A, B, C] by [C, A]",
                vec![A, B, C],
                vec![C, A],
                vec![(C, 0), (A, 1), (B, 2)],
                vec![(A, 1), (B, 2), (C, 0)],
            ),
            (
                "[A, B, C, D] by [D, B]",
                vec![A, B, C, D],
                vec![D, B],
                vec![(A, 0), (D, 1), (B, 2), (C, 3)],
                vec![(B, 2), (C, 3), (D, 1)],
            ),
        ] {
            let typed: Vec<(u64, u8)> = children.iter().map(|id| (*id, T_PRIM)).collect();
            let mut state = typed_parent_state(P, T_PRIM, &typed);
            let mut live = ClientModel::joined_from_snapshot(Reader::Ovlive, &state);

            let mutations = apply_both(&mut state, &mut live, batch(vec![reorder_cmd(P, order)]));

            assert_eq!(
                listed_keys(&state, P),
                expected,
                "{label}: USD's order, each position keeping its key"
            );
            assert_eq!(
                rewritten_keys(&mutations),
                rewritten,
                "{label}: a SetSection for exactly the children that moved"
            );
            assert!(
                set_section_rows(&mutations)
                    .values()
                    .all(|s| s.last_version == state.sequence_num),
                "{label}: each rewritten row carries the reorder's version"
            );
            assert_live_and_snapshot_parity(label, &live, &state);
        }
    }

    /// OpenUSD's own `SdfApplyListOrdering` vectors (`testSdfListOp.py:166-189`
    /// @ `ee47c679`): ordering by `[1, 2, 3]` turns `[]`, `[2]`, `[2, 1]`,
    /// `[2, 4, 1]` and `[0, 2, 4, 1, 5, 3]` into `[]`, `[2]`, `[1, 2]`,
    /// `[1, 2, 4]` and `[0, 1, 5, 2, 4, 3]`. Run through the engine as one
    /// child list (item `v` is section `100 + v`, a prim under a prim), and
    /// through the model's reference, which must reproduce them too.
    ///
    /// Discriminated by: `apply_list_ordering` appending each named item's run
    /// without the unnamed items that follow it (`[2, 4, 1]` then gives
    /// `[4, 1, 2]`).
    #[test]
    fn apply_reorder_children_reproduces_openusd_ordering_vectors() {
        const P: u64 = 1;
        let id = |v: &u64| 100 + v;
        let order: Vec<u64> = vec![1, 2, 3];

        for (list, expected) in [
            (vec![], vec![]),
            (vec![2], vec![2]),
            (vec![2, 1], vec![1, 2]),
            (vec![2, 4, 1], vec![1, 2, 4]),
            (vec![0, 2, 4, 1, 5, 3], vec![0, 1, 5, 2, 4, 3]),
        ] {
            assert_eq!(
                usd_list_ordering(&list, &order),
                expected,
                "the model's reference reproduces USD's vector for {list:?}"
            );

            let typed: Vec<(u64, u8)> = list.iter().map(|v| (id(v), T_PRIM)).collect();
            let mut state = typed_parent_state(P, T_PRIM, &typed);
            apply_delta(
                &mut state,
                Some(batch(vec![reorder_cmd(P, order.iter().map(id).collect())])),
            )
            .expect("a reorder never rejects");
            assert_eq!(
                state.children.get(&P).cloned().unwrap_or_default(),
                expected.iter().map(id).collect::<Vec<_>>(),
                "the engine reproduces USD's vector for {list:?}"
            );
        }
    }

    /// A reorder orders each child list among the positions that list holds
    /// and leaves every other child list where it is (P a prim; keys follow
    /// list position):
    ///
    /// * `[A, p1, B, p2]` by `[B, A]` reorders the prims only — `[B, p1, A,
    ///   p2]`, re-keying B (0) and A (2);
    /// * `[p1, A, p2, B]` by `[p2, p1]` reorders the properties only —
    ///   `[p2, A, p1, B]`;
    /// * attributes and relationships share the properties list — `[A, r1,
    ///   p1]` by `[p1, r1]` gives `[A, p1, r1]`;
    /// * under a parent of type Unknown every child shares the one fallback
    ///   list, so the same shapes order as one list: `[B, p2, A, p1]` and
    ///   `[p2, B, p1, A]`.
    ///
    /// Fails on the pre-change engine, which installed the named list as the
    /// whole list and so dropped the other kind. The first two cases fail an
    /// engine ordering the whole list as one; the Unknown-parent cases fail one
    /// that ignores the fallback list (the named kind alone would move).
    #[test]
    fn apply_reorder_children_orders_each_child_list_in_its_own_positions() {
        const P: u64 = 1;
        const A: u64 = 10;
        const B: u64 = 11;
        const P1: u64 = 20;
        const P2: u64 = 21;
        const R1: u64 = 30;
        let interleaved = vec![
            (A, T_PRIM),
            (P1, T_ATTRIBUTE),
            (B, T_PRIM),
            (P2, T_ATTRIBUTE),
        ];
        let property_first = vec![
            (P1, T_ATTRIBUTE),
            (A, T_PRIM),
            (P2, T_ATTRIBUTE),
            (B, T_PRIM),
        ];

        for (label, parent_ty, children, order, expected, rewritten) in [
            (
                "prims under a prim",
                T_PRIM,
                interleaved.clone(),
                vec![B, A],
                vec![B, P1, A, P2],
                vec![(A, 2), (B, 0)],
            ),
            (
                "properties under a prim",
                T_PRIM,
                property_first.clone(),
                vec![P2, P1],
                vec![P2, A, P1, B],
                vec![(P1, 2), (P2, 0)],
            ),
            (
                "attributes and relationships share a list",
                T_PRIM,
                vec![(A, T_PRIM), (R1, T_RELATIONSHIP), (P1, T_ATTRIBUTE)],
                vec![P1, R1],
                vec![A, P1, R1],
                vec![(P1, 1), (R1, 2)],
            ),
            (
                "prims under an Unknown parent",
                T_UNKNOWN,
                interleaved,
                vec![B, A],
                vec![B, P2, A, P1],
                vec![(A, 2), (B, 0), (P1, 3), (P2, 1)],
            ),
            (
                "properties under an Unknown parent",
                T_UNKNOWN,
                property_first,
                vec![P2, P1],
                vec![P2, B, P1, A],
                vec![(A, 3), (B, 1), (P1, 2), (P2, 0)],
            ),
        ] {
            let mut state = typed_parent_state(P, parent_ty, &children);
            let mut live = ClientModel::joined_from_snapshot(Reader::Ovlive, &state);

            let mutations = apply_both(&mut state, &mut live, batch(vec![reorder_cmd(P, order)]));

            assert_eq!(state.children[&P], expected, "{label}");
            assert_eq!(
                rewritten_keys(&mutations),
                rewritten,
                "{label}: only the moved children are re-keyed, to their positions' keys"
            );
            assert_live_and_snapshot_parity(label, &live, &state);
        }
    }

    /// Every `(parent type, child type, child type)` triple over the twelve
    /// `SdfSpecType` values, so every row of the child-list table — and every
    /// pair it sends to the fallback list — is pinned.
    ///
    /// The parent lists `[x1, y1, x2, y2]`, the x's of the first child type and
    /// the y's of the second, keyed 0..3, and the reorder names all four in
    /// reverse, `[y2, x2, y1, x1]`. If the two types share a child list (by
    /// the model's own table, [`client_child_list`]) the whole list reverses:
    /// `[y2, x2, y1, x1]`. If they do not, each list reverses within its own
    /// positions: `[x2, y2, x1, y1]`.
    ///
    /// Discriminated by: moving one row of `child_list` to another list
    /// number (Attribute → Expression into the Connection list), and dropping
    /// a single-list row to the fallback (Mapper → MapperArg).
    #[test]
    fn apply_reorder_children_groups_every_type_pair_by_the_child_list_table() {
        const P: u64 = 1;
        const X1: u64 = 10;
        const Y1: u64 = 11;
        const X2: u64 = 12;
        const Y2: u64 = 13;

        let mut wrong = Vec::new();
        for parent_ty in 0u8..=11 {
            for x_ty in 0u8..=11 {
                for y_ty in 0u8..=11 {
                    let mut state = typed_parent_state(
                        P,
                        parent_ty,
                        &[(X1, x_ty), (Y1, y_ty), (X2, x_ty), (Y2, y_ty)],
                    );
                    apply_delta(
                        &mut state,
                        Some(batch(vec![reorder_cmd(P, vec![Y2, X2, Y1, X1])])),
                    )
                    .expect("a reorder never rejects");
                    let shared = client_child_list(Some(parent_ty), x_ty)
                        == client_child_list(Some(parent_ty), y_ty);
                    let expected = if shared {
                        vec![Y2, X2, Y1, X1]
                    } else {
                        vec![X2, Y2, X1, Y1]
                    };
                    if state.children[&P] != expected {
                        wrong.push((parent_ty, x_ty, y_ty, state.children[&P].clone()));
                    }
                }
            }
        }
        assert!(
            wrong.is_empty(),
            "(parent type, x type, y type, list) the engine grouped differently from the \
             child-list table: {wrong:?}"
        );
    }

    // ── Ignored ids ──────────────────────────────────────────────────────────

    /// P (5) under the root listing A, B and C (1, 2, 3) at keys 0, 1, 2. Every
    /// row is type 1 (Attribute), a pair with no USD child list, so the three
    /// share the fallback list.
    fn ignored_id_fixture() -> (ObjectState, [u64; 3]) {
        let mut state = make_state();
        seed_section(&mut state, 5, 0, "P", 0);
        seed_section(&mut state, 1, 5, "A", 0);
        seed_section(&mut state, 2, 5, "B", 1);
        seed_section(&mut state, 3, 5, "C", 2);
        state.children.insert(0, vec![5]);
        state.children.insert(5, vec![1, 2, 3]);
        (state, [1, 2, 3])
    }

    /// Apply `ReorderChildren(parent, named)` to one copy of `state` and the
    /// same reorder naming only `remaining` — the ids of `named` that are the
    /// parent's listed children, first occurrence each — to another, then
    /// assert what an ignored id must not change: both commit, advancing
    /// `sequence_num` by one; the two copies end in the same state and write
    /// the same rows; and every other parent's list, and every row parented
    /// elsewhere, is as it was. Returns the first copy.
    fn assert_reorder_ignores(
        label: &str,
        state: &ObjectState,
        parent: u64,
        named: Vec<u64>,
        remaining: Vec<u64>,
    ) -> ObjectState {
        assert_reorder_ignores_then(label, state, parent, named, remaining, Vec::new())
    }

    /// [`assert_reorder_ignores`], with `tail` applied after the reorder in the
    /// same delta on both copies.
    fn assert_reorder_ignores_then(
        label: &str,
        state: &ObjectState,
        parent: u64,
        named: Vec<u64>,
        remaining: Vec<u64>,
        tail: Vec<StructCommand>,
    ) -> ObjectState {
        let run = |list: Vec<u64>| {
            let mut copy = state.clone();
            let mut cmds = vec![reorder_cmd(parent, list)];
            cmds.extend(tail.iter().cloned());
            let mutations = apply_delta(&mut copy, Some(batch(cmds)))
                .unwrap_or_else(|e| panic!("{label}: the reorder must commit, got {e:?}"));
            (copy, mutations)
        };
        let (after, mutations) = run(named);
        let (reference, reference_mutations) = run(remaining);

        assert_eq!(
            after.sequence_num,
            state.sequence_num + 1,
            "{label}: the delta commits"
        );
        assert_eq!(
            structural_shape(&after),
            structural_shape(&reference),
            "{label}: the reorder equals the one over the ids that remain"
        );
        assert_eq!(
            set_section_rows(&mutations),
            set_section_rows(&reference_mutations),
            "{label}: and writes the same rows"
        );
        for (p, list) in &state.children {
            if *p != parent {
                assert_eq!(
                    after.children.get(p),
                    Some(list),
                    "{label}: parent {p}'s list is untouched"
                );
            }
        }
        for (id, row) in &state.sections {
            if row.parent_id != parent {
                assert_eq!(
                    after.sections.get(id),
                    Some(row),
                    "{label}: row {id}, parented elsewhere, is untouched"
                );
            }
        }
        after
    }

    /// Case (b): a reorder naming a child a peer's earlier delta deleted
    /// commits as if the child were not named. B is gone, so `[C, B, A]` acts
    /// as `[C, A]` over `[A, C]`, and the two keep their positions' keys (0
    /// and 2).
    ///
    /// Discriminated by: the early return described at
    /// `apply_reorder_children_with_an_unknown_section_id_ignores_it`.
    #[test]
    fn apply_reorder_children_naming_a_child_a_peer_deleted_ignores_it() {
        const P: u64 = 5;
        let (mut state, [a, b, c]) = ignored_id_fixture();
        apply_delta(&mut state, Some(batch(vec![delete_cmd(b)]))).expect("the peer's delete");

        let after = assert_reorder_ignores("deleted child", &state, P, vec![c, b, a], vec![c, a]);
        assert_eq!(listed_keys(&after, P), vec![(c, 0), (a, 2)]);
    }

    /// Case (e), policy 12: a legacy row naming P that P's list lost is not a
    /// listed child, so a reorder naming it ignores it and leaves it unlisted.
    ///
    /// The reorder alone would end the delta cross-map inconsistent (the row
    /// stays unlisted), tripping the debug audit, so the delta goes on to
    /// re-list L the way the engine does — a move onto its stored parent, which
    /// inserts it by its key (1). Where L lands tells what the reorder did:
    ///
    /// * ignored — the reorder leaves `[C, A, B]` keyed 0, 1, 2 and the move
    ///   puts L after A: `[C, A, L, B]`;
    /// * listed at its named place — `[C, L, A, B]`, which the move keeps;
    /// * appended after the list (revision 2) — `[C, A, B, L]`.
    #[test]
    fn apply_reorder_children_naming_a_legacy_unlisted_child_leaves_it_unlisted() {
        const P: u64 = 5;
        const L: u64 = 4;
        let (mut state, [a, b, c]) = ignored_id_fixture();
        seed_section(&mut state, L, P, "L", 1);
        let relist = move_cmd(L, 0, P, "L");

        let after = assert_reorder_ignores_then(
            "legacy unlisted child",
            &state,
            P,
            vec![c, L, a],
            vec![c, a],
            vec![relist],
        );
        assert_eq!(after.children[&P], vec![c, a, L, b]);
        assert_eq!(after.sections[&L].section_order, 1, "L keeps its key");
    }

    /// A reorder naming no listed child — only an unknown id, another parent's
    /// child and a repeat — is a no-op: no `SetChildren` for P, no
    /// `SetSection`, and the state is unchanged except `sequence_num`.
    ///
    /// Discriminated by: removing `apply_reorder_children`'s `order.is_empty()`
    /// return, which re-installs P's list and emits its `SetChildren`.
    #[test]
    fn apply_reorder_children_naming_only_ignored_ids_changes_nothing_but_the_sequence() {
        const P: u64 = 5;
        const Q: u64 = 9;
        const X: u64 = 7;
        const GHOST: u64 = 42;
        let (mut state, _) = ignored_id_fixture();
        seed_section(&mut state, Q, 0, "Q", 1);
        seed_section(&mut state, X, Q, "X", 0);
        state.children.insert(0, vec![P, Q]);
        state.children.insert(Q, vec![X]);

        let mut after = state.clone();
        let mutations = apply_delta(
            &mut after,
            Some(batch(vec![reorder_cmd(P, vec![GHOST, X, GHOST])])),
        )
        .expect("a reorder never rejects");

        assert!(
            mutations
                .iter()
                .all(|m| matches!(m, StorageMutation::SetObjectHeader { .. })),
            "only the header is written, got {mutations:?}"
        );
        let mut expected = state.clone();
        expected.sequence_num += 1;
        assert_eq!(after, expected, "nothing but the sequence changes");
    }

    // ── Keys ─────────────────────────────────────────────────────────────────

    /// Normal arm: on keys that strictly increase — here with gaps, 0 5 7 9 —
    /// each position keeps its key. `[A, B, C, D]` by `[D, B]` gives `[A, D, B,
    /// C]` keyed A0 D5 B7 C9: the key multiset is unchanged, A (unmoved) is
    /// not rewritten, and a create afterwards appends at the pre-reorder
    /// largest key + 1 (10).
    ///
    /// Discriminated by: forcing `reorder_layout` onto its dense `0..n-1` arm,
    /// which keys the list 0 1 2 3, rewrites C with key 3 and appends at 4.
    #[test]
    fn apply_reorder_children_keeps_each_positions_key_and_the_append_key() {
        const P: u64 = 1;
        const A: u64 = 10;
        const B: u64 = 11;
        const C: u64 = 12;
        const D: u64 = 13;
        const E: u64 = 14;

        let mut state = make_state();
        let mut live = ClientModel::new(Reader::Ovlive);
        apply_both(
            &mut state,
            &mut live,
            batch(vec![
                change_cmd(0, P, "P", 0),
                change_cmd(P, A, "A", 0),
                change_cmd(P, B, "B", 5),
                change_cmd(P, C, "C", 7),
                change_cmd(P, D, "D", 9),
            ]),
        );
        let mutations = apply_both(
            &mut state,
            &mut live,
            batch(vec![reorder_cmd(P, vec![D, B])]),
        );

        assert_eq!(
            listed_keys(&state, P),
            vec![(A, 0), (D, 5), (B, 7), (C, 9)],
            "each position keeps its key; the key multiset is unchanged"
        );
        assert_eq!(
            rewritten_keys(&mutations),
            vec![(B, 7), (C, 9), (D, 5)],
            "the unmoved A is not rewritten"
        );

        apply_both(&mut state, &mut live, batch(vec![create_cmd(P, E, "E")]));
        assert_eq!(
            state.sections[&E].section_order, 10,
            "the append key is the pre-reorder largest key + 1"
        );
        assert_eq!(state.children[&P], vec![A, D, B, C, E]);
        assert_live_and_snapshot_parity("position keys", &live, &state);
    }

    /// Degenerate arm: keys that do not strictly increase — every child at 0,
    /// as seeded siblings can arrive — are re-keyed `0..n-1` in list order, and
    /// a create afterwards appends at n.
    ///
    /// * `[C, A]` reorders to `[C, A, B]`, keyed 0 1 2: A and B are rewritten,
    ///   C keeps 0.
    /// * `[A]` moves nothing, but the list is still re-keyed: B and C, which
    ///   did not move, are rewritten to 1 and 2.
    ///
    /// Discriminated by: forcing `reorder_layout` onto its position-key arm,
    /// which leaves every key 0 and appends the create at 1.
    #[test]
    fn apply_reorder_children_re_keys_a_list_whose_keys_tie() {
        const P: u64 = 1;
        const A: u64 = 10;
        const B: u64 = 11;
        const C: u64 = 12;
        const D: u64 = 13;

        for (label, order, expected, rewritten) in [
            (
                "partial reorder",
                vec![C, A],
                vec![(C, 0), (A, 1), (B, 2)],
                vec![(A, 1), (B, 2)],
            ),
            (
                "nothing moves",
                vec![A],
                vec![(A, 0), (B, 1), (C, 2)],
                vec![(B, 1), (C, 2)],
            ),
        ] {
            let mut state = make_state();
            apply_delta(
                &mut state,
                Some(batch(vec![
                    change_cmd(0, P, "P", 0),
                    change_cmd(P, A, "A", 0),
                    change_cmd(P, B, "B", 0),
                    change_cmd(P, C, "C", 0),
                ])),
            )
            .expect("the seed applies");
            assert_eq!(
                state.children[&P],
                vec![A, B, C],
                "{label}: precondition, ties in arrival order"
            );
            // The live client joins after the seed: the C++ client library
            // reads tied orders as one index, which inserts each tied row
            // ahead of the last and reverses them, so no client applies this
            // upload's ties as the server stores them.
            let mut live = ClientModel::joined_from_snapshot(Reader::Ovlive, &state);

            let mutations = apply_both(&mut state, &mut live, batch(vec![reorder_cmd(P, order)]));
            assert_eq!(listed_keys(&state, P), expected, "{label}");
            assert_eq!(rewritten_keys(&mutations), rewritten, "{label}");

            apply_both(&mut state, &mut live, batch(vec![create_cmd(P, D, "D")]));
            assert_eq!(state.sections[&D].section_order, 3, "{label}: appends at n");
            assert_live_and_snapshot_parity(label, &live, &state);
        }
    }

    // ── Readers that rebuild the list ────────────────────────────────────────

    /// A snapshot-mode reconnector holding the list from before a partial
    /// reorder receives `plan_faithful_coalesced`'s existing rows: the
    /// children the reorder re-keyed — D, B and C, the ones that moved in
    /// `[A, B, C, D]` by `[D, B]` — and, because P now has a changed child,
    /// its unchanged child A too, so P's whole children sequence is restated.
    /// Applying them gives a reconnector of either reader the server's
    /// `[A, D, B, C]`.
    ///
    /// Discriminated by: dropping the re-key's `last_version` bump, which
    /// leaves no changed row, so P is not touched, the coalesced diff carries
    /// no modified row and the reconnector stays at `[A, B, C, D]`; and by
    /// restating the changed rows only, which leaves A out of the modified
    /// set.
    #[test]
    fn apply_reorder_children_partial_reorder_reaches_a_snapshot_mode_reconnector() {
        const P: u64 = 1;
        const A: u64 = 10;
        const B: u64 = 11;
        const C: u64 = 12;
        const D: u64 = 13;

        let mut state = make_state();
        apply_delta(
            &mut state,
            Some(batch(vec![
                typed_change_cmd(0, P, "P", T_PRIM, 0),
                typed_change_cmd(P, A, "A", T_PRIM, 0),
                typed_change_cmd(P, B, "B", T_PRIM, 1),
                typed_change_cmd(P, C, "C", T_PRIM, 2),
                typed_change_cmd(P, D, "D", T_PRIM, 3),
            ])),
        )
        .expect("seed");
        let have_version = state.sequence_num;
        let reconnectors: Vec<ClientModel> = READERS
            .iter()
            .map(|reader| ClientModel::joined_from_snapshot(*reader, &state))
            .collect();

        apply_delta(&mut state, Some(batch(vec![reorder_cmd(P, vec![D, B])])))
            .expect("a reorder never rejects");
        assert_eq!(state.children[&P], vec![A, D, B, C]);

        let crate::domain::diff::DiffPlan::Incremental {
            new_sections,
            modified_sections,
            deleted_sections,
            ..
        } = crate::domain::diff::DiffEngine::plan_faithful_coalesced(&state, &[], have_version)
            .expect("the coalesced diff must plan")
        else {
            panic!("a reconnector behind the head plans Incremental");
        };
        assert!(new_sections.is_empty() && deleted_sections.is_empty());
        let mut modified: Vec<u64> = modified_sections.iter().map(|s| s.section_id).collect();
        modified.sort_unstable();
        assert_eq!(
            modified,
            vec![A, B, C, D],
            "the re-keyed children (the moved ones) and their unchanged sibling A: P's whole \
             children sequence is restated"
        );

        let catch_up = OwnedDelta {
            is_diff: true,
            base_version: have_version,
            struct_commands: modified_sections.iter().map(section_as_change).collect(),
            ..Default::default()
        };
        for mut reconnector in reconnectors {
            reconnector.apply(&catch_up);
            assert_eq!(
                reconnector.lists(),
                server_lists(&state),
                "{:?}: the reconnector reaches the server's order",
                reconnector.reader
            );
        }
    }

    // ── Reconnectors of either reader, over gapped keys (b.ze1) ─────────────
    //
    // A reconnector's keys can have gaps the server's positions do not: a
    // client never renumbers on a delete, and a catch-up retracts after it
    // restates. The Kit client reads a catch-up order as a key, displacing a
    // sibling that holds it; the C++ client library reads it as an index and
    // renumbers. `plan_faithful_coalesced` restates every listed child of a
    // parent with a new or changed child at its position, which both readings
    // rebuild into the server's order. Each test folds the plan, in the wire
    // order `build_delta` (`src/storage/diff_read.rs`) sends it, into a
    // reconnector of each reader and compares every USD child list with the
    // server's.
    //
    // Fails on the builder that restated only the changed children: the Kit
    // reader misplaces a child in each case (measured; the C++ client
    // library's reader, which reads positions as indexes, is traced by hand
    // to the server's order there, and the loop stops at the Kit reader).

    /// A `CreateSection` with an explicit `SdfSpecType`.
    fn typed_create_cmd(parent_id: u64, section_id: u64, name: &str, ty: u8) -> StructCommand {
        StructCommand::CreateSection(OodfCreateSection {
            parent_id,
            section_id,
            section_name: name.to_string(),
            section_type: ty,
        })
    }

    /// The tombstones a delta's mutations wrote, as storage keeps them.
    fn tombstones_written(mutations: &[StorageMutation]) -> Vec<(u64, u64, u64)> {
        mutations
            .iter()
            .filter_map(|m| match m {
                StorageMutation::CreateTombstone {
                    section_id,
                    first_version,
                    last_version,
                } => Some((*section_id, *first_version, *last_version)),
                _ => None,
            })
            .collect()
    }

    /// The snapshot-mode catch-up for a reconnector at `have_version`, in
    /// `build_delta`'s wire order: new sections, then existing ones, then
    /// retractions.
    fn coalesced_catch_up(
        state: &ObjectState,
        tombstones: &[(u64, u64, u64)],
        have_version: u64,
    ) -> OwnedDelta {
        let crate::domain::diff::DiffPlan::Incremental {
            new_sections,
            modified_sections,
            deleted_sections,
            ..
        } = crate::domain::diff::DiffEngine::plan_faithful_coalesced(
            state,
            tombstones,
            have_version,
        )
        .expect("the coalesced diff must plan")
        else {
            panic!("a reconnector behind the head plans Incremental");
        };
        let mut struct_commands: Vec<StructCommand> = new_sections
            .iter()
            .chain(&modified_sections)
            .map(section_as_change)
            .collect();
        struct_commands.extend(deleted_sections.iter().map(|(section_id, _, _)| {
            StructCommand::DiffDeleteSection(oodf_ffi::CommandDiffDeleteSection {
                section_id: *section_id,
            })
        }));
        OwnedDelta {
            is_diff: true,
            base_version: have_version,
            struct_commands,
            ..Default::default()
        }
    }

    /// Apply `delta` to the server and to every reconnector still following
    /// live.
    fn apply_to_all(
        state: &mut ObjectState,
        reconnectors: &mut [ClientModel],
        delta: OwnedDelta,
    ) -> Vec<StorageMutation> {
        for reconnector in reconnectors.iter_mut() {
            reconnector.apply(&delta);
        }
        apply_delta(state, Some(delta)).expect("delta must apply")
    }

    /// Fold `catch_up` into each reconnector and require the server's order in
    /// every child list.
    fn assert_reconnectors_reach_the_server(
        label: &str,
        reconnectors: Vec<ClientModel>,
        catch_up: &OwnedDelta,
        state: &ObjectState,
    ) {
        let server = server_child_lists(state);
        for mut reconnector in reconnectors {
            reconnector.apply(catch_up);
            assert_eq!(
                reconnector.child_lists(),
                server,
                "{label} / {:?}: the reconnector must hold the server's order in every child \
                 list",
                reconnector.reader
            );
        }
    }

    /// The ticket's repro. P's prims are A, B, C, X; while the reconnector is
    /// away B is deleted and D created, so the server holds `[A, C, X, D]`
    /// with D at key 4. Restating only D at position 3 makes the Kit reader
    /// move X, which still holds key 3, to the end: `[A, C, D, X]`.
    #[test]
    fn plan_faithful_coalesced_delete_then_create_reaches_a_reconnector_of_either_reader() {
        const P: u64 = 1;
        const A: u64 = 10;
        const B: u64 = 11;
        const C: u64 = 12;
        const X: u64 = 13;
        const D: u64 = 14;

        let mut state = make_state();
        apply_delta(
            &mut state,
            Some(batch(vec![
                typed_change_cmd(0, P, "P", T_PRIM, 0),
                typed_change_cmd(P, A, "A", T_PRIM, 0),
                typed_change_cmd(P, B, "B", T_PRIM, 1),
                typed_change_cmd(P, C, "C", T_PRIM, 2),
                typed_change_cmd(P, X, "X", T_PRIM, 3),
            ])),
        )
        .expect("seed");
        let have_version = state.sequence_num;
        let reconnectors: Vec<ClientModel> = READERS
            .iter()
            .map(|reader| ClientModel::joined_from_snapshot(*reader, &state))
            .collect();

        let mut tombstones =
            tombstones_written(&apply_delta(&mut state, Some(batch(vec![delete_cmd(B)]))).unwrap());
        tombstones.extend(tombstones_written(
            &apply_delta(
                &mut state,
                Some(batch(vec![typed_create_cmd(P, D, "D", T_PRIM)])),
            )
            .unwrap(),
        ));
        assert_eq!(state.children[&P], vec![A, C, X, D], "precondition");

        assert_reconnectors_reach_the_server(
            "delete then create",
            reconnectors,
            &coalesced_catch_up(&state, &tombstones, have_version),
            &state,
        );
    }

    /// An interleaved prim/attribute parent whose keys have gaps on the
    /// reconnector: P holds A x G B y F H D z (prims A G B F H D, attributes
    /// x y z), and the reconnector follows the deletes of G, F and H live, so
    /// it keys A0 x1 B3 y4 D7 z8 while the server's positions are 0..5. A prim
    /// C is then created (server prims `[A, B, D, C]`). Restating only C at
    /// position 6 lands it between B (3) and D (7) for the Kit reader:
    /// `[A, B, C, D]`.
    #[test]
    fn plan_faithful_coalesced_create_under_a_gapped_interleaved_parent_reaches_either_reader() {
        const P: u64 = 1;
        let prims = [
            (10, "A"),
            (12, "G"),
            (13, "B"),
            (15, "F"),
            (16, "H"),
            (17, "D"),
        ];
        let attrs = [(11, "x"), (14, "y"), (18, "z")];
        let mut seed: Vec<(u64, &str, u8)> = prims
            .iter()
            .map(|(id, n)| (*id, *n, T_PRIM))
            .chain(attrs.iter().map(|(id, n)| (*id, *n, T_ATTRIBUTE)))
            .collect();
        seed.sort_unstable_by_key(|(id, _, _)| *id);
        let mut cmds = vec![typed_change_cmd(0, P, "P", T_PRIM, 0)];
        for (key, (id, name, ty)) in seed.iter().enumerate() {
            cmds.push(typed_change_cmd(P, *id, name, *ty, key as u32));
        }

        let mut state = make_state();
        apply_delta(&mut state, Some(batch(cmds))).expect("seed");
        let mut reconnectors: Vec<ClientModel> = READERS
            .iter()
            .map(|reader| ClientModel::joined_from_snapshot(*reader, &state))
            .collect();
        for gone in [12, 15, 16] {
            apply_to_all(&mut state, &mut reconnectors, batch(vec![delete_cmd(gone)]));
        }
        let have_version = state.sequence_num;

        apply_delta(
            &mut state,
            Some(batch(vec![typed_create_cmd(P, 19, "C", T_PRIM)])),
        )
        .expect("the create applies");
        assert_eq!(
            state.children[&P],
            vec![10, 11, 13, 14, 17, 18, 19],
            "precondition"
        );

        assert_reconnectors_reach_the_server(
            "create under a gapped interleaved parent",
            reconnectors,
            &coalesced_catch_up(&state, &[], have_version),
            &state,
        );
    }

    /// A partial reorder after a create, over gapped keys. The reconnector
    /// keys P's prims A0 B2 C3 E5 (it followed the deletes of X and Y live).
    /// While it is away D is created (key 6) and a writer that has not seen D
    /// reorders `[C, A, B, E]`: the server holds `[C, A, B, E, D]`, re-keying
    /// C, A and B only. Restating D at 4 and C, A, B at 0, 1, 2 without E
    /// leaves the Kit reader with E (key 5) after D: `[C, A, B, D, E]`.
    #[test]
    fn plan_faithful_coalesced_partial_reorder_after_a_create_reaches_either_reader() {
        const P: u64 = 1;
        const A: u64 = 10;
        const X: u64 = 11;
        const B: u64 = 12;
        const C: u64 = 13;
        const Y: u64 = 14;
        const E: u64 = 15;
        const D: u64 = 16;

        let mut state = make_state();
        apply_delta(
            &mut state,
            Some(batch(vec![
                typed_change_cmd(0, P, "P", T_PRIM, 0),
                typed_change_cmd(P, A, "A", T_PRIM, 0),
                typed_change_cmd(P, X, "X", T_PRIM, 1),
                typed_change_cmd(P, B, "B", T_PRIM, 2),
                typed_change_cmd(P, C, "C", T_PRIM, 3),
                typed_change_cmd(P, Y, "Y", T_PRIM, 4),
                typed_change_cmd(P, E, "E", T_PRIM, 5),
            ])),
        )
        .expect("seed");
        let mut reconnectors: Vec<ClientModel> = READERS
            .iter()
            .map(|reader| ClientModel::joined_from_snapshot(*reader, &state))
            .collect();
        for gone in [X, Y] {
            apply_to_all(&mut state, &mut reconnectors, batch(vec![delete_cmd(gone)]));
        }
        let have_version = state.sequence_num;

        apply_delta(
            &mut state,
            Some(batch(vec![typed_create_cmd(P, D, "D", T_PRIM)])),
        )
        .expect("the create applies");
        apply_delta(
            &mut state,
            Some(batch(vec![reorder_cmd(P, vec![C, A, B, E])])),
        )
        .expect("a reorder never rejects");
        assert_eq!(state.children[&P], vec![C, A, B, E, D], "precondition");

        assert_reconnectors_reach_the_server(
            "partial reorder after a create",
            reconnectors,
            &coalesced_catch_up(&state, &[], have_version),
            &state,
        );
    }

    // ── Legacy non-child entries ─────────────────────────────────────────────

    /// A reorder over a stored list holding entries that are not P's children
    /// — a ghost G with no row, a stale listing of X (whose row, and list,
    /// are Q's) and a repeat of A — drops them before ordering:
    ///
    /// * a partial list `[C, A]` gives `[C, A, B]`, strictly increasing (C0
    ///   A1 B2); Q's list and X's row are untouched and X gets no
    ///   `SetSection`; and the delta ends cross-map consistent, so the
    ///   delta-boundary audits pass;
    /// * a list naming every listed child of one child list — here P is a prim
    ///   and A, B, C its prims, with an attribute p1 among them — leaves that
    ///   child list exactly in the list's order, `[C, A, B]`, around p1.
    ///
    /// After each, a client joining from a fresh snapshot and the checkpoint
    /// round trip hold the same order as the server.
    ///
    /// Discriminated by: removing the working list's parent filter, which
    /// keeps X listed under P (the reverse audit then fires); and dropping the
    /// re-key's `section_order` write, after which the round trip sorts by the
    /// old keys and rebuilds `[A, B, C]`.
    #[test]
    fn apply_reorder_children_over_legacy_non_child_entries_drops_them() {
        const P: u64 = 5;
        const Q: u64 = 9;
        const X: u64 = 7;
        const G: u64 = 42;
        const P1: u64 = 20;

        let legacy = |state: &mut ObjectState, list: Vec<u64>| {
            seed_section(state, Q, 0, "Q", 1);
            seed_section(state, X, Q, "X", 0);
            state.children.insert(0, vec![P, Q]);
            state.children.insert(Q, vec![X]);
            state.children.insert(P, list);
        };

        let (mut state, [a, b, c]) = ignored_id_fixture();
        legacy(&mut state, vec![a, G, X, b, a, c]);
        let before = state.clone();
        let mutations = apply_delta(&mut state, Some(batch(vec![reorder_cmd(P, vec![c, a])])))
            .expect("a reorder never rejects");
        assert_eq!(
            listed_keys(&state, P),
            vec![(c, 0), (a, 1), (b, 2)],
            "the non-child entries are dropped and the list is strictly increasing"
        );
        assert_eq!(state.children[&Q], vec![X], "Q's list is untouched");
        assert_eq!(
            state.sections[&X], before.sections[&X],
            "X's row is untouched"
        );
        assert!(
            !set_section_rows(&mutations).contains_key(&X),
            "X gets no SetSection"
        );
        assert_snapshot_and_round_trip_parity("partial list over legacy entries", &state);

        let mut state = typed_parent_state(
            P,
            T_PRIM,
            &[(a, T_PRIM), (P1, T_ATTRIBUTE), (b, T_PRIM), (c, T_PRIM)],
        );
        legacy(&mut state, vec![a, G, P1, X, b, a, c]);
        apply_delta(&mut state, Some(batch(vec![reorder_cmd(P, vec![c, a, b])])))
            .expect("a reorder never rejects");
        assert_eq!(
            state.children[&P],
            vec![c, P1, a, b],
            "the prims are exactly in the list's order, around the attribute"
        );
        assert_snapshot_and_round_trip_parity("whole child list over legacy entries", &state);
    }

    /// The readers that rebuild a list from stored state agree with it: a
    /// client of either reader joining from a fresh snapshot, and the
    /// checkpoint round trip (which places rows by their stored keys), hold the
    /// server's child order under every parent.
    fn assert_snapshot_and_round_trip_parity(label: &str, state: &ObjectState) {
        let stored = server_lists(state);
        for reader in READERS {
            assert_eq!(
                ClientModel::joined_from_snapshot(reader, state).lists(),
                stored,
                "{label} / {reader:?}: a client joining from a fresh snapshot must hold the \
                 server's order"
            );
        }
        assert_eq!(
            server_lists(&full_content_round_trip(state)),
            stored,
            "{label}: the checkpoint round trip must rebuild the server's order"
        );
    }

    // ═════════════════════════════════════════════════════════════════════════
    // b.a4k — orphans and adoption (SDD §10.4)
    //
    // A row whose parent has no row is an orphan, listed nowhere: no children
    // list exists under an id with no row. When the parent is born — by a
    // `CreateSection` or a new `DiffChangeSection` row — every row naming it
    // joins its list in `(section_order, section_id)` order before the delta's
    // next command, and each adopted row is re-stamped with the delta's
    // version. The start states below are the read model's shape: orphan rows
    // and no list under their parent, because every snapshot read leaves out a
    // stored list under a parent with no row.
    // ═════════════════════════════════════════════════════════════════════════

    /// The id every orphan below names until it is born.
    const A4K_X: u64 = 9;

    /// Three orphans under [`A4K_X`] whose `(key, id)` order differs from
    /// their id order and includes a key tie: `(2, 20)`, `(2, 30)`, `(5, 10)`.
    /// The state's sequence is 3, so a delta applied to it runs at version 4
    /// and every seeded row carries version 1.
    fn a4k_orphan_state() -> ObjectState {
        let mut state = make_state();
        state.sequence_num = 3;
        seed_section(&mut state, 10, A4K_X, "o10", 5);
        seed_section(&mut state, 20, A4K_X, "o20", 2);
        seed_section(&mut state, 30, A4K_X, "o30", 2);
        state
    }

    /// The adopted list of [`a4k_orphan_state`], in `(key, id)` order.
    const A4K_ADOPTED: [u64; 3] = [20, 30, 10];

    /// The two commands that give [`A4K_X`] a row, each placing it under the
    /// root. Every test below that covers a birth runs once per command, as
    /// its own test, so each birth passes or fails on its own.
    fn a4k_birth_by_create() -> StructCommand {
        create_cmd(0, A4K_X, "X")
    }

    fn a4k_birth_by_change() -> StructCommand {
        change_cmd(0, A4K_X, "X", 0)
    }

    /// Every children-list mutation naming `parent`.
    fn children_mutations_for(mutations: &[StorageMutation], parent: u64) -> usize {
        mutations
            .iter()
            .filter(|m| {
                matches!(m,
                    StorageMutation::SetChildren { section_id, .. }
                    | StorageMutation::DeleteChildrenList { section_id }
                    if *section_id == parent)
            })
            .count()
    }

    /// How many `SetSection`s in `mutations` write `id`.
    fn set_section_count(mutations: &[StorageMutation], id: u64) -> usize {
        mutations
            .iter()
            .filter(|m| matches!(m, StorageMutation::SetSection { section, .. } if section.section_id == id))
            .count()
    }

    /// T1. A new `DiffChangeSection` row under a parent with no row keeps no
    /// list under that parent and emits no children-list mutation for it; the
    /// row itself is written with its parent and key.
    ///
    /// Discriminated by: removing `insert_child_into_parent`'s missing-parent
    /// early return, which materializes `state.children[X]` and writes
    /// `SetChildren(X, …)` (in a debug build the delta-boundary cross-map
    /// check's third half fires first, on the same list).
    #[test]
    fn apply_delta_diff_change_section_new_row_under_a_missing_parent_keeps_no_list() {
        const O: u64 = 1;

        let mut state = make_state();
        let (mutations, _) = apply_counted(&mut state, batch(vec![change_cmd(A4K_X, O, "O", 5)]));
        assert!(
            !state.children.contains_key(&A4K_X),
            "new row: no list under the missing parent, got {:?}",
            state.children
        );
        assert_eq!(
            children_mutations_for(&mutations, A4K_X),
            0,
            "new row: no children-list mutation names the missing parent"
        );
        let row = &state.sections[&O];
        assert_eq!(
            (row.parent_id, row.section_order),
            (A4K_X, 5),
            "new row: the orphan keeps its parent and key"
        );
    }

    /// T1, the reparent half: a live row a `DiffChangeSection` moves under a
    /// parent with no row leaves its old parent's list and joins no list; no
    /// children-list mutation names the missing parent.
    ///
    /// Discriminated by: the same early return as the new-row half.
    #[test]
    fn apply_delta_diff_change_section_reparent_under_a_missing_parent_keeps_no_list() {
        const O: u64 = 1;
        const P: u64 = 2;

        let mut state = make_state();
        seed_section(&mut state, P, 0, "P", 0);
        seed_section(&mut state, O, P, "O", 0);
        state.children.insert(0, vec![P]);
        state.children.insert(P, vec![O]);
        let (mutations, _) = apply_counted(&mut state, batch(vec![change_cmd(A4K_X, O, "O", 3)]));
        assert!(
            !state.children.contains_key(&A4K_X),
            "reparent: no list under the missing parent, got {:?}",
            state.children
        );
        assert_eq!(
            children_mutations_for(&mutations, A4K_X),
            0,
            "reparent: no children-list mutation names the missing parent"
        );
        assert_eq!(
            set_children_writes(&mutations),
            vec![(P, Vec::new())],
            "reparent: the old parent's list is rewritten without the row"
        );
        let row = &state.sections[&O];
        assert_eq!(
            (row.parent_id, row.section_order),
            (A4K_X, 3),
            "reparent: the orphan carries its new parent and key"
        );
    }

    /// T2. Each command that gives a row to an id adopts the rows naming it:
    /// the new list holds them in `(key, id)` order, and each adopted row
    /// gets exactly one `SetSection`, with `last_version` = the delta's
    /// version and its parent and key unchanged. One test per birth command.
    ///
    /// Discriminated by: removing the `adopt_orphans` call (the list is
    /// absent; in a debug build the forward cross-map check fires first,
    /// because the orphans now have a live parent that does not list them),
    /// and removing the `last_version` rewrite (no `SetSection` for the
    /// adopted rows).
    #[test]
    fn apply_delta_create_section_birth_adopts_its_orphans_in_key_then_id_order() {
        assert_birth_adopts_its_orphans_in_key_then_id_order(
            "CreateSection",
            a4k_birth_by_create(),
        );
    }

    #[test]
    fn apply_delta_diff_change_section_birth_adopts_its_orphans_in_key_then_id_order() {
        assert_birth_adopts_its_orphans_in_key_then_id_order(
            "DiffChangeSection",
            a4k_birth_by_change(),
        );
    }

    fn assert_birth_adopts_its_orphans_in_key_then_id_order(label: &str, birth: StructCommand) {
        let mut state = a4k_orphan_state();
        let before = state.clone();
        let (mutations, _) = apply_counted(&mut state, batch(vec![birth]));
        let new_version = state.sequence_num;
        assert_eq!(new_version, 4, "{label}: the delta runs at version 4");

        assert_eq!(
            state.children.get(&A4K_X),
            Some(&A4K_ADOPTED.to_vec()),
            "{label}: the orphans are listed in (key, id) order"
        );
        assert_eq!(
            set_children_writes(&mutations),
            vec![(0, vec![A4K_X]), (A4K_X, A4K_ADOPTED.to_vec())],
            "{label}: the new list is written once, beside the root's"
        );
        for id in A4K_ADOPTED {
            let old = &before.sections[&id];
            let row = &state.sections[&id];
            assert_eq!(
                (row.parent_id, row.section_order, row.first_version),
                (old.parent_id, old.section_order, old.first_version),
                "{label}: adopted row {id} keeps its parent, key and first version"
            );
            assert_eq!(
                row.last_version, new_version,
                "{label}: adopted row {id} is re-stamped with the delta's version"
            );
            assert_eq!(
                set_section_count(&mutations, id),
                1,
                "{label}: adopted row {id} gets exactly one SetSection"
            );
            assert_eq!(
                set_section_rows(&mutations)[&id],
                *row,
                "{label}: the SetSection carries the re-stamped row"
            );
        }
    }

    /// T2, second half: a row the same delta already wrote before its parent
    /// was born is adopted without a second `SetSection` — its own write
    /// already carries the delta's version and the current row. One test per
    /// birth command.
    #[test]
    fn apply_delta_create_section_adopting_a_row_written_earlier_in_the_delta_emits_no_second_set_section(
    ) {
        assert_adopting_an_in_delta_row_emits_no_second_set_section(
            "CreateSection",
            a4k_birth_by_create(),
        );
    }

    #[test]
    fn apply_delta_diff_change_section_adopting_a_row_written_earlier_in_the_delta_emits_no_second_set_section(
    ) {
        assert_adopting_an_in_delta_row_emits_no_second_set_section(
            "DiffChangeSection",
            a4k_birth_by_change(),
        );
    }

    fn assert_adopting_an_in_delta_row_emits_no_second_set_section(
        label: &str,
        birth: StructCommand,
    ) {
        const O: u64 = 40;
        let mut state = make_state();
        let (mutations, _) =
            apply_counted(&mut state, batch(vec![change_cmd(A4K_X, O, "o", 1), birth]));
        assert_eq!(state.children.get(&A4K_X), Some(&vec![O]), "{label}");
        assert_eq!(
            set_section_count(&mutations, O),
            1,
            "{label}: only the row's own write"
        );
    }

    /// T3. Adoption runs before the delta's next command, so a command that
    /// builds the new parent's list in the same delta acts on the adopted
    /// list (b461447a's "known remaining case"):
    ///
    /// * `CreateSection` under X appends after the largest adopted key (5),
    ///   at key 6;
    /// * `DiffChangeSection` under X lands at its explicit key, 3 — after the
    ///   two key-2 rows, before the key-5 row;
    /// * `MoveSection` from P into X appends, at key 6;
    /// * `ReorderChildren(X, [10, 20])` orders the adopted list by USD's
    ///   rule: `[10, 20, 30]`, re-keyed `0..2` because the old keys tie.
    ///
    /// Discriminated by: removing the adoption (the orphans stay unlisted
    /// under a live parent; in a debug build the forward cross-map check
    /// fires), and by breaking its key ties in descending id order (the
    /// `CreateSection` case lists `[30, 20, 10, 50]`).
    ///
    /// One test per birth command × follow-up command, generated below.
    fn assert_adoption_precedes_the_next_command(
        label: &str,
        birth: StructCommand,
        follow: StructCommand,
        expected: &[u64],
    ) {
        let mut state = a4k_orphan_state();
        seed_section(&mut state, A4K_P, 0, "P", 0);
        seed_section(&mut state, A4K_M, A4K_P, "m", 0);
        state.children.insert(0, vec![A4K_P]);
        state.children.insert(A4K_P, vec![A4K_M]);

        apply_counted(&mut state, batch(vec![birth, follow]));

        assert_eq!(
            state.children.get(&A4K_X).map(Vec::as_slice),
            Some(expected),
            "{label}: the command acts on the adopted list"
        );
        assert_lists_sorted(label, &state);
    }

    /// A row the follow-up command creates under X.
    const A4K_N: u64 = 50;
    /// A live row under [`A4K_P`] that the follow-up move takes into X.
    const A4K_M: u64 = 60;
    /// A live top-level section.
    const A4K_P: u64 = 2;

    macro_rules! a4k_same_delta_tests {
        ($($name:ident: $birth:expr, $follow:expr => $expected:expr;)*) => {
            $(
                #[test]
                fn $name() {
                    assert_adoption_precedes_the_next_command(
                        stringify!($name),
                        $birth,
                        $follow,
                        &$expected,
                    );
                }
            )*
        };
    }

    a4k_same_delta_tests! {
        apply_delta_create_section_birth_then_create_under_it_appends_after_the_adopted_rows:
            a4k_birth_by_create(), create_cmd(A4K_X, A4K_N, "n") => [20, 30, 10, A4K_N];
        apply_delta_diff_change_section_birth_then_create_under_it_appends_after_the_adopted_rows:
            a4k_birth_by_change(), create_cmd(A4K_X, A4K_N, "n") => [20, 30, 10, A4K_N];
        apply_delta_create_section_birth_then_diff_change_under_it_lands_among_the_adopted_rows:
            a4k_birth_by_create(), change_cmd(A4K_X, A4K_N, "n", 3) => [20, 30, A4K_N, 10];
        apply_delta_diff_change_section_birth_then_diff_change_under_it_lands_among_the_adopted_rows:
            a4k_birth_by_change(), change_cmd(A4K_X, A4K_N, "n", 3) => [20, 30, A4K_N, 10];
        apply_delta_create_section_birth_then_move_into_it_appends_after_the_adopted_rows:
            a4k_birth_by_create(), move_cmd(A4K_M, A4K_P, A4K_X, "m") => [20, 30, 10, A4K_M];
        apply_delta_diff_change_section_birth_then_move_into_it_appends_after_the_adopted_rows:
            a4k_birth_by_change(), move_cmd(A4K_M, A4K_P, A4K_X, "m") => [20, 30, 10, A4K_M];
        apply_delta_create_section_birth_then_reorder_of_it_orders_the_adopted_rows:
            a4k_birth_by_create(), reorder_cmd(A4K_X, vec![10, 20]) => [10, 20, 30];
        apply_delta_diff_change_section_birth_then_reorder_of_it_orders_the_adopted_rows:
            a4k_birth_by_change(), reorder_cmd(A4K_X, vec![10, 20]) => [10, 20, 30];
    }

    /// T3, the keys: the append after a birth takes the largest adopted key
    /// + 1, and the reorder re-keys the tied adopted list `0..n-1`.
    #[test]
    fn apply_delta_adoption_then_append_and_reorder_write_the_expected_keys() {
        let mut state = a4k_orphan_state();
        apply_counted(
            &mut state,
            batch(vec![
                create_cmd(0, A4K_X, "X"),
                create_cmd(A4K_X, A4K_N, "n"),
            ]),
        );
        assert_eq!(
            state.sections[&A4K_N].section_order, 6,
            "append after key 5"
        );
        let mut state = a4k_orphan_state();
        apply_counted(
            &mut state,
            batch(vec![
                create_cmd(0, A4K_X, "X"),
                reorder_cmd(A4K_X, vec![10, 20]),
            ]),
        );
        assert_eq!(
            listed_keys(&state, A4K_X),
            vec![(10, 0), (20, 1), (30, 2)],
            "the reorder re-keys the tied adopted list 0..n-1"
        );
    }

    /// T4. Adoption's result does not depend on how a client split its
    /// commands into deltas: one delta and one delta per command give the
    /// same rows and lists. Covers an orphan stored before the delta (40), two
    /// made inside it before the birth (30, 10), a key tie (30 and 40 at key
    /// 2) and keys out of id order, then an explicit-order insert and an
    /// append under the new parent.
    ///
    /// Discriminated by: breaking the adoption's key ties in descending id
    /// order (measured: `[40, 30, 20, 10, 50]`, caught by the explicit list
    /// below — the split and batched runs agree with each other, equally
    /// wrong). Adopting in plain id order is not a defect it can see: the
    /// inserts place by key, so only the tie order matters, and ascending id
    /// is the specified tie order. One test per birth command.
    #[test]
    fn apply_delta_create_section_adoption_matches_one_command_per_delta() {
        assert_adoption_matches_one_command_per_delta("CreateSection", a4k_birth_by_create());
    }

    #[test]
    fn apply_delta_diff_change_section_adoption_matches_one_command_per_delta() {
        assert_adoption_matches_one_command_per_delta("DiffChangeSection", a4k_birth_by_change());
    }

    fn assert_adoption_matches_one_command_per_delta(label: &str, birth: StructCommand) {
        let start = {
            let mut state = make_state();
            seed_section(&mut state, 40, A4K_X, "o40", 2);
            state
        };
        let delta = batch(vec![
            change_cmd(A4K_X, 30, "o30", 2),
            change_cmd(A4K_X, 10, "o10", 5),
            birth,
            change_cmd(A4K_X, 20, "o20", 2),
            create_cmd(A4K_X, 50, "o50"),
        ]);

        let mut batched = start.clone();
        apply_delta(&mut batched, Some(delta.clone())).expect("batched delta must apply");
        let mut split = start.clone();
        apply_one_command_per_delta(&mut split, &delta);

        assert_eq!(
            structural_shape(&batched),
            structural_shape(&split),
            "{label}: one delta and one delta per command must agree"
        );
        assert_eq!(
            batched.children.get(&A4K_X),
            Some(&vec![30, 40, 20, 10, 50]),
            "{label}: adopted (2,30) (2,40) (5,10); then 20 at key 2 after both \
             key-2 rows, and 50 appended"
        );
    }

    /// T5. Creating a parent and deleting it in the same delta takes its
    /// adopted orphan with it — tombstone written, field and time sample
    /// purged — and so does a cascade from an ancestor of the just-created
    /// parent. Then the ticket's replay: re-creating the id afterwards finds
    /// nothing to adopt, and the next structural delta applies.
    ///
    /// Discriminated by: removing the adoption — the cascade, which follows
    /// the lists, then leaves the orphan behind with its field.
    #[test]
    fn apply_delta_create_then_delete_of_a_parent_deletes_its_adopted_orphan() {
        assert_cascade_deletes_the_adopted_orphan(
            "create X then delete X",
            vec![create_cmd(0, A4K_X, "X"), delete_cmd(A4K_X)],
        );
    }

    /// T5, through a cascade from an ancestor of the just-created parent:
    /// create A, create X under A, delete A.
    #[test]
    fn apply_delta_cascade_from_an_ancestor_of_a_new_parent_deletes_its_adopted_orphan() {
        const A: u64 = 5;
        assert_cascade_deletes_the_adopted_orphan(
            "create A, create X under A, delete A",
            vec![
                create_cmd(0, A, "A"),
                create_cmd(A, A4K_X, "X"),
                delete_cmd(A),
            ],
        );
    }

    fn assert_cascade_deletes_the_adopted_orphan(label: &str, cmds: Vec<StructCommand>) {
        const O: u64 = 1;
        {
            let mut state = make_state();
            seed_section(&mut state, O, A4K_X, "o", 0);
            seed_field(&mut state, O, "f");
            seed_time_sample(&mut state, O, 1.0);
            let (mutations, _) = apply_counted(&mut state, batch(cmds));
            assert!(
                !state.sections.contains_key(&O),
                "{label}: the adopted orphan is deleted with its parent"
            );
            assert!(
                mutations.iter().any(
                    |m| matches!(m, StorageMutation::CreateTombstone { section_id, .. } if *section_id == O)
                ),
                "{label}: the orphan is tombstoned"
            );
            assert!(
                mutations.iter().any(|m| matches!(m,
                    StorageMutation::DeleteField { section_id, key_name } if *section_id == O && key_name == "f")),
                "{label}: the orphan's field is purged"
            );
            assert!(
                state.fields.is_empty() && state.time_samples.is_empty(),
                "{label}: no field or time sample of the orphan survives"
            );
            assert!(
                state.children.keys().all(|p| *p == 0),
                "{label}: only the root's list is left"
            );

            // Re-create the parent, then one more structural delta.
            apply_counted(&mut state, batch(vec![create_cmd(0, A4K_X, "X")]));
            assert_eq!(
                state.children.get(&A4K_X),
                None,
                "{label}: the re-created parent has nothing to adopt"
            );
            apply_counted(&mut state, batch(vec![create_cmd(A4K_X, 2, "next")]));
            assert_eq!(state.children.get(&A4K_X), Some(&vec![2]), "{label}");
        }
    }

    /// T6. A `ReorderChildren` naming a parent with no row changes nothing:
    /// no list appears, no orphan is re-keyed or re-stamped — including an
    /// orphan the same delta made just before it.
    ///
    /// Discriminated by: removing `insert_child_into_parent`'s missing-parent
    /// early return, which gives the in-delta orphan a list the reorder then
    /// acts on.
    #[test]
    fn apply_reorder_children_of_a_missing_parent_changes_nothing() {
        let mut state = a4k_orphan_state();
        let before = state.clone();
        let (mutations, _) = apply_counted(
            &mut state,
            batch(vec![
                change_cmd(A4K_X, 40, "o40", 2),
                reorder_cmd(A4K_X, vec![40, 30, 20, 10]),
            ]),
        );
        assert!(
            !state.children.contains_key(&A4K_X),
            "no list under the missing parent, got {:?}",
            state.children
        );
        assert_eq!(children_mutations_for(&mutations, A4K_X), 0);
        for id in A4K_ADOPTED {
            assert_eq!(
                state.sections[&id], before.sections[&id],
                "orphan {id} is untouched"
            );
        }
        assert_eq!(
            state.sections[&40].section_order, 2,
            "the in-delta orphan keeps its key"
        );
        assert_eq!(
            set_section_rows(&mutations)
                .keys()
                .copied()
                .collect::<Vec<u64>>(),
            vec![40],
            "only the in-delta orphan's own write"
        );
    }

    /// T7. The cross-map check's third half: a children list under an id that
    /// is neither the root nor a section with a row fires it.
    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "neither the root nor a section with a row")]
    fn debug_assert_cross_map_consistent_panics_on_a_list_under_a_missing_parent() {
        let mut state = make_state();
        seed_section(&mut state, 1, A4K_X, "o", 0);
        state.children.insert(A4K_X, vec![1]);
        debug_assert_cross_map_consistent(&state);
    }

    /// T7, the other side: the root's list — the one list with no row behind
    /// it — does not fire the third half, empty or not.
    #[test]
    fn debug_assert_cross_map_consistent_accepts_the_root_list() {
        let mut state = make_state();
        state.children.insert(0, Vec::new());
        debug_assert_cross_map_consistent(&state);
        seed_section(&mut state, 1, 0, "a", 0);
        state.children.insert(0, vec![1]);
        debug_assert_cross_map_consistent(&state);
    }

    /// T8. A snapshot-mode reconnector whose cut lies between a row's
    /// orphaning and its adoption receives the row under its new parent:
    /// adoption re-stamps it, so `plan_faithful_coalesced` re-emits it as
    /// modified, and the reconnector ends with the server's lists.
    ///
    /// Discriminated by: removing the adoption's `last_version` rewrite — the
    /// row stays at the orphaning's version, the coalesced diff carries only
    /// X, and the reconnector never lists the row.
    #[test]
    fn plan_faithful_coalesced_re_emits_a_row_adopted_after_the_cut() {
        const O: u64 = 1;

        let mut state = make_state();
        apply_delta(&mut state, Some(batch(vec![change_cmd(A4K_X, O, "o", 0)])))
            .expect("orphaning");
        let have_version = state.sequence_num;
        let reconnectors: Vec<ClientModel> = READERS
            .iter()
            .map(|reader| ClientModel::joined_from_snapshot(*reader, &state))
            .collect();

        apply_delta(&mut state, Some(batch(vec![create_cmd(0, A4K_X, "X")]))).expect("the birth");

        let crate::domain::diff::DiffPlan::Incremental {
            new_sections,
            modified_sections,
            ..
        } = crate::domain::diff::DiffEngine::plan_faithful_coalesced(&state, &[], have_version)
            .expect("the coalesced diff must plan")
        else {
            panic!("a reconnector behind the head plans Incremental");
        };
        let modified: Vec<(u64, u64)> = modified_sections
            .iter()
            .map(|s| (s.section_id, s.parent_id))
            .collect();
        assert_eq!(
            modified,
            vec![(O, A4K_X)],
            "the adopted row is re-emitted under its new parent"
        );

        let catch_up = OwnedDelta {
            is_diff: true,
            base_version: have_version,
            struct_commands: new_sections
                .iter()
                .chain(&modified_sections)
                .map(section_as_change)
                .collect(),
            ..Default::default()
        };
        for mut reconnector in reconnectors {
            reconnector.apply(&catch_up);
            assert_eq!(
                reconnector.lists(),
                server_lists(&state),
                "{:?}: the reconnector reaches the server's lists",
                reconnector.reader
            );
        }
        assert_eq!(server_lists(&state)[&A4K_X], vec![O]);
    }

    /// T9. A birth that closes a parent cycle — X created under its own
    /// orphan O — adopts O, leaving the consistent cycle O → X → O that the
    /// root cannot reach (the delta-boundary cross-map check passes on it).
    /// A fresh snapshot drops both, and deleting either deletes both.
    ///
    /// Discriminated by: removing the adoption (O sits under the live X and
    /// is listed nowhere).
    #[test]
    fn apply_delta_birth_under_its_own_orphan_leaves_an_unreachable_consistent_cycle() {
        const O: u64 = 1;

        let cycle = || {
            let mut state = make_state();
            seed_section(&mut state, O, A4K_X, "o", 0);
            apply_delta(&mut state, Some(batch(vec![create_cmd(O, A4K_X, "X")])))
                .expect("a birth under a live row applies");
            state
        };

        let state = cycle();
        assert_eq!(state.children.get(&O), Some(&vec![A4K_X]));
        assert_eq!(state.children.get(&A4K_X), Some(&vec![O]));
        assert_eq!(state.children.get(&0).map(Vec::len).unwrap_or(0), 0);
        match crate::domain::diff::DiffEngine::plan_faithful_snapshot(&state)
            .expect("the snapshot must plan")
        {
            crate::domain::diff::DiffPlan::FullContent { sections, .. } => assert!(
                sections.is_empty(),
                "the root reaches neither section: {sections:?}"
            ),
            other => panic!("a snapshot plans FullContent, got {other:?}"),
        }

        for doomed in [O, A4K_X] {
            let mut state = cycle();
            apply_delta(&mut state, Some(batch(vec![delete_cmd(doomed)])))
                .expect("the delete applies");
            assert!(
                state.sections.is_empty(),
                "deleting {doomed} deletes both: {:?}",
                state.sections
            );
            assert!(
                state.children.is_empty(),
                "deleting {doomed} leaves no list: {:?}",
                state.children
            );
        }
    }

    /// A chain of `len` rows hanging off the missing [`A4K_X`]: row `1000`
    /// names X (an orphan, listed nowhere), and each further row is listed
    /// under the one before it — the read model's shape for a subtree whose
    /// root lost its parent. Born under the root, X puts the chain's last row
    /// at depth `len + 1`.
    fn a4k_orphan_chain_state(len: usize) -> ObjectState {
        let mut state = make_state();
        state.sequence_num = 3;
        for i in 0..len as u64 {
            let id = 1000 + i;
            let parent = if i == 0 { A4K_X } else { id - 1 };
            seed_section(&mut state, id, parent, &format!("c{id}"), 0);
            if i > 0 {
                state.children.insert(parent, vec![id]);
            }
        }
        state
    }

    /// Adoption brings an orphan's subtree under the end-of-delta depth check:
    /// a birth whose adopted subtree puts the deepest row one past
    /// `MAX_SECTION_TREE_DEPTH` is refused with `SectionTreeTooDeep`, and the
    /// sequence does not advance. Before the birth the root reached none of
    /// the chain, so the same rows passed every earlier delta.
    ///
    /// Discriminated by: removing the adoption — the chain stays unreachable,
    /// the depth is 1, and the birth is accepted.
    fn assert_birth_over_a_too_deep_orphan_chain_is_refused(label: &str, birth: StructCommand) {
        use crate::domain::limits::MAX_SECTION_TREE_DEPTH;

        let mut state = a4k_orphan_chain_state(MAX_SECTION_TREE_DEPTH);
        apply_delta(&mut state.clone(), None).expect("fixture: the unreachable chain passes");
        let err = apply_delta(&mut state, Some(batch(vec![birth])))
            .expect_err("a birth that adopts a too-deep subtree must be refused");
        assert!(
            matches!(
                err,
                DeltaError::SectionTreeTooDeep { depth, cap }
                    if depth == MAX_SECTION_TREE_DEPTH + 1 && cap == MAX_SECTION_TREE_DEPTH
            ),
            "{label}: expected SectionTreeTooDeep(depth=cap+1), got {err:?}"
        );
        assert_eq!(
            state.sequence_num, 3,
            "{label}: the sequence does not advance"
        );
    }

    #[test]
    fn apply_delta_create_section_birth_adopting_a_too_deep_subtree_returns_section_tree_too_deep()
    {
        assert_birth_over_a_too_deep_orphan_chain_is_refused(
            "CreateSection",
            a4k_birth_by_create(),
        );
    }

    #[test]
    fn apply_delta_diff_change_section_birth_adopting_a_too_deep_subtree_returns_section_tree_too_deep(
    ) {
        assert_birth_over_a_too_deep_orphan_chain_is_refused(
            "DiffChangeSection",
            a4k_birth_by_change(),
        );
    }

    /// The twin one level shallower: the deepest adopted row lands exactly at
    /// `MAX_SECTION_TREE_DEPTH`, and the birth is accepted with the whole
    /// chain reachable.
    #[test]
    fn apply_delta_birth_adopting_a_subtree_at_the_depth_cap_succeeds() {
        use crate::domain::limits::MAX_SECTION_TREE_DEPTH;

        let mut state = a4k_orphan_chain_state(MAX_SECTION_TREE_DEPTH - 1);
        apply_delta(&mut state, Some(batch(vec![a4k_birth_by_create()])))
            .expect("depth == cap must be accepted");
        assert_eq!(state.sequence_num, 4);
        assert_eq!(state.children.get(&A4K_X), Some(&vec![1000]));
    }

    /// A second birth target: a missing id named by the orphan [`A4K_ORPHAN_Y`].
    const A4K_Y: u64 = 40;
    /// The orphan naming [`A4K_Y`] in [`a4k_two_parent_orphan_state`].
    const A4K_ORPHAN_Y: u64 = 41;
    /// A row a `DiffChangeSection` creates under [`A4K_X`] after X's cascade.
    const A4K_LATE: u64 = 50;

    /// [`a4k_orphan_state`] plus the orphan [`A4K_ORPHAN_Y`] naming the
    /// missing [`A4K_Y`], so both birth targets have a row to adopt.
    fn a4k_two_parent_orphan_state() -> ObjectState {
        let mut state = a4k_orphan_state();
        seed_section(&mut state, A4K_ORPHAN_Y, A4K_Y, "o41", 0);
        state
    }

    /// The adoption index is built at most once per delta, however many
    /// sections the delta births: four births by both commands — X, Y, the
    /// new row 50 under the missing X, and a re-birth of X after a
    /// `DeleteSection` cascaded X and its adopted rows — build it once, by
    /// the first birth. Every later birth reads the index that
    /// `ChildPlan::name_inserted` kept current, which is why X's re-birth
    /// still adopts 50, written after the build. One test per command
    /// giving the delta its first birth.
    ///
    /// A rebuild per birth is an O(S) pass over every row each time, the
    /// cost the scoped index exists to remove. The adoption result is the
    /// same with or without it, so `adoption_index_builds` is the only signal
    /// that sees it.
    ///
    /// Discriminated by: making `adopt_orphans` rebuild the index (and bump
    /// the counter) on every birth. Both tests then fail on the count, with 4.
    fn assert_multi_birth_delta_builds_the_adoption_index_once(
        label: &str,
        cmds: Vec<StructCommand>,
    ) {
        let mut state = a4k_two_parent_orphan_state();
        let (_, stats) = apply_counted(&mut state, batch(cmds));

        // The four births happened and adopted what they should.
        assert_eq!(
            state.children.get(&A4K_Y),
            Some(&vec![A4K_ORPHAN_Y]),
            "{label}: Y's birth adopts its orphan"
        );
        for id in A4K_ADOPTED {
            assert!(
                !state.sections.contains_key(&id),
                "{label}: X's first birth adopted {id}, so X's cascade deletes it"
            );
        }
        assert_eq!(
            state.children.get(&A4K_X),
            Some(&vec![A4K_LATE]),
            "{label}: X's re-birth after the cascade adopts the row named after the build"
        );

        assert_eq!(
            stats.adoption_index_builds, 1,
            "{label}: four births in one delta build the adoption index once"
        );
    }

    #[test]
    fn apply_delta_multi_birth_delta_led_by_create_section_builds_the_adoption_index_once() {
        assert_multi_birth_delta_builds_the_adoption_index_once(
            "first birth by CreateSection",
            vec![
                create_cmd(0, A4K_X, "X"),
                change_cmd(0, A4K_Y, "Y", 1),
                delete_cmd(A4K_X),
                change_cmd(A4K_X, A4K_LATE, "o50", 0),
                create_cmd(0, A4K_X, "X"),
            ],
        );
    }

    #[test]
    fn apply_delta_multi_birth_delta_led_by_diff_change_section_builds_the_adoption_index_once() {
        assert_multi_birth_delta_builds_the_adoption_index_once(
            "first birth by DiffChangeSection",
            vec![
                change_cmd(0, A4K_X, "X", 0),
                create_cmd(0, A4K_Y, "Y"),
                delete_cmd(A4K_X),
                change_cmd(A4K_X, A4K_LATE, "o50", 0),
                change_cmd(0, A4K_X, "X", 0),
            ],
        );
    }

    /// A delta that births no section builds no adoption index, even when its
    /// `CreateSection` and `DiffChangeSection` commands name ids, which
    /// makes those ids index candidates, and a row names a missing parent. A
    /// `DiffChangeSection` of a live row (a re-key under the same parent, and
    /// the orphan 41 reparented under a live parent) is not a birth, and
    /// neither is a `CreateSection` the engine skips for a missing parent. A
    /// field-only delta names no candidate at all.
    ///
    /// Discriminated by: making `apply_diff_change_section`'s live-row branch
    /// call `adopt_orphans` as well. The structural delta then builds the
    /// index once.
    #[test]
    fn apply_delta_without_a_birth_builds_no_adoption_index() {
        const P: u64 = 2;
        const LIVE: u64 = 1;
        const SKIPPED: u64 = 60;

        let mut state = a4k_two_parent_orphan_state();
        seed_section(&mut state, P, 0, "P", 0);
        seed_section(&mut state, LIVE, P, "live", 0);
        state.children.insert(0, vec![P]);
        state.children.insert(P, vec![LIVE]);

        let (_, stats) = apply_counted(
            &mut state,
            batch(vec![
                change_cmd(P, LIVE, "live", 3),
                change_cmd(P, A4K_ORPHAN_Y, "o41", 5),
                create_cmd(A4K_Y, SKIPPED, "skipped"),
            ]),
        );
        assert_eq!(
            state.children.get(&P),
            Some(&vec![LIVE, A4K_ORPHAN_Y]),
            "the re-key and the reparent both land under P"
        );
        assert!(
            !state.sections.contains_key(&SKIPPED),
            "the CreateSection under the missing Y is skipped"
        );
        assert_eq!(
            stats.adoption_index_builds, 0,
            "a structural delta with no birth builds no adoption index"
        );

        let (_, stats) = apply_counted(
            &mut state,
            OwnedDelta {
                is_diff: false,
                set_fields: vec![OodfSetField {
                    section_id: LIVE,
                    key_name: "k".to_string(),
                    value_or_ext_hash: vec![1],
                    ext_value_size: 0,
                    ext_value_index: 0,
                    set_order: 0,
                }],
                ..Default::default()
            },
        );
        assert_eq!(
            stats.adoption_index_builds, 0,
            "a field-only delta builds no adoption index"
        );
    }

    // ── The bounded plan: `apply_delta_bounded` (b.26e) ─────────────────────

    /// The parent the bounded-plan fixtures edit.
    const BOUNDED_PARENT: u64 = 5;
    /// The first of its children's ids.
    const FIRST_CHILD: u64 = 100;
    /// A long children list: long enough that a rewrite growing with it is
    /// unmistakable next to the delta's one element.
    const LONG_LIST: usize = 1_000;
    /// Distinct costs per rewritten row and per rewritten list entry, so a
    /// count landing in the wrong term changes the cost.
    const ROW_COST: u64 = 7;
    const LIST_ENTRY_COST: u64 = 3;

    /// A state whose `BOUNDED_PARENT` (under the root) has `n` childless
    /// children, ids from `FIRST_CHILD`, keyed `0..n` in list order — the
    /// strictly increasing keys every list this engine builds carries.
    fn parent_with_children(n: usize) -> ObjectState {
        let mut state = make_state();
        state.sequence_num = 1;
        state
            .sections
            .insert(BOUNDED_PARENT, make_section(BOUNDED_PARENT, 0, "Parent"));
        state.children.insert(0, vec![BOUNDED_PARENT]);
        let ids: Vec<u64> = (0..n as u64).map(|i| FIRST_CHILD + i).collect();
        for (i, &id) in ids.iter().enumerate() {
            let mut section = make_section(id, BOUNDED_PARENT, &format!("c{i}"));
            section.section_order = u32::try_from(i).expect("small fixture");
            state.sections.insert(id, section);
        }
        state.children.insert(BOUNDED_PARENT, ids);
        state
    }

    fn reorder_parent(children_list: Vec<u64>) -> OwnedDelta {
        OwnedDelta {
            struct_commands: vec![StructCommand::ReorderChildren(OodfReorderChildren {
                section_id: BOUNDED_PARENT,
                children_list_id: 0,
                children_list,
            })],
            ..Default::default()
        }
    }

    /// The rewrite `mutations` make for `delta`, counted independently of the
    /// bounded sink from what `apply_delta_with_stats` returns: every
    /// mutation less the delta's own elements; and, per parent, the larger of
    /// the length of the `SetChildren` it emits and — for each parent in
    /// `listed`, the parents whose lists the plan reads (`(parent, its
    /// pre-delta length)`) — that whole list, which the plan projects before
    /// it lists it. So deleting one of a parent's `L` children counts `L`,
    /// not the `L − 1` it emits, and a parent listed and never emitted keeps
    /// its `L`.
    fn rewrite_of(
        mutations: &[StorageMutation],
        delta: Option<&OwnedDelta>,
        listed: &[(u64, u64)],
    ) -> RewriteCount {
        let own = delta.map_or(0, |d| {
            (d.struct_commands.len() + d.set_fields.len() + d.time_samples.len()) as u64
        });
        let mut per_parent: std::collections::BTreeMap<u64, u64> = listed.iter().copied().collect();
        for m in mutations {
            if let StorageMutation::SetChildren {
                section_id,
                children,
            } = m
            {
                let entries = per_parent.entry(*section_id).or_default();
                *entries = (*entries).max(children.len() as u64);
            }
        }
        RewriteCount {
            rows: (mutations.len() as u64).saturating_sub(own),
            list_entries: per_parent.values().sum(),
            indexed_sections: 0,
            adoption_entries: 0,
            walked_sections: 0,
            rollback_rebuild_bytes: 0,
        }
    }

    /// A bounded-plan shape: its name, state and delta, and the parents
    /// whose lists its plan reads, each with its pre-delta length — what the
    /// plan projects before it lists the parent ([`rewrite_of`]).
    type BoundedShape = (
        &'static str,
        ObjectState,
        Option<OwnedDelta>,
        Vec<(u64, u64)>,
    );

    /// The bounded-plan shapes. Each but the last two rewrites existing rows
    /// or a list the delta's own elements do not bound. The cascade lists
    /// the deleted parent's own parent, the root (one child, which it emits
    /// as an empty list), and not the deleted parent's list, which it deletes
    /// by walking it.
    fn bounded_shapes() -> Vec<BoundedShape> {
        let field_section = {
            let mut state = make_state();
            state.sections.insert(1, make_section(1, 0, "S"));
            state.children.insert(0, vec![1]);
            state
        };
        let long = LONG_LIST as u64;
        vec![
            (
                "move the first child to the end",
                parent_with_children(LONG_LIST),
                Some(reorder_parent(vec![FIRST_CHILD + 1, FIRST_CHILD])),
                vec![(BOUNDED_PARENT, long)],
            ),
            (
                "one new child under a long list",
                parent_with_children(LONG_LIST),
                Some(delta_create_section(BOUNDED_PARENT, 99_999, "new")),
                vec![(BOUNDED_PARENT, long)],
            ),
            (
                "cascade delete of a parent and its children",
                parent_with_children(50),
                Some(delta_delete_section(BOUNDED_PARENT)),
                vec![(0, 1)],
            ),
            (
                "reset of a populated state",
                parent_with_children(50),
                Some(OwnedDelta {
                    is_diff: true,
                    base_version: 0,
                    ..Default::default()
                }),
                vec![],
            ),
            (
                "fields on an existing section",
                field_section,
                Some(OwnedDelta {
                    set_fields: (0..3)
                        .map(|i| OodfSetField {
                            section_id: 1,
                            key_name: format!("k{i}"),
                            value_or_ext_hash: vec![i],
                            ..Default::default()
                        })
                        .collect(),
                    ..Default::default()
                }),
                vec![],
            ),
            ("the empty delta", parent_with_children(3), None, vec![]),
        ]
    }

    fn allowance(bytes: u64) -> MutationAllowance {
        MutationAllowance {
            bytes,
            row_bytes: ROW_COST,
            list_entry_bytes: LIST_ENTRY_COST,
            index_entry_bytes: 0,
            adoption_entry_bytes: 0,
            walk_entry_bytes: 0,
        }
    }

    /// A budget that refuses every extension: the plan is bounded at its
    /// first allowance.
    fn refuse() -> impl FnMut(RewriteCount) -> Option<u64> {
        |_| None
    }

    /// The reference plan of `delta` on a copy of `state`: the planned state,
    /// the mutations, the stats.
    fn reference_plan(
        state: &ObjectState,
        delta: Option<OwnedDelta>,
    ) -> (ObjectState, Vec<StorageMutation>, ApplyDeltaStats) {
        let mut planned = state.clone();
        let mut stats = ApplyDeltaStats::default();
        let mutations =
            apply_delta_with_stats(&mut planned, delta, &mut stats).expect("fixtures apply");
        (planned, mutations, stats)
    }

    /// Within an allowance that covers the rewrite — exactly, or with room to
    /// spare — the bounded plan is `apply_delta_with_stats`'s: the same
    /// planned state, mutations and stats, and the rewrite they make.
    #[test]
    fn apply_delta_bounded_within_allowance_matches_apply_delta_with_stats() {
        for (name, state, delta, listed) in bounded_shapes() {
            let (want_state, want_mutations, want_stats) = reference_plan(&state, delta.clone());
            let want_rewrite = rewrite_of(&want_mutations, delta.as_ref(), &listed);
            let exact = want_rewrite.bytes(ROW_COST, LIST_ENTRY_COST, 0, 0, 0);
            for bytes in [exact, u64::MAX] {
                let mut planned = state.clone();
                let mut stats = ApplyDeltaStats::default();
                let plan = apply_delta_bounded(
                    &mut planned,
                    delta.clone(),
                    allowance(bytes),
                    &mut refuse(),
                    &mut stats,
                )
                .expect("fixtures apply");
                let BoundedPlan::Within { mutations, rewrite } = plan else {
                    panic!("{name}: allowance {bytes} covers {exact}: got {plan:?}");
                };
                assert_eq!(mutations, want_mutations, "{name}: mutations");
                assert_eq!(rewrite, want_rewrite, "{name}: rewrite");
                assert_eq!(planned, want_state, "{name}: planned state");
                assert_eq!(stats, want_stats, "{name}: stats");
            }
        }
    }

    /// An allowance one byte short of the rewrite — or none at all — aborts
    /// the plan, which still reports the exact need: the same count a
    /// covering allowance plans within.
    #[test]
    fn apply_delta_bounded_past_allowance_reports_the_exact_rewrite_needed() {
        for (name, state, delta, listed) in bounded_shapes() {
            let (_, want_mutations, _) = reference_plan(&state, delta.clone());
            let want_rewrite = rewrite_of(&want_mutations, delta.as_ref(), &listed);
            let exact = want_rewrite.bytes(ROW_COST, LIST_ENTRY_COST, 0, 0, 0);
            assert!(exact > 0, "fixture self-check: {name} rewrites something");
            for bytes in [exact - 1, 0] {
                let mut planned = state.clone();
                let mut stats = ApplyDeltaStats::default();
                let plan = apply_delta_bounded(
                    &mut planned,
                    delta.clone(),
                    allowance(bytes),
                    &mut refuse(),
                    &mut stats,
                )
                .expect("fixtures apply");
                let BoundedPlan::Overflow { needed } = plan else {
                    panic!("{name}: allowance {bytes} is short of {exact}: got {plan:?}");
                };
                assert_eq!(needed, want_rewrite, "{name}: allowance {bytes}");
            }
        }
    }

    /// The rewrite grows with the object, not the delta: moving the first of
    /// a long list's children to the end re-keys every child (one row each,
    /// plus the list and the header, less the one command), and one new child
    /// rewrites the whole list it joins; a true swap re-keys only the two.
    #[test]
    fn apply_delta_bounded_rewrite_counts_rekeyed_rows_and_rewritten_list_entries() {
        let n = LONG_LIST as u64;
        for (name, delta, want) in [
            (
                "move the first child to the end",
                reorder_parent(vec![FIRST_CHILD + 1, FIRST_CHILD]),
                RewriteCount {
                    rows: n + 1,
                    list_entries: n,
                    indexed_sections: 0,
                    adoption_entries: 0,
                    walked_sections: 0,
                    rollback_rebuild_bytes: 0,
                },
            ),
            (
                "true swap of the first two",
                reorder_parent(vec![FIRST_CHILD + 1, FIRST_CHILD, FIRST_CHILD + 2]),
                RewriteCount {
                    rows: 3,
                    list_entries: n,
                    indexed_sections: 0,
                    adoption_entries: 0,
                    walked_sections: 0,
                    rollback_rebuild_bytes: 0,
                },
            ),
            (
                "one new child",
                delta_create_section(BOUNDED_PARENT, 99_999, "new"),
                RewriteCount {
                    rows: 2,
                    list_entries: n + 1,
                    indexed_sections: 0,
                    adoption_entries: 0,
                    walked_sections: 0,
                    rollback_rebuild_bytes: 0,
                },
            ),
        ] {
            let mut state = parent_with_children(LONG_LIST);
            let plan = apply_delta_bounded(
                &mut state,
                Some(delta),
                allowance(u64::MAX),
                &mut refuse(),
                &mut ApplyDeltaStats::default(),
            )
            .expect("fixtures apply");
            let BoundedPlan::Within { rewrite, .. } = plan else {
                panic!("{name}: an unbounded allowance plans within: got {plan:?}");
            };
            assert_eq!(rewrite, want, "{name}");
        }
    }

    /// A listed parent counts its whole pre-delta list, whatever the plan
    /// then emits for it: the plan projects the list
    /// before it builds the listing, and per parent the larger of that
    /// projection and the emitted `SetChildren` is kept. So deleting one of a
    /// parent's `L` children counts `L` list entries though the plan emits a
    /// list of `L − 1`. A parent listed and then deleted, whose list the plan
    /// emits nowhere, keeps its projection of `L` in what the plan asks for
    /// and in its overflow need, which are what it held; the kept count a
    /// plan within its allowance reports, which only the mutation list
    /// outlives, does not carry it.
    #[test]
    fn apply_delta_bounded_counts_a_listed_parents_whole_list_whatever_it_emits() {
        let n = LONG_LIST as u64;
        let within = |delta: OwnedDelta| {
            let mut state = parent_with_children(LONG_LIST);
            let plan = apply_delta_bounded(
                &mut state,
                Some(delta),
                allowance(u64::MAX),
                &mut refuse(),
                &mut ApplyDeltaStats::default(),
            )
            .expect("fixtures apply");
            let BoundedPlan::Within { mutations, rewrite } = plan else {
                panic!("an unbounded allowance plans within: got {plan:?}");
            };
            (mutations, rewrite)
        };
        let emitted = |mutations: &[StorageMutation], parent: u64| {
            mutations.iter().find_map(|m| match m {
                StorageMutation::SetChildren {
                    section_id,
                    children,
                } if *section_id == parent => Some(children.len() as u64),
                _ => None,
            })
        };

        let (mutations, rewrite) = within(delta_delete_section(FIRST_CHILD));
        assert_eq!(
            emitted(&mutations, BOUNDED_PARENT),
            Some(n - 1),
            "fixture self-check: the plan emits the parent's list less the deleted child"
        );
        assert_eq!(
            rewrite.list_entries, n,
            "deleting 1 of {n} children counts the whole list of {n}"
        );

        let listed_then_deleted = || {
            batch(vec![
                create_cmd(BOUNDED_PARENT, 99_999, "new"),
                delete_cmd(BOUNDED_PARENT),
            ])
        };
        let (mutations, rewrite) = within(listed_then_deleted());
        assert_eq!(
            emitted(&mutations, BOUNDED_PARENT),
            None,
            "fixture self-check: the deleted parent's list is emitted nowhere"
        );
        assert_eq!(
            emitted(&mutations, 0),
            Some(0),
            "fixture self-check: the root's list of one is emitted empty"
        );
        assert_eq!(
            rewrite.list_entries, 1,
            "kept: the root's projection of 1 over its empty list; the deleted parent's list is \
             plan scratch the mutation list does not hold"
        );

        let mut state = parent_with_children(LONG_LIST);
        let plan = apply_delta_bounded(
            &mut state,
            Some(listed_then_deleted()),
            allowance(0),
            &mut refuse(),
            &mut ApplyDeltaStats::default(),
        )
        .expect("fixtures apply");
        let BoundedPlan::Overflow { needed } = plan else {
            panic!("no allowance overflows: got {plan:?}");
        };
        assert_eq!(
            needed.list_entries,
            n + 1,
            "the need: the deleted parent keeps its projection of {n}, and the root its 1"
        );
    }

    /// Prices for every rewrite term, each distinct and nonzero, so a
    /// projection of any kind passes an allowance of 0 and asks.
    fn priced_allowance(bytes: u64) -> MutationAllowance {
        MutationAllowance {
            bytes,
            row_bytes: ROW_COST,
            list_entry_bytes: LIST_ENTRY_COST,
            index_entry_bytes: 5,
            adoption_entry_bytes: 11,
            walk_entry_bytes: 0,
        }
    }

    /// One ask of a recording budget: what the plan asked for, and the
    /// build counters (`ParentCache::build` calls, adoption-index records)
    /// when it asked.
    type Ask = (RewriteCount, u64, u64);

    /// Plan `delta` on `state` from an allowance of 0 under a budget that
    /// grants exactly each ask and records it with the build counters at the
    /// ask; the counters start at 0. Returns the asks and the plan.
    fn plan_recording_asks(state: &ObjectState, delta: OwnedDelta) -> (Vec<Ask>, BoundedPlan) {
        test_counters::PARENT_CACHE_BUILDS.with(|c| c.set(0));
        test_counters::ADOPTION_INDEX_RECORDS.with(|c| c.set(0));
        let mut asks = Vec::new();
        let mut planned = state.clone();
        let plan = {
            let mut grant_exactly = |needed: RewriteCount| {
                asks.push((
                    needed,
                    test_counters::PARENT_CACHE_BUILDS.with(std::cell::Cell::get),
                    test_counters::ADOPTION_INDEX_RECORDS.with(std::cell::Cell::get),
                ));
                Some(needed.bytes(ROW_COST, LIST_ENTRY_COST, 5, 11, 0))
            };
            apply_delta_bounded(
                &mut planned,
                Some(delta),
                priced_allowance(0),
                &mut grant_exactly,
                &mut ApplyDeltaStats::default(),
            )
            .expect("fixtures apply")
        };
        (asks, plan)
    }

    /// A listing is projected before it is built. One `CreateSection` under
    /// a 1,000-child parent, from an allowance of 0: the plan's first ask
    /// carries the parent's whole list (`{rows 0, list_entries 1,000}`)
    /// while no `ParentCache` has been built. A plan that asked only at its
    /// emissions would first ask with `{1, 1,001}`, after the build.
    #[test]
    fn apply_delta_bounded_projects_a_listing_before_it_builds_it() {
        let n = LONG_LIST as u64;
        let (asks, plan) = plan_recording_asks(
            &parent_with_children(LONG_LIST),
            delta_create_section(BOUNDED_PARENT, 99_999, "new"),
        );
        assert!(
            matches!(plan, BoundedPlan::Within { .. }),
            "every ask granted: {plan:?}"
        );
        let first = asks.first().copied().expect("the plan asks");
        assert_eq!(
            first,
            (
                RewriteCount {
                    rows: 0,
                    list_entries: n,
                    indexed_sections: 0,
                    adoption_entries: 0,
                    walked_sections: 0,
                    rollback_rebuild_bytes: 0,
                },
                0,
                0
            ),
            "the first ask is the listing's projection, before the cache is built; asks: \
             {asks:?}"
        );
    }

    /// The adoption-index tests' fixture: the 1,000-child parent re-keyed
    /// under the root
    /// (`DiffChangeSection`, a candidate the adoption index records every row
    /// naming) and a new section `Y` under the root (a birth, which builds
    /// the index), in `p_first` order.
    fn rekey_and_birth(p_first: bool) -> OwnedDelta {
        let rekey = change_cmd(0, BOUNDED_PARENT, "Parent", 1);
        let birth = create_cmd(0, 88_888, "Y");
        batch(if p_first {
            vec![rekey, birth]
        } else {
            vec![birth, rekey]
        })
    }

    /// Each id the adoption index records is asked for before it is
    /// recorded. Under a budget granting exactly each ask, the
    /// ask carrying `adoption_entries = k` sees `k − 1` records, for every k
    /// in 1..=1,000. Discriminated by removing the projection from the
    /// record site: the counter then runs ahead of every ask.
    #[test]
    fn apply_delta_bounded_asks_before_each_adoption_index_record() {
        for p_first in [true, false] {
            let (asks, plan) =
                plan_recording_asks(&parent_with_children(LONG_LIST), rekey_and_birth(p_first));
            assert!(
                matches!(plan, BoundedPlan::Within { .. }),
                "every ask granted: {plan:?}"
            );
            let mut seen = std::collections::BTreeMap::new();
            for (needed, _, records) in &asks {
                seen.entry(needed.adoption_entries).or_insert(*records);
            }
            for k in 1..=LONG_LIST as u64 {
                assert_eq!(
                    seen.get(&k),
                    Some(&(k - 1)),
                    "p_first {p_first}: the ask for adoption entry {k} comes with {} recorded",
                    k - 1
                );
            }
        }
    }

    /// The adoption index's ids are counted, plan-only.
    /// With a refusing budget the need carries all 1,000 recorded ids in
    /// both command orders, and 1,001 when a later `CreateSection` under the
    /// re-keyed parent hooks one more; a plan within its allowance keeps
    /// none in its rewrite; and a one-new-child delta on a base with no
    /// orphans asks no adoption entries.
    #[test]
    fn apply_delta_bounded_counts_adoption_index_entries_plan_only() {
        let base = parent_with_children(LONG_LIST);
        let overflow = |delta: OwnedDelta| {
            let mut planned = base.clone();
            match apply_delta_bounded(
                &mut planned,
                Some(delta),
                priced_allowance(0),
                &mut refuse(),
                &mut ApplyDeltaStats::default(),
            )
            .expect("fixtures apply")
            {
                BoundedPlan::Overflow { needed } => needed,
                plan => panic!("no allowance overflows: got {plan:?}"),
            }
        };
        for p_first in [true, false] {
            assert_eq!(
                overflow(rekey_and_birth(p_first)).adoption_entries,
                LONG_LIST as u64,
                "p_first {p_first}: the need carries every recorded id"
            );
        }
        let mut hooked = rekey_and_birth(true);
        hooked
            .struct_commands
            .push(create_cmd(BOUNDED_PARENT, 77_777, "Z"));
        assert_eq!(
            overflow(hooked).adoption_entries,
            LONG_LIST as u64 + 1,
            "a later birth under the re-keyed parent hooks one more id"
        );

        let (_, plan) = plan_recording_asks(&base, rekey_and_birth(true));
        let BoundedPlan::Within { rewrite, .. } = plan else {
            panic!("every ask granted: {plan:?}");
        };
        assert_eq!(
            rewrite.adoption_entries, 0,
            "Within keeps no adoption entries"
        );

        let (asks, _) =
            plan_recording_asks(&base, delta_create_section(BOUNDED_PARENT, 99_999, "n"));
        assert!(
            asks.iter()
                .all(|(needed, _, _)| needed.adoption_entries == 0),
            "a one-new-child delta asks no adoption entries: {asks:?}"
        );
    }

    /// The adoption index's ids, per adopted orphan: a birth adopting `k` orphans asks
    /// for `k` adoption-index entries **and** `k` rows — the adopted rows'
    /// re-stamps — and its plan emits exactly one `SetSection` per adopted
    /// orphan. One run per birth command, at k = 1, 3 and 64: the need of a
    /// refusing budget carries `adoption_entries = k`, its rows grow by one
    /// per orphan (everything else the birth writes — its own row, the
    /// root's list and its new list — is the same at every k), and the
    /// largest ask of a budget granting each ask equals that need.
    #[test]
    fn apply_delta_bounded_birth_adopting_k_orphans_asks_k_entries_and_k_rows() {
        let orphans = |k: u64| {
            let mut state = make_state();
            state.sequence_num = 3;
            for i in 0..k {
                seed_section(&mut state, 1_000 + i, A4K_X, &format!("o{i}"), 0);
            }
            state
        };
        for (label, birth) in [
            ("CreateSection", a4k_birth_by_create()),
            ("DiffChangeSection", a4k_birth_by_change()),
        ] {
            let mut rows_at_one = None;
            for k in [1_u64, 3, 64] {
                let state = orphans(k);
                let mut planned = state.clone();
                let needed = match apply_delta_bounded(
                    &mut planned,
                    Some(batch(vec![birth.clone()])),
                    priced_allowance(0),
                    &mut refuse(),
                    &mut ApplyDeltaStats::default(),
                )
                .expect("fixtures apply")
                {
                    BoundedPlan::Overflow { needed } => needed,
                    plan => panic!("{label} k={k}: no allowance overflows: got {plan:?}"),
                };
                assert_eq!(
                    needed.adoption_entries, k,
                    "{label} k={k}: one adoption-index entry per orphan"
                );
                let rows_at_one = *rows_at_one.get_or_insert(needed.rows);
                assert_eq!(
                    needed.rows - rows_at_one,
                    k - 1,
                    "{label} k={k}: one rewritten row per adopted orphan ({needed:?})"
                );
                assert!(needed.rows >= k, "{label} k={k}: {needed:?}");

                let (asks, plan) = plan_recording_asks(&state, batch(vec![birth.clone()]));
                let BoundedPlan::Within { mutations, .. } = plan else {
                    panic!("{label} k={k}: every ask granted: {plan:?}");
                };
                let largest = asks
                    .iter()
                    .fold(RewriteCount::default(), |most, (asked, _, _)| {
                        RewriteCount {
                            rows: most.rows.max(asked.rows),
                            list_entries: most.list_entries.max(asked.list_entries),
                            indexed_sections: most.indexed_sections.max(asked.indexed_sections),
                            adoption_entries: most.adoption_entries.max(asked.adoption_entries),
                            walked_sections: most.walked_sections.max(asked.walked_sections),
                            rollback_rebuild_bytes: most
                                .rollback_rebuild_bytes
                                .max(asked.rollback_rebuild_bytes),
                        }
                    });
                assert_eq!(
                    (largest.rows, largest.adoption_entries),
                    (needed.rows, k),
                    "{label} k={k}: the asks reach the need; asks: {asks:?}"
                );
                for id in (0..k).map(|i| 1_000 + i) {
                    assert_eq!(
                        set_section_count(&mutations, id),
                        1,
                        "{label} k={k}: adopted orphan {id} gets exactly one SetSection"
                    );
                }
                assert_eq!(
                    mutations
                        .iter()
                        .filter(|m| matches!(m, StorageMutation::SetSection { .. }))
                        .count() as u64,
                    k + 1,
                    "{label} k={k}: the birth's own row and one per orphan"
                );
            }
        }
    }

    /// The name probe's conflict answer: the plan's first
    /// `(parent_id, name)` probe is answered by a scan of `state.sections`
    /// ([`scan_child_by_name`]) and its second by the index it builds, and
    /// both give the same conflict id over duplicate `(parent, name)` rows:
    /// the lowest surviving id, other than `except_id`. After the index is
    /// built, a holder deleted, reparented or renamed under it is filtered by
    /// the index's re-verification exactly as a fresh scan leaves it out.
    #[test]
    fn find_child_by_name_scan_and_index_give_the_same_conflict_id() {
        const P: u64 = 1;
        const Q: u64 = 2;
        let base = || {
            let mut state = make_state();
            seed_section(&mut state, P, 0, "P", 0);
            seed_section(&mut state, Q, 0, "Q", 0);
            for id in [30, 10, 20] {
                seed_section(&mut state, id, P, "A", 0);
            }
            seed_section(&mut state, 40, P, "B", 0);
            state
        };
        type Change = fn(&mut ObjectState);
        let cases: [(&str, Change, Option<u64>, Option<u64>); 8] = [
            ("the lowest of three holders", |_| {}, None, Some(10)),
            ("the lowest excepted", |_| {}, Some(10), Some(20)),
            ("a higher holder excepted", |_| {}, Some(20), Some(10)),
            (
                "the lowest deleted",
                |s| {
                    s.sections.remove(&10);
                },
                None,
                Some(20),
            ),
            (
                "the lowest reparented",
                |s| {
                    s.sections.get_mut(&10).expect("seeded").parent_id = Q;
                },
                None,
                Some(20),
            ),
            (
                "the lowest renamed",
                |s| {
                    s.sections.get_mut(&10).expect("seeded").name = "Z".to_owned();
                },
                None,
                Some(20),
            ),
            (
                "the lowest deleted, the next excepted",
                |s| {
                    s.sections.remove(&10);
                },
                Some(20),
                Some(30),
            ),
            (
                "every holder deleted",
                |s| {
                    for id in [10, 20, 30] {
                        s.sections.remove(&id);
                    }
                },
                None,
                None,
            ),
        ];
        for (label, change, except_id, expected) in cases {
            let mut state = base();
            let mut plan = ChildPlan::default();
            let mut stats = ApplyDeltaStats::default();
            let scanned = find_child_by_name(&state, &mut plan, &mut stats, P, "A", except_id);
            assert!(
                plan.names.is_none(),
                "{label}: the first probe builds nothing"
            );
            let indexed = find_child_by_name(&state, &mut plan, &mut stats, P, "A", except_id);
            assert!(plan.names.is_some(), "{label}: the second probe builds");
            assert_eq!(scanned, indexed, "{label}: before the change");
            change(&mut state);
            let indexed = find_child_by_name(&state, &mut plan, &mut stats, P, "A", except_id);
            let scanned = scan_child_by_name(&state.sections, P, "A", except_id);
            assert_eq!(
                (indexed, scanned),
                (expected, expected),
                "{label}: the index and a scan agree after the change"
            );
        }
    }

    /// The name probe, a one-probe plan: a single `CreateSection`
    /// onto a duplicated name resolves its conflict by the scan — the lowest
    /// id's subtree is cascade-deleted, as the index would pick — and builds
    /// no index (`name_index_builds` 0, no `NameIndex::build` call).
    #[test]
    fn apply_delta_one_probe_plan_builds_no_name_index() {
        let mut state = make_state();
        seed_section(&mut state, 1, 0, "P", 0);
        seed_section(&mut state, 20, 1, "A", 0);
        seed_section(&mut state, 10, 1, "A", 0);
        seed_section(&mut state, 11, 10, "child-of-10", 0);
        state.children.insert(0, vec![1]);
        state.children.insert(1, vec![10, 20]);
        state.children.insert(10, vec![11]);
        test_counters::NAME_INDEX_BUILDS.with(|c| c.set(0));

        let (_, stats) = apply_counted(&mut state, batch(vec![create_cmd(1, 30, "A")]));

        assert_eq!(
            (stats.name_index_probes, stats.name_index_builds),
            (1, 0),
            "one probe, answered by the scan"
        );
        assert_eq!(
            test_counters::NAME_INDEX_BUILDS.with(std::cell::Cell::get),
            0,
            "no NameIndex::build"
        );
        assert!(
            !state.sections.contains_key(&10) && !state.sections.contains_key(&11),
            "the lowest holder's subtree is cascade-deleted"
        );
        assert_eq!(state.children[&1], vec![20, 30]);
    }

    /// The name probe, a two-probe plan: its second probe projects the
    /// index at the object's sections plus the delta's births and moves
    /// before it builds it — the ask that first carries the index sees no
    /// `NameIndex::build` call — and the index is built once. "The object's
    /// sections" are counted when the index is built, at the second probe,
    /// so a birth that landed before it is counted twice (once as a section,
    /// once as a birth): `S + 1 + 2` here, one over `S + births`, the safe
    /// direction.
    #[test]
    fn apply_delta_bounded_asks_for_the_name_index_before_it_builds_it() {
        let state = parent_with_children(LONG_LIST);
        let sections = state.sections.len() as u64;
        let delta = batch(vec![
            create_cmd(BOUNDED_PARENT, 99_998, "x"),
            create_cmd(BOUNDED_PARENT, 99_999, "y"),
        ]);
        test_counters::NAME_INDEX_BUILDS.with(|c| c.set(0));
        let mut asks = Vec::new();
        let mut planned = state.clone();
        let mut stats = ApplyDeltaStats::default();
        let plan = {
            let mut grant_exactly = |needed: RewriteCount| {
                asks.push((
                    needed,
                    test_counters::NAME_INDEX_BUILDS.with(std::cell::Cell::get),
                ));
                Some(needed.bytes(ROW_COST, LIST_ENTRY_COST, 5, 11, 0))
            };
            apply_delta_bounded(
                &mut planned,
                Some(delta),
                priced_allowance(0),
                &mut grant_exactly,
                &mut stats,
            )
            .expect("fixtures apply")
        };
        assert!(
            matches!(plan, BoundedPlan::Within { .. }),
            "every ask granted: {plan:?}"
        );
        assert_eq!(
            stats.name_index_builds, 1,
            "two probes build the index once"
        );
        let first_index_ask = asks
            .iter()
            .find(|(needed, _)| needed.indexed_sections > 0)
            .copied()
            .expect("the index is asked for");
        assert_eq!(
            (first_index_ask.0.indexed_sections, first_index_ask.1),
            (sections + 1 + 2, 0),
            "the index is projected at the sections then (S and the first birth) plus the \
             delta's two births, before it is built; asks: {asks:?}"
        );
        assert_eq!(
            test_counters::NAME_INDEX_BUILDS.with(std::cell::Cell::get),
            1
        );
    }

    /// A rebuilt listing: a parent's cache dropped by an
    /// order-side write and rebuilt over the same list projects nothing new —
    /// the listing is projected once, at its first build — so the need
    /// carries the parent's list once, not once per build.
    #[test]
    fn apply_delta_bounded_rebuilt_listing_projects_nothing_new() {
        const P: u64 = 1;
        const C: u64 = 1_000;
        let mut state = make_state();
        seed_section(&mut state, P, 0, "P", 0);
        state.children.insert(0, vec![P]);
        let list: Vec<u64> = (0..C).map(|i| 11 + i).collect();
        for (i, &cid) in list.iter().enumerate() {
            seed_section(&mut state, cid, P, &format!("c{cid}"), i as u32 + 1);
        }
        state.children.insert(P, list);
        // The first order raise sorts P's freshly built listing out of order,
        // so the sort arm drops the cache and derives it again.
        let delta = batch(vec![
            change_cmd(P, 11, "c11", 5_000),
            change_cmd(P, 12, "c12", 5_001),
        ]);
        let (_, _, reference_stats) = reference_plan(&state, Some(delta.clone()));
        assert_eq!(
            reference_stats.parent_cache_builds, 2,
            "fixture self-check: the listing is built twice"
        );
        let mut planned = state.clone();
        let plan = apply_delta_bounded(
            &mut planned,
            Some(delta),
            priced_allowance(0),
            &mut refuse(),
            &mut ApplyDeltaStats::default(),
        )
        .expect("fixtures apply");
        let BoundedPlan::Overflow { needed } = plan else {
            panic!("no allowance overflows: got {plan:?}");
        };
        assert_eq!(
            needed.list_entries, C,
            "P's list of {C} once, though it was listed twice"
        );
    }

    /// What `Within` keeps: only what its mutation list holds.
    /// The name index's sections and a projection the plan never retires —
    /// here a cascade's over-projection of a ghost id it lists, three rows
    /// for a node that emits nothing — are in the overflow need (what the
    /// plan held at its peak) and not in the kept rewrite, which equals the
    /// count of the mutations themselves. (How much of the ghost's
    /// projection is still pending at the peak depends on which unprojected
    /// emissions retired part of it first, so the ghost case asserts only
    /// that the need's rows pass the kept rows.)
    #[test]
    fn apply_delta_bounded_within_keeps_neither_the_index_nor_pending_projections() {
        const GHOST: u64 = 999;
        let n = LONG_LIST as u64;
        let mut ghostly = parent_with_children(LONG_LIST);
        ghostly
            .children
            .get_mut(&BOUNDED_PARENT)
            .expect("seeded")
            .push(GHOST);
        let two_probes = batch(vec![
            create_cmd(BOUNDED_PARENT, 99_998, "x"),
            create_cmd(BOUNDED_PARENT, 99_999, "y"),
        ]);
        // (name, state, delta, listed parents, the need's indexed sections,
        // whether rows are pending beside the kept ones at the peak)
        for (name, state, delta, listed, indexed, rows_pending) in [
            (
                "a two-probe plan (the name index)",
                parent_with_children(LONG_LIST),
                two_probes,
                vec![(BOUNDED_PARENT, n)],
                // The sections at the second probe (the parent, its children
                // and the first birth) plus the two births.
                n + 2 + 2,
                false,
            ),
            (
                "a cascade listing a ghost (three rows projected, none emitted)",
                ghostly,
                delta_delete_section(BOUNDED_PARENT),
                vec![(0, 1)],
                0,
                true,
            ),
        ] {
            let (_, want_mutations, _) = reference_plan(&state, Some(delta.clone()));
            let kept = rewrite_of(&want_mutations, Some(&delta), &listed);
            let plan_at = |bytes: u64, budget: &mut dyn RewriteBudget| {
                let mut planned = state.clone();
                apply_delta_bounded(
                    &mut planned,
                    Some(delta.clone()),
                    priced_allowance(bytes),
                    budget,
                    &mut ApplyDeltaStats::default(),
                )
                .expect("fixtures apply")
            };
            let BoundedPlan::Within { rewrite, .. } = plan_at(u64::MAX, &mut refuse()) else {
                panic!("{name}: an unbounded allowance plans within");
            };
            assert_eq!(rewrite, kept, "{name}: Within keeps the mutations' count");
            let BoundedPlan::Overflow { needed } = plan_at(0, &mut refuse()) else {
                panic!("{name}: no allowance overflows");
            };
            assert_eq!(
                (
                    needed.list_entries,
                    needed.indexed_sections,
                    needed.adoption_entries
                ),
                (kept.list_entries, indexed, 0),
                "{name}: the need carries the index the plan held"
            );
            if rows_pending {
                assert!(
                    needed.rows > kept.rows,
                    "{name}: the need carries rows projected and never emitted: {needed:?} vs \
                     {kept:?}"
                );
            } else {
                assert_eq!(needed.rows, kept.rows, "{name}");
            }
        }
    }

    /// The refusal-semantics shapes: plans whose first ask is a listing, a
    /// reset's drain, a cascade's walk, the name index and the adoption
    /// index.
    fn refusal_shapes() -> Vec<(&'static str, ObjectState, OwnedDelta)> {
        vec![
            (
                "one new child under a long list",
                parent_with_children(LONG_LIST),
                delta_create_section(BOUNDED_PARENT, 99_999, "new"),
            ),
            (
                "a reset",
                parent_with_children(LONG_LIST),
                OwnedDelta {
                    is_diff: true,
                    base_version: 0,
                    ..Default::default()
                },
            ),
            (
                "a cascade",
                parent_with_children(LONG_LIST),
                delta_delete_section(BOUNDED_PARENT),
            ),
            (
                "two probes",
                parent_with_children(LONG_LIST),
                batch(vec![
                    create_cmd(BOUNDED_PARENT, 99_998, "x"),
                    create_cmd(BOUNDED_PARENT, 99_999, "y"),
                ]),
            ),
            (
                "a re-key and a birth",
                parent_with_children(LONG_LIST),
                rekey_and_birth(true),
            ),
        ]
    }

    /// [`priced_allowance`]'s prices: what a [`RewriteCount`] costs.
    fn priced(count: RewriteCount) -> u64 {
        count.bytes(ROW_COST, LIST_ENTRY_COST, 5, 11, 0)
    }

    /// Plan `delta` on a copy of `state` from [`priced_allowance`]`(bytes)`
    /// under `budget`.
    fn plan_priced(
        state: &ObjectState,
        delta: &OwnedDelta,
        bytes: u64,
        budget: &mut dyn RewriteBudget,
    ) -> BoundedPlan {
        let mut planned = state.clone();
        apply_delta_bounded(
            &mut planned,
            Some(delta.clone()),
            priced_allowance(bytes),
            budget,
            &mut ApplyDeltaStats::default(),
        )
        .expect("fixtures apply")
    }

    /// Refusal semantics: a plan refused at its first ask counts on to its
    /// end, so its overflow need is the need a plan refused only at its last
    /// growth reports — one byte short of that need's cost — and an
    /// allowance of exactly that cost plans within, asking nothing. Over
    /// [`refusal_shapes`].
    #[test]
    fn apply_delta_bounded_need_does_not_depend_on_where_the_plan_was_refused() {
        for (name, state, delta) in refusal_shapes() {
            let BoundedPlan::Overflow { needed: at_first } =
                plan_priced(&state, &delta, 0, &mut refuse())
            else {
                panic!("{name}: no allowance overflows");
            };
            let cost = priced(at_first);
            let BoundedPlan::Overflow { needed: at_last } =
                plan_priced(&state, &delta, cost - 1, &mut refuse())
            else {
                panic!("{name}: one byte short of the need overflows");
            };
            assert_eq!(
                at_first, at_last,
                "{name}: the need does not depend on where the plan was refused"
            );
            assert!(
                matches!(
                    plan_priced(&state, &delta, cost, &mut refuse()),
                    BoundedPlan::Within { .. }
                ),
                "{name}: the need's cost plans within without asking"
            );
        }
    }

    /// [`RewriteBudget`]'s answer rules, a refusal's two spellings: an
    /// answer that does not cover what the plan asked for (`Some` of one
    /// byte less than the ask's cost) is a refusal, as `None` is. Either
    /// way the plan ends `Overflow` with the need a refusing budget's plan
    /// reports, and a budget that counts its calls is asked exactly once:
    /// after its refusal the plan never asks again. Over
    /// [`refusal_shapes`].
    #[test]
    fn rewrite_budget_short_answer_is_a_refusal_and_a_refused_budget_is_never_asked_again() {
        for (name, state, delta) in refusal_shapes() {
            let BoundedPlan::Overflow { needed: refused } =
                plan_priced(&state, &delta, 0, &mut refuse())
            else {
                panic!("{name}: no allowance overflows");
            };
            for (answer, short) in [("Some(short)", true), ("None", false)] {
                let mut calls = 0_u32;
                let mut budget = |needed: RewriteCount| {
                    calls += 1;
                    short.then(|| priced(needed) - 1)
                };
                let BoundedPlan::Overflow { needed } = plan_priced(&state, &delta, 0, &mut budget)
                else {
                    panic!("{name}, {answer}: the plan overflows; it planned within");
                };
                assert_eq!(
                    needed, refused,
                    "{name}, {answer}: the need a refusing budget's plan reports"
                );
                assert_eq!(
                    calls, 1,
                    "{name}, {answer}: the budget is asked once and never after it refused"
                );
            }
        }
    }

    /// [`RewriteBudget`]'s answer rules, a refusal mid-plan: a budget that
    /// grants exactly the plan's first `k` asks and refuses its next ends the
    /// plan `Overflow` with the same need a refusal at the first ask reports,
    /// and is asked `k + 1` times. For `k` of 1, half the plan's asks and all
    /// but its last, over the [`refusal_shapes`] that ask more than once.
    #[test]
    fn rewrite_budget_refusal_after_grants_reports_the_first_ask_refusals_need() {
        let mut several = 0;
        for (name, state, delta) in refusal_shapes() {
            let mut asks = 0_usize;
            let plan = plan_priced(&state, &delta, 0, &mut |needed: RewriteCount| {
                asks += 1;
                Some(priced(needed))
            });
            assert!(
                matches!(plan, BoundedPlan::Within { .. }),
                "{name}: fixture self-check: every ask granted plans within"
            );
            if asks < 2 {
                continue;
            }
            several += 1;
            let BoundedPlan::Overflow { needed: at_first } =
                plan_priced(&state, &delta, 0, &mut refuse())
            else {
                panic!("{name}: no allowance overflows");
            };
            let mut grants: Vec<usize> = vec![1, asks / 2, asks - 1];
            grants.dedup();
            for k in grants.into_iter().filter(|&k| k >= 1) {
                let mut calls = 0_usize;
                let plan = plan_priced(&state, &delta, 0, &mut |needed: RewriteCount| {
                    calls += 1;
                    (calls <= k).then(|| priced(needed))
                });
                let BoundedPlan::Overflow { needed } = plan else {
                    panic!(
                        "{name}: refused at ask {}: the plan overflows; it planned within",
                        k + 1
                    );
                };
                assert_eq!(
                    needed,
                    at_first,
                    "{name}: refused at ask {} of {asks}: the first ask's refusal's need",
                    k + 1
                );
                assert_eq!(
                    calls,
                    k + 1,
                    "{name}: {k} grants and one refusal, then never asked again"
                );
            }
        }
        assert!(
            several >= 3,
            "fixture self-check: at least three shapes ask more than once ({several})"
        );
    }

    /// A reset projects its drain before its first emission. From an
    /// allowance of 0, under a budget granting exactly each ask, the reset's
    /// first ask already carries three rows per section (its
    /// `DeleteSection`, `CreateTombstone` and `DeleteChildrenList`) plus the
    /// root's list delete — a plan that asked only at its emissions would
    /// first ask at the first emitted row, with 1.
    #[test]
    fn apply_delta_bounded_reset_asks_for_its_drain_before_it_emits() {
        let state = parent_with_children(LONG_LIST);
        let sections = state.sections.len() as u64;
        let (asks, plan) = plan_recording_asks(
            &state,
            OwnedDelta {
                is_diff: true,
                base_version: 0,
                ..Default::default()
            },
        );
        assert!(
            matches!(plan, BoundedPlan::Within { .. }),
            "every ask granted: {plan:?}"
        );
        let first = asks.first().expect("the reset asks").0;
        assert!(
            first.rows >= 3 * sections,
            "the first ask carries the drain's {} rows; asks: {:?}",
            3 * sections,
            &asks[..asks.len().min(4)]
        );
        assert_eq!(first.rows, 3 * sections + 1, "and the root's list delete");
    }

    /// A cascade of a 1,000-node subtree asks before its
    /// walk grows: the first asks that raise the plan's rows are, in order,
    /// the subtree root's three rows (before the walk starts) and its 999
    /// children's (before they are pushed on the walk's stack), before any
    /// cascade row is emitted — each emitted row retires one projected, so
    /// the cascade's 3,000 emissions raise nothing. The one later raise is
    /// what the plan emits beyond its projections, reaching the plan's whole
    /// rewrite. A plan that asked only at its emissions would raise its rows
    /// one emission at a time.
    #[test]
    fn apply_delta_bounded_cascade_asks_before_its_walk_grows() {
        const NODES: u64 = 1_000;
        let state = parent_with_children(NODES as usize - 1);
        let delta = delta_delete_section(BOUNDED_PARENT);
        let (_, reference_mutations, _) = reference_plan(&state, Some(delta.clone()));
        let whole = rewrite_of(&reference_mutations, Some(&delta), &[(0, 1)]);
        let (asks, plan) = plan_recording_asks(&state, delta);
        assert!(
            matches!(plan, BoundedPlan::Within { .. }),
            "every ask granted: {plan:?}"
        );
        let mut rows_raised: Vec<u64> = Vec::new();
        for (needed, _, _) in &asks {
            if rows_raised.last().is_none_or(|&last| needed.rows > last) {
                rows_raised.push(needed.rows);
            }
        }
        assert_eq!(
            rows_raised,
            vec![3, 3 * NODES, whole.rows],
            "the root's rows, then the walk's, then the rest of the plan's rewrite; asks: {:?}",
            &asks[..asks.len().min(6)]
        );
        assert!(whole.rows - 3 * NODES <= 1, "fixture self-check: {whole:?}");
    }

    /// A pin of the adoption index's "live or not" rule: the adoption index
    /// records the rows naming every candidate, live or not. A legacy row L
    /// names live P but is absent from P's list. The delta births Y (which
    /// builds the index while P is still live), deletes P (its cascade walks
    /// P's list, so L survives as an orphan) and re-creates P under the
    /// root, which adopts L: L is listed under the new P with one
    /// `SetSection`, and the batched result is the one a command-per-delta
    /// replay reaches (`Y` and the new `P` under the root, `L` alone under
    /// `P`; the listed child gone with the cascade). The replay itself
    /// cannot run here in a debug build: its first boundary, after `Y`'s
    /// birth alone, still holds the legacy row unlisted under a live parent,
    /// which the cross-map check rejects, so the replay's result is spelled
    /// out. Discriminated by building the index from rows naming an id with
    /// no row only: L is then never recorded and stays unlisted (in a debug
    /// build the forward cross-map check fires first).
    #[test]
    fn apply_delta_adoption_index_records_a_lost_row_naming_a_live_candidate() {
        const P: u64 = 2;
        const LISTED: u64 = 3;
        const LOST: u64 = 4;
        const Y: u64 = 5;
        let arrange = || {
            let mut state = make_state();
            // Version 3, so the delta runs at 4 and re-stamps every adopted
            // row the seeding left at version 1.
            state.sequence_num = 3;
            seed_section(&mut state, P, 0, "P", 0);
            seed_section(&mut state, LISTED, P, "listed", 0);
            seed_section(&mut state, LOST, P, "lost", 1);
            state.children.insert(0, vec![P]);
            state.children.insert(P, vec![LISTED]);
            state
        };
        let delta = batch(vec![
            create_cmd(0, Y, "Y"),
            delete_cmd(P),
            create_cmd(0, P, "P"),
        ]);
        let mut batched = arrange();
        let (mutations, _) = apply_counted(&mut batched, delta);

        assert_eq!(
            batched.children.get(&P),
            Some(&vec![LOST]),
            "the lost row is adopted by the re-created P"
        );
        assert_eq!(
            set_section_count(&mutations, LOST),
            1,
            "the adopted row gets one SetSection"
        );
        // The command-per-delta result: `Y` appended under the root (key
        // 1), `P` gone and re-created after it (key 2), `L` keeping its own
        // key under the new `P`, the listed child cascaded away.
        let mut replayed = make_state();
        seed_section(&mut replayed, Y, 0, "Y", 1);
        seed_section(&mut replayed, P, 0, "P", 2);
        seed_section(&mut replayed, LOST, P, "lost", 1);
        replayed.children.insert(0, vec![Y, P]);
        replayed.children.insert(P, vec![LOST]);
        assert_eq!(
            structural_shape(&batched),
            structural_shape(&replayed),
            "batched application reaches the command-per-delta result"
        );
    }

    /// The bounded sink, driven directly: within the allowance it collects;
    /// the push that passes it drops everything collected and frees the
    /// list's buffer, every later push collects nothing (its list entries
    /// still counted), and `finish` reports the whole rewrite, exact.
    #[test]
    fn bounded_sink_past_its_allowance_frees_its_list_and_collects_nothing_more() {
        let one_row = |id| StorageMutation::DeleteChildrenList { section_id: id };
        let list = |id| StorageMutation::SetChildren {
            section_id: id,
            children: vec![1, 2, 3],
        };
        // One own element: the first row is the delta's, not the rewrite's.
        let mut no_extension = refuse();
        let meter = std::cell::RefCell::new(BoundedMeter::new(
            allowance(2 * ROW_COST),
            1,
            &mut no_extension,
        ));
        let mut sink = BoundedSink { meter: &meter };
        for id in 0..3 {
            sink.push_with(|| one_row(id));
        }
        assert!(
            !meter.borrow().overflowed,
            "two rewritten rows fit two rows' bytes"
        );
        assert_eq!(meter.borrow().mutations.len(), 3);

        sink.push_with(|| one_row(3));
        assert!(
            meter.borrow().overflowed,
            "a third rewritten row passes the allowance"
        );
        assert!(
            meter.borrow().mutations.is_empty(),
            "the collected mutations are dropped"
        );
        assert_eq!(
            meter.borrow().mutations.capacity(),
            0,
            "and their buffer freed"
        );

        for id in 4..6 {
            sink.push_with(|| list(id));
            assert!(
                meter.borrow().mutations.is_empty(),
                "nothing is collected past it"
            );
            assert_eq!(
                meter.borrow().mutations.capacity(),
                0,
                "nothing is allocated past it"
            );
        }
        let BoundedPlan::Overflow { needed } = meter.into_inner().finish() else {
            panic!("a sink past its allowance finishes as an overflow");
        };
        assert_eq!(
            needed,
            RewriteCount {
                rows: 5,
                list_entries: 6,
                indexed_sections: 0,
                adoption_entries: 0,
                walked_sections: 0,
                rollback_rebuild_bytes: 0,
            },
            "every push is counted, the ones past the allowance too, less the own element"
        );
    }

    /// The same through the engine: a plan an allowance one byte short of
    /// its rewrite leaves the sink with no mutations and no buffer, and the
    /// exact need.
    #[test]
    fn apply_delta_engine_past_the_allowance_leaves_the_bounded_sink_empty() {
        for (name, state, delta, listed) in bounded_shapes() {
            let (_, want_mutations, _) = reference_plan(&state, delta.clone());
            let want_rewrite = rewrite_of(&want_mutations, delta.as_ref(), &listed);
            let exact = want_rewrite.bytes(ROW_COST, LIST_ENTRY_COST, 0, 0, 0);
            let own = delta.as_ref().map_or(0, own_element_count);
            let mut no_extension = refuse();
            let meter = std::cell::RefCell::new(BoundedMeter::new(
                allowance(exact - 1),
                own,
                &mut no_extension,
            ));
            let mut sink = BoundedSink { meter: &meter };
            apply_delta_engine(
                &mut state.clone(),
                delta,
                &mut sink,
                Some(&meter),
                &mut ApplyDeltaStats::default(),
                ApplyMode::Commit,
                NoHooks,
            )
            .expect("fixtures apply");
            let meter = meter.into_inner();
            assert!(meter.overflowed, "{name}: one byte short overflows");
            assert!(meter.mutations.is_empty(), "{name}: nothing kept");
            assert_eq!(meter.mutations.capacity(), 0, "{name}: no buffer kept");
            let BoundedPlan::Overflow { needed } = meter.finish() else {
                panic!("{name}: an overflowed sink finishes as an overflow");
            };
            assert_eq!(needed, want_rewrite, "{name}");
        }
    }

    /// A delta the engine refuses is refused the same way under a bound.
    #[test]
    fn apply_delta_bounded_refused_delta_returns_the_engines_error() {
        let delta = delta_create_section(77, 77, "self");
        let reference = apply_delta_with_stats(
            &mut parent_with_children(3),
            Some(delta.clone()),
            &mut ApplyDeltaStats::default(),
        )
        .expect_err("fixture self-check: a self-parenting create is refused");
        let bounded = apply_delta_bounded(
            &mut parent_with_children(3),
            Some(delta),
            allowance(u64::MAX),
            &mut refuse(),
            &mut ApplyDeltaStats::default(),
        )
        .expect_err("refused under a bound too");
        assert_eq!(bounded.to_string(), reference.to_string());
    }

    /// A delta's own elements are its struct commands, set fields and time
    /// samples, each counted once.
    #[test]
    fn own_element_count_sums_struct_commands_set_fields_and_time_samples() {
        let delta = OwnedDelta {
            struct_commands: vec![StructCommand::DeleteSection(OodfDeleteSection {
                parent_id: 0,
                section_id: 1,
            })],
            set_fields: vec![OodfSetField::default(); 2],
            time_samples: vec![OodfTimeSample::default(); 4],
            ..Default::default()
        };
        assert_eq!(own_element_count(&delta), 7);
        assert_eq!(own_element_count(&OwnedDelta::default()), 0);
    }

    // ── The cascade, reset and sink scratch models (b.fsz dq-1) ──────────────
    //
    // The structural cache's per-record repair charge (RP2) prices a replayed
    // record's scratch from these models before the record applies. Each is
    // checked here against what it models: `subtree_pushes` against the
    // pushes the engine's own walk makes (`collect_subtree`, counted through
    // its row projections), the byte models against the std collections the
    // engine builds, built here the same way, and the list-delete sink
    // against the collecting sink it stands in for.
    //
    // Group widths: the byte models' tables are priced with 16 trailing
    // control bytes (`alloc_model::TABLE_TRAILING_CONTROL_BYTES`), and the
    // std tables they are checked against are read through the same
    // `table_bytes`, from their real capacities; hashbrown's bucket count for
    // a capacity does not depend on its group width (8 here on aarch64, 16 on
    // x86_64), and every table here is built by inserts from empty, with no
    // tombstone. So each comparison is the same at either width, and a table
    // at width 16 allocates no more than the 16 trailing bytes priced.

    /// A plan meter that only counts the ids `collect_subtree` pushes: it
    /// projects three rows per id before it pushes it (the root, then each
    /// list's children), so a third of its row projections.
    struct PushCounter(std::cell::Cell<u64>);

    impl PlanMeter for PushCounter {
        fn project(&self, projection: Projection) {
            if let Projection::Rows(rows) = projection {
                self.0.set(self.0.get() + rows / 3);
            }
        }
    }

    /// The ids the engine's cascade walk pushes from `root`, and the ids it
    /// visits: `collect_subtree` itself, under a [`PushCounter`].
    fn engine_walk(state: &ObjectState, root: u64) -> (u64, usize) {
        let counter = PushCounter(std::cell::Cell::new(0));
        let plan = ChildPlan::new(Some(&counter), StructuralInserts::default(), NoHooks);
        let visited = collect_subtree(state, root, &plan);
        (counter.0.get(), visited.len())
    }

    /// A state with `rows` live sections under `parent` (no list entry each)
    /// and `lists` children lists, as given.
    fn shape(rows: &[(u64, u64)], lists: &[(u64, &[u64])]) -> ObjectState {
        let mut state = make_state();
        for &(id, parent) in rows {
            state
                .sections
                .insert(id, make_section(id, parent, &format!("s{id}")));
        }
        for &(parent, list) in lists {
            state.children.insert(parent, list.to_vec());
        }
        state
    }

    /// `subtree_pushes` counts the root and every list of every id it
    /// reaches, which on a tree is exactly the engine's pushes; under a
    /// ghost (a listed id with no row, whose stale list the engine still
    /// pushes) it walks the ghost's list as the engine does; on a stale DAG
    /// (an id listed under two parents) it counts the shared id's list each
    /// time it reaches it, more than the engine, which visits it once; and it
    /// answers `None` past its cap (on a cycle, which never ends, and on a
    /// tree one push over) and past `SUBTREE_WALK_FRAMES` levels below its
    /// root, but not at that depth. On every finite shape its count is at
    /// least the engine's pushes, which are at least its visits.
    #[test]
    fn subtree_pushes_bounds_the_engines_walk_and_answers_none_past_its_cap_or_depth() {
        // A tree: 1 → [2, 3]; 2 → [4, 5, 6]; 3 → [7].
        let tree = shape(
            &[(1, 0), (2, 1), (3, 1), (4, 2), (5, 2), (6, 2), (7, 3)],
            &[(1, &[2, 3]), (2, &[4, 5, 6]), (3, &[7])],
        );
        // A ghost: 1 → [2, 99], 99 has no row but a stale list → [100, 101].
        let ghost = shape(
            &[(1, 0), (2, 1), (100, 99), (101, 99)],
            &[(1, &[2, 99]), (99, &[100, 101])],
        );
        // A stale DAG: 4 listed under both 2 and 3, with a list of its own.
        let dag = shape(
            &[(1, 0), (2, 1), (3, 1), (4, 2), (5, 4), (6, 4)],
            &[(1, &[2, 3]), (2, &[4]), (3, &[4]), (4, &[5, 6])],
        );
        for (what, state, want, engine_pushes) in [
            ("a tree", &tree, 7, 7),
            ("a ghost's stale list", &ghost, 5, 5),
            ("a stale DAG", &dag, 9, 7),
        ] {
            let (pushes, visits) = engine_walk(state, 1);
            assert_eq!(
                (pushes, subtree_pushes(state, 1, u64::MAX)),
                (engine_pushes, Some(want)),
                "{what}: the engine's pushes and the walk's count"
            );
            assert!(
                want >= pushes && pushes >= visits as u64,
                "{what}: count {want} ≥ the engine's pushes {pushes} ≥ its visits {visits}"
            );
        }
        assert_eq!(subtree_pushes(&tree, 1, 7), Some(7), "at its cap");
        assert_eq!(subtree_pushes(&tree, 1, 6), None, "one push past its cap");
        assert_eq!(
            subtree_pushes(&tree, 1, 0),
            None,
            "the root alone is past a cap of 0"
        );
        assert_eq!(
            subtree_pushes(&tree, 7, u64::MAX),
            Some(1),
            "a leaf: the root alone"
        );

        // A cycle: 1 → [2], 2 → [1]. The engine ends it on its visited set;
        // the walk, which keeps none, at its cap or its depth limit.
        let cycle = shape(&[(1, 0), (2, 1)], &[(1, &[2]), (2, &[1])]);
        assert_eq!(
            engine_walk(&cycle, 1),
            (3, 2),
            "fixture: the engine ends it"
        );
        assert_eq!(subtree_pushes(&cycle, 1, 10), None, "a cycle, at the cap");
        assert_eq!(
            subtree_pushes(&cycle, 1, u64::MAX),
            None,
            "a cycle, at the depth limit"
        );

        // A chain of `n` sections, each the only child of the one before:
        // `n - 1` lists below the root, so its last list sits `n - 2` levels
        // below it.
        let chain = |n: u64| {
            let rows: Vec<(u64, u64)> = (1..=n).map(|id| (id, id - 1)).collect();
            let mut state = shape(&rows, &[]);
            for id in 1..n {
                state.children.insert(id, vec![id + 1]);
            }
            state
        };
        let frames = SUBTREE_WALK_FRAMES as u64;
        let deepest = chain(frames + 1);
        assert_eq!(
            subtree_pushes(&deepest, 1, u64::MAX),
            Some(frames + 1),
            "{frames} lists on the path: every frame used"
        );
        assert_eq!(engine_walk(&deepest, 1).0, frames + 1);
        assert_eq!(
            subtree_pushes(&chain(frames + 2), 1, u64::MAX),
            None,
            "one level deeper than its frames"
        );
    }

    /// The real peak bytes of a collection grown one element at a time from
    /// empty (capacity 0), read off its `capacity()` after each: its last
    /// allocation, or at a growth the old allocation and the new one alive
    /// at once, whichever is more. `bytes` is the allocation at a capacity.
    fn grown_peak(
        capacities: impl IntoIterator<Item = usize>,
        bytes: impl Fn(usize) -> u64,
    ) -> u64 {
        let mut peak = 0;
        let mut current = 0;
        for capacity in capacities {
            if capacity != current {
                let old = if current == 0 { 0 } else { bytes(current) };
                peak = peak.max(old + bytes(capacity));
                current = capacity;
            }
        }
        peak.max(if current == 0 { 0 } else { bytes(current) })
    }

    /// The real peak of a `HashSet<u64>` built by `n` inserts from empty.
    fn set_peak(n: u64) -> u64 {
        let mut set = std::collections::HashSet::new();
        grown_peak(
            (0..n).map(|id| {
                set.insert(id);
                set.capacity()
            }),
            |capacity| crate::domain::alloc_model::table_bytes(capacity, 8),
        )
    }

    /// The real peak of a `Vec<T>` while `n` pushes land on one of `len`
    /// items at `capacity`, beyond the buffer it starts with (its holder's,
    /// as `vec_growth_bytes` reads it): at each growth the old buffer and
    /// the new one alive at once, less the starting buffer; 0 with no
    /// growth.
    fn vec_peak<T: Default>(len: usize, capacity: usize, n: u64) -> u64 {
        let mut vec: Vec<T> = Vec::with_capacity(capacity);
        vec.resize_with(len, T::default);
        let element = std::mem::size_of::<T>() as u64;
        let bytes = |capacity: usize| allocation_bytes(capacity as u64 * element);
        let start = vec.capacity();
        let mut current = start;
        let mut peak = 0;
        for _ in 0..n {
            vec.push(T::default());
            if vec.capacity() != current {
                peak = peak.max(bytes(current) + bytes(vec.capacity()));
                current = vec.capacity();
            }
        }
        peak.saturating_sub(bytes(start))
    }

    /// `cascade_scratch_bytes` is at least what the std collections a cascade
    /// of `P` pushes builds can hold at once, each at its own peak, summed:
    /// the walk's stack at the largest of three shapes — it starts as the
    /// root's one-entry buffer and the root is popped first, so it holds at
    /// most `P - 1` ids: the root's `P - 1` children extended at once; many
    /// short lists growing it by doubling, the old buffer alive at each
    /// growth; and the root's `P - 3` children extended at once (a buffer of
    /// exactly `P - 3`) followed by a last child's list of 2, which doubles
    /// it once (`Vec` grows to max(2 × capacity, required)) with 8 (P − 3)
    /// and 16 (P − 3) B alive together — the shape the model's
    /// `allocation_bytes(16 P) + allocation_bytes(8 P)` prices, the largest
    /// at every count here from 7 up (at 1 the walk never grows its stack) —
    /// the walk's result and visited
    /// set, the plan's `deleted_lists` and `purge_ids`, the hooked
    /// touched-list set (the cascade's `P` lists and 3 the record writes),
    /// the purge's probe vector (exact length) and the name interner's free
    /// list, every live name of 2,048 pushed onto a free list of 5 at a
    /// capacity of 8 — at the pushes of a placed asset (28), a group of the
    /// dev shape (3,151), a 20,000-child section (20,001) and the sizes
    /// around them and the table boundaries (7, 897, 1,000, 1,793).
    /// `pushes` 0 prices nothing.
    ///
    /// The stack term is the one this test discriminates, by arithmetic
    /// (self-checked at the end, with no source change). At 20,001 pushes
    /// the extend-then-double stack holds 19,998 ids' buffer and its
    /// doubling to 39,996 at once — `allocation_bytes(159,984)` +
    /// `allocation_bytes(319,968)` = 163,840 + 327,680 = 491,520 B — and the
    /// collections sum to 3,039,200 B. The model is 3,039,200 B, tight: its
    /// stack term is `allocation_bytes(320,016)` + `allocation_bytes(160,008)`
    /// = 327,680 + 163,840. A stack term of 16·P — the old buffer's 163,840
    /// dropped, or the 16·P term halved to 2 × 163,840 — leaves 2,875,360 B,
    /// short by 163,840 B. The doubling form, `allocation_bytes(8) +
    /// vec_growth_bytes(1, 1, P, 8)` (16,384 ids' buffer and 32,768 alive,
    /// 393,216 B), leaves 2,940,896 B, short by 98,304 B. So each fails here
    /// (and at 28, 897, 1,000, 1,793 and 3,151 pushes as well). The model is
    /// also tight at 28, 1,000 and 3,151.
    #[test]
    fn cascade_scratch_bytes_covers_the_std_collections_a_cascade_builds() {
        let touched = 3u64;
        let names = (2_048usize, 5usize, 8usize);
        assert_eq!(cascade_scratch_bytes(0, touched, true, names), 0);
        let mut short = Vec::new();
        let mut weaker_short = [0; 3];
        for pushes in [1u64, 7, 28, 897, 1_000, 1_793, 3_151, 20_001] {
            // `collect_subtree`'s stack, at the largest of three shapes. It
            // starts as the root's one-element buffer (`vec![root]`) and the
            // root is popped first, so it holds at most `P - 1` ids.
            //
            // 1. The root's `P - 1` children extended onto it at once.
            let mut stack: Vec<u64> = vec![0];
            stack.pop();
            let before = stack.capacity();
            stack.extend_from_slice(&vec![1u64; pushes as usize - 1]);
            let single_extend = allocation_bytes(before as u64 * 8)
                + if stack.capacity() == before {
                    0
                } else {
                    allocation_bytes(stack.capacity() as u64 * 8)
                };
            // 2. Many short lists, each push growing it by doubling with the
            //    old buffer alive at each growth, up to `P - 1` ids.
            let doubling = allocation_bytes(8) + vec_peak::<u64>(0, 1, pushes - 1);
            // 3. The root's `P - 3` children extended at once (a buffer of
            //    exactly `P - 3`), then, the last child popped, its list of 2
            //    extended: `Vec` grows to max(2 × capacity, required), so the
            //    buffer doubles once, both alive.
            let extend_then_double = if pushes < 5 {
                0
            } else {
                let grown = |stack: &mut Vec<u64>, ids: usize| {
                    let old = stack.capacity();
                    stack.extend_from_slice(&vec![1u64; ids]);
                    if stack.capacity() == old {
                        0
                    } else {
                        allocation_bytes(old as u64 * 8)
                            + allocation_bytes(stack.capacity() as u64 * 8)
                    }
                };
                let mut stack: Vec<u64> = vec![0];
                stack.pop();
                let first = grown(&mut stack, pushes as usize - 3);
                stack.pop();
                first.max(grown(&mut stack, 2))
            };
            let stack_peak = single_extend.max(doubling).max(extend_then_double);
            let real = stack_peak
                + vec_peak::<u64>(0, 0, pushes)
                + set_peak(pushes)
                + set_peak(pushes)
                + set_peak(pushes)
                + set_peak(pushes + touched)
                + allocation_bytes(pushes * 8)
                + vec_peak::<crate::domain::types::NameId>(names.1, names.2, names.0 as u64);
            let modelled = cascade_scratch_bytes(pushes, touched, true, names);
            if modelled < real {
                short.push(format!("{pushes} pushes: {modelled} B < {real} B"));
            }
            // The model with weaker stack terms (the doc's arithmetic): 16·P
            // instead of 24·P, the old buffer's 8·P dropped or the 16·P term
            // halved; and the doubling form, the growth of pushes one at a
            // time from the root's buffer.
            let stack_term = allocation_bytes(pushes * 16) + allocation_bytes(pushes * 8);
            let base = modelled - stack_term;
            let weaker = [
                base + allocation_bytes(pushes * 16),
                base + 2 * allocation_bytes(pushes * 8),
                base + allocation_bytes(8) + vec_growth_bytes(1, 1, pushes, 8),
            ];
            eprintln!(
                "CASCADE-MODEL {pushes} pushes: model {modelled} B, collections {real} B; stack \
                 single extend {single_extend} B, doubling {doubling} B, extend then double \
                 {extend_then_double} B; weaker stack terms (old buffer dropped, halved, \
                 doubling form): {weaker:?} B"
            );
            for (count, weaker) in weaker_short.iter_mut().zip(weaker) {
                if weaker < real {
                    *count += 1;
                }
            }
            assert_eq!(
                modelled - cascade_scratch_bytes(pushes, touched, false, names),
                table_growth_bytes(0, 0, pushes + touched, 8),
                "{pushes} pushes: unhooked, no touched-list set"
            );
        }
        assert!(
            short.is_empty(),
            "the model covers the collections' bytes at every push count: {short:?}"
        );
        assert!(
            weaker_short.iter().all(|&count| count > 0),
            "self-check: each weaker stack term — 16·P with the old buffer dropped, 16·P \
             halved, and the doubling form — falls short of the collections at some push \
             count, so this test discriminates the 24·P term; short at (dropped, halved, \
             doubling form) = {weaker_short:?} push counts"
        );
    }

    /// `reset_scratch_bytes` is at least what the std collections a reset
    /// builds can hold at once, each at its own peak: the children keys
    /// (exact length), the plan's `parents` (a `(u64, ParentEntry)` slot per
    /// list) and `order`, its `deleted_lists` (one per section) and, hooked,
    /// the touched-list set (one per list) — at a 9,000-deep chain's
    /// sections and lists, the dev shape's and small and boundary counts.
    #[test]
    fn reset_scratch_bytes_covers_the_std_collections_a_reset_builds() {
        let mut short = Vec::new();
        for (sections, lists) in [
            (1u64, 1u64),
            (28, 7),
            (897, 897),
            (7_167, 1_216),
            (9_001, 9_001),
        ] {
            let mut parents: std::collections::HashMap<u64, ParentEntry> =
                std::collections::HashMap::new();
            let parent_slot = std::mem::size_of::<(u64, ParentEntry)>() as u64;
            let parents_peak = grown_peak(
                (0..lists).map(|id| {
                    parents.insert(id, ParentEntry::default());
                    parents.capacity()
                }),
                |capacity| crate::domain::alloc_model::table_bytes(capacity, parent_slot),
            );
            let real = allocation_bytes(lists * 8)
                + parents_peak
                + vec_peak::<u64>(0, 0, lists)
                + set_peak(sections)
                + set_peak(lists);
            let modelled = reset_scratch_bytes(sections, lists, true);
            eprintln!(
                "RESET-MODEL {sections} sections, {lists} lists: model {modelled} B, collections \
                 {real} B"
            );
            if modelled < real {
                short.push(format!(
                    "{sections} sections, {lists} lists: {modelled} B < {real} B"
                ));
            }
            assert_eq!(
                modelled - reset_scratch_bytes(sections, lists, false),
                table_growth_bytes(0, 0, lists, 8),
                "unhooked, no touched-list set"
            );
        }
        assert!(
            short.is_empty(),
            "the model covers the collections' bytes at every count: {short:?}"
        );
    }

    /// Exactly what the structural cache's dangling-parent upkeep reads off
    /// a collected mutation list: the id of each `DeleteChildrenList`, and of
    /// each `SetChildren` whose list is empty, in emission order — matched
    /// here on the variants themselves, independently of
    /// `StorageMutation::deleted_list`.
    fn collected_list_deletes(mutations: &[StorageMutation]) -> Vec<u64> {
        mutations
            .iter()
            .filter_map(|mutation| match mutation {
                StorageMutation::DeleteChildrenList { section_id } => Some(*section_id),
                StorageMutation::SetChildren {
                    section_id,
                    children,
                } if children.is_empty() => Some(*section_id),
                _ => None,
            })
            .collect()
    }

    /// The list-delete sink (`MutationSink::ListDeletes`) keeps exactly the
    /// ids a collecting sink's `DeleteChildrenList`s and empty `SetChildren`s
    /// carry, in the same order, and leaves the state exactly as the
    /// collecting sink does — for a cascade, a reset, a create-then-delete in
    /// one delta, a move that empties its parent's list, a move that keeps
    /// it, and a field write — so the structural cache's dangling-parent
    /// upkeep, which reads nothing else off a replayed record, is the same
    /// through either sink. Self-check: the cases' collected mutations carry
    /// at least one `DeleteChildrenList`, one empty `SetChildren` and one
    /// non-empty `SetChildren` (which neither keeps). This runs the unindexed
    /// engine; the indexed replay a cache hit's repair runs is compared the
    /// same way by `storage::structural_cache_repair`'s
    /// `indexed_replay_through_the_list_delete_sink_matches_the_collecting_sink`.
    #[test]
    fn list_deletes_sink_keeps_exactly_the_collecting_sinks_list_delete_ids() {
        let base = make_three_level_tree();
        let mut with_sibling = make_three_level_tree();
        with_sibling
            .sections
            .insert(4, make_section_versioned(4, 1, "Sibling", 1));
        with_sibling.children.insert(1, vec![2, 4]);
        let reset = OwnedDelta {
            is_diff: true,
            base_version: 0,
            struct_commands: vec![StructCommand::CreateSection(OodfCreateSection {
                parent_id: 0,
                section_id: 9,
                section_name: "Fresh".to_owned(),
                section_type: 0,
            })],
            ..Default::default()
        };
        let create_then_delete = OwnedDelta {
            struct_commands: vec![
                StructCommand::CreateSection(OodfCreateSection {
                    parent_id: 3,
                    section_id: 8,
                    section_name: "Transient".to_owned(),
                    section_type: 0,
                }),
                StructCommand::DeleteSection(OodfDeleteSection {
                    parent_id: 3,
                    section_id: 8,
                }),
            ],
            ..Default::default()
        };
        let field_write = OwnedDelta {
            set_fields: vec![OodfSetField {
                section_id: 2,
                key_name: "k".to_owned(),
                value_or_ext_hash: vec![1],
                ..Default::default()
            }],
            ..Default::default()
        };
        let cases: [(&str, &ObjectState, OwnedDelta); 6] = [
            ("a cascade", &base, delta_delete_section(1)),
            ("a reset", &base, reset),
            ("a create-then-delete", &base, create_then_delete),
            (
                "a move emptying its parent's list",
                &base,
                delta_move_section(3, 2, 1, "Moved"),
            ),
            (
                "a move keeping its parent's list",
                &with_sibling,
                delta_move_section(4, 1, 2, "Moved"),
            ),
            ("a field write", &base, field_write),
        ];
        let (mut deletes, mut empty_sets, mut kept_sets) = (0, 0, 0);
        for (what, state, delta) in cases {
            let mut collected_state = state.clone();
            let mut collecting = MutationSink::collecting();
            apply_delta_into(
                &mut collected_state,
                Some(delta.clone()),
                &mut collecting,
                &mut ApplyDeltaStats::default(),
            )
            .expect("fixtures apply");
            let mut listed_state = state.clone();
            let mut listing = MutationSink::list_deletes();
            apply_delta_into(
                &mut listed_state,
                Some(delta),
                &mut listing,
                &mut ApplyDeltaStats::default(),
            )
            .expect("fixtures apply");
            let MutationSink::Collect(mutations) = collecting else {
                panic!("a collecting sink");
            };
            let MutationSink::ListDeletes(ids) = listing else {
                panic!("a list-delete sink");
            };

            assert_eq!(
                ids,
                collected_list_deletes(&mutations),
                "{what}: the list-delete sink keeps the collected list deletes, in order"
            );
            assert_eq!(
                listed_state, collected_state,
                "{what}: the sink changes nothing in the state"
            );
            for mutation in &mutations {
                match mutation {
                    StorageMutation::DeleteChildrenList { .. } => deletes += 1,
                    StorageMutation::SetChildren { children, .. } if children.is_empty() => {
                        empty_sets += 1;
                    }
                    StorageMutation::SetChildren { .. } => kept_sets += 1,
                    _ => {}
                }
            }
        }
        assert!(
            deletes > 0 && empty_sets > 0 && kept_sets > 0,
            "fixture self-check: the cases emit list deletes ({deletes}), empty list writes \
             ({empty_sets}) and kept list writes ({kept_sets})"
        );
    }

    /// `StorageMutation::deleted_list` is the id of a `DeleteChildrenList`
    /// or an empty `SetChildren`, and `None` for a non-empty `SetChildren`
    /// and every other mutation.
    #[test]
    fn deleted_list_names_a_children_list_delete_or_an_empty_list_write_only() {
        assert_eq!(
            StorageMutation::DeleteChildrenList { section_id: 4 }.deleted_list(),
            Some(4)
        );
        assert_eq!(
            StorageMutation::SetChildren {
                section_id: 5,
                children: Vec::new(),
            }
            .deleted_list(),
            Some(5)
        );
        assert_eq!(
            StorageMutation::SetChildren {
                section_id: 6,
                children: vec![7],
            }
            .deleted_list(),
            None
        );
        assert_eq!(
            StorageMutation::CreateTombstone {
                section_id: 8,
                first_version: 1,
                last_version: 2,
            }
            .deleted_list(),
            None
        );
    }

    /// The list-delete sink's model: its id vector at the lists deleted
    /// inline and the touched parents, and one mutation in flight carrying
    /// at most two names of `MAX_NAME_BYTES` — its floor at no list.
    #[test]
    fn list_delete_sink_bytes_prices_its_ids_and_one_mutation_in_flight() {
        let name = allocation_bytes(crate::domain::limits::MAX_NAME_BYTES as u64);
        assert_eq!(list_delete_sink_bytes(0, 0), 2 * name, "the floor");
        for (deleted, touched) in [(1u64, 0u64), (0, 1), (3_151, 2), (20_001, 1)] {
            assert_eq!(
                list_delete_sink_bytes(deleted, touched),
                vec_growth_bytes(0, 0, deleted + touched, 8) + 2 * name,
                "{deleted} deleted, {touched} touched"
            );
            assert!(
                list_delete_sink_bytes(deleted, touched)
                    >= vec_peak::<u64>(0, 0, deleted + touched) + 2 * name,
                "{deleted} deleted, {touched} touched: covers the id vector std grows"
            );
        }
    }
}
