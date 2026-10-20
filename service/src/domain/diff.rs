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

// Diff reconstruction: build a catch-up delta for a subscriber at a given version.
// Pure domain logic — zero I/O, zero async. See SDD §10.

use std::collections::{HashMap, HashSet};

use thiserror::Error;

use crate::domain::limits::MAX_SECTION_TREE_DEPTH;
use crate::domain::types::{Field, FieldValue, ObjectState, Section, TimeSample};

// ── DiffError ────────────────────────────────────────────────────────────────

/// Errors produced by [`DiffEngine::plan`] when the section tree violates a
/// hard invariant. All variants indicate malformed input (e.g. a section-tree
/// depth that would previously have blown the thread stack).
#[derive(Debug, Error, PartialEq, Eq)]
pub enum DiffError {
    /// The traversed section-tree depth exceeded [`MAX_SECTION_TREE_DEPTH`].
    ///
    /// A recursive DFS would blow the thread stack on the same input; the
    /// iterative walker returns this error instead. Sized to absorb any
    /// realistic depth while capping producer bugs / malicious input.
    #[error("section tree depth {depth} exceeds cap {cap}")]
    SectionTreeTooDeep { depth: usize, cap: usize },
}

// ── DiffPlan ─────────────────────────────────────────────────────────────────

/// The result of planning a diff for a subscriber.
///
/// Routing follows SDD §10.1:
/// - Case 1: object_id mismatch or have_version == 0 → FullContent
/// - Case 2: same object_id, have_version < current  → Incremental
/// - Case 3: have_version >= current_version          → UpToDate
#[derive(Debug, Clone, PartialEq)]
pub enum DiffPlan {
    /// Subscriber is current — nothing to send (Case 3).
    UpToDate,

    /// Full content delivery: object_id mismatch or have_version=0 (Case 1).
    /// Sections are in DFS order over the children lists. From
    /// [`DiffEngine::plan_faithful_snapshot`]: reachable sections only, each
    /// after its parent on any stored shape. From the legacy
    /// `plan_full_content`: each after the section whose list names it, which
    /// on a stale listing stored before b.xje is not its parent, with orphans
    /// appended after the reachable sections.
    FullContent {
        sections: Vec<Section>,
        fields: Vec<Field>,
        time_samples: Vec<TimeSample>,
    },

    /// Incremental diff: same object_id, subscriber is behind (Case 2).
    Incremental {
        /// Sections with first_version > have_version, in DFS order.
        new_sections: Vec<Section>,
        /// Existing sections (`first_version <= have_version`) the subscriber
        /// must re-apply. From the legacy `plan_incremental`: those with
        /// `last_version > have_version`. From
        /// [`DiffEngine::plan_faithful_coalesced`]: those, plus every other
        /// listed child of a parent that has an emitted (new or changed)
        /// child, so that parent's whole children sequence is restated.
        modified_sections: Vec<Section>,
        /// Tombstones for sections the subscriber knew about that were deleted since.
        /// Each entry is (section_id, first_version, last_version).
        deleted_sections: Vec<(u64, u64, u64)>,
        /// Fields with version > have_version.
        changed_fields: Vec<Field>,
        /// Time samples with version > have_version.
        changed_time_samples: Vec<TimeSample>,
    },
}

// ── DiffEngine ───────────────────────────────────────────────────────────────

pub struct DiffEngine;

/// [`DiffEngine::faithful_walk`]'s result: the walked sections in DFS
/// parent-first order, their `section_id`s, and each one's position among
/// its parent's true children.
type FaithfulWalk = (Vec<Section>, HashSet<u64>, HashMap<u64, u32>);

impl DiffEngine {
    /// Plan a diff for a subscriber who reports `(have_object_id, have_version)`.
    ///
    /// `tombstones` are `(section_id, first_version, last_version)` tuples loaded
    /// from storage — they are not part of ObjectState because deleted sections
    /// are no longer in the live section index.
    ///
    /// `upper_version` optionally bounds the diff window. `None` emits everything
    /// newer than `have_version`. `Some(u)` filters to the half-open interval
    /// `(have_version, u]` — any change with version > u is skipped. Used by the
    /// per-commit replay path (`catch_up_in_window`'s per-commit reads) so
    /// that the diff labelled for a given sequence_num doesn't accidentally
    /// include content from sequences committed after it under a write burst.
    pub fn plan(
        state: &ObjectState,
        tombstones: &[(u64, u64, u64)],
        have_object_id: u64,
        have_version: u64,
        upper_version: Option<u64>,
    ) -> Result<DiffPlan, DiffError> {
        // Case 1: object_id mismatch or subscriber at version 0 → full content.
        // FullContent ignores the upper bound — it's a cold-start delivery and
        // the caller (Regime 1 / Regime 3) wants everything currently stored.
        if have_object_id != state.object_id || have_version == 0 {
            return Self::plan_full_content(state);
        }

        // Case 3: subscriber is up to date (or ahead — treat the same).
        if have_version >= state.sequence_num {
            return Ok(DiffPlan::UpToDate);
        }

        // Case 2: incremental diff.
        Self::plan_incremental(state, tombstones, have_version, upper_version)
    }

    /// Case 1: emit everything in DFS order, orphans after reachable sections.
    ///
    /// **Every one of the three emitted collections is a TOTAL order over the
    /// CONTENT of [`ObjectState`]** — not over any `HashMap`'s internal table
    /// layout (b.gcb / b.vz7):
    ///
    /// - `sections` — the reachable-from-root DFS prefix (deterministic: it walks
    ///   `state.children`'s authored `Vec`s) followed by the orphan tail sorted on
    ///   `section_id`, the `state.sections` map key. See
    ///   [`sections_in_dfs_order`](Self::sections_in_dfs_order).
    /// - `fields` — sorted on `(version, authoring_set_order, section_id,
    ///   key_name)` via the shared
    ///   [`sort_fields_by_authoring`](Self::sort_fields_by_authoring) comparator.
    ///   `(section_id, key_name)` is the `ObjectState::fields` map key, so the
    ///   4-tuple is total even when a non-conforming writer reuses a raw
    ///   authoring rank within one delta.
    /// - `time_samples` — sorted on `(version, section_id, time)`, the key both
    ///   faithful builders already use; `(section_id, time)` is the
    ///   `ObjectState::time_samples` map key, so the triple is total.
    ///
    /// **SORT ONLY — the cumulative `set_order` is preserved verbatim.** This
    /// path deliberately does NOT call
    /// [`dense_reindex_by_authoring`](Self::dense_reindex_by_authoring), which
    /// overwrites `set_order` with the dense emit index. Two consumers depend on
    /// the cumulative cursor surviving: the checkpoint assembler
    /// (`build_reset_ompf_from_state` overwrites the slot with
    /// `authoring_set_order` itself) and the legacy catch-up cold-start wire,
    /// which emits the stored cursor.
    ///
    /// **Why the total order matters (SR-9.2).** Both consumers of this plan —
    /// the checkpoint base assembly and `DiffEngine::plan`'s `have_version == 0`
    /// cold-start branch — previously inherited `HashMap` iteration order, so the
    /// same logical snapshot produced byte-divergent (occasionally
    /// different-LENGTH, via FlatBuffers vtable sharing) base OMPF blobs and a
    /// byte-divergent ext-refs sidecar across replicas. The sidecar's
    /// first-appearance hash order arrives here transitively:
    /// [`collect_external_value_hashes`] scans `fields` then `time_samples` into
    /// ONE shared pool, which is why the time-sample sort is load-bearing for the
    /// sidecar and not merely cosmetic. With the total order, identical state ⇒
    /// identical bytes ⇒ a re-base PUT at the same key is byte-idempotent, giving
    /// SR-9.2's cross-replica duplicate-leader argument a structural ground
    /// rather than a set-semantic one.
    ///
    /// **Scope fence.** [`plan_incremental`](Self::plan_incremental)'s
    /// `modified_sections` / `changed_fields` / `changed_time_samples` stay
    /// unsorted: it is not a checkpoint consumer, and reordering the legacy
    /// incremental catch-up wire is a separate parity surface (deliberately left
    /// alone by b.3pe).
    fn plan_full_content(state: &ObjectState) -> Result<DiffPlan, DiffError> {
        let sections = Self::sections_in_dfs_order(state)?;

        let mut fields: Vec<Field> = state.fields.values().cloned().collect();
        Self::sort_fields_by_authoring(&mut fields);

        let mut time_samples: Vec<TimeSample> = state.time_samples.values().cloned().collect();
        time_samples.sort_by_key(|ts| (ts.version, ts.section_id, ts.time));

        Ok(DiffPlan::FullContent {
            sections,
            fields,
            time_samples,
        })
    }

    /// Case 2: filter by version, emit new sections in DFS order.
    ///
    /// When `upper_version` is `Some(u)`, every version-indexed filter also
    /// caps at `u` so the diff reflects only commits in `(have_version, u]`.
    /// `None` preserves the open-ended behavior — everything newer than
    /// `have_version` is emitted.
    ///
    /// **Deliberately outside the b.gcb/b.vz7 determinism scope.**
    /// `modified_sections` / `changed_fields` / `changed_time_samples` are
    /// collected in `HashMap` iteration order and stay that way: this path is not
    /// a checkpoint consumer, and reordering the legacy incremental catch-up wire
    /// is a separate parity surface that b.3pe left alone. Only `new_sections`
    /// gains determinism, transitively, because it filters
    /// [`sections_in_dfs_order`](Self::sections_in_dfs_order)'s output.
    fn plan_incremental(
        state: &ObjectState,
        tombstones: &[(u64, u64, u64)],
        have_version: u64,
        upper_version: Option<u64>,
    ) -> Result<DiffPlan, DiffError> {
        // Helper: `true` if `v` is within the upper bound (or unbounded).
        let within_upper = |v: u64| upper_version.is_none_or(|u| v <= u);

        // -- Sections (3 passes per SDD §10.1 Case 2) --

        // Collect new section IDs for DFS-ordered emission.
        // A section is "new to the subscriber" if first_version > have_version.
        // Under an upper bound, also require the section to have been created
        // at or before `u` — sections created in commits > u belong to a later
        // per-commit diff, not this one.
        let new_section_ids: HashSet<u64> = state
            .sections
            .values()
            .filter(|s| s.first_version > have_version && within_upper(s.first_version))
            .map(|s| s.section_id)
            .collect();

        // New sections in DFS order (parent-first per §10.3, except on a stale
        // listing stored before b.xje; see `reachable_sections_in_dfs_order`).
        let all_dfs = Self::sections_in_dfs_order(state)?;
        let new_sections: Vec<Section> = all_dfs
            .into_iter()
            .filter(|s| new_section_ids.contains(&s.section_id))
            .collect();

        // Modified sections: existed before have_version but changed since.
        // Under an upper bound, the latest modification must be at or before u
        // — otherwise it belongs to a later per-commit diff.
        let modified_sections: Vec<Section> = state
            .sections
            .values()
            .filter(|s| {
                s.first_version <= have_version
                    && s.last_version > have_version
                    && within_upper(s.last_version)
            })
            .cloned()
            .collect();

        // Deleted sections: tombstones the subscriber needs to know about.
        // The subscriber knew the section (first_version <= have_version) and
        // it was deleted after their version (last_version > have_version).
        // Under an upper bound, the delete must have happened at or before u.
        let deleted_sections: Vec<(u64, u64, u64)> = tombstones
            .iter()
            .filter(|(_, first_ver, last_ver)| {
                *first_ver <= have_version && *last_ver > have_version && within_upper(*last_ver)
            })
            .copied()
            .collect();

        // -- Fields --
        let changed_fields: Vec<Field> = state
            .fields
            .values()
            .filter(|f| f.version > have_version && within_upper(f.version))
            .cloned()
            .collect();

        // -- Time samples --
        let changed_time_samples: Vec<TimeSample> = state
            .time_samples
            .values()
            .filter(|ts| ts.version > have_version && within_upper(ts.version))
            .cloned()
            .collect();

        Ok(DiffPlan::Incremental {
            new_sections,
            modified_sections,
            deleted_sections,
            changed_fields,
            changed_time_samples,
        })
    }

    /// Build a faithful full-state snapshot for a from-empty (cold-start)
    /// joiner — the corrected catch-up path (SDD §12.9, SR-5.2/SR-5.6).
    ///
    /// Distinct from [`plan_full_content`](Self::plan_full_content), which stays
    /// **lossy** and is **stable for a stated reason**. The two words are
    /// separate claims, and only the first is a prohibition:
    ///
    /// - *Lossy* — no net-live pruning, no dense re-index, cumulative `set_order`
    ///   emitted verbatim. Still true, and still binding: the checkpoint
    ///   assembler's `build_reset_ompf` depends on it, so the lossy semantics and
    ///   the cumulative `set_order` MUST NOT change.
    /// - *Stable* — the emitted order. This used to be accidentally FALSE (it was
    ///   `HashMap` iteration order); b.gcb/b.vz7 made it true by giving all three
    ///   emitted collections a total order over state content. The "MUST NOT
    ///   change" above does NOT scope to the emission order — that order is now
    ///   the deliberate ground for base byte-idempotency (SR-9.2).
    ///
    /// This builder corrects three defects for the empty-tree joiner:
    ///
    /// 1. **Net-live (abandoned-prim) pruning.** Only sections reachable from
    ///    root (`parent_id = 0`) through the
    ///    [`faithful_walk`](Self::faithful_walk), which follows each section
    ///    only from its own parent's list, are emitted. Scratch prims that were
    ///    created and then reparented / detached mid-session — leaving them unreachable
    ///    from root — are DROPPED, not appended as orphans. A from-empty client
    ///    cannot faithfully apply an abandoned subtree. Liveness is DERIVED here
    ///    from the current section tree (`live_prim_ids`, the reachable-from-root
    ///    set) — there is NO stored per-prim tombstone/flag; the walk is bounded
    ///    at O(current live sections), satisfying SR-5.6. (`live_prim_ids` is the
    ///    per-prim net-live concept; deliberately NOT named `superseded`, which
    ///    is the unrelated object-level CreateLayer-overwrite marker — SR-5.2.)
    /// 2. **Dense-unique `set_order`.** Each field is emitted with a dense
    ///    unique index `0..N-1`, assigned by sorting the emitted net-live
    ///    fields on `(version, authoring_set_order)` and taking each field's
    ///    0-based position. This reproduces the legacy C++ catch-up's RELATIVE
    ///    field ordering (its global authoring counter is monotonic in
    ///    `(edit-version, raw-authoring-rank)`) while GUARANTEEING GLOBAL
    ///    UNIQUENESS across the whole catch-up — a requirement the from-empty
    ///    Kit client enforces via an ordered `map` keyed on `(saveCount,
    ///    setOrder)` (all catch-up fields share one `saveCount`), where a
    ///    duplicate `setOrder` silently drops the field. Neither the raw
    ///    `authoring_set_order` (per-delta rank; resets each delta → collides
    ///    across prims) nor the cumulative `set_order` cursor is emitted. See
    ///    the b.bdx note below.
    ///
    /// 3. **Real sibling `section_order`.** Each emitted section carries its
    ///    0-based position among its parent's true children, in the parent's
    ///    one children list with all child kinds together (the
    ///    [`faithful_walk`](Self::faithful_walk)'s position map), rather than
    ///    the stored `section_order`, matching the C++ catch-up so the
    ///    from-empty client reconstructs xformOp order (translate → rotate →
    ///    scale). Oracle-verified byte-exact against the committed C++
    ///    catch-up signature (b.bdx). On a list carrying no entry the walk
    ///    skips, the position equals the entry's index in the list.
    ///
    /// **b.bdx set_order — why dense-unique, not the raw C++ integers.**
    /// The legacy C++ catch-up emits a *global authoring counter*
    /// `set_order = base_delta + raw_rank`, where `base_delta` is the running
    /// count of authoring operations in all prior deltas (cumulative
    /// `max_raw_rank + 1`) captured at the delta's start. Those exact integers
    /// are NOT reconstructable from a materialized `ObjectState` — deltas whose
    /// fields were all later overwritten leave no trace of their `max_raw_rank`,
    /// so `base_delta` for later deltas is unrecoverable. But the exact
    /// integers do not matter: the Kit client re-bases every incoming
    /// `setOrder` onto its own running counter, so it needs only (1) global
    /// uniqueness and (2) the correct RELATIVE order (the apply order). C++'s
    /// global counter is monotonic in `(edit-version, raw-authoring-rank)` =
    /// `(field.version, field.authoring_set_order)`, so sorting the emitted
    /// fields on that key and assigning dense indices `0..N-1` yields the SAME
    /// field sequence C++ emits, compressed to dense values — reproducing the
    /// relative order exactly and guaranteeing uniqueness (the dense index is
    /// unique by construction). Emitting the raw `authoring_set_order` instead
    /// would collide (it resets per delta, so the first field of every batch is
    /// `0`), and the cumulative `set_order` cursor is a per-field running
    /// accumulator (b.qw6) that inflates within a delta (e.g. root `upAxis`
    /// authoring-rank 4 → stored 14) and is sensitive to field-vector order —
    /// neither is emitted. This is EMIT-TIME only: no new per-field value is
    /// persisted, and the write paths / checkpoint round-trip are untouched
    /// (`authoring_set_order` persistence remains, now consumed as the
    /// within-version sort key here).
    ///
    /// No tombstones / deleted-section entries are produced — a from-empty
    /// joiner needs none. External-value references inside fields / time samples
    /// are preserved unchanged so the storage layer can attach blobs.
    pub fn plan_faithful_snapshot(state: &ObjectState) -> Result<DiffPlan, DiffError> {
        // Parent-first walk over true children only, with each emitted
        // section's 0-based position among its parent's true children.
        let (dfs_sections, live_prim_ids, positions) = Self::faithful_walk(state)?;

        // Emit each section's position in `section_order` — matching the
        // legacy C++ catch-up, which the from-empty client needs for correct
        // xformOp ordering. See `walked_position`.
        let sections: Vec<Section> = dfs_sections
            .into_iter()
            .map(|s| Section {
                section_order: Self::walked_position(&positions, &s),
                ..s
            })
            .collect();

        // Fields only for reachable (net-live) prims, re-indexed with the
        // shared dense-unique emit-time `set_order` rule (see the b.bdx note in
        // this function's doc comment and
        // [`dense_reindex_by_authoring`](Self::dense_reindex_by_authoring) for
        // the full rationale). Emit-time ONLY — nothing new is persisted.
        let live_fields: Vec<Field> = state
            .fields
            .values()
            .filter(|f| live_prim_ids.contains(&f.section_id))
            .cloned()
            .collect();
        let fields = Self::dense_reindex_by_authoring(live_fields);

        // Time samples only for reachable prims; external refs untouched.
        // Sorted on `(version, section_id, time)` — `(section_id, time)` is the
        // map key, so the triple is a total order and the emitted sequence is
        // deterministic rather than `HashMap`-iteration-dependent. Ordering
        // only; time samples carry no emitted ordering slot to change.
        let mut time_samples: Vec<TimeSample> = state
            .time_samples
            .values()
            .filter(|ts| live_prim_ids.contains(&ts.section_id))
            .cloned()
            .collect();
        time_samples.sort_by_key(|ts| (ts.version, ts.section_id, ts.time));

        Ok(DiffPlan::FullContent {
            sections,
            fields,
            time_samples,
        })
    }

    /// Build a faithful **coalesced incremental** diff for a reconnector — the
    /// corrected catch-up path for BOTH reconnect regimes (Plan b.msb, Task 2;
    /// SDD §12.3 Regimes 2 & 3 / §12.9). Under the shipped default
    /// `LIVE_EDIT_CATCHUP_SNAPSHOT_ENABLED=true` the in-window (Regime 2) and
    /// out-of-window (Regime 3) paths CONVERGE on this builder — the only
    /// surviving distinction is the regime metric label (see the dispatch in
    /// `src/grpc/subscribe.rs::catch_up_in_window`). `false` reverts Regime 2 to
    /// legacy per-commit replay and Regime 3 to the legacy coalesced diff.
    ///
    /// The reconnector has, by definition, `0 < have_version < current_seq` on
    /// the SAME `object_id`, so its per-object sequence HAS advanced past 0. It
    /// therefore CANNOT be handed the from-empty
    /// [`plan_faithful_snapshot`](Self::plan_faithful_snapshot)
    /// (`base_version = 0`) — the client rejects a `base=0` reset once its
    /// sequence has advanced (SDD §12.3, §12.9; the Kit client's behaviour at
    /// the revision recorded in `docs-internal/cross-repo-debugging.md`).
    /// This builder therefore emits a `DiffPlan::Incremental` so the
    /// storage layer keeps the exact client-accepted envelope today's legacy
    /// coalesced diff uses (`is_diff = true`, `base_version = have_version`) —
    /// it corrects the CONTENT inside that envelope, not the envelope.
    ///
    /// Three corrections over the legacy version-diffing
    /// [`plan_incremental`](Self::plan_incremental):
    ///
    /// 0. **Real sibling `section_order`, restated per touched parent.** Every
    ///    emitted section (new and modified) carries its 0-based position
    ///    among its parent's true children, in the parent's one children list
    ///    with all child kinds together — the same position map the
    ///    from-empty builder emits (b.bdx; [`faithful_walk`](Self::faithful_walk)).
    ///    A parent is *touched* when one of its walked children is new or
    ///    changed since `have_version`; `modified_sections` then carries every
    ///    other walked child of that parent too, changed or not. Positions are
    ///    not keys: the Kit client reads a diff's `section_order` as a key with
    ///    displacement and the C++ client library reads it as an index, and a position
    ///    for only the changed children places them differently under the two
    ///    readers whenever the client's own keys have gaps (a Kit delete leaves
    ///    one). Restating the whole sequence gives every child of the parent a
    ///    unique, increasing value, so both readers end with the server's list
    ///    whatever keys the client held. A parent that only lost children is
    ///    not touched: removal keeps the remaining children's relative order
    ///    under both readers. Each section is emitted at most once. A restated
    ///    section existed at `have_version` and is still live, so neither
    ///    retraction arm below names it: arm (b) takes only sections that are
    ///    not live, and an id arm (a) names was deleted after `have_version`,
    ///    so a live row under it was re-created since and is new, not
    ///    restated (that re-created case is SDD §9.2's Known gap).
    ///
    /// 1. **Net-live pruning + unified ghost reconciliation.** Liveness is
    ///    DERIVED from the current section tree via the
    ///    [`faithful_walk`](Self::faithful_walk) (same
    ///    `MAX_SECTION_TREE_DEPTH` bound → bounded memory). Only
    ///    net-live prims — and the fields / time samples anchored to them — are
    ///    emitted as change content; abandoned / unreachable scratch prims are
    ///    NEVER emitted as live content. `deleted_sections` is the UNIFIED ghost
    ///    set: it retracts (via `DiffDeleteSection`) every section the
    ///    reconnector could still hold (`first_version <= have_version`) that is
    ///    no longer net-live — a SUPERSET of the legacy tombstone-only
    ///    retraction, covering BOTH explicitly-deleted sections (tombstones, as
    ///    the legacy path does) AND sections that dropped out of the reachable
    ///    set during the trimmed window (which the legacy version-diff leaves as
    ///    stale ghosts). Sections first created AFTER `have_version` that are now
    ///    unreachable are simply DROPPED (the client never had them; nothing to
    ///    retract) — matching the from-empty builder's drop semantics.
    ///
    /// 2. **Dense-unique `set_order` — the SAME rule the from-empty builder
    ///    uses.** Changed fields are re-emitted through the shared
    ///    [`dense_reindex_by_authoring`](Self::dense_reindex_by_authoring)
    ///    helper, exactly as
    ///    [`plan_faithful_snapshot`](Self::plan_faithful_snapshot) does (b.3pe) —
    ///    the two builders are CONVERGED on this rule via that single
    ///    implementation; there is no divergence to preserve. The helper's doc
    ///    carries the full rationale. What is specific to THIS builder: emitting
    ///    the stored cumulative `set_order` verbatim (which the legacy coalesced
    ///    diff does) re-orders every multi-field commit collapsed into this
    ///    window, because that cursor encodes WIRE-ARRIVAL order and the Kit
    ///    writer serializes a commit's fields out of a `std::unordered_map` —
    ///    hash-bucket order, uncorrelated with the writer's declared
    ///    `authoring_set_order`.
    ///
    /// The in-place `base = have_version` reconciliation was validated live
    /// against two real Kit clients on 2026-08-24 (b.vsw, closed PASSED).
    /// Caveat: that pass predates the b.3pe dense re-index above and does not
    /// cover it — the post-b.3pe live re-check is intentionally untracked (the
    /// owner verifies at the next deploy), and the re-index's standing evidence
    /// is the offline captured-traffic golden replay at a mid-window reconnect.
    /// The fallback if a future live run disproves the in-place path (a forced
    /// clean re-bootstrap, needing a client mid-stream discard affordance or a
    /// fresh `object_id` generation) is deliberately not implemented here. See
    /// the wiring site in `src/grpc/subscribe.rs`.
    ///
    /// Kept DISTINCT from both `plan_incremental` (byte-stable for the legacy
    /// path) and `plan_faithful_snapshot`; it shares the
    /// [`faithful_walk`](Self::faithful_walk) and
    /// [`dense_reindex_by_authoring`](Self::dense_reindex_by_authoring) with
    /// the latter.
    pub fn plan_faithful_coalesced(
        state: &ObjectState,
        tombstones: &[(u64, u64, u64)],
        have_version: u64,
    ) -> Result<DiffPlan, DiffError> {
        // Net-live section set (reachable-from-root through true children
        // only), DFS parent-first order, with each section's position.
        let (live_sections, live_prim_ids, positions) = Self::faithful_walk(state)?;

        // Touched parents: those with a walked child that is new or changed
        // since have_version. Each one's whole children sequence is restated.
        let touched_parents: HashSet<u64> = live_sections
            .iter()
            .filter(|s| s.first_version > have_version || s.last_version > have_version)
            .map(|s| s.parent_id)
            .collect();

        // New sections: net-live, created after have_version, in DFS order.
        // Each carries its position in `section_order` (see `walked_position`).
        let new_sections: Vec<Section> = live_sections
            .iter()
            .filter(|s| s.first_version > have_version)
            .map(|s| Section {
                section_order: Self::walked_position(&positions, s),
                ..s.clone()
            })
            .collect();

        // Modified sections: net-live, existed at/before have_version, and
        // either changed since or a child of a touched parent. Disjoint from
        // `new_sections` on `first_version`, and the walk yields each section
        // once, so no id is emitted twice. DFS order (from `live_sections`), so
        // a touched parent's children follow in list order. Same position
        // override as above.
        let modified_sections: Vec<Section> = live_sections
            .iter()
            .filter(|s| {
                s.first_version <= have_version
                    && (s.last_version > have_version || touched_parents.contains(&s.parent_id))
            })
            .map(|s| Section {
                section_order: Self::walked_position(&positions, s),
                ..s.clone()
            })
            .collect();

        // Unified ghost set. Two sources, disjoint as filtered: a deleted
        // section leaves the live index, and a `section_id` re-created after
        // its deletion (reuse is allowed, SDD §9.2) has `first_version` at or
        // above its old tombstone's `last_version`, so no id passes both
        // filters. Such an id can still be a new live section AND an arm-(a)
        // ghost in the same diff (SDD §9.2).
        //
        //   (a) Tombstoned deletes — SAME filter the legacy `plan_incremental`
        //       uses: the subscriber knew the section (first <= have_version)
        //       and it was deleted after their version (last > have_version).
        //   (b) Abandoned / detached-to-unreachable prims — sections still in
        //       the live index but NOT reached by the faithful walk, that the
        //       reconnector could still hold (first_version <= have_version).
        //       That includes a section stored before b.xje that is listed only
        //       under a parent other than its own. Their version tuple is
        //       sourced from the live section row. This is the arm the legacy
        //       version-diff omits, leaving stale ghosts.
        let mut deleted_sections: Vec<(u64, u64, u64)> = tombstones
            .iter()
            .filter(|(_, first_ver, last_ver)| {
                *first_ver <= have_version && *last_ver > have_version
            })
            .copied()
            .collect();
        deleted_sections.extend(
            state
                .sections
                .values()
                .filter(|s| {
                    s.first_version <= have_version && !live_prim_ids.contains(&s.section_id)
                })
                .map(|s| (s.section_id, s.first_version, s.last_version)),
        );
        // Neither arm arrives in a stable order — (a) is materialized from a
        // Valkey SET read, (b) from a `HashMap` — so the unified set is sorted
        // on `section_id` (unique across both arms, since the two are disjoint)
        // to make the emitted retraction sequence deterministic for identical
        // state. Ordering only; no emitted value changes, and the byte-stable
        // legacy `plan_incremental` is untouched.
        deleted_sections.sort_unstable();

        // Changed fields: newer than have_version AND anchored to a net-live
        // prim (drop those anchored to now-unreachable prims). Re-indexed with
        // the SAME shared dense-unique emit-time rule the from-empty builder
        // uses — the single call site below is what enforces that convergence
        // (b.3pe); see [`dense_reindex_by_authoring`](Self::dense_reindex_by_authoring)
        // for the rationale. Emit-time only — nothing new is persisted.
        let changed_fields: Vec<Field> = state
            .fields
            .values()
            .filter(|f| f.version > have_version && live_prim_ids.contains(&f.section_id))
            .cloned()
            .collect();
        let changed_fields = Self::dense_reindex_by_authoring(changed_fields);

        // Changed time samples: same net-live + version filter; external refs
        // untouched so the storage layer can attach blobs. Sorted on
        // `(version, section_id, time)` — `(section_id, time)` is the map key,
        // so the triple is a total order and the emitted sequence is
        // deterministic rather than `HashMap`-iteration-dependent. No emitted
        // VALUE changes here; time samples carry no ordering slot.
        let mut changed_time_samples: Vec<TimeSample> = state
            .time_samples
            .values()
            .filter(|ts| ts.version > have_version && live_prim_ids.contains(&ts.section_id))
            .cloned()
            .collect();
        changed_time_samples.sort_by_key(|ts| (ts.version, ts.section_id, ts.time));

        Ok(DiffPlan::Incremental {
            new_sections,
            modified_sections,
            deleted_sections,
            changed_fields,
            changed_time_samples,
        })
    }

    /// Assign the emit-time **dense unique `set_order`** both faithful catch-up
    /// builders use, returning the fields in the order they must be emitted.
    ///
    /// The single implementation shared by
    /// [`plan_faithful_snapshot`](Self::plan_faithful_snapshot) and
    /// [`plan_faithful_coalesced`](Self::plan_faithful_coalesced) (b.3pe) — the
    /// two builders' convergence on this rule is enforced HERE, by both calling
    /// this function, not by parallel prose in two doc comments.
    ///
    /// **The rule.** Sort on the composite key `(version, authoring_set_order,
    /// section_id, key_name)` — via the shared
    /// [`sort_fields_by_authoring`](Self::sort_fields_by_authoring) comparator,
    /// which [`plan_full_content`](Self::plan_full_content) also calls (sort-only,
    /// no re-index) — then overwrite each field's `set_order` with its 0-based
    /// position `0..N-1`. This function is exactly "that comparator, plus the
    /// dense re-index"; the ordering rationale below is the comparator's, and the
    /// re-index rationale further down is this function's alone.
    ///
    /// **Why that ordering key.** The legacy C++ catch-up emits a *global
    /// authoring counter* that is monotonic in `(edit-version,
    /// raw-authoring-rank)` — precisely `(field.version,
    /// field.authoring_set_order)` — so sorting on that pair reproduces C++'s
    /// RELATIVE field sequence exactly. The `(section_id, key_name)` suffix is
    /// load-bearing for determinism, not for parity: on conforming input each
    /// `SetField` in one commit carries a distinct raw rank (across commits the
    /// version differs), so the leading pair already orders the fields — but the
    /// server does NOT enforce within-delta `set_order` uniqueness
    /// (`authoring_set_order` is copied verbatim from the writer, with only an
    /// upper-bound check), so a non-conforming writer that reused a raw rank
    /// across two `SetField`s in one delta would tie on the pair and fall back to
    /// `HashMap` iteration order — a non-deterministic wire order across
    /// catch-ups and processes. `(section_id, key_name)` is the
    /// [`ObjectState::fields`](crate::domain::types::ObjectState) map key and is
    /// therefore unique, making the 4-tuple a TOTAL order; the sort can be
    /// unstable without ambiguity.
    ///
    /// **Why dense indices rather than either persisted rank.** The Kit client
    /// keys catch-up fields in an ordered `std::map<pair<saveCount, setOrder>>`
    /// where every field in one catch-up shares a single `saveCount`: a
    /// duplicate `setOrder` collides on `emplace` and the field is silently
    /// dropped (release) / asserts (debug). It also re-bases every incoming
    /// `setOrder` onto its own running counter, so only RELATIVE order and
    /// uniqueness are observable — never the absolute integers. A dense index is
    /// unique by construction. Emitting the raw `authoring_set_order` instead
    /// would collide massively (it resets per delta, so the first field of every
    /// edit-batch is `0`); emitting the stored cumulative `set_order` cursor is
    /// also wrong (a per-field running accumulator, b.qw6, that neither
    /// byte-matches the C++ counter nor reliably preserves authoring order).
    ///
    /// EMIT-TIME ONLY: nothing new is persisted, and the write paths /
    /// checkpoint round-trip are untouched — the dense index is computed here
    /// from the already-persisted `(version, authoring_set_order)`.
    fn dense_reindex_by_authoring(mut fields: Vec<Field>) -> Vec<Field> {
        Self::sort_fields_by_authoring(&mut fields);
        for (idx, field) in fields.iter_mut().enumerate() {
            field.set_order = idx as u64;
        }
        fields
    }

    /// Sort `fields` into the canonical authoring order
    /// `(version, authoring_set_order, section_id, key_name)` — **ordering only,
    /// no field value is touched**.
    ///
    /// The single implementation of that comparator, shared by
    /// [`dense_reindex_by_authoring`](Self::dense_reindex_by_authoring) (which
    /// then overwrites `set_order` with the dense emit index) and
    /// [`plan_full_content`](Self::plan_full_content) (which must PRESERVE the
    /// cumulative `set_order`, so it stops here). The full rationale for the key
    /// — including why the 4-tuple is a TOTAL order and the sort may therefore
    /// run unstable — lives on `dense_reindex_by_authoring`.
    fn sort_fields_by_authoring(fields: &mut [Field]) {
        // Borrow-based comparator: `sort_by_key` on a tuple containing
        // `key_name` would clone that String on every comparison
        // (O(n log n) allocations over a whole layer's fields, which no
        // per-delta limit bounds).
        fields.sort_unstable_by(|a, b| {
            (a.version, a.authoring_set_order, a.section_id)
                .cmp(&(b.version, b.authoring_set_order, b.section_id))
                .then_with(|| a.key_name.cmp(&b.key_name))
        });
    }

    /// Collect all sections in DFS order, with orphans appended.
    ///
    /// Walks the children map starting from root (parent_id=0) through
    /// [`reachable_sections_in_dfs_order`](Self::reachable_sections_in_dfs_order),
    /// whose order is parent-first except on a stale listing stored before
    /// b.xje (see that function). Sections not reachable from root are orphans
    /// and are emitted after all reachable ones (SDD §10.4). Feeds only the
    /// legacy [`plan_full_content`](Self::plan_full_content) and
    /// [`plan_incremental`](Self::plan_incremental); the faithful builders
    /// walk with [`faithful_walk`](Self::faithful_walk) instead and append no
    /// orphans.
    ///
    /// **The reachable DFS prefix is byte-for-byte identical to the prior
    /// single-function implementation. The orphan TAIL is not — it gained an
    /// order** (b.gcb/b.vz7). The prefix walks `state.children`'s authored `Vec`s
    /// and was always deterministic; the orphan append walked
    /// `state.sections.values()`, i.e. `HashMap` iteration order. Orphans are now
    /// sorted on `section_id` — the `state.sections` map key, hence a total order
    /// (same shape as `deleted_sections.sort_unstable()` in
    /// [`plan_faithful_coalesced`](Self::plan_faithful_coalesced)). That change
    /// only ADDS determinism: it replaces a sequence that had no defined order
    /// with one that does, and never reorders a sequence that was already
    /// deterministic.
    ///
    /// One transitive effect is in scope and intended:
    /// [`plan_incremental`](Self::plan_incremental) filters this function's output
    /// for its `new_sections`, so ITS orphaned-new sections become deterministic
    /// too. That is the same add-only improvement, not a wire change — and it is
    /// the ONLY determinism `plan_incremental` gains here; its own
    /// `modified_sections` / `changed_fields` / `changed_time_samples` remain
    /// unsorted by design (see [`plan_full_content`](Self::plan_full_content)'s
    /// scope fence).
    fn sections_in_dfs_order(state: &ObjectState) -> Result<Vec<Section>, DiffError> {
        let (mut ordered, reachable_ids) = Self::reachable_sections_in_dfs_order(state)?;

        // Orphans: any section not reached by the DFS walk. Collected first and
        // sorted on `section_id` (the `state.sections` map key, so a total
        // order) rather than pushed in `HashMap` iteration order — see the
        // doc-comment above.
        let mut orphans: Vec<Section> = state
            .sections
            .values()
            .filter(|s| !reachable_ids.contains(&s.section_id))
            .cloned()
            .collect();
        orphans.sort_unstable_by_key(|s| s.section_id);
        ordered.append(&mut orphans);

        Ok(ordered)
    }

    /// Walk the children lists from root, returning the reached sections in DFS
    /// order together with the set of their `section_id`s.
    ///
    /// The legacy walk, used only by
    /// [`sections_in_dfs_order`](Self::sections_in_dfs_order) (which appends
    /// orphans on top of this result) for the legacy
    /// [`plan_full_content`](Self::plan_full_content) /
    /// [`plan_incremental`](Self::plan_incremental). This helper NEVER appends
    /// orphans — the returned `Vec` and id set contain reached sections only.
    ///
    /// **Unchanged by design, stale listings included.** It descends through
    /// every listed id, whatever that id's `parent_id`, so on a stale listing
    /// stored before b.xje (a child also or only listed under a parent other
    /// than its own) it can emit the child ahead of its real parent. It stays
    /// that way because its output is the checkpoint base's bytes, which two
    /// leaders on different builds must write identically (SDD §22.20.2), and
    /// the byte-stable rollback wire. The faithful builders use
    /// [`faithful_walk`](Self::faithful_walk), which is parent-first on any
    /// stored shape.
    ///
    /// Iterative — a linear parent → child chain of ~10K–80K sections (trivially
    /// within the 100 MB layer cap) would previously have blown the thread stack
    /// under the recursive walker. The explicit `Vec` stack mirrors the pattern
    /// used by [`collect_subtree`](crate::domain::delta::collect_subtree) for
    /// cascade deletes and caps traversal depth at
    /// [`MAX_SECTION_TREE_DEPTH`] — any breach returns
    /// [`DiffError::SectionTreeTooDeep`] rather than unbounded growth.
    fn reachable_sections_in_dfs_order(
        state: &ObjectState,
    ) -> Result<(Vec<Section>, HashSet<u64>), DiffError> {
        let mut visited: HashSet<u64> = HashSet::new();
        let mut ordered: Vec<Section> = Vec::new();
        let mut reachable_ids: HashSet<u64> = HashSet::new();

        // Stack of `(section_id, depth)` — "emit this section, then expand
        // its children". Seed with the root's direct children (children of
        // sentinel parent_id = 0) at depth 1. Children are pushed in reverse
        // so the first child sits on top of the stack and is visited first,
        // matching the parent-first / left-to-right order of the prior
        // recursive walker.
        let mut stack: Vec<(u64, usize)> = Vec::new();
        if let Some(root_children) = state.children.get(&0) {
            for &child_id in root_children.iter().rev() {
                stack.push((child_id, 1));
            }
        }

        while let Some((section_id, depth)) = stack.pop() {
            // Cycle guard — a child that appears in multiple children lists
            // (malformed state) is emitted at most once.
            if !visited.insert(section_id) {
                continue;
            }

            if depth > MAX_SECTION_TREE_DEPTH {
                return Err(DiffError::SectionTreeTooDeep {
                    depth,
                    cap: MAX_SECTION_TREE_DEPTH,
                });
            }

            if let Some(section) = state.sections.get(&section_id) {
                ordered.push(section.clone());
                reachable_ids.insert(section_id);
            }

            if let Some(children) = state.children.get(&section_id) {
                for &child_id in children.iter().rev() {
                    if !visited.contains(&child_id) {
                        stack.push((child_id, depth + 1));
                    }
                }
            }
        }

        Ok((ordered, reachable_ids))
    }

    /// The faithful builders' walk: the sections reached from root through
    /// **true children only**, in DFS parent-first order, with the set of
    /// their `section_id`s and each one's position among its parent's true
    /// children.
    ///
    /// Used by [`plan_faithful_snapshot`](Self::plan_faithful_snapshot) and
    /// [`plan_faithful_coalesced`](Self::plan_faithful_coalesced) only. The
    /// legacy [`reachable_sections_in_dfs_order`](Self::reachable_sections_in_dfs_order)
    /// stays as it is for the checkpoint base and the rollback wire (see that
    /// function).
    ///
    /// Starting from list `0`, for each parent `P` the walk reaches, it reads
    /// `P`'s children list in order. An entry `c` is a **true child** of `P`
    /// iff `state.sections[c].parent_id == P` and `c` has not occurred earlier
    /// in `P`'s list. True children get consecutive positions `0, 1, 2, …` in
    /// list order and are walked. Every other entry is skipped without being
    /// marked visited and without descending through it:
    ///
    /// - a **ghost** (an id with no `state.sections` row) takes no position, so
    ///   it shifts no later sibling;
    /// - a **stale listing** (a section whose `parent_id` is another parent) is
    ///   reached, if at all, only from its own parent's list, so it is emitted
    ///   after that parent and with that list's position. A section stored
    ///   before b.xje that is listed only under a parent other than its own is
    ///   not reached, so the faithful builders treat it as not live;
    /// - a **repeat** of an id earlier in the same list keeps its first
    ///   position.
    ///
    /// Each section's `parent_id` names one parent and that parent's list is
    /// read once, so a section takes at most one position, and the result is a
    /// function of the stored content alone, never of `HashMap` iteration
    /// order. On a list with none of those entries, a section's position is
    /// its index in the list, the value the legacy C++ catch-up emits (b.bdx).
    ///
    /// Positions are built per reached parent in one O(list length) pass, so
    /// a wide fan-out costs O(total listed children), not
    /// Σ|children(parent)|². They are transient: built per call, read by the
    /// two faithful builders, and dropped with the call.
    ///
    /// Iterative, with the same [`MAX_SECTION_TREE_DEPTH`] cap and visited
    /// guard as the legacy walk; a breach returns
    /// [`DiffError::SectionTreeTooDeep`].
    fn faithful_walk(state: &ObjectState) -> Result<FaithfulWalk, DiffError> {
        let mut visited: HashSet<u64> = HashSet::new();
        let mut ordered: Vec<Section> = Vec::new();
        let mut live_ids: HashSet<u64> = HashSet::new();
        let mut positions: HashMap<u64, u32> = HashMap::new();

        // Stack of `(section_id, depth)`: "emit this section, then expand its
        // true children". Every pushed id is a true child, so it has a row.
        let mut stack: Vec<(u64, usize)> = Vec::new();
        // The list to expand before the next pop, as `(parent_id, depth of
        // its children)`: the root's list at depth 1 first, then each
        // emitted section's own list.
        let mut expand: Option<(u64, usize)> = Some((0, 1));
        // Reused per expanded list: the list's true children in order, and
        // the ids already seen in it (to skip repeats).
        let mut true_children: Vec<u64> = Vec::new();
        let mut seen_in_list: HashSet<u64> = HashSet::new();

        loop {
            let to_expand = expand.take().and_then(|(parent_id, child_depth)| {
                state
                    .children
                    .get(&parent_id)
                    .map(|children| (parent_id, child_depth, children))
            });
            if let Some((parent_id, child_depth, children)) = to_expand {
                true_children.clear();
                seen_in_list.clear();
                for &child_id in children {
                    // Only an id's first occurrence in the list can be a
                    // true child.
                    if !seen_in_list.insert(child_id) {
                        continue;
                    }
                    let is_true_child = state
                        .sections
                        .get(&child_id)
                        .is_some_and(|s| s.parent_id == parent_id);
                    if is_true_child {
                        true_children.push(child_id);
                    }
                }
                for (position, &child_id) in true_children.iter().enumerate() {
                    // No list in a layer reaches `u32::MAX` entries; saturate
                    // rather than wrap.
                    positions.insert(child_id, u32::try_from(position).unwrap_or(u32::MAX));
                }
                // Reversed, so the first child is popped first (parent-first,
                // left-to-right).
                for &child_id in true_children.iter().rev() {
                    if !visited.contains(&child_id) {
                        stack.push((child_id, child_depth));
                    }
                }
            }

            let Some((section_id, depth)) = stack.pop() else {
                break;
            };

            // Visited guard. A section is a true child of its one `parent_id`
            // only, and each list is read once, so no input should reach a
            // section twice; the guard keeps any that does from emitting it
            // twice.
            if !visited.insert(section_id) {
                continue;
            }

            if depth > MAX_SECTION_TREE_DEPTH {
                return Err(DiffError::SectionTreeTooDeep {
                    depth,
                    cap: MAX_SECTION_TREE_DEPTH,
                });
            }

            let Some(section) = state.sections.get(&section_id) else {
                continue;
            };
            ordered.push(section.clone());
            live_ids.insert(section_id);

            // A row whose id is `0` shares the root's list key (see
            // `delta::adopt_orphans`); that list was expanded first, so it is
            // not read again.
            if section_id != 0 {
                expand = Some((section_id, depth + 1));
            }
        }

        Ok((ordered, live_ids, positions))
    }

    /// The `section_order` a faithful builder emits for `section`: its
    /// position from the [`faithful_walk`](Self::faithful_walk) that emitted
    /// it.
    ///
    /// Every section the walk emits has a position, so the fallback to the
    /// stored `section_order` is unreachable; it is kept so a broken
    /// invariant degrades rather than panics in release builds.
    fn walked_position(positions: &HashMap<u64, u32>, section: &Section) -> u32 {
        let position = positions.get(&section.section_id).copied();
        debug_assert!(
            position.is_some(),
            "faithful_walk emitted section {} without a position",
            section.section_id
        );
        position.unwrap_or(section.section_order)
    }
}

// ── External value identification ──────────────────────────────────────────

/// Result of scanning a DiffPlan for external value references.
///
/// Contains the deduplicated hashes in first-appearance order and a mapping
/// from hash to 1-based OMPF part index. Inline-only diffs produce an empty
/// result.
pub struct ExternalValueScan {
    /// Deduplicated hashes in the order they were first encountered.
    /// Used to build the OMPF parts slice after fetching data.
    pub ordered_hashes: Vec<[u8; 32]>,
    /// Hash → 1-based OMPF part index. Two fields with the same hash share
    /// the same index (and a single OMPF part).
    pub hash_to_index: HashMap<[u8; 32], u32>,
}

impl ExternalValueScan {
    /// Returns true if no external values were found.
    pub fn is_empty(&self) -> bool {
        self.ordered_hashes.is_empty()
    }
}

/// Insert `hash` into the scan if not already present, assigning the next
/// 1-based index. Saturates at `u32::MAX` rather than wrapping — OMPF part
/// indices are u32 in the wire format, so we cannot represent more than
/// `u32::MAX - 1` distinct external values in one diff. A diff is built from
/// a whole layer, which no per-delta limit bounds; a real layer's distinct
/// external values are orders of magnitude below that (four billion distinct
/// values would be far beyond any layer's memory), and the saturation is a
/// belt-and-braces guard.
fn add_external_hash(
    hash: &[u8; 32],
    ordered_hashes: &mut Vec<[u8; 32]>,
    hash_to_index: &mut HashMap<[u8; 32], u32>,
) {
    if hash_to_index.contains_key(hash) {
        return;
    }
    // `ordered_hashes.len() + 1` fits in u32 iff the current length is
    // <= u32::MAX - 1. A saturating `try_from` is sufficient for the panic-
    // free path; the explicit `saturating_add(1)` prevents a second-step
    // overflow if the try_from ever yields exactly `u32::MAX`.
    let index = u32::try_from(ordered_hashes.len())
        .unwrap_or(u32::MAX)
        .saturating_add(1);
    ordered_hashes.push(*hash);
    hash_to_index.insert(*hash, index);
}

/// Scan fields and time samples for `FieldValue::External` entries.
///
/// Returns deduplicated hashes with stable 1-based indices (order of first
/// appearance). Pure function — no I/O, no async.
pub fn collect_external_value_hashes(
    fields: &[Field],
    time_samples: &[TimeSample],
) -> ExternalValueScan {
    let mut ordered_hashes: Vec<[u8; 32]> = Vec::new();
    let mut hash_to_index: HashMap<[u8; 32], u32> = HashMap::new();

    for field in fields {
        if let FieldValue::External { hash, .. } = &field.value {
            add_external_hash(hash, &mut ordered_hashes, &mut hash_to_index);
        }
    }

    for ts in time_samples {
        if let FieldValue::External { hash, .. } = &ts.value {
            add_external_hash(hash, &mut ordered_hashes, &mut hash_to_index);
        }
    }

    ExternalValueScan {
        ordered_hashes,
        hash_to_index,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ordered_float::OrderedFloat;

    use crate::domain::types::{FieldValue, ObjectState};

    // ── helpers ──────────────────────────────────────────────────────────────

    fn make_section(id: u64, parent: u64, first: u64, last: u64) -> Section {
        Section {
            section_id: id,
            parent_id: parent,
            name: format!("sec_{id}"),
            section_type: 0,
            section_order: id as u32,
            first_version: first,
            last_version: last,
        }
    }

    fn make_field(section_id: u64, key: &str, version: u64) -> Field {
        Field {
            section_id,
            key_name: key.to_string(),
            value: FieldValue::Inline(vec![version as u8]),
            set_order: 0,
            authoring_set_order: 0,
            version,
        }
    }

    fn make_time_sample(section_id: u64, time: f64, version: u64) -> TimeSample {
        TimeSample {
            section_id,
            time: OrderedFloat(time),
            value: FieldValue::Inline(vec![version as u8]),
            version,
        }
    }

    /// Insert a section into state and register it in the children map.
    fn insert_section(state: &mut ObjectState, sec: Section) {
        let parent = sec.parent_id;
        let id = sec.section_id;
        state.sections.insert(id, sec);
        state.children.entry(parent).or_default().push(id);
    }

    fn insert_field(state: &mut ObjectState, f: Field) {
        state.fields.insert((f.section_id, f.key_name.clone()), f);
    }

    fn insert_time_sample(state: &mut ObjectState, ts: TimeSample) {
        state.time_samples.insert((ts.section_id, ts.time), ts);
    }

    fn section_ids(sections: &[Section]) -> Vec<u64> {
        sections.iter().map(|s| s.section_id).collect()
    }

    // ── Case 3: Up to date ──────────────────────────────────────────────────

    #[test]
    fn plan_have_version_equals_sequence_num_returns_up_to_date() {
        let state = ObjectState {
            object_id: 1,
            sequence_num: 5,
            ..Default::default()
        };
        let result = DiffEngine::plan(&state, &[], 1, 5, None).unwrap();
        assert_eq!(result, DiffPlan::UpToDate);
    }

    #[test]
    fn plan_have_version_greater_than_sequence_num_returns_up_to_date() {
        let state = ObjectState {
            object_id: 1,
            sequence_num: 5,
            ..Default::default()
        };
        let result = DiffEngine::plan(&state, &[], 1, 10, None).unwrap();
        assert_eq!(result, DiffPlan::UpToDate);
    }

    // ── Case 1: Full content ────────────────────────────────────────────────

    #[test]
    fn plan_object_id_mismatch_returns_full_content() {
        let state = ObjectState {
            object_id: 42,
            sequence_num: 5,
            ..Default::default()
        };
        let result = DiffEngine::plan(&state, &[], 99, 3, None).unwrap();
        assert!(
            matches!(result, DiffPlan::FullContent { .. }),
            "object_id mismatch must produce FullContent, got {result:?}"
        );
    }

    #[test]
    fn plan_have_version_zero_returns_full_content() {
        let state = ObjectState {
            object_id: 1,
            sequence_num: 5,
            ..Default::default()
        };
        // Same object_id, but have_version=0 → still FullContent.
        let result = DiffEngine::plan(&state, &[], 1, 0, None).unwrap();
        assert!(
            matches!(result, DiffPlan::FullContent { .. }),
            "have_version=0 must produce FullContent, got {result:?}"
        );
    }

    #[test]
    fn plan_full_content_empty_object_returns_empty_collections() {
        let state = ObjectState {
            object_id: 1,
            sequence_num: 1,
            ..Default::default()
        };
        let result = DiffEngine::plan(&state, &[], 1, 0, None).unwrap();
        match result {
            DiffPlan::FullContent {
                sections,
                fields,
                time_samples,
            } => {
                assert!(sections.is_empty(), "empty object should have no sections");
                assert!(fields.is_empty(), "empty object should have no fields");
                assert!(
                    time_samples.is_empty(),
                    "empty object should have no time_samples"
                );
            }
            other => panic!("expected FullContent, got {other:?}"),
        }
    }

    #[test]
    fn plan_full_content_three_level_tree_dfs_parent_before_children() {
        // Tree:
        //   root (0)
        //     ├── 1
        //     │   ├── 3
        //     │   └── 4
        //     └── 2
        //         └── 5
        let mut state = ObjectState {
            object_id: 1,
            sequence_num: 5,
            ..Default::default()
        };
        insert_section(&mut state, make_section(1, 0, 1, 5));
        insert_section(&mut state, make_section(2, 0, 1, 5));
        insert_section(&mut state, make_section(3, 1, 1, 5));
        insert_section(&mut state, make_section(4, 1, 1, 5));
        insert_section(&mut state, make_section(5, 2, 1, 5));

        let result = DiffEngine::plan(&state, &[], 1, 0, None).unwrap();
        match result {
            DiffPlan::FullContent { sections, .. } => {
                let ids = section_ids(&sections);
                assert_eq!(ids.len(), 5, "should include all 5 sections");

                // Parent-before-children constraints:
                let pos = |id: u64| ids.iter().position(|&x| x == id).unwrap();
                assert!(pos(1) < pos(3), "section 1 must precede child 3");
                assert!(pos(1) < pos(4), "section 1 must precede child 4");
                assert!(pos(2) < pos(5), "section 2 must precede child 5");
            }
            other => panic!("expected FullContent, got {other:?}"),
        }
    }

    #[test]
    fn plan_full_content_orphan_emitted_after_reachable_sections() {
        // Orphan: section 99 has parent_id=77, but 77 doesn't exist.
        let mut state = ObjectState {
            object_id: 1,
            sequence_num: 5,
            ..Default::default()
        };
        insert_section(&mut state, make_section(1, 0, 1, 5));
        // Insert orphan — parent_id 77 is not in the tree.
        let orphan = make_section(99, 77, 1, 5);
        state.sections.insert(99, orphan);
        // Don't register 99 in children under 77 because 77 itself is not in sections;
        // the DFS from root won't reach it either way.

        let result = DiffEngine::plan(&state, &[], 1, 0, None).unwrap();
        match result {
            DiffPlan::FullContent { sections, .. } => {
                let ids = section_ids(&sections);
                assert_eq!(ids.len(), 2, "should have reachable section + orphan");
                let pos_reachable = ids.iter().position(|&x| x == 1).unwrap();
                let pos_orphan = ids.iter().position(|&x| x == 99).unwrap();
                assert!(
                    pos_reachable < pos_orphan,
                    "orphan must appear after all reachable sections"
                );
            }
            other => panic!("expected FullContent, got {other:?}"),
        }
    }

    #[test]
    fn plan_full_content_includes_all_fields_and_time_samples() {
        let mut state = ObjectState {
            object_id: 1,
            sequence_num: 5,
            ..Default::default()
        };
        insert_section(&mut state, make_section(1, 0, 1, 5));
        insert_field(&mut state, make_field(1, "color", 2));
        insert_field(&mut state, make_field(1, "size", 3));
        insert_time_sample(&mut state, make_time_sample(1, 0.0, 1));
        insert_time_sample(&mut state, make_time_sample(1, 1.0, 4));

        let result = DiffEngine::plan(&state, &[], 1, 0, None).unwrap();
        match result {
            DiffPlan::FullContent {
                fields,
                time_samples,
                ..
            } => {
                assert_eq!(fields.len(), 2, "should include both fields");
                assert_eq!(time_samples.len(), 2, "should include both time samples");
            }
            other => panic!("expected FullContent, got {other:?}"),
        }
    }

    // ── Case 2: Incremental ─────────────────────────────────────────────────

    #[test]
    fn plan_incremental_new_sections_included() {
        let mut state = ObjectState {
            object_id: 1,
            sequence_num: 10,
            ..Default::default()
        };
        // Section created at version 8 — after subscriber's version 5.
        insert_section(&mut state, make_section(1, 0, 8, 10));

        let result = DiffEngine::plan(&state, &[], 1, 5, None).unwrap();
        match result {
            DiffPlan::Incremental {
                new_sections,
                modified_sections,
                ..
            } => {
                assert_eq!(new_sections.len(), 1, "one new section");
                assert_eq!(new_sections[0].section_id, 1);
                assert!(modified_sections.is_empty(), "no modified sections");
            }
            other => panic!("expected Incremental, got {other:?}"),
        }
    }

    #[test]
    fn plan_incremental_modified_sections_included() {
        let mut state = ObjectState {
            object_id: 1,
            sequence_num: 10,
            ..Default::default()
        };
        // Section existed before version 5 (first_version=2), modified after (last_version=7).
        insert_section(&mut state, make_section(1, 0, 2, 7));

        let result = DiffEngine::plan(&state, &[], 1, 5, None).unwrap();
        match result {
            DiffPlan::Incremental {
                new_sections,
                modified_sections,
                ..
            } => {
                assert!(
                    new_sections.is_empty(),
                    "not new — existed before have_version"
                );
                assert_eq!(modified_sections.len(), 1, "one modified section");
                assert_eq!(modified_sections[0].section_id, 1);
            }
            other => panic!("expected Incremental, got {other:?}"),
        }
    }

    #[test]
    fn plan_incremental_unchanged_sections_excluded() {
        let mut state = ObjectState {
            object_id: 1,
            sequence_num: 10,
            ..Default::default()
        };
        // Section untouched since before subscriber's version.
        insert_section(&mut state, make_section(1, 0, 1, 4));

        let result = DiffEngine::plan(&state, &[], 1, 5, None).unwrap();
        match result {
            DiffPlan::Incremental {
                new_sections,
                modified_sections,
                ..
            } => {
                assert!(new_sections.is_empty(), "section not new");
                assert!(
                    modified_sections.is_empty(),
                    "section not modified after have_version"
                );
            }
            other => panic!("expected Incremental, got {other:?}"),
        }
    }

    #[test]
    fn plan_incremental_deleted_sections_from_tombstones() {
        let state = ObjectState {
            object_id: 1,
            sequence_num: 10,
            ..Default::default()
        };
        // Tombstone: section 50 existed before have_version (first=2), deleted after (last=7).
        let tombstones = vec![(50, 2, 7)];

        let result = DiffEngine::plan(&state, &tombstones, 1, 5, None).unwrap();
        match result {
            DiffPlan::Incremental {
                deleted_sections, ..
            } => {
                assert_eq!(deleted_sections.len(), 1);
                assert_eq!(deleted_sections[0], (50, 2, 7));
            }
            other => panic!("expected Incremental, got {other:?}"),
        }
    }

    #[test]
    fn plan_incremental_tombstone_before_have_version_excluded() {
        let state = ObjectState {
            object_id: 1,
            sequence_num: 10,
            ..Default::default()
        };
        // Tombstone: deleted before subscriber's version — they already know.
        let tombstones = vec![(50, 1, 3)];

        let result = DiffEngine::plan(&state, &tombstones, 1, 5, None).unwrap();
        match result {
            DiffPlan::Incremental {
                deleted_sections, ..
            } => {
                assert!(
                    deleted_sections.is_empty(),
                    "tombstone with last_version <= have_version should be excluded"
                );
            }
            other => panic!("expected Incremental, got {other:?}"),
        }
    }

    #[test]
    fn plan_incremental_tombstone_created_after_have_version_excluded() {
        let state = ObjectState {
            object_id: 1,
            sequence_num: 10,
            ..Default::default()
        };
        // Tombstone: section created AND deleted after subscriber's version.
        // Subscriber never knew about it → no need to tell them it was deleted.
        let tombstones = vec![(50, 7, 9)];

        let result = DiffEngine::plan(&state, &tombstones, 1, 5, None).unwrap();
        match result {
            DiffPlan::Incremental {
                deleted_sections, ..
            } => {
                assert!(
                    deleted_sections.is_empty(),
                    "tombstone with first_version > have_version should be excluded"
                );
            }
            other => panic!("expected Incremental, got {other:?}"),
        }
    }

    #[test]
    fn plan_incremental_changed_fields_included_unchanged_excluded() {
        let mut state = ObjectState {
            object_id: 1,
            sequence_num: 10,
            ..Default::default()
        };
        insert_section(&mut state, make_section(1, 0, 1, 1));
        // Field changed after subscriber's version.
        insert_field(&mut state, make_field(1, "color", 7));
        // Field unchanged — version at or before subscriber's version.
        insert_field(&mut state, make_field(1, "size", 3));

        let result = DiffEngine::plan(&state, &[], 1, 5, None).unwrap();
        match result {
            DiffPlan::Incremental { changed_fields, .. } => {
                assert_eq!(changed_fields.len(), 1, "only changed field included");
                assert_eq!(changed_fields[0].key_name, "color");
            }
            other => panic!("expected Incremental, got {other:?}"),
        }
    }

    #[test]
    fn plan_incremental_changed_time_samples_included_unchanged_excluded() {
        let mut state = ObjectState {
            object_id: 1,
            sequence_num: 10,
            ..Default::default()
        };
        insert_section(&mut state, make_section(1, 0, 1, 1));
        // Time sample changed after subscriber's version.
        insert_time_sample(&mut state, make_time_sample(1, 0.0, 8));
        // Time sample unchanged.
        insert_time_sample(&mut state, make_time_sample(1, 1.0, 2));

        let result = DiffEngine::plan(&state, &[], 1, 5, None).unwrap();
        match result {
            DiffPlan::Incremental {
                changed_time_samples,
                ..
            } => {
                assert_eq!(
                    changed_time_samples.len(),
                    1,
                    "only changed sample included"
                );
                assert_eq!(changed_time_samples[0].time, OrderedFloat(0.0));
            }
            other => panic!("expected Incremental, got {other:?}"),
        }
    }

    #[test]
    fn plan_incremental_section_created_and_deleted_in_same_range() {
        // Section created at version 7, deleted at version 9. Subscriber is at version 5.
        // The section is gone from ObjectState but the tombstone covers it.
        // Because first_version > have_version, the subscriber never knew this section,
        // so the tombstone should NOT be sent.
        let state = ObjectState {
            object_id: 1,
            sequence_num: 10,
            ..Default::default()
        };
        let tombstones = vec![(50, 7, 9)];

        let result = DiffEngine::plan(&state, &tombstones, 1, 5, None).unwrap();
        match result {
            DiffPlan::Incremental {
                deleted_sections, ..
            } => {
                assert!(
                    deleted_sections.is_empty(),
                    "section created+deleted after have_version should not produce a tombstone"
                );
            }
            other => panic!("expected Incremental, got {other:?}"),
        }
    }

    // ── Edge cases ──────────────────────────────────────────────────────────

    #[test]
    fn plan_full_content_single_section_no_children() {
        let mut state = ObjectState {
            object_id: 1,
            sequence_num: 3,
            ..Default::default()
        };
        insert_section(&mut state, make_section(1, 0, 1, 3));

        let result = DiffEngine::plan(&state, &[], 1, 0, None).unwrap();
        match result {
            DiffPlan::FullContent { sections, .. } => {
                assert_eq!(sections.len(), 1);
                assert_eq!(sections[0].section_id, 1);
            }
            other => panic!("expected FullContent, got {other:?}"),
        }
    }

    #[test]
    fn plan_full_content_multiple_orphans() {
        let mut state = ObjectState {
            object_id: 1,
            sequence_num: 5,
            ..Default::default()
        };
        // One reachable section.
        insert_section(&mut state, make_section(1, 0, 1, 5));
        // Two orphans with non-existent parents.
        let orphan_a = make_section(90, 77, 1, 5);
        let orphan_b = make_section(91, 88, 1, 5);
        state.sections.insert(90, orphan_a);
        state.sections.insert(91, orphan_b);

        let result = DiffEngine::plan(&state, &[], 1, 0, None).unwrap();
        match result {
            DiffPlan::FullContent { sections, .. } => {
                assert_eq!(sections.len(), 3, "1 reachable + 2 orphans");
                let ids = section_ids(&sections);
                let pos_reachable = ids.iter().position(|&x| x == 1).unwrap();
                let pos_orphan_a = ids.iter().position(|&x| x == 90).unwrap();
                let pos_orphan_b = ids.iter().position(|&x| x == 91).unwrap();
                assert!(
                    pos_reachable < pos_orphan_a && pos_reachable < pos_orphan_b,
                    "all orphans must appear after reachable sections"
                );
            }
            other => panic!("expected FullContent, got {other:?}"),
        }
    }

    #[test]
    fn plan_incremental_new_sections_in_dfs_order() {
        // Verify that new sections emitted in incremental diff maintain DFS order.
        //   root(0)
        //     └── 1 (old, v1)
        //         ├── 2 (new, v7)
        //         │   └── 4 (new, v7)
        //         └── 3 (new, v7)
        let mut state = ObjectState {
            object_id: 1,
            sequence_num: 10,
            ..Default::default()
        };
        insert_section(&mut state, make_section(1, 0, 1, 1));
        insert_section(&mut state, make_section(2, 1, 7, 7));
        insert_section(&mut state, make_section(3, 1, 7, 7));
        insert_section(&mut state, make_section(4, 2, 7, 7));

        let result = DiffEngine::plan(&state, &[], 1, 5, None).unwrap();
        match result {
            DiffPlan::Incremental {
                new_sections,
                modified_sections,
                ..
            } => {
                assert!(modified_sections.is_empty());
                let ids = section_ids(&new_sections);
                assert_eq!(ids.len(), 3, "sections 2, 3, 4 are new");
                // DFS order: 2 before 4 (parent before child), 4 before 3 (depth first).
                let pos = |id: u64| ids.iter().position(|&x| x == id).unwrap();
                assert!(
                    pos(2) < pos(4),
                    "parent 2 must precede child 4 in DFS order"
                );
            }
            other => panic!("expected Incremental, got {other:?}"),
        }
    }

    #[test]
    fn plan_incremental_boundary_last_version_equals_have_version_excluded() {
        // last_version == have_version means the section has NOT changed since the
        // subscriber's version. It should be excluded from modified sections.
        let mut state = ObjectState {
            object_id: 1,
            sequence_num: 10,
            ..Default::default()
        };
        insert_section(&mut state, make_section(1, 0, 2, 5));

        let result = DiffEngine::plan(&state, &[], 1, 5, None).unwrap();
        match result {
            DiffPlan::Incremental {
                new_sections,
                modified_sections,
                ..
            } => {
                assert!(new_sections.is_empty());
                assert!(
                    modified_sections.is_empty(),
                    "section with last_version == have_version should not be modified"
                );
            }
            other => panic!("expected Incremental, got {other:?}"),
        }
    }

    #[test]
    fn plan_incremental_field_at_have_version_excluded() {
        // Field with version == have_version should be excluded (not strictly >).
        let mut state = ObjectState {
            object_id: 1,
            sequence_num: 10,
            ..Default::default()
        };
        insert_section(&mut state, make_section(1, 0, 1, 1));
        insert_field(&mut state, make_field(1, "exact", 5));

        let result = DiffEngine::plan(&state, &[], 1, 5, None).unwrap();
        match result {
            DiffPlan::Incremental { changed_fields, .. } => {
                assert!(
                    changed_fields.is_empty(),
                    "field with version == have_version should be excluded"
                );
            }
            other => panic!("expected Incremental, got {other:?}"),
        }
    }

    #[test]
    fn plan_incremental_time_sample_at_have_version_excluded() {
        // Time sample with version == have_version should be excluded.
        let mut state = ObjectState {
            object_id: 1,
            sequence_num: 10,
            ..Default::default()
        };
        insert_section(&mut state, make_section(1, 0, 1, 1));
        insert_time_sample(&mut state, make_time_sample(1, 0.0, 5));

        let result = DiffEngine::plan(&state, &[], 1, 5, None).unwrap();
        match result {
            DiffPlan::Incremental {
                changed_time_samples,
                ..
            } => {
                assert!(
                    changed_time_samples.is_empty(),
                    "time sample with version == have_version should be excluded"
                );
            }
            other => panic!("expected Incremental, got {other:?}"),
        }
    }

    // ── upper_version bound tests ───────────────────────────────────────

    #[test]
    fn plan_incremental_upper_version_bounds_changed_fields() {
        // have=1, upper=5, sequence_num=10. Fields at versions 2, 5, 7 should
        // yield only versions 2 and 5 (in (have, upper]).
        let mut state = ObjectState {
            object_id: 1,
            sequence_num: 10,
            ..Default::default()
        };
        insert_section(&mut state, make_section(1, 0, 1, 10));
        insert_field(&mut state, make_field(1, "a", 2));
        insert_field(&mut state, make_field(1, "b", 5));
        insert_field(&mut state, make_field(1, "c", 7));

        let result = DiffEngine::plan(&state, &[], 1, 1, Some(5)).unwrap();
        match result {
            DiffPlan::Incremental { changed_fields, .. } => {
                let versions: Vec<u64> = changed_fields.iter().map(|f| f.version).collect();
                assert!(
                    versions.contains(&2) && versions.contains(&5),
                    "fields at versions 2 and 5 must be included, got {versions:?}"
                );
                assert!(
                    !versions.contains(&7),
                    "field at version 7 must be excluded by upper bound 5, got {versions:?}"
                );
            }
            other => panic!("expected Incremental, got {other:?}"),
        }
    }

    #[test]
    fn plan_incremental_upper_version_bounds_time_samples() {
        let mut state = ObjectState {
            object_id: 1,
            sequence_num: 10,
            ..Default::default()
        };
        insert_section(&mut state, make_section(1, 0, 1, 10));
        insert_time_sample(&mut state, make_time_sample(1, 0.0, 3));
        insert_time_sample(&mut state, make_time_sample(1, 1.0, 6));
        insert_time_sample(&mut state, make_time_sample(1, 2.0, 8));

        let result = DiffEngine::plan(&state, &[], 1, 2, Some(6)).unwrap();
        match result {
            DiffPlan::Incremental {
                changed_time_samples,
                ..
            } => {
                let versions: Vec<u64> = changed_time_samples.iter().map(|ts| ts.version).collect();
                assert!(
                    versions.contains(&3) && versions.contains(&6),
                    "samples at versions 3 and 6 must be included, got {versions:?}"
                );
                assert!(
                    !versions.contains(&8),
                    "sample at version 8 must be excluded by upper bound, got {versions:?}"
                );
            }
            other => panic!("expected Incremental, got {other:?}"),
        }
    }

    #[test]
    fn plan_incremental_upper_version_bounds_new_sections() {
        // Section 1 created at v2, section 2 created at v8. have=1, upper=5 →
        // only section 1 emitted.
        let mut state = ObjectState {
            object_id: 1,
            sequence_num: 10,
            ..Default::default()
        };
        insert_section(&mut state, make_section(1, 0, 2, 2));
        insert_section(&mut state, make_section(2, 0, 8, 8));

        let result = DiffEngine::plan(&state, &[], 1, 1, Some(5)).unwrap();
        match result {
            DiffPlan::Incremental { new_sections, .. } => {
                let ids: Vec<u64> = new_sections.iter().map(|s| s.section_id).collect();
                assert!(
                    ids.contains(&1),
                    "section created at v2 must be emitted, got {ids:?}"
                );
                assert!(
                    !ids.contains(&2),
                    "section created at v8 must be excluded by upper 5, got {ids:?}"
                );
            }
            other => panic!("expected Incremental, got {other:?}"),
        }
    }

    #[test]
    fn plan_incremental_upper_version_bounds_deleted_sections() {
        let state = ObjectState {
            object_id: 1,
            sequence_num: 10,
            ..Default::default()
        };
        // tombstones: section 1 existed at v1, deleted at v4; section 2 existed at v1, deleted at v9.
        let tombstones = vec![(1u64, 1u64, 4u64), (2u64, 1u64, 9u64)];

        let result = DiffEngine::plan(&state, &tombstones, 1, 2, Some(5)).unwrap();
        match result {
            DiffPlan::Incremental {
                deleted_sections, ..
            } => {
                let ids: Vec<u64> = deleted_sections.iter().map(|(sid, ..)| *sid).collect();
                assert!(
                    ids.contains(&1),
                    "tombstone deleted at v4 must be in (have=2, upper=5], got {ids:?}"
                );
                assert!(
                    !ids.contains(&2),
                    "tombstone deleted at v9 must be excluded by upper=5, got {ids:?}"
                );
            }
            other => panic!("expected Incremental, got {other:?}"),
        }
    }

    #[test]
    fn plan_incremental_upper_version_none_matches_original_behavior() {
        // With upper_version=None, behavior must be identical to the old plan.
        let mut state = ObjectState {
            object_id: 1,
            sequence_num: 10,
            ..Default::default()
        };
        insert_section(&mut state, make_section(1, 0, 1, 10));
        insert_field(&mut state, make_field(1, "a", 2));
        insert_field(&mut state, make_field(1, "b", 9));

        let result = DiffEngine::plan(&state, &[], 1, 1, None).unwrap();
        match result {
            DiffPlan::Incremental { changed_fields, .. } => {
                let versions: Vec<u64> = changed_fields.iter().map(|f| f.version).collect();
                assert!(
                    versions.contains(&2) && versions.contains(&9),
                    "None upper bound must include everything > have_version, got {versions:?}"
                );
            }
            other => panic!("expected Incremental, got {other:?}"),
        }
    }

    // ── External value identification tests ─────────────────────────────

    fn test_hash(byte: u8) -> [u8; 32] {
        [byte; 32]
    }

    fn make_ext_field(section_id: u64, key: &str, version: u64, hash_byte: u8) -> Field {
        Field {
            section_id,
            key_name: key.to_string(),
            value: FieldValue::External {
                hash: test_hash(hash_byte),
                size: 1024,
                ext_value_index: 0,
            },
            set_order: 0,
            authoring_set_order: 0,
            version,
        }
    }

    fn make_ext_time_sample(section_id: u64, time: f64, version: u64, hash_byte: u8) -> TimeSample {
        TimeSample {
            section_id,
            time: OrderedFloat(time),
            value: FieldValue::External {
                hash: test_hash(hash_byte),
                size: 2048,
                ext_value_index: 0,
            },
            version,
        }
    }

    #[test]
    fn collect_ext_hashes_all_inline_returns_empty() {
        let fields = vec![
            make_field(1, "a", 1),
            make_field(1, "b", 2),
            make_field(2, "c", 3),
            make_field(2, "d", 4),
            make_field(3, "e", 5),
        ];
        let scan = collect_external_value_hashes(&fields, &[]);
        assert!(scan.is_empty());
        assert!(scan.ordered_hashes.is_empty());
        assert!(scan.hash_to_index.is_empty());
    }

    #[test]
    fn collect_ext_hashes_all_external_returns_all() {
        let fields = vec![
            make_ext_field(1, "a", 1, 0x01),
            make_ext_field(1, "b", 2, 0x02),
            make_ext_field(2, "c", 3, 0x03),
            make_ext_field(2, "d", 4, 0x04),
            make_ext_field(3, "e", 5, 0x05),
        ];
        let scan = collect_external_value_hashes(&fields, &[]);
        assert_eq!(scan.ordered_hashes.len(), 5);
        assert_eq!(scan.hash_to_index.len(), 5);
        // All indices should be 1-based and distinct.
        let mut indices: Vec<u32> = scan.hash_to_index.values().copied().collect();
        indices.sort();
        assert_eq!(indices, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn collect_ext_hashes_dedup_same_hash_one_entry() {
        let fields = vec![
            make_ext_field(1, "a", 1, 0xAA),
            make_ext_field(2, "b", 2, 0xAA),
            make_ext_field(3, "c", 3, 0xAA),
        ];
        let scan = collect_external_value_hashes(&fields, &[]);
        assert_eq!(scan.ordered_hashes.len(), 1, "dedup: one unique hash");
        assert_eq!(scan.hash_to_index.len(), 1);
        assert_eq!(scan.hash_to_index[&test_hash(0xAA)], 1);
    }

    #[test]
    fn collect_ext_hashes_mixed_inline_external() {
        let fields = vec![
            make_field(1, "inline1", 1),
            make_ext_field(1, "ext1", 2, 0x01),
            make_field(2, "inline2", 3),
            make_ext_field(2, "ext2", 4, 0x02),
        ];
        let scan = collect_external_value_hashes(&fields, &[]);
        assert_eq!(scan.ordered_hashes.len(), 2, "only external hashes");
        assert!(scan.hash_to_index.contains_key(&test_hash(0x01)));
        assert!(scan.hash_to_index.contains_key(&test_hash(0x02)));
    }

    #[test]
    fn collect_ext_hashes_empty_input() {
        let scan = collect_external_value_hashes(&[], &[]);
        assert!(scan.is_empty());
    }

    #[test]
    fn collect_ext_hashes_time_samples_shared_pool_with_fields() {
        let fields = vec![make_ext_field(1, "f1", 1, 0x01)];
        let time_samples = vec![
            make_ext_time_sample(1, 0.0, 2, 0x02),
            make_ext_time_sample(1, 1.0, 3, 0x01), // same hash as field
        ];
        let scan = collect_external_value_hashes(&fields, &time_samples);
        assert_eq!(
            scan.ordered_hashes.len(),
            2,
            "hash 0x01 deduped across field and time sample"
        );
        // Both the field and time sample with hash 0x01 share index 1.
        assert_eq!(scan.hash_to_index[&test_hash(0x01)], 1);
        assert_eq!(scan.hash_to_index[&test_hash(0x02)], 2);
    }

    #[test]
    fn collect_ext_hashes_deterministic_ordering() {
        // Run twice with same input; ordering must be identical.
        let fields = vec![
            make_ext_field(1, "a", 1, 0x03),
            make_ext_field(2, "b", 2, 0x01),
            make_ext_field(3, "c", 3, 0x02),
        ];
        let scan1 = collect_external_value_hashes(&fields, &[]);
        let scan2 = collect_external_value_hashes(&fields, &[]);
        assert_eq!(scan1.ordered_hashes, scan2.ordered_hashes);
        assert_eq!(scan1.hash_to_index, scan2.hash_to_index);
    }

    // ── OMPF packing tests (via build_diff_payload) ─────────────────────

    #[test]
    fn build_diff_payload_zero_ext_values_one_ompf_part() {
        use crate::domain::ompf::OmpfContainer;
        use crate::storage::diff_read::build_diff_payload;

        let plan = DiffPlan::FullContent {
            sections: vec![make_section(1, 0, 1, 1)],
            fields: vec![make_field(1, "inline", 1)],
            time_samples: vec![],
        };
        let scan = ExternalValueScan {
            ordered_hashes: vec![],
            hash_to_index: HashMap::new(),
        };
        let ompf_bytes = build_diff_payload(&plan, "usd", 0, &scan, &HashMap::new())
            .expect("build_diff_payload should succeed in test");
        let container = OmpfContainer::parse(&ompf_bytes).unwrap();
        assert!(
            container.external_values.is_empty(),
            "inline-only diff must have no external OMPF parts"
        );
    }

    #[test]
    fn build_diff_payload_single_ext_value_two_ompf_parts() {
        use crate::domain::ompf::OmpfContainer;
        use crate::storage::diff_read::build_diff_payload;

        let h = test_hash(0xAA);
        let fields = vec![make_ext_field(1, "tex", 1, 0xAA)];
        let scan = collect_external_value_hashes(&fields, &[]);

        let mut ext_data = HashMap::new();
        ext_data.insert(h, b"external-blob-data".to_vec());

        let plan = DiffPlan::FullContent {
            sections: vec![make_section(1, 0, 1, 1)],
            fields,
            time_samples: vec![],
        };
        let ompf_bytes = build_diff_payload(&plan, "usd", 0, &scan, &ext_data)
            .expect("build_diff_payload should succeed in test");
        let container = OmpfContainer::parse(&ompf_bytes).unwrap();
        assert_eq!(
            container.external_values.len(),
            1,
            "one external value → one extra OMPF part"
        );
        assert_eq!(container.external_values[0], b"external-blob-data");
    }

    #[test]
    fn build_diff_payload_multiple_ext_values_correct_indices() {
        use crate::domain::ompf::OmpfContainer;
        use crate::storage::diff_read::build_diff_payload;

        let fields = vec![
            make_ext_field(1, "f1", 1, 0x01),
            make_ext_field(2, "f2", 2, 0x02),
            make_ext_field(3, "f3", 3, 0x03),
        ];
        let scan = collect_external_value_hashes(&fields, &[]);

        let mut ext_data = HashMap::new();
        ext_data.insert(test_hash(0x01), b"data-1".to_vec());
        ext_data.insert(test_hash(0x02), b"data-2".to_vec());
        ext_data.insert(test_hash(0x03), b"data-3".to_vec());

        let plan = DiffPlan::FullContent {
            sections: vec![
                make_section(1, 0, 1, 3),
                make_section(2, 0, 1, 3),
                make_section(3, 0, 1, 3),
            ],
            fields,
            time_samples: vec![],
        };
        let ompf_bytes = build_diff_payload(&plan, "usd", 0, &scan, &ext_data)
            .expect("build_diff_payload should succeed in test");
        let container = OmpfContainer::parse(&ompf_bytes).unwrap();
        assert_eq!(
            container.external_values.len(),
            3,
            "3 distinct hashes → 3 OMPF parts"
        );
        // Parts are in first-appearance order.
        assert_eq!(container.external_values[0], b"data-1");
        assert_eq!(container.external_values[1], b"data-2");
        assert_eq!(container.external_values[2], b"data-3");

        // Verify ext_value_index in decoded OwnedDelta.
        let delta = oodf_ffi::parse_delta(container.delta_bytes).unwrap();
        assert_eq!(delta.set_fields[0].ext_value_index, 1);
        assert_eq!(delta.set_fields[1].ext_value_index, 2);
        assert_eq!(delta.set_fields[2].ext_value_index, 3);
    }

    #[test]
    fn build_diff_payload_deduped_same_index_one_ompf_part() {
        use crate::domain::ompf::OmpfContainer;
        use crate::storage::diff_read::build_diff_payload;

        // Two fields with the same hash → same ext_value_index, one OMPF part.
        let fields = vec![
            make_ext_field(1, "tex1", 1, 0xBB),
            make_ext_field(2, "tex2", 2, 0xBB),
        ];
        let scan = collect_external_value_hashes(&fields, &[]);

        let mut ext_data = HashMap::new();
        ext_data.insert(test_hash(0xBB), b"shared-data".to_vec());

        let plan = DiffPlan::FullContent {
            sections: vec![make_section(1, 0, 1, 2), make_section(2, 0, 1, 2)],
            fields,
            time_samples: vec![],
        };
        let ompf_bytes = build_diff_payload(&plan, "usd", 0, &scan, &ext_data)
            .expect("build_diff_payload should succeed in test");
        let container = OmpfContainer::parse(&ompf_bytes).unwrap();
        assert_eq!(
            container.external_values.len(),
            1,
            "deduped: one OMPF part for shared hash"
        );
        assert_eq!(container.external_values[0], b"shared-data");

        // Both fields must have the same ext_value_index.
        let delta = oodf_ffi::parse_delta(container.delta_bytes).unwrap();
        assert_eq!(delta.set_fields[0].ext_value_index, 1);
        assert_eq!(delta.set_fields[1].ext_value_index, 1);
    }

    #[test]
    fn build_diff_payload_up_to_date_returns_empty() {
        use crate::storage::diff_read::build_diff_payload;

        let plan = DiffPlan::UpToDate;
        let scan = ExternalValueScan {
            ordered_hashes: vec![],
            hash_to_index: HashMap::new(),
        };
        let ompf_bytes = build_diff_payload(&plan, "usd", 5, &scan, &HashMap::new())
            .expect("build_diff_payload should succeed in test");
        assert!(ompf_bytes.is_empty());
    }

    // ── C-4 regression: deep-chain stack-overflow protection ─────────────────
    //
    // Before b.55p, `sections_in_dfs_order` used a recursive walker. A linear
    // parent → child chain of ~10K–80K sections (trivially within the 100 MB
    // layer cap) would blow the thread stack on `plan(…) → plan_full_content`.
    // The iterative rewrite caps traversal at `MAX_SECTION_TREE_DEPTH` and
    // returns `DiffError::SectionTreeTooDeep` on breach instead of panicking.
    //
    // The small-stack worker thread is what makes a regression fail
    // deterministically: a 64 KiB stack makes a recursive walker overflow long
    // before 100K frames (and well before the cap), where a default-sized
    // stack might survive a shallower chain. It does not contain the failure:
    // a stack overflow on any thread aborts the whole test process. Under
    // nextest (process-per-test) that fails only this test; under plain
    // `cargo test` it takes the whole test binary down with it.

    /// Build a chain `root(0) → 1 → 2 → 3 → … → depth` and seed state so
    /// every link lives in `state.sections` and `state.children`. `depth == 0`
    /// returns an empty state.
    fn build_linear_chain(depth: usize) -> ObjectState {
        let mut state = ObjectState {
            object_id: 1,
            sequence_num: depth as u64,
            ..Default::default()
        };
        if depth == 0 {
            return state;
        }
        for i in 1..=depth {
            let parent = if i == 1 { 0 } else { (i - 1) as u64 };
            insert_section(&mut state, make_section(i as u64, parent, 1, 1));
        }
        state
    }

    /// Run `f` on a dedicated worker thread with a 64 KiB stack. Any recursion
    /// beyond ~a few hundred frames will overflow that budget. A stack
    /// overflow is not a panic the `join` below can catch: Rust's overflow
    /// handler prints `thread '<label>' (<tid>) has overflowed its stack` and
    /// aborts the whole test process, so the thread's name is the only
    /// pointer to the offending test. The `join` error arm catches an
    /// ordinary panic in `f` only.
    fn run_on_small_stack<R, F>(label: &str, f: F) -> R
    where
        F: FnOnce() -> R + Send + 'static,
        R: Send + 'static,
    {
        std::thread::Builder::new()
            .name(label.into())
            .stack_size(64 * 1024)
            .spawn(f)
            .expect("spawn small-stack worker")
            .join()
            .unwrap_or_else(|_| panic!("small-stack worker for {label} panicked"))
    }

    #[test]
    fn sections_in_dfs_order_depth_at_cap_succeeds() {
        // Boundary: chain of exactly MAX_SECTION_TREE_DEPTH must succeed.
        // Executed on a small-stack worker so any regression to recursive DFS
        // overflows deterministically (aborting the test process, see
        // `run_on_small_stack`).
        let ordered = run_on_small_stack("dfs-depth-at-cap", || {
            let state = build_linear_chain(MAX_SECTION_TREE_DEPTH);
            DiffEngine::sections_in_dfs_order(&state).expect("depth == cap must be accepted")
        });
        assert_eq!(
            ordered.len(),
            MAX_SECTION_TREE_DEPTH,
            "every section in the linear chain must be emitted exactly once"
        );
        // Parent-first ordering: section i emitted before section i+1.
        for (idx, sec) in ordered.iter().enumerate() {
            assert_eq!(
                sec.section_id,
                (idx + 1) as u64,
                "DFS must walk the chain in parent-first order"
            );
        }
    }

    #[test]
    fn sections_in_dfs_order_depth_one_over_cap_returns_too_deep() {
        // Boundary: one more than MAX_SECTION_TREE_DEPTH must return
        // SectionTreeTooDeep rather than blow the stack.
        let err = run_on_small_stack("dfs-depth-one-over-cap", || {
            let state = build_linear_chain(MAX_SECTION_TREE_DEPTH + 1);
            DiffEngine::sections_in_dfs_order(&state).unwrap_err()
        });
        let DiffError::SectionTreeTooDeep { depth, cap } = err;
        assert_eq!(cap, MAX_SECTION_TREE_DEPTH);
        // The walker checks depth > cap after popping, so the first offending
        // depth observed is cap + 1.
        assert_eq!(depth, MAX_SECTION_TREE_DEPTH + 1);
    }

    #[test]
    fn sections_in_dfs_order_deep_chain_does_not_blow_stack() {
        // The core DoS regression (C-4): a ~100K-section linear chain used to
        // crash the process with a recursive DFS stack overflow. On a 64 KiB
        // worker stack, the iterative walker must refuse with SectionTreeTooDeep
        // (via the cap) and never panic.
        let err = run_on_small_stack("dfs-100k-linear-chain", || {
            let state = build_linear_chain(100_000);
            DiffEngine::sections_in_dfs_order(&state).unwrap_err()
        });
        assert!(
            matches!(err, DiffError::SectionTreeTooDeep { .. }),
            "100K-chain must return SectionTreeTooDeep, got {err:?}"
        );
    }

    /// The sections a faithful builder emits for `state`, or its error:
    /// `plan_faithful_snapshot`'s sections, or `plan_faithful_coalesced`'s
    /// new and modified sections at `have_version` 0 (every chain section is
    /// newer, so it is new). Both walk with `faithful_walk`, which has its
    /// own depth check, separate from the legacy walk's.
    fn faithful_walk_sections(
        coalesced: bool,
        state: &ObjectState,
    ) -> Result<Vec<Section>, DiffError> {
        let plan = if coalesced {
            DiffEngine::plan_faithful_coalesced(state, &[], 0)?
        } else {
            DiffEngine::plan_faithful_snapshot(state)?
        };
        match plan {
            DiffPlan::FullContent { sections, .. } => Ok(sections),
            DiffPlan::Incremental {
                new_sections,
                modified_sections,
                ..
            } => Ok(new_sections.into_iter().chain(modified_sections).collect()),
            other => panic!("a faithful builder planned {other:?}"),
        }
    }

    /// The faithful builders, by label.
    const FAITHFUL_BUILDERS: [(&str, bool); 2] = [("snapshot", false), ("coalesced", true)];

    /// Boundary: a chain of exactly `MAX_SECTION_TREE_DEPTH` is walked whole,
    /// parent-first, by both faithful builders. Discriminated by an
    /// off-by-one in `faithful_walk`'s check (`>=`), which refuses it.
    #[test]
    fn faithful_builders_depth_at_cap_succeed_parent_first() {
        for (label, coalesced) in FAITHFUL_BUILDERS {
            let sections = run_on_small_stack("faithful-depth-at-cap", move || {
                let state = build_linear_chain(MAX_SECTION_TREE_DEPTH);
                faithful_walk_sections(coalesced, &state)
            })
            .unwrap_or_else(|e| panic!("{label}: depth == cap must be accepted, got {e:?}"));
            assert_eq!(
                section_ids(&sections),
                (1..=MAX_SECTION_TREE_DEPTH as u64).collect::<Vec<u64>>(),
                "{label}: every chain section once, parent-first"
            );
        }
    }

    /// Boundary: one section over the cap is refused with the first offending
    /// depth, `cap + 1`. Discriminated by removing `faithful_walk`'s check,
    /// which walks the chain and plans it.
    #[test]
    fn faithful_builders_depth_one_over_cap_return_too_deep() {
        for (label, coalesced) in FAITHFUL_BUILDERS {
            let result = run_on_small_stack("faithful-depth-one-over-cap", move || {
                let state = build_linear_chain(MAX_SECTION_TREE_DEPTH + 1);
                faithful_walk_sections(coalesced, &state).map(|s| s.len())
            });
            assert_eq!(
                result,
                Err(DiffError::SectionTreeTooDeep {
                    depth: MAX_SECTION_TREE_DEPTH + 1,
                    cap: MAX_SECTION_TREE_DEPTH,
                }),
                "{label}"
            );
        }
    }

    /// C-4 for the faithful walk: a 100K-section chain is refused with
    /// `SectionTreeTooDeep` on a 64 KiB stack, without overflowing it.
    /// Discriminated by removing `faithful_walk`'s check, which plans the
    /// whole chain instead.
    #[test]
    fn faithful_builders_deep_chain_return_too_deep_without_blowing_the_stack() {
        for (label, coalesced) in FAITHFUL_BUILDERS {
            let result = run_on_small_stack("faithful-100k-linear-chain", move || {
                let state = build_linear_chain(100_000);
                faithful_walk_sections(coalesced, &state).map(|s| s.len())
            });
            assert!(
                matches!(result, Err(DiffError::SectionTreeTooDeep { .. })),
                "{label}: a 100K chain must return SectionTreeTooDeep, got {result:?}"
            );
        }
    }

    #[test]
    fn sections_in_dfs_order_emits_orphans_after_reachable_post_rewrite() {
        // Regression for the H-dom-2 concern: the iterative rewrite must
        // preserve the orphan-after-reachable ordering contract. Builds a
        // 3-section reachable chain plus one orphan and verifies the orphan
        // appears strictly after every reachable section.
        let mut state = ObjectState {
            object_id: 1,
            sequence_num: 1,
            ..Default::default()
        };
        insert_section(&mut state, make_section(1, 0, 1, 1));
        insert_section(&mut state, make_section(2, 1, 1, 1));
        insert_section(&mut state, make_section(3, 2, 1, 1));
        // Orphan (parent 77 doesn't exist in state.sections).
        let orphan = make_section(90, 77, 1, 1);
        state.sections.insert(90, orphan);

        let ordered = DiffEngine::sections_in_dfs_order(&state).unwrap();
        let ids: Vec<u64> = ordered.iter().map(|s| s.section_id).collect();
        assert_eq!(
            ids.len(),
            4,
            "must emit all reachable sections and the orphan exactly once"
        );
        let pos_orphan = ids.iter().position(|&x| x == 90).unwrap();
        for reachable in [1u64, 2, 3] {
            let pos_reachable = ids.iter().position(|&x| x == reachable).unwrap();
            assert!(
                pos_reachable < pos_orphan,
                "orphan 90 at pos {pos_orphan} must come after reachable {reachable} at pos {pos_reachable}: {ids:?}"
            );
        }
    }

    // ── Corrected snapshot builder (plan_faithful_snapshot) tests ────────────
    //
    // The faithful builder (SR-1.1 / SR-1.2 / SR-5.2) derives a from-empty
    // full-state snapshot from persisted per-commit state. It differs from the
    // lossy `plan_full_content` (retained for the checkpoint assembler) on
    // three axes (b.bdx): it emits a DENSE UNIQUE `set_order` (`0..N-1`, ordered
    // by `(version, authoring_set_order)`) rather than the cumulative cursor; it
    // emits each section's derived per-parent sibling `section_order` (0-based
    // position within its parent's ordered children list) rather than the stored
    // `section_order`; and it drops net-live-unreachable (abandoned) prims
    // instead of appending them as orphans. It never emits tombstones —
    // `FullContent` has no deleted-section list at all.

    /// A field whose cumulative `set_order` cursor differs from its persisted
    /// authoring rank, so a test can prove which one the builder emits.
    fn make_authored_field(
        section_id: u64,
        key: &str,
        version: u64,
        cumulative_set_order: u64,
        authoring_set_order: u64,
    ) -> Field {
        Field {
            section_id,
            key_name: key.to_string(),
            value: FieldValue::Inline(vec![version as u8]),
            set_order: cumulative_set_order,
            authoring_set_order,
            version,
        }
    }

    // (a) Dense-unique set_order ordered by (version, authoring_set_order) — b.bdx.

    #[test]
    fn faithful_snapshot_emits_dense_unique_set_order_by_version_then_authoring() {
        // b.bdx: the corrected snapshot emits a DENSE UNIQUE `set_order`
        // (`0..N-1`), assigned by sorting the net-live fields on `(version,
        // authoring_set_order)` and taking each field's 0-based position — NOT
        // the raw authoring rank and NOT the cumulative cursor. Here all three
        // fields share one version, so the ordering key reduces to the authoring
        // rank; the dense indices therefore follow authoring order 0/1/2.
        let mut state = ObjectState {
            object_id: 1,
            sequence_num: 5,
            ..Default::default()
        };
        insert_section(&mut state, make_section(1, 0, 1, 5));
        // Cumulative cursors (100/101/102) differ from the authoring ranks
        // (2/0/1), and the authoring ranks are NOT in HashMap key order.
        insert_field(&mut state, make_authored_field(1, "a", 5, 100, 2));
        insert_field(&mut state, make_authored_field(1, "b", 5, 101, 0));
        insert_field(&mut state, make_authored_field(1, "c", 5, 102, 1));

        let plan = DiffEngine::plan_faithful_snapshot(&state).unwrap();
        match plan {
            DiffPlan::FullContent { fields, .. } => {
                assert_eq!(fields.len(), 3);
                let by_key = |k: &str| fields.iter().find(|f| f.key_name == k).unwrap();
                // Dense-unique indices ordered by (version, authoring): b(auth 0)
                // → 0, c(auth 1) → 1, a(auth 2) → 2.
                assert_eq!(by_key("b").set_order, 0);
                assert_eq!(by_key("c").set_order, 1);
                assert_eq!(
                    by_key("a").set_order,
                    2,
                    "field a (authoring rank 2) must emit dense index 2, not the cumulative cursor 100"
                );
                // Global uniqueness: emitted set_orders are exactly the dense
                // range {0,1,2}.
                let mut orders: Vec<u64> = fields.iter().map(|f| f.set_order).collect();
                orders.sort_unstable();
                assert_eq!(
                    orders,
                    vec![0, 1, 2],
                    "emitted set_order must be the dense unique range 0..N, never the cumulative cursor"
                );
                // The cumulative cursor values (100/101/102) must never appear.
                for f in &fields {
                    assert!(
                        f.set_order < 3,
                        "emitted set_order must be a dense index, not the cumulative cursor, got {} for {}",
                        f.set_order,
                        f.key_name
                    );
                }
            }
            other => panic!("expected FullContent, got {other:?}"),
        }
    }

    #[test]
    fn faithful_snapshot_set_order_globally_unique_across_colliding_authoring_ranks() {
        // b.bdx — the load-bearing property: raw per-delta authoring ranks
        // RESET each commit and so COLLIDE across commits/prims (the first field
        // of every batch is rank 0). The Kit client keys catch-up fields into an
        // ordered `map<pair<saveCount,setOrder>>` under one shared `saveCount`, so
        // a duplicate `setOrder` is SILENTLY DROPPED. The dense-unique scheme must
        // therefore emit GLOBALLY-UNIQUE set_orders even when the raw authoring
        // ranks collide — this is what protects the client. Two prims, two commits,
        // authoring ranks {0,1} reused in each → a raw-rank emission would collide.
        let mut state = ObjectState {
            object_id: 1,
            sequence_num: 4,
            ..Default::default()
        };
        insert_section(&mut state, make_section(1, 0, 1, 4));
        insert_section(&mut state, make_section(2, 0, 1, 4));
        // Prim 1, commit version 3: authoring ranks 0,1.
        insert_field(&mut state, make_authored_field(1, "a", 3, 10, 0));
        insert_field(&mut state, make_authored_field(1, "b", 3, 11, 1));
        // Prim 2, commit version 4: authoring ranks 0,1 AGAIN (collision).
        insert_field(&mut state, make_authored_field(2, "c", 4, 20, 0));
        insert_field(&mut state, make_authored_field(2, "d", 4, 21, 1));

        let plan = DiffEngine::plan_faithful_snapshot(&state).unwrap();
        match plan {
            DiffPlan::FullContent { fields, .. } => {
                assert_eq!(fields.len(), 4);
                // (b) GLOBAL UNIQUENESS despite colliding raw authoring ranks.
                let mut orders: Vec<u64> = fields.iter().map(|f| f.set_order).collect();
                orders.sort_unstable();
                assert_eq!(
                    orders,
                    vec![0, 1, 2, 3],
                    "emitted set_order must be globally unique (dense 0..N) even when raw \
                     authoring ranks collide across prims/commits — a raw-rank emission would \
                     produce {{0,1,0,1}} and the client would drop half the fields"
                );
                // (c) RELATIVE ORDER = sort by (version, authoring): a<b<c<d.
                let by_key = |k: &str| fields.iter().find(|f| f.key_name == k).unwrap().set_order;
                assert!(
                    by_key("a") < by_key("b")
                        && by_key("b") < by_key("c")
                        && by_key("c") < by_key("d"),
                    "dense set_order must preserve the (version, authoring) apply order a<b<c<d"
                );
            }
            other => panic!("expected FullContent, got {other:?}"),
        }
    }

    // (b) Sections parent→child, each carrying its real per-parent sibling index.

    #[test]
    fn faithful_snapshot_sections_parent_before_child_with_sibling_order() {
        // b.bdx: the corrected snapshot emits each section's REAL per-parent
        // sibling index (0-based position within its parent's ordered children
        // list) in `section_order`, replacing the stored value — the from-empty
        // Kit client needs it to reconstruct xformOp order (translate→rotate→scale).
        // Same three-level tree shape as the plan_full_content DFS test.
        //   root(0) ├─1 ├─3 └─4   └─2 └─5
        //   root children [1,2] → sec1=0, sec2=1
        //   sec1 children [3,4] → sec3=0, sec4=1
        //   sec2 children [5]   → sec5=0
        let mut state = ObjectState {
            object_id: 1,
            sequence_num: 5,
            ..Default::default()
        };
        insert_section(&mut state, make_section(1, 0, 1, 5));
        insert_section(&mut state, make_section(2, 0, 1, 5));
        insert_section(&mut state, make_section(3, 1, 1, 5));
        insert_section(&mut state, make_section(4, 1, 1, 5));
        insert_section(&mut state, make_section(5, 2, 1, 5));

        let plan = DiffEngine::plan_faithful_snapshot(&state).unwrap();
        match plan {
            DiffPlan::FullContent { sections, .. } => {
                let ids = section_ids(&sections);
                assert_eq!(ids.len(), 5, "all 5 reachable sections emitted");
                let pos = |id: u64| ids.iter().position(|&x| x == id).unwrap();
                assert!(pos(1) < pos(3), "section 1 must precede child 3");
                assert!(pos(1) < pos(4), "section 1 must precede child 4");
                assert!(pos(2) < pos(5), "section 2 must precede child 5");
                // The emitted section_order is the sibling index within the
                // parent's children list, NOT the stored `section_order`
                // (make_section persists section_order == section_id).
                let sibling_order = |id: u64| {
                    sections
                        .iter()
                        .find(|s| s.section_id == id)
                        .unwrap()
                        .section_order
                };
                assert_eq!(sibling_order(1), 0, "sec1 is child index 0 under root");
                assert_eq!(sibling_order(2), 1, "sec2 is child index 1 under root");
                assert_eq!(sibling_order(3), 0, "sec3 is child index 0 under sec1");
                assert_eq!(sibling_order(4), 1, "sec4 is child index 1 under sec1");
                assert_eq!(sibling_order(5), 0, "sec5 is child index 0 under sec2");
            }
            other => panic!("expected FullContent, got {other:?}"),
        }
    }

    // (c) Net-live-only drop.

    #[test]
    fn faithful_snapshot_drops_abandoned_prim_with_no_tombstone() {
        let mut state = ObjectState {
            object_id: 1,
            sequence_num: 5,
            ..Default::default()
        };
        // Net-live tree: root -> 1 -> 3.
        insert_section(&mut state, make_section(1, 0, 1, 5));
        insert_section(&mut state, make_section(3, 1, 1, 5));
        // Prim 99 created then abandoned: its parent 77 was deleted mid-session,
        // leaving 99 unreachable from root. Its field + time sample go with it.
        state.sections.insert(99, make_section(99, 77, 1, 5));
        state.children.entry(77).or_default().push(99);
        insert_field(&mut state, make_field(99, "ghost", 5));
        insert_time_sample(&mut state, make_time_sample(99, 0.0, 5));

        let plan = DiffEngine::plan_faithful_snapshot(&state).unwrap();
        match plan {
            // FullContent structurally carries NO deleted-section / tombstone list.
            DiffPlan::FullContent {
                sections,
                fields,
                time_samples,
            } => {
                let ids = section_ids(&sections);
                assert_eq!(ids.len(), 2, "only the two net-live sections, got {ids:?}");
                assert!(
                    !ids.contains(&99),
                    "abandoned prim must be dropped, never appended as an orphan or tombstone"
                );
                assert!(
                    !fields.iter().any(|f| f.section_id == 99),
                    "field on abandoned prim must be dropped"
                );
                assert!(
                    !time_samples.iter().any(|ts| ts.section_id == 99),
                    "time sample on abandoned prim must be dropped"
                );
            }
            other => panic!("expected FullContent (no tombstone list), got {other:?}"),
        }

        // Contrast pin: the lossy checkpoint path STILL appends the abandoned
        // prim as an orphan — the net-live drop is unique to the faithful path.
        match DiffEngine::plan_full_content(&state).unwrap() {
            DiffPlan::FullContent { sections, .. } => {
                assert!(
                    section_ids(&sections).contains(&99),
                    "plan_full_content must still emit the orphan (unchanged lossy behavior)"
                );
            }
            other => panic!("expected FullContent, got {other:?}"),
        }
    }

    #[test]
    fn faithful_snapshot_reparented_prim_appears_once_under_final_parent() {
        let mut state = ObjectState {
            object_id: 1,
            sequence_num: 5,
            ..Default::default()
        };
        // Parents X=1 and Y=2 under root.
        insert_section(&mut state, make_section(1, 0, 1, 5));
        insert_section(&mut state, make_section(2, 0, 1, 5));
        // Prim 5 created under X=1, then reparented to Y=2: its stored parent_id
        // is the FINAL parent 2, and it is linked only under 2's children.
        insert_section(&mut state, make_section(5, 2, 1, 5));

        let plan = DiffEngine::plan_faithful_snapshot(&state).unwrap();
        match plan {
            DiffPlan::FullContent { sections, .. } => {
                let occurrences = sections.iter().filter(|s| s.section_id == 5).count();
                assert_eq!(
                    occurrences, 1,
                    "reparented prim must appear exactly once, never duplicated"
                );
                let sec5 = sections.iter().find(|s| s.section_id == 5).unwrap();
                assert_eq!(
                    sec5.parent_id, 2,
                    "reparented prim must sit under its net-live parent Y=2, not stale under X=1"
                );
                let ids = section_ids(&sections);
                let pos = |id: u64| ids.iter().position(|&x| x == id).unwrap();
                assert!(
                    pos(2) < pos(5),
                    "prim 5 must follow its final parent 2 in DFS order, got {ids:?}"
                );
            }
            other => panic!("expected FullContent, got {other:?}"),
        }
    }

    // (d) Functional: xformOp sibling order (b.bdx), independent of C++ parity.

    #[test]
    fn faithful_snapshot_xform_op_children_get_ascending_sibling_order() {
        // b.bdx FUNCTIONAL correctness (no C++ oracle involved): a prim whose
        // ordered children are the transform ops translate → rotateXYZ → scale →
        // xformOpOrder must emit ASCENDING sibling indices 0,1,2,3 — NOT all 0.
        // The from-empty Kit client reconstructs xformOp application order
        // (translate→rotate→scale) from this sibling order, so getting it wrong
        // (the pre-fix hardcoded 0) transforms the object incorrectly.
        let mut state = ObjectState {
            object_id: 1,
            sequence_num: 2,
            ..Default::default()
        };
        // root(0) → prim(1); prim(1)'s children authored in xformOp order.
        insert_section(&mut state, make_section(1, 0, 1, 2));
        let mut xform = |id: u64, name: &str| {
            let mut s = make_section(id, 1, 1, 2);
            s.name = name.to_string();
            // Persist a DISTINCT non-monotonic stored section_order so the test
            // proves the emitted value is the sibling index, not the stored one.
            s.section_order = 900 - id as u32;
            insert_section(&mut state, s);
        };
        xform(10, "xformOp:translate");
        xform(11, "xformOp:rotateXYZ");
        xform(12, "xformOp:scale");
        xform(13, "xformOpOrder");

        let plan = DiffEngine::plan_faithful_snapshot(&state).unwrap();
        match plan {
            DiffPlan::FullContent { sections, .. } => {
                let order_of = |id: u64| {
                    sections
                        .iter()
                        .find(|s| s.section_id == id)
                        .unwrap()
                        .section_order
                };
                assert_eq!(order_of(10), 0, "translate must be sibling 0");
                assert_eq!(order_of(11), 1, "rotateXYZ must be sibling 1");
                assert_eq!(order_of(12), 2, "scale must be sibling 2");
                assert_eq!(order_of(13), 3, "xformOpOrder must be sibling 3");
                // The transform-op apply order is strictly ascending, and the
                // indices are NOT all 0 (the pre-fix bug).
                assert!(
                    order_of(10) < order_of(11)
                        && order_of(11) < order_of(12)
                        && order_of(12) < order_of(13),
                    "xformOp sibling order must be strictly ascending translate<rotate<scale<order"
                );
                let all_zero = [10u64, 11, 12, 13].iter().all(|&id| order_of(id) == 0);
                assert!(
                    !all_zero,
                    "sibling order must not collapse to all-0 (pre-fix bug)"
                );
            }
            other => panic!("expected FullContent, got {other:?}"),
        }
    }

    // (e) Time samples + external values emitted; no tombstones.

    #[test]
    fn faithful_snapshot_emits_time_samples_and_external_fields_no_tombstones() {
        let mut state = ObjectState {
            object_id: 1,
            sequence_num: 5,
            ..Default::default()
        };
        insert_section(&mut state, make_section(1, 0, 1, 5));
        insert_time_sample(&mut state, make_time_sample(1, 0.0, 3));
        // A field whose value is an external (content-addressed) reference.
        insert_field(&mut state, make_ext_field(1, "tex", 4, 0xAB));

        let plan = DiffEngine::plan_faithful_snapshot(&state).unwrap();
        match plan {
            // FullContent has no deleted-section field — asserting the variant
            // proves the plan carries no tombstones.
            DiffPlan::FullContent {
                fields,
                time_samples,
                ..
            } => {
                assert_eq!(time_samples.len(), 1, "time sample must be carried");
                assert_eq!(time_samples[0].section_id, 1);
                let ext = fields
                    .iter()
                    .find(|f| f.key_name == "tex")
                    .expect("external-valued field must be emitted, not dropped");
                match &ext.value {
                    FieldValue::External { hash, size, .. } => {
                        assert_eq!(*hash, test_hash(0xAB), "external hash carried unchanged");
                        assert_eq!(*size, 1024, "external size carried unchanged");
                    }
                    other => panic!("expected External value, got {other:?}"),
                }
            }
            other => panic!("plan must be FullContent (no tombstone list), got {other:?}"),
        }
    }

    // Non-regression pin: the lossy plan_full_content path is unchanged.

    #[test]
    fn plan_full_content_unchanged_for_representative_state() {
        // Representative multi-section tree + orphan, fields (one with a
        // cumulative cursor distinct from its authoring rank), and time samples.
        // Pins today's lossy behavior so the additive faithful builder cannot
        // perturb the checkpoint-assembler caller.
        let mut state = ObjectState {
            object_id: 1,
            sequence_num: 5,
            ..Default::default()
        };
        insert_section(&mut state, make_section(1, 0, 1, 5));
        insert_section(&mut state, make_section(2, 1, 1, 5));
        insert_section(&mut state, make_section(3, 1, 1, 5));
        state.sections.insert(90, make_section(90, 77, 1, 5)); // orphan
        insert_field(&mut state, make_authored_field(1, "color", 5, 42, 1));
        insert_field(&mut state, make_field(2, "size", 5));
        insert_time_sample(&mut state, make_time_sample(1, 0.0, 3));
        insert_time_sample(&mut state, make_time_sample(1, 1.0, 4));

        let plan = DiffEngine::plan_full_content(&state).unwrap();
        match plan {
            DiffPlan::FullContent {
                sections,
                fields,
                time_samples,
            } => {
                let ids = section_ids(&sections);
                assert_eq!(ids.len(), 4, "3 reachable + 1 orphan");
                let pos = |id: u64| ids.iter().position(|&x| x == id).unwrap();
                assert!(
                    pos(1) < pos(2) && pos(1) < pos(3),
                    "parent before children, got {ids:?}"
                );
                assert!(
                    pos(1) < pos(90) && pos(2) < pos(90) && pos(3) < pos(90),
                    "orphan appended after all reachable, got {ids:?}"
                );
                assert_eq!(fields.len(), 2, "all fields emitted regardless of liveness");
                let color = fields.iter().find(|f| f.key_name == "color").unwrap();
                assert_eq!(
                    color.set_order, 42,
                    "plan_full_content must preserve the CUMULATIVE set_order, not the authoring rank"
                );
                assert_eq!(time_samples.len(), 2, "both time samples emitted");
            }
            other => panic!("expected FullContent, got {other:?}"),
        }
    }

    // ── Corrected coalesced builder (plan_faithful_coalesced) tests ──────────
    //
    // b.3pe: the reconnect catch-up builder used to emit the STORED CUMULATIVE
    // `set_order` verbatim. That cursor is a per-`SetField` accumulator bumped
    // in WIRE-ARRIVAL order, and the Kit writer serializes a commit's fields out
    // of a `std::unordered_map`, so arrival order is hash-bucket order —
    // uncorrelated with the writer's declared `authoring_set_order`. The
    // corrected builder re-emits a DENSE UNIQUE index `0..N-1` assigned over the
    // total-order key `(version, authoring_set_order, section_id, key_name)`,
    // converging on the rule `plan_faithful_snapshot` already used (b.bdx).
    //
    // The tests below mirror the two snapshot dense-order tests, plus the case
    // the snapshot suite structurally CANNOT express: a state whose cumulative
    // cursor order DISAGREES with the authoring order inside ONE version, where
    // the pre-fix verbatim emission and the corrected emission produce different
    // relative orders. Each of these fails against the pre-fix builder.

    /// Sorted-by-emitted-`set_order` `(section_id, key_name)` sequence — the only
    /// two properties the Kit client can observe (it re-bases the absolute
    /// integers onto its own counter).
    fn emitted_order(fields: &[Field]) -> Vec<(u64, String)> {
        let mut v: Vec<(u64, u64, String)> = fields
            .iter()
            .map(|f| (f.set_order, f.section_id, f.key_name.clone()))
            .collect();
        v.sort();
        v.into_iter().map(|(_, s, k)| (s, k)).collect()
    }

    /// The three coalesced-plan vectors b.3pe made deterministic:
    /// `(changed_fields, changed_time_samples, deleted_sections)`.
    type IncrementalParts = (Vec<Field>, Vec<TimeSample>, Vec<(u64, u64, u64)>);

    fn unwrap_incremental(plan: DiffPlan) -> IncrementalParts {
        match plan {
            DiffPlan::Incremental {
                changed_fields,
                changed_time_samples,
                deleted_sections,
                ..
            } => (changed_fields, changed_time_samples, deleted_sections),
            other => panic!("expected Incremental, got {other:?}"),
        }
    }

    // (a) Dense-unique set_order ordered by (version, authoring_set_order) — b.3pe.

    #[test]
    fn faithful_coalesced_set_order_globally_unique_across_colliding_authoring_ranks() {
        // Coalesced sibling of
        // `faithful_snapshot_set_order_globally_unique_across_colliding_authoring_ranks`.
        // Raw per-delta authoring ranks RESET each commit, so they collide across
        // the commits a coalesced window folds together. The Kit client keys
        // catch-up fields into an ordered `map<pair<saveCount, setOrder>>` under
        // ONE shared `saveCount`, so a duplicate `setOrder` is silently dropped —
        // the dense re-index is what makes the emission globally unique.
        let mut state = ObjectState {
            object_id: 1,
            sequence_num: 4,
            ..Default::default()
        };
        insert_section(&mut state, make_section(1, 0, 1, 4));
        insert_section(&mut state, make_section(2, 0, 1, 4));
        // Prim 1, commit version 3: authoring ranks 0,1.
        insert_field(&mut state, make_authored_field(1, "a", 3, 10, 0));
        insert_field(&mut state, make_authored_field(1, "b", 3, 11, 1));
        // Prim 2, commit version 4: authoring ranks 0,1 AGAIN (collision).
        insert_field(&mut state, make_authored_field(2, "c", 4, 20, 0));
        insert_field(&mut state, make_authored_field(2, "d", 4, 21, 1));

        let plan = DiffEngine::plan_faithful_coalesced(&state, &[], 2).unwrap();
        let (fields, _, _) = unwrap_incremental(plan);
        assert_eq!(fields.len(), 4);
        let mut orders: Vec<u64> = fields.iter().map(|f| f.set_order).collect();
        orders.sort_unstable();
        assert_eq!(
            orders,
            vec![0, 1, 2, 3],
            "emitted set_order must be globally unique (dense 0..N) even when raw authoring \
             ranks collide across the commits a coalesced window folds together — a raw-rank \
             emission would produce {{0,1,0,1}} and the client would drop half the fields"
        );
        let by_key = |k: &str| fields.iter().find(|f| f.key_name == k).unwrap().set_order;
        assert!(
            by_key("a") < by_key("b") && by_key("b") < by_key("c") && by_key("c") < by_key("d"),
            "dense set_order must preserve the (version, authoring) apply order a<b<c<d"
        );
    }

    #[test]
    fn faithful_coalesced_set_order_follows_authoring_rank_not_cumulative_cursor() {
        // THE b.3pe REGRESSION CRUX — the case the snapshot suite cannot express,
        // because only this builder ever emitted the cumulative cursor verbatim.
        //
        // Real Kit-writer shape: one commit (version 6) writes four fields whose
        // WIRE-ARRIVAL order (and therefore the cumulative cursor the server
        // accumulated: 50/51/52/53) is the writer's `unordered_map` bucket order,
        // which DISAGREES with the writer's declared authoring ranks (3/1/0/2).
        //
        // Pre-fix (cumulative emitted verbatim) the client would apply
        //   w(50) → x(51) → y(52) → z(53)
        // Post-fix (dense over (version, authoring, section, key)) it applies
        //   y(auth 0) → x(auth 1) → z(auth 2) → w(auth 3)
        // The two sequences are disjoint in every position, so this assertion is
        // a hard fail against the pre-fix builder.
        let mut state = ObjectState {
            object_id: 1,
            sequence_num: 6,
            ..Default::default()
        };
        insert_section(&mut state, make_section(1, 0, 1, 6));
        insert_field(&mut state, make_authored_field(1, "w", 6, 50, 3));
        insert_field(&mut state, make_authored_field(1, "x", 6, 51, 1));
        insert_field(&mut state, make_authored_field(1, "y", 6, 52, 0));
        insert_field(&mut state, make_authored_field(1, "z", 6, 53, 2));

        let plan = DiffEngine::plan_faithful_coalesced(&state, &[], 5).unwrap();
        let (fields, _, _) = unwrap_incremental(plan);
        assert_eq!(fields.len(), 4);

        let keys: Vec<String> = emitted_order(&fields).into_iter().map(|(_, k)| k).collect();
        assert_eq!(
            keys,
            vec!["y", "x", "z", "w"],
            "emitted relative order must follow the writer's declared authoring ranks \
             (y=0, x=1, z=2, w=3), NOT the wire-arrival cumulative cursor"
        );
        // The pre-fix sequence must NOT be what came out.
        assert_ne!(
            keys,
            vec!["w", "x", "y", "z"],
            "emitting the stored cumulative set_order verbatim reproduces hash-bucket \
             arrival order — the exact b.3pe defect"
        );
        // And the cumulative cursor values themselves are gone from the wire.
        for f in &fields {
            assert!(
                f.set_order < 4,
                "emitted set_order must be a dense index, got cumulative-looking {} for {}",
                f.set_order,
                f.key_name
            );
        }
        // The persisted authoring ranks survive the emit-time re-index
        // un-mutated — the dense index is derived, nothing new is stored and
        // the sort key itself is not rewritten.
        let by_key = |k: &str| fields.iter().find(|f| f.key_name == k).unwrap();
        assert_eq!(by_key("w").authoring_set_order, 3);
        assert_eq!(by_key("x").authoring_set_order, 1);
        assert_eq!(by_key("y").authoring_set_order, 0);
        assert_eq!(by_key("z").authoring_set_order, 2);
    }

    #[test]
    fn faithful_coalesced_dense_index_spans_only_the_post_have_version_slice() {
        // The dense range is `0..N-1` over the FILTERED (version > have_version,
        // net-live) set — not over all persisted fields. A pre-`have_version`
        // field must neither be emitted nor consume an index, and the surviving
        // fields must still be dense from 0. Also pins that a field anchored to a
        // net-live prim but authored at exactly `have_version` is excluded.
        let mut state = ObjectState {
            object_id: 1,
            sequence_num: 7,
            ..Default::default()
        };
        insert_section(&mut state, make_section(1, 0, 1, 7));
        insert_section(&mut state, make_section(2, 1, 1, 7));
        // Below / at the watermark — already applied by the reconnector.
        insert_field(&mut state, make_authored_field(1, "old", 2, 5, 0));
        insert_field(&mut state, make_authored_field(1, "at_hv", 4, 6, 1));
        // Above the watermark — the coalesced slice.
        insert_field(&mut state, make_authored_field(2, "n1", 6, 300, 1));
        insert_field(&mut state, make_authored_field(2, "n2", 6, 301, 0));
        insert_field(&mut state, make_authored_field(1, "n3", 7, 302, 0));

        let plan = DiffEngine::plan_faithful_coalesced(&state, &[], 4).unwrap();
        let (fields, _, _) = unwrap_incremental(plan);
        let keys: Vec<String> = emitted_order(&fields).into_iter().map(|(_, k)| k).collect();
        assert_eq!(
            keys,
            vec!["n2", "n1", "n3"],
            "only fields with version > have_version are emitted, ordered by \
             (version, authoring_set_order)"
        );
        let mut orders: Vec<u64> = fields.iter().map(|f| f.set_order).collect();
        orders.sort_unstable();
        assert_eq!(
            orders,
            vec![0, 1, 2],
            "the dense range covers the emitted slice only — it must start at 0 and have no \
             gaps left by the filtered-out pre-have_version fields"
        );
    }

    #[test]
    fn faithful_coalesced_drops_fields_on_unreachable_prims_without_index_gaps() {
        // Content filter × dense index: a field anchored to a net-live-unreachable
        // (abandoned) prim is DROPPED, and dropping it must not punch a hole in
        // the dense range emitted to the client.
        let mut state = ObjectState {
            object_id: 1,
            sequence_num: 6,
            ..Default::default()
        };
        insert_section(&mut state, make_section(1, 0, 1, 6));
        insert_section(&mut state, make_section(2, 1, 1, 6));
        // Section 90 hangs off a non-existent parent 77 → unreachable from root.
        state.sections.insert(90, make_section(90, 77, 1, 6));
        insert_field(&mut state, make_authored_field(2, "live_a", 6, 10, 1));
        insert_field(&mut state, make_authored_field(90, "ghost", 6, 11, 0));
        insert_field(&mut state, make_authored_field(1, "live_b", 6, 12, 2));

        let plan = DiffEngine::plan_faithful_coalesced(&state, &[], 3).unwrap();
        let (fields, _, deleted) = unwrap_incremental(plan);
        let keys: Vec<String> = emitted_order(&fields).into_iter().map(|(_, k)| k).collect();
        assert_eq!(
            keys,
            vec!["live_a", "live_b"],
            "fields anchored to unreachable prims are dropped"
        );
        let mut orders: Vec<u64> = fields.iter().map(|f| f.set_order).collect();
        orders.sort_unstable();
        assert_eq!(
            orders,
            vec![0, 1],
            "the dense range must close over the dropped ghost field, leaving no gap"
        );
        // The abandoned prim is still retracted (unified ghost set, arm (b)).
        assert!(
            deleted.iter().any(|(sid, _, _)| *sid == 90),
            "abandoned prim 90 must appear in the unified ghost retraction set, got {deleted:?}"
        );
    }

    // (b) Deterministic emission order across independently-built equal states.

    #[test]
    fn faithful_coalesced_plan_is_deterministic_for_equal_state() {
        // b.3pe determinism: `changed_fields`, `changed_time_samples` and
        // `deleted_sections` are all materialized from `HashMap` iteration (and,
        // for the tombstone arm, an unordered Valkey SET read), so an unsorted
        // builder emits a different VECTOR ORDER per process / per map instance
        // even for identical state. Two reconnectors served the same state must
        // receive the same plan.
        //
        // Repeating on ONE state would not prove this — a single `HashMap`
        // iterates consistently within its own lifetime. The states below are
        // built INDEPENDENTLY (own `RandomState` seed) with the insertion order
        // of the UNORDERED containers reversed, which is what actually varies
        // the iteration order.
        //
        // `children` insertion order is deliberately held FIXED: it is not
        // incidental map order but real persisted data (a snapshot read takes
        // each parent's child list with `LRANGE`, which is ordered), and
        // it is what the emitted per-parent `section_order` sibling index means.
        // Permuting it would change the state's meaning, not just its layout.
        let build = |reverse: bool| {
            let mut state = ObjectState {
                object_id: 1,
                sequence_num: 9,
                ..Default::default()
            };
            for s in [
                make_section(1, 0, 1, 9),
                make_section(2, 1, 1, 9),
                make_section(3, 1, 1, 9),
                make_section(4, 2, 5, 9),
            ] {
                insert_section(&mut state, s);
            }
            // Abandoned prims (unreachable from root) → ghost-retraction arm (b).
            let mut ghosts: Vec<Section> = vec![
                make_section(80, 77, 1, 9),
                make_section(81, 77, 1, 9),
                make_section(82, 77, 1, 9),
            ];
            // Colliding authoring ranks across prims and commits, and cumulative
            // cursors deliberately out of step with them.
            let mut fields: Vec<Field> = vec![
                make_authored_field(2, "f1", 6, 900, 1),
                make_authored_field(3, "f2", 6, 901, 1),
                make_authored_field(4, "f3", 7, 902, 0),
                make_authored_field(1, "f4", 7, 903, 0),
                make_authored_field(2, "f5", 8, 904, 2),
                make_authored_field(3, "f6", 9, 905, 0),
            ];
            let mut samples: Vec<TimeSample> = vec![
                make_time_sample(2, 3.0, 7),
                make_time_sample(2, 1.0, 7),
                make_time_sample(3, 2.0, 8),
                make_time_sample(4, 0.5, 9),
            ];
            if reverse {
                ghosts.reverse();
                fields.reverse();
                samples.reverse();
            }
            for g in ghosts {
                // Inserted WITHOUT registering in `children`, matching the other
                // orphan fixtures in this module.
                state.sections.insert(g.section_id, g);
            }
            for f in fields {
                insert_field(&mut state, f);
            }
            for ts in samples {
                insert_time_sample(&mut state, ts);
            }
            state
        };

        // Tombstone arm (a), also supplied in two different orders.
        let tombstones_fwd: Vec<(u64, u64, u64)> = vec![(70, 1, 6), (71, 2, 7), (72, 1, 9)];
        let tombstones_rev: Vec<(u64, u64, u64)> =
            tombstones_fwd.iter().rev().copied().collect::<Vec<_>>();

        let plan_a =
            DiffEngine::plan_faithful_coalesced(&build(false), &tombstones_fwd, 5).unwrap();
        let plan_b = DiffEngine::plan_faithful_coalesced(&build(true), &tombstones_rev, 5).unwrap();

        assert_eq!(
            plan_a, plan_b,
            "plan_faithful_coalesced must emit an identical plan — including the VECTOR ORDER of \
             changed_fields, changed_time_samples and deleted_sections — for logically identical \
             state, regardless of HashMap iteration or tombstone-set arrival order"
        );

        // Repeating on the SAME state is also stable (weaker, but pins that no
        // interior mutability or clock/randomness leaks into the plan).
        let state = build(false);
        assert_eq!(
            DiffEngine::plan_faithful_coalesced(&state, &tombstones_fwd, 5).unwrap(),
            DiffEngine::plan_faithful_coalesced(&state, &tombstones_fwd, 5).unwrap(),
            "two calls on one state must agree"
        );

        // Spot-check the sorted invariants directly, so a future regression names
        // itself rather than surfacing only as an opaque plan inequality.
        let (fields, samples, deleted) = unwrap_incremental(plan_a);
        assert!(
            deleted.windows(2).all(|w| w[0].0 < w[1].0),
            "deleted_sections must be sorted by section_id, got {deleted:?}"
        );
        assert_eq!(
            fields.iter().map(|f| f.set_order).collect::<Vec<u64>>(),
            (0..fields.len() as u64).collect::<Vec<u64>>(),
            "changed_fields must be emitted already in dense set_order sequence"
        );
        assert!(
            samples
                .windows(2)
                .all(|w| (w[0].version, w[0].section_id, w[0].time)
                    < (w[1].version, w[1].section_id, w[1].time)),
            "changed_time_samples must be sorted on (version, section_id, time)"
        );
    }

    #[test]
    fn faithful_snapshot_time_samples_emitted_in_deterministic_order() {
        // Sibling of the coalesced determinism pin, for the from-empty builder's
        // `time_samples` vector (b.3pe made it sorted too). Same construction:
        // two independently-seeded states, reversed insertion order.
        let build = |reverse: bool| {
            let mut state = ObjectState {
                object_id: 1,
                sequence_num: 9,
                ..Default::default()
            };
            insert_section(&mut state, make_section(1, 0, 1, 9));
            insert_section(&mut state, make_section(2, 1, 1, 9));
            let mut samples: Vec<TimeSample> = vec![
                make_time_sample(2, 3.0, 5),
                make_time_sample(1, 1.0, 5),
                make_time_sample(2, 1.0, 5),
                make_time_sample(1, 2.0, 6),
                make_time_sample(2, 0.0, 7),
            ];
            if reverse {
                samples.reverse();
            }
            for ts in samples {
                insert_time_sample(&mut state, ts);
            }
            state
        };

        let plan_a = DiffEngine::plan_faithful_snapshot(&build(false)).unwrap();
        let plan_b = DiffEngine::plan_faithful_snapshot(&build(true)).unwrap();
        assert_eq!(
            plan_a, plan_b,
            "plan_faithful_snapshot must emit time_samples in a stable order for identical state"
        );
        match plan_a {
            DiffPlan::FullContent { time_samples, .. } => {
                assert_eq!(time_samples.len(), 5);
                assert!(
                    time_samples
                        .windows(2)
                        .all(|w| (w[0].version, w[0].section_id, w[0].time)
                            < (w[1].version, w[1].section_id, w[1].time)),
                    "time_samples must be sorted on (version, section_id, time)"
                );
            }
            other => panic!("expected FullContent, got {other:?}"),
        }
    }

    // ── b.gcb / b.vz7: plan_full_content deterministic emission ──────────────
    //
    // `plan_full_content` is the SHARED emitter behind the checkpoint base
    // assembly (`build_reset_ompf_from_state`) and the legacy catch-up
    // cold-start branch (`DiffEngine::plan` with `have_version == 0`). All three
    // of its emitted vectors used to inherit `HashMap` iteration order, so the
    // same logical snapshot produced byte-divergent base OMPF blobs (and a
    // byte-divergent ext-refs sidecar) across replicas — the SR-9.2
    // duplicate-leader re-base could not be argued byte-idempotent.
    //
    // The construction below is the b.3pe sibling's
    // (`faithful_coalesced_plan_is_deterministic_for_equal_state`, above): two
    // INDEPENDENTLY-built states, each unordered container filled in the reverse
    // insertion order in the second build. See that test's doc-comment for why
    // an in-process pair is a genuine witness for the cross-replica claim
    // (`RandomState::new()` draws a fresh key per INSTANCE) and why `children`
    // insertion order is deliberately held FIXED (it is real `LRANGE`-ordered
    // persisted data, so permuting it changes the state's MEANING). The
    // genuinely cross-PROCESS measurement is the golden generator's job
    // (t3.dfh.be.q7.f2); nothing here claims it.

    /// An external-valued field with an explicit cumulative cursor and authoring
    /// rank — `make_authored_field`'s counterpart for the ext-sidecar fixture,
    /// where the leading `(version, authoring_set_order)` pair must be made to
    /// tie while each field still references a distinct blob.
    fn make_authored_ext_field(
        section_id: u64,
        key: &str,
        version: u64,
        cumulative_set_order: u64,
        authoring_set_order: u64,
        hash_byte: u8,
    ) -> Field {
        Field {
            section_id,
            key_name: key.to_string(),
            value: FieldValue::External {
                hash: test_hash(hash_byte),
                size: 1024,
                ext_value_index: 0,
            },
            set_order: cumulative_set_order,
            authoring_set_order,
            version,
        }
    }

    /// b.gcb / b.vz7 — `plan_full_content` emits a TOTAL order over state
    /// content, so two replicas holding the same logical snapshot serve the same
    /// plan (and therefore the same base OMPF bytes).
    ///
    /// **Mutation evidence.** Each of the three landed sorts was neutralised in
    /// turn (scratch edit, never committed) and this test observed red:
    ///
    /// | neutralised | first assertion red | message |
    /// |---|---|---|
    /// | `Self::sort_fields_by_authoring(&mut fields)` call removed | whole-plan `assert_eq!` | `plan_full_content must emit an identical plan — including the VECTOR ORDER of sections, fields and time_samples — …` — left/right differ in `fields` order |
    /// | `time_samples.sort_by_key(…)` removed | whole-plan `assert_eq!` | same message; left/right differ in `time_samples` order |
    /// | `orphans.sort_unstable_by_key(…)` removed | whole-plan `assert_eq!` | same message; left/right differ in the orphan tail (`…, 82, 80, 81]` vs `…, 82, 81, 80]`) |
    ///
    /// Each neutralisation was then re-run with the whole-plan `assert_eq!`
    /// temporarily inert, to confirm the per-invariant spot checks name the
    /// regression instead of leaving it an opaque plan inequality:
    ///
    /// | neutralised | spot check red | message |
    /// |---|---|---|
    /// | field sort | (1) | `fields must be strictly ascending on (version, authoring_set_order, section_id, key_name), got [(6, 1, 4, "s4_k"), (7, 0, 3, "later"), (5, 2, 1, "early"), …]` |
    /// | time-sample sort | (3) | `time_samples must be strictly ascending on (version, section_id, time), got [(7, 2, 3.0), (7, 3, 2.0), (7, 2, 1.0), …]` |
    /// | orphan sort | (4) | `orphan tail must be emitted section_id-ascending, got [1, 2, 4, 3, 82, 80, 81]` |
    #[test]
    fn plan_full_content_is_deterministic_for_equal_state() {
        let build = |reverse: bool| {
            let mut state = ObjectState {
                object_id: 1,
                sequence_num: 9,
                ..Default::default()
            };
            // Reachable tree. `children` insertion order held FIXED in both
            // builds (see the section comment above).
            //   root(0) └─1 ├─2 └─3   ; 2 └─4
            for s in [
                make_section(1, 0, 1, 9),
                make_section(2, 1, 1, 9),
                make_section(3, 1, 1, 9),
                make_section(4, 2, 5, 9),
            ] {
                insert_section(&mut state, s);
            }

            // NON-VACUITY (a) — THREE ORPHAN sections, inserted into
            // `state.sections` WITHOUT registering in `children`, so the DFS
            // never reaches them and they land in the orphan tail. Their
            // insertion order (82, 80, 81) is deliberately not ascending, so the
            // emitted tail cannot be ascending by accident. Remove them and
            // `sections_in_dfs_order`'s third `HashMap` walk (engineer seam
            // E-1(ii)) is completely unfenced: an unsorted orphan append would
            // still pass every remaining assertion here.
            let mut ghosts: Vec<Section> = vec![
                make_section(82, 77, 1, 9),
                make_section(80, 77, 1, 9),
                make_section(81, 77, 1, 9),
            ];

            // NON-VACUITY (b) — five fields share the leading `(version,
            // authoring_set_order)` pair `(6, 1)`: three across DIFFERENT
            // sections (2/3/4) and two within ONE section (1) under different
            // `key_name`s. Without the first group the `section_id` component of
            // the 4-tuple is never exercised; without the second the `key_name`
            // tiebreak is never exercised — either way a leading-pair-only sort
            // (or an unstable sort with no total key) would pass. Colliding
            // authoring ranks are the ORDINARY case: `authoring_set_order` is a
            // raw per-delta rank and resets every commit.
            //
            // The cumulative `set_order` cursors are deliberately out of step
            // with the emission order, so a dense re-index cannot hide.
            let mut fields: Vec<Field> = vec![
                make_authored_field(1, "early", 5, 903, 2),
                make_authored_field(1, "aa", 6, 901, 1),
                make_authored_field(1, "bb", 6, 907, 1),
                make_authored_field(2, "s2_k", 6, 900, 1),
                make_authored_field(3, "s3_k", 6, 905, 1),
                make_authored_field(4, "s4_k", 6, 902, 1),
                make_authored_field(3, "later", 7, 904, 0),
            ];

            // NON-VACUITY (c) — three time samples share `(version=7,
            // section_id=2)` and differ only in `time`. Without them the `time`
            // component of the `(version, section_id, time)` key is never
            // exercised and a `(version, section_id)`-only sort would pass.
            let mut samples: Vec<TimeSample> = vec![
                make_time_sample(2, 3.0, 7),
                make_time_sample(4, 0.5, 9),
                make_time_sample(2, 1.0, 7),
                make_time_sample(3, 2.0, 7),
                make_time_sample(2, 2.0, 7),
            ];

            if reverse {
                ghosts.reverse();
                fields.reverse();
                samples.reverse();
            }
            for g in ghosts {
                state.sections.insert(g.section_id, g);
            }
            for f in fields {
                insert_field(&mut state, f);
            }
            for ts in samples {
                insert_time_sample(&mut state, ts);
            }
            state
        };

        // `have_object_id = 0` mismatches `state.object_id = 1` AND
        // `have_version = 0` — the cold-start route into `plan_full_content`.
        let plan_a = DiffEngine::plan(&build(false), &[], 0, 0, None).unwrap();
        let plan_b = DiffEngine::plan(&build(true), &[], 0, 0, None).unwrap();
        assert_eq!(
            plan_a, plan_b,
            "plan_full_content must emit an identical plan — including the VECTOR ORDER of \
             sections, fields and time_samples — for logically identical state, regardless of \
             HashMap iteration order"
        );

        // Repeating on the SAME state is also stable (weaker, but pins that no
        // interior mutability or clock/randomness leaks into the plan).
        let state = build(false);
        assert_eq!(
            DiffEngine::plan(&state, &[], 0, 0, None).unwrap(),
            DiffEngine::plan(&state, &[], 0, 0, None).unwrap(),
            "two calls on one state must agree"
        );

        let (sections, fields, time_samples) = match plan_a {
            DiffPlan::FullContent {
                sections,
                fields,
                time_samples,
            } => (sections, fields, time_samples),
            other => panic!("expected FullContent, got {other:?}"),
        };

        // (1) E-2 — fields STRICTLY ascending on the landed 4-tuple. Strict (not
        // merely non-descending) is the real claim: `(section_id, key_name)` is
        // the `ObjectState::fields` map key, so the 4-tuple is TOTAL and ties are
        // impossible.
        assert_eq!(fields.len(), 7);
        assert!(
            fields.windows(2).all(|w| (
                w[0].version,
                w[0].authoring_set_order,
                w[0].section_id,
                &w[0].key_name
            ) < (
                w[1].version,
                w[1].authoring_set_order,
                w[1].section_id,
                &w[1].key_name
            )),
            "fields must be strictly ascending on (version, authoring_set_order, section_id, \
             key_name), got {:?}",
            fields
                .iter()
                .map(|f| (
                    f.version,
                    f.authoring_set_order,
                    f.section_id,
                    f.key_name.clone()
                ))
                .collect::<Vec<_>>()
        );

        // (2) SORT ONLY — engineer seam E-3 and the Task's 2026-09-01 amendment:
        // `plan_full_content` reorders fields but PRESERVES each field's
        // cumulative `set_order` verbatim. It must not have grown a
        // `dense_reindex_by_authoring` call, which would rewrite the legacy
        // catch-up cold-start wire's `set_order` to the dense emit index.
        for f in &fields {
            let stored = &state.fields[&(f.section_id, f.key_name.clone())];
            assert_eq!(
                f.set_order, stored.set_order,
                "field ({}, {}) must carry its stored cumulative set_order verbatim",
                f.section_id, f.key_name
            );
        }
        // Explicit form of the same claim: the emitted cursors follow the
        // EMISSION order and are neither sorted nor dense.
        assert_eq!(
            fields.iter().map(|f| f.set_order).collect::<Vec<u64>>(),
            vec![903, 901, 907, 900, 905, 902, 904],
            "the cumulative cursor must survive the sort verbatim — a dense re-index would emit \
             0..6 here"
        );

        // (3) E-4 — time samples strictly ascending on (version, section_id,
        // time). Total for the same reason: `(section_id, time)` is the
        // `ObjectState::time_samples` map key.
        assert_eq!(time_samples.len(), 5);
        assert!(
            time_samples
                .windows(2)
                .all(|w| (w[0].version, w[0].section_id, w[0].time)
                    < (w[1].version, w[1].section_id, w[1].time)),
            "time_samples must be strictly ascending on (version, section_id, time), got {:?}",
            time_samples
                .iter()
                .map(|ts| (ts.version, ts.section_id, ts.time))
                .collect::<Vec<_>>()
        );

        // (4) E-1(ii) / E-5 — the ORPHAN TAIL gains an order; the reachable DFS
        // prefix MUST NOT move, so nothing NEW is asserted about it beyond the
        // parent-before-child shape the existing DFS coverage already carries.
        let ids = section_ids(&sections);
        assert_eq!(ids.len(), 7, "4 reachable + 3 orphans");
        let (prefix, tail) = ids.split_at(4);
        assert_eq!(
            tail,
            [80, 81, 82],
            "orphan tail must be emitted section_id-ascending, got {ids:?}"
        );
        let mut prefix_sorted = prefix.to_vec();
        prefix_sorted.sort_unstable();
        assert_eq!(
            prefix_sorted,
            vec![1, 2, 3, 4],
            "every reachable section must precede every orphan, got {ids:?}"
        );
        let pos = |id: u64| prefix.iter().position(|&x| x == id).unwrap();
        assert!(
            pos(1) < pos(2) && pos(1) < pos(3) && pos(2) < pos(4),
            "reachable prefix must stay parent-before-child, got {prefix:?}"
        );
    }

    /// b.gcb / b.vz7 — the ext-refs sidecar's first-appearance hash order is
    /// deterministic at its DOMAIN source, for the same state that produced a
    /// deterministic plan.
    ///
    /// The hashes are compared as VECTORS, never as sets: a set comparison is
    /// exactly the weak observable b.vz7 complains about downstream, and it
    /// would be satisfied by every permutation the pre-sort code emitted.
    ///
    /// **Mutation evidence** (scratch neutralisations, never committed):
    ///
    /// | neutralised | first assertion red | message |
    /// |---|---|---|
    /// | `Self::sort_fields_by_authoring(&mut fields)` call removed | cross-build `ordered_hashes` vector equality | `the sidecar's ordered hashes must be equal as VECTORS across two independently-built equal states — …` (`[A2, A4, A3, A1, B1, B2]` vs `[A3, A4, A2, A1, B1, B2]`) |
    /// | `time_samples.sort_by_key(…)` removed | the same assertion | same message; the `B1`/`B2` suffix permutes |
    /// | `orphans.sort_unstable_by_key(…)` removed | — (green) | this fixture has no orphan sections; the orphan tail is `plan_full_content_is_deterministic_for_equal_state`'s to fence |
    ///
    /// With the cross-build equality temporarily inert, both of the first two
    /// neutralisations turned the explicit expected-order assertion red instead
    /// (`the sidecar's first-appearance hash order must follow the emitted field
    /// order, then the emitted time-sample order`) — though only *flakily* for
    /// the time-sample row, since a single state's `HashMap` can happen to
    /// iterate four samples in the right order. That is precisely why the
    /// cross-build vector comparison is the primary gate here and the explicit
    /// order is the naming diagnostic behind it.
    ///
    /// The time-sample row is the whole reason the `0xB1`/`0xB2` half of the
    /// fixture exists: sorting fields alone leaves the sidecar
    /// non-deterministic (engineer seam E-1(i)).
    #[test]
    fn plan_full_content_ext_hash_order_is_deterministic_for_equal_state() {
        let build = |reverse: bool| {
            let mut state = ObjectState {
                object_id: 1,
                sequence_num: 9,
                ..Default::default()
            };
            for s in [
                make_section(1, 0, 1, 9),
                make_section(2, 1, 1, 9),
                make_section(3, 1, 1, 9),
            ] {
                insert_section(&mut state, s);
            }

            // NON-VACUITY (a) — four ext-referencing fields TIE on the leading
            // `(version, authoring_set_order)` pair `(6, 1)`, two of them within
            // one section. Without the tie the sidecar order would be pinned by
            // the leading pair alone and the `(section_id, key_name)` suffix
            // would never be exercised.
            // `e_dup` re-uses `0xA1` so the dedup path is covered too — a second
            // appearance must NOT append a second entry.
            let mut fields: Vec<Field> = vec![
                make_authored_ext_field(2, "e_a", 6, 900, 1, 0xA1),
                make_authored_ext_field(3, "e_b", 6, 901, 1, 0xA2),
                make_authored_ext_field(1, "e_c", 6, 902, 1, 0xA3),
                make_authored_ext_field(1, "e_d", 6, 903, 1, 0xA4),
                make_authored_ext_field(3, "e_dup", 7, 904, 0, 0xA1),
                // An inline field, so the scan's "skip non-external" branch is
                // exercised inside the sorted sequence rather than around it.
                make_authored_field(2, "inline", 5, 899, 0),
            ];

            // NON-VACUITY (b) — `0xB1` and `0xB2` appear FIRST on a TIME SAMPLE
            // and on NO field. This is what makes the sidecar's order depend on
            // the time-sample ordering: `collect_external_value_hashes` scans
            // fields THEN time samples into ONE shared first-appearance pool
            // (seam E-1(i)), so with fields sorted but time samples unsorted the
            // sidecar is still non-deterministic. Remove these two and the
            // assertion below degrades to a fields-only claim and the
            // time-sample sort goes untested here.
            // `0xA2` reappears on a sample already seen on a field → no new entry.
            let mut samples: Vec<TimeSample> = vec![
                make_ext_time_sample(2, 3.0, 7, 0xB2),
                make_ext_time_sample(3, 2.0, 8, 0xA2),
                make_ext_time_sample(2, 1.0, 7, 0xB1),
                make_time_sample(1, 0.5, 6),
            ];

            if reverse {
                fields.reverse();
                samples.reverse();
            }
            for f in fields {
                insert_field(&mut state, f);
            }
            for ts in samples {
                insert_time_sample(&mut state, ts);
            }
            state
        };

        let plan_a = DiffEngine::plan(&build(false), &[], 0, 0, None).unwrap();
        let plan_b = DiffEngine::plan(&build(true), &[], 0, 0, None).unwrap();

        let unwrap = |plan: DiffPlan| match plan {
            DiffPlan::FullContent {
                fields,
                time_samples,
                ..
            } => (fields, time_samples),
            other => panic!("expected FullContent, got {other:?}"),
        };
        let (fields_a, samples_a) = unwrap(plan_a);
        let (fields_b, samples_b) = unwrap(plan_b);

        let scan_a = collect_external_value_hashes(&fields_a, &samples_a);
        let scan_b = collect_external_value_hashes(&fields_b, &samples_b);

        assert_eq!(
            scan_a.ordered_hashes, scan_b.ordered_hashes,
            "the sidecar's ordered hashes must be equal as VECTORS across two independently-built \
             equal states — a set comparison here would be satisfied by every permutation the \
             pre-sort emitter produced"
        );

        // The concrete order the total order pins:
        //   fields   (version, authoring, section, key): (5,0,2,inline) inline,
        //            (6,1,1,e_c)=A3, (6,1,1,e_d)=A4, (6,1,2,e_a)=A1,
        //            (6,1,3,e_b)=A2, (7,0,3,e_dup)=A1(dup)
        //   samples  (version, section, time): (6,1,0.5) inline,
        //            (7,2,1.0)=B1, (7,2,3.0)=B2, (8,3,2.0)=A2(dup)
        assert_eq!(
            scan_a.ordered_hashes,
            vec![
                test_hash(0xA3),
                test_hash(0xA4),
                test_hash(0xA1),
                test_hash(0xA2),
                test_hash(0xB1),
                test_hash(0xB2),
            ],
            "the sidecar's first-appearance hash order must follow the emitted field order, then \
             the emitted time-sample order"
        );

        // The 1-based OMPF part indices ride the same vector — pinned here so a
        // reordering that somehow preserved `ordered_hashes` could not slip the
        // index map.
        assert_eq!(scan_a.hash_to_index[&test_hash(0xA3)], 1);
        assert_eq!(scan_a.hash_to_index[&test_hash(0xB2)], 6);
        assert_eq!(
            scan_a.hash_to_index, scan_b.hash_to_index,
            "hash → part-index map must agree across the two builds"
        );
    }

    // ═════════════════════════════════════════════════════════════════════════
    // b.ze1 — the faithful walk and the coalesced restatement.
    //
    // The walk reaches a section only from its own parent's list, numbers the
    // true children of each list 0, 1, 2… (a ghost, a stale listing or a
    // repeat takes no position) and emits every section after its parent. The
    // coalesced diff restates every listed child of a parent with a new or
    // changed child, at its position. Whether the restatement rebuilds the
    // server's order under each client's reading is checked by the reader
    // tests in `src/domain/delta.rs` (`ClientModel`); these pin the plans.
    // ═════════════════════════════════════════════════════════════════════════

    /// `SdfSpecType` values used below.
    const ZE1_PRIM: u8 = 6;
    const ZE1_ATTRIBUTE: u8 = 1;

    fn ze1_state(sequence_num: u64) -> ObjectState {
        ObjectState {
            object_id: 1,
            sequence_num,
            ..Default::default()
        }
    }

    /// A row only, listed nowhere: the caller writes every list itself.
    fn ze1_row(state: &mut ObjectState, id: u64, parent: u64, first: u64, last: u64) {
        state
            .sections
            .insert(id, make_section(id, parent, first, last));
    }

    /// `(section_id, emitted section_order)` in emission order.
    fn ze1_emitted(sections: &[Section]) -> Vec<(u64, u32)> {
        sections
            .iter()
            .map(|s| (s.section_id, s.section_order))
            .collect()
    }

    fn ze1_snapshot(state: &ObjectState) -> Vec<Section> {
        match DiffEngine::plan_faithful_snapshot(state).expect("the snapshot plans") {
            DiffPlan::FullContent { sections, .. } => sections,
            other => panic!("expected FullContent, got {other:?}"),
        }
    }

    /// `(new_sections, modified_sections, deleted_sections)`.
    type Ze1Coalesced = (Vec<Section>, Vec<Section>, Vec<(u64, u64, u64)>);

    fn ze1_coalesced(
        state: &ObjectState,
        tombstones: &[(u64, u64, u64)],
        have_version: u64,
    ) -> Ze1Coalesced {
        match DiffEngine::plan_faithful_coalesced(state, tombstones, have_version)
            .expect("the coalesced diff plans")
        {
            DiffPlan::Incremental {
                new_sections,
                modified_sections,
                deleted_sections,
                ..
            } => (new_sections, modified_sections, deleted_sections),
            other => panic!("expected Incremental, got {other:?}"),
        }
    }

    // ── (iv) The walk over legacy stored shapes ──────────────────────────────

    /// A ghost entry — an id with no row — in P's list `[A, G, C]` takes no
    /// position: C is P's second true child and is emitted at 1, not at its
    /// list index 2.
    #[test]
    fn faithful_snapshot_ghost_entry_takes_no_position() {
        const P: u64 = 1;
        const A: u64 = 10;
        const C: u64 = 12;
        const G: u64 = 99;
        let mut state = ze1_state(2);
        ze1_row(&mut state, P, 0, 1, 1);
        ze1_row(&mut state, A, P, 1, 1);
        ze1_row(&mut state, C, P, 1, 1);
        state.children.insert(0, vec![P]);
        state.children.insert(P, vec![A, G, C]);

        assert_eq!(
            ze1_emitted(&ze1_snapshot(&state)),
            vec![(P, 0), (A, 0), (C, 1)],
            "the ghost is not emitted and shifts no later sibling"
        );
    }

    /// A stale double listing: S's row names P, and Q's list (walked first)
    /// also names S. S is emitted once, after P, at its position in P's list
    /// (1); in Q's list it takes no position, so Y is Q's second child (1).
    #[test]
    fn faithful_snapshot_stale_double_listing_is_emitted_under_its_true_parent_after_it() {
        const P: u64 = 1;
        const Q: u64 = 2;
        const A: u64 = 10;
        const X: u64 = 20;
        const Y: u64 = 21;
        const S: u64 = 30;
        let mut state = ze1_state(2);
        for (id, parent) in [(P, 0), (Q, 0), (A, P), (X, Q), (Y, Q), (S, P)] {
            ze1_row(&mut state, id, parent, 1, 1);
        }
        state.children.insert(0, vec![Q, P]);
        state.children.insert(Q, vec![X, S, Y]);
        state.children.insert(P, vec![A, S]);

        assert_eq!(
            ze1_emitted(&ze1_snapshot(&state)),
            vec![(Q, 0), (X, 0), (Y, 1), (P, 1), (A, 0), (S, 1)],
            "S comes after its parent P with P's index; Q's list numbers only its true children"
        );
    }

    /// A repeat in one list, `[A, B, A]`, keeps the first occurrence's
    /// position: A at 0, B at 1, and A is emitted once.
    #[test]
    fn faithful_snapshot_repeated_entry_keeps_its_first_position() {
        const P: u64 = 1;
        const A: u64 = 10;
        const B: u64 = 11;
        let mut state = ze1_state(2);
        ze1_row(&mut state, P, 0, 1, 1);
        ze1_row(&mut state, A, P, 1, 1);
        ze1_row(&mut state, B, P, 1, 1);
        state.children.insert(0, vec![P]);
        state.children.insert(P, vec![A, B, A]);

        assert_eq!(
            ze1_emitted(&ze1_snapshot(&state)),
            vec![(P, 0), (A, 0), (B, 1)]
        );
    }

    /// A ghost id that carries a list of its own is not descended through: Z,
    /// whose row names the ghost G as its parent, is unreachable, so the
    /// snapshot drops it and a reconnector that could hold it is told to
    /// retract it.
    #[test]
    fn faithful_builders_do_not_descend_through_a_ghost_entry() {
        const P: u64 = 1;
        const A: u64 = 10;
        const G: u64 = 99;
        const Z: u64 = 50;
        let mut state = ze1_state(3);
        ze1_row(&mut state, P, 0, 1, 1);
        ze1_row(&mut state, A, P, 1, 1);
        ze1_row(&mut state, Z, G, 1, 1);
        state.children.insert(0, vec![P]);
        state.children.insert(P, vec![A, G]);
        state.children.insert(G, vec![Z]);

        assert_eq!(
            ze1_emitted(&ze1_snapshot(&state)),
            vec![(P, 0), (A, 0)],
            "the snapshot does not reach Z through the ghost"
        );
        let (new, modified, deleted) = ze1_coalesced(&state, &[], 2);
        assert!(new.is_empty() && modified.is_empty());
        assert_eq!(deleted, vec![(Z, 1, 1)], "Z is retracted (arm b)");
    }

    /// A section stored before b.xje that is listed only under a parent other
    /// than its own is not live to the faithful builders: S (row parent P,
    /// listed only in Q's list) is dropped from a snapshot and retracted from
    /// a reconnector that could hold it, and Q's list numbers only its true
    /// children.
    #[test]
    fn faithful_builders_drop_a_section_listed_only_under_a_foreign_parent() {
        const P: u64 = 1;
        const Q: u64 = 2;
        const A: u64 = 10;
        const X: u64 = 20;
        const Y: u64 = 21;
        const S: u64 = 30;
        let mut state = ze1_state(3);
        for (id, parent) in [(P, 0), (Q, 0), (A, P), (X, Q), (Y, Q), (S, P)] {
            ze1_row(&mut state, id, parent, 1, 1);
        }
        state.children.insert(0, vec![P, Q]);
        state.children.insert(P, vec![A]);
        state.children.insert(Q, vec![X, S, Y]);

        assert_eq!(
            ze1_emitted(&ze1_snapshot(&state)),
            vec![(P, 0), (A, 0), (Q, 1), (X, 0), (Y, 1)],
            "S is not emitted, and Y is Q's second true child"
        );
        let (new, modified, deleted) = ze1_coalesced(&state, &[], 2);
        assert!(new.is_empty() && modified.is_empty());
        assert_eq!(deleted, vec![(S, 1, 1)], "S is retracted (arm b)");
    }

    /// Both faithful plans are a function of the stored content alone. Each
    /// state below holds the same stale double listing (S's row names P, and
    /// Q's list names S too), built in a fresh `HashMap` with its own random
    /// seed and with the lists inserted in a varying order; every plan must
    /// be identical, with S at P's position.
    #[test]
    fn faithful_plans_are_independent_of_children_map_order_on_a_stale_listing() {
        const P: u64 = 1;
        const Q: u64 = 2;
        const A: u64 = 10;
        const X: u64 = 20;
        const S: u64 = 30;
        const BUILDS: usize = 32;
        let build = |rotation: usize| {
            let mut state = ze1_state(4);
            for (id, parent, last) in [(P, 0, 1), (Q, 0, 1), (A, P, 1), (X, Q, 1), (S, P, 4)] {
                ze1_row(&mut state, id, parent, 1, last);
            }
            let mut lists = vec![(0, vec![P, Q]), (P, vec![A, S]), (Q, vec![S, X])];
            lists.rotate_left(rotation % 3);
            for (parent, list) in lists {
                state.children.insert(parent, list);
            }
            state
        };

        let snapshots: Vec<Vec<(u64, u32)>> = (0..BUILDS)
            .map(|i| ze1_emitted(&ze1_snapshot(&build(i))))
            .collect();
        assert!(
            snapshots.iter().all(|s| *s == snapshots[0]),
            "every snapshot must be identical: {snapshots:?}"
        );
        assert_eq!(
            snapshots[0],
            vec![(P, 0), (A, 0), (S, 1), (Q, 1), (X, 0)],
            "S at its position in P's list, and X first among Q's true children"
        );

        let coalesced: Vec<Ze1Coalesced> = (0..BUILDS)
            .map(|i| ze1_coalesced(&build(i), &[], 3))
            .collect();
        assert!(
            coalesced.iter().all(|c| *c == coalesced[0]),
            "every coalesced plan must be identical: {coalesced:?}"
        );
        assert_eq!(
            ze1_emitted(&coalesced[0].1),
            vec![(A, 0), (S, 1)],
            "S changed, so P's whole sequence is restated at P's positions"
        );
    }

    // ── (i) The coalesced restatement ────────────────────────────────────────

    /// The ticket's repro: P's children were A, B, C, X; B is deleted and D
    /// created while the reconnector is away. Every listed child of P is
    /// restated at its position in `[A, C, X, D]` — D new at 3, A, C and X at
    /// 0, 1 and 2 — and B is retracted.
    #[test]
    fn faithful_coalesced_restates_every_listed_child_of_a_parent_with_a_new_child() {
        const P: u64 = 1;
        const A: u64 = 10;
        const B: u64 = 11;
        const C: u64 = 12;
        const X: u64 = 13;
        const D: u64 = 14;
        let mut state = ze1_state(3);
        ze1_row(&mut state, P, 0, 1, 1);
        for id in [A, C, X] {
            ze1_row(&mut state, id, P, 1, 1);
        }
        ze1_row(&mut state, D, P, 3, 3);
        state.children.insert(0, vec![P]);
        state.children.insert(P, vec![A, C, X, D]);
        let tombstones = [(B, 1, 2)];

        let (new, modified, deleted) = ze1_coalesced(&state, &tombstones, 1);
        assert_eq!(ze1_emitted(&new), vec![(D, 3)], "D is new, at its position");
        assert_eq!(
            ze1_emitted(&modified),
            vec![(A, 0), (C, 1), (X, 2)],
            "P's unchanged children are restated, in list order, at their positions"
        );
        assert_eq!(deleted, vec![(B, 1, 2)], "B is retracted");
    }

    /// Positions count every child kind together: under a prim P holding prims
    /// A and B and an attribute x (`[A, x, B]`), a created prim C is new at 3
    /// and the attribute is restated with the prims, at 1.
    #[test]
    fn faithful_coalesced_restatement_spans_every_child_list_of_the_parent() {
        const P: u64 = 1;
        const A: u64 = 10;
        const B: u64 = 11;
        const C: u64 = 12;
        const X_ATTR: u64 = 20;
        let mut state = ze1_state(3);
        let mut typed = |id: u64, parent: u64, ty: u8, first: u64| {
            let mut s = make_section(id, parent, first, first);
            s.section_type = ty;
            state.sections.insert(id, s);
        };
        typed(P, 0, ZE1_PRIM, 1);
        typed(A, P, ZE1_PRIM, 1);
        typed(X_ATTR, P, ZE1_ATTRIBUTE, 1);
        typed(B, P, ZE1_PRIM, 1);
        typed(C, P, ZE1_PRIM, 3);
        state.children.insert(0, vec![P]);
        state.children.insert(P, vec![A, X_ATTR, B, C]);

        let (new, modified, deleted) = ze1_coalesced(&state, &[], 2);
        assert_eq!(ze1_emitted(&new), vec![(C, 3)]);
        assert_eq!(
            ze1_emitted(&modified),
            vec![(A, 0), (X_ATTR, 1), (B, 2)],
            "every listed child of P is restated, the attribute included"
        );
        assert!(deleted.is_empty());
    }

    /// A parent whose children did not change adds nothing: a field-only edit
    /// restates no section, a parent that only lost a child restates none of
    /// the others, and of a move only the destination's children are restated,
    /// not the source's.
    #[test]
    fn faithful_coalesced_untouched_parents_add_nothing() {
        const P: u64 = 1;
        const Q: u64 = 2;
        const A: u64 = 10;
        const B: u64 = 11;
        const C: u64 = 12;
        const X: u64 = 20;
        let base = || {
            let mut state = ze1_state(3);
            for (id, parent) in [(P, 0), (Q, 0), (A, P), (C, P), (X, Q)] {
                ze1_row(&mut state, id, parent, 1, 1);
            }
            state.children.insert(0, vec![P, Q]);
            state.children.insert(P, vec![A, C]);
            state.children.insert(Q, vec![X]);
            state
        };

        // A field-only edit on A.
        let mut state = base();
        insert_field(&mut state, make_field(A, "f", 3));
        let (new, modified, deleted) = ze1_coalesced(&state, &[], 2);
        assert!(
            new.is_empty() && modified.is_empty() && deleted.is_empty(),
            "field-only: no section is emitted, got new {new:?} modified {modified:?}"
        );

        // P only lost B, deleted after the cut.
        let state = base();
        let (new, modified, deleted) = ze1_coalesced(&state, &[(B, 1, 3)], 2);
        assert!(
            new.is_empty() && modified.is_empty(),
            "delete-only: P's remaining children are not restated, got {modified:?}"
        );
        assert_eq!(deleted, vec![(B, 1, 3)]);

        // B moved from P to Q after the cut: Q is touched, P is not.
        let mut state = base();
        ze1_row(&mut state, B, Q, 1, 3);
        state.children.insert(Q, vec![X, B]);
        let (new, modified, deleted) = ze1_coalesced(&state, &[], 2);
        assert!(new.is_empty() && deleted.is_empty());
        assert_eq!(
            ze1_emitted(&modified),
            vec![(X, 0), (B, 1)],
            "move-out: only the destination's children are restated; A and C are not"
        );
    }

    /// Over a history touching several parents at several depths — a new
    /// parent with new children under a parent with old ones, a changed child,
    /// a delete, a detached subtree — every section is emitted at most once,
    /// and no emitted section is also retracted.
    #[test]
    fn faithful_coalesced_emits_each_section_once_and_never_retracts_an_emitted_one() {
        const HAVE: u64 = 2;
        let mut state = ze1_state(5);
        // root → R; R → [A (old), N (new), C (changed)]; N → [N1, N2] (new);
        // C → [C1 (old)]; detached D (old) under the missing 77.
        ze1_row(&mut state, 1, 0, 1, 1);
        ze1_row(&mut state, 10, 1, 1, 1);
        ze1_row(&mut state, 11, 1, 4, 4);
        ze1_row(&mut state, 12, 1, 1, 5);
        ze1_row(&mut state, 20, 11, 4, 4);
        ze1_row(&mut state, 21, 11, 4, 4);
        ze1_row(&mut state, 30, 12, 1, 1);
        ze1_row(&mut state, 40, 77, 1, 3);
        state.children.insert(0, vec![1]);
        state.children.insert(1, vec![10, 11, 12]);
        state.children.insert(11, vec![20, 21]);
        state.children.insert(12, vec![30]);
        let tombstones = [(13, 1, 3), (14, 3, 4)];

        let (new, modified, deleted) = ze1_coalesced(&state, &tombstones, HAVE);
        let emitted: Vec<u64> = new.iter().chain(&modified).map(|s| s.section_id).collect();
        let unique: HashSet<u64> = emitted.iter().copied().collect();
        assert_eq!(
            unique.len(),
            emitted.len(),
            "no section is emitted twice: {emitted:?}"
        );
        let retracted: HashSet<u64> = deleted.iter().map(|(id, _, _)| *id).collect();
        assert!(
            unique.is_disjoint(&retracted),
            "no emitted section is retracted: emitted {emitted:?}, retracted {retracted:?}"
        );
        // The shape itself, so the checks above do not pass over an empty set.
        assert_eq!(section_ids(&new), vec![11, 20, 21]);
        assert_eq!(
            section_ids(&modified),
            vec![10, 12],
            "R's old children are restated; C's child C1 is not (C itself only changed)"
        );
        assert_eq!(
            deleted.iter().map(|(id, _, _)| *id).collect::<Vec<u64>>(),
            vec![13, 40]
        );
    }
}
