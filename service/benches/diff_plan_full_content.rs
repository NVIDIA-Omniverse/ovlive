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

//! Criterion microbench for the **full-content diff plan** —
//! `DiffEngine::plan(&state, &[], 0, 0, None)`, i.e. the `plan_full_content`
//! branch — and, isolated beside it, the cost of the total-order field sort
//! that branch acquired for SR-9.2 byte-deterministic base emission
//! (b.gcb / b.vz7).
//!
//! ## Why this file exists
//!
//! The Task that added the sort carries the guard "no bench regression on
//! the assembly path". When it was written there was **no instrument on
//! that path at all**: `benches/` held only `ext_mirror_enqueue` and
//! `ext_mirror_store_values`, both SR-3.1 ext-mirror, neither reaching
//! `DiffEngine::plan`. "Run the existing bench" was not executable, so the
//! guard is discharged by creating the instrument.
//!
//! A true before/after criterion diff is also **not obtainable in-lane**:
//! the pre-sort code is already gone from the branch, and capturing a
//! baseline would mean checking out a pre-sort commit — shared-branch git
//! manipulation this work must not do. The three arms below get the same
//! answer without it: measure the whole call, measure the sort alone on the
//! same generated state, and read the two against a builder that already
//! performs the same class of sort on a hotter path.
//!
//! ## The three arms
//!
//! Swept over three field counts (~100 / ~1_000 / ~10_000) of the SAME
//! generated [`ObjectState`], built once per size outside every measured
//! closure so allocation is never folded into the number:
//!
//! - `plan_full_content/<n>` — the whole `DiffEngine::plan(&state, &[], 0,
//!   0, None)` call. `have_object_id = 0` never matches the generated
//!   `object_id`, so this is unconditionally the `plan_full_content`
//!   branch: DFS section walk + orphan-tail sort, field clone + field
//!   sort, time-sample clone + sort.
//! - `sort_only/<n>` — a pre-cloned field `Vec` in `HashMap` order, sorted
//!   with the same comparator and nothing else. This is the delta the sort
//!   introduced, isolated: everything else in `plan_full_content` (the
//!   clone out of the map, the DFS walk, the time-sample pass) predates it.
//!   The clone happens in `iter_batched`'s untimed setup, so the timed
//!   region is the comparator and only the comparator.
//! - `plan_faithful_snapshot/<n>` — the comparative anchor.
//!   [`DiffEngine::plan_faithful_snapshot`] already performs the same class
//!   of sort (it calls the identical private comparator) **plus** a dense
//!   `set_order` reindex and a sibling-index build, and it sits on the
//!   catch-up path, which is far hotter than the checkpoint path this Task
//!   touched. If `plan_full_content` lands inside that envelope, the added
//!   cost is inside one this repo already accepts.
//!
//! `sort_only / plan_full_content` is therefore the added fraction,
//! measured rather than argued.
//!
//! ## The comparator is replicated here, deliberately
//!
//! `DiffEngine::sort_fields_by_authoring` is a **private** helper.
//! [`sort_fields_like_diff_engine`] below mirrors its key
//! `(version, authoring_set_order, section_id)` then `key_name`, via the
//! same `sort_unstable_by` shape. Making the production helper `pub` purely
//! to bench it would widen the domain crate's API surface for a measurement,
//! so the copy is the deliberate trade. If the production key ever changes,
//! this arm silently starts measuring the wrong comparator — the `sort_only`
//! number would then simply stop being a valid decomposition of
//! `plan_full_content`, which the two arms sitting side by side in one report
//! makes visible (a `sort_only` that no longer tracks `plan_full_content`'s
//! shape is the tell). The authoritative implementation is
//! `src/domain/diff.rs::DiffEngine::sort_fields_by_authoring`.
//!
//! ## Generated state, not a checked-in fixture
//!
//! Per the test-writing guide's "generate rather than check in binary
//! fixtures" rule (`docs-internal/test-writing-guide.md` §
//! Performance-Sensitive Tests). Nothing here touches Valkey, so no keys are
//! minted and the bounded-keyspace half of that rule does not bite; the
//! state is bounded anyway (the largest arm is about 10 % of `MAX_DELTA_ENTRIES`).
//!
//! The generator's shape is load-bearing — see [`generate_object_state`].
//! Fields deliberately carry the **tie-heavy** `(version,
//! authoring_set_order)` distribution real authored data has: a fixture
//! with all-distinct leading keys would let every comparison resolve on the
//! first `u64` and would understate the comparator's work by skipping the
//! `section_id` and `key_name` fall-throughs entirely.
//!
//! ## Report-only: nothing gates on this
//!
//! This is a **LOCAL same-machine instrument**, the posture the guide's
//! "The precise instrument is local" bullet describes. It is deliberately
//! NOT wired into CI: `.gitlab-ci.yml`'s `perf:ext-mirror-bench` job is not
//! extended, and `tools/check_bench_thresholds.py` gates exactly one named
//! id (`ext_mirror_enqueue/mirror_drained/4096`) — an id added here would be
//! reported informationally and could never fail a pipeline, so adding one
//! would be a no-op dressed as a gate. **No CI ceiling gates any id in this
//! file.** There is likewise no wall-clock assertion anywhere in the repo's
//! gating suite for this path; `cargo test` / `cargo nextest run` select no
//! bench target, so the release gate's runtime is untouched, while
//! `cargo clippy --all-targets` and `cargo test --benches` do build this
//! file, which is what keeps it from bit-rotting.
//!
//! The comparison to trust is criterion's own `change: +X%` line against the
//! previous run of the same ids on the same machine, stored in
//! `target/criterion/`.
//!
//! **Read that line across at least two runs before believing it.** The two
//! `plan_*` arms allocate heavily inside the timed region (a `Vec<Field>`
//! per call, a `String` and a payload `Vec<u8>` per field), so on a shared
//! container with other builds running they are sensitive to allocator and
//! page-cache state in a way `sort_only` — allocation-free once its input is
//! cloned in untimed setup — is not. During this bench's own commissioning,
//! one run in four came in ~42 % low on all three `plan_full_content` sizes
//! while `sort_only` moved under 3 %; the other three agreed within a few
//! percent. A single `Performance has improved.` on a `plan_*` id is
//! therefore not evidence on its own. Re-run until two consecutive runs
//! agree, and treat a change that moves `plan_full_content` without moving
//! `sort_only` proportionally as suspect.
//!
//! ## Running it
//!
//! ```text
//! cargo bench -j 4 -p live-edit --bench diff_plan_full_content
//! cargo bench -j 4 -p live-edit --bench diff_plan_full_content -- --output-format bencher
//! ```
//!
//! No `required-features` and no Valkey: `DiffEngine::plan`,
//! `DiffEngine::plan_faithful_snapshot` and `DiffPlan` are all `pub` and the
//! measured code is pure `domain` logic. The target runs bare exactly as
//! above.
//!
//! Criterion's windows are shortened from the defaults (see the
//! `criterion_group!` config at the bottom): the 10_000-field arms cost
//! milliseconds per iteration, and the stock 3 s warm-up + 5 s measurement
//! across nine ids pushes the sweep past the ten-minute ceiling the agent
//! harness allows for one command. The shortened windows still collect
//! criterion's full sample count at every size.

use std::collections::HashMap;
use std::hint::black_box;
use std::time::Duration;

use criterion::{criterion_group, criterion_main, BatchSize, BenchmarkId, Criterion, Throughput};
use ordered_float::OrderedFloat;

use live_edit::domain::diff::{DiffEngine, DiffPlan};
use live_edit::domain::types::{Field, FieldValue, ObjectState, Section, TimeSample};

/// Field counts swept by every arm.
///
/// 100 is a small authored layer where per-call fixed costs (the DFS walk,
/// the `Vec` allocations) still dominate; 10_000 is about 10 % of
/// `domain::limits::MAX_DELTA_ENTRIES` and is where an `O(n log n)` term with a
/// `String` comparison in it becomes the whole signal. 1_000 brackets them
/// so the trend between the two is visible rather than inferred.
const FIELD_COUNTS: [usize; 3] = [100, 1_000, 10_000];

/// Fields written per simulated commit — i.e. how many fields share one
/// `version`. Chosen to give the 10_000-field arm ~156 distinct versions,
/// the shape a layer edited over a session actually has. A single version
/// for all fields would collapse the leading sort key to a constant; one
/// version per field would make it a unique key and skip every
/// fall-through.
const FIELDS_PER_VERSION: usize = 64;

/// How many fields share one `(version, authoring_set_order)` pair.
///
/// The tie-heaviness knob. `authoring_set_order` is the writer's raw
/// per-delta authoring rank, which real non-conforming writers reuse within
/// one delta, so ties on the leading two keys are the common case rather
/// than the exception. At 4, every comparison inside a tie group falls
/// through to `section_id`, and — because [`generate_object_state`] puts
/// consecutive field pairs on the SAME section — half of those fall through
/// again to the `key_name` `String` comparison. That is the expensive path
/// the comparator actually walks in production.
const FIELDS_PER_AUTHORING_RANK: usize = 4;

/// Fields per section. Keeps the section count proportional to the field
/// count so the DFS walk in `plan_full_content` scales with the arm rather
/// than staying a fixed cost that the largest size amortizes away.
const FIELDS_PER_SECTION: usize = 5;

/// Unreachable (orphan) sections added per arm, as a fraction of the
/// reachable section count.
///
/// Present because the orphan tail is one of the two sequences the Task's
/// sort covers (`sections_in_dfs_order` sorts orphans on `section_id`;
/// before the change they were appended in `HashMap` order). Orphans carry
/// **no fields and no time samples** — deliberately, because
/// `plan_faithful_snapshot` drops unreachable prims' fields, and giving
/// orphans fields would hand the anchor arm a smaller field set than
/// `plan_full_content` sorts and quietly invalidate the comparison.
const ORPHAN_SECTION_DIVISOR: usize = 20;

/// Time samples per arm, as a fraction of the field count. Non-zero because
/// the time-sample sort `(version, section_id, time)` is part of the same
/// change and feeds the shared ext-hash pool; small because authored layers
/// carry far fewer samples than fields.
const TIME_SAMPLES_DIVISOR: usize = 10;

/// Inline field payload size in bytes. Large enough that the per-field
/// `Vec<u8>` clone `plan_full_content` performs is a real allocation rather
/// than a rounding error, small enough that the 10_000-field arm holds well
/// under a megabyte of payload.
const INLINE_VALUE_BYTES: usize = 48;

/// Every Nth field is a `FieldValue::External` reference instead of inline
/// bytes, matching the mixed shape a layer with large values has.
const EXTERNAL_VALUE_STRIDE: usize = 8;

/// `object_id` of the generated state. Any non-zero value works: the
/// `plan_full_content` arm passes `have_object_id = 0`, which cannot match,
/// so `DiffEngine::plan` takes the full-content branch unconditionally.
const OBJECT_ID: u64 = 42;

/// A deterministic field key name.
///
/// The long shared prefix is intentional: real USD attribute names collide
/// for many bytes before diverging, so a `String` comparison in the
/// comparator's last fall-through is not resolved by the first character.
/// A short unique name would make that fall-through nearly free and
/// understate the sort.
fn key_name_for(index: usize) -> String {
    format!("xformOp:transform:attribute_{index:06}")
}

/// A deterministic field value: inline bytes for most fields, an external
/// hash reference for every [`EXTERNAL_VALUE_STRIDE`]th one.
fn field_value_for(index: usize) -> FieldValue {
    if index.is_multiple_of(EXTERNAL_VALUE_STRIDE) {
        let mut hash = [0u8; 32];
        for (slot, byte) in hash.iter_mut().enumerate() {
            *byte = ((index + slot) % 251) as u8;
        }
        FieldValue::External {
            hash,
            size: 1024 + index as u64,
            ext_value_index: (index / EXTERNAL_VALUE_STRIDE) as u32,
        }
    } else {
        FieldValue::Inline(
            (0..INLINE_VALUE_BYTES)
                .map(|b| ((index + b) % 251) as u8)
                .collect(),
        )
    }
}

/// Build an [`ObjectState`] carrying `field_count` fields.
///
/// Shape, and why each part of it is what it is:
///
/// - **Sections** — a two-level reachable tree (root → group → leaf), so
///   the DFS walk has real branching rather than one flat sibling list, and
///   `plan_faithful_snapshot`'s sibling-index build has something to index.
///   Depth 2 is far under `MAX_SECTION_TREE_DEPTH`; this bench measures the
///   sort, not the depth guard.
/// - **Orphans** — appended with a parent id that exists in no children
///   list, so the DFS never reaches them and `sections_in_dfs_order` hits
///   its orphan-tail sort. Fieldless, per [`ORPHAN_SECTION_DIVISOR`].
/// - **Fields** — `version` advances every [`FIELDS_PER_VERSION`] fields
///   (a simulated commit); within a version `authoring_set_order` advances
///   every [`FIELDS_PER_AUTHORING_RANK`] fields, so each
///   `(version, authoring_set_order)` pair is shared by that many fields;
///   consecutive field pairs share a `section_id`, so half of the
///   within-tie comparisons fall all the way through to `key_name`. The
///   cumulative `set_order` cursor is a running counter, as production
///   stores it — `plan_full_content` preserves it verbatim and
///   `plan_faithful_snapshot` overwrites it, which is exactly the
///   difference between the two arms.
/// - **Time samples** — spread over reachable sections with a unique
///   `(section_id, time)` map key and a tie-heavy `version`, so the
///   time-sample sort is exercised too.
///
/// Deterministic: no RNG, so two runs on the same machine generate
/// byte-identical input and criterion's run-to-run diff reflects code
/// changes only.
fn generate_object_state(field_count: usize) -> ObjectState {
    let reachable_section_count = (field_count / FIELDS_PER_SECTION).max(4);
    // A roughly square fan-out: `group_count` groups hang off the root and
    // the rest of the sections are distributed across them round-robin.
    let group_count = (reachable_section_count as f64).sqrt().ceil() as usize;
    let group_count = group_count.clamp(1, reachable_section_count);

    let mut sections: HashMap<u64, Section> = HashMap::new();
    let mut children: HashMap<u64, Vec<u64>> = HashMap::new();

    // Section ids are 1-based; 0 is the root sentinel `parent_id`.
    for slot in 0..reachable_section_count {
        let section_id = (slot + 1) as u64;
        let is_group = slot < group_count;
        let parent_id = if is_group {
            0
        } else {
            // Round-robin the leaves across the groups.
            ((slot - group_count) % group_count + 1) as u64
        };
        let version = 1 + (slot / FIELDS_PER_VERSION) as u64;

        children.entry(parent_id).or_default().push(section_id);
        sections.insert(
            section_id,
            Section {
                section_id,
                parent_id,
                name: format!("/World/prim_{slot:06}"),
                section_type: 1,
                section_order: (slot % 64) as u32,
                first_version: version,
                last_version: version,
            },
        );
    }

    // Orphans: unreachable from root (their parent appears in no children
    // list), fieldless, so both `plan_full_content` and
    // `plan_faithful_snapshot` sort the identical field set.
    let orphan_count = reachable_section_count / ORPHAN_SECTION_DIVISOR;
    let orphan_parent = u64::MAX;
    for slot in 0..orphan_count {
        let section_id = (reachable_section_count + slot + 1) as u64;
        sections.insert(
            section_id,
            Section {
                section_id,
                parent_id: orphan_parent,
                name: format!("/Orphan/prim_{slot:06}"),
                section_type: 1,
                section_order: slot as u32,
                first_version: 1,
                last_version: 1,
            },
        );
    }

    let mut fields: HashMap<(u64, String), Field> = HashMap::new();
    for index in 0..field_count {
        let within_version = index % FIELDS_PER_VERSION;
        let version = 1 + (index / FIELDS_PER_VERSION) as u64;
        let authoring_set_order = (within_version / FIELDS_PER_AUTHORING_RANK) as u64;
        // Consecutive pairs share a section, so within each tie group of
        // `FIELDS_PER_AUTHORING_RANK` fields two distinct sections appear
        // and each contributes two fields — the `key_name` fall-through.
        let section_id = ((index / 2) % reachable_section_count + 1) as u64;
        let key_name = key_name_for(index);

        fields.insert(
            (section_id, key_name.clone()),
            Field {
                section_id,
                key_name,
                value: field_value_for(index),
                set_order: index as u64,
                authoring_set_order,
                version,
            },
        );
    }

    let sample_count = field_count / TIME_SAMPLES_DIVISOR;
    let mut time_samples: HashMap<(u64, OrderedFloat<f64>), TimeSample> = HashMap::new();
    for index in 0..sample_count {
        let section_id = (index % reachable_section_count + 1) as u64;
        // `(section_id, time)` is the map key: the round is distinct for
        // every repeat visit to a section, so the pair is unique.
        let round = index / reachable_section_count;
        let time = OrderedFloat(round as f64 * 0.5);
        let version = 1 + (index / FIELDS_PER_AUTHORING_RANK) as u64;

        time_samples.insert(
            (section_id, time),
            TimeSample {
                section_id,
                time,
                value: field_value_for(index),
                version,
            },
        );
    }

    ObjectState {
        object_id: OBJECT_ID,
        sequence_num: 1 + (field_count / FIELDS_PER_VERSION) as u64,
        first_version: 1,
        source_format: "usda".to_string(),
        next_set_order: field_count as u64,
        sections,
        children,
        fields,
        time_samples,
    }
}

/// **Hand-maintained mirror** of the private
/// `DiffEngine::sort_fields_by_authoring`.
///
/// Same key — `(version, authoring_set_order, section_id)` then `key_name`
/// — and the same `sort_unstable_by` with a borrow-based comparator (a
/// `sort_by_key` on a tuple containing `key_name` would clone that `String`
/// on every comparison, which would make this arm measure allocation rather
/// than comparison). The production helper is private and stays private:
/// widening `domain`'s API surface to bench it is not a trade this
/// measurement is worth. See the module header's "The comparator is
/// replicated here, deliberately".
///
/// Authoritative implementation:
/// `src/domain/diff.rs::DiffEngine::sort_fields_by_authoring`.
fn sort_fields_like_diff_engine(fields: &mut [Field]) {
    fields.sort_unstable_by(|a, b| {
        (a.version, a.authoring_set_order, a.section_id)
            .cmp(&(b.version, b.authoring_set_order, b.section_id))
            .then_with(|| a.key_name.cmp(&b.key_name))
    });
}

/// Prove, before any measurement at a given size, that the generated state
/// actually has the four properties the arms' claims rest on.
///
/// Without this the sweep would still produce confident numbers on a
/// degenerate fixture, and every one of the four failure modes is silent:
///
/// 1. **The plan really takes the full-content branch** and emits every
///    generated field. A dispatch that fell through to `UpToDate` or
///    `Incremental` would report a near-empty call as "the assembly path".
/// 2. **`plan_faithful_snapshot` sorts the SAME field set.** It drops
///    fields on unreachable prims, so an orphan that accidentally acquired
///    fields would shrink the anchor arm's input and make the
///    `plan_full_content` vs `plan_faithful_snapshot` comparison a
///    comparison of two different workloads.
/// 3. **The orphan tail is non-empty**, so `sections_in_dfs_order`'s
///    orphan-`section_id` sort — one of the two sequences the Task's change
///    covers — is actually exercised rather than skipped.
/// 4. **The `(version, authoring_set_order)` ties are real**, and some of
///    them extend to `section_id` so the `key_name` `String` comparison is
///    reached. An all-distinct leading key resolves every comparison on the
///    first `u64` and would understate the comparator by a wide margin —
///    the exact way a hand-written fixture flatters a sort.
///
/// Panicking here fails the bench run, which is the right outcome: the
/// alternative is a plausible-looking number for the wrong input.
fn assert_state_is_representative(state: &ObjectState, field_count: usize) {
    let full = DiffEngine::plan(state, &[], 0, 0, None).expect("generated state must plan cleanly");
    let DiffPlan::FullContent {
        sections, fields, ..
    } = &full
    else {
        panic!(
            "DiffEngine::plan did not take the full-content branch for the \
             {field_count}-field state; the plan_full_content arm would be \
             measuring some other branch entirely"
        );
    };

    assert_eq!(
        fields.len(),
        field_count,
        "plan_full_content emitted {} of {field_count} generated fields",
        fields.len()
    );

    let snapshot =
        DiffEngine::plan_faithful_snapshot(state).expect("generated state must plan cleanly");
    let DiffPlan::FullContent {
        fields: snapshot_fields,
        ..
    } = &snapshot
    else {
        panic!("plan_faithful_snapshot must return FullContent");
    };
    assert_eq!(
        snapshot_fields.len(),
        fields.len(),
        "the anchor arm sorts {} fields while plan_full_content sorts {} — the \
         two arms would not be comparable. plan_faithful_snapshot drops fields \
         on unreachable prims, so this means an orphan section acquired fields.",
        snapshot_fields.len(),
        fields.len()
    );

    assert!(
        sections.len() > state.children.values().map(Vec::len).sum::<usize>(),
        "no orphan sections in the {field_count}-field state: the orphan-tail \
         sort in sections_in_dfs_order is not being exercised"
    );

    // Tie census on the generated fields, not on the sorted output — the
    // property belongs to the input.
    let distinct_leading_pairs: std::collections::HashSet<(u64, u64)> = state
        .fields
        .values()
        .map(|f| (f.version, f.authoring_set_order))
        .collect();
    assert!(
        distinct_leading_pairs.len() * 2 <= field_count,
        "the {field_count}-field state has {} distinct (version, \
         authoring_set_order) pairs — too few ties for the comparator's \
         section_id/key_name fall-throughs to be exercised",
        distinct_leading_pairs.len()
    );

    let distinct_triples: std::collections::HashSet<(u64, u64, u64)> = state
        .fields
        .values()
        .map(|f| (f.version, f.authoring_set_order, f.section_id))
        .collect();
    assert!(
        distinct_triples.len() < field_count,
        "every (version, authoring_set_order, section_id) triple in the \
         {field_count}-field state is unique, so the key_name String \
         comparison — the expensive fall-through — is never reached"
    );
}

fn bench_diff_plan_full_content(c: &mut Criterion) {
    let mut group = c.benchmark_group("diff_plan_full_content");

    for field_count in FIELD_COUNTS {
        // Built ONCE per size, outside every measured closure, so neither
        // the generation nor the map construction lands in any number.
        let state = generate_object_state(field_count);
        assert_state_is_representative(&state, field_count);

        // The unsorted source for `sort_only`: the fields in `HashMap`
        // iteration order, i.e. exactly what `plan_full_content` collects
        // before it sorts. Cloned (untimed) per iteration below so every
        // measured sort starts from the same unsorted permutation rather
        // than re-sorting already-sorted data.
        let unsorted_fields: Vec<Field> = state.fields.values().cloned().collect();

        group.throughput(Throughput::Elements(field_count as u64));

        // ── Arm 1: the whole full-content plan. ─────────────────────────
        // `have_object_id = 0` never matches `OBJECT_ID`, so `plan`
        // dispatches to `plan_full_content` on every iteration.
        group.bench_with_input(
            BenchmarkId::new("plan_full_content", field_count),
            &field_count,
            |b, _| {
                b.iter(|| {
                    let plan = DiffEngine::plan(black_box(&state), black_box(&[]), 0, 0, None)
                        .expect("generated state must plan cleanly");
                    black_box(plan)
                });
            },
        );

        // ── Arm 2: the sort alone — the delta the Task introduced. ───────
        // `iter_batched` so the `Vec` clone is setup (untimed) and the
        // timed region is nothing but the comparator. `LargeInput` keeps
        // criterion from batching enough 10_000-field clones to matter.
        group.bench_with_input(
            BenchmarkId::new("sort_only", field_count),
            &field_count,
            |b, _| {
                b.iter_batched(
                    || unsorted_fields.clone(),
                    |mut fields| {
                        sort_fields_like_diff_engine(black_box(&mut fields));
                        black_box(fields)
                    },
                    BatchSize::LargeInput,
                );
            },
        );

        // ── Arm 3: the comparative anchor. ──────────────────────────────
        // Same sort class (it calls the same private comparator) plus a
        // dense `set_order` reindex and a sibling-index build, on the far
        // hotter catch-up path.
        group.bench_with_input(
            BenchmarkId::new("plan_faithful_snapshot", field_count),
            &field_count,
            |b, _| {
                b.iter(|| {
                    let plan = DiffEngine::plan_faithful_snapshot(black_box(&state))
                        .expect("generated state must plan cleanly");
                    black_box(plan)
                });
            },
        );
    }

    group.finish();
}

criterion_group! {
    name = benches;
    // Shorter than criterion's 3 s warm-up / 5 s measurement defaults. The
    // 10_000-field arms cost milliseconds per iteration, so the stock
    // windows would put this nine-id sweep past the ten-minute ceiling one
    // agent-harness command gets. 500 ms / 2 s still collects criterion's
    // full 100-sample default at every size, and the numbers stay
    // comparable run-to-run because the config is fixed here rather than
    // passed on the command line.
    config = Criterion::default()
        .warm_up_time(Duration::from_millis(500))
        .measurement_time(Duration::from_secs(2));
    targets = bench_diff_plan_full_content
}
criterion_main!(benches);
