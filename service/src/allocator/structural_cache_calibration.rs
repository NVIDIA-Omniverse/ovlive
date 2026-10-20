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

//! Calibration of the structural-planning cache's weight estimate,
//! `storage::structural_cache::estimate_state_bytes`, against what jemalloc
//! actually allocates (structural-edits spec SR-3.3; Epic 4 breakdown decision
//! E3).
//!
//! It lives in the service binary's tests because only the binary allocates
//! through jemalloc (`#[global_allocator]` is bin-only, linux-gnu only): lib
//! and integration tests run on the system allocator. It is compiled, like its
//! parent module, only on linux-gnu, so it is skipped everywhere else. Run it
//! with `cargo test -p live-edit --bin live-edit structural_cache_calibration`
//! (add `-- --nocapture` to see the recorded error under plain `cargo test`;
//! nextest prints it for these tests through `.config/nextest.toml`).
//!
//! **What is measured.** Each state is built the way the service builds
//! one — `apply_delta` over the deltas below, then
//! `normalize_state` — and put in the form the cache stores it, its tables
//! compacted by `compact_state` as `admit` and `put_back` do. The measurement
//! is a `clone()` of it into a `Box`:
//! a planning clone is how the cache's entries are made (SR-3.5), and the
//! `Box` charges the `ObjectState` itself to the heap as a cache slot does.
//! The bytes held are jemalloc's per-thread allocated counter less its
//! deallocated counter across the clone, so allocations on other test threads
//! do not count and a clone's own temporaries cancel out. Both counters are in
//! jemalloc's size classes, which the estimate rounds to.
//!
//! **The states.** One layer grown by creates: the pseudo-root, `World` and
//! 16 zones under it, then prims each created with 3–8 attribute sections.
//! A new prim goes under a random zone at least 40 % of the time — always
//! while no prim exists yet, and whenever the earlier prim drawn as its parent
//! already sits at depth 8 — and otherwise under that earlier prim, so no prim
//! sits deeper than 8 levels. Section ids are random and non-zero. Each prim
//! carries a `specifier` and a `typeName`, and each attribute a `typeName`,
//! `custom`, `variability`
//! and `default`, as the C++ client library authored a prim at an inspected
//! revision; the values are deterministic filler sized per key (a `default`
//! is 12–48 bytes). Measured at 1k and 5k sections; at 5k after deleting
//! prims until half the sections are gone; at 7,150 sections — near the
//! sections table's load limit — after deleting three quarters of them, the
//! shape whose tombstones made the uncompacted estimate take a table for half
//! its size; and at 1k with every prim's transform animated (time samples)
//! and some attribute values external, so the time-sample table and the
//! external-value arm of the estimate are measured too. Each asserts ±30 %.
//!
//! **Both row forms.** The cache's entries are the compact form
//! (`CompactState`: per value row its key and external hash, the field names
//! interned), weighed by the `CompactRows` rule; the full form's rule
//! (`FullRows`) is kept for the measurement tools, which compare the two. The
//! 1k, 5k, half-removed, three-quarters-removed and animated-external states
//! are each built and measured in both forms, the compact one planned on
//! directly by the engine (the deletes of the two thinned states thin its
//! compact tables and name interner, which `compact_state` then compacts).
//!
//! Each test prints one `CALIBRATION` line with the measured error.

use std::hint::black_box;

use live_edit::domain::delta::apply_delta;
use live_edit::domain::replay::normalize_state;
use live_edit::domain::types::{
    CompactRows, FieldRows, FieldValue, FullRows, ObjectStateOf, SampleRows, ValueRows,
};
use live_edit::storage::structural_cache::{compact_state, estimate_state_bytes, WeighedRows};
use oodf_ffi::{
    CommandCreateSection, CommandDeleteSection, CommandDiffChangeSection, OwnedDelta, SetField,
    StructCommand, TimeSample,
};

use super::read;

/// The estimate's allowed error against the measured allocation (Epic 4
/// acceptance).
const TOLERANCE: f64 = 0.30;

/// Sdf spec types as the C++ client library sends them.
const SECTION_TYPE_ATTRIBUTE: u8 = 1;
const SECTION_TYPE_PRIM: u8 = 6;
const SECTION_TYPE_PSEUDO_ROOT: u8 = 7;
/// The pseudo-root's section id, under the stored root `0`.
const PSEUDO_ROOT: u64 = 1;
/// Zones under `World`, the shared parents a new prim may take.
const ZONES: usize = 16;
/// An xformable prim's attribute specs, in authoring order; a prim takes the
/// first 3–8.
const ATTR_NAMES: [&str; 8] = [
    "xformOp:translate",
    "xformOp:rotateXYZ",
    "xformOp:scale",
    "xformOpOrder",
    "extent",
    "size",
    "primvars:displayColor",
    "visibility",
];
const PRIM_TYPES: [&str; 4] = ["Xform", "Cube", "Mesh", "Sphere"];
/// The deepest a created prim may sit (zones are depth 2).
const MAX_DEPTH: usize = 8;
/// Prims created per delta while building a state. Only how many deltas a
/// build takes depends on it.
const PRIMS_PER_DELTA: usize = 50;
/// Prims deleted per delta while thinning a state: small, so thinning stops
/// close to its target.
const DELETES_PER_DELTA: usize = 5;
/// Time samples per animated transform: one second of animation at 24 fps.
const FRAMES: usize = 24;
/// The bytes of one animated `xformOp:translate` sample: a two-byte tag and
/// three `f64`s.
const TRANSLATE_VALUE_BYTES: usize = 26;
/// The attribute specs whose `default` is made an external value in the
/// animated state.
const EXTERNAL_ATTRS: [&str; 2] = ["primvars:displayColor", "extent"];
/// The stored size of each such external value (its bytes live in the store;
/// the state keeps only the hash and size).
const EXTERNAL_VALUE_BYTES: u64 = 64 * 1024;
/// The seed of every state's random choices, so each run measures the same
/// states.
const SEED: u64 = 0x5eed_b20c;

/// A small deterministic generator (SplitMix64).
struct Rng(u64);

impl Rng {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    fn below(&mut self, n: u64) -> u64 {
        self.next_u64() % n
    }
}

/// `bytes` bytes of deterministic filler.
fn filler(bytes: usize, seed: u64) -> Vec<u8> {
    (0..bytes)
        .map(|i| (seed.wrapping_add(i as u64) % 251) as u8)
        .collect()
}

fn field(section_id: u64, key: &str, bytes: usize, seed: u64, set_order: u64) -> SetField {
    SetField {
        section_id,
        key_name: key.to_owned(),
        value_or_ext_hash: filler(bytes, seed),
        ext_value_size: 0,
        ext_value_index: 0,
        set_order,
    }
}

/// One layer object growing under the create mix the module doc describes,
/// applied through the service's own delta engine, in the row form `R`: the
/// full form (`FullRows`), or the compact form the cache stores
/// (`CompactRows`), which the engine plans on directly as a repair or a
/// structural commit does. One seed gives both forms the same sections and
/// rows.
struct Layer<R: ValueRows> {
    rng: Rng,
    state: ObjectStateOf<R>,
    zones: Vec<u64>,
    /// Live prims and their depth.
    prims: Vec<(u64, usize)>,
    names: u64,
}

impl<R: ValueRows> Layer<R> {
    /// The fresh layer: the pseudo-root, `World` and the zones, as one reset
    /// delta.
    fn seeded() -> Self {
        let mut rng = Rng(SEED);
        let world = rng.next_u64() | 2;
        let zones: Vec<u64> = (0..ZONES).map(|_| rng.next_u64() | 2).collect();
        let change = |parent_id, section_id, name: &str, section_type, section_order| {
            StructCommand::DiffChangeSection(CommandDiffChangeSection {
                parent_id,
                section_id,
                section_name: name.to_owned(),
                section_type,
                section_order,
            })
        };
        let mut struct_commands = vec![
            change(0, PSEUDO_ROOT, "", SECTION_TYPE_PSEUDO_ROOT, 0),
            change(PSEUDO_ROOT, world, "World", SECTION_TYPE_PRIM, 0),
        ];
        let mut set_fields = vec![
            field(world, "specifier", 4, 1, 1),
            field(world, "typeName", 9, 2, 2),
        ];
        for (i, &zone) in zones.iter().enumerate() {
            struct_commands.push(change(
                world,
                zone,
                &format!("Zone{i}"),
                SECTION_TYPE_PRIM,
                i as u32,
            ));
            let order = set_fields.len() as u64;
            set_fields.push(field(zone, "specifier", 4, zone, order + 1));
            set_fields.push(field(zone, "typeName", 9, zone + 1, order + 2));
        }
        let mut state = ObjectStateOf::<R> {
            object_id: 1,
            source_format: "usd".to_owned(),
            ..ObjectStateOf::default()
        };
        apply_delta(
            &mut state,
            Some(OwnedDelta {
                is_diff: true,
                base_version: 0,
                struct_commands,
                set_fields,
                ..OwnedDelta::default()
            }),
        )
        .expect("the seed delta applies");
        Self {
            rng,
            state,
            zones,
            prims: Vec::new(),
            names: 0,
        }
    }

    fn sections(&self) -> usize {
        self.state.sections.len()
    }

    fn fresh_id(&mut self) -> u64 {
        loop {
            let id = self.rng.next_u64();
            if id > PSEUDO_ROOT && !self.state.sections.contains_key(&id) {
                return id;
            }
        }
    }

    /// A parent for a new prim: a random zone 40 % of the time (and always
    /// while no prim exists yet), else a random earlier prim, or a zone when
    /// that prim is already at [`MAX_DEPTH`]. Returns it with the new prim's
    /// depth.
    fn pick_parent(&mut self) -> (u64, usize) {
        if !self.prims.is_empty() && self.rng.below(10) >= 4 {
            let (prim, depth) = self.prims[self.rng.below(self.prims.len() as u64) as usize];
            if depth < MAX_DEPTH {
                return (prim, depth + 1);
            }
        }
        (self.zones[self.rng.below(ZONES as u64) as usize], 3)
    }

    /// Create prims, [`PRIMS_PER_DELTA`] per delta, until the state holds at
    /// least `target` sections.
    fn grow_to(&mut self, target: usize) {
        while self.sections() < target {
            let mut struct_commands = Vec::new();
            let mut set_fields = Vec::new();
            let mut created = 0;
            for _ in 0..PRIMS_PER_DELTA {
                if self.sections() + created >= target {
                    break;
                }
                let (parent, depth) = self.pick_parent();
                let id = self.fresh_id();
                self.names += 1;
                let type_name = PRIM_TYPES[self.rng.below(PRIM_TYPES.len() as u64) as usize];
                let seed = self.rng.next_u64();
                struct_commands.push(StructCommand::CreateSection(CommandCreateSection {
                    parent_id: parent,
                    section_id: id,
                    section_name: format!("w{}_p{}", self.names % 8, self.names),
                    section_type: SECTION_TYPE_PRIM,
                }));
                let mut order = set_fields.len() as u64;
                set_fields.push(field(id, "specifier", 4, seed, order + 1));
                set_fields.push(field(
                    id,
                    "typeName",
                    type_name.len() + 4,
                    seed + 1,
                    order + 2,
                ));
                let attrs = 3 + self.rng.below(6) as usize;
                for (k, name) in ATTR_NAMES.iter().take(attrs).enumerate() {
                    let attr = self.fresh_id();
                    struct_commands.push(StructCommand::CreateSection(CommandCreateSection {
                        parent_id: id,
                        section_id: attr,
                        section_name: (*name).to_owned(),
                        section_type: SECTION_TYPE_ATTRIBUTE,
                    }));
                    let attr_seed = seed.wrapping_add(10 * (k as u64 + 1));
                    for (key, bytes) in [
                        ("typeName", 8),
                        ("custom", 2),
                        ("variability", 4),
                        ("default", 12 + (attr_seed % 37) as usize),
                    ] {
                        order = set_fields.len() as u64;
                        set_fields.push(field(attr, key, bytes, attr_seed, order + 1));
                    }
                }
                self.prims.push((id, depth));
                created += 1 + attrs;
            }
            self.apply(struct_commands, set_fields);
        }
    }

    /// Delete random prims, whole subtrees, [`DELETES_PER_DELTA`] per delta,
    /// until the state holds at most `target` sections.
    fn thin_to(&mut self, target: usize) {
        while self.sections() > target && !self.prims.is_empty() {
            let mut struct_commands = Vec::new();
            for _ in 0..DELETES_PER_DELTA.min(self.prims.len()) {
                let (prim, _) = self.prims[self.rng.below(self.prims.len() as u64) as usize];
                let parent_id = self.state.sections[&prim].parent_id;
                struct_commands.push(StructCommand::DeleteSection(CommandDeleteSection {
                    parent_id,
                    section_id: prim,
                }));
            }
            self.apply(struct_commands, Vec::new());
            let sections = &self.state.sections;
            self.prims.retain(|(prim, _)| sections.contains_key(prim));
        }
    }

    fn apply(&mut self, struct_commands: Vec<StructCommand>, set_fields: Vec<SetField>) {
        self.apply_with_samples(struct_commands, set_fields, Vec::new());
    }

    fn apply_with_samples(
        &mut self,
        struct_commands: Vec<StructCommand>,
        set_fields: Vec<SetField>,
        time_samples: Vec<TimeSample>,
    ) {
        apply_delta(
            &mut self.state,
            Some(OwnedDelta {
                struct_commands,
                set_fields,
                time_samples,
                ..OwnedDelta::default()
            }),
        )
        .expect("a generated delta applies");
    }

    /// Animate every `xformOp:translate` with [`FRAMES`] inline time
    /// samples, and make the `default` of every [`EXTERNAL_ATTRS`] attribute
    /// an external value, in one delta.
    fn animate_and_externalize(&mut self) {
        let mut attributes: Vec<(u64, String)> = self
            .state
            .sections
            .values()
            .filter(|section| section.section_type == SECTION_TYPE_ATTRIBUTE)
            .map(|section| (section.section_id, section.name.clone()))
            .collect();
        attributes.sort_unstable();
        let mut time_samples = Vec::new();
        let mut set_fields = Vec::new();
        for (attr, name) in attributes {
            if name == ATTR_NAMES[0] {
                for frame in 0..FRAMES {
                    time_samples.push(TimeSample {
                        section_id: attr,
                        time: frame as f64,
                        value_or_ext_hash: filler(TRANSLATE_VALUE_BYTES, attr ^ frame as u64),
                        ext_value_size: 0,
                        ext_value_index: 0,
                    });
                }
            } else if EXTERNAL_ATTRS.contains(&name.as_str()) {
                set_fields.push(SetField {
                    section_id: attr,
                    key_name: "default".to_owned(),
                    value_or_ext_hash: filler(32, attr),
                    ext_value_size: EXTERNAL_VALUE_BYTES,
                    ext_value_index: 1,
                    set_order: set_fields.len() as u64 + 1,
                });
            }
        }
        self.apply_with_samples(Vec::new(), set_fields, time_samples);
    }

    /// The state as the cache stores it: normalized (SR-3.2), and its tables
    /// compacted, as `StructuralCache::admit` and `CheckedOut::put_back` do
    /// (on the compact form, its name interner's free ids trimmed too).
    fn cached(self) -> ObjectStateOf<R> {
        let mut state = self.normalized();
        compact_state(&mut state);
        state
    }

    /// The state normalized (SR-3.2) but not compacted.
    fn normalized(mut self) -> ObjectStateOf<R> {
        let _dangling = normalize_state(&mut self.state);
        self.state
    }
}

/// The bytes jemalloc holds for a boxed `clone()` of `state`, and
/// [`estimate_state_bytes`] of that clone.
///
/// On the compact form the clone shares each interned name's `Arc<str>` with
/// `state` (a clone bumps its count and allocates nothing), where the
/// estimate charges every name once per state: a layer holds 5–15 distinct
/// field names, so the difference is a few hundred bytes.
fn measure_clone<R: WeighedRows>(state: &ObjectStateOf<R>) -> (u64, u64) {
    let allocated: *mut u64 =
        read(c"thread.allocatedp").expect("`thread.allocatedp` must be readable");
    let deallocated: *mut u64 =
        read(c"thread.deallocatedp").expect("`thread.deallocatedp` must be readable");
    // SAFETY: both pointers are the calling thread's own jemalloc counters,
    // valid for the thread's lifetime, and only read here on that thread.
    let before = unsafe { (allocated.read_volatile(), deallocated.read_volatile()) };
    let held = black_box(Box::new(state.clone()));
    // SAFETY: as above.
    let after = unsafe { (allocated.read_volatile(), deallocated.read_volatile()) };
    let measured = (after.0 - before.0) - (after.1 - before.1);
    let estimate = estimate_state_bytes(&held);
    drop(held);
    (measured, estimate)
}

/// Measure `state`, print the `CALIBRATION` record, and assert the estimate
/// is within [`TOLERANCE`] of the measurement.
fn calibrate<R: WeighedRows>(label: &str, state: &ObjectStateOf<R>) {
    let (measured, estimate) = measure_clone(state);
    let error = (estimate as f64 - measured as f64) / measured as f64;
    eprintln!(
        "CALIBRATION estimate_state_bytes {label}: sections={} fields={} children_lists={} \
         time_samples={} measured_bytes={measured} estimate_bytes={estimate} error={:+.1}%",
        state.sections.len(),
        FieldRows::len(&state.fields),
        state.children.len(),
        SampleRows::len(&state.time_samples),
        error * 100.0,
    );
    assert!(
        error.abs() <= TOLERANCE,
        "{label}: estimate_state_bytes = {estimate} B is {:+.1}% off jemalloc's measured \
         {measured} B, outside ±{:.0}%",
        error * 100.0,
        TOLERANCE * 100.0
    );
}

/// The 1k state, in row form `R`, as the cache stores it.
fn at_1k<R: WeighedRows>() -> ObjectStateOf<R> {
    let mut layer = Layer::<R>::seeded();
    layer.grow_to(1_000);
    layer.cached()
}

/// The 5k state, in row form `R`, as the cache stores it.
fn at_5k<R: WeighedRows>() -> ObjectStateOf<R> {
    let mut layer = Layer::<R>::seeded();
    layer.grow_to(5_000);
    layer.cached()
}

/// The 5k state, in row form `R`, after deleting prims until at most half its
/// sections are left, as the cache stores it. On the compact form the
/// deletes run on the compact state, so its row tables and name interner are
/// the ones a cascade thinned and `compact_state` then compacted.
fn half_removed<R: WeighedRows>() -> ObjectStateOf<R> {
    let mut layer = Layer::<R>::seeded();
    layer.grow_to(5_000);
    let built = layer.sections();
    layer.thin_to(built / 2);
    assert!(
        layer.sections() <= built / 2,
        "fixture: at least half of the {built} sections must have been removed; {} remain",
        layer.sections()
    );
    layer.cached()
}

#[test]
fn estimate_state_bytes_within_30_percent_of_jemalloc_at_1k_sections() {
    calibrate("loadtest_1k", &at_1k::<FullRows>());
}

#[test]
fn estimate_state_bytes_within_30_percent_of_jemalloc_at_5k_sections() {
    calibrate("loadtest_5k", &at_5k::<FullRows>());
}

#[test]
fn estimate_state_bytes_within_30_percent_of_jemalloc_after_many_section_removals() {
    calibrate("loadtest_5k_half_removed", &half_removed::<FullRows>());
}

// ── The compact form, the cache's entry (b.mr6) ─────────────────────────────

#[test]
fn compact_estimate_state_bytes_within_30_percent_of_jemalloc_at_1k_sections() {
    calibrate("compact_loadtest_1k", &at_1k::<CompactRows>());
}

#[test]
fn compact_estimate_state_bytes_within_30_percent_of_jemalloc_at_5k_sections() {
    calibrate("compact_loadtest_5k", &at_5k::<CompactRows>());
}

#[test]
fn compact_estimate_state_bytes_within_30_percent_of_jemalloc_after_many_section_removals() {
    calibrate(
        "compact_loadtest_5k_half_removed",
        &half_removed::<CompactRows>(),
    );
}

/// The compact form's time-sample table and its boxed external hashes: the
/// animated, partly external 1k state of the full form's test below, built on
/// the compact form.
#[test]
fn compact_estimate_state_bytes_within_30_percent_of_jemalloc_with_time_samples_and_external_values(
) {
    let mut layer = Layer::<CompactRows>::seeded();
    layer.grow_to(1_000);
    layer.animate_and_externalize();
    let state = layer.cached();
    let external = state.fields.external_rows();
    assert!(
        !state.time_samples.is_empty() && external > 0,
        "fixture: the state must hold time samples ({}) and external values ({external})",
        SampleRows::len(&state.time_samples)
    );

    calibrate("compact_loadtest_1k_animated_external", &state);
}

/// The state of 7,150 sections, in row form `R`, after deleting prims until at
/// most a quarter of them are left: normalized, but not yet compacted, so a
/// test can print the uncompacted error before it compacts. On the compact
/// form the deletes run on the compact state, so its sections, fields and
/// samples tables and its name interner are the ones a cascade thinned.
fn hard_thinned<R: WeighedRows>() -> ObjectStateOf<R> {
    let mut layer = Layer::<R>::seeded();
    layer.grow_to(7_150);
    let built = layer.sections();
    layer.thin_to(built / 4);
    assert!(
        layer.sections() <= built / 4,
        "fixture: at least three quarters of the {built} sections must have been removed; \
         {} remain",
        layer.sections()
    );
    layer.normalized()
}

/// A table thinned hard from near its 7/8 load limit: 7,150 sections fill the
/// sections table's 8,192 buckets to 87 %, and thinning to a quarter leaves it
/// with more tombstones than items, so its `capacity()` falls below half its
/// buckets. Uncompacted, the estimate took each such table for half its size
/// (−47 % when recorded); the cache compacts every state it stores
/// (`compact_state`), and the compacted state is the one asserted. (Thinning
/// the 5k state, at 61 % load, frees most erased slots outright instead.)
///
/// The uncompacted error is printed as information only: the cache never
/// stores that form.
#[test]
fn estimate_state_bytes_within_30_percent_of_jemalloc_after_hard_thinning_from_near_the_load_limit()
{
    let mut state = hard_thinned::<FullRows>();

    let (measured, estimate) = measure_clone(&state);
    eprintln!(
        "INFO estimate_state_bytes loadtest_7k_three_quarters_removed uncompacted (not \
         asserted; never stored): sections_capacity={} fields_capacity={} \
         measured_bytes={measured} estimate_bytes={estimate} error={:+.1}%",
        state.sections.capacity(),
        state.fields.capacity(),
        (estimate as f64 - measured as f64) / measured as f64 * 100.0,
    );
    compact_state(&mut state);
    calibrate("loadtest_7k_three_quarters_removed", &state);
}

/// The hard-thinned state of the full form's test above, built and thinned on
/// the compact form: the compact rule's sections, fields and samples tables
/// after heavy thinning from near the sections table's load limit, compacted
/// (with the name interner's free ids trimmed) as the cache stores it.
///
/// The uncompacted error is printed as information only: the cache never
/// stores that form.
#[test]
fn compact_estimate_state_bytes_within_30_percent_of_jemalloc_after_hard_thinning_from_near_the_load_limit(
) {
    let mut state = hard_thinned::<CompactRows>();

    let (measured, estimate) = measure_clone(&state);
    eprintln!(
        "INFO estimate_state_bytes compact_loadtest_7k_three_quarters_removed uncompacted (not \
         asserted; never stored): sections_capacity={} fields_capacity={} \
         measured_bytes={measured} estimate_bytes={estimate} error={:+.1}%",
        state.sections.capacity(),
        state.fields.capacity(),
        (estimate as f64 - measured as f64) / measured as f64 * 100.0,
    );
    compact_state(&mut state);
    calibrate("compact_loadtest_7k_three_quarters_removed", &state);
}

/// The time-sample table and the external-value arm of the estimate: every
/// prim's transform animated over [`FRAMES`] inline samples, and the
/// `default` of every [`EXTERNAL_ATTRS`] attribute an external value (only
/// its hash and size are held).
#[test]
fn estimate_state_bytes_within_30_percent_of_jemalloc_with_time_samples_and_external_values() {
    let mut layer = Layer::<FullRows>::seeded();
    layer.grow_to(1_000);
    layer.animate_and_externalize();
    let state = layer.cached();
    let external = state
        .fields
        .values()
        .filter(|field| matches!(field.value, FieldValue::External { .. }))
        .count();
    assert!(
        !state.time_samples.is_empty() && external > 0,
        "fixture: the state must hold time samples ({}) and external values ({external})",
        state.time_samples.len()
    );

    calibrate("loadtest_1k_animated_external", &state);
}
