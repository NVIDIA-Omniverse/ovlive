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

//! A restored object's sequence band (SDD SR-2.5, §22.6): the arithmetic that
//! seeds a restored object's writer cursor above every sequence number its
//! pre-restore subscribers may hold, and the version flattening that makes
//! every restored entry visible at that cursor.
//!
//! One definition, two callers: the on-access restore
//! (`checkpoint::apply::apply_restore`) and the restore Job's writer
//! (`dr::job::execute`). Pure: no I/O, no logging. Each caller keeps its own
//! log lines and errors at its call site.
//!
//! - [`seq_band`]: `restore_floor = max(snap_seq, max_seq_observed)`, `band =
//!   max(safety_margin, observed_commit_span)`, `seeded_seq = restore_floor +
//!   band`, refused at or above the f64 safe-integer ceiling (2^53 − 1).
//! - [`band_is_outsized`]: whether a measured span is far enough above the
//!   configured margin to earn the caller's operator WARN.
//! - [`flatten_to_seeded_seq`]: every section's `last_version`, every field's
//!   `version` and every time sample's `version` set to `seeded_seq`, each
//!   section's `first_version` kept, and `sequence_num` set to `seeded_seq`.

use crate::checkpoint::apply::{F64_SAFE_INT_MAX, OUTSIZED_BAND_MARGIN_MULTIPLE};
use crate::domain::types::ObjectState;

/// One restored object's band: the floor it can prove, the guard band above
/// it, and the seeded writer cursor `restore_floor + band`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SeqBand {
    /// `max(snap_seq, max_seq_observed)`: the highest sequence number the
    /// restore can prove was issued.
    pub restore_floor: u64,
    /// `max(safety_margin, observed_commit_span)`: the guard band, stamped as
    /// `restore_seq_bump`.
    pub band: u64,
    /// `restore_floor + band` (saturating): the writer cursor the restored
    /// header carries as `seq`.
    pub seeded_seq: u64,
}

/// A band whose seeded cursor reaches the f64 safe-integer ceiling
/// (2^53 − 1): both commit paths refuse every write at or above it, so a
/// restore seeded there would be unwritable. Carries the band as computed,
/// for the caller's log line.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SeqCeiling(pub SeqBand);

/// The band of a restore at `snap_seq` (SR-2.5, b.8y5, b.vnw): the floor is
/// `snap_seq` raised to `max_seq_observed` (0 when the caller read no
/// surviving high-water), the band is the configured `safety_margin` raised
/// to the measured `observed_commit_span`, and the cursor is their sum.
/// `Err` when the cursor reaches the f64 safe-integer ceiling.
pub fn seq_band(
    snap_seq: u64,
    max_seq_observed: u64,
    observed_commit_span: u64,
    safety_margin: u64,
) -> Result<SeqBand, SeqCeiling> {
    let restore_floor = snap_seq.max(max_seq_observed);
    let band = safety_margin.max(observed_commit_span);
    let seeded_seq = restore_floor.saturating_add(band);
    let banded = SeqBand {
        restore_floor,
        band,
        seeded_seq,
    };
    if seeded_seq >= F64_SAFE_INT_MAX {
        return Err(SeqCeiling(banded));
    }
    Ok(banded)
}

/// Whether a measured `observed_commit_span` exceeds the configured
/// `safety_margin` by more than `apply::OUTSIZED_BAND_MARGIN_MULTIPLE` (8):
/// the band is applied unclamped either way, but such a span is worth an
/// operator's attention (b.vnw).
pub fn band_is_outsized(observed_commit_span: u64, safety_margin: u64) -> bool {
    observed_commit_span > safety_margin.saturating_mul(OUTSIZED_BAND_MARGIN_MULTIPLE)
}

/// Flatten `state`'s per-entry versions onto `seeded_seq` (b.2ti) and set
/// its `sequence_num` to `seeded_seq`. `Section::first_version` is kept.
pub fn flatten_to_seeded_seq(state: &mut ObjectState, seeded_seq: u64) {
    state.sequence_num = seeded_seq;
    // Flatten the per-entry versions a diff builder FILTERS on —
    // `Section::last_version`, `Field::version`, `TimeSample::version` — onto
    // `seeded_seq` (b.2ti). `Section::first_version` is deliberately exempt;
    // see the last paragraph.
    //
    // A restore is ONE write of the WHOLE object at `seeded_seq`, so every
    // entry it materializes must be visible as of that sequence. The restore
    // preserves `object_id` (the SR-2.4 identity guard on the on-access path;
    // the plan's identity check in the restore Job), so a subscriber
    // that survives it does NOT re-bootstrap — it reconnects and takes the
    // incremental path, where `src/domain/diff.rs`
    // (`plan_faithful_coalesced`) emits only entries with `version >
    // have_version` (it also restates the other children of a parent with
    // such a child, and nothing else). Carrying the pre-restore stamps
    // would therefore withhold the restored content from exactly the
    // subscribers that need correcting:
    // the base decode leaves everything at `version = 1`, and a composed
    // lineage leaves base entries below its tail entries.
    //
    // What the guard band does and does not buy. `restore_floor`
    // ([`seq_band`]) provably dominates every sequence this server can PROVE
    // the object issued — the checkpoint's `snap_seq`, and the surviving
    // stream top when the probe returned one. The commits lost in the RPO
    // window are by definition not provable; the band above the floor
    // (`max(configured margin, measured commit span)`) is what COVERS them. So
    // `seeded_seq` dominates a reconnector's `have_version` only insofar as
    // that band is sized for the layer's real per-interval commit rate. If the
    // band is undersized a subscriber can hold `have_version >= seeded_seq`,
    // and for that subscriber this stamp does not help — its diff still
    // filters the restored entries out. Closing that residual is the band's
    // job, not this step's (SDD §22.6).
    //
    // Determinism, not order preservation. A constant `version` keeps
    // `sort_fields_by_authoring`'s key `(version, authoring_set_order,
    // section_id, key_name)` a TOTAL order, so the emit sequence stays
    // deterministic — but it is a DIFFERENT sequence. With `version` constant
    // the primary key collapses to `authoring_set_order`, which resets to 0
    // per delta, so on a composed lineage base-era and tail-era fields
    // interleave where composition had kept the eras separated. That
    // reordering propagates into `dense_reindex_by_authoring`'s emitted
    // indices and into the bytes of the re-base artifact named below.
    //
    // The time-sample comparator loses its primary key the same way, and in
    // all three builders that use it: `plan_full_content`,
    // `plan_faithful_snapshot` and `plan_faithful_coalesced` all sort on
    // `(version, section_id, time)`, so with `version` constant the key
    // collapses to `(section_id, time)` and samples authored in different eras
    // interleave. For the re-base artifact that is not cosmetic — per
    // `plan_full_content`'s rustdoc, `collect_external_value_hashes` scans
    // `fields` THEN `time_samples` into ONE first-appearance pool, and
    // `build_reset_ompf_from_state` forces that arm, so the artifact's ext-refs
    // sidecar hash order (`ExternalValueScan::ordered_hashes`) moves with the
    // time-sample order too. Still total, still deterministic — a different
    // total order.
    //
    // Scope. `compose_lineage`'s own per-record stamping is left alone — it is
    // the in-order-replay witness. But this flattening does not stop at what
    // Valkey persists: on the on-access path `AppliedCheckpoint::state` is
    // moved into `BaseBootstrap::spawn_restore_rebase`
    // (`src/checkpoint/base_bootstrap.rs`; the recovery path calls it from
    // `src/checkpoint/recovery.rs`) and rebuilt by `build_reset_ompf_from_state`
    // (`src/checkpoint/assembler.rs`), so the SR-3.6 restore-minted re-base
    // artifact is assembled from these flattened versions too.
    //
    // Why `Section::first_version` is exempt. `plan_faithful_coalesced`'s
    // abandoned-prim retraction arm fires on `first_version <= have_version &&
    // !live_prim_ids.contains(id)`. Raising `first_version` to `seeded_seq`
    // would put it above every reconnector's `have_version`, so the orphan
    // sections a restored base carries (`plan_full_content` emits unreachable
    // sections into the base OMPF) could never be retracted. Left at the value
    // the decoder produced (`1`) or the composition engine replayed (the real
    // tail sequence, for a tail-created section), that arm keeps working, and
    // new-vs-modified classification stays truthful for tail-created sections
    // on a composed lineage — base-era sections read as pre-existing by
    // construction, since the decoder stamps `first_version = 1` (on a
    // base-only restore that is EVERY section, so the retraction arm above
    // matches every base-carried orphan for every reconnector, including
    // orphans created after its `have_version` that it never held). The
    // collapsed classification is harmless because `build_delta`
    // (`src/storage/diff_read.rs`) emits the same `DiffChangeSection` upsert
    // for both classes, so the wire shape is unchanged either way — the
    // load-bearing reason to preserve `first_version` is the retraction arm,
    // not the classification. `first_version <= last_version` still holds
    // everywhere, and not by luck: the restore's `snap_seq` is its lineage's
    // head, cross-checked equal to `composed.composed_seq` — the terminal
    // replay cursor — and so bounds every version composition stamped, while
    // a base-only decode stamps `1`; with `restore_floor >= snap_seq` and
    // `band >= 1` (`checkpoint_seq_safety_margin` is zero-rejected in
    // `Config::from_env`, and the band is a `max` over it), `seeded_seq =
    // restore_floor + band` is strictly above the composed head and at least
    // `1`, hence at or above every preserved `first_version`. SDD §22.6
    // (guard band), §12.10 (catch-up / reconnect).
    for section in state.sections.values_mut() {
        section.last_version = seeded_seq;
    }
    for field in state.fields.values_mut() {
        field.version = seeded_seq;
    }
    for sample in state.time_samples.values_mut() {
        sample.version = seeded_seq;
    }
    // The exemption's ordering derivation above rests on the floor/band
    // arithmetic and on `snap_seq == composed_seq` upstream; pin it here so a
    // change to either trips in-crate instead of shipping sections whose
    // `first_version` outruns their `last_version`. `debug_assert!` rather than
    // `assert!` because a restore runs on a request path — a release build must
    // not abort a recovery it can still serve over an ordering invariant.
    debug_assert!(
        state
            .sections
            .values()
            .all(|section| section.first_version <= section.last_version),
        "restore must leave first_version <= last_version: seeded_seq is \
         restore_floor + band, with restore_floor >= snap_seq (the composed \
         head, which bounds every replayed version) and band >= 1, so it \
         dominates every preserved first_version"
    );
}

#[cfg(test)]
mod tests {
    //! U3 (pins): the band arithmetic and the flattening both restore paths
    //! share, read against hand-computed figures.

    use std::collections::HashMap;

    use ordered_float::OrderedFloat;

    use super::*;
    use crate::domain::types::{Field, FieldValue, Section, TimeSample};

    /// The floor is `snap_seq` raised to a surviving high-water, never
    /// lowered by one below it; the band is the margin raised to a measured
    /// span; the cursor is their sum.
    #[test]
    fn seq_band_floors_at_the_higher_of_snap_seq_and_high_water_and_bands_by_the_larger_term() {
        assert_eq!(
            seq_band(100, 140, 0, 1_000),
            Ok(SeqBand {
                restore_floor: 140,
                band: 1_000,
                seeded_seq: 1_140,
            }),
            "a high-water above snap_seq raises the floor; the margin is the band"
        );
        assert_eq!(
            seq_band(100, 60, 0, 1_000),
            Ok(SeqBand {
                restore_floor: 100,
                band: 1_000,
                seeded_seq: 1_100,
            }),
            "a high-water below snap_seq does not lower it"
        );
        assert_eq!(
            seq_band(100, 0, 2_500, 1_000),
            Ok(SeqBand {
                restore_floor: 100,
                band: 2_500,
                seeded_seq: 2_600,
            }),
            "a measured span above the margin is the band"
        );
    }

    /// A cursor at or above 2^53 − 1 is refused, carrying the band as
    /// computed; one below it is not.
    #[test]
    fn seq_band_refuses_a_cursor_at_the_f64_safe_integer_ceiling() {
        let below = seq_band(F64_SAFE_INT_MAX - 1_001, 0, 0, 1_000);
        assert_eq!(below.map(|b| b.seeded_seq), Ok(F64_SAFE_INT_MAX - 1));
        assert_eq!(
            seq_band(F64_SAFE_INT_MAX - 1_000, 0, 0, 1_000),
            Err(SeqCeiling(SeqBand {
                restore_floor: F64_SAFE_INT_MAX - 1_000,
                band: 1_000,
                seeded_seq: F64_SAFE_INT_MAX,
            }))
        );
        assert!(
            matches!(seq_band(u64::MAX - 5, 0, 0, 1_000), Err(SeqCeiling(b)) if b.seeded_seq == u64::MAX),
            "the sum saturates rather than wrapping below the ceiling"
        );
    }

    /// A span is outsized only strictly above eight margins.
    #[test]
    fn band_is_outsized_only_above_eight_times_the_margin() {
        assert_eq!(OUTSIZED_BAND_MARGIN_MULTIPLE, 8);
        assert!(!band_is_outsized(8_000, 1_000));
        assert!(band_is_outsized(8_001, 1_000));
        assert!(!band_is_outsized(0, 1_000));
    }

    /// The flattening stamps every section's `last_version`, every field's
    /// and every time sample's `version` and `sequence_num` with the seeded
    /// cursor, and keeps each section's `first_version` and everything else.
    #[test]
    fn flatten_to_seeded_seq_stamps_every_version_and_keeps_first_version() {
        let section = |id: u64, first: u64, last: u64| Section {
            section_id: id,
            parent_id: 0,
            name: format!("s{id}"),
            section_type: 6,
            section_order: 0,
            first_version: first,
            last_version: last,
        };
        let mut state = ObjectState {
            sequence_num: 7,
            first_version: 1,
            sections: HashMap::from([(1, section(1, 1, 3)), (2, section(2, 5, 7))]),
            ..ObjectState::default()
        };
        state.fields.insert(
            (1, "a".to_owned()),
            Field {
                section_id: 1,
                key_name: "a".to_owned(),
                value: FieldValue::Inline(vec![1]),
                set_order: 4,
                authoring_set_order: 2,
                version: 3,
            },
        );
        state.time_samples.insert(
            (2, OrderedFloat(1.5)),
            TimeSample {
                section_id: 2,
                time: OrderedFloat(1.5),
                value: FieldValue::Inline(vec![2]),
                version: 6,
            },
        );
        let mut want = state.clone();

        flatten_to_seeded_seq(&mut state, 1_007);

        want.sequence_num = 1_007;
        for section in want.sections.values_mut() {
            section.last_version = 1_007;
        }
        for field in want.fields.values_mut() {
            field.version = 1_007;
        }
        for sample in want.time_samples.values_mut() {
            sample.version = 1_007;
        }
        assert_eq!(state, want);
        assert_eq!(state.sections[&2].first_version, 5, "first_version kept");
    }
}
