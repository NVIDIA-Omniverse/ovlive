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

//! Core domain types: [`Section`], [`SectionKey`], [`Field`], [`TimeSample`],
//! [`FieldValue`] and an object's materialized state, [`ObjectStateOf`]. No
//! I/O, no async — pure data.
//!
//! # Two forms of an object's state
//!
//! [`ObjectStateOf`] is generic over the store that holds its value rows (its
//! fields and time samples), [`ValueRows`]; everything else — the header
//! scalars, the sections and the children lists — is the same in both forms.
//!
//! - [`ObjectState`] (= `ObjectStateOf<FullRows>`) holds every row whole:
//!   its value, size, set order and version. It is what every reader decodes
//!   and serves (catch-up, checkpoint, ext-GC), and what `CreateLayer` builds.
//!   Its time-sample times are canonical: a reader holds a stored `-0.0` row
//!   at `+0.0`, resolving a stored `-0.0` / `+0.0` pair by
//!   [`resolve_signed_zero_pair`] ([`time_samples_from_stored_rows`]), and a
//!   delta's times are canonical already (`oodf_ffi::parse_delta`).
//! - [`CompactState`] (= `ObjectStateOf<CompactRows>`) holds, per row, only
//!   its key and its external hash, if any: everything structural planning
//!   and the structural commit's pre-images read of a row. Field names are
//!   interned per state ([`NameInterner`]), and a time-sample row is keyed by
//!   its stored time bits, so `0.0` and `-0.0` are two rows, as they are two
//!   stored keys. That stays so for good: this build writes no `-0.0` row,
//!   but a legacy one (from an older build, during a rolling upgrade or after
//!   a rollback, until the reconciler's signed-zero merge folds it into
//!   `0.0`) is a real stored row a writer must delete and count as such. It
//!   is the structural write-planning cache's entry and the state a
//!   structural plan runs on.
//!
//! The two are distinct types, so a compact state can never reach a reader:
//! a reader takes [`ObjectState`], and handing it a [`CompactState`] is a type
//! error. [`ObjectState`] is a type alias rather than a defaulted type
//! parameter so that every `ObjectState { .. }` and `ObjectState::default()`
//! in expression position keeps naming the full form.
//!
//! The planning engine reaches the rows only through [`FieldRows`] and
//! [`SampleRows`], which expose what its value-row sites, the pre-images and
//! the cache's weight estimate use: insert a row, purge the rows of a set of
//! sections, drain every row, test a row's existence, read a row's external
//! hash, walk the rows with their hashes, and the store's housekeeping.

use ordered_float::OrderedFloat;
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::Arc;

/// Discriminated value: inline bytes or external hash reference.
#[derive(Debug, Clone, PartialEq)]
pub enum FieldValue {
    Inline(Vec<u8>),
    External {
        hash: [u8; 32],
        size: u64,
        ext_value_index: u32,
    },
}

/// A section in the hierarchical document tree.
#[derive(Debug, Clone, PartialEq)]
pub struct Section {
    pub section_id: u64,
    pub parent_id: u64,
    pub name: String,
    pub section_type: u8,
    pub section_order: u32,
    pub first_version: u64,
    pub last_version: u64,
}

/// The `(parent_id, name)` identity of a stored [`Section`] row — what a
/// by-name child lookup keys on.
///
/// Carried by `StorageMutation::SetSection` (`previous`) and
/// `StorageMutation::DeleteSection` to name the row each mutation replaces
/// or removes. No structural record carries them and neither commit path
/// writes anything from them; only the structural commit's plan fingerprint
/// reads them, as plan identity. The structural commit's section digests come
/// from the pre-plan state, never from these keys — a per-mutation key can
/// name a row an earlier mutation of the same delta wrote, not the stored one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SectionKey {
    pub parent_id: u64,
    pub name: String,
}

/// A field value attached to a section.
#[derive(Debug, Clone, PartialEq)]
pub struct Field {
    pub section_id: u64,
    pub key_name: String,
    pub value: FieldValue,
    pub set_order: u64,
    /// Raw per-delta authoring rank (`OodfSetField.set_order`) of the last
    /// LWW-winning write to this field. Distinct from the cumulative
    /// `set_order` above (which folds authoring rank into a running cursor);
    /// carried for the corrected catch-up snapshot builder.
    pub authoring_set_order: u64,
    pub version: u64,
}

/// A time-keyed sample value attached to a section.
#[derive(Debug, Clone, PartialEq)]
pub struct TimeSample {
    pub section_id: u64,
    pub time: OrderedFloat<f64>,
    pub value: FieldValue,
    pub version: u64,
}

/// Which row of a stored signed-zero pair survives: the pair is one section's
/// time-sample row at `+0.0` and its row at `-0.0`, two stored keys for one
/// USD time code. [`resolve_signed_zero_pair`] decides.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SignedZeroSurvivor {
    /// The `+0.0` row's sample survives.
    PositiveZero,
    /// The `-0.0` row's sample survives (stored, after a merge, at `+0.0`).
    NegativeZero,
}

/// The one rule that resolves a stored signed-zero pair — the same section's
/// time-sample rows at `+0.0` and at `-0.0` — shared by every full-form reader
/// ([`time_samples_from_stored_rows`]) and the reference-count reconciler's
/// merge (`storage::time_sample_merge`), so a merge changes nothing a reader
/// serves.
///
/// In order:
/// 1. the row with the higher `version` wins: it was written by the later
///    commit;
/// 2. on a version tie, a row holding a value beats an erase row (an inline
///    row with an empty value, which is how an erase is stored): one update
///    that erased at one zero and set at the other meant the set;
/// 3. on a further tie, the `+0.0` row wins.
///
/// Pure; reads only the two rows' `version` and `value`.
#[must_use]
pub fn resolve_signed_zero_pair(
    positive_zero: &TimeSample,
    negative_zero: &TimeSample,
) -> SignedZeroSurvivor {
    use std::cmp::Ordering;
    match negative_zero.version.cmp(&positive_zero.version) {
        Ordering::Greater => SignedZeroSurvivor::NegativeZero,
        Ordering::Less => SignedZeroSurvivor::PositiveZero,
        Ordering::Equal => {
            if is_erase_value(&positive_zero.value) && !is_erase_value(&negative_zero.value) {
                SignedZeroSurvivor::NegativeZero
            } else {
                SignedZeroSurvivor::PositiveZero
            }
        }
    }
}

/// Whether `value` is an erase: an inline value with no bytes. An external
/// value always names a blob, so it is never an erase.
fn is_erase_value(value: &FieldValue) -> bool {
    matches!(value, FieldValue::Inline(bytes) if bytes.is_empty())
}

/// Build the full form's time-sample map from an object's stored time-sample
/// rows, one [`TimeSample`] per stored row, each carrying its stored time.
///
/// Every time is canonical in the result: a `-0.0` row is held at `+0.0`, so
/// the full form never holds `-0.0`. When a section has both a `+0.0` row and
/// a `-0.0` row — a stored signed-zero pair, which only an older build or a
/// row not yet merged by the reconciler leaves — [`resolve_signed_zero_pair`]
/// picks the survivor, held at `+0.0` whichever row it came from, so which
/// row a reader serves does not depend on read order. Any other row is held
/// under its own key; a key given twice keeps the later row.
///
/// The decode step of every full-form reader
/// (`storage::diff_read::snapshot_to_object_state`).
pub fn time_samples_from_stored_rows(
    rows: impl IntoIterator<Item = TimeSample>,
) -> HashMap<(u64, OrderedFloat<f64>), TimeSample> {
    let rows = rows.into_iter();
    let mut held = HashMap::with_capacity(rows.size_hint().0);
    // Per section, its `+0.0` and `-0.0` rows, resolved once every row is in.
    let mut zeros: HashMap<u64, (Option<TimeSample>, Option<TimeSample>)> = HashMap::new();
    for sample in rows {
        let time = sample.time.into_inner();
        if time != 0.0 {
            held.insert((sample.section_id, sample.time), sample);
            continue;
        }
        let pair = zeros.entry(sample.section_id).or_default();
        if time.is_sign_negative() {
            pair.1 = Some(sample);
        } else {
            pair.0 = Some(sample);
        }
    }
    for (section_id, pair) in zeros {
        let mut survivor = match pair {
            (Some(positive), Some(negative)) => {
                match resolve_signed_zero_pair(&positive, &negative) {
                    SignedZeroSurvivor::PositiveZero => positive,
                    SignedZeroSurvivor::NegativeZero => negative,
                }
            }
            (Some(only), None) | (None, Some(only)) => only,
            (None, None) => continue,
        };
        survivor.time = OrderedFloat(0.0);
        held.insert((section_id, OrderedFloat(0.0)), survivor);
    }
    held
}

/// Materialized state of a layer object, with its value rows held by `R`
/// (module docs, "Two forms of an object's state").
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ObjectStateOf<R: ValueRows> {
    pub object_id: u64,
    pub sequence_num: u64,
    pub first_version: u64,
    pub source_format: String,
    pub next_set_order: u64,
    pub sections: HashMap<u64, Section>,
    pub children: HashMap<u64, Vec<u64>>,
    /// The field rows: for [`FullRows`], a map from `(section_id, key_name)`
    /// to the whole [`Field`].
    pub fields: R::Fields,
    /// The time-sample rows: for [`FullRows`], a map from `(section_id,
    /// time)` to the whole [`TimeSample`].
    pub time_samples: R::Samples,
}

/// The full form: every value row whole. What every reader decodes and serves.
pub type ObjectState = ObjectStateOf<FullRows>;

/// The compact form: per value row, its key and its external hash only. The
/// structural write-planning cache's entry and the state a structural plan
/// runs on; never handed to a reader.
pub type CompactState = ObjectStateOf<CompactRows>;

/// A store of an object's value rows: which field store and time-sample store
/// an [`ObjectStateOf`] holds. Implemented by the two markers [`FullRows`] and
/// [`CompactRows`].
pub trait ValueRows: Clone + fmt::Debug + Default + PartialEq + Send + Sync + 'static {
    /// The field rows.
    type Fields: FieldRows;
    /// The time-sample rows.
    type Samples: SampleRows;
}

/// [`ObjectState`]'s row store: today's two maps, holding every row whole.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FullRows;

impl ValueRows for FullRows {
    type Fields = HashMap<(u64, String), Field>;
    type Samples = HashMap<(u64, OrderedFloat<f64>), TimeSample>;
}

/// [`CompactState`]'s row store: per row, its key and its external hash.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CompactRows;

impl ValueRows for CompactRows {
    type Fields = CompactFields;
    type Samples = CompactSamples;
}

/// An object's field rows, as the planning engine, the structural commit's
/// pre-images and the cache's weight estimate reach them. A row is keyed by
/// `(section_id, key_name)`.
///
/// The methods are named apart from `HashMap`'s inherent ones where the two
/// would differ (`insert_row`, `drain_rows`), so a call on the full form's map
/// never silently picks the other.
pub trait FieldRows: Clone + fmt::Debug + Default + PartialEq + Send + Sync {
    /// Insert `field`, replacing the row with its key. The compact form keeps
    /// only the key and the external hash. Reports the row it replaced and
    /// whether the write interned a new field name ([`RowWrite`]).
    fn insert_row(&mut self, field: Field) -> RowWrite;

    /// Remove every row whose section is in `section_ids`, calling
    /// `on_removed(section_id, key_name)` once per removed row, in no
    /// particular order. The key is lent, so a caller that needs it owned
    /// clones it only when it uses it.
    fn purge_sections(
        &mut self,
        section_ids: &HashSet<u64>,
        mut on_removed: impl FnMut(u64, &str),
    ) {
        self.purge_sections_with(section_ids, |section_id, key_name, _, _| {
            on_removed(section_id, key_name);
        });
    }

    /// [`Self::purge_sections`], also lending each removed row's external
    /// hash (`None` for an inline row) and, when removing it released its
    /// interned field name (the last row using it went), the id the name
    /// held — always `None` on the full form, which interns nothing. The
    /// planning engine's writer (`domain::delta_write`) reports both to its
    /// hooks.
    fn purge_sections_with(
        &mut self,
        section_ids: &HashSet<u64>,
        on_removed: impl FnMut(u64, &str, Option<&[u8; 32]>, Option<NameId>),
    );

    /// Remove every row, calling `on_removed(section_id, key_name)` once per
    /// row, in no particular order. Leaves the store empty.
    fn drain_rows(&mut self, on_removed: impl FnMut(u64, &str));

    /// Move every row out, leaving an empty store whose tables keep their
    /// capacity and hasher, as [`Self::drain_rows`] leaves them: so inserts
    /// after it land where they would after a drain. A reset hands the
    /// removed rows to its hooks rather than dropping them one by one.
    fn take_all(&mut self) -> Self;

    /// Call `f(section_id, key_name)` once per row, in the order
    /// [`Self::drain_rows`] would hand the rows over and with the same
    /// arguments, without removing any.
    fn for_each_in_drain_order(&self, f: impl FnMut(u64, &str));

    /// Whether the row `(section_id, key_name)` is held.
    fn contains(&self, section_id: u64, key_name: &str) -> bool;

    /// The external hash the row `(section_id, key_name)` holds; `None` for
    /// an inline row or an absent one.
    fn ext_hash(&self, section_id: u64, key_name: &str) -> Option<[u8; 32]>;

    /// Call `f(section_id, key_name, external hash)` once per row, in no
    /// particular order.
    fn for_each(&self, f: impl FnMut(u64, &str, Option<&[u8; 32]>));

    /// The number of rows.
    fn len(&self) -> usize;

    /// Whether no row is held.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Release spare capacity. The compact form also drops the interned names
    /// no row uses any more.
    fn shrink_to_fit(&mut self);

    /// Set every external row's in-memory `ext_value_index` to 0, the value a
    /// stored row decodes to. A no-op on the compact form, which holds none.
    fn zero_ext_indexes(&mut self);

    /// The rows the row table takes before an insert call would grow it
    /// (`capacity() − len()`), for a hooked plan's rollback-rebuild check
    /// (`domain::plan_index::IndexedHooks`). Only the compact form, which
    /// hooked plans write, reports it; the full form reports `usize::MAX`.
    fn row_room(&self) -> usize {
        usize::MAX
    }

    /// The names the interner's name → id table takes before an insert call
    /// would grow it, as [`Self::row_room`]; `usize::MAX` for the full form,
    /// which interns nothing.
    fn name_room(&self) -> usize {
        usize::MAX
    }

    /// The row table's capacity (`HashMap::capacity`), which a hooked plan's
    /// removal reports to the plan indexes' capacity high-water
    /// (`domain::plan_index::PlanIndexes::note_capacity`). Only the compact
    /// form, which hooked plans write, reports it; the full form reports 0.
    fn row_capacity(&self) -> usize {
        0
    }

    /// The interner's name → id table's capacity, as [`Self::row_capacity`];
    /// 0 for the full form, which interns nothing.
    fn name_capacity(&self) -> usize {
        0
    }

    /// Whether a write of a row named `key_name` makes an insert call on the
    /// interner's name → id table: the compact form's name is not interned
    /// yet. Always `false` for the full form.
    fn interns(&self, _key_name: &str) -> bool {
        false
    }
}

/// An object's time-sample rows, as [`FieldRows`] is for its fields. A row is
/// keyed by `(section_id, time bits)`: the stored key spells the time by its
/// bit pattern, so `0.0` and `-0.0` name two rows.
///
/// The full form's map is keyed by `OrderedFloat`, whose equality conflates
/// `0.0` and `-0.0`; its methods compare the held sample's own time bits, so
/// a lookup by the other zero finds nothing, as the stored key would.
pub trait SampleRows: Clone + fmt::Debug + Default + PartialEq + Send + Sync {
    /// Insert `sample`, replacing the row with its key. The compact form keeps
    /// only the key (the sample's time bits) and the external hash. The full
    /// form keeps today's `OrderedFloat` key, so `0.0` replaces `-0.0`'s row
    /// and the reverse. Reports the replaced row's external hash: `None` when
    /// no row had the key, `Some(None)` for an inline one.
    fn insert_row(&mut self, sample: TimeSample) -> Option<Option<[u8; 32]>>;

    /// Remove every row whose section is in `section_ids`, calling
    /// `on_removed(section_id, time)` once per removed row, in no particular
    /// order. The full form passes its map key's time, the compact form the
    /// row's stored time.
    fn purge_sections(
        &mut self,
        section_ids: &HashSet<u64>,
        mut on_removed: impl FnMut(u64, OrderedFloat<f64>),
    ) {
        self.purge_sections_with(section_ids, |section_id, time, _| {
            on_removed(section_id, time);
        });
    }

    /// [`Self::purge_sections`], also lending each removed row's external
    /// hash (`None` for an inline row).
    fn purge_sections_with(
        &mut self,
        section_ids: &HashSet<u64>,
        on_removed: impl FnMut(u64, OrderedFloat<f64>, Option<&[u8; 32]>),
    );

    /// Remove every row, calling `on_removed(section_id, time)` once per row
    /// (the time as [`Self::purge_sections`] passes it), in no particular
    /// order. Leaves the store empty.
    fn drain_rows(&mut self, on_removed: impl FnMut(u64, OrderedFloat<f64>));

    /// As [`FieldRows::take_all`].
    fn take_all(&mut self) -> Self;

    /// As [`FieldRows::for_each_in_drain_order`]: `f(section_id, time)`, the
    /// time as [`Self::drain_rows`] passes it.
    fn for_each_in_drain_order(&self, f: impl FnMut(u64, OrderedFloat<f64>));

    /// Whether the row `(section_id, time_bits)` is held.
    fn contains(&self, section_id: u64, time_bits: u64) -> bool;

    /// The external hash the row `(section_id, time_bits)` holds; `None` for
    /// an inline row or an absent one.
    fn ext_hash(&self, section_id: u64, time_bits: u64) -> Option<[u8; 32]>;

    /// Call `f(section_id, time_bits, external hash)` once per row, in no
    /// particular order: the full form passes each held sample's own time
    /// bits.
    fn for_each(&self, f: impl FnMut(u64, u64, Option<&[u8; 32]>));

    /// The number of rows.
    fn len(&self) -> usize;

    /// Whether no row is held.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Release spare capacity.
    fn shrink_to_fit(&mut self);

    /// As [`FieldRows::zero_ext_indexes`].
    fn zero_ext_indexes(&mut self);

    /// As [`FieldRows::row_room`].
    fn row_room(&self) -> usize {
        usize::MAX
    }

    /// As [`FieldRows::row_capacity`].
    fn row_capacity(&self) -> usize {
        0
    }
}

/// The external hash `value` holds, if it is external.
fn external_hash(value: &FieldValue) -> Option<&[u8; 32]> {
    match value {
        FieldValue::External { hash, .. } => Some(hash),
        FieldValue::Inline(_) => None,
    }
}

fn zero_ext_value_index(value: &mut FieldValue) {
    if let FieldValue::External {
        ext_value_index, ..
    } = value
    {
        *ext_value_index = 0;
    }
}

/// What one row write replaced, as [`FieldRows::insert_row`] reports it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RowWrite {
    /// The replaced row's external hash: `None` when no row had the key,
    /// `Some(None)` for an inline one.
    pub replaced: Option<Option<[u8; 32]>>,
    /// Whether the write interned a field name no row used before (the
    /// compact form only; the full form interns nothing).
    pub name_interned: bool,
}

/// An empty map with `map`'s capacity and hasher: the shape `map.drain()` or
/// `map.clear()` leaves, so the next inserts land where they would there.
fn emptied<K, V>(map: &HashMap<K, V>) -> HashMap<K, V> {
    HashMap::with_capacity_and_hasher(map.capacity(), map.hasher().clone())
}

impl FieldRows for HashMap<(u64, String), Field> {
    fn insert_row(&mut self, field: Field) -> RowWrite {
        let replaced = self.insert((field.section_id, field.key_name.clone()), field);
        RowWrite {
            replaced: replaced.map(|old| external_hash(&old.value).copied()),
            name_interned: false,
        }
    }

    fn purge_sections_with(
        &mut self,
        section_ids: &HashSet<u64>,
        mut on_removed: impl FnMut(u64, &str, Option<&[u8; 32]>, Option<NameId>),
    ) {
        self.retain(|(section_id, key_name), field| {
            if section_ids.contains(section_id) {
                on_removed(*section_id, key_name, external_hash(&field.value), None);
                false
            } else {
                true
            }
        });
    }

    fn drain_rows(&mut self, mut on_removed: impl FnMut(u64, &str)) {
        for ((section_id, key_name), _) in self.drain() {
            on_removed(section_id, &key_name);
        }
    }

    fn take_all(&mut self) -> Self {
        let empty = emptied(self);
        std::mem::replace(self, empty)
    }

    fn for_each_in_drain_order(&self, mut f: impl FnMut(u64, &str)) {
        for (section_id, key_name) in self.keys() {
            f(*section_id, key_name);
        }
    }

    fn contains(&self, section_id: u64, key_name: &str) -> bool {
        self.contains_key(&(section_id, key_name.to_owned()))
    }

    fn ext_hash(&self, section_id: u64, key_name: &str) -> Option<[u8; 32]> {
        self.get(&(section_id, key_name.to_owned()))
            .and_then(|field| external_hash(&field.value))
            .copied()
    }

    fn for_each(&self, mut f: impl FnMut(u64, &str, Option<&[u8; 32]>)) {
        for ((section_id, key_name), field) in self {
            f(*section_id, key_name, external_hash(&field.value));
        }
    }

    fn len(&self) -> usize {
        HashMap::len(self)
    }

    fn shrink_to_fit(&mut self) {
        HashMap::shrink_to_fit(self);
    }

    fn zero_ext_indexes(&mut self) {
        for field in self.values_mut() {
            zero_ext_value_index(&mut field.value);
        }
    }
}

impl SampleRows for HashMap<(u64, OrderedFloat<f64>), TimeSample> {
    fn insert_row(&mut self, sample: TimeSample) -> Option<Option<[u8; 32]>> {
        self.insert((sample.section_id, sample.time), sample)
            .map(|old| external_hash(&old.value).copied())
    }

    fn purge_sections_with(
        &mut self,
        section_ids: &HashSet<u64>,
        mut on_removed: impl FnMut(u64, OrderedFloat<f64>, Option<&[u8; 32]>),
    ) {
        self.retain(|(section_id, time), sample| {
            if section_ids.contains(section_id) {
                on_removed(*section_id, *time, external_hash(&sample.value));
                false
            } else {
                true
            }
        });
    }

    fn drain_rows(&mut self, mut on_removed: impl FnMut(u64, OrderedFloat<f64>)) {
        for ((section_id, time), _) in self.drain() {
            on_removed(section_id, time);
        }
    }

    fn take_all(&mut self) -> Self {
        let empty = emptied(self);
        std::mem::replace(self, empty)
    }

    fn for_each_in_drain_order(&self, mut f: impl FnMut(u64, OrderedFloat<f64>)) {
        for (section_id, time) in self.keys() {
            f(*section_id, *time);
        }
    }

    fn contains(&self, section_id: u64, time_bits: u64) -> bool {
        self.get(&(section_id, OrderedFloat(f64::from_bits(time_bits))))
            .is_some_and(|held| held.time.into_inner().to_bits() == time_bits)
    }

    fn ext_hash(&self, section_id: u64, time_bits: u64) -> Option<[u8; 32]> {
        self.get(&(section_id, OrderedFloat(f64::from_bits(time_bits))))
            .filter(|held| held.time.into_inner().to_bits() == time_bits)
            .and_then(|held| external_hash(&held.value))
            .copied()
    }

    fn for_each(&self, mut f: impl FnMut(u64, u64, Option<&[u8; 32]>)) {
        for ((section_id, _), sample) in self {
            f(
                *section_id,
                sample.time.into_inner().to_bits(),
                external_hash(&sample.value),
            );
        }
    }

    fn len(&self) -> usize {
        HashMap::len(self)
    }

    fn shrink_to_fit(&mut self) {
        HashMap::shrink_to_fit(self);
    }

    fn zero_ext_indexes(&mut self) {
        for sample in self.values_mut() {
            zero_ext_value_index(&mut sample.value);
        }
    }
}

/// The id a [`NameInterner`] gives a field name.
pub type NameId = u32;

/// One state's field names, each stored once, with the number of rows that
/// use it.
///
/// A name is interned when the first row using it is inserted and released
/// when the last such row goes: its id is then free for the next new name,
/// and its string is dropped. [`Self::shrink_to_fit`] also trims the trailing
/// free ids and the spare capacity. Ids are meaningful only within the
/// interner that gave them, which is why [`CompactFields`] compares and
/// prints resolved names, never ids.
///
/// A plan's rollback (`domain::plan_journal`) puts every name back at the id
/// it held before the plan: it releases the names of the rows the plan
/// created, takes out a pre-plan row the plan left keyed by a name it
/// released and interned again (a new id), and re-acquires each released
/// name at its pre-plan id (`CompactFields::recreate_at_pre_ids`). So the
/// rollback leaves the same ids live and the same ids free as before the
/// plan, and never grows the table. Only the free list's order can differ,
/// which no weight reads, and the free slots the plan appended, which
/// compaction trims: compacted, the interner weighs what it weighed
/// compacted before the plan, byte for byte, however many attempts were
/// rolled back before.
#[derive(Clone, Default)]
pub struct NameInterner {
    /// Name → id. The `Arc` is the same allocation as the slot's, so each
    /// name's bytes are held once.
    ids: HashMap<Arc<str>, NameId>,
    /// Id → name and use count; a free id's slot holds no name.
    slots: Vec<NameSlot>,
    /// The free ids, reused before the table grows.
    free: Vec<NameId>,
}

#[derive(Clone, Default)]
struct NameSlot {
    name: Option<Arc<str>>,
    uses: u64,
}

impl NameInterner {
    /// The id of `name`, interning it if no row uses it yet, with its use
    /// count raised by one; `true` when this call interned it.
    fn acquire(&mut self, name: &str) -> (NameId, bool) {
        if let Some(&id) = self.ids.get(name) {
            self.slots[id as usize].uses += 1;
            return (id, false);
        }
        let name: Arc<str> = Arc::from(name);
        let id = match self.free.pop() {
            Some(id) => id,
            None => {
                let id = NameId::try_from(self.slots.len())
                    .expect("more distinct field names in one object than a u32 id can number");
                self.slots.push(NameSlot::default());
                id
            }
        };
        self.slots[id as usize] = NameSlot {
            name: Some(Arc::clone(&name)),
            uses: 1,
        };
        self.ids.insert(name, id);
        (id, true)
    }

    /// Lower `id`'s use count by one, freeing the id and dropping its name
    /// when no row uses it any more; `true` when this call dropped it.
    fn release(&mut self, id: NameId) -> bool {
        let slot = &mut self.slots[id as usize];
        slot.uses -= 1;
        if slot.uses == 0 {
            if let Some(name) = slot.name.take() {
                self.ids.remove(&name);
            }
            self.free.push(id);
            return true;
        }
        false
    }

    /// Intern `name` at `id`, a free slot, with a use count of one: a plan's
    /// rollback putting a name the plan released back at the id it held
    /// before the plan. The free list is left as it is, still holding `id`,
    /// until the caller takes the ids it filled off it in one pass
    /// ([`Self::claim_free`]); nothing may acquire in between. `false`,
    /// writing nothing, when `name` is interned or `id` is not a free slot.
    fn occupy(&mut self, name: &str, id: NameId) -> bool {
        if self.ids.contains_key(name) {
            return false;
        }
        let Some(slot) = self.slots.get_mut(id as usize) else {
            return false;
        };
        if slot.name.is_some() {
            return false;
        }
        let name: Arc<str> = Arc::from(name);
        *slot = NameSlot {
            name: Some(Arc::clone(&name)),
            uses: 1,
        };
        self.ids.insert(name, id);
        true
    }

    /// Take `claimed`, the ids [`Self::occupy`] filled, off the free list.
    /// Searched from the end and stopping at the last one found: each was
    /// freed by the plan or its rollback, so it sits among the entries
    /// pushed since the plan began — O(the plan's and the rollback's name
    /// releases), not O(the free list).
    fn claim_free(&mut self, claimed: &[NameId]) {
        if claimed.is_empty() {
            return;
        }
        let mut pending: HashSet<NameId> = claimed.iter().copied().collect();
        let mut at = self.free.len();
        while !pending.is_empty() && at > 0 {
            at -= 1;
            if pending.remove(&self.free[at]) {
                // The entry swapped in from the end was already passed over.
                self.free.swap_remove(at);
            }
        }
        debug_assert!(
            pending.is_empty(),
            "an occupied id was not on the free list: {pending:?}"
        );
    }

    /// Move every name out, leaving an empty interner whose tables keep their
    /// capacity (and the name table its hasher), as [`Self::clear`] leaves
    /// them.
    fn take_all(&mut self) -> Self {
        let empty = Self {
            ids: emptied(&self.ids),
            slots: Vec::with_capacity(self.slots.capacity()),
            free: Vec::with_capacity(self.free.capacity()),
        };
        std::mem::replace(self, empty)
    }

    /// The number of rows using `id`; 0 for a free id.
    fn uses(&self, id: NameId) -> u64 {
        self.slots.get(id as usize).map_or(0, |slot| slot.uses)
    }

    /// The id of `name`, if a row uses it.
    pub fn id_of(&self, name: &str) -> Option<NameId> {
        self.ids.get(name).copied()
    }

    /// The name `id` stands for. An id no row uses resolves to `""`.
    pub fn name(&self, id: NameId) -> &str {
        self.slots
            .get(id as usize)
            .and_then(|slot| slot.name.as_deref())
            .unwrap_or("")
    }

    /// The interned names, in no particular order.
    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.ids.keys().map(|name| &**name)
    }

    /// One entry of the name → id table, for a weight estimate.
    pub const ID_ENTRY_BYTES: usize = std::mem::size_of::<(Arc<str>, NameId)>();

    /// One slot of the id → name table, for a weight estimate.
    pub const SLOT_BYTES: usize = std::mem::size_of::<NameSlot>();

    /// The heap bytes of one interned name's `Arc<str>` allocation of
    /// `name_len` bytes: the two reference counts, then the bytes.
    pub fn name_allocation_bytes(name_len: usize) -> usize {
        2 * std::mem::size_of::<usize>() + name_len
    }

    /// The name → id table's capacity, in entries.
    pub fn ids_capacity(&self) -> usize {
        self.ids.capacity()
    }

    /// The id → name table's capacity, in slots.
    pub fn slots_capacity(&self) -> usize {
        self.slots.capacity()
    }

    /// The free-id list's capacity, in ids.
    pub fn free_capacity(&self) -> usize {
        self.free.capacity()
    }

    /// The free-id list's length, in ids.
    pub fn free_len(&self) -> usize {
        self.free.len()
    }

    /// The number of interned names.
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    /// Whether no name is interned.
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    fn clear(&mut self) {
        self.ids.clear();
        self.slots.clear();
        self.free.clear();
    }

    /// A clone with room to intern `names` more names without growing: the
    /// name → id table at its length plus `names` when that is above its
    /// capacity, and the id → name table at its length plus `names`.
    fn clone_with_room(&self, names: usize) -> Self {
        let mut slots = Vec::with_capacity(self.slots.len().saturating_add(names));
        slots.extend(self.slots.iter().cloned());
        Self {
            ids: clone_with_room(&self.ids, names),
            slots,
            free: self.free.clone(),
        }
    }

    /// The free ids, sorted ascending: the set the free list holds, without
    /// its order, which a plan's rollback does not restore. Test-support
    /// only, for tests comparing two interners' free ids.
    #[cfg(any(test, feature = "test-support"))]
    pub fn free_ids_sorted_for_test(&self) -> Vec<NameId> {
        let mut free = self.free.clone();
        free.sort_unstable();
        free
    }

    /// Trim the free ids at the end of the table and release spare capacity.
    fn shrink_to_fit(&mut self) {
        while self.slots.last().is_some_and(|slot| slot.name.is_none()) {
            self.slots.pop();
        }
        let len = self.slots.len();
        self.free.retain(|&id| (id as usize) < len);
        self.ids.shrink_to_fit();
        self.slots.shrink_to_fit();
        self.free.shrink_to_fit();
    }
}

/// The external hash of a compact row; `None` for an inline row. Boxed, so
/// an inline row's value is one null pointer.
type CompactHash = Option<Box<[u8; 32]>>;

/// How many times more a probe-arm step of the compact field purge may cost
/// than a scanned row before the probe arm stops paying
/// ([`CompactFields::purges_by_probe`]).
///
/// The compact field purge ([`FieldRows::purge_sections_with`] on
/// [`CompactFields`]) finds a purged section's rows one of two ways. The
/// probe arm walks the interner's `N` id slots, live and free, for every
/// purged section, looking up `(section, id)` at each live one: `D × N`
/// steps for `D` sections, a count set by the edit, not the layer. Each step
/// is a random lookup into the row table, so its cost rises as the table
/// outgrows the CPU caches, then plateaus near one DRAM miss (below): the
/// probe arm costs the edit's steps times a bounded per-step cost, not
/// O(F). The scan arm is a `retain` over every row, a membership test of
/// the purge set each: `F` rows, the layer's size. The probe arm runs when
/// `D × N × R ≤ F`, so a purge takes O(min(D × N, F)) steps, however many
/// distinct field names a client wrote.
///
/// `R = max(1, ⌈median(ns per probe-arm (section, slot) step) ÷
/// median(ns per scanned row)⌉)`, the largest over the shapes measured: the
/// dev shape at 60k sections (7 names, 224,820 rows) and a many-names shape
/// (60k sections, 256 names, 238,868 rows), each at 32 and 1,024 purged
/// sections, both arms forced on clones of one store, the first arm
/// alternating per rep, 100 reps after 5 warm-ups, release build, aarch64,
/// measured with the internal flatness harness. Measured ratios:
///
/// | Shape | D | Probe ns / step | Scan ns / row | Ratio |
/// |---|---|---|---|---|
/// | dev, 7 names | 32 | 71.05 | 10.83 | **6.56** |
/// | dev, 7 names | 1,024 | 46.72 | 12.49 | 3.74 |
/// | 256 names | 32 | 21.31 | 11.28 | 1.89 |
/// | 256 names | 1,024 | 15.42 | 13.07 | 1.18 |
///
/// So `R = ⌈6.56⌉ = 7`. Record:
/// `docs-internal/validation/structural-delete-purge-b-k39-20261010/raw/purge-ratio.json`.
///
/// By layer size (`D` = 32, 100 reps after 10 warm-ups, same build and
/// platform), the per-step ratio rises with the field table while a scanned
/// row stays at 10.4–11.5 ns:
///
/// | Sections | Dev rows | Dev ns/step (224) | Ratio | 256 names ns/step (8,192) | Ratio |
/// |---|---|---|---|---|---|
/// | 10k | 37,404 | 29.8 | 2.75 | 14.5 | 1.32 |
/// | 60k | 224,820 | 61.9 | 5.92 | 19.8 | 1.81 |
/// | 190k | 711,901 | 156.8 | 15.0 | 38.8 | 3.69 |
/// | 500k | 1,873,322 | 248.5 | 21.6 | 70.0 | 6.21 |
/// | 1M | 3,746,667 | 268.0 | 24.3 | 80.2 | 7.15 |
///
/// A step plateaus near one DRAM miss, about 270 ns on the dev shape at
/// 500k–1M (1.08× from 500k to 1M), so the probe arm's cost has a ceiling
/// per step. These purges are far below the crossover, where the probe arm
/// is far cheaper in absolute terms: 60 µs against 41 ms at 1M sections on
/// the dev shape. Records: `raw/purge-ratio-by-size.json` (10k, 60k, 190k)
/// and `raw/purge-ratio-plateau.json` (500k, 1M), beside the record above.
///
/// At the crossover itself (`D × N × 7 ≈ F`, 190k sections, 30 reps after 5
/// warm-ups) the probe arm measured below the scan on both shapes: dev
/// (`F` = 711,901, `D` = 14,500) 5.48 ms against 9.73 ms, 0.56×; 256 names
/// (`F` = 755,830, `D` = 420) 2.94 ms against 21.6 ms, 0.14×. So `R` = 7 is
/// conservative there: the arms' true cost crossover lies past the switch
/// point, and far past it the scan wins (256 names, `D` = 7,000: probe
/// 1.72× the scan). Record: `raw/purge-ratio-crossover.json`.
///
/// Every figure here was measured on aarch64 only; x86_64, the CI and
/// release target, is unmeasured, and an x86_64 run of the same arms is
/// owed by the time-sample purge follow-up. A wrong `R` moves only the
/// crossover between the arms: the probe arm still runs only where
/// `D × N × R ≤ F`, so the purge stays within a constant factor of the scan
/// either way.
pub const PURGE_PROBE_RATIO: usize = 7;

/// A test-support seam on the compact field purge's arm
/// ([`PURGE_PROBE_RATIO`]): force the probe or the scan arm on this thread,
/// and count what each purge did. Compiled out of every other build, where
/// [`CompactFields::purges_by_probe`] alone picks the arm.
///
/// The forced arm is never paused, unlike `domain::delta::failpoint`: a
/// test-support oracle that plans a second time beside the plan under test
/// (the structural cache's shadow check, a proptest's copy oracle) runs on
/// the same thread and takes the same arm, so the two emit their purge runs
/// in the same order. A purge of an empty store, or of no sections, returns
/// before either arm and counts nothing.
#[cfg(any(test, feature = "test-support"))]
pub mod purge_arm {
    use std::cell::Cell;

    /// The compact field purge's two arms ([`super::PURGE_PROBE_RATIO`]).
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum PurgeArm {
        /// A lookup per purged section and live interned name.
        Probe,
        /// A `retain` over every row.
        Scan,
    }

    /// What the compact field purges on this thread did since the last
    /// [`reset`].
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
    pub struct PurgeCounts {
        /// Purges that ran the probe arm.
        pub probe_purges: u64,
        /// Purges that ran the scan arm.
        pub scan_purges: u64,
        /// The steps the purges took: on the probe arm one per
        /// `(section, id)` lookup (each purged section times each name live
        /// when it is reached), on the scan arm one per row the `retain`
        /// visits (every row).
        pub steps_visited: u64,
        /// Rows removed.
        pub rows_removed: u64,
        /// Interned names released (the last row using the name went).
        pub names_released: u64,
    }

    thread_local! {
        static FORCED: Cell<Option<PurgeArm>> = const { Cell::new(None) };
        static COUNTS: Cell<PurgeCounts> = const {
            Cell::new(PurgeCounts {
                probe_purges: 0,
                scan_purges: 0,
                steps_visited: 0,
                rows_removed: 0,
                names_released: 0,
            })
        };
    }

    /// Restores the arm forced before [`force`] when dropped, on unwind too.
    #[must_use = "the forced arm is restored when the guard drops"]
    pub struct ForceGuard {
        previous: Option<PurgeArm>,
    }

    impl Drop for ForceGuard {
        fn drop(&mut self) {
            FORCED.with(|f| f.set(self.previous));
        }
    }

    /// Force `arm` on this thread's compact field purges until the guard
    /// drops (`None`: the predicate picks, as in production).
    pub fn force(arm: Option<PurgeArm>) -> ForceGuard {
        ForceGuard {
            previous: FORCED.with(|f| f.replace(arm)),
        }
    }

    /// The arm forced on this thread, if any.
    pub fn forced() -> Option<PurgeArm> {
        FORCED.with(Cell::get)
    }

    /// This thread's counts since the last [`reset`].
    pub fn counts() -> PurgeCounts {
        COUNTS.with(Cell::get)
    }

    /// Zero this thread's counts. The forced arm is left as it is.
    pub fn reset() {
        COUNTS.with(|c| c.set(PurgeCounts::default()));
    }

    /// Count one purge on `arm`.
    pub(super) fn record(arm: PurgeArm, steps: u64, rows_removed: u64, names_released: u64) {
        COUNTS.with(|c| {
            let mut counts = c.get();
            match arm {
                PurgeArm::Probe => counts.probe_purges += 1,
                PurgeArm::Scan => counts.scan_purges += 1,
            }
            counts.steps_visited += steps;
            counts.rows_removed += rows_removed;
            counts.names_released += names_released;
            c.set(counts);
        });
    }
}

/// A field row a plan's rollback re-creates: `(section_id, key_name, hash)`,
/// `hash` `None` for an inline row.
pub(crate) type FieldRowImage = (u64, Box<str>, Option<[u8; 32]>);

/// A [`FieldRowImage`] with its name's pre-plan id:
/// `(section_id, key_name, pre_id, hash)`
/// ([`CompactFields::recreate_at_pre_ids`]).
pub(crate) type RowAtPreId = (u64, Box<str>, NameId, Option<[u8; 32]>);

/// The compact form's field rows: `(section_id, name id) → external hash`,
/// the names held by a per-state [`NameInterner`].
///
/// `PartialEq` and `Debug` are semantic: two stores are equal when they hold
/// the same `(section_id, name)` rows with the same hashes, whatever ids
/// their interners gave the names, and a store prints resolved names.
#[derive(Clone, Default)]
pub struct CompactFields {
    names: NameInterner,
    rows: HashMap<(u64, NameId), CompactHash>,
}

impl CompactFields {
    /// An empty store with room for `rows` rows.
    pub fn with_capacity(rows: usize) -> Self {
        Self {
            names: NameInterner::default(),
            rows: HashMap::with_capacity(rows),
        }
    }

    /// Insert the row `(section_id, key_name)` holding `hash` (`None` for an
    /// inline row), replacing the row with that key. Reports the replaced
    /// row and whether the name was interned by this write ([`RowWrite`]).
    pub fn insert(&mut self, section_id: u64, key_name: &str, hash: Option<[u8; 32]>) -> RowWrite {
        let (id, name_interned) = self.names.acquire(key_name);
        let replaced = self.rows.insert((section_id, id), hash.map(Box::new));
        if replaced.is_some() {
            // The replaced row already held a use of the name, so this
            // release never drops it.
            self.names.release(id);
        }
        RowWrite {
            replaced: replaced.map(|old| old.map(|hash| *hash)),
            name_interned,
        }
    }

    /// Remove the row `(section_id, key_name)`: its external hash (`None` for
    /// an inline row) and whether removing it released its interned name, or
    /// `None` when no row had the key. A plan's rollback removes through it
    /// a row its plan inserted, and a pre-plan row the plan left keyed by a
    /// name it released and interned again, at a new id.
    pub fn remove(&mut self, section_id: u64, key_name: &str) -> Option<(Option<[u8; 32]>, bool)> {
        let id = self.names.id_of(key_name)?;
        let hash = self.rows.remove(&(section_id, id))?;
        let released = self.names.release(id);
        Some((hash.map(|hash| *hash), released))
    }

    /// Put `hash` into the existing row `(section_id, key_name)` in place,
    /// returning the hash it replaced (`Some(None)` for an inline row), or
    /// `None`, writing nothing, when no row has the key. No insert call: the
    /// row keeps its slot and its name id, and the interner is untouched. A
    /// plan's rollback restores a row the plan overwrote through it, so the
    /// restore can never grow the table (std's `HashMap::insert` reserves a
    /// slot before it looks the key up, overwrite or not).
    pub fn restore_hash_in_place(
        &mut self,
        section_id: u64,
        key_name: &str,
        hash: Option<[u8; 32]>,
    ) -> Option<Option<[u8; 32]>> {
        let id = self.names.id_of(key_name)?;
        let slot = self.rows.get_mut(&(section_id, id))?;
        let old = std::mem::replace(slot, hash.map(Box::new));
        Some(old.map(|hash| *hash))
    }

    /// Re-create rows a plan removed whose names the plan released, each
    /// `(section_id, key_name, pre_id, hash)` keyed by its name's pre-plan id
    /// `pre_id`: a name no row uses is interned at that id, which must be
    /// free, and the ids so filled are taken off the free list once all rows
    /// are in (`NameInterner::claim_free`). A plan's rollback calls it after
    /// releasing every name the plan's own rows used, so each pre-plan id is
    /// free again (`domain::plan_journal`). `interned(key_name)` is called
    /// for each name this interns. Answers the rows it could not key that
    /// way — a name interned at another id, or an id not free, which that
    /// order rules out — for the caller to re-create by [`Self::insert`].
    pub(crate) fn recreate_at_pre_ids(
        &mut self,
        rows: Vec<RowAtPreId>,
        mut interned: impl FnMut(&str),
    ) -> Vec<FieldRowImage> {
        let mut claimed = Vec::new();
        let mut rest = Vec::new();
        for (section_id, key_name, pre_id, hash) in rows {
            match self.names.id_of(&key_name) {
                // A row before this one re-acquired the name.
                Some(id) if id == pre_id => self.names.slots[id as usize].uses += 1,
                Some(_) => {
                    rest.push((section_id, key_name, hash));
                    continue;
                }
                None => {
                    if !self.names.occupy(&key_name, pre_id) {
                        rest.push((section_id, key_name, hash));
                        continue;
                    }
                    claimed.push(pre_id);
                    interned(&key_name);
                }
            }
            if self
                .rows
                .insert((section_id, pre_id), hash.map(Box::new))
                .is_some()
            {
                // The replaced row already held a use of the name.
                self.names.release(pre_id);
            }
        }
        self.names.claim_free(&claimed);
        rest
    }

    /// The row map's room: the rows it takes before its next insert call
    /// would grow it (`capacity() − len()`).
    pub fn row_room(&self) -> usize {
        self.rows.capacity().saturating_sub(self.rows.len())
    }

    /// Rebuild the row map at a capacity of `len` rows ([`rebuild_map_at`]).
    pub fn rebuild_rows_at(&mut self, len: usize) {
        rebuild_map_at(&mut self.rows, len);
    }

    /// The interner's name → id table's room ([`Self::row_room`]'s rule).
    pub fn name_room(&self) -> usize {
        self.names
            .ids
            .capacity()
            .saturating_sub(self.names.ids.len())
    }

    /// Rebuild the interner's name → id table at a capacity of `len` names
    /// ([`rebuild_map_at`]).
    pub fn rebuild_names_at(&mut self, len: usize) {
        rebuild_map_at(&mut self.names.ids, len);
    }

    /// The interner holding this store's names.
    pub fn names(&self) -> &NameInterner {
        &self.names
    }

    /// Whether a purge of `purge_ids` sections runs the probe arm
    /// ([`PURGE_PROBE_RATIO`]): `purge_ids × id slots × R ≤ rows`, saturating.
    /// The id slots count the free ids as well as the live ones, so an
    /// interner left full of freed ids cannot make the probe arm cost more
    /// than the scan. It reads counts only — never a table's capacity, its
    /// buckets or its tombstones — so a given store and purge set take the
    /// same arm whatever the hasher seed or the platform's hash-group width.
    pub fn purges_by_probe(&self, purge_ids: usize) -> bool {
        purge_ids
            .saturating_mul(self.names.slots.len())
            .saturating_mul(PURGE_PROBE_RATIO)
            <= self.rows.len()
    }

    /// One slot of the row map, for a weight estimate.
    pub const ROW_SLOT_BYTES: usize = std::mem::size_of::<((u64, NameId), CompactHash)>();

    /// The heap bytes of one external row's boxed hash (time samples' too).
    pub const HASH_BYTES: usize = std::mem::size_of::<[u8; 32]>();

    /// The row map's capacity, in rows.
    pub fn capacity(&self) -> usize {
        self.rows.capacity()
    }

    /// The number of rows holding an external hash.
    pub fn external_rows(&self) -> usize {
        self.rows.values().filter(|hash| hash.is_some()).count()
    }

    /// A clone with room for `rows` more rows and `names` more interned
    /// names without growing any table ([`clone_with_room`]): the
    /// test-support plan copy (`storage::structural_cache::CheckedOut::plan_state`),
    /// sized for the delta's own field rows. The cached route plans in place.
    pub fn clone_with_room(&self, rows: usize, names: usize) -> Self {
        Self {
            names: self.names.clone_with_room(names),
            rows: clone_with_room(&self.rows, rows),
        }
    }
}

impl FieldRows for CompactFields {
    fn insert_row(&mut self, field: Field) -> RowWrite {
        self.insert(
            field.section_id,
            &field.key_name,
            external_hash(&field.value).copied(),
        )
    }

    /// Two arms, picked by [`CompactFields::purges_by_probe`]
    /// ([`PURGE_PROBE_RATIO`]); both call `on_removed` once per removed row,
    /// lending the name before the release that can drop it.
    ///
    /// - **Probe**, O(D × N) for `D` purged sections and `N` id slots, live
    ///   and free: for each
    ///   purged section in ascending id order, and each live interned id in
    ///   ascending order, remove the row `(section, id)` if there is one.
    ///   The order is a function of the store's contents and the set's
    ///   members, never of a hash table's layout or the set's hasher: so a
    ///   clone of the store (which keeps the interner's ids) purged by an
    ///   equal set emits the same run in the same order, and the structural
    ///   cache's shadow check and the in-place oracles compare it exactly.
    /// - **Scan**, O(F) for `F` rows: a `retain` over every row, in the row
    ///   table's hash order — also the same on a clone, which keeps the
    ///   table's layout.
    ///
    /// Either way the purge only removes, so no table's room falls while it
    /// runs. Which arm ran is observable only through the order of the run
    /// and the free list's order (the arms release names in different
    /// orders), and so the id that any name interned later takes — and what
    /// that changes later; never through the rows, the released names' ids,
    /// the live and free id sets, or the tables' layouts after the purge. An
    /// erase leaves a tombstone exactly when the run of non-empty buckets
    /// through its slot (counted up to a hash group's width each side) is at
    /// least a group long. A tombstone leaves every run as it was, and an
    /// empty bucket arises only inside a run already shorter than a group,
    /// so within a pass that only removes, which slots end as tombstones does
    /// not depend on the order of the erases.
    fn purge_sections_with(
        &mut self,
        section_ids: &HashSet<u64>,
        mut on_removed: impl FnMut(u64, &str, Option<&[u8; 32]>, Option<NameId>),
    ) {
        if self.rows.is_empty() || section_ids.is_empty() {
            return;
        }
        let probe = self.purges_by_probe(section_ids.len());
        #[cfg(any(test, feature = "test-support"))]
        let probe = match purge_arm::forced() {
            Some(arm) => arm == purge_arm::PurgeArm::Probe,
            None => probe,
        };
        #[cfg(any(test, feature = "test-support"))]
        let (mut steps, mut removed, mut released_names) = (0u64, 0u64, 0u64);
        let Self { names, rows } = self;
        if probe {
            let mut sections: Vec<u64> = section_ids.iter().copied().collect();
            sections.sort_unstable();
            // Releases free slots but never remove one, so the slot count
            // holds for the whole purge.
            let slots = names.slots.len();
            for section_id in sections {
                for slot in 0..slots {
                    if names.slots[slot].name.is_none() {
                        continue;
                    }
                    let id = NameId::try_from(slot).expect("an interned id fits a NameId");
                    #[cfg(any(test, feature = "test-support"))]
                    {
                        steps += 1;
                    }
                    let Some(hash) = rows.remove(&(section_id, id)) else {
                        continue;
                    };
                    // Lent before the release, which can drop the name.
                    let released = (names.uses(id) == 1).then_some(id);
                    on_removed(section_id, names.name(id), hash.as_deref(), released);
                    names.release(id);
                    #[cfg(any(test, feature = "test-support"))]
                    {
                        removed += 1;
                        released_names += u64::from(released.is_some());
                    }
                }
            }
        } else {
            #[cfg(any(test, feature = "test-support"))]
            {
                steps = rows.len() as u64;
            }
            rows.retain(|&(section_id, id), hash| {
                if section_ids.contains(&section_id) {
                    // Lent before the release, which can drop the name.
                    let released = (names.uses(id) == 1).then_some(id);
                    on_removed(section_id, names.name(id), hash.as_deref(), released);
                    names.release(id);
                    #[cfg(any(test, feature = "test-support"))]
                    {
                        removed += 1;
                        released_names += u64::from(released.is_some());
                    }
                    false
                } else {
                    true
                }
            });
        }
        #[cfg(any(test, feature = "test-support"))]
        purge_arm::record(
            if probe {
                purge_arm::PurgeArm::Probe
            } else {
                purge_arm::PurgeArm::Scan
            },
            steps,
            removed,
            released_names,
        );
    }

    fn drain_rows(&mut self, mut on_removed: impl FnMut(u64, &str)) {
        let Self { names, rows } = self;
        for ((section_id, id), _) in rows.drain() {
            on_removed(section_id, names.name(id));
        }
        names.clear();
    }

    fn take_all(&mut self) -> Self {
        let rows = emptied(&self.rows);
        Self {
            names: self.names.take_all(),
            rows: std::mem::replace(&mut self.rows, rows),
        }
    }

    fn for_each_in_drain_order(&self, mut f: impl FnMut(u64, &str)) {
        for &(section_id, id) in self.rows.keys() {
            f(section_id, self.names.name(id));
        }
    }

    fn contains(&self, section_id: u64, key_name: &str) -> bool {
        self.names
            .id_of(key_name)
            .is_some_and(|id| self.rows.contains_key(&(section_id, id)))
    }

    fn ext_hash(&self, section_id: u64, key_name: &str) -> Option<[u8; 32]> {
        let id = self.names.id_of(key_name)?;
        self.rows.get(&(section_id, id))?.as_deref().copied()
    }

    fn for_each(&self, mut f: impl FnMut(u64, &str, Option<&[u8; 32]>)) {
        for (&(section_id, id), hash) in &self.rows {
            f(section_id, self.names.name(id), hash.as_deref());
        }
    }

    fn len(&self) -> usize {
        self.rows.len()
    }

    fn shrink_to_fit(&mut self) {
        self.rows.shrink_to_fit();
        self.names.shrink_to_fit();
    }

    fn zero_ext_indexes(&mut self) {}

    fn row_room(&self) -> usize {
        CompactFields::row_room(self)
    }

    fn name_room(&self) -> usize {
        CompactFields::name_room(self)
    }

    fn row_capacity(&self) -> usize {
        self.rows.capacity()
    }

    fn name_capacity(&self) -> usize {
        self.names.ids.capacity()
    }

    fn interns(&self, key_name: &str) -> bool {
        self.names.id_of(key_name).is_none()
    }
}

impl PartialEq for CompactFields {
    fn eq(&self, other: &Self) -> bool {
        self.rows.len() == other.rows.len()
            && self.rows.iter().all(|(&(section_id, id), hash)| {
                other
                    .names
                    .id_of(self.names.name(id))
                    .and_then(|other_id| other.rows.get(&(section_id, other_id)))
                    .is_some_and(|other_hash| other_hash == hash)
            })
    }
}

impl fmt::Debug for CompactFields {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut rows: Vec<((u64, &str), HashDebug<'_>)> = self
            .rows
            .iter()
            .map(|(&(section_id, id), hash)| {
                (
                    (section_id, self.names.name(id)),
                    HashDebug(hash.as_deref()),
                )
            })
            .collect();
        rows.sort_unstable_by(|a, b| a.0.cmp(&b.0));
        f.debug_map().entries(rows).finish()
    }
}

/// The compact form's time-sample rows: `(section_id, stored time bits) →
/// external hash`.
///
/// Keyed by the stored bits, not the time's value, permanently: a legacy
/// `-0.0` row beside a `+0.0` row of the same section is two stored keys, and
/// a writer that collapsed them would delete, or count in `obj:ext`, one of
/// the two only (module docs, "Two forms of an object's state").
#[derive(Clone, Default, PartialEq)]
pub struct CompactSamples {
    rows: HashMap<(u64, u64), CompactHash>,
}

impl CompactSamples {
    /// An empty store with room for `rows` rows.
    pub fn with_capacity(rows: usize) -> Self {
        Self {
            rows: HashMap::with_capacity(rows),
        }
    }

    /// Insert the row `(section_id, time_bits)` holding `hash` (`None` for an
    /// inline row), replacing the row with that key.
    /// Reports the replaced row's external hash: `None` when no row had the
    /// key, `Some(None)` for an inline one.
    pub fn insert(
        &mut self,
        section_id: u64,
        time_bits: u64,
        hash: Option<[u8; 32]>,
    ) -> Option<Option<[u8; 32]>> {
        self.rows
            .insert((section_id, time_bits), hash.map(Box::new))
            .map(|old| old.map(|hash| *hash))
    }

    /// Remove the row `(section_id, time_bits)`: its external hash (`None`
    /// for an inline row), or `None` when no row had the key. A plan's
    /// rollback removes a row its plan inserted through it.
    pub fn remove(&mut self, section_id: u64, time_bits: u64) -> Option<Option<[u8; 32]>> {
        self.rows
            .remove(&(section_id, time_bits))
            .map(|old| old.map(|hash| *hash))
    }

    /// Put `hash` into the existing row `(section_id, time_bits)` in place,
    /// returning the hash it replaced, or `None`, writing nothing, when no
    /// row has the key: [`CompactFields::restore_hash_in_place`]'s rule.
    pub fn restore_hash_in_place(
        &mut self,
        section_id: u64,
        time_bits: u64,
        hash: Option<[u8; 32]>,
    ) -> Option<Option<[u8; 32]>> {
        let slot = self.rows.get_mut(&(section_id, time_bits))?;
        let old = std::mem::replace(slot, hash.map(Box::new));
        Some(old.map(|hash| *hash))
    }

    /// The row map's room ([`CompactFields::row_room`]'s rule).
    pub fn row_room(&self) -> usize {
        self.rows.capacity().saturating_sub(self.rows.len())
    }

    /// Rebuild the row map at a capacity of `len` rows ([`rebuild_map_at`]).
    pub fn rebuild_rows_at(&mut self, len: usize) {
        rebuild_map_at(&mut self.rows, len);
    }

    /// One slot of the row map, for a weight estimate.
    pub const ROW_SLOT_BYTES: usize = std::mem::size_of::<((u64, u64), CompactHash)>();

    /// The row map's capacity, in rows.
    pub fn capacity(&self) -> usize {
        self.rows.capacity()
    }

    /// The number of rows holding an external hash.
    pub fn external_rows(&self) -> usize {
        self.rows.values().filter(|hash| hash.is_some()).count()
    }

    /// A clone with room for `rows` more rows without growing
    /// ([`clone_with_room`]): the test-support plan copy
    /// (`storage::structural_cache::CheckedOut::plan_state`), sized for the
    /// delta's own time samples. The cached route plans in place.
    pub fn clone_with_room(&self, rows: usize) -> Self {
        Self {
            rows: clone_with_room(&self.rows, rows),
        }
    }
}

/// Rebuild `map` in place at a capacity of `len` entries, keeping its hasher:
/// a fresh table allocated with that capacity, every entry moved into it
/// (`extend(drain())`), then swapped in, the old table freed after the swap.
///
/// A structural plan's rollback rebuilds a table this way when it lacks the
/// room to re-create the keys its plan removed (`domain::plan_journal`):
/// with `len` the table's pre-plan length, the result is the minimal,
/// tombstone-free table for that length — never more buckets than the
/// current one when the current one held that many entries — so the
/// re-creations land without an insert call growing it, and the cache's
/// compaction leaves it alone. The move reserves nothing past `len` while
/// the table holds at most `len` entries.
pub fn rebuild_map_at<K: Eq + std::hash::Hash, V, S: std::hash::BuildHasher + Clone>(
    map: &mut HashMap<K, V, S>,
    len: usize,
) {
    let mut fresh = HashMap::with_capacity_and_hasher(len, map.hasher().clone());
    fresh.extend(map.drain());
    *map = fresh;
}

/// A clone of `map` that takes `room` more entries without growing: built at
/// a capacity of its length plus `room` when that is above its capacity, so
/// the clone is never a table that rehashes into twice its buckets once the
/// entries arrive; a plain clone (same bucket layout) otherwise. Built
/// directly at that capacity, so no copy at the old layout is made first.
pub fn clone_with_room<K: Clone + Eq + std::hash::Hash, V: Clone>(
    map: &HashMap<K, V>,
    room: usize,
) -> HashMap<K, V> {
    let wanted = map.len().saturating_add(room);
    if wanted <= map.capacity() {
        return map.clone();
    }
    let mut clone = HashMap::with_capacity(wanted);
    clone.extend(map.iter().map(|(k, v)| (k.clone(), v.clone())));
    clone
}

/// **Test support.** The usable slots of `map`'s bucket array — what its
/// `capacity()` reads with no tombstones — however many tombstones its
/// removals left: the true-bucket oracle the structural cache's weight checks
/// compare against (`domain::alloc_model::table_bytes` reads it exactly). A
/// clone allocates the same buckets (hashbrown's `RawTable::clone`), and
/// draining the clone marks every control byte empty and resets its growth
/// to the buckets' whole load (`RawDrain`'s drop, `clear_no_drop`); `clear()`
/// would skip that on a table with no items left. O(n).
#[cfg(any(test, feature = "test-support"))]
pub fn true_capacity_for_test<K: Clone, V: Clone, S: Clone>(map: &HashMap<K, V, S>) -> usize {
    let mut clone = map.clone();
    drop(clone.drain());
    clone.capacity()
}

/// **Test support.** One hash table's `capacity()` and its true capacity
/// ([`true_capacity_for_test`]).
#[cfg(any(test, feature = "test-support"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MapCapacities {
    /// `HashMap::capacity`.
    pub capacity: usize,
    /// [`true_capacity_for_test`].
    pub true_capacity: usize,
}

#[cfg(any(test, feature = "test-support"))]
impl MapCapacities {
    /// `map`'s two capacities.
    pub fn of<K: Clone, V: Clone, S: Clone>(map: &HashMap<K, V, S>) -> Self {
        Self {
            capacity: map.capacity(),
            true_capacity: true_capacity_for_test(map),
        }
    }
}

#[cfg(any(test, feature = "test-support"))]
impl CompactFields {
    /// **Test support.** The store's hash tables — its row map, then the
    /// interner's name → id table — with their capacities. Destructured field
    /// by field, with no `..`, so a field added to the store or its interner
    /// fails to build here until it is classified: a hash table, listed here
    /// and in the cache's table registry (`domain::delta_write::RebuildTable`),
    /// or not one (`domain::plan_index::hash_tables_for_test`).
    pub fn hash_tables_for_test(&self) -> (MapCapacities, MapCapacities) {
        let Self { names, rows } = self;
        let NameInterner {
            ids,
            slots: _,
            free: _,
        } = names;
        (MapCapacities::of(rows), MapCapacities::of(ids))
    }
}

#[cfg(any(test, feature = "test-support"))]
impl CompactSamples {
    /// **Test support.** The store's one hash table, its row map, with its
    /// capacities; destructured as [`CompactFields::hash_tables_for_test`].
    pub fn hash_tables_for_test(&self) -> MapCapacities {
        let Self { rows } = self;
        MapCapacities::of(rows)
    }
}

impl SampleRows for CompactSamples {
    fn insert_row(&mut self, sample: TimeSample) -> Option<Option<[u8; 32]>> {
        self.insert(
            sample.section_id,
            sample.time.into_inner().to_bits(),
            external_hash(&sample.value).copied(),
        )
    }

    /// A `retain` over every row, O(T) for `T` rows, in the row table's hash
    /// order; nothing at all for an empty store or set. Unlike the field
    /// purge there is no probe arm: the entry keeps no set of the layer's
    /// distinct times to probe, the way the interner keeps the field names,
    /// so a delete on a layer holding time samples still visits every one of
    /// them. Building such a set is the time-sample purge follow-up's index.
    fn purge_sections_with(
        &mut self,
        section_ids: &HashSet<u64>,
        mut on_removed: impl FnMut(u64, OrderedFloat<f64>, Option<&[u8; 32]>),
    ) {
        if self.rows.is_empty() || section_ids.is_empty() {
            return;
        }
        self.rows.retain(|&(section_id, bits), hash| {
            if section_ids.contains(&section_id) {
                on_removed(
                    section_id,
                    OrderedFloat(f64::from_bits(bits)),
                    hash.as_deref(),
                );
                false
            } else {
                true
            }
        });
    }

    fn drain_rows(&mut self, mut on_removed: impl FnMut(u64, OrderedFloat<f64>)) {
        for ((section_id, bits), _) in self.rows.drain() {
            on_removed(section_id, OrderedFloat(f64::from_bits(bits)));
        }
    }

    fn take_all(&mut self) -> Self {
        let empty = emptied(&self.rows);
        Self {
            rows: std::mem::replace(&mut self.rows, empty),
        }
    }

    fn for_each_in_drain_order(&self, mut f: impl FnMut(u64, OrderedFloat<f64>)) {
        for &(section_id, bits) in self.rows.keys() {
            f(section_id, OrderedFloat(f64::from_bits(bits)));
        }
    }

    fn contains(&self, section_id: u64, time_bits: u64) -> bool {
        self.rows.contains_key(&(section_id, time_bits))
    }

    fn ext_hash(&self, section_id: u64, time_bits: u64) -> Option<[u8; 32]> {
        self.rows.get(&(section_id, time_bits))?.as_deref().copied()
    }

    fn for_each(&self, mut f: impl FnMut(u64, u64, Option<&[u8; 32]>)) {
        for (&(section_id, bits), hash) in &self.rows {
            f(section_id, bits, hash.as_deref());
        }
    }

    fn len(&self) -> usize {
        self.rows.len()
    }

    fn shrink_to_fit(&mut self) {
        self.rows.shrink_to_fit();
    }

    fn zero_ext_indexes(&mut self) {}

    fn row_room(&self) -> usize {
        CompactSamples::row_room(self)
    }

    fn row_capacity(&self) -> usize {
        self.rows.capacity()
    }
}

impl fmt::Debug for CompactSamples {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut rows: Vec<((u64, f64), HashDebug<'_>)> = self
            .rows
            .iter()
            .map(|(&(section_id, bits), hash)| {
                (
                    (section_id, f64::from_bits(bits)),
                    HashDebug(hash.as_deref()),
                )
            })
            .collect();
        rows.sort_unstable_by(|a, b| (a.0 .0, a.0 .1.to_bits()).cmp(&(b.0 .0, b.0 .1.to_bits())));
        f.debug_map().entries(rows).finish()
    }
}

/// A compact row's hash, printed as hex (`inline` for none).
struct HashDebug<'a>(Option<&'a [u8; 32]>);

impl fmt::Debug for HashDebug<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.0 {
            None => f.write_str("inline"),
            Some(hash) => {
                f.write_str("ext:")?;
                for byte in hash {
                    write!(f, "{byte:02x}")?;
                }
                Ok(())
            }
        }
    }
}

/// The compact projection of `state`: the same header scalars, sections and
/// children lists, and per value row its key and external hash. A time-sample
/// row is keyed by its sample's own time bits. Pure; for tests and the
/// measurement tools, which build without `test-support` too (no production
/// path projects a full state: the cache decodes its entry straight into the
/// compact form).
pub fn compact(state: &ObjectState) -> CompactState {
    let mut fields = CompactFields::default();
    for ((section_id, key_name), field) in &state.fields {
        fields.insert(*section_id, key_name, external_hash(&field.value).copied());
    }
    let mut time_samples = CompactSamples::default();
    for ((section_id, _), sample) in &state.time_samples {
        time_samples.insert(
            *section_id,
            sample.time.into_inner().to_bits(),
            external_hash(&sample.value).copied(),
        );
    }
    CompactState {
        object_id: state.object_id,
        sequence_num: state.sequence_num,
        first_version: state.first_version,
        source_format: state.source_format.clone(),
        next_set_order: state.next_set_order,
        sections: state.sections.clone(),
        children: state.children.clone(),
        fields,
        time_samples,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── b.8jc rule B: one survivor for a stored signed-zero pair ───────────

    /// The section every rule-B fixture row is on.
    const PAIR_SECTION: u64 = 7;

    /// A stored time-sample row of [`PAIR_SECTION`] at `time`: an erase (an
    /// empty inline value) when `value` is empty, else that inline value.
    fn stored_row(time: f64, version: u64, value: &[u8]) -> TimeSample {
        TimeSample {
            section_id: PAIR_SECTION,
            time: OrderedFloat(time),
            value: FieldValue::Inline(value.to_vec()),
            version,
        }
    }

    /// Rule B over every combination the design names — the `-0.0` row's
    /// version higher, lower or equal × each row an erase or a value — with
    /// the expected survivor written out by hand rather than derived from the
    /// rule. Each case is checked on [`resolve_signed_zero_pair`] directly
    /// and through [`time_samples_from_stored_rows`] with the two rows given
    /// in both orders: the reader holds exactly one row, at `+0.0` (bits 0),
    /// carrying the survivor's value and version, whatever the read order.
    #[test]
    fn resolve_signed_zero_pair_table_picks_the_same_survivor_in_both_read_orders() {
        use SignedZeroSurvivor::{NegativeZero as Neg, PositiveZero as Pos};
        const VALUE: &[u8] = b"set";
        const ERASE: &[u8] = b"";
        /// (positive version, positive value, negative version, negative
        /// value, survivor)
        type Case = (u64, &'static [u8], u64, &'static [u8], SignedZeroSurvivor);
        let cases: [Case; 12] = [
            // The -0.0 row is newer: it wins, erase or not.
            (1, VALUE, 2, VALUE, Neg),
            (1, VALUE, 2, ERASE, Neg),
            (1, ERASE, 2, VALUE, Neg),
            (1, ERASE, 2, ERASE, Neg),
            // The +0.0 row is newer: it wins, erase or not.
            (2, VALUE, 1, VALUE, Pos),
            (2, VALUE, 1, ERASE, Pos),
            (2, ERASE, 1, VALUE, Pos),
            (2, ERASE, 1, ERASE, Pos),
            // A version tie: a value beats an erase, else +0.0 wins.
            (3, VALUE, 3, VALUE, Pos),
            (3, VALUE, 3, ERASE, Pos),
            (3, ERASE, 3, VALUE, Neg),
            (3, ERASE, 3, ERASE, Pos),
        ];
        for (pos_version, pos_value, neg_version, neg_value, survivor) in cases {
            let positive = stored_row(0.0, pos_version, pos_value);
            let negative = stored_row(-0.0, neg_version, neg_value);
            let context =
                format!("+0.0 v{pos_version} {pos_value:?} / -0.0 v{neg_version} {neg_value:?}");
            assert_eq!(
                resolve_signed_zero_pair(&positive, &negative),
                survivor,
                "{context}"
            );
            let expected = match survivor {
                Pos => &positive,
                Neg => &negative,
            };
            for (order, rows) in [
                ("+0.0 first", vec![positive.clone(), negative.clone()]),
                ("-0.0 first", vec![negative.clone(), positive.clone()]),
            ] {
                let held = time_samples_from_stored_rows(rows);
                assert_eq!(held.len(), 1, "{context}, {order}: one row held");
                let row = held
                    .get(&(PAIR_SECTION, OrderedFloat(0.0)))
                    .unwrap_or_else(|| panic!("{context}, {order}: held at time 0"));
                assert_eq!(
                    row.time.into_inner().to_bits(),
                    0,
                    "{context}, {order}: held at +0.0"
                );
                assert_eq!(
                    (&row.value, row.version),
                    (&expected.value, expected.version),
                    "{context}, {order}: the survivor's value and version"
                );
            }
        }
    }

    /// An external value is never an erase: on a version tie, a `-0.0` row
    /// holding an external value beats an empty `+0.0` row, and an external
    /// `+0.0` row beats an empty `-0.0` row.
    #[test]
    fn resolve_signed_zero_pair_tie_external_value_beats_an_erase() {
        let external = |time: f64| TimeSample {
            value: FieldValue::External {
                hash: [0x5A; 32],
                size: 64,
                ext_value_index: 0,
            },
            ..stored_row(time, 4, b"")
        };
        assert_eq!(
            resolve_signed_zero_pair(&stored_row(0.0, 4, b""), &external(-0.0)),
            SignedZeroSurvivor::NegativeZero
        );
        assert_eq!(
            resolve_signed_zero_pair(&external(0.0), &stored_row(-0.0, 4, b"")),
            SignedZeroSurvivor::PositiveZero
        );
    }

    /// A lone `-0.0` row decodes to `+0.0` (bits 0) with its value and
    /// version; a lone `+0.0` row and a row at any other time keep their key;
    /// and two sections' pairs resolve independently.
    #[test]
    fn time_samples_from_stored_rows_holds_a_lone_negative_zero_row_at_positive_zero() {
        let other = |section_id: u64, time: f64, version: u64| TimeSample {
            section_id,
            ..stored_row(time, version, b"x")
        };
        let held = time_samples_from_stored_rows([
            stored_row(-0.0, 5, b"lone"),
            other(8, 0.0, 2),
            other(8, -0.0, 1),
            other(PAIR_SECTION, 1.5, 3),
            other(9, 0.0, 6),
        ]);

        assert_eq!(held.len(), 4, "one row per (section, USD time): {held:?}");
        let lone = &held[&(PAIR_SECTION, OrderedFloat(0.0))];
        assert_eq!(lone.time.into_inner().to_bits(), 0, "held at +0.0");
        assert_eq!(
            (&lone.value, lone.version),
            (&FieldValue::Inline(b"lone".to_vec()), 5)
        );
        assert_eq!(
            held[&(8, OrderedFloat(0.0))].version,
            2,
            "section 8's +0.0 row is newer"
        );
        assert_eq!(held[&(PAIR_SECTION, OrderedFloat(1.5))].version, 3);
        assert_eq!(held[&(9, OrderedFloat(0.0))].version, 6);
        assert!(
            held.keys()
                .all(|(_, time)| !(time.into_inner() == 0.0 && time.is_sign_negative())),
            "no key holds -0.0"
        );
        assert!(
            held.values()
                .all(|row| !(row.time.into_inner() == 0.0 && row.time.is_sign_negative())),
            "no row holds -0.0"
        );
    }

    #[test]
    fn field_value_inline_equality_same_bytes_equal() {
        let a = FieldValue::Inline(vec![1, 2, 3]);
        let b = FieldValue::Inline(vec![1, 2, 3]);
        assert_eq!(a, b);
    }

    #[test]
    fn field_value_inline_equality_different_bytes_not_equal() {
        let a = FieldValue::Inline(vec![1, 2, 3]);
        let b = FieldValue::Inline(vec![4, 5, 6]);
        assert_ne!(a, b);
    }

    #[test]
    fn field_value_external_equality_same_hash_equal() {
        let hash = [0xabu8; 32];
        let a = FieldValue::External {
            hash,
            size: 1024,
            ext_value_index: 1,
        };
        let b = FieldValue::External {
            hash,
            size: 1024,
            ext_value_index: 1,
        };
        assert_eq!(a, b);
    }

    #[test]
    fn field_value_external_equality_different_hash_not_equal() {
        let hash_a = [0xabu8; 32];
        let mut hash_b = [0xabu8; 32];
        hash_b[0] = 0x01;
        let a = FieldValue::External {
            hash: hash_a,
            size: 1024,
            ext_value_index: 1,
        };
        let b = FieldValue::External {
            hash: hash_b,
            size: 1024,
            ext_value_index: 1,
        };
        assert_ne!(a, b);
    }

    #[test]
    fn field_value_external_equality_different_size_not_equal() {
        let hash = [0xabu8; 32];
        let a = FieldValue::External {
            hash,
            size: 1024,
            ext_value_index: 1,
        };
        let b = FieldValue::External {
            hash,
            size: 2048,
            ext_value_index: 1,
        };
        assert_ne!(a, b);
    }

    #[test]
    fn object_state_default_is_zero_and_empty() {
        let state = ObjectState::default();
        assert_eq!(state.object_id, 0);
        assert_eq!(state.sequence_num, 0);
        assert_eq!(state.first_version, 0);
        assert_eq!(state.source_format, "");
        assert_eq!(state.next_set_order, 0);
        assert!(state.sections.is_empty());
        assert!(state.children.is_empty());
        assert!(state.fields.is_empty());
        assert!(state.time_samples.is_empty());
    }

    #[test]
    fn time_sample_keyed_by_ordered_float_round_trip_in_hashmap() {
        let mut map: HashMap<(u64, OrderedFloat<f64>), TimeSample> = HashMap::new();
        let section_id = 42u64;
        let time = OrderedFloat(1.5f64);
        let sample = TimeSample {
            section_id,
            time,
            value: FieldValue::Inline(vec![7, 8, 9]),
            version: 3,
        };
        map.insert((section_id, time), sample.clone());

        let retrieved = map.get(&(section_id, OrderedFloat(1.5f64)));
        assert!(
            retrieved.is_some(),
            "should find entry by same OrderedFloat key"
        );
        assert_eq!(*retrieved.unwrap(), sample);
    }

    #[test]
    fn time_sample_ordered_float_distinct_keys_no_collision() {
        let mut map: HashMap<(u64, OrderedFloat<f64>), TimeSample> = HashMap::new();
        let section_id = 1u64;
        for i in 0..5u32 {
            let t = OrderedFloat(f64::from(i));
            let sample = TimeSample {
                section_id,
                time: t,
                value: FieldValue::Inline(vec![i as u8]),
                version: u64::from(i),
            };
            map.insert((section_id, t), sample);
        }
        assert_eq!(map.len(), 5);
        let retrieved = map.get(&(section_id, OrderedFloat(3.0f64)));
        assert_eq!(retrieved.unwrap().version, 3);
    }

    #[test]
    fn section_construction_round_trip() {
        let section = Section {
            section_id: 10,
            parent_id: 0,
            name: "root".to_string(),
            section_type: 1,
            section_order: 5,
            first_version: 1,
            last_version: 7,
        };
        assert_eq!(section.section_id, 10);
        assert_eq!(section.parent_id, 0);
        assert_eq!(section.name, "root");
        assert_eq!(section.section_type, 1);
        assert_eq!(section.section_order, 5);
        assert_eq!(section.first_version, 1);
        assert_eq!(section.last_version, 7);
    }

    #[test]
    fn section_equality_same_fields_equal() {
        let a = Section {
            section_id: 10,
            parent_id: 0,
            name: "root".to_string(),
            section_type: 1,
            section_order: 5,
            first_version: 1,
            last_version: 7,
        };
        let b = a.clone();
        assert_eq!(a, b);
    }

    #[test]
    fn field_construction_round_trip() {
        let value = FieldValue::Inline(vec![0xde, 0xad]);
        let field = Field {
            section_id: 99,
            key_name: "myKey".to_string(),
            value: value.clone(),
            set_order: 2,
            authoring_set_order: 2,
            version: 11,
        };
        assert_eq!(field.section_id, 99);
        assert_eq!(field.key_name, "myKey");
        assert_eq!(field.value, value);
        assert_eq!(field.set_order, 2);
        assert_eq!(field.version, 11);
    }

    #[test]
    fn field_equality_same_fields_equal() {
        let a = Field {
            section_id: 1,
            key_name: "k".to_string(),
            value: FieldValue::Inline(vec![0]),
            set_order: 0,
            authoring_set_order: 0,
            version: 0,
        };
        let b = a.clone();
        assert_eq!(a, b);
    }

    #[test]
    fn children_map_stores_and_retrieves_ordered_child_lists() {
        let mut state = ObjectState::default();
        let parent_id = 1u64;
        let children = vec![2u64, 3u64, 4u64];
        state.children.insert(parent_id, children.clone());

        let retrieved = state.children.get(&parent_id).unwrap();
        assert_eq!(retrieved.len(), 3);
        assert_eq!(*retrieved, children, "child order must be preserved");
    }

    #[test]
    fn object_state_sections_and_fields_round_trip() {
        let mut state = ObjectState {
            object_id: 5,
            sequence_num: 2,
            first_version: 1,
            source_format: "usd".to_string(),
            next_set_order: 3,
            ..Default::default()
        };

        let section = Section {
            section_id: 100,
            parent_id: 0,
            name: "prim".to_string(),
            section_type: 0,
            section_order: 0,
            first_version: 1,
            last_version: 2,
        };
        state.sections.insert(section.section_id, section.clone());

        let field = Field {
            section_id: 100,
            key_name: "xformOp".to_string(),
            value: FieldValue::Inline(vec![1, 0, 0, 0]),
            set_order: 1,
            authoring_set_order: 1,
            version: 1,
        };
        state
            .fields
            .insert((field.section_id, field.key_name.clone()), field.clone());

        assert_eq!(state.sections[&100], section);
        assert_eq!(state.fields[&(100, "xformOp".to_string())], field);
        assert_eq!(state.source_format, "usd");
    }

    // ── The compact form ────────────────────────────────────────────────────

    /// An external value of hash `tag` repeated.
    fn external(tag: u8) -> FieldValue {
        FieldValue::External {
            hash: [tag; 32],
            size: 100,
            ext_value_index: 7,
        }
    }

    fn field(section_id: u64, key_name: &str, value: FieldValue) -> Field {
        Field {
            section_id,
            key_name: key_name.to_owned(),
            value,
            set_order: 3,
            authoring_set_order: 2,
            version: 9,
        }
    }

    fn sample(section_id: u64, time: f64, value: FieldValue) -> TimeSample {
        TimeSample {
            section_id,
            time: OrderedFloat(time),
            value,
            version: 9,
        }
    }

    /// The names a compact field store has interned, sorted.
    fn interned(fields: &CompactFields) -> Vec<String> {
        let mut names: Vec<String> = fields.names().names().map(str::to_owned).collect();
        names.sort_unstable();
        names
    }

    /// T9: equality is semantic. Two stores holding the same rows, their names
    /// interned in different orders (so each name has another id in each),
    /// compare equal; a different hash on one row makes them unequal.
    #[test]
    fn compact_fields_same_rows_interned_in_different_orders_compare_equal() {
        let rows = [
            (1, "x", Some([1; 32])),
            (1, "y", None),
            (2, "z", Some([2; 32])),
        ];
        let mut forward = CompactFields::default();
        for (section_id, name, hash) in rows {
            forward.insert(section_id, name, hash);
        }
        let mut backward = CompactFields::default();
        for (section_id, name, hash) in rows.into_iter().rev() {
            backward.insert(section_id, name, hash);
        }
        assert_ne!(
            forward.names().id_of("x"),
            backward.names().id_of("x"),
            "fixture: the two interners must number the names differently"
        );

        assert_eq!(forward, backward);

        let mut rehashed = backward.clone();
        rehashed.insert(1, "x", Some([3; 32]));
        assert_ne!(forward, rehashed, "one row's hash differs");
        let mut inlined = backward.clone();
        inlined.insert(1, "x", None);
        assert_ne!(forward, inlined, "one row turned inline");
    }

    /// T9: the same `(section_id, name id)` pairs under different names are
    /// different rows: equality resolves the ids through each store's own
    /// interner, never compares them raw.
    #[test]
    fn compact_fields_same_ids_with_different_names_compare_unequal() {
        let mut a = CompactFields::default();
        a.insert(1, "x", None);
        a.insert(1, "y", None);
        let mut b = CompactFields::default();
        b.insert(1, "x", None);
        b.insert(1, "w", None);
        assert_eq!(
            a.names().id_of("y"),
            b.names().id_of("w"),
            "fixture: `y` and `w` must have the same id in their interners"
        );

        assert_ne!(a, b);
        assert_ne!(b, a, "inequality is symmetric");
    }

    /// T9: `Debug` prints the resolved names (never the ids), sorted by key,
    /// and each row's hash as hex or `inline`; time samples print their time.
    #[test]
    fn compact_rows_debug_shows_names_hashes_and_times() {
        let mut fields = CompactFields::default();
        fields.insert(2, "xformOp", Some([0xAB; 32]));
        fields.insert(1, "visibility", None);
        let printed = format!("{fields:?}");
        let ext = format!("ext:{}", "ab".repeat(32));
        assert_eq!(
            printed,
            format!("{{(1, \"visibility\"): inline, (2, \"xformOp\"): {ext}}}")
        );

        let mut samples = CompactSamples::default();
        samples.insert(3, (-0.0f64).to_bits(), None);
        samples.insert(3, 0.0f64.to_bits(), Some([0xAB; 32]));
        assert_eq!(
            format!("{samples:?}"),
            format!("{{(3, 0.0): {ext}, (3, -0.0): inline}}"),
            "each stored row prints, the two zeros apart, in time-bits order"
        );
    }

    /// The interner counts a name's rows: a replaced row keeps one use, a
    /// purge releases a name only when its last row goes, and a drain empties
    /// the interner (M2).
    #[test]
    fn compact_fields_intern_each_name_while_a_row_uses_it() {
        let mut fields = CompactFields::default();
        fields.insert_row(field(1, "a", FieldValue::Inline(vec![1])));
        fields.insert_row(field(2, "a", external(4)));
        fields.insert_row(field(1, "b", FieldValue::Inline(vec![2])));
        // Replaces (1, "b"): still one row, one use.
        fields.insert_row(field(1, "b", external(5)));
        assert_eq!(fields.len(), 3);
        assert_eq!(interned(&fields), ["a", "b"]);
        assert_eq!(fields.ext_hash(1, "b"), Some([5; 32]), "the replacement");

        let mut removed = Vec::new();
        fields.purge_sections(&HashSet::from([1]), |section_id, name| {
            removed.push((section_id, name.to_owned()));
        });
        removed.sort_unstable();
        assert_eq!(removed, [(1, "a".to_owned()), (1, "b".to_owned())]);
        assert_eq!(
            interned(&fields),
            ["a"],
            "`b` had one row (the replacement kept one use) and is released; `a` keeps (2, a)"
        );
        assert!(fields.contains(2, "a") && !fields.contains(1, "a"));

        fields.insert_row(field(7, "c", FieldValue::Inline(vec![3])));
        let mut drained = Vec::new();
        fields.drain_rows(|section_id, name| drained.push((section_id, name.to_owned())));
        drained.sort_unstable();
        assert_eq!(drained, [(2, "a".to_owned()), (7, "c".to_owned())]);
        assert!(fields.is_empty());
        assert!(fields.names().is_empty(), "a drain empties the interner");
    }

    /// A freed id is reused by the next new name, and `shrink_to_fit` trims
    /// the trailing free ids, so the interner's tables fall back to what its
    /// live names need (the cache's interner GC at `compact_state`).
    #[test]
    fn name_interner_reuses_freed_ids_and_shrink_trims_trailing_free_ids() {
        let mut fields = CompactFields::default();
        for (section_id, name) in [(1, "a"), (2, "b"), (3, "c"), (4, "d")] {
            fields.insert(section_id, name, None);
        }
        let id_b = fields.names().id_of("b").expect("b interned");
        fields.purge_sections(&HashSet::from([2]), |_, _| {});
        fields.insert(5, "e", None);
        assert_eq!(
            fields.names().id_of("e"),
            Some(id_b),
            "the new name takes the id `b` freed"
        );

        // Free the two highest ids (`c`'s and `d`'s rows), then shrink: both
        // slots go.
        assert_eq!(
            (fields.names().id_of("c"), fields.names().id_of("d")),
            (Some(2), Some(3)),
            "fixture: `c` and `d` hold the two highest ids"
        );
        let slots_before = fields.names().slots_capacity();
        fields.purge_sections(&HashSet::from([3, 4]), |_, _| {});
        FieldRows::shrink_to_fit(&mut fields);
        assert_eq!(interned(&fields), ["a", "e"]);
        assert!(
            fields.names().slots_capacity() < slots_before,
            "the trailing free slots are trimmed ({} vs {slots_before})",
            fields.names().slots_capacity()
        );
        assert_eq!(fields.names().free_capacity(), 0, "no free id is left");
        fields.insert(6, "f", None);
        assert_eq!(
            fields.names().id_of("f"),
            Some(2),
            "a trimmed id is never handed out from the free list twice"
        );
        assert_eq!(fields.names().name(2), "f");
    }

    /// The compact sample store keys a row by its stored time bits: `0.0` and
    /// `-0.0` are two rows, each found only by its own bits. The full form's
    /// `OrderedFloat` key holds one row for both, and its lookups compare the
    /// held sample's own bits, as the stored key would.
    #[test]
    fn compact_samples_keep_zero_and_negative_zero_as_two_rows() {
        let (zero, negative_zero) = (0.0f64.to_bits(), (-0.0f64).to_bits());
        let mut compact_rows = CompactSamples::default();
        compact_rows.insert_row(sample(1, 0.0, external(1)));
        compact_rows.insert_row(sample(1, -0.0, external(2)));
        assert_eq!(compact_rows.len(), 2);
        assert_eq!(compact_rows.ext_hash(1, zero), Some([1; 32]));
        assert_eq!(compact_rows.ext_hash(1, negative_zero), Some([2; 32]));

        let mut full: HashMap<(u64, OrderedFloat<f64>), TimeSample> = HashMap::new();
        full.insert_row(sample(1, 0.0, external(1)));
        assert!(full.contains(1, zero));
        assert!(
            !full.contains(1, negative_zero),
            "the other zero is not held"
        );
        assert_eq!(full.ext_hash(1, negative_zero), None);
        full.insert_row(sample(1, -0.0, external(2)));
        assert_eq!(SampleRows::len(&full), 1, "the full form conflates them");
    }

    /// `compact` keeps the header scalars, the sections and the children
    /// lists, and per value row its key and external hash; a time-sample row
    /// is keyed by the held sample's own time bits, not its map key's.
    #[test]
    fn compact_projects_every_row_to_its_key_and_hash() {
        let mut state = ObjectState {
            object_id: 5,
            sequence_num: 6,
            first_version: 2,
            source_format: "usda".to_owned(),
            next_set_order: 11,
            ..ObjectState::default()
        };
        state.sections.insert(
            1,
            Section {
                section_id: 1,
                parent_id: 0,
                name: "a".to_owned(),
                section_type: 1,
                section_order: 0,
                first_version: 1,
                last_version: 2,
            },
        );
        state.children.insert(0, vec![1]);
        state
            .fields
            .insert_row(field(1, "in", FieldValue::Inline(vec![1, 2])));
        state.fields.insert_row(field(1, "ex", external(3)));
        // Under the `+0.0` map key, a sample whose own time is `-0.0` (the
        // full map keeps the first key when a row is replaced).
        state.time_samples.insert_row(sample(1, 0.0, external(4)));
        state.time_samples.insert_row(sample(1, -0.0, external(5)));

        let projected = compact(&state);

        assert_eq!(
            (
                projected.object_id,
                projected.sequence_num,
                projected.first_version,
                projected.source_format.as_str(),
                projected.next_set_order,
            ),
            (5, 6, 2, "usda", 11)
        );
        assert_eq!(projected.sections, state.sections);
        assert_eq!(projected.children, state.children);
        let mut fields = CompactFields::default();
        fields.insert(1, "in", None);
        fields.insert(1, "ex", Some([3; 32]));
        assert_eq!(projected.fields, fields);
        let mut samples = CompactSamples::default();
        samples.insert(1, (-0.0f64).to_bits(), Some([5; 32]));
        assert_eq!(projected.time_samples, samples);
    }

    // ── The compact field purge's two arms (`PURGE_PROBE_RATIO`) ────────────
    //
    // Every test here holds at either hash-group width and under any hasher
    // seed: the arm predicate reads counts only, each arm comparison forces
    // the arm on clones of one store, and nothing compares a table's room,
    // capacity or free-list order across the arms.

    use super::purge_arm::{self, PurgeArm, PurgeCounts};

    /// One row the purge removed, as `on_removed` lent it.
    type Removed = (u64, String, Option<[u8; 32]>, Option<NameId>);

    /// A compact field store of `sections` sections (1 to `sections`), each
    /// holding `shared`, its own `own{s}` and `pair{s / 2}` (two sections
    /// share each), an external hash on every third row; then the rows of
    /// `own3` and `own7` removed, so the interner holds free ids below live
    /// ones (interior free slots).
    fn purge_store(sections: u64) -> CompactFields {
        let mut fields = CompactFields::default();
        let mut row = 0u8;
        for s in 1..=sections {
            for name in [
                "shared".to_owned(),
                format!("own{s}"),
                format!("pair{}", s / 2),
            ] {
                let hash = row.is_multiple_of(3).then_some([row; 32]);
                fields.insert(s, &name, hash);
                row = row.wrapping_add(1);
            }
        }
        for (s, name) in [(3, "own3"), (7, "own7")] {
            assert!(
                fields.remove(s, name).is_some_and(|(_, released)| released),
                "fixture: {name}'s only row is removed and its id freed"
            );
        }
        fields
    }

    /// Purge `ids` from `store` with `arm` forced on this thread: the store
    /// after it, every row it removed in emission order, and the seam's
    /// counts for the one purge.
    fn purge_forced(
        store: &CompactFields,
        ids: &HashSet<u64>,
        arm: Option<PurgeArm>,
    ) -> (CompactFields, Vec<Removed>, PurgeCounts) {
        let mut store = store.clone();
        let _forced = purge_arm::force(arm);
        purge_arm::reset();
        let mut removed = Vec::new();
        store.purge_sections_with(ids, |section_id, name, hash, released| {
            removed.push((section_id, name.to_owned(), hash.copied(), released));
        });
        (store, removed, purge_arm::counts())
    }

    /// Every live interned name with its id, sorted.
    fn live_ids(fields: &CompactFields) -> Vec<(String, NameId)> {
        let mut live: Vec<(String, NameId)> = fields
            .names()
            .names()
            .map(|name| (name.to_owned(), fields.names().id_of(name).expect("live")))
            .collect();
        live.sort_unstable();
        live
    }

    /// The removed rows as a sorted multiset of `(section, name, hash)`.
    fn removed_rows(removed: &[Removed]) -> Vec<(u64, String, Option<[u8; 32]>)> {
        let mut rows: Vec<_> = removed
            .iter()
            .map(|(section_id, name, hash, _)| (*section_id, name.clone(), *hash))
            .collect();
        rows.sort_unstable();
        rows
    }

    /// The released names with the ids they held, as a sorted set.
    fn released_names(removed: &[Removed]) -> Vec<(String, NameId)> {
        let mut released: Vec<(String, NameId)> = removed
            .iter()
            .filter_map(|(_, name, _, released)| released.map(|id| (name.clone(), id)))
            .collect();
        released.sort_unstable();
        released.dedup();
        released
    }

    /// T1: forced on clones of one store with one purge set, the probe and
    /// the scan arm leave equal stores, remove the same rows (multiset of
    /// `(section, name, hash)`), release the same names at the same ids, and
    /// leave the same live and free id sets. Which row carries a release, the
    /// free list's order, and the tables' room are not compared: they may
    /// differ by arm. The purge set holds an id no row has, and the store
    /// free ids below live ones.
    #[test]
    fn compact_purge_probe_and_scan_arms_remove_the_same_rows_and_release_the_same_names() {
        let store = purge_store(12);
        // 2 to 5 and an absent id: `own2`, `own4`, `own5`, `pair1` and
        // `pair2` lose their last row; `shared` keeps its others; section 3
        // lost `own3` before the purge.
        let ids: HashSet<u64> = [2, 3, 4, 5, 99].into_iter().collect();
        assert!(
            store.names().free_ids_sorted_for_test().len() == 2 && !store.contains(99, "shared"),
            "fixture: two interior free ids, and an absent purged section"
        );

        let (probed, probe_removed, probe_counts) =
            purge_forced(&store, &ids, Some(PurgeArm::Probe));
        let (scanned, scan_removed, scan_counts) = purge_forced(&store, &ids, Some(PurgeArm::Scan));

        assert_eq!(
            (probe_counts.probe_purges, probe_counts.scan_purges),
            (1, 0),
            "the probe arm ran"
        );
        assert_eq!(
            (scan_counts.probe_purges, scan_counts.scan_purges),
            (0, 1),
            "the scan arm ran"
        );
        assert_eq!(probed, scanned, "the same store after the purge");
        assert_eq!(
            removed_rows(&probe_removed),
            removed_rows(&scan_removed),
            "the same rows removed"
        );
        assert_eq!(
            removed_rows(&probe_removed).len(),
            11,
            "fixture: sections 2, 4 and 5 lose three rows each, section 3 two"
        );
        let released = released_names(&probe_removed);
        assert_eq!(
            released,
            released_names(&scan_removed),
            "the same names released at the same ids"
        );
        let mut released_set: Vec<&str> = released.iter().map(|(n, _)| n.as_str()).collect();
        released_set.sort_unstable();
        assert_eq!(released_set, ["own2", "own4", "own5", "pair1", "pair2"]);
        for (name, id) in &released {
            assert_eq!(
                store.names().id_of(name),
                Some(*id),
                "{name} is released at the id it held before the purge"
            );
        }
        assert_eq!(live_ids(&probed), live_ids(&scanned), "the live ids");
        assert_eq!(
            probed.names().free_ids_sorted_for_test(),
            scanned.names().free_ids_sorted_for_test(),
            "the free ids"
        );
    }

    /// T2: the probe arm's emission order is a function of the store and the
    /// purge set's members, never of the set's hasher: two clones of one
    /// store purged through two separately built `HashSet`s of the same ids
    /// (two `RandomState` seeds, iterating in different orders) emit the
    /// same run, row for row.
    #[test]
    fn compact_purge_probe_arm_emits_the_same_run_whatever_the_purge_sets_hasher() {
        const PURGED: u64 = 32;
        let store = purge_store(48);
        let first: HashSet<u64> = (1..=PURGED).collect();
        // A second set of the same ids that iterates in another order.
        let second = (0..64)
            .map(|_| (1..=PURGED).collect::<HashSet<u64>>())
            .find(|set| !set.iter().eq(first.iter()))
            .expect("fixture: a second hasher seed iterates 32 ids in another order");

        let (probed_first, run_first, counts_first) =
            purge_forced(&store, &first, Some(PurgeArm::Probe));
        let (probed_second, run_second, counts_second) =
            purge_forced(&store, &second, Some(PurgeArm::Probe));

        assert_eq!(
            (counts_first.probe_purges, counts_second.probe_purges),
            (1, 1),
            "both purges ran the probe arm"
        );
        assert!(
            run_first.len() > 64,
            "fixture: a long run ({} rows)",
            run_first.len()
        );
        assert_eq!(run_first, run_second, "the same run, in the same order");
        assert_eq!(probed_first, probed_second);
    }

    /// A compact field store of exactly `rows` rows over `names` live names
    /// and no free id: row `i` is `(1 + i / names, n{i % names})`.
    fn store_with(rows: usize, names: usize) -> CompactFields {
        assert!(rows >= names, "fixture: every name has a row");
        let mut fields = CompactFields::default();
        for i in 0..rows {
            fields.insert(1 + (i / names) as u64, &format!("n{}", i % names), None);
        }
        assert_eq!(FieldRows::len(&fields), rows, "fixture: {rows} rows");
        fields
    }

    /// T3: the arm predicate's boundary, exactly. `purges_by_probe(D)` is
    /// true at `rows = D × N × R` and false one row below, `N` the
    /// interner's id slots (live and free), `R` [`PURGE_PROBE_RATIO`]: a free
    /// slot below a live one counts (a predicate over the live names alone
    /// would put the boundary at `D × live × R`, which both stores pass).
    /// The predicate saturates rather than overflow. Counts only, so the
    /// same at either hash-group width.
    #[test]
    fn purges_by_probe_is_true_at_exactly_d_n_r_rows_and_false_one_row_below() {
        const D: usize = 2;
        const N: usize = 3;
        let boundary = D * N * PURGE_PROBE_RATIO;
        assert!(store_with(boundary, N).purges_by_probe(D), "at D·N·R rows");
        assert!(
            !store_with(boundary - 1, N).purges_by_probe(D),
            "at D·N·R − 1 rows"
        );

        // Four id slots, one of them free below a live one: the boundary is
        // D·4·R, though only three names are live.
        let slotted = |rows: usize| {
            // `n0` to `n3` interned at ids 0 to 3, then `n1`'s one row
            // removed, freeing id 1 below `n2` and `n3`; then rows of `n0`
            // up to `rows`.
            let mut fields = CompactFields::default();
            for name in ["n0", "n1", "n2", "n3"] {
                fields.insert(1, name, None);
            }
            fields.remove(1, "n1");
            let mut section_id = 2u64;
            while FieldRows::len(&fields) < rows {
                fields.insert(section_id, "n0", None);
                section_id += 1;
            }
            assert_eq!(
                fields.names().free_ids_sorted_for_test(),
                [1],
                "fixture: id 1 free below live ids"
            );
            assert_eq!(fields.names().len(), N, "fixture: three live names");
            fields
        };
        let slot_boundary = D * (N + 1) * PURGE_PROBE_RATIO;
        assert!(
            slotted(slot_boundary).purges_by_probe(D),
            "at D·(live + free)·R rows"
        );
        assert!(
            !slotted(slot_boundary - 1).purges_by_probe(D),
            "one row below D·(live + free)·R, though above D·live·R = {}",
            D * N * PURGE_PROBE_RATIO
        );

        assert!(
            !store_with(boundary, N).purges_by_probe(usize::MAX),
            "the product saturates"
        );
    }

    /// T3: natural selection, far from the boundary on each side. A store of
    /// `4 × D × N × R` rows purged of `D` sections takes the probe arm and
    /// visits `D × N` steps (no purged section releases a name, so every
    /// slot is live at every visit); a store of `D × N × R ÷ 4` rows purged
    /// of `D` sections takes the scan arm and visits every row. Sized from
    /// [`PURGE_PROBE_RATIO`], so the fixtures move with `R`.
    #[test]
    fn compact_purge_takes_the_probe_arm_far_below_the_crossover_and_the_scan_arm_far_above() {
        const N: usize = 3;

        // Probe: D = 2 sections of a store 4× past the boundary.
        const D_PROBE: usize = 2;
        let store = store_with(4 * D_PROBE * N * PURGE_PROBE_RATIO, N);
        let ids: HashSet<u64> = (1..=D_PROBE as u64).collect();
        let (_, removed, counts) = purge_forced(&store, &ids, None);
        assert_eq!(
            (counts.probe_purges, counts.scan_purges),
            (1, 0),
            "the probe arm, naturally"
        );
        assert_eq!(counts.steps_visited, (D_PROBE * N) as u64, "D × N steps");
        assert_eq!(removed.len(), D_PROBE * N, "every row of the D sections");

        // Scan: D sections where D × N × R is 4× the rows.
        let rows = 2 * N * PURGE_PROBE_RATIO;
        let d_scan = 4 * rows / (N * PURGE_PROBE_RATIO);
        let store = store_with(rows, N);
        assert!(
            d_scan as u64 <= 1 + (rows / N) as u64,
            "fixture: the purged sections are the store's"
        );
        let ids: HashSet<u64> = (1..=d_scan as u64).collect();
        let (_, removed, counts) = purge_forced(&store, &ids, None);
        assert_eq!(
            (counts.probe_purges, counts.scan_purges),
            (0, 1),
            "the scan arm, naturally"
        );
        assert_eq!(counts.steps_visited, rows as u64, "every row");
        assert_eq!(removed.len(), d_scan * N, "every row of the D sections");
    }

    /// T5: a purge of an empty store, or by an empty set, visits nothing and
    /// allocates nothing: it returns before either arm (the seam counts no
    /// purge; the probe arm's sorted id vector is its one allocation, and an
    /// empty store with no interned names would otherwise take that arm,
    /// `0 ≤ 0`), calls `on_removed` for no row, and leaves every table's
    /// capacity as it was. The time-sample purge likewise.
    #[test]
    fn compact_purge_of_an_empty_store_or_by_an_empty_set_visits_and_allocates_nothing() {
        let ids: HashSet<u64> = (1..=8).collect();
        let empty_set = HashSet::new();

        // A never-used store (no id slot) and an emptied one (free slots
        // only, its row table keeping its capacity).
        let mut emptied = store_with(12, 3);
        emptied.purge_sections(&(1..=4).collect(), |_, _| {});
        assert!(
            emptied.is_empty() && !emptied.names().free_ids_sorted_for_test().is_empty(),
            "fixture: no row, free id slots"
        );
        for (label, store, set) in [
            ("a never-used store", CompactFields::default(), &ids),
            ("an emptied store", emptied, &ids),
            ("an empty purge set", store_with(12, 3), &empty_set),
        ] {
            for arm in [None, Some(PurgeArm::Probe), Some(PurgeArm::Scan)] {
                // A clone, as `purge_forced` purges one: a cloned id → name
                // vector holds its length's capacity.
                let before = store.clone();
                let (after, removed, counts) = purge_forced(&store, set, arm);
                assert_eq!(
                    counts,
                    PurgeCounts::default(),
                    "{label}, {arm:?}: no purge ran"
                );
                assert!(removed.is_empty(), "{label}, {arm:?}: no row removed");
                assert_eq!(
                    (
                        after.capacity(),
                        after.names().slots_capacity(),
                        after.names().free_capacity()
                    ),
                    (
                        before.capacity(),
                        before.names().slots_capacity(),
                        before.names().free_capacity()
                    ),
                    "{label}, {arm:?}: no table changed"
                );
                assert_eq!(after, store, "{label}, {arm:?}: the same rows");
            }
        }

        let mut samples = CompactSamples::with_capacity(16);
        let capacity = samples.capacity();
        let mut visited = 0;
        samples.purge_sections_with(&ids, |_, _, _| visited += 1);
        assert_eq!(
            (visited, samples.capacity()),
            (0, capacity),
            "an empty sample store"
        );
        samples.insert(1, 1.0f64.to_bits(), None);
        samples.purge_sections_with(&empty_set, |_, _, _| visited += 1);
        assert_eq!(
            (visited, SampleRows::len(&samples)),
            (0, 1),
            "an empty purge set"
        );
    }
}
