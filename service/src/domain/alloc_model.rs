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

//! The allocation model the structural write cache weighs its entries by:
//! jemalloc's size classes ([`allocation_bytes`]), a std `HashMap`'s
//! allocation at a given capacity ([`table_bytes`]), and what a map or a
//! `Vec` can allocate as inserts grow it ([`table_growth_bytes`],
//! [`vec_growth_bytes`]).
//!
//! The one definition of each, read by the cache's state estimate
//! (`storage::structural_cache::estimate_state_bytes`), by the persistent
//! plan indexes (`domain::plan_index`), which keep running sums of the same
//! allocations so the entry's weight can be read in O(1) — equal to the
//! estimate at every compaction, and above it while a table thinned since
//! then hides buckets the estimate's `capacity()` reading misses — and by the
//! charges that price a plan's growth before it runs. Pure arithmetic: no
//! I/O, no allocation.

/// The control bytes a hash table allocates past its buckets (hashbrown's
/// group width).
pub const TABLE_TRAILING_CONTROL_BYTES: u64 = 16;

/// The allocation of a hash table reporting `capacity` usable slots, at
/// `slot_bytes` per slot: buckets times the slot size, plus one control byte
/// per bucket and the trailing group.
///
/// The bucket count is hashbrown's (`capacity_to_buckets`, 0.16.1, for slots
/// of 4 bytes or more): 4 buckets for up to 3 slots, 8 for up to 7, then the
/// power of two whose 7/8 load covers `capacity`. A std `HashMap`'s
/// `capacity()` is its items plus its remaining growth: never below its
/// length, and below its buckets' usable slots by the tombstones removals
/// left. On a table that is minimal for its length — every state the cache
/// stores, after `storage::structural_cache::compact_state` — both ends of
/// that range round to the same bucket count, so the bucket count is exact,
/// tombstones or not; so is a 4-bucket table's, whatever its tombstones.
///
/// An uncompacted table thinned by removals is read low: one power of two
/// (half its buckets) once its tombstones reach half its usable slots, two
/// once they reach three quarters, and nothing at all once tombstones fill
/// every usable slot. A caller weighing such a table passes its held
/// capacity instead — the largest `capacity()` it reported just before a
/// removal since it was last compacted, or its `capacity()` when that is
/// larger (`domain::plan_index::PlanIndexes::held_capacity`).
pub fn table_bytes(capacity: usize, slot_bytes: u64) -> u64 {
    buckets_bytes(table_buckets(capacity as u64), slot_bytes)
}

/// The bucket count [`table_bytes`] reads a table of `capacity` usable slots
/// as (0 for none). Monotone in `capacity`, and the inverse of
/// [`bucket_capacity`] on a bucket count: `table_buckets(bucket_capacity(b))
/// == b`.
pub fn table_buckets(capacity: u64) -> u64 {
    if capacity == 0 {
        return 0;
    }
    // hashbrown's `capacity_to_buckets` for the slot sizes weighed here:
    // 4 buckets up to 3 slots, 8 up to 7, then the power of two whose 7/8
    // load covers the slots.
    if capacity < 4 {
        return 4;
    }
    if capacity < 8 {
        return 8;
    }
    (capacity.saturating_mul(8) / 7)
        .checked_next_power_of_two()
        .unwrap_or(u64::MAX)
}

/// The usable slots of a table of `buckets` buckets: hashbrown's load limit
/// (`bucket_mask_to_capacity`), one slot short of the buckets up to 8 of
/// them, 7/8 of them from 16 up; 0 for none. What `capacity()` reports for
/// such a table with no tombstones.
pub fn bucket_capacity(buckets: u64) -> u64 {
    if buckets <= 8 {
        buckets.saturating_sub(1)
    } else {
        (buckets / 8).saturating_mul(7)
    }
}

/// The allocation of a table of `buckets` buckets of `slot_bytes` each.
fn buckets_bytes(buckets: u64, slot_bytes: u64) -> u64 {
    if buckets == 0 {
        return 0;
    }
    allocation_bytes(
        buckets
            .saturating_mul(slot_bytes.saturating_add(1))
            .saturating_add(TABLE_TRAILING_CONTROL_BYTES),
    )
}

/// The most a std `HashMap` holding `len` items at `capacity`
/// (`HashMap::capacity`) can allocate beyond its current table while
/// `insert_calls` more inserts land on it: 0 when it has room for them, else
/// the tables it grows through that are alive at once.
///
/// - **Every insert call counts, not only a new key.** `HashMap::insert`
///   reserves one slot before it looks its key up (hashbrown's
///   `find_or_find_insert_index`), so an overwrite on a table with no room
///   left grows it too; `entry()` reserves only for a vacant key. A table
///   whose length plus its insert calls stays within its capacity never
///   grows: each call finds at least one slot of growth left. A structural
///   plan's rollback is not priced here: it restores rewritten keys in
///   place and makes insert calls only to re-create removed ones, rebuilding
///   a table without the room at its pre-plan length rather than growing it
///   (`domain::plan_journal`, charged on its own).
/// - **A growth at least doubles the buckets.** hashbrown grows a full table
///   to `max(items + 1, full + 1)` slots (`reserve_rehash_inner`), `full`
///   being the usable slots of its current bucket count. Removals leave
///   tombstones that `capacity()` does not count and that `shrink_to_fit`
///   keeps on a table already minimal for its length, so the next growth can
///   come with the length well below `full`, and still doubles. The current
///   bucket count is read from `capacity` by [`table_bytes`]' own arithmetic,
///   exact on a minimal table; on a table thinned since it was compacted,
///   where `capacity` reads low, [`table_growth_bytes_held`] reads it from
///   the held capacity instead. (When enough of a full table is tombstones,
///   hashbrown rehashes in place instead and allocates nothing; the bound
///   ignores that.)
/// - **Each growth doubles again**, until the table holds `len +
///   insert_calls`. The old table is freed once its items move, so what is
///   alive at once is the last table grown to and the one before it — but
///   the one before is the current table, which its holder already counts,
///   when the growth is a single doubling.
///
/// [`table_growth_bytes_held`] with the held capacity at `capacity`.
pub fn table_growth_bytes(len: usize, capacity: usize, insert_calls: u64, slot_bytes: u64) -> u64 {
    table_growth_bytes_held(len, capacity, capacity, insert_calls, slot_bytes)
}

/// [`table_growth_bytes`] of a table whose holder weighs it at
/// `held_capacity` (`domain::plan_index::PlanIndexes::held_capacity`, never
/// below `capacity`): the room the inserts take still comes from `capacity`
/// (`capacity − len`), but the current bucket count — what a growth doubles
/// from, and the table the holder counts — from `held_capacity`, which
/// tombstones cannot hide. With `held_capacity` at `capacity` it is
/// [`table_growth_bytes`] exactly.
pub fn table_growth_bytes_held(
    len: usize,
    capacity: usize,
    held_capacity: usize,
    insert_calls: u64,
    slot_bytes: u64,
) -> u64 {
    let current = table_buckets(held_capacity.max(capacity) as u64);
    table_growth(len, capacity, current, current, insert_calls, slot_bytes).bytes
}

/// What [`table_growth_bounded`] prices for one table: the bytes, and the
/// bucket count the growth reaches (the current one when the inserts fit).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TableGrowth {
    /// The bytes alive at once beyond what the holder counts.
    pub bytes: u64,
    /// The bucket count of the last table grown to: at least what hashbrown
    /// can grow the table to through the inserts, from any bucket count up
    /// to the bound's. The bound's own when the inserts fit the room.
    pub grown_buckets: u64,
}

/// [`table_growth_bytes`] of a table known only to hold at most the bucket
/// count `bound_capacity` reads as (`storage::structural_cache::CopyBucketBound`,
/// the sampled audit's index-less copy): the room from `capacity`, the
/// doublings from the bound's bucket count — so the tables grown through are
/// at least the true ones. One doubling answers the table grown to, from at
/// most `2b` for the bound's `b` buckets: sound because the copy's charge
/// holds each table at `b`. When the inserts outgrow one doubling, the last
/// two tables less the current table at `capacity`'s reading, which is never
/// above the table's true bucket count — the conservative choice, which does
/// not rely on the charge holding the table at `b`. The soundness argument
/// is `storage::structural_cache::estimate_replay_record_growth_bounded`'s.
/// Also answers the bucket count it grew to, for the bound's raise.
pub fn table_growth_bounded(
    len: usize,
    capacity: usize,
    bound_capacity: usize,
    insert_calls: u64,
    slot_bytes: u64,
) -> TableGrowth {
    let current = table_buckets(bound_capacity.max(capacity) as u64);
    let counted = table_buckets(capacity as u64);
    table_growth(len, capacity, current, counted, insert_calls, slot_bytes)
}

/// The growth arithmetic of [`table_growth_bytes_held`] and
/// [`table_growth_bounded`]: the doublings run from `current` buckets until
/// the table holds `len + insert_calls`; a single doubling answers the table
/// grown to, several answer the last two less `counted` buckets, the table
/// the holder counts.
fn table_growth(
    len: usize,
    capacity: usize,
    current: u64,
    counted: u64,
    insert_calls: u64,
    slot_bytes: u64,
) -> TableGrowth {
    let wanted = (len as u64).saturating_add(insert_calls);
    if wanted <= capacity as u64 {
        return TableGrowth {
            bytes: 0,
            grown_buckets: current,
        };
    }
    let mut previous = current;
    // hashbrown's smallest allocated table has 4 buckets (3 usable slots).
    let mut grown = table_buckets(bucket_capacity(current).saturating_add(1)).max(4);
    while bucket_capacity(grown) < wanted && grown < u64::MAX / 2 {
        previous = grown;
        grown = grown.saturating_mul(2);
    }
    let last = buckets_bytes(grown, slot_bytes);
    let bytes = if previous == current {
        last
    } else {
        last.saturating_add(buckets_bytes(previous, slot_bytes))
            .saturating_sub(buckets_bytes(counted, slot_bytes))
    };
    TableGrowth {
        bytes,
        grown_buckets: grown,
    }
}

/// The most a `Vec` of `capacity` elements holding `len` can allocate beyond
/// its current buffer while `pushes` more elements are pushed onto it one at
/// a time: 0 when it has room, else the buffers it grows through that are
/// alive at once, as [`table_growth_bytes`]. A push past the capacity grows
/// it to `max(2 × capacity, len + 1, 4)` (`RawVec`'s amortized growth, its
/// least non-zero capacity being 4 for elements up to 1 KiB).
pub fn vec_growth_bytes(len: usize, capacity: usize, pushes: u64, element_bytes: u64) -> u64 {
    let capacity = capacity as u64;
    let wanted = (len as u64).saturating_add(pushes);
    if wanted <= capacity {
        return 0;
    }
    let bytes = |elements: u64| allocation_bytes(elements.saturating_mul(element_bytes));
    let mut previous = capacity;
    let mut grown = capacity.saturating_mul(2).max(4);
    while grown < wanted && grown < u64::MAX / 2 {
        previous = grown;
        grown = grown.saturating_mul(2);
    }
    if previous == capacity {
        bytes(grown)
    } else {
        bytes(grown)
            .saturating_add(bytes(previous))
            .saturating_sub(bytes(capacity))
    }
}

/// The bytes jemalloc hands out for an `n`-byte request: 8, then 16-byte
/// steps to 128, then four size classes per doubling.
pub fn allocation_bytes(n: u64) -> u64 {
    match n {
        0 => 0,
        1..=8 => 8,
        9..=128 => n.div_ceil(16) * 16,
        _ => {
            // 2^k ≤ n − 1 < 2^(k+1), so the classes above 2^k are 2^(k−2) apart.
            let k = 63 - (n - 1).leading_zeros();
            let step = 1u64 << (k - 2);
            n.div_ceil(step).saturating_mul(step)
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::mem::size_of;

    use super::*;

    /// One slot of the `HashMap<u64, u64>` the std checks below grow.
    const SLOT: u64 = size_of::<(u64, u64)>() as u64;

    /// A `HashMap<u64, u64>` holding `0..len`, reserved for `capacity`.
    fn map_of(len: u64, capacity: usize) -> HashMap<u64, u64> {
        let mut map = HashMap::with_capacity(capacity);
        for key in 0..len {
            map.insert(key, key);
        }
        map
    }

    #[test]
    fn allocation_bytes_rounds_up_to_jemallocs_size_classes() {
        for (request, class) in [
            (0, 0),
            (1, 8),
            (8, 8),
            (9, 16),
            (100, 112),
            (128, 128),
            (129, 160),
            (160, 160),
            (161, 192),
            (1_000, 1_024),
            (1_025, 1_280),
            (4_097, 5_120),
        ] {
            assert_eq!(allocation_bytes(request), class, "a {request}-byte request");
        }
    }

    #[test]
    fn table_bytes_reads_hashbrowns_bucket_count_from_the_capacity() {
        assert_eq!(table_bytes(0, SLOT), 0, "no table, no allocation");
        // 4 buckets up to 3 slots, 8 up to 7, then 7/8 load.
        for (capacity, buckets) in [(1, 4), (3, 4), (4, 8), (7, 8), (14, 16), (28, 32)] {
            assert_eq!(
                table_bytes(capacity, SLOT),
                allocation_bytes(buckets * (SLOT + 1) + TABLE_TRAILING_CONTROL_BYTES),
                "capacity {capacity} is {buckets} buckets"
            );
        }
        // And std's own tables: a map reserved for n reports the capacity of
        // the bucket count the arithmetic reads back.
        for reserved in [1usize, 3, 4, 7, 8, 14, 15, 100, 1_000] {
            let map: HashMap<u64, u64> = HashMap::with_capacity(reserved);
            assert_eq!(
                table_bytes(map.capacity(), SLOT),
                table_bytes(reserved, SLOT),
                "std reserved {reserved} and reports {}",
                map.capacity()
            );
        }
    }

    #[test]
    fn table_growth_bytes_is_zero_while_the_calls_fit_the_capacity() {
        assert_eq!(table_growth_bytes(0, 0, 0, SLOT), 0);
        assert_eq!(table_growth_bytes(3, 7, 4, SLOT), 0, "7 of 7: no growth");
        assert_eq!(table_growth_bytes(14, 14, 0, SLOT), 0);
    }

    /// One growth past the capacity is the doubled table alone (the current
    /// one is its holder's), and it is what std grows to, whether the call
    /// that needs it adds a key or overwrites one.
    #[test]
    fn table_growth_bytes_of_one_doubling_is_the_table_std_grows_to() {
        for (len, capacity) in [(3u64, 3usize), (7, 7), (14, 14), (28, 28)] {
            let predicted = table_growth_bytes(len as usize, capacity, 1, SLOT);
            let mut map = map_of(len, capacity);
            assert_eq!(map.capacity(), capacity, "fixture: a full table");
            // An overwrite: the key exists, and std still reserves first.
            map.insert(0, 1);
            assert!(map.capacity() > capacity, "std grew the full table");
            assert_eq!(
                predicted,
                table_bytes(map.capacity(), SLOT),
                "one insert call on a full table of {capacity}: the doubled table"
            );
        }
        // From nothing, the first insert allocates the 4-bucket table.
        assert_eq!(table_growth_bytes(0, 0, 1, SLOT), table_bytes(3, SLOT));
    }

    /// Several growths: the last table and the one before it are alive at
    /// once while the items move, less the current table its holder counts.
    #[test]
    fn table_growth_bytes_of_several_doublings_holds_the_last_two_tables() {
        let (len, capacity, calls) = (14u64, 14usize, 20u64);
        let predicted = table_growth_bytes(len as usize, capacity, calls, SLOT);
        let mut map = map_of(len, capacity);
        assert_eq!(map.capacity(), capacity, "fixture: a full table");
        let mut grown_through = Vec::new();
        for key in len..len + calls {
            let before = map.capacity();
            map.insert(key, key);
            if map.capacity() != before {
                grown_through.push(map.capacity());
            }
        }
        assert_eq!(
            grown_through.len(),
            2,
            "fixture: two growths, {grown_through:?}"
        );
        let (previous, last) = (grown_through[0], grown_through[1]);
        assert_eq!(
            predicted,
            table_bytes(last, SLOT) + table_bytes(previous, SLOT) - table_bytes(capacity, SLOT),
            "the last table and the one before it, less the current one"
        );
    }

    #[test]
    fn vec_growth_bytes_is_zero_with_room_and_doubles_from_at_least_four() {
        const ELEMENT: u64 = 8;
        let bytes = |elements: u64| allocation_bytes(elements * ELEMENT);
        assert_eq!(vec_growth_bytes(3, 4, 1, ELEMENT), 0, "room for the push");
        assert_eq!(
            vec_growth_bytes(0, 0, 1, ELEMENT),
            bytes(4),
            "the least non-zero is 4"
        );
        assert_eq!(vec_growth_bytes(4, 4, 1, ELEMENT), bytes(8), "one doubling");
        assert_eq!(
            vec_growth_bytes(4, 4, 10, ELEMENT),
            bytes(16) + bytes(8) - bytes(4),
            "two doublings: the last buffer and the one before it, less the current"
        );
        // And std's own growth: a full Vec grows to the capacity predicted.
        for capacity in [0usize, 1, 4, 5, 100] {
            let mut v: Vec<u64> = Vec::with_capacity(capacity);
            v.extend(0..capacity as u64);
            let predicted = vec_growth_bytes(v.len(), v.capacity(), 1, ELEMENT);
            v.push(0);
            assert_eq!(
                predicted,
                bytes(v.capacity() as u64),
                "one push past a full Vec of capacity {capacity}"
            );
        }
    }

    /// R-T1, the premise the rollback's restore-in-place rule rests on
    /// (`domain::plan_journal`, "Rollback without growth"): on std's map at
    /// `capacity() == len()`, inserting an existing key grows the table,
    /// while a `get_mut` assignment of the same key does not. Fails if a std
    /// upgrade stops reserving before the lookup (the rule would then be
    /// over-cautious, not wrong) or starts reserving on `get_mut`.
    #[test]
    fn std_map_insert_of_an_existing_key_grows_a_full_table_and_get_mut_does_not() {
        for capacity in [3usize, 7, 14, 112] {
            let mut map = map_of(capacity as u64, capacity);
            assert_eq!(
                (map.len(), map.capacity()),
                (capacity, capacity),
                "fixture: a table at capacity() == len()"
            );
            *map.get_mut(&0).expect("key 0 is present") = 99;
            assert_eq!(
                map.capacity(),
                capacity,
                "a get_mut assignment does not reserve"
            );
            map.insert(0, 100);
            assert!(
                map.capacity() > capacity,
                "insert of an existing key on a full table grows it ({capacity} -> {})",
                map.capacity()
            );
            assert_eq!(map.len(), capacity, "it only overwrote");
        }
    }

    /// The `(len, capacity, insert_calls)` shapes the held and bounded
    /// growth checks sweep: no room, some room, room to spare, one doubling
    /// and several, from nothing and from 4 to 64 buckets.
    const GROWTH_SHAPES: [(usize, usize, u64); 9] = [
        (0, 0, 1),
        (3, 3, 1),
        (5, 7, 2),
        (7, 7, 1),
        (7, 7, 100),
        (14, 14, 20),
        (20, 28, 9),
        (27, 27, 1),
        (56, 56, 300),
    ];

    /// T7 (b.5h3): with the held capacity at the table's own capacity — or
    /// below it, which the held reading never is but `max` absorbs —
    /// `table_growth_bytes_held` is `table_growth_bytes` exactly, so every
    /// caller weighing an untouched table prices as before.
    #[test]
    fn table_growth_bytes_held_at_the_capacity_is_table_growth_bytes() {
        for (len, capacity, calls) in GROWTH_SHAPES {
            let plain = table_growth_bytes(len, capacity, calls, SLOT);
            assert_eq!(
                table_growth_bytes_held(len, capacity, capacity, calls, SLOT),
                plain,
                "({len}, {capacity}, {calls}) held at its capacity"
            );
            assert_eq!(
                table_growth_bytes_held(len, capacity, capacity / 2, calls, SLOT),
                plain,
                "({len}, {capacity}, {calls}) held below its capacity"
            );
        }
    }

    /// T7 (b.5h3): a table read at a held capacity above its `capacity()` —
    /// 27 slots shown of the 56 its 64 buckets hold, the rest hidden by
    /// tombstones — still takes its room from `capacity()` (no growth while
    /// the calls fit it), but doubles from the held 64 buckets: one call past
    /// the room prices the 128-bucket table, where the live capacity alone
    /// would price the 64-bucket one it already holds; several doublings
    /// price the last two tables less the held one, which its holder counts.
    #[test]
    fn table_growth_bytes_held_above_the_capacity_doubles_from_the_held_buckets() {
        let (shown, held) = (27usize, 56usize);
        assert_eq!(
            (table_buckets(shown as u64), table_buckets(held as u64)),
            (32, 64),
            "fixture: the shown capacity reads half the held buckets"
        );
        assert_eq!(
            table_growth_bytes_held(20, shown, held, 7, SLOT),
            0,
            "the calls fit the room capacity() shows"
        );
        assert_eq!(
            table_growth_bytes_held(20, shown, held, 8, SLOT),
            buckets_bytes(128, SLOT),
            "one call past the room: doubled from the held 64 buckets"
        );
        assert_eq!(
            table_growth_bytes_held(shown, shown, held, 1, SLOT),
            buckets_bytes(128, SLOT),
            "one call on no room: the 128-bucket table"
        );
        assert_eq!(
            table_growth_bytes(shown, shown, 1, SLOT),
            buckets_bytes(64, SLOT),
            "the live capacity alone would price the table it already holds"
        );
        assert_eq!(
            table_growth_bytes_held(shown, shown, held, 100, SLOT),
            buckets_bytes(256, SLOT) + buckets_bytes(128, SLOT) - buckets_bytes(64, SLOT),
            "several doublings: the last two tables less the held one"
        );
    }

    /// T7 (b.5h3), `table_growth_bounded`, the sampled audit copy's pricing:
    /// the room comes from `capacity()`, the doublings from the bound's
    /// bucket count, and the answer names the bucket count grown to (the
    /// bound's own when the calls fit). With the bound at the capacity it is
    /// `table_growth_bytes`. On the multi-doubling branch the current table
    /// subtracted is the one `capacity()` reads — 8 buckets here, under a
    /// bound of 32 — never the bound's: subtracting at `capacity()`'s reading
    /// rather than the bound's 32 buckets is the conservative choice; it does
    /// not rely on the charge holding the table at `b`.
    #[test]
    fn table_growth_bounded_doubles_from_the_bound_and_subtracts_the_table_capacity_reads() {
        for (len, capacity, calls) in GROWTH_SHAPES {
            assert_eq!(
                table_growth_bounded(len, capacity, capacity, calls, SLOT).bytes,
                table_growth_bytes(len, capacity, calls, SLOT),
                "({len}, {capacity}, {calls}) bounded at its own capacity"
            );
        }
        let (shown, bound) = (7usize, 28usize);
        assert_eq!(
            (table_buckets(shown as u64), table_buckets(bound as u64)),
            (8, 32),
            "fixture: a bound above what capacity() reads"
        );
        assert_eq!(
            table_growth_bounded(5, shown, bound, 2, SLOT),
            TableGrowth {
                bytes: 0,
                grown_buckets: 32
            },
            "the calls fit the room: no growth, at the bound's buckets"
        );
        assert_eq!(
            table_growth_bounded(shown, shown, bound, 1, SLOT),
            TableGrowth {
                bytes: buckets_bytes(64, SLOT),
                grown_buckets: 64
            },
            "one doubling, from the bound's 32 buckets"
        );
        assert_eq!(
            table_growth_bounded(shown, shown, bound, 100, SLOT),
            TableGrowth {
                bytes: buckets_bytes(128, SLOT) + buckets_bytes(64, SLOT) - buckets_bytes(8, SLOT),
                grown_buckets: 128
            },
            "several doublings: the last two tables less the 8 buckets capacity() reads"
        );
    }
}
