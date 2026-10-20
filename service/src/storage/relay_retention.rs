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

//! The per-object relay byte budget: the `relay_gens` header column and the
//! fold every commit applies to it.
//!
//! Every `UpdateLayer` commit `XADD`s the writer's whole inbound OMPF to the
//! object's stream as the entry's `ompf` field, so the stream's memory cost is
//! in bytes, while `MAXLEN ~ LIVE_EDIT_LAYER_STREAM_MAXLEN` bounds it only in
//! entries. `LIVE_EDIT_LAYER_STREAM_MAX_BYTES` (B) adds a byte bound: the
//! stream keeps the newest entries whose relay bytes fit B (to within one
//! generation, B/K), never fewer than [`RELAY_MIN_ENTRIES`], and never more
//! than the entry cap. Retained relay bytes per object are therefore at most
//! about B plus [`RELAY_MIN_ENTRIES`] × `LIVE_EDIT_MAX_LAYER_SIZE`.
//!
//! # The column
//!
//! `obj:hdr` carries [`HEADER_FIELD_RELAY_GENS`]: at most K + 1 `start:bytes`
//! pairs, oldest first, joined by `,`, every number canonical decimal. `bytes`
//! is the summed length of the `ompf` fields of the commits from `start` up to
//! the next pair's `start`. The value is advanced by [`RelayRetention::fold`]
//! in the same header `HSET` that advances `seq` — `update_layer_fast.lua` step
//! 7 in both script modes, and the WATCH path's `SetObjectHeader` `HSET` — so
//! it is written in the commit's own atomic unit and is byte-identical
//! whichever route committed (the `commit_chain` pattern, SDD §7.2). A refused
//! commit, or one answered from its commit token, writes nothing.
//!
//! An absent, malformed or out-of-range value is read as fresh
//! (`[(new_seq, 0)]`): an object materialized before the column existed, by an
//! older replica, or by a restore starts accounting at its next commit, and an
//! entry it did not account stays bounded by the entry cap until a trim removes
//! it. Like `commit_chain` the column is not one of the structural header
//! fields and is removed with the header. With B = 0 the column is neither read
//! nor written; a stale value left from an earlier B > 0 can only over-count,
//! which trims earlier, never later.
//!
//! # The trim
//!
//! When the fold drops a generation, the commit trims the stream head with
//! `XTRIM stream MINID ~ {cut}-0`, after its own `XADD` succeeded and in the
//! same atomic unit, where `cut = min(oldest kept start, max(1, new_seq −
//! MIN + 1))`. Trimming is head-only and never reaches the entry just added,
//! because `cut ≤ new_seq`. `~` trims whole stream nodes only, so it can keep
//! slightly more than the cut names, never less.
//!
//! # Lua parity
//!
//! `update_layer_fast.lua` runs the same algorithm, statement for statement,
//! on the header its step 1 read; B, T, K and MIN reach it through the shared
//! `ARGV` prefix ([`crate::storage::lua::UpdateScriptPrefix`]), so the script
//! derives no constant of its own. Every number in the column, and every sum
//! the fold compares or stores, stays within `2^53 − 1` (sums are clamped to
//! it), the largest integer a Lua number holds exactly, so the two
//! implementations agree on every input.

/// The `obj:hdr` column holding the relay byte accounting (see the module
/// documentation). `pub` because integration tests in the external test crate
/// read it back by name.
pub const HEADER_FIELD_RELAY_GENS: &str = "relay_gens";

/// K: the number of generations the byte budget is split into. A generation
/// closes once it holds at least B/K bytes, so the budget is enforced to within
/// B/K, and the column holds at most K + 1 pairs.
pub const RELAY_GENERATIONS: u64 = 8;

/// MIN: the fewest entries a byte trim leaves on the stream. It binds only when
/// entries are larger than about B/2, and gives a live reader one commit of
/// slack when two such commits land within one of its reads.
pub const RELAY_MIN_ENTRIES: u64 = 2;

/// `2^53 − 1`: the largest integer a Lua number (an IEEE 754 double) holds
/// exactly. Every value the fold parses, sums or stores is kept at or below it,
/// so `update_layer_fast.lua` computes the identical result.
const F64_INT_CEILING: u64 = (1u64 << 53) - 1;

/// The separator between two `start:bytes` pairs in the column.
const PAIR_SEPARATOR: char = ',';
/// The separator between a pair's `start` and its `bytes`.
const START_BYTES_SEPARATOR: char = ':';

/// The byte-budget parameters every commit carries: B, T, K and MIN.
///
/// Built once per store from `LIVE_EDIT_LAYER_STREAM_MAX_BYTES`
/// ([`Self::from_budget`]) and sent unchanged on every route, so the script and
/// the WATCH path fold with the same numbers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RelayRetention {
    /// B: the per-object byte budget for retained relay bytes. `0` turns the
    /// byte budget off — the column is neither read nor written and no trim is
    /// issued (count-only retention).
    pub budget_bytes: u64,
    /// T = max(1, ⌈B/K⌉): the bytes at which the newest generation closes.
    pub generation_bytes: u64,
    /// K ([`RELAY_GENERATIONS`]).
    pub generations: u64,
    /// MIN ([`RELAY_MIN_ENTRIES`]).
    pub min_entries: u64,
}

impl RelayRetention {
    /// The parameters for a byte budget of `budget_bytes` (B; `0` = off), with
    /// K = [`RELAY_GENERATIONS`] and MIN = [`RELAY_MIN_ENTRIES`].
    pub fn from_budget(budget_bytes: u64) -> Self {
        Self {
            budget_bytes,
            generation_bytes: budget_bytes.div_ceil(RELAY_GENERATIONS).max(1),
            generations: RELAY_GENERATIONS,
            min_entries: RELAY_MIN_ENTRIES,
        }
    }

    /// Whether the byte budget is on (B > 0).
    pub fn enabled(&self) -> bool {
        self.budget_bytes > 0
    }

    /// Fold one commit at `new_seq`, whose relay OMPF is `relay_bytes` long
    /// (`0` for a commit without one), into the column's previous value
    /// `prev` (`None` when the header has no such column). `None` when the
    /// byte budget is off: nothing is written and nothing trimmed.
    ///
    /// The algorithm, `update_layer_fast.lua`'s statement for statement:
    ///
    /// 1. Parse `prev`. Read it as fresh, `[(new_seq, 0)]`, when it is absent
    ///    or malformed (anything but canonical-digit `start:bytes` pairs joined
    ///    by `,`, or a number above `2^53 − 1`), when any `start` is above
    ///    `new_seq`, or when the starts are not strictly increasing.
    /// 2. Add `relay_bytes` to the newest generation's bytes.
    /// 3. While more than one generation remains and their bytes sum above B,
    ///    drop the oldest.
    /// 4. If the newest generation now holds at least T bytes, open a new one,
    ///    `(new_seq + 1, 0)`.
    /// 5. While more than K + 1 generations remain, merge the two oldest.
    /// 6. Format the result.
    /// 7. If step 3 dropped a generation, trim below
    ///    `min(oldest kept start, max(1, new_seq − MIN + 1))`.
    ///
    /// Every sum is clamped to `2^53 − 1`, as the script's is.
    pub fn fold(&self, prev: Option<&str>, new_seq: u64, relay_bytes: u64) -> Option<RelayFold> {
        if !self.enabled() {
            return None;
        }
        // Step 1.
        let mut gens = prev
            .and_then(|value| parse_generations(value, new_seq))
            .unwrap_or_else(|| vec![Generation::fresh(new_seq)]);

        // Step 2.
        let newest = gens.last_mut().expect("parse never yields an empty list");
        newest.bytes = clamped_sum(newest.bytes, relay_bytes);

        // Step 3: keep the longest run of newest generations whose bytes sum
        // to at most B, and always the newest one.
        let mut first_kept = gens.len() - 1;
        let mut kept_bytes = gens[first_kept].bytes;
        while first_kept > 0 {
            let candidate = gens[first_kept - 1].bytes;
            if kept_bytes + candidate > self.budget_bytes {
                break;
            }
            kept_bytes += candidate;
            first_kept -= 1;
        }
        let dropped = first_kept > 0;
        gens.drain(..first_kept);

        // Step 4.
        let newest_bytes = gens.last().expect("the newest generation is kept").bytes;
        if newest_bytes >= self.generation_bytes {
            gens.push(Generation::fresh(new_seq + 1));
        }

        // Step 5.
        let max_len = usize::try_from(self.generations.saturating_add(1)).unwrap_or(usize::MAX);
        while gens.len() > max_len {
            let second = gens.remove(1);
            gens[0].bytes = clamped_sum(gens[0].bytes, second.bytes);
        }

        // Steps 6 and 7.
        let trim_below_seq = dropped.then(|| {
            let floor = new_seq
                .saturating_add(1)
                .saturating_sub(self.min_entries)
                .max(1);
            gens[0].start.min(floor)
        });
        Some(RelayFold {
            value: format_generations(&gens),
            trim_below_seq,
        })
    }
}

/// The result of one [`RelayRetention::fold`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RelayFold {
    /// The column's new value, written in the header `HSET` that advances
    /// `seq`.
    pub value: String,
    /// `Some(cut)` when the fold dropped a generation: the commit trims the
    /// stream with `XTRIM MINID ~ {cut}-0` after its `XADD`. `None`: no trim.
    pub trim_below_seq: Option<u64>,
}

impl RelayFold {
    /// The `XTRIM MINID` threshold id for [`Self::trim_below_seq`].
    pub fn trim_min_id(&self) -> Option<String> {
        self.trim_below_seq.map(|cut| format!("{cut}-0"))
    }
}

/// One `start:bytes` pair of the column.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Generation {
    start: u64,
    bytes: u64,
}

impl Generation {
    fn fresh(start: u64) -> Self {
        Self { start, bytes: 0 }
    }
}

/// `a + b`, clamped to `2^53 − 1` — the script's `math.min(a + b, CEILING)`.
/// Both inputs are at or below the ceiling, so the true sum fits a `u64`.
fn clamped_sum(a: u64, b: u64) -> u64 {
    a.saturating_add(b).min(F64_INT_CEILING)
}

/// Parse the column under the strict grammar, or `None` when the value must be
/// read as fresh (see step 1 of [`RelayRetention::fold`]).
fn parse_generations(value: &str, new_seq: u64) -> Option<Vec<Generation>> {
    let mut gens: Vec<Generation> = Vec::new();
    for pair in value.split(PAIR_SEPARATOR) {
        let (start, bytes) = pair.split_once(START_BYTES_SEPARATOR)?;
        let start = parse_decimal(start)?;
        let bytes = parse_decimal(bytes)?;
        if start > new_seq {
            return None;
        }
        if gens.last().is_some_and(|previous| start <= previous.start) {
            return None;
        }
        gens.push(Generation { start, bytes });
    }
    // `split` yields at least one piece, and every piece was pushed or
    // returned `None`.
    Some(gens)
}

/// One number of the column: one or more ASCII digits, at most `2^53 − 1`.
/// `str::parse` alone would also accept a leading `+`, which the script's
/// `%d+` pattern does not.
fn parse_decimal(digits: &str) -> Option<u64> {
    if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    digits.parse::<u64>().ok().filter(|n| *n <= F64_INT_CEILING)
}

fn format_generations(gens: &[Generation]) -> String {
    let mut out = String::new();
    for (i, generation) in gens.iter().enumerate() {
        if i > 0 {
            out.push(PAIR_SEPARATOR);
        }
        out.push_str(&generation.start.to_string());
        out.push(START_BYTES_SEPARATOR);
        out.push_str(&generation.bytes.to_string());
    }
    out
}

#[cfg(test)]
mod tests {
    //! The fold, step by step, on hand-computed inputs. The invariants over
    //! arbitrary commit sequences (the byte bound, no over-trimming, exact
    //! accounting, at most K + 1 pairs) are the property tests in
    //! `tests/proptests.rs`; that the script folds byte-identically is the
    //! route-parity suite in `tests/integration/storage/relay_retention_test.rs`.
    use super::*;

    /// B = 80 bytes, so T = 10: small enough to hand-compute every step.
    const B: u64 = 80;

    fn fold(prev: Option<&str>, new_seq: u64, relay_bytes: u64) -> RelayFold {
        RelayRetention::from_budget(B)
            .fold(prev, new_seq, relay_bytes)
            .expect("the budget is on")
    }

    #[test]
    fn from_budget_derives_generation_size_count_and_minimum() {
        assert_eq!(
            RelayRetention::from_budget(134_217_728),
            RelayRetention {
                budget_bytes: 134_217_728,
                generation_bytes: 16_777_216,
                generations: 8,
                min_entries: 2,
            },
            "the production default splits 128 MiB into eight 16 MiB generations"
        );
        assert_eq!(
            RelayRetention::from_budget(9).generation_bytes,
            2,
            "T is ⌈B/K⌉, rounded up"
        );
        assert_eq!(
            RelayRetention::from_budget(5).generation_bytes,
            1,
            "T is at least 1 when B < K"
        );
        let off = RelayRetention::from_budget(0);
        assert!(!off.enabled(), "B = 0 turns the byte budget off");
        assert_eq!(
            off.generation_bytes, 1,
            "the script refuses T = 0 with the budget on; with it off T is still a valid 1"
        );
    }

    #[test]
    fn fold_with_the_budget_off_writes_no_column_and_no_trim() {
        assert_eq!(
            RelayRetention::from_budget(0).fold(Some("1:500,4:0"), 5, 1_000),
            None,
            "B = 0 is count-only retention: nothing to write, nothing to trim"
        );
    }

    #[test]
    fn fold_of_an_absent_column_starts_fresh_at_the_new_sequence() {
        assert_eq!(
            fold(None, 7, 6),
            RelayFold {
                value: "7:6".to_owned(),
                trim_below_seq: None,
            }
        );
    }

    #[test]
    fn fold_adds_the_relay_bytes_to_the_newest_generation_only() {
        assert_eq!(
            fold(Some("3:10,5:4"), 6, 3),
            RelayFold {
                value: "3:10,5:7".to_owned(),
                trim_below_seq: None,
            }
        );
    }

    #[test]
    fn fold_of_an_empty_commit_rewrites_the_column_unchanged() {
        assert_eq!(fold(Some("3:10,5:4"), 6, 0).value, "3:10,5:4");
    }

    #[test]
    fn fold_opens_a_generation_at_the_next_sequence_once_the_newest_reaches_t() {
        assert_eq!(
            fold(Some("2:9"), 3, 1),
            RelayFold {
                value: "2:10,4:0".to_owned(),
                trim_below_seq: None,
            },
            "10 bytes is exactly T = 10: the generation closes and the next commit (4) opens one"
        );
        assert_eq!(
            fold(Some("2:8"), 3, 1).value,
            "2:9",
            "below T the generation stays open"
        );
    }

    #[test]
    fn fold_drops_the_oldest_generations_past_the_budget_and_trims_below_the_oldest_kept() {
        // Newest 8:15 + 10 = 25; + 5:30 = 55 ≤ 80 keeps 5; + 2:30 = 85 > 80
        // drops 2. The newest (25 ≥ T) closes and 10:0 opens. The cut is the
        // oldest kept start, 5, below the MIN floor 9 − 2 + 1 = 8.
        let folded = fold(Some("2:30,5:30,8:15"), 9, 10);
        assert_eq!(
            folded,
            RelayFold {
                value: "5:30,8:25,10:0".to_owned(),
                trim_below_seq: Some(5),
            }
        );
        assert_eq!(folded.trim_min_id().as_deref(), Some("5-0"));
    }

    #[test]
    fn fold_keeps_generations_whose_bytes_sum_exactly_to_the_budget() {
        // 40 + 40 = 80 = B: the boundary is "sum above B", so nothing drops.
        assert_eq!(
            fold(Some("2:40,5:30"), 6, 10),
            RelayFold {
                value: "2:40,5:40,7:0".to_owned(),
                trim_below_seq: None,
            }
        );
    }

    #[test]
    fn fold_of_a_commit_larger_than_the_budget_keeps_the_minimum_entries() {
        // The newest alone (100) is above B, so every older generation drops;
        // the newest is always kept. The oldest kept start (6) is the commit
        // itself, so the MIN floor 6 − 2 + 1 = 5 sets the cut: entries 5 and 6
        // stay, the two-entry minimum.
        assert_eq!(
            fold(Some("4:70,6:0"), 6, 100),
            RelayFold {
                value: "6:100,7:0".to_owned(),
                trim_below_seq: Some(5),
            }
        );
    }

    #[test]
    fn fold_cut_is_never_below_one() {
        // MIN floor 2 − 2 + 1 = 1, the stream's first possible sequence.
        assert_eq!(fold(Some("1:90,2:0"), 2, 50).trim_below_seq, Some(1));
    }

    #[test]
    fn fold_merges_the_two_oldest_generations_past_k_plus_one() {
        // Nine 10-byte generations, left by a smaller budget (T was 10); at
        // B = 800 (T = 100) nothing drops, the newest closes at 105, and the
        // tenth pair it opens is folded away by merging the two oldest.
        let prev = "1:10,2:10,3:10,4:10,5:10,6:10,7:10,8:10,9:10";
        let folded = RelayRetention::from_budget(800)
            .fold(Some(prev), 10, 95)
            .expect("the budget is on");
        assert_eq!(
            folded,
            RelayFold {
                value: "1:20,3:10,4:10,5:10,6:10,7:10,8:10,9:105,11:0".to_owned(),
                trim_below_seq: None,
            }
        );
        assert_eq!(
            folded.value.split(',').count() as u64,
            RELAY_GENERATIONS + 1,
            "at most K + 1 pairs"
        );
    }

    #[test]
    fn fold_reads_a_malformed_or_inconsistent_column_as_fresh() {
        // Each of these is read as absent: the commit at 8 starts a fresh
        // column, `8:4`, and trims nothing. The script applies the same grammar
        // (`%d+:%d+` pairs joined by `,`, numbers at most 2^53 − 1, starts
        // strictly increasing and at most the new sequence).
        let rejected = [
            ("", "empty"),
            ("garbage", "not a pair"),
            ("1:2,", "trailing separator"),
            (",1:2", "leading separator"),
            ("1:2,,3:4", "empty pair"),
            ("1:2:3", "three numbers"),
            ("1:2,3", "a pair without bytes"),
            (":2", "a pair without a start"),
            ("+1:2", "a sign"),
            ("1:-2", "a negative"),
            (" 1:2", "whitespace"),
            ("1:2 ", "trailing whitespace"),
            ("1.0:2", "a fraction"),
            ("2:5,2:6", "a repeated start"),
            ("3:5,2:6", "decreasing starts"),
            ("9:5", "a start above the new sequence"),
            ("1:9007199254740992", "bytes above 2^53 - 1"),
            ("9007199254740992:1", "a start above 2^53 - 1"),
            ("1:99999999999999999999999", "bytes past u64"),
        ];
        for (value, why) in rejected {
            assert_eq!(
                fold(Some(value), 8, 4),
                RelayFold {
                    value: "8:4".to_owned(),
                    trim_below_seq: None,
                },
                "{why} ({value:?}) must be read as a fresh column"
            );
        }
    }

    #[test]
    fn fold_accepts_a_start_equal_to_the_new_sequence() {
        // A generation opened by the previous commit starts at this one.
        assert_eq!(fold(Some("2:10,8:0"), 8, 4).value, "2:10,8:4");
    }

    #[test]
    fn fold_accepts_leading_zeros_and_writes_canonical_decimals() {
        assert_eq!(fold(Some("007:003"), 8, 4).value, "7:7");
    }

    #[test]
    fn fold_clamps_every_sum_to_the_lua_integer_ceiling() {
        let ceiling = (1u64 << 53) - 1;
        let folded = RelayRetention::from_budget(ceiling)
            .fold(Some(&format!("1:{ceiling}")), 2, 10)
            .expect("the budget is on");
        assert_eq!(
            folded.value,
            format!("1:{ceiling},3:0"),
            "the newest generation's sum stays at 2^53 − 1, the largest integer the script's \
             numbers hold exactly"
        );
        assert_eq!(folded.trim_below_seq, None);
    }

    #[test]
    fn trim_min_id_names_the_cut_as_a_stream_id() {
        let no_trim = RelayFold {
            value: "1:1".to_owned(),
            trim_below_seq: None,
        };
        assert_eq!(no_trim.trim_min_id(), None);
        let trim = RelayFold {
            value: "5:1".to_owned(),
            trim_below_seq: Some(5),
        };
        assert_eq!(trim.trim_min_id().as_deref(), Some("5-0"));
    }
}
