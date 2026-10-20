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

// OMPF container parse and serialize.
//
// Wire format per the public Omniverse OMPF spec:
//   [4B magic "OMPF" (0x46504d4f)]
//   [4B num_parts (u32 LE)]
//   [num_parts * 8B offsets (u64 LE)]
//   [part 0 bytes][part 1 bytes]...[part N-1 bytes]
//
// The offset table has exactly `num_parts` entries — there is NO trailing
// sentinel. `offsets[i]` marks where part `i` begins; part `i`'s end is
// `offsets[i+1]` for `i < num_parts - 1`, and the last part runs to the end
// of the buffer.

use thiserror::Error;

const MAGIC: u32 = 0x46504d4f;
const MAGIC_SIZE: usize = 4;
const NUM_PARTS_SIZE: usize = 4;
const OFFSET_SIZE: usize = 8;
const HEADER_BASE: usize = MAGIC_SIZE + NUM_PARTS_SIZE;

#[derive(Debug, Error, PartialEq)]
pub enum OmpfError {
    #[error("bad magic: expected 0x{expected:08x}, got 0x{got:08x}")]
    BadMagic { expected: u32, got: u32 },

    #[error("truncated: needed {needed} bytes, available {available}")]
    Truncated { needed: usize, available: usize },

    #[error("offset table not monotonically increasing at index {index}")]
    OffsetNotMonotonic { index: usize },

    #[error("offset {offset} at index {index} is out of bounds (data size {size})")]
    OffsetOutOfBounds {
        offset: u64,
        index: usize,
        size: usize,
    },

    #[error("num_parts is 0 but data is not empty")]
    NoParts,

    #[error("num_parts {num_parts} exceeds maximum {max} implied by data size {size}")]
    NumPartsTooLarge {
        num_parts: usize,
        max: usize,
        size: usize,
    },

    #[error(
        "offset[0] = {offset} overlaps offset table ending at {table_end} (num_parts = {num_parts})"
    )]
    OffsetOverlapsTable {
        offset: u64,
        table_end: usize,
        num_parts: usize,
    },

    #[error("offset {offset} at index {index} does not fit in usize on this platform")]
    OffsetNotAddressable { offset: u64, index: usize },

    #[error("serialize: num_parts {num_parts} exceeds u32::MAX")]
    SerializeNumPartsOverflow { num_parts: usize },
}

pub struct OmpfContainer<'a> {
    pub delta_bytes: &'a [u8],
    pub external_values: Vec<&'a [u8]>,
}

impl<'a> OmpfContainer<'a> {
    /// Parse an OMPF binary container from `data`.
    ///
    /// Empty input is valid (SDD §8.4) and returns an empty container.
    pub fn parse(data: &'a [u8]) -> Result<Self, OmpfError> {
        if data.is_empty() {
            return Ok(OmpfContainer {
                delta_bytes: &[],
                external_values: vec![],
            });
        }

        if data.len() < MAGIC_SIZE {
            return Err(OmpfError::Truncated {
                needed: MAGIC_SIZE,
                available: data.len(),
            });
        }
        let mut magic_buf = [0u8; MAGIC_SIZE];
        magic_buf.copy_from_slice(&data[0..MAGIC_SIZE]);
        let got_magic = u32::from_le_bytes(magic_buf);
        if got_magic != MAGIC {
            return Err(OmpfError::BadMagic {
                expected: MAGIC,
                got: got_magic,
            });
        }

        if data.len() < HEADER_BASE {
            return Err(OmpfError::Truncated {
                needed: HEADER_BASE,
                available: data.len(),
            });
        }
        let mut num_parts_buf = [0u8; NUM_PARTS_SIZE];
        num_parts_buf.copy_from_slice(&data[MAGIC_SIZE..HEADER_BASE]);
        let num_parts = u32::from_le_bytes(num_parts_buf) as usize;

        if num_parts == 0 {
            return Err(OmpfError::NoParts);
        }

        // Bound num_parts by the input size before any allocation. An 8-byte
        // header claiming num_parts = u32::MAX would otherwise drive a ~34 GB
        // Vec::with_capacity below. `data.len() / OFFSET_SIZE` is a hard cap
        // since the offset table alone needs that many bytes.
        let max_parts = data.len() / OFFSET_SIZE;
        if num_parts > max_parts {
            return Err(OmpfError::NumPartsTooLarge {
                num_parts,
                max: max_parts,
                size: data.len(),
            });
        }

        // Offset table: `num_parts` u64 LE values starting at byte 8.
        // `checked_mul` guards against overflow on 32-bit targets; `checked_add`
        // guards the final sum. Both are theoretically reachable only if the
        // num_parts cap above is ever loosened.
        let offsets_bytes =
            num_parts
                .checked_mul(OFFSET_SIZE)
                .ok_or(OmpfError::NumPartsTooLarge {
                    num_parts,
                    max: max_parts,
                    size: data.len(),
                })?;
        let offsets_end =
            HEADER_BASE
                .checked_add(offsets_bytes)
                .ok_or(OmpfError::NumPartsTooLarge {
                    num_parts,
                    max: max_parts,
                    size: data.len(),
                })?;
        if data.len() < offsets_end {
            return Err(OmpfError::Truncated {
                needed: offsets_end,
                available: data.len(),
            });
        }

        let mut offsets = Vec::with_capacity(num_parts);
        for i in 0..num_parts {
            let start = HEADER_BASE + i * OFFSET_SIZE;
            let mut off_buf = [0u8; OFFSET_SIZE];
            off_buf.copy_from_slice(&data[start..start + OFFSET_SIZE]);
            offsets.push(u64::from_le_bytes(off_buf));
        }

        // `usize::try_from` on every u64 offset — on 32-bit targets an
        // offset > u32::MAX silently truncates via `as usize` and aliases
        // onto valid bytes. Reject explicitly with `OffsetNotAddressable`.
        let mut offset_usizes: Vec<usize> = Vec::with_capacity(num_parts);
        for (i, off) in offsets.iter().enumerate() {
            let off_us = usize::try_from(*off).map_err(|_| OmpfError::OffsetNotAddressable {
                offset: *off,
                index: i,
            })?;
            offset_usizes.push(off_us);
        }

        // offsets[0] must not point inside the header + offset table. Without
        // this check, an adversarial OMPF can alias part bodies onto header
        // bytes and downstream parsers may mis-parse.
        if offset_usizes[0] < offsets_end {
            return Err(OmpfError::OffsetOverlapsTable {
                offset: offsets[0],
                table_end: offsets_end,
                num_parts,
            });
        }

        // Validate offsets: monotonic and within bounds.
        for i in 0..num_parts {
            if offset_usizes[i] > data.len() {
                return Err(OmpfError::OffsetOutOfBounds {
                    offset: offsets[i],
                    index: i,
                    size: data.len(),
                });
            }
            if i > 0 && offsets[i] < offsets[i - 1] {
                return Err(OmpfError::OffsetNotMonotonic { index: i });
            }
        }

        // Slice parts. Part i ends at offsets[i+1] for i < num_parts-1; the
        // last part ends at data.len().
        let delta_end = if num_parts == 1 {
            data.len()
        } else {
            offset_usizes[1]
        };
        let delta_bytes = &data[offset_usizes[0]..delta_end];

        let mut external_values = Vec::with_capacity(num_parts.saturating_sub(1));
        for i in 1..num_parts {
            let end = if i == num_parts - 1 {
                data.len()
            } else {
                offset_usizes[i + 1]
            };
            external_values.push(&data[offset_usizes[i]..end]);
        }

        Ok(OmpfContainer {
            delta_bytes,
            external_values,
        })
    }

    /// Serialize an OMPF container to owned bytes.
    ///
    /// Empty delta and no external values returns an empty Vec (SDD §8.4).
    ///
    /// Returns [`OmpfError::SerializeNumPartsOverflow`] if `1 + external_values.len()`
    /// doesn't fit in a `u32` (the wire-format count field). This is unreachable
    /// on 64-bit hosts with realistic inputs — a Vec with >4 B entries would
    /// exhaust memory long before — but the explicit check replaces a silent
    /// `as u32` truncation that would have emitted a bogus header on 64-bit
    /// pathological inputs.
    pub fn serialize(delta: &[u8], external_values: &[&[u8]]) -> Result<Vec<u8>, OmpfError> {
        if delta.is_empty() && external_values.is_empty() {
            return Ok(Vec::new());
        }

        let num_parts = 1 + external_values.len();
        let num_parts_u32 = u32::try_from(num_parts)
            .map_err(|_| OmpfError::SerializeNumPartsOverflow { num_parts })?;
        let header_size = MAGIC_SIZE + NUM_PARTS_SIZE + num_parts * OFFSET_SIZE;

        let total_payload: usize =
            delta.len() + external_values.iter().map(|v| v.len()).sum::<usize>();
        let mut out = Vec::with_capacity(header_size + total_payload);

        out.extend_from_slice(&MAGIC.to_le_bytes());
        out.extend_from_slice(&num_parts_u32.to_le_bytes());

        // Offset table: `num_parts` entries, no sentinel. offsets[0] is the
        // start of the delta (= header_size); subsequent entries are the
        // start of each external value.
        let mut offset = header_size as u64;
        out.extend_from_slice(&offset.to_le_bytes());
        offset += delta.len() as u64;
        for ext in external_values {
            out.extend_from_slice(&offset.to_le_bytes());
            offset += ext.len() as u64;
        }

        out.extend_from_slice(delta);
        for ext in external_values {
            out.extend_from_slice(ext);
        }

        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use base64::{engine::general_purpose::STANDARD as B64, Engine as _};

    #[test]
    fn empty_input_returns_empty_container() {
        let c = OmpfContainer::parse(&[]).unwrap();
        assert!(c.delta_bytes.is_empty());
        assert!(c.external_values.is_empty());
    }

    #[test]
    fn serialize_empty_returns_empty_vec() {
        assert!(OmpfContainer::serialize(&[], &[]).unwrap().is_empty());
    }

    #[test]
    fn roundtrip_delta_only() {
        let delta = b"flatbuffers-delta-data";
        let bytes = OmpfContainer::serialize(delta, &[]).unwrap();
        let c = OmpfContainer::parse(&bytes).unwrap();
        assert_eq!(c.delta_bytes, delta);
        assert!(c.external_values.is_empty());
    }

    #[test]
    fn roundtrip_delta_with_external_values() {
        let delta = b"delta";
        let ext1 = b"external-value-1";
        let ext2 = b"external-value-2";
        let bytes = OmpfContainer::serialize(delta, &[ext1, ext2]).unwrap();
        let c = OmpfContainer::parse(&bytes).unwrap();
        assert_eq!(c.delta_bytes, delta);
        assert_eq!(c.external_values.len(), 2);
        assert_eq!(c.external_values[0], ext1);
        assert_eq!(c.external_values[1], ext2);
    }

    #[test]
    fn header_layout_matches_spec_no_sentinel() {
        // Verify the emitted byte layout matches the public OMPF spec:
        // 4B magic | 4B num_parts | num_parts * 8B offsets | parts...
        let delta = b"delta";
        let ext = b"e";
        let bytes = OmpfContainer::serialize(delta, &[ext]).unwrap();
        // num_parts = 2, so header = 4 + 4 + 2*8 = 24 bytes. Total = 24 + 5 + 1 = 30.
        assert_eq!(bytes.len(), 30);
        assert_eq!(&bytes[0..4], b"OMPF");
        assert_eq!(u32::from_le_bytes(bytes[4..8].try_into().unwrap()), 2);
        // offset[0] = 24 (start of delta), offset[1] = 29 (start of ext).
        assert_eq!(u64::from_le_bytes(bytes[8..16].try_into().unwrap()), 24);
        assert_eq!(u64::from_le_bytes(bytes[16..24].try_into().unwrap()), 29);
    }

    #[test]
    fn bad_magic_returns_error() {
        let mut bytes = OmpfContainer::serialize(b"delta", &[]).unwrap();
        bytes[0] = 0x00;
        assert!(matches!(
            OmpfContainer::parse(&bytes),
            Err(OmpfError::BadMagic { .. })
        ));
    }

    #[test]
    fn truncated_before_magic_returns_error() {
        assert!(matches!(
            OmpfContainer::parse(&[0x46]),
            Err(OmpfError::Truncated { .. })
        ));
    }

    #[test]
    fn truncated_header_returns_error() {
        let bytes = MAGIC.to_le_bytes();
        assert!(matches!(
            OmpfContainer::parse(&bytes),
            Err(OmpfError::Truncated { .. })
        ));
    }

    #[test]
    fn zero_parts_returns_no_parts_error() {
        let mut out = Vec::new();
        out.extend_from_slice(&MAGIC.to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes());
        assert!(matches!(
            OmpfContainer::parse(&out),
            Err(OmpfError::NoParts)
        ));
    }

    #[test]
    fn offset_out_of_bounds_returns_error() {
        let mut bytes = OmpfContainer::serialize(b"delta", &[]).unwrap();
        // Overwrite offset[0] at byte 8 with a huge value.
        let huge: u64 = u64::MAX;
        bytes[8..16].copy_from_slice(&huge.to_le_bytes());
        assert!(matches!(
            OmpfContainer::parse(&bytes),
            Err(OmpfError::OffsetOutOfBounds { .. })
        ));
    }

    #[test]
    fn non_monotonic_offsets_return_error() {
        let delta = b"delta";
        let ext = b"ext";
        let mut bytes = OmpfContainer::serialize(delta, &[ext]).unwrap();
        // With num_parts=2, offsets are at bytes 8 and 16. Swap so offset[1] < offset[0].
        let o0 = u64::from_le_bytes(bytes[8..16].try_into().unwrap());
        let o1 = u64::from_le_bytes(bytes[16..24].try_into().unwrap());
        bytes[8..16].copy_from_slice(&o1.to_le_bytes());
        bytes[16..24].copy_from_slice(&o0.to_le_bytes());
        assert!(matches!(
            OmpfContainer::parse(&bytes),
            Err(OmpfError::OffsetNotMonotonic { .. })
        ));
    }

    #[test]
    fn truncated_offset_table_returns_error() {
        let mut out = Vec::new();
        out.extend_from_slice(&MAGIC.to_le_bytes());
        out.extend_from_slice(&1u32.to_le_bytes());
        out.push(0x00);
        assert!(matches!(
            OmpfContainer::parse(&out),
            Err(OmpfError::Truncated { .. })
        ));
    }

    #[test]
    fn zero_length_delta_part_is_accepted() {
        let bytes = OmpfContainer::serialize(b"", &[b"some-ext"]).unwrap();
        let c = OmpfContainer::parse(&bytes).unwrap();
        assert!(c.delta_bytes.is_empty());
        assert_eq!(c.external_values.len(), 1);
        assert_eq!(c.external_values[0], b"some-ext");
    }

    #[test]
    fn zero_length_external_value_is_accepted() {
        let bytes = OmpfContainer::serialize(b"delta", &[b""]).unwrap();
        let c = OmpfContainer::parse(&bytes).unwrap();
        assert_eq!(c.delta_bytes, b"delta");
        assert_eq!(c.external_values.len(), 1);
        assert!(c.external_values[0].is_empty());
    }

    #[test]
    fn roundtrip_single_part_identity() {
        let delta = b"deterministic-delta-payload";
        let serialized = OmpfContainer::serialize(delta, &[]).unwrap();
        let parsed = OmpfContainer::parse(&serialized).unwrap();
        assert_eq!(parsed.delta_bytes, delta);
        assert!(parsed.external_values.is_empty());
    }

    #[test]
    fn roundtrip_multi_part_identity() {
        let delta = b"main-delta";
        let ext0 = b"external-blob-zero";
        let ext1 = b"external-blob-one";
        let ext2 = b"external-blob-two";
        let serialized = OmpfContainer::serialize(delta, &[ext0, ext1, ext2]).unwrap();
        let parsed = OmpfContainer::parse(&serialized).unwrap();
        assert_eq!(parsed.delta_bytes, delta);
        assert_eq!(parsed.external_values.len(), 3);
        assert_eq!(parsed.external_values[0], ext0);
        assert_eq!(parsed.external_values[1], ext1);
        assert_eq!(parsed.external_values[2], ext2);
    }

    /// Regression test for b.jcd item 1: an 8-byte header claiming
    /// `num_parts = u32::MAX` must be rejected before any large allocation.
    /// Prior to the fix, the parser reached `Vec::with_capacity(num_parts)`
    /// and attempted to reserve ~34 GB.
    #[test]
    fn num_parts_overflow_rejected_before_allocation() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&MAGIC.to_le_bytes());
        bytes.extend_from_slice(&u32::MAX.to_le_bytes());
        assert_eq!(bytes.len(), HEADER_BASE);
        match OmpfContainer::parse(&bytes).err() {
            Some(OmpfError::NumPartsTooLarge {
                num_parts,
                max,
                size,
            }) => {
                assert_eq!(num_parts, u32::MAX as usize);
                assert_eq!(size, HEADER_BASE);
                assert_eq!(max, HEADER_BASE / OFFSET_SIZE);
            }
            other => panic!("expected NumPartsTooLarge, got {other:?}"),
        }
    }

    /// Regression test for b.jcd item 1: a header whose stated num_parts
    /// requires more offset-table bytes than exist in the buffer must be
    /// rejected with `NumPartsTooLarge`, not `Truncated`. This keeps the
    /// with_capacity cap tied to actual input size.
    #[test]
    fn num_parts_exceeding_input_size_rejected() {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&MAGIC.to_le_bytes());
        bytes.extend_from_slice(&1_000_000u32.to_le_bytes());
        // A few spurious tail bytes — far fewer than num_parts * 8.
        bytes.extend_from_slice(&[0u8; 16]);
        assert!(matches!(
            OmpfContainer::parse(&bytes),
            Err(OmpfError::NumPartsTooLarge { .. })
        ));
    }

    /// Regression test for b.jcd item 2: `offsets[0]` pointing inside the
    /// header/offset-table region must be rejected. Without this check, part
    /// bodies can alias onto header bytes and confuse downstream parsers.
    #[test]
    fn offset_zero_pointing_into_offset_table_rejected() {
        let mut bytes = OmpfContainer::serialize(b"delta", &[]).unwrap();
        // With num_parts=1, offsets table ends at byte 16. Overwrite offset[0]
        // with 0 so the "part" starts at the magic bytes.
        bytes[8..16].copy_from_slice(&0u64.to_le_bytes());
        match OmpfContainer::parse(&bytes).err() {
            Some(OmpfError::OffsetOverlapsTable {
                offset,
                table_end,
                num_parts,
            }) => {
                assert_eq!(offset, 0);
                assert_eq!(table_end, HEADER_BASE + OFFSET_SIZE);
                assert_eq!(num_parts, 1);
            }
            other => panic!("expected OffsetOverlapsTable, got {other:?}"),
        }
    }

    /// b.jcd item 2: a `offsets[0]` that lands exactly on `offset_table_end`
    /// must still be accepted — the check is `<`, not `<=`.
    #[test]
    fn offset_zero_equal_to_offset_table_end_is_accepted() {
        // serialize() places offset[0] == HEADER_BASE + num_parts * OFFSET_SIZE
        // (i.e. exactly at offset_table_end) for num_parts=1. Verify that the
        // boundary is inclusive.
        let bytes = OmpfContainer::serialize(b"x", &[]).unwrap();
        let o0 = u64::from_le_bytes(bytes[8..16].try_into().unwrap());
        assert_eq!(o0 as usize, HEADER_BASE + OFFSET_SIZE);
        assert!(OmpfContainer::parse(&bytes).is_ok());
    }

    /// b.jcd item 2 boundary: `offsets[0] = offset_table_end - 1` lands on the
    /// last byte of the offset table itself. Any value strictly less than
    /// `offset_table_end` must be rejected — this locks the `<` vs `<=` edge
    /// from the opposite side of the boundary test above.
    #[test]
    fn offset_zero_one_below_offset_table_end_rejected() {
        // num_parts = 1 → offsets_end = HEADER_BASE + OFFSET_SIZE = 16.
        // Craft a buffer large enough that OffsetOutOfBounds doesn't fire first.
        let mut bytes = OmpfContainer::serialize(b"payload", &[]).unwrap();
        let offsets_end = (HEADER_BASE + OFFSET_SIZE) as u64;
        let one_below = offsets_end - 1;
        bytes[8..16].copy_from_slice(&one_below.to_le_bytes());
        match OmpfContainer::parse(&bytes).err() {
            Some(OmpfError::OffsetOverlapsTable {
                offset,
                table_end,
                num_parts,
            }) => {
                assert_eq!(offset, one_below);
                assert_eq!(table_end as u64, offsets_end);
                assert_eq!(num_parts, 1);
            }
            other => panic!("expected OffsetOverlapsTable, got {other:?}"),
        }
    }

    /// b.jcd item 2: multi-part variant of the overlap check. The offsets_end
    /// computation must account for `num_parts * OFFSET_SIZE`, so an attacker
    /// who craft an `offsets[0]` that would be valid for num_parts=1 (e.g.
    /// 16) but falls inside a larger num_parts=3 offset table (end = 32)
    /// must still be rejected.
    #[test]
    fn multi_part_offset_zero_inside_larger_offset_table_rejected() {
        let delta = b"delta";
        let bytes = OmpfContainer::serialize(delta, &[b"e1", b"e2"]).unwrap();
        // num_parts = 3, offsets_end = 8 + 3*8 = 32. Overwrite offset[0] to 16
        // (valid for num_parts=1, but inside this 3-offset table).
        let mut poisoned = bytes.clone();
        poisoned[8..16].copy_from_slice(&16u64.to_le_bytes());
        match OmpfContainer::parse(&poisoned).err() {
            Some(OmpfError::OffsetOverlapsTable {
                offset,
                table_end,
                num_parts,
            }) => {
                assert_eq!(offset, 16);
                assert_eq!(table_end, HEADER_BASE + 3 * OFFSET_SIZE);
                assert_eq!(num_parts, 3);
            }
            other => panic!("expected OffsetOverlapsTable, got {other:?}"),
        }
    }

    /// b.jcd item 1: specifically exercise the boundary where `num_parts`
    /// equals the cap `data.len() / OFFSET_SIZE`. This must be accepted by the
    /// NumPartsTooLarge gate (the gate uses `>`, not `>=`) — rejection then
    /// falls through to a different error, not `NumPartsTooLarge`. Locks the
    /// `>` vs `>=` edge on the cap.
    #[test]
    fn num_parts_equal_to_max_cap_not_rejected_as_too_large() {
        // Craft a buffer of exactly 24 bytes: HEADER_BASE (8) + 2 * OFFSET_SIZE.
        // num_parts = 3 exactly matches data.len() / OFFSET_SIZE = 24/8 = 3,
        // so the cap check must not fire (3 > 3 is false).
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&MAGIC.to_le_bytes());
        bytes.extend_from_slice(&3u32.to_le_bytes());
        // Pad out to exactly 24 bytes with two offset-table slots. Parse will
        // reject this for a non-cap reason (e.g. Truncated when reading the
        // full offset table that requires 32 bytes) — what matters is that we
        // do NOT get `NumPartsTooLarge`.
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        assert_eq!(bytes.len(), 24);
        assert_eq!(bytes.len() / OFFSET_SIZE, 3);
        let err = OmpfContainer::parse(&bytes).err().unwrap();
        assert!(
            !matches!(err, OmpfError::NumPartsTooLarge { .. }),
            "expected a non-cap error; got {err:?}"
        );
    }

    /// b.jcd item 1 companion: `num_parts == max_parts + 1` must be rejected
    /// as `NumPartsTooLarge`. Pairs with the equal-to-cap test above to pin
    /// the `>` vs `>=` edge on both sides, the same way the offset-overlap
    /// tests pin the `<` vs `<=` edge on both sides.
    #[test]
    fn num_parts_one_above_cap_rejected_as_too_large() {
        // Same 24-byte buffer (data.len() / OFFSET_SIZE = 3), but num_parts = 4
        // — exactly one above the cap. Must fire the cap check.
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&MAGIC.to_le_bytes());
        bytes.extend_from_slice(&4u32.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        bytes.extend_from_slice(&0u64.to_le_bytes());
        assert_eq!(bytes.len(), 24);
        match OmpfContainer::parse(&bytes).err() {
            Some(OmpfError::NumPartsTooLarge {
                num_parts,
                max,
                size,
            }) => {
                assert_eq!(num_parts, 4);
                assert_eq!(max, 3);
                assert_eq!(size, 24);
            }
            other => panic!("expected NumPartsTooLarge, got {other:?}"),
        }
    }

    // Guard test: pinned by name in scripts/guard-tests.sh; rename or remove it there too.
    /// b.jcd item 3: explicitly assert that `.try_into().unwrap()` no longer
    /// appears on untrusted input slices. This is a mechanical grep-style
    /// guard; if a future refactor reintroduces the pattern the test fails at
    /// build time via the regex. Kept narrow: only flags `[0..X].try_into()`
    /// forms on byte-slice reads of input, not legitimate uses elsewhere.
    #[test]
    fn source_has_no_try_into_unwrap_on_input_slices() {
        let src = include_str!("ompf.rs");
        // Walk the `parse` function body only. The pattern we're banning is
        // `data[..].try_into().unwrap()` — a conversion from an input byte
        // slice into a fixed-size array that would panic on length mismatch.
        // The fix replaces these with explicit `copy_from_slice` into a
        // stack-allocated array.
        let parse_fn_start = src
            .find("pub fn parse(data: &'a [u8]) -> Result<Self, OmpfError> {")
            .expect("parse fn must exist");
        // Anchor to the next item's doc-comment header rather than a matching
        // brace pattern — rustfmt re-indents and nested blocks can both make
        // `\n    }\n` match inside parse() rather than at its close.
        let parse_fn_end = parse_fn_start
            + src[parse_fn_start..]
                .find("\n    /// Serialize an OMPF container to owned bytes.")
                .expect("next item (serialize) must follow parse");
        let parse_body = &src[parse_fn_start..parse_fn_end];
        assert!(
            !parse_body.contains(".try_into().unwrap()"),
            "OMPF parse() must not use `.try_into().unwrap()` on input bytes \
             (b.jcd item 3). Use copy_from_slice into a fixed-size array \
             instead."
        );
    }

    /// Regression test: a CreateLayer OMPF chunk captured from the C++
    /// live-edit service. The old Rust parser required an `num_parts + 1`
    /// offset table and rejected this chunk with `OffsetOutOfBounds` because
    /// it read the FlatBuffers data as a bogus sentinel. This test locks in
    /// the fix: no-sentinel Layout per the public Omniverse OMPF spec.
    #[test]
    fn parses_cpp_service_ompf_chunk() {
        const B64_CHUNK: &str = "T01QRgEAAAAQAAAAAAAAABwAAABPT0RGAAAAABAAGAAHAAAACAAMABAAFAAQAAAAAAAAASgAAAAYAAAADAAAAAQAAAAAAAAAAQAAACAAAAACAAAAvAAAAIQAAAACAAAABQUKABQADAAEAAgACgAAABAAAAAYAAAABQAAAAAAAAAGAAAAZmllbGQwAABAAAAARUlYUURSS0tNRFBWUFlPRFpLRE9HTU1ZREZJUURRUVVHTUtGQVhMRFlLRFhDRUJKR0RVTEpMVllRRlJYR1dNTgwAIAAMABQACAAHAAwAAAAAAAABGAAAAAEAAAAAAAAABQAAAAAAAAAAAAAABQAAAFByaW01AAoAEAAAAAgABAAKAAAADAAAAAEAAAAAAAAAAAAAAAAAAAA=";
        let bytes = B64.decode(B64_CHUNK).expect("valid base64");
        let c = OmpfContainer::parse(&bytes).expect("C++-emitted OMPF must parse");
        // num_parts=1, offset[0]=16 → delta spans bytes 16..len.
        assert_eq!(c.delta_bytes.len(), bytes.len() - 16);
        assert!(c.external_values.is_empty());
        // Sanity-check the FlatBuffers file_identifier "OODF" at offset 4
        // of the delta slice (FlatBuffers convention).
        assert_eq!(&c.delta_bytes[4..8], b"OODF");
    }

    // ── b.55p OMPF boundaries ────────────────────────────────────────────────
    //
    // The boundary tests below lock in the exact `offsets[0] > data.len()` vs
    // `offsets[0] == data.len()` split and the `num_parts > u32::MAX`
    // serialize guard. Pre-fix `serialize` used a silent `as u32` truncation
    // and returned a `Vec<u8>`; post-fix it returns `Result<Vec<u8>, _>` and
    // rejects with `SerializeNumPartsOverflow`.

    #[test]
    fn parse_num_parts_one_offset_equals_data_len_yields_empty_delta() {
        // Boundary: offsets[0] == data.len() is the "zero-length delta" shape.
        // With num_parts=1, the parser computes `delta_end = data.len()` and
        // the delta slice is `data[offset..delta_end]` = empty. This is
        // legal per SDD §8.4 and must NOT trip OffsetOutOfBounds.
        //
        // Build the buffer by hand: header (magic + num_parts + 1 offset)
        // with offsets[0] pointing exactly at data.len() = HEADER_BASE + OFFSET_SIZE.
        let header_size = HEADER_BASE + OFFSET_SIZE;
        let mut bytes = Vec::with_capacity(header_size);
        bytes.extend_from_slice(&MAGIC.to_le_bytes());
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&(header_size as u64).to_le_bytes());
        assert_eq!(bytes.len(), header_size);

        let c = OmpfContainer::parse(&bytes).expect("offset == data.len() boundary must parse");
        assert!(c.delta_bytes.is_empty());
        assert!(c.external_values.is_empty());
    }

    #[test]
    fn parse_num_parts_one_offset_one_past_data_len_rejects() {
        // One byte past data.len() must fail with OffsetOutOfBounds — this
        // locks in the `>` vs `>=` split at the data.len() boundary.
        let header_size = HEADER_BASE + OFFSET_SIZE;
        let mut bytes = Vec::with_capacity(header_size);
        bytes.extend_from_slice(&MAGIC.to_le_bytes());
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&((header_size + 1) as u64).to_le_bytes());

        let err = match OmpfContainer::parse(&bytes) {
            Ok(_) => panic!("offset = data.len() + 1 must be rejected"),
            Err(e) => e,
        };
        assert!(
            matches!(err, OmpfError::OffsetOutOfBounds { .. }),
            "offset = data.len() + 1 must return OffsetOutOfBounds, got {err:?}"
        );
    }

    /// Serialize `1 + external_values.len() = u32::MAX + 1` parts cannot be
    /// constructed in practice on a 64-bit host (would OOM long before), so
    /// the `SerializeNumPartsOverflow` arm is documented as unreachable in
    /// production. This test asserts the error arm is reachable IF triggered
    /// and locks in the Result contract at the type level.
    ///
    /// The test exercises the same `u32::try_from(num_parts)` path the real
    /// code uses by `assume` — since we can't actually build u32::MAX+1
    /// entries, we instead verify the contract via the wrapped check:
    /// a normal call path still returns Ok.
    #[test]
    fn serialize_normal_case_returns_ok_result() {
        // The post-fix type contract is `Result<Vec<u8>, OmpfError>`. A
        // normal serialize call must still return Ok; this protects against
        // a future regression that flips the default arm to Err or drops
        // the Result entirely.
        let res = OmpfContainer::serialize(b"delta", &[b"ext"]);
        assert!(res.is_ok(), "normal serialize must return Ok, got {res:?}");
    }

    /// Static compile-time assertion that the serialize signature returns a
    /// `Result<_, OmpfError>`. If a future refactor accidentally reverts to
    /// the pre-b.55p `-> Vec<u8>` shape (losing `SerializeNumPartsOverflow`),
    /// the `?` below fails to type-check and this test stops compiling.
    #[test]
    fn serialize_signature_returns_result() {
        fn _assert_returns_result() -> Result<(), OmpfError> {
            let _bytes = OmpfContainer::serialize(b"", &[])?;
            Ok(())
        }
    }
}
