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

//! The `commit_chain` header column: an object's fence identity.
//!
//! `obj:hdr` carries [`HEADER_FIELD_COMMIT_CHAIN`], a 40-lowercase-hex value
//! that names one materialization of an object and the exact history of
//! commits on top of it:
//!
//! - **Every materialization** — `CreateLayer`, an overwrite, a DR restore —
//!   writes [`fresh_commit_chain`], a new random value, in the header `HSET` of
//!   the shared materializer body (`layer::queue_object_state`). Two restores
//!   of the same checkpoint generation therefore carry different values even
//!   though they copy the same `created_at` and may land at the same `seq`.
//! - **Every commit**, an empty one included, advances it to
//!   [`commit_chain_fold`] of the previous value and the exact `update|…`
//!   payload that commit `XADD`s to the object's stream
//!   ([`update_stream_payload`]), in the same atomic unit as the header `HSET`
//!   that advances `seq`. Both commit paths fold: `update_layer_fast.lua` in
//!   its step-7 header `HSET` (`redis.sha1hex`), and the WATCH path in the
//!   `SetObjectHeader` `HSET` of its `MULTI` (this module's fold, over the
//!   value its `WATCH`ed header read carried). A refused commit writes nothing,
//!   the column included.
//!
//! Retirement's `deleted` stream entries are not commits and do not fold.
//!
//! So a holder of an earlier `(seq, commit_chain)` pair can fold the stream
//! entries after its `seq` and compare against the header: a mismatch means
//! the object was re-materialized, or the backing store rolled back and was
//! re-advanced by different commits, since the pair was taken. `seq` alone
//! cannot tell either case apart from ordinary progress.
//!
//! The column is deliberately NOT one of
//! [`HEADER_STRUCTURAL_FIELDS`](super::layer::HEADER_STRUCTURAL_FIELDS): an
//! object materialized before the column existed, or by an older replica that
//! does not write it, must not read as structurally incomplete (and therefore
//! restore-eligible). Such an object reads the column as `""`, and the chain
//! folds from there. The column is removed with the header, like every other
//! header field.
//!
//! SHA-1 here is an identity hash, not a security boundary: the values are
//! never authenticated and never leave the backing store. SHA-1 is used because
//! it is the digest Valkey's Lua runtime exposes (`redis.sha1hex`), and the fold
//! must be byte-identical whether a script or Rust computes it.
//!
//! [`HEADER_FIELD_COMMIT_CHAIN`]: super::layer::HEADER_FIELD_COMMIT_CHAIN

use sha1::{Digest, Sha1};

use super::layer::hex_encode_bytes;

/// The separator [`commit_chain_fold`] places between the previous value and
/// the payload. A single byte that can appear in neither a 40-hex value nor
/// the `""` of an object with no `commit_chain`, so the concatenation is
/// unambiguous at the boundary.
const FOLD_SEPARATOR: &[u8] = b"\n";

/// Advance a `commit_chain` by one commit.
///
/// Returns the lowercase 40-hex SHA-1 of `prev`, a single `\n`, then
/// `payload` — byte-identical to Lua's
/// `redis.sha1hex(prev .. "\n" .. payload)`. `prev` is `""` for an object whose
/// header has no `commit_chain` (Lua's `HGET` of an absent field returns
/// `false`, which a script must map to `""` before folding); `payload` is the
/// exact `update|…` string the commit `XADD`s.
///
/// The one Rust definition of the fold: a Rust-side writer or reader must call
/// this rather than re-derive it, so the Rust paths cannot drift from each
/// other, only (testably) from the script.
pub fn commit_chain_fold(prev: &str, payload: &str) -> String {
    let mut hasher = Sha1::new();
    hasher.update(prev.as_bytes());
    hasher.update(FOLD_SEPARATOR);
    hasher.update(payload.as_bytes());
    hex_encode_bytes(&hasher.finalize())
}

/// The `payload` field of a commit's stream entry:
/// `update|{object_id}|{seq}|{client_id}|{client_ts}|{server_ts}`.
///
/// The one Rust copy of the format. The WATCH path's `XADD` and CreateLayer's
/// initial `1-0` entry build their payload here, and every Rust-side
/// derivation of a successor `commit_chain` folds over this function's output,
/// so the bytes a Rust fold hashes cannot drift from the bytes the stream
/// carries. `update_layer_fast.lua` builds the same string by concatenation
/// (its contract invariant 3); `LayerNotification::parse` is the reader.
///
/// `client_id` is written verbatim: the validation layer refuses a client id
/// containing `|`, which would otherwise make the payload ambiguous to parse.
pub fn update_stream_payload(
    object_id: u64,
    seq: u64,
    client_id: &str,
    client_ts: u64,
    server_ts: u64,
) -> String {
    format!("update|{object_id}|{seq}|{client_id}|{client_ts}|{server_ts}")
}

/// A new `commit_chain` for a materialization: 160 random bits, rendered as 40
/// lowercase hex digits — the same width and alphabet as a fold result, so the
/// column has one shape whatever wrote it.
///
/// Drawn from `rand::random`, the same thread-local CSPRNG the commit token
/// (`layer::mint_commit_token`) uses. Random rather than derived from the
/// materialized state, because two materializations of identical state (two
/// restores of one checkpoint generation) must still differ.
pub fn fresh_commit_chain() -> String {
    hex_encode_bytes(&rand::random::<[u8; 20]>())
}

#[cfg(test)]
mod tests {
    use super::*;

    // Every expected digest below was computed outside this crate, with
    // Python's `hashlib.sha1(prev.encode("utf-8") + b"\n" +
    // payload.encode("utf-8")).hexdigest()`, so a vector cannot agree with the
    // fold merely by sharing its implementation. The cross-language pin against
    // Valkey's own `redis.sha1hex` is
    // `tests/integration/storage/commit_chain_tests.rs`.

    /// A 40-lowercase-hex `commit_chain`, as a fold result or a fresh value
    /// would carry. Not itself a fold result: the input to the hex-prev cases.
    const HEX_PREV: &str = "0123456789abcdef0123456789abcdef01234567";

    /// The exact shape of the stream payload a commit `XADD`s
    /// (`update|<object_id>|<new_seq>|<client_id>|<client_ts>|<server_ts>`).
    const PAYLOAD_A: &str = "update|42|7|client-abc|1000000|2000000";

    /// The commit after [`PAYLOAD_A`] on the same object.
    const PAYLOAD_B: &str = "update|42|8|client-abc|1000001|2000001";

    fn assert_commit_chain_shape(value: &str, what: &str) {
        assert_eq!(
            value.len(),
            40,
            "{what} must be 40 hex digits (160 bits), got {value:?}"
        );
        assert!(
            value
                .bytes()
                .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)),
            "{what} must be lowercase hex only, got {value:?}"
        );
    }

    #[test]
    fn commit_chain_fold_empty_prev_and_empty_payload_is_sha1_of_the_separator() {
        // SHA-1 of the single byte "\n": the separator is present even when both
        // sides are empty, and nothing else is hashed.
        assert_eq!(
            commit_chain_fold("", ""),
            "adc83b19e793491b1c6ea0fd8b46cd9f32e592fc"
        );
    }

    #[test]
    fn commit_chain_fold_empty_prev_with_update_payload_matches_reference_vector() {
        // `prev = ""` is an object with no `commit_chain` column (pre-feature,
        // or written by an older replica).
        assert_eq!(
            commit_chain_fold("", PAYLOAD_A),
            "8a1695c1d27a6322263c9d11b6cab54ef7024c64"
        );
    }

    #[test]
    fn commit_chain_fold_hex_prev_with_update_payload_matches_reference_vector() {
        assert_eq!(
            commit_chain_fold(HEX_PREV, PAYLOAD_A),
            "9837b9e61222943a95561f76133b34426dda6e62"
        );
    }

    #[test]
    fn commit_chain_fold_hex_prev_with_empty_payload_matches_reference_vector() {
        assert_eq!(
            commit_chain_fold(HEX_PREV, ""),
            "6cbc808cfbe07ad63e5781d59276dc0f7685bc10"
        );
    }

    #[test]
    fn commit_chain_fold_non_ascii_payload_hashes_its_utf8_bytes() {
        // A client id with a two-byte (é) and two three-byte (漢字) UTF-8
        // sequences: the fold hashes the payload's UTF-8 bytes, which is what
        // Lua's byte-string concatenation sees.
        let payload = "update|42|7|cli\u{e9}nt-\u{6f22}\u{5b57}|1000000|2000000";
        assert_eq!(
            commit_chain_fold("", payload),
            "b808816709d7fe1878e88ab165897ea37eb92c0e"
        );
    }

    #[test]
    fn commit_chain_fold_control_bytes_in_payload_match_reference_vector() {
        // U+00FF (UTF-8 `c3 bf`; a `&str` cannot carry a bare 0xff byte), NUL
        // and an embedded "\n" inside the payload: hashed as-is, with no
        // escaping and no truncation at NUL.
        let payload = "update|42|7|\u{ff}\u{0}\n|0|0";
        assert_eq!(
            commit_chain_fold(HEX_PREV, payload),
            "94fa800a63ff7a58cc7b0ac433ee5089c20dd7de"
        );
    }

    #[test]
    fn commit_chain_fold_two_commits_folded_in_opposite_orders_differ() {
        let a_then_b = commit_chain_fold(&commit_chain_fold("", PAYLOAD_A), PAYLOAD_B);
        let b_then_a = commit_chain_fold(&commit_chain_fold("", PAYLOAD_B), PAYLOAD_A);

        // Exact values first, so the inequality below is between the two
        // reference chains rather than between two arbitrary strings.
        assert_eq!(a_then_b, "58651f1c3f6b143d646a6fc6624346d0281397f3");
        assert_eq!(b_then_a, "6787df390a9f9bf2778184663b097a789d2e34bc");
        assert_ne!(
            a_then_b, b_then_a,
            "the chain must name the commit ORDER, not just the set of commits"
        );
    }

    #[test]
    fn fresh_commit_chain_is_40_lowercase_hex() {
        assert_commit_chain_shape(&fresh_commit_chain(), "fresh_commit_chain()");
    }

    #[test]
    fn fresh_commit_chain_two_calls_differ() {
        // 160 random bits: a collision between two draws is not a flake this
        // test can hit; equality means the value is not random.
        let first = fresh_commit_chain();
        let second = fresh_commit_chain();
        assert_ne!(
            first, second,
            "two materializations must never share a fresh commit_chain"
        );
    }
}
