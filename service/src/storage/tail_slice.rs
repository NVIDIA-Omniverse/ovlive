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

//! Bounded, paged read of a per-object live-update stream slice — the *read*
//! half of tail capture's completeness evidence (SDD SR-1.2 / SR-1.4).
//!
//! # What this module is
//!
//! One function, [`read_tail_slice`], which reads `le:{object_id}:obj:stream`
//! forward from `{S0+1}-0` in bounded pages, decodes each entry into owned
//! storage types, and stops on one of three conditions: the stream ran out, an
//! entry cap was reached, or a byte cap was reached. It does **not** classify.
//! Deciding which prefix of the returned slice is a sound, replayable
//! continuation of the lineage — and which of SR-1.2's refusal arms applies
//! otherwise — belongs to the checkpoint-side classifier, which consumes the
//! value this function returns.
//!
//! The split is a module-boundary requirement, not a stylistic one. CLAUDE.md's
//! *Module Dependency Rules* forbid a new concrete `storage → checkpoint` edge,
//! so nothing here names a checkpoint type: no `TailRecord`, no tail-codec
//! import, no checkpoint error enum. Everything crossing the boundary is a
//! storage-owned type built from `LayerNotification`, `String`, and `Bytes`.
//!
//! # The S0 contract
//!
//! `base_snap_seq` (S0) is **the newest restorable generation's sequence per
//! the checkpoint STORE LISTING** — the lineage head, of either kind (base or
//! tail). It is *never* the `le:{object_id}:obj:hdr` `last_checkpoint_seq`
//! cursor: SDD §22.7 step 8 tolerates that cursor lagging when its write fails,
//! so a lagging cursor would silently re-read commits an existing generation
//! already covers and mint overlapping tails.
//!
//! This function takes S0 as an argument and **does not enforce its
//! provenance** — enforcement lives at the capture unit's call site, which is
//! the only code that holds the store listing. Two further preconditions the
//! call site owns, recorded here so a reader of this module can see the whole
//! contract:
//!
//! - **S0 ≥ 1 always.** SR-7.1 roots every lineage at `snap_seq = 1`, and a
//!   layer whose listing cannot root a lineage refuses before any slice is read
//!   — `refused_no_base` when the listing is empty, `refused_lineage_unrootable`
//!   when it holds only tails or its head sits at sequence 0. This is
//!   load-bearing, not incidental: `CreateLayer`'s seed stream entry at `1-0`
//!   is `update`-typed and deliberately carries no `ompf` field
//!   while accompanying the *full initial layer content* (SR-7.3). Under the
//!   classifier's rules it would decode as an empty commit and replay as a
//!   no-op over real content. `S0 ≥ 1` is what keeps the read window
//!   `{S0+1}-0` from ever reaching back to `1-0`.
//! - **Post-restore fossil regions.** See *Cap lookahead* below.
//!
//! # Evidence discipline: never `XLEN`, never a count
//!
//! SR-1.2 pins the completeness evidence to *the returned slice itself*. This
//! module therefore issues `XRANGE` and nothing else — no `XLEN`, no `XINFO
//! STREAM`, no `XREVRANGE`, no count of any kind compared against anything. A
//! count is a claim about entries the reader never saw; the whole point of the
//! design is that only delivered entries count as evidence. A future change
//! that adds a length probe here to "sanity check" the slice would reintroduce
//! exactly the reasoning SR-1.2 removed.
//!
//! # Paging soundness (SR-1.2, in full)
//!
//! fred 10.1.0's `xrange_values` issues **exactly one** `XRANGE key start end
//! [COUNT n]` per call — no internal pagination and no re-issue (verified in
//! the vendored crate source: `commands/impls/streams.rs::xrange` builds one
//! command and returns one reply). Paging is therefore a loop over calls, and
//! the slice is assembled across several non-atomic round-trips. That is still
//! sound, for four reasons that must hold together:
//!
//! 1. **Each page is atomic server-side.** Valkey executes one `XRANGE` as a
//!    single command; a concurrent `XADD` or trim can never interleave *inside*
//!    one reply.
//! 2. **Delivered entries are irretractable.** Streams are append-only and
//!    nothing in this service issues `XDEL`. Once an entry is in our
//!    `Vec`, no later server-side event can invalidate it — which is why the
//!    check the classifier runs over the in-memory slice needs **no quiescence
//!    requirement** at all.
//! 3. **Trimming is head-only.** Both trims evict from the head: the entry
//!    cap's `MAXLEN ~` on every `XADD` (`LIVE_EDIT_LAYER_STREAM_MAXLEN`), and
//!    the relay byte budget's `XTRIM MINID ~` below a cut that never reaches
//!    the newest entry (`LIVE_EDIT_LAYER_STREAM_MAX_BYTES`,
//!    `storage::relay_retention`). Between two pages, a trim can therefore only remove
//!    entries at or below ones we already hold, or — if it outran us entirely —
//!    entries we have not yet read.
//! 4. **Both failure shapes land on a refusal arm.** A trim that outran the
//!    reader surfaces in exactly two ways: as a *first-page* in-window failure
//!    (the first returned entry's ID is not `{S0+1}-0`, which the classifier
//!    calls `trimmed`), or as a *page-boundary ascending jump* (the first entry
//!    of a later page skips the `+1` progression, which the classifier calls
//!    `seq_gap` and which refuses the entire accumulated prefix). Neither can
//!    produce a slice that looks contiguous while hiding a hole.
//!
//! # Preserve, never skip — the deliberate contrast with the subscriber loop
//!
//! The stream multiplexer's reader (`parse_payload` in
//! `src/storage/stream_mux/reader.rs`, the
//! ``stream entry missing `payload` field; skipping`` /
//! ``stream entry `payload` is not valid UTF-8; skipping`` arms) **skips**
//! undecodable entries and advances its subscribers' cursors past them. That is
//! right for a subscriber: one corrupt entry should not stall live delivery.
//!
//! It is exactly **wrong** here, and the difference is load-bearing. A skipped
//! entry would silently vanish from the evidence, leaving the classifier
//! looking at a slice whose IDs still read as a contiguous `+1` progression
//! across a hole it can no longer see — certifying a prefix that jumped over a
//! real commit. Every entry the server returned MUST appear in
//! [`TailSlice::entries`], in server order, decodable or not; anything that
//! fails to decode is preserved as [`TailEntryPayload::Undecodable`] and the
//! classifier terminates the prefix there. **Do not "harmonise" these two
//! loops.**
//!
//! # Byte accounting, and the caller's framing obligation
//!
//! [`TailStreamEntry::payload_bytes`] and [`TailSlice::payload_bytes`] count
//! **raw verbatim payload bytes only** — the byte length of the entry's `ompf`
//! stream field, and `0` when that field is absent (empty commits, sentinels).
//! Nothing else is counted: not the `payload` metadata field, not the stream
//! ID, not any envelope framing.
//!
//! Envelope framing headroom is the **caller's** to reserve when deriving
//! [`TailSliceCaps::max_payload_bytes`] from `LIVE_EDIT_CHECKPOINT_TAIL_MAX_BYTES`.
//! The tail envelope adds a fixed 44-byte header plus a fixed 21-byte prefix
//! per record on top of the raw payloads, so a caller that passed the artifact
//! cap through unreduced could build an artifact slightly over it. That
//! subtraction lives at the call site precisely so the tail codec — which owns
//! those two constants — stays out of `storage`.
//!
//! # Cap lookahead, and what it does *not* reach
//!
//! When a cap stops the read, this function issues one further `COUNT 1`
//! `XRANGE` and returns the result as [`TailSlice::lookahead`]. It is counted
//! against neither cap and is never part of `entries`. [`TailSlice::cap_hit`]
//! is `Some(_)` **only** when that probe returned an entry, so `cap_hit` means
//! precisely "genuinely truncated, with more pending" — a read that reached a
//! cap exactly as the stream ended reports `cap_hit == None`.
//!
//! Its sole purpose is to let the classifier apply SR-1.2's
//! refuse-everything-on-jump rule *at the cap boundary*. Without it, a cap
//! landing inside a post-restore fossil region would hide the ascending jump
//! immediately beyond it, and the capped prefix would extend a lineage with
//! commits from an abandoned pre-restore timeline — the precise outcome SR-1.2
//! forbids, because composing them mints a generation whose restore resurrects
//! state the live timeline has already superseded.
//!
//! **The lookahead closes only the *boundary* case.** A jump further beyond
//! the cap stays invisible to the slice by construction. The complementary
//! guard is a check on the object header's `restore_seq_bump` field being
//! non-zero — stamped by every restore (`src/checkpoint/apply.rs`, the
//! restore-seed path) and cleared once a post-restore generation lands
//! (`src/checkpoint/checkpointer.rs`) — and it belongs to the capture unit's
//! call site, **not here**. Recorded so it cannot be forgotten.
//!
//! # Pool
//!
//! Reads ride `client.pool`, the default multiplexed data pool, matching the
//! assembler's read posture and its stated reason (`src/checkpoint/assembler.rs`:
//! reads are read-only and benefit from multiplexing). SR-9.1's original
//! "maintenance pool" placement was superseded at the Epic-3 breakdown — the
//! maintenance pool defaults to size 1 and is dedicated to sweep/GC cadence, so
//! bounded paging there would serialize captures behind maintenance traffic.

use std::collections::HashMap;

use bytes::Bytes;
use fred::interfaces::StreamsInterface;

use crate::error::ServiceError;
use crate::storage::client::{map_snapshot_load_err, FredErrOrigin, ValkeyClient, POOL_LABEL_DATA};
use crate::storage::keys::KeySchema;
use crate::storage::notifications::{
    LayerNotification, NOTIFICATION_PAYLOAD_FIELD, RELAY_OMPF_FIELD,
};
use crate::storage::stream_mux::{paged_xrange_count, stream_entry_bytes};

/// The most entries one paging `XRANGE` requests. SR-1.2's measured page size:
/// 512 entries measured at 2-5 ms per page for 1-10 KiB entries against
/// single-node Valkey.
///
/// A ceiling, not the page size: each page's `COUNT` is byte-sized by
/// [`crate::storage::stream_mux::paged_xrange_count`] from the largest entry
/// read so far (the stream multiplexer's soft `ReadSizing::count_for` rule).
/// Soft, so it bounds a page only by entries already seen: after a run of
/// small entries the next page can still ask for 512 entries, and if those
/// are large ones, one reply carries 512 of them. That worst case is bounded
/// by the smaller of the retained window — about the relay byte budget
/// (`LIVE_EDIT_LAYER_STREAM_MAX_BYTES`) plus two `LIVE_EDIT_MAX_LAYER_SIZE`
/// commits — and this ceiling's 512 entries × `LIVE_EDIT_MAX_LAYER_SIZE`.
/// With the budget at 0 only the ceiling binds: about 50 GB at the 100 MB
/// `LIVE_EDIT_MAX_LAYER_SIZE` default. A page is a
/// *round-trip* bound, not a correctness bound — the entry and byte caps in
/// [`TailSliceCaps`] are what bound the slice. Tuning either trades per-call
/// latency against the number of round-trips; it changes no outcome.
pub const TAIL_SLICE_PAGE_COUNT: u64 = crate::storage::stream_mux::PAGED_XRANGE_MAX_COUNT;

/// 2^53 - 1. Mirrors `SEQUENCE_CEILING` (`src/domain/delta.rs`) and the Lua
/// fast path's `9007199254740991` literal, duplicated rather than imported:
/// the domain copy is function-local, and `storage` must not grow a dependency
/// on the checkpoint codec's copy. Both commit paths refuse to mint a sequence
/// above this (the Lua f64 integer limit), so no readable stream entry can
/// exist beyond it.
const SEQUENCE_CEILING: u64 = 9_007_199_254_740_991;

/// Bounds for one [`read_tail_slice`] call.
///
/// **Caps arrive as arguments; this module never reads config.** Keeping the
/// primitive config-free is what makes it callable (and unit-testable) without
/// a `Config`, and keeps the `LIVE_EDIT_CHECKPOINT_TAIL_*` knob derivation —
/// including the framing-headroom subtraction described in the module header —
/// in one place at the capture call site.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TailSliceCaps {
    /// Maximum number of entries the returned slice may carry. Derived from
    /// `LIVE_EDIT_CHECKPOINT_TAIL_MAX_ENTRIES`.
    pub max_entries: usize,
    /// Maximum total **raw verbatim payload bytes** the returned slice may
    /// carry — see the module header's byte-accounting section, and note that
    /// the caller must reserve envelope framing headroom below
    /// `LIVE_EDIT_CHECKPOINT_TAIL_MAX_BYTES` before passing this.
    pub max_payload_bytes: u64,
    /// The most entries one paging `XRANGE` requests; each page's `COUNT` is
    /// byte-sized below it ([`TAIL_SLICE_PAGE_COUNT`]).
    /// [`TAIL_SLICE_PAGE_COUNT`] is the measured default. Values below 1 are
    /// treated as 1 (a `COUNT 0` reply would never terminate the paging loop).
    pub page_count: u64,
}

/// Which cap stopped a read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TailSliceCap {
    /// [`TailSliceCaps::max_entries`] was reached.
    Entries,
    /// Admitting the next entry would have pushed the slice past
    /// [`TailSliceCaps::max_payload_bytes`]. That entry is **excluded**, never
    /// partially included.
    Bytes,
}

/// The decoded form of one stream entry, or the reason it could not be decoded.
///
/// See the module header: an entry that fails to decode is **preserved** here
/// rather than skipped, so the classifier can terminate the prefix at it
/// instead of silently stepping over a hole.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TailEntryPayload {
    /// A successfully parsed notification. `Update` variants carry their
    /// verbatim `ompf` bytes in `relay_ompf` when the entry had that field.
    Notification(LayerNotification),
    /// The entry could not be decoded: no `payload` field, a non-UTF-8
    /// `payload` field, or a `payload` field that failed
    /// [`LayerNotification::parse`]. `reason` is for logs and refusal context
    /// only — the classifier treats every `Undecodable` identically.
    Undecodable { reason: String },
}

/// One entry of a tail slice, in the order the server returned it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TailStreamEntry {
    /// The Valkey stream ID exactly as returned (`{ms}-{n}` for auto-ID
    /// sentinels, `{seq}-0` for commits). **Authoritative** for the entry's
    /// sequence — never re-derived from the payload text.
    pub stream_id: String,
    /// Decoded payload, or the reason decoding failed.
    pub payload: TailEntryPayload,
    /// Byte length of this entry's `ompf` field; `0` when absent. Raw verbatim
    /// payload bytes only — see the module header.
    pub payload_bytes: u64,
}

/// The result of one bounded, paged tail read.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TailSlice {
    /// Every entry the server returned inside the read window, in server
    /// order, 1:1 — nothing dropped, nothing reordered, nothing skipped.
    pub entries: Vec<TailStreamEntry>,
    /// The one entry immediately past a cap stop, if a cap stopped the read
    /// and an entry was still pending. Counted against neither cap and never
    /// part of `entries`. See the module header for why it exists and what it
    /// does not reach.
    pub lookahead: Option<TailStreamEntry>,
    /// Which cap stopped the read, and `None` when the stream simply ran out.
    /// `Some(_)` **only** when `lookahead` is `Some(_)`.
    pub cap_hit: Option<TailSliceCap>,
    /// Sum of `entries[..].payload_bytes`. Excludes `lookahead`.
    pub payload_bytes: u64,
    /// Number of paging `XRANGE` calls issued. Excludes the `COUNT 1`
    /// lookahead probe. Diagnostic only — never evidence.
    pub pages_read: u32,
}

/// Read the bounded stream slice starting at `{base_snap_seq + 1}-0`.
///
/// `base_snap_seq` is S0 — the newest restorable generation per the checkpoint
/// **store listing**, never the `obj:hdr` cursor. The provenance of that value,
/// and the `S0 >= 1` precondition that keeps the read window clear of
/// `CreateLayer`'s ompf-less `1-0` seed entry, are the caller's to guarantee;
/// see the module header for the full contract and why it is load-bearing.
///
/// # Behaviour
///
/// Pages forward with `XRANGE key start + COUNT n`, where `n` is byte-sized
/// from the largest entry read so far and at most `caps.page_count`
/// ([`crate::storage::stream_mux::paged_xrange_count`]), advancing the
/// cursor to `({last_observed_id}` — the **observed** ID, never a synthesised
/// `{seq+1}-0`, because auto-ID sentinels (`deleted` / overwrite markers) carry
/// `{ms}-{n}` IDs and a synthesised cursor would step straight past one. Stops
/// when a page comes back shorter than the `COUNT` it asked for (stream
/// exhausted), when
/// `entries.len()` reaches `caps.max_entries`, or when admitting the next entry
/// would push the running byte total past `caps.max_payload_bytes` — in which
/// case that entry is **excluded**, never partially included.
///
/// A single first entry whose `ompf` field alone exceeds
/// `caps.max_payload_bytes` is likewise excluded, and logs one WARN. That layer
/// is SR-8.3's documented residual class: it will refuse observably on every
/// tick rather than capture a partial commit. The two operator levers are
/// `LIVE_EDIT_CHECKPOINT_TAIL_MAX_BYTES` (raise it above the largest single
/// commit) and `LIVE_EDIT_MAX_LAYER_SIZE` (bound how large a single commit may
/// get); the WARN names both.
///
/// # Errors
///
/// - [`ServiceError::Internal`] when `base_snap_seq + 1` would exceed the
///   `2^53 - 1` sequence ceiling both commit paths enforce — there is no
///   readable window above it.
/// - Whatever `map_snapshot_load_err` classifies a failed `XRANGE` as:
///   [`ServiceError::Aborted`] for transient I/O (the in-band retry signal),
///   [`ServiceError::Internal`] otherwise.
pub async fn read_tail_slice(
    client: &ValkeyClient,
    keys: &KeySchema,
    object_id: u64,
    base_snap_seq: u64,
    caps: TailSliceCaps,
) -> Result<TailSlice, ServiceError> {
    // `max_entries == 0` would refuse every capture while reporting itself
    // configured; `Config::from_env` zero-rejects the knob it comes from.
    debug_assert!(
        caps.max_entries >= 1,
        "TailSliceCaps::max_entries must be >= 1 (LIVE_EDIT_CHECKPOINT_TAIL_MAX_ENTRIES \
         zero-rejects); got 0"
    );
    debug_assert!(
        caps.page_count >= 1,
        "TailSliceCaps::page_count must be >= 1; got 0"
    );
    // Release-safe floor for the same hazard: `COUNT 0` returns an empty reply
    // that would compare `0 < 0 == false` and spin the paging loop forever.
    let max_page_count = caps.page_count.max(1);

    let start_seq = match base_snap_seq.checked_add(1) {
        Some(seq) if seq <= SEQUENCE_CEILING => seq,
        _ => {
            return Err(ServiceError::Internal(format!(
                "tail slice read for object {object_id}: base_snap_seq {base_snap_seq} leaves no \
                 readable window below the {SEQUENCE_CEILING} sequence ceiling both commit paths \
                 enforce"
            )));
        }
    };

    let stream_key = keys.layer_stream(object_id);
    let start_id = format!("{start_seq}-0");

    let mut cursor = start_id.clone();
    let mut entries: Vec<TailStreamEntry> = Vec::new();
    let mut payload_bytes: u64 = 0;
    let mut pages_read: u32 = 0;
    let mut cap_hit: Option<TailSliceCap> = None;
    // The last ID we ADMITTED, which is what both the page cursor and the
    // lookahead probe anchor on. In the no-cap case it is also the last ID the
    // server returned, because a full page is admitted in full.
    let mut last_admitted_id: Option<String> = None;
    // The largest entry any page has returned, sizing the next page's `COUNT`
    // (`stream_mux::paged_xrange_count`). Diagnostic sizing only — never
    // evidence.
    let mut largest_entry_bytes: Option<u64> = None;

    'paging: loop {
        let page_count = paged_xrange_count(largest_entry_bytes).min(max_page_count);
        // Field VALUES decode as `Bytes`, not `String`: the per-object stream
        // carries a binary `ompf` field alongside the UTF-8 `payload` field
        // (b.gze), and a `String` decode would fail the whole slice on the OMPF
        // bytes' invalid UTF-8 — poisoning every entry over one binary field.
        // Same reason `subscribe.rs`'s rollback-mode in-window replay decodes
        // as `Bytes`.
        let page: Vec<(String, HashMap<String, Bytes>)> = client
            .pool
            .xrange_values(&stream_key, cursor.clone(), "+", Some(page_count))
            .await
            .map_err(|e| {
                map_snapshot_load_err(
                    FredErrOrigin::client(client, POOL_LABEL_DATA),
                    "XRANGE tail slice",
                    e,
                )
            })?;

        pages_read = pages_read.saturating_add(1);
        let page_len = page.len() as u64;
        largest_entry_bytes = page
            .iter()
            .map(|(id, fields)| stream_entry_bytes(id, fields))
            .chain(largest_entry_bytes)
            .max();

        for (id, fields) in page {
            if entries.len() >= caps.max_entries {
                cap_hit = Some(TailSliceCap::Entries);
                break 'paging;
            }

            let entry_bytes = fields
                .get(RELAY_OMPF_FIELD)
                .map_or(0u64, |b| b.len() as u64);
            if payload_bytes.saturating_add(entry_bytes) > caps.max_payload_bytes {
                if entries.is_empty() {
                    // SR-8.3's residual class: one commit whose verbatim bytes
                    // do not fit the per-tail artifact budget at all. Excluded
                    // whole (never partially included), so this layer refuses
                    // observably every tick until an operator moves one of the
                    // two byte knobs named below.
                    tracing::warn!(
                        object_id = object_id,
                        stream_id = %id,
                        entry_payload_bytes = entry_bytes,
                        max_payload_bytes = caps.max_payload_bytes,
                        "tail slice: first stream entry alone exceeds the tail byte budget; \
                         excluding it, so this layer cannot be tail-captured. Raise \
                         LIVE_EDIT_CHECKPOINT_TAIL_MAX_BYTES above the largest single commit, or \
                         lower LIVE_EDIT_MAX_LAYER_SIZE to bound how large one commit may get."
                    );
                }
                cap_hit = Some(TailSliceCap::Bytes);
                break 'paging;
            }

            payload_bytes = payload_bytes.saturating_add(entry_bytes);
            entries.push(TailStreamEntry {
                stream_id: id.clone(),
                payload: decode_entry_payload(&fields),
                payload_bytes: entry_bytes,
            });
            last_admitted_id = Some(id);
        }

        // A short page means the server had nothing more to give: `XRANGE`
        // with `COUNT n` returns exactly `min(n, available)`.
        if page_len < page_count {
            break;
        }
        // Full page, no cap fired, so every entry was admitted and
        // `last_admitted_id` is the page's last observed ID. Exclusive start
        // (`(id`, Valkey 7.2+) advances past it without re-reading it.
        let Some(last) = last_admitted_id.as_deref() else {
            // Unreachable: a full page with `page_count >= 1` admitted at least
            // one entry. Break rather than spin on an unchanged cursor.
            break;
        };
        cursor = format!("({last}");
    }

    let lookahead = if cap_hit.is_some() {
        // Anchor the probe so it returns *the entry the cap refused*: exclusive
        // of the last admitted entry, or — when the byte cap excluded the very
        // first entry and there is nothing admitted to anchor on — the original
        // inclusive window start, which returns that oversize entry itself.
        //
        // That second case is load-bearing downstream: it is the only way the
        // classifier can tell an oversize-first-entry read (empty `entries`,
        // `cap_hit == Some(Bytes)`) apart from a genuinely empty window (empty
        // `entries`, `cap_hit == None`). Folding the two together would point
        // the operator at the stream-retention knob for a problem that is
        // entirely about commit size.
        let probe_cursor = match last_admitted_id.as_deref() {
            Some(last) => format!("({last}"),
            None => start_id,
        };
        let probe: Vec<(String, HashMap<String, Bytes>)> = client
            .pool
            .xrange_values(&stream_key, probe_cursor, "+", Some(1))
            .await
            .map_err(|e| {
                map_snapshot_load_err(
                    FredErrOrigin::client(client, POOL_LABEL_DATA),
                    "XRANGE tail-slice lookahead probe",
                    e,
                )
            })?;

        probe
            .into_iter()
            .next()
            .map(|(id, fields)| TailStreamEntry {
                stream_id: id,
                payload_bytes: fields
                    .get(RELAY_OMPF_FIELD)
                    .map_or(0u64, |b| b.len() as u64),
                payload: decode_entry_payload(&fields),
            })
    } else {
        None
    };

    // `cap_hit` must mean "genuinely truncated, with more pending". A cap that
    // landed exactly on the end of the stream — or whose pending entry was
    // trimmed away between the page read and the probe — is not a truncation.
    if lookahead.is_none() {
        cap_hit = None;
    }

    Ok(TailSlice {
        entries,
        lookahead,
        cap_hit,
        payload_bytes,
        pages_read,
    })
}

/// Decode one stream entry's fields into a [`TailEntryPayload`].
///
/// Total by construction: every failure mode becomes
/// [`TailEntryPayload::Undecodable`], never an `Err` and never a skip. See the
/// module header for why preserving these entries is load-bearing.
fn decode_entry_payload(fields: &HashMap<String, Bytes>) -> TailEntryPayload {
    let Some(raw) = fields.get(NOTIFICATION_PAYLOAD_FIELD) else {
        return TailEntryPayload::Undecodable {
            reason: format!("missing `{NOTIFICATION_PAYLOAD_FIELD}` field"),
        };
    };
    let text = match std::str::from_utf8(raw) {
        Ok(text) => text,
        Err(e) => {
            return TailEntryPayload::Undecodable {
                reason: format!("`{NOTIFICATION_PAYLOAD_FIELD}` field is not valid UTF-8: {e}"),
            };
        }
    };
    match LayerNotification::parse(text) {
        // The writer's raw inbound OMPF rides on the separate binary `ompf`
        // field and is attached here, exactly as the subscriber loop does.
        // A no-op on `Deleted` sentinels, which never carry one.
        Ok(notification) => TailEntryPayload::Notification(
            notification.with_relay_ompf(fields.get(RELAY_OMPF_FIELD).cloned()),
        ),
        Err(e) => TailEntryPayload::Undecodable {
            reason: format!("`{NOTIFICATION_PAYLOAD_FIELD}` field failed to parse: {e}"),
        },
    }
}

#[cfg(test)]
mod tests {
    //! In-`src` tests for [`read_tail_slice`], against real Valkey via
    //! `TestValkey` (the `src/storage/counters.rs` precedent — an in-crate
    //! `#[cfg(test)]` module gets `test-support` for free).
    //!
    //! **Why real Valkey and not a fake.** Everything this primitive is
    //! responsible for is a property of *how Valkey answers `XRANGE`*: that
    //! `COUNT n` returns exactly `min(n, available)` and so a short page means
    //! exhaustion; that the exclusive `(id` start form advances past an observed
    //! ID without re-reading it; that an auto-ID sentinel's `{ms}-{n}` sorts
    //! where it does relative to explicit `{seq}-0` commit IDs; and that a
    //! binary `ompf` field survives a `Bytes` decode alongside a UTF-8 `payload`
    //! field. A hand-rolled fake would encode the same beliefs the code under
    //! test encodes, and would agree with it whether or not either is right.
    //!
    //! The *classifier* that consumes these slices needs none of this: it is a
    //! pure function over the returned value, and its exhaustive arm matrix
    //! lives in [`crate::checkpoint::tail_classify`]'s own test module with no
    //! I/O at all.
    //!
    //! **Namespace discipline (shared Valkey).** Every case touches only
    //! `tv.key_schema().layer_stream(OBJECT_ID)`, which carries the fixture's
    //! per-test UUID prefix. Nothing here issues `FLUSHALL` / `FLUSHDB` — the
    //! backing instance is shared with concurrently-running agents and CI jobs,
    //! and `TestValkey`'s `Drop` already SCAN+UNLINKs the prefix. The
    //! multi-thread runtime flavour is load-bearing for that cleanup: `Drop`
    //! skips it entirely on a current-thread runtime.
    //!
    //! **No timing assertions and no sleeps.** Every case plants its entries,
    //! reads once, and asserts on the returned value; `pages_read` is the
    //! round-trip observable, which is a count, not a clock.
    use super::*;
    use crate::storage::notifications::DELETED_STREAM_PAYLOAD;
    use crate::storage::test_support::TestValkey;
    use fred::types::streams::XCap;
    use fred::types::Value;

    /// One object id for the whole module — key isolation comes from the
    /// fixture's per-test prefix, not from the id.
    const OBJECT_ID: u64 = 9001;
    /// S0 for the matrix. `> 1`, matching the production precondition, so the
    /// window starts at `8-0` and never reaches `CreateLayer`'s `1-0` seed.
    const S0: u64 = 7;
    /// First sequence the window may contain.
    const FIRST: u64 = S0 + 1;
    const SERVER_TS: i64 = 1_700_000_000_000;

    fn caps(max_entries: usize, max_payload_bytes: u64, page_count: u64) -> TailSliceCaps {
        TailSliceCaps {
            max_entries,
            max_payload_bytes,
            page_count,
        }
    }

    /// Caps generous enough that neither fires — for the cases about paging.
    fn uncapped(page_count: u64) -> TailSliceCaps {
        caps(1024, 1 << 20, page_count)
    }

    /// Raw `XADD` against the fixture's stream key. Returns the ID the server
    /// assigned, which is what the auto-ID sentinel cases need.
    async fn xadd(tv: &TestValkey, id: &str, fields: Vec<(&str, Value)>) -> String {
        let stream_key = tv.key_schema().layer_stream(OBJECT_ID);
        tv.client()
            .pool
            .xadd(&stream_key, false, XCap::from(None::<()>), id, fields)
            .await
            .expect("XADD must succeed")
    }

    /// `n` bytes of verbatim payload — content is irrelevant, length is not.
    fn ompf_of(n: usize) -> Vec<u8> {
        vec![b'x'; n]
    }

    /// Plant a well-formed commit entry at the explicit `{seq}-0` ID both
    /// commit paths use, in the field shape `src/storage/notifications.rs`
    /// documents: UTF-8 `payload` metadata plus an optional binary `ompf`.
    async fn plant_commit(tv: &TestValkey, seq: u64, ompf: Option<&[u8]>) {
        let payload = format!(
            "update|{OBJECT_ID}|{seq}|matrix-client|0|{}",
            SERVER_TS + seq as i64
        );
        let mut fields: Vec<(&str, Value)> = vec![(NOTIFICATION_PAYLOAD_FIELD, payload.into())];
        if let Some(bytes) = ompf {
            fields.push((
                RELAY_OMPF_FIELD,
                Value::Bytes(Bytes::copy_from_slice(bytes)),
            ));
        }
        // The returned id is the one we asked for, so it carries no
        // information; the sentinel helper below is the case that needs it.
        xadd(tv, &format!("{seq}-0"), fields).await;
    }

    /// Plant the layer-retirement sentinel exactly as its writers do: auto ID,
    /// bare `deleted` payload, no `ompf` field.
    async fn plant_sentinel(tv: &TestValkey) -> String {
        xadd(
            tv,
            "*",
            vec![(NOTIFICATION_PAYLOAD_FIELD, DELETED_STREAM_PAYLOAD.into())],
        )
        .await
    }

    async fn read(tv: &TestValkey, caps: TailSliceCaps) -> TailSlice {
        read_tail_slice(tv.client(), tv.key_schema(), OBJECT_ID, S0, caps)
            .await
            .expect("tail slice read must succeed")
    }

    #[track_caller]
    fn assert_commit(entry: &TailStreamEntry, seq: u64) {
        assert_eq!(entry.stream_id, format!("{seq}-0"), "wrong stream id");
        match &entry.payload {
            TailEntryPayload::Notification(LayerNotification::Update { sequence_num, .. }) => {
                assert_eq!(*sequence_num, seq, "payload sequence disagrees with the id")
            }
            other => panic!("expected a decoded update at {seq}-0, got {other:?}"),
        }
    }

    #[track_caller]
    fn assert_undecodable(entry: &TailStreamEntry, stream_id: &str) {
        assert_eq!(entry.stream_id, stream_id, "wrong stream id");
        assert!(
            matches!(entry.payload, TailEntryPayload::Undecodable { .. }),
            "entry {stream_id} must be PRESERVED as Undecodable, got {:?}",
            entry.payload
        );
    }

    fn ids(slice: &TailSlice) -> Vec<&str> {
        slice.entries.iter().map(|e| e.stream_id.as_str()).collect()
    }

    // ── 21-23: paging ───────────────────────────────────────────────────────

    /// Case 21 — a prefix longer than one page is assembled across the page
    /// boundary with server order preserved.
    ///
    /// The `pages_read > 1` assertion is the anti-vacuity guard: with a generous
    /// `page_count` this test would pass having exercised no boundary at all,
    /// and the exclusive-`(id`-cursor logic — the part that can silently
    /// duplicate or skip an entry — would go untested.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn read_tail_slice_prefix_spanning_multiple_pages_preserves_order() {
        let tv = TestValkey::setup().await;
        for seq in FIRST..FIRST + 5 {
            plant_commit(&tv, seq, Some(&ompf_of(8))).await;
        }

        let slice = read(&tv, uncapped(2)).await;

        assert_eq!(
            ids(&slice),
            vec!["8-0", "9-0", "10-0", "11-0", "12-0"],
            "entries must be 1:1 with the server's reply, in server order"
        );
        assert!(
            slice.pages_read > 1,
            "page_count=2 over 5 entries must cross a page boundary; pages_read={} \
             means this case proved nothing about paging",
            slice.pages_read
        );
        assert_eq!(slice.pages_read, 3, "2 full pages + 1 short page");
        assert_eq!(slice.cap_hit, None, "neither cap was near");
        assert_eq!(slice.lookahead, None, "no cap, so no lookahead probe");
        assert_eq!(slice.payload_bytes, 5 * 8);
    }

    /// Case 22 — the entry cap stops the *paging*, not just the output.
    ///
    /// A read that over-fetched every page and truncated afterwards would return
    /// an identical `entries` vec, so `entries.len()` cannot catch it;
    /// `pages_read` can. With 10 planted entries at `page_count = 2`, an
    /// over-reading implementation issues 5-6 pages where 2 suffice — on a hot
    /// layer with a full retention window that is the difference between a
    /// bounded read and dragging thousands of entries per tick.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn read_tail_slice_stops_issuing_pages_once_the_entry_cap_is_reached() {
        let tv = TestValkey::setup().await;
        for seq in FIRST..FIRST + 10 {
            plant_commit(&tv, seq, Some(&ompf_of(4))).await;
        }

        let slice = read(&tv, caps(3, 1 << 20, 2)).await;

        assert_eq!(ids(&slice), vec!["8-0", "9-0", "10-0"]);
        assert_eq!(
            slice.pages_read, 2,
            "the cap must stop the paging loop, not merely truncate its result"
        );
        assert_eq!(slice.cap_hit, Some(TailSliceCap::Entries));
    }

    /// Case 23 — a page shorter than the `COUNT` it asked for means the server
    /// had nothing more to give (`XRANGE COUNT n` returns exactly
    /// `min(n, available)`), so
    /// the loop ends there. An implementation that issued one more "just to be
    /// sure" would read 3 pages here instead of 2 — one wasted round trip per
    /// capture, per layer, per tick.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn read_tail_slice_short_final_page_ends_the_read_without_an_extra_call() {
        let tv = TestValkey::setup().await;
        for seq in FIRST..FIRST + 3 {
            plant_commit(&tv, seq, Some(&ompf_of(4))).await;
        }

        let slice = read(&tv, uncapped(2)).await;

        assert_eq!(ids(&slice), vec!["8-0", "9-0", "10-0"]);
        assert_eq!(
            slice.pages_read, 2,
            "one full page then one short page; the short page terminates the loop"
        );
        assert_eq!(slice.cap_hit, None);
    }

    /// Case 23b (b.yvh) — page sizes follow entry sizes. Each page's `COUNT`
    /// is byte-sized from the largest entry read so far
    /// (`stream_mux::paged_xrange_count`, at most `caps.page_count`), so once
    /// a large entry has been read, later pages read a few entries at a time
    /// rather than the [`TAIL_SLICE_PAGE_COUNT`] ceiling. (The rule is soft:
    /// a page sized after small entries can still return large ones — see
    /// [`TAIL_SLICE_PAGE_COUNT`].)
    ///
    /// The first page's `COUNT` is 64 (the 1 MiB page target over the 16 KiB
    /// default estimate), and its 64th entry is 600 KiB. That sizes every
    /// later page at one entry (1 MiB over 600 KiB): the three entries after
    /// it take three pages, and a fifth finds the stream's end. A reader that
    /// asked for the 512-entry ceiling every time reads all 67 in one page.
    /// The page sizes change no outcome: the 67 entries come back 1:1.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn read_tail_slice_sizes_each_page_from_the_largest_entry_read_so_far() {
        const LARGE: usize = 600 * 1024;
        let tv = TestValkey::setup().await;
        for seq in FIRST..FIRST + 63 {
            plant_commit(&tv, seq, Some(&ompf_of(8))).await;
        }
        plant_commit(&tv, FIRST + 63, Some(&ompf_of(LARGE))).await;
        for seq in FIRST + 64..FIRST + 67 {
            plant_commit(&tv, seq, Some(&ompf_of(8))).await;
        }

        let slice = read(&tv, caps(1024, 1 << 30, TAIL_SLICE_PAGE_COUNT)).await;

        let want: Vec<String> = (FIRST..FIRST + 67).map(|seq| format!("{seq}-0")).collect();
        assert_eq!(
            ids(&slice),
            want.iter().map(String::as_str).collect::<Vec<_>>(),
            "the slice is 1:1 with the stream whatever the page sizes"
        );
        assert_eq!(
            slice.pages_read, 5,
            "64 entries, then one entry per page after the 600 KiB one, then the empty page \
             that ends the read"
        );
        assert_eq!(slice.cap_hit, None);
    }

    // ── 24: preserve, never skip ────────────────────────────────────────────

    /// Case 24 — the opposite of the stream reader's skip (`parse_payload` in
    /// `storage::stream_mux::reader`), over all three decode
    /// failure modes: missing `payload` field, non-UTF-8 `payload`, and a
    /// `payload` that fails `LayerNotification::parse`.
    ///
    /// The subscriber loop **skips** these and advances its cursor past them,
    /// which is right for live delivery. Here it would be silent corruption: a
    /// skipped entry vanishes from the evidence, and the classifier then sees a
    /// slice whose IDs still read as a contiguous `+1` progression across a hole
    /// it can no longer see — certifying a prefix that jumped a real commit.
    /// Every returned entry must appear, in server order, decodable or not.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn read_tail_slice_undecodable_entries_are_preserved_in_order_never_skipped() {
        let tv = TestValkey::setup().await;
        plant_commit(&tv, FIRST, Some(&ompf_of(8))).await;
        // No `payload` field at all — only the binary relay field.
        xadd(
            &tv,
            "9-0",
            vec![(
                RELAY_OMPF_FIELD,
                Value::Bytes(Bytes::copy_from_slice(&ompf_of(16))),
            )],
        )
        .await;
        // A `payload` field that is not valid UTF-8.
        xadd(
            &tv,
            "10-0",
            vec![(
                NOTIFICATION_PAYLOAD_FIELD,
                Value::Bytes(Bytes::from_static(&[0xff, 0xfe, 0x00])),
            )],
        )
        .await;
        // Valid UTF-8 that `LayerNotification::parse` rejects.
        xadd(
            &tv,
            "11-0",
            vec![(NOTIFICATION_PAYLOAD_FIELD, "not-a-notification".into())],
        )
        .await;
        plant_commit(&tv, FIRST + 4, Some(&ompf_of(8))).await;

        let slice = read(&tv, uncapped(TAIL_SLICE_PAGE_COUNT)).await;

        assert_eq!(
            ids(&slice),
            vec!["8-0", "9-0", "10-0", "11-0", "12-0"],
            "no entry may be dropped, and none may be reordered"
        );
        assert_commit(&slice.entries[0], FIRST);
        assert_undecodable(&slice.entries[1], "9-0");
        assert_undecodable(&slice.entries[2], "10-0");
        assert_undecodable(&slice.entries[3], "11-0");
        assert_commit(&slice.entries[4], FIRST + 4);
        assert_eq!(
            slice.entries[1].payload_bytes, 16,
            "byte accounting reads the `ompf` field directly, so it holds even \
             for an entry whose `payload` never decoded"
        );
    }

    // ── 25-26: cap semantics and the lookahead probe ────────────────────────

    /// Case 25a — `cap_hit` means "genuinely truncated, **with more pending**".
    /// A read whose cap coincides with the end of the stream truncated nothing,
    /// so it reports `None` and the classifier treats it as an ordinary
    /// `captured`, not `captured_capped`.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn read_tail_slice_cap_reached_exactly_at_stream_end_reports_no_cap_hit() {
        let tv = TestValkey::setup().await;
        for seq in FIRST..FIRST + 3 {
            plant_commit(&tv, seq, Some(&ompf_of(4))).await;
        }

        let slice = read(&tv, caps(3, 1 << 20, TAIL_SLICE_PAGE_COUNT)).await;

        assert_eq!(slice.entries.len(), 3);
        assert_eq!(
            slice.cap_hit, None,
            "max_entries == the number of available entries is not a truncation"
        );
        assert_eq!(slice.lookahead, None);
    }

    /// Case 25b — the lookahead is carried *beside* the slice: it is not an
    /// entry, and it is counted against neither cap. Distinct payload sizes make
    /// the byte-accounting assertion sharp — if the probe's entry leaked into
    /// the total, `payload_bytes` would read 94 rather than 30.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn read_tail_slice_lookahead_entry_is_excluded_from_entries_and_byte_accounting() {
        let tv = TestValkey::setup().await;
        for seq in FIRST..FIRST + 3 {
            plant_commit(&tv, seq, Some(&ompf_of(10))).await;
        }
        plant_commit(&tv, FIRST + 3, Some(&ompf_of(64))).await;

        let slice = read(&tv, caps(3, 1 << 20, TAIL_SLICE_PAGE_COUNT)).await;

        assert_eq!(ids(&slice), vec!["8-0", "9-0", "10-0"]);
        assert_eq!(slice.cap_hit, Some(TailSliceCap::Entries));
        let lookahead = slice.lookahead.as_ref().expect("cap_hit implies lookahead");
        assert_eq!(
            lookahead.stream_id, "11-0",
            "the probe returns the very next entry"
        );
        assert!(
            !ids(&slice).contains(&"11-0"),
            "the lookahead must never appear in `entries`"
        );
        assert_eq!(
            slice.payload_bytes, 30,
            "payload_bytes sums `entries` only, excluding the lookahead"
        );
    }

    /// Case 26 — SR-8.3's residual class: a first entry whose `ompf` field alone
    /// exceeds the byte budget. It is excluded whole, never partially included,
    /// and the probe anchors on the *window start* so the lookahead is that very
    /// entry.
    ///
    /// That last detail is load-bearing downstream and is why it is asserted by
    /// stream id here: `entries` empty + `cap_hit == Some(Bytes)` is the only
    /// signal the classifier has to tell this apart from a genuinely empty
    /// window, and the two arms send an operator to different knobs
    /// (`LIVE_EDIT_CHECKPOINT_TAIL_MAX_BYTES` vs the stream-retention knobs,
    /// `LIVE_EDIT_LAYER_STREAM_MAXLEN` and `LIVE_EDIT_LAYER_STREAM_MAX_BYTES`).
    ///
    /// **The WARN this path logs is deliberately not asserted.** The test-writing
    /// guide's *What NOT to Test* rules exact log messages out as observability
    /// rather than contract, and there is no logging-assertion convention in this
    /// module to follow. Capturing it would also mean pinning a thread-local
    /// `tracing` subscriber across an `await` on a multi-thread runtime — and the
    /// multi-thread flavour is not optional here, since `TestValkey`'s `Drop`
    /// silently skips its key cleanup on a current-thread runtime.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn read_tail_slice_first_entry_alone_over_the_byte_cap_is_excluded() {
        let tv = TestValkey::setup().await;
        plant_commit(&tv, FIRST, Some(&ompf_of(128))).await;
        plant_commit(&tv, FIRST + 1, Some(&ompf_of(4))).await;

        let slice = read(&tv, caps(1024, 16, TAIL_SLICE_PAGE_COUNT)).await;

        assert!(
            slice.entries.is_empty(),
            "the oversize entry is excluded whole, never partially included"
        );
        assert_eq!(slice.payload_bytes, 0);
        assert_eq!(slice.cap_hit, Some(TailSliceCap::Bytes));
        let lookahead = slice.lookahead.as_ref().expect("cap_hit implies lookahead");
        assert_eq!(
            lookahead.stream_id, "8-0",
            "with nothing admitted the probe anchors on the window start, so it \
             returns the excluded entry itself"
        );
        assert_eq!(lookahead.payload_bytes, 128);
    }

    /// Case 26b — the ordinary byte-cap stop, with a non-empty admitted prefix.
    /// The entry that would have crossed the budget is excluded, and the running
    /// total reflects only what was admitted.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn read_tail_slice_byte_cap_excludes_the_entry_that_would_exceed_it() {
        let tv = TestValkey::setup().await;
        for seq in FIRST..FIRST + 3 {
            plant_commit(&tv, seq, Some(&ompf_of(20))).await;
        }

        let slice = read(&tv, caps(1024, 45, TAIL_SLICE_PAGE_COUNT)).await;

        assert_eq!(
            ids(&slice),
            vec!["8-0", "9-0"],
            "40 fits under 45, 60 does not"
        );
        assert_eq!(slice.payload_bytes, 40);
        assert_eq!(slice.cap_hit, Some(TailSliceCap::Bytes));
        assert_eq!(
            slice.lookahead.as_ref().map(|e| e.stream_id.as_str()),
            Some("10-0")
        );
    }

    // ── 27: auto-ID cursor correctness ──────────────────────────────────────

    /// Case 27 — an auto-ID sentinel mid-stream is delivered, not stepped over.
    ///
    /// The cursor advances on the **observed** ID (`({last}`), never on a
    /// synthesised `{seq+1}-0`. A synthesised cursor breaks in both directions
    /// here: it cannot address the sentinel's `{ms}-{n}` ID, and after the
    /// sentinel it would rewind to an explicit ID far below it and re-read the
    /// same entry forever. `page_count = 1` forces the cursor to be recomputed
    /// at every entry, including across the sentinel, so the bug would surface
    /// as a wrong entry list or a non-terminating read rather than being masked
    /// by a single large page.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn read_tail_slice_auto_id_sentinel_mid_stream_is_delivered_not_skipped() {
        let tv = TestValkey::setup().await;
        plant_commit(&tv, FIRST, Some(&ompf_of(8))).await;
        plant_commit(&tv, FIRST + 1, None).await;
        let sentinel_id = plant_sentinel(&tv).await;

        let slice = read(&tv, uncapped(1)).await;

        assert_eq!(
            ids(&slice),
            vec!["8-0", "9-0", sentinel_id.as_str()],
            "the sentinel must be delivered in order, after both commits"
        );
        assert_commit(&slice.entries[0], FIRST);
        assert_commit(&slice.entries[1], FIRST + 1);
        assert_eq!(
            slice.entries[2].payload,
            TailEntryPayload::Notification(LayerNotification::Deleted),
        );
        assert_eq!(
            slice.entries[2].payload_bytes, 0,
            "sentinels carry no `ompf` field"
        );
        assert_eq!(slice.cap_hit, None);
    }

    /// Companion to case 27: an `ompf`-less commit (the §8.4 empty-commit shape)
    /// decodes as an `Update` with `relay_ompf == None`, not as an
    /// `Undecodable` — the distinction the classifier's empty-delta arm rests
    /// on. Also pins that its byte contribution is zero.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn read_tail_slice_ompf_less_commit_decodes_as_an_update_with_no_relay_bytes() {
        let tv = TestValkey::setup().await;
        plant_commit(&tv, FIRST, None).await;

        let slice = read(&tv, uncapped(TAIL_SLICE_PAGE_COUNT)).await;

        match &slice.entries[0].payload {
            TailEntryPayload::Notification(LayerNotification::Update {
                sequence_num,
                relay_ompf,
                server_ts,
                ..
            }) => {
                assert_eq!(*sequence_num, FIRST);
                assert_eq!(*server_ts, SERVER_TS + FIRST as i64);
                assert!(
                    relay_ompf.is_none(),
                    "an absent `ompf` field must decode as None, not as empty bytes"
                );
            }
            other => panic!("expected a decoded update, got {other:?}"),
        }
        assert_eq!(slice.payload_bytes, 0);
    }

    // ── Sequence-ceiling guard (engineer-diff gap coverage) ─────────────────

    /// The read window is refused rather than issued when `base_snap_seq + 1`
    /// would sit above the `2^53 - 1` ceiling both commit paths enforce: no
    /// readable entry can exist there, so an `XRANGE` would be a round trip that
    /// can only ever come back empty — and an empty window folds to `Trimmed`,
    /// which would send an operator to the retention knob for an arithmetic
    /// limit. `u64::MAX` additionally exercises the `checked_add` overflow arm.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn read_tail_slice_base_snap_seq_at_or_above_the_ceiling_returns_internal() {
        let tv = TestValkey::setup().await;

        for base in [SEQUENCE_CEILING, u64::MAX] {
            let err = read_tail_slice(
                tv.client(),
                tv.key_schema(),
                OBJECT_ID,
                base,
                uncapped(TAIL_SLICE_PAGE_COUNT),
            )
            .await
            .expect_err("no readable window exists above the sequence ceiling");
            assert!(
                matches!(err, ServiceError::Internal(_)),
                "base_snap_seq={base} must be a permanent Internal, never the \
                 retriable Aborted signal: retrying cannot make the window exist"
            );
        }
    }

    /// Boundary complement: one below the ceiling still yields a readable window
    /// (start `= SEQUENCE_CEILING`), so the guard is `>`-shaped and does not
    /// refuse the last legal sequence.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn read_tail_slice_base_snap_seq_one_below_the_ceiling_is_readable() {
        let tv = TestValkey::setup().await;

        let slice = read_tail_slice(
            tv.client(),
            tv.key_schema(),
            OBJECT_ID,
            SEQUENCE_CEILING - 1,
            uncapped(TAIL_SLICE_PAGE_COUNT),
        )
        .await
        .expect("the last legal window must still be readable");

        assert!(slice.entries.is_empty(), "nothing was planted that high");
        assert_eq!(slice.cap_hit, None);
    }
}
