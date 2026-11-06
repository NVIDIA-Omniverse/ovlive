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

//! A catch-up payload built as 1 MiB pages, and the reads that fill it
//! (SDD §11.2, §12.3, §12.6).
//!
//! A catch-up payload is an OMPF container: the delta and every external value
//! the delta references. It is held as [`OmpfPayload`], one allocation of at
//! most [`OMPF_PAGE_BYTES`] per page, and page `i` is the stream's chunk `i`
//! (`grpc::streaming::chunk_pages_for_stream`), so the bytes are copied once,
//! into their page, and never again: no whole-payload buffer, no copy per
//! chunk.
//!
//! **Build order.** The builder (`storage::diff_read`) plans the payload,
//! serialises the delta and probes the stored length of each external value
//! ([`ExternalValueStore::probe_value_lengths`]). When it builds with
//! fetch-on-miss and the probe finds values missing, it restores none of them
//! itself: it answers them, each with the size its plan declares, beside the
//! layout it planned, and its caller restores them in batches charged to the
//! catch-up payload budget `D_c` ([`restore_in_batches`], below) before it
//! finishes the layout from their restored lengths. A restore writes the value
//! back to the store before it returns, so only its length is kept and its
//! bytes are dropped. The container is laid out from those lengths
//! ([`PayloadLayout::lay_out`]): the header, offset table and delta are known
//! before any value byte is read. Its caller reserves `D_c` for
//! [`PayloadLayout::reservation_bytes`] — the subscription's catch-up does,
//! inside its slot or build turn (`grpc::subscribe`) — and only then are the
//! values read, every one of them in ranges, a restored one read back from
//! where its restore wrote it ([`PayloadLayout::fill`]).
//!
//! **Restores before the layout.** [`restore_in_batches`] takes the missing
//! values in plan order, in batches of at most [`EXT_RESTORE_PARALLELISM`]
//! values whose charges together fit in `D_c`'s total; a value whose charge
//! alone exceeds it is a batch of its own, which the budget clamps to its
//! total so it runs alone. A value's charge is its declared size times the
//! caller's per-byte charge ([`restore_charge_bytes`]: a catch-up passes the
//! restore charge [`EXT_RESTORE_CHARGE_CENTI`], 4.24 per byte), rounded up
//! to whole KiB. For each batch it re-probes the values' lengths — one
//! another catch-up restored meanwhile is laid out at its stored length,
//! neither fetched nor charged — and reserves the rest's charges through the
//! caller's reservation (for a catch-up, its capacity wait on `D_c`). Once
//! the reservation is granted it probes them once more, waited or not, and
//! gives back the share of every value now stored before it fetches
//! anything, so a batch that waited while another catch-up restored its
//! values neither fetches them again nor keeps their charge. Two catch-ups
//! whose probes both run before either's write-back each fetch the value:
//! on one replica the store's single-flight merges their fetches when they
//! overlap, but it is per process, so across replicas nothing merges them
//! and a fleet-wide wave can fetch a value once per replica. It then splits
//! the grant into one share per value and restores the values at once
//! ([`ExternalValueStore::recover_value`]). Each share is repaid, and its
//! value's bytes dropped, as soon as that value's write-back has returned;
//! the charge of a restored value's bytes beyond its declared size is a
//! forced top-up held from its fetch until its write-back returns. The next
//! batch reserves only once every share of this one is repaid, and the
//! payload's own reservation only once the last is: nothing of `D_c` is held
//! while one of the catch-up's reservations waits. Only a fetch-on-miss
//! layout ([`RestoreLayout`]) can be restored this way ([`restore_in_batches`]
//! says why).
//!
//! **The ranged reads.** Each value is read with
//! [`ExternalValueStore::read_value_range`] in ranges of at most
//! [`EXT_VALUE_READ_CHUNK_BYTES`] (65,472 B) that never cross a page boundary,
//! at most [`EXT_RANGE_WINDOW`] in flight per build. Each reply is copied into
//! its page and dropped, so a build holds its pages and at most that many
//! replies, which its reservation covers. A range is never larger: one
//! connection's reply frames stay inside the read buffer the connection-buffer
//! bound counts (SDD §15.3, §22.17.1).
//!
//! **Fail-closed.** A range shorter than asked, or a value whose length at the
//! closing check ([`ExternalValueStore::probe_value_lengths`] again) differs
//! from the one it was laid out with, reads as missing. The closing check runs
//! on every fill. A probe build checks every value that did not already come
//! back short, answers [`Built::Missing`] with every value that reads as
//! missing, and the caller runs the miss protocol (a rebuild under the `{gc}`
//! read hold). A fetch-on-miss build reads every range first and checks every
//! value, then restores through the checkpoint adapter
//! ([`ExternalValueStore::recover_value`]), one at a time, only the values
//! the check finds not stored. A value still stored at another length is not
//! missing — a
//! verified upload of another length replaced it while it was read — nor is
//! one stored at its laid-out length that came back short, and a value
//! restored before the layout that does not read back whole at its restored
//! length (its write-back refused, or replaced while read) is never restored a
//! second time: each fails the fill `UNAVAILABLE` before anything is restored,
//! and the retry lays it out afresh from what is stored then. So does a
//! payload any byte of which was left unread: it is never frozen. One exit is
//! `INTERNAL` instead: a value the checkpoint adapter cannot restore either,
//! whose restore fails with the store's integrity error, the value genuinely
//! missing from the store and the adapter alike (SDD §11.2). A payload is
//! never sent with a torn value.
//!
//! **Memory.** The reservation's grant is split across the pages when they are
//! frozen: each page carries its own share in its [`Bytes`] owner, and repays
//! it when the page is dropped — when tonic has encoded its chunk, or with the
//! message on any other exit (a discard, a disconnect, an error). A grant the
//! budget clamped to its total covers the payload's first pages, which go out
//! first. The share for the replies in flight is repaid as the frozen pages
//! are built. Bytes restored through the adapter before the layout are held
//! only under their batch's grant and the top-up for any bytes beyond their
//! declared size (above), and dropped before the payload
//! reserves, so nothing of them is held across that wait, and the payload's
//! grant covers them as it covers every other value's. A value restored
//! during the fill lands in pages the grant already covers; its charge
//! ([`restore_charge_bytes`]) is taken as a forced top-up on the grant's
//! budget before it is fetched and held only while its bytes are — inside the
//! payload's grant, where nothing may wait — one value at a time, and bytes
//! fetched beyond its laid-out length are charged the same way as a restore
//! before the layout charges its bytes beyond their declared size.

use std::collections::HashMap;
use std::future::Future;

use bytes::Bytes;
use futures::stream::StreamExt;

use crate::domain::ompf::OmpfContainer;
use crate::error::ServiceError;
use crate::storage::diff_read::{Built, RestoreLayout};
use crate::storage::ext_backup::EXT_VALUE_READ_CHUNK_BYTES;
use crate::storage::external_values::{ExtValueLengths, ExternalValueStore};
use crate::storage::memory_budget::{kib_for_bytes, MemoryGrant, MemoryTopUp};

/// The size of a payload page, and of a stream chunk: 1 MiB (SDD §12.7).
/// Every page is this size but the last.
pub const OMPF_PAGE_BYTES: usize = 1_048_576;

/// The most external-value ranges one build has in flight at once: 16
/// [assumed, compiled]. With ranges of at most
/// [`EXT_VALUE_READ_CHUNK_BYTES`], a build's replies in flight are at most
/// about 1 MiB, which its reservation covers
/// ([`PayloadLayout::reservation_bytes`]).
pub const EXT_RANGE_WINDOW: usize = 16;

/// The bytes a build's ranged reads can hold at once: [`EXT_RANGE_WINDOW`]
/// replies of at most [`EXT_VALUE_READ_CHUNK_BYTES`].
pub const EXT_RANGE_WINDOW_BYTES: u64 =
    (EXT_RANGE_WINDOW as u64) * (EXT_VALUE_READ_CHUNK_BYTES as u64);

/// The most values one catch-up restores through the checkpoint adapter at
/// once before its layout ([`restore_in_batches`]): 4 [assumed, compiled].
/// With a catch-up build in every catch-up slot, at most 4 × the slot count
/// backup-store fetches and their write-backs run per replica.
pub const EXT_RESTORE_PARALLELISM: usize = 4;

/// A catch-up's charge per declared byte of a value it restores, in
/// hundredths: 424, so 4.24 per byte, ⌈1.25 × 3.39 × 100⌉. 3.39 is the
/// resident growth per restored byte measured at 4 MiB values in batches of
/// 4 (record `restore-memory-envelope-tv3-20261022`), the value, the
/// adapter's collect of its response and the store's write-back; the heap
/// fits at 2.0, and the excess is jemalloc keeping freed pages across
/// batches (dirty decay 10 s). Restore-only: uploads keep
/// `grpc::upload_admission::EXT_EXPANSION_CENTI` (3.09), their own fitted
/// expansion, which this never raises.
///
/// Deferred refinement: purge the allocator between restore batches through
/// the hook the accept-loss Job's value removal uses
/// (`storage::allocator_purge::AllocatorPurge`), which would bring the
/// resident growth back toward the heap's 2.0 and let this fall. A catch-up
/// runs in a serving replica, which installs no purge today: doing so needs
/// an install path outside the Job and a measurement of what an all-arena
/// purge costs concurrent requests. Revisit on a restore-throughput
/// complaint or measurement.
pub const EXT_RESTORE_CHARGE_CENTI: u64 = 424;

/// The catch-up payload budget charge of holding a restored value of `len`
/// bytes: `len × charge_centi ÷ 100`, rounded up, saturating. `charge_centi`
/// is the caller's per-byte charge in hundredths; a catch-up passes
/// [`EXT_RESTORE_CHARGE_CENTI`], chosen to cover the value, the adapter's
/// collect of its response and the store's write-back (SDD §12.3).
pub fn restore_charge_bytes(len: u64, charge_centi: u64) -> u64 {
    let scaled = (u128::from(len) * u128::from(charge_centi)).div_ceil(100);
    u64::try_from(scaled).unwrap_or(u64::MAX)
}

/// [`restore_charge_bytes`] in whole KiB, as the budget takes it, in bytes:
/// what one value's share of its batch's grant holds.
fn restore_share_bytes(len: u64, charge_centi: u64) -> u64 {
    kib_for_bytes(restore_charge_bytes(len, charge_centi)).saturating_mul(1024)
}

/// One value a fetch-on-miss layout found not stored, to restore before the
/// layout is finished ([`restore_in_batches`]). Its fields are private and it
/// reaches [`restore_in_batches`] only inside a
/// [`RestoreLayout`](crate::storage::diff_read::RestoreLayout), which only a
/// fetch-on-miss layout builds (that function's docs say why).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RestoreValue {
    /// The value's hash.
    hash: [u8; 32],
    /// The size the plan declares for it (the largest of its references'),
    /// which its charge is reserved for before it is fetched.
    declared: u64,
}

impl RestoreValue {
    /// The value `hash`, whose plan declares `declared` bytes for it.
    pub(crate) fn new(hash: [u8; 32], declared: u64) -> Self {
        RestoreValue { hash, declared }
    }

    /// The value's hash.
    pub fn hash(&self) -> [u8; 32] {
        self.hash
    }

    /// The size the plan declares for the value (the largest of its
    /// references'), which its charge is reserved for before it is fetched.
    pub fn declared(&self) -> u64 {
        self.declared
    }
}

/// What [`restore_in_batches`] found of one value: the length to lay it out
/// at, and whether this catch-up restored it — fetched it through the
/// checkpoint adapter and wrote it back — or its batch's re-probe found it
/// stored again (another catch-up restored it meanwhile).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RestoredLength {
    /// The value's stored length.
    pub len: u64,
    /// Whether this catch-up restored it.
    pub restored: bool,
}

/// How many of `charges` — the shares ([`restore_share_bytes`]) of the values
/// still to restore, in plan order — the next batch takes: at most
/// [`EXT_RESTORE_PARALLELISM`], while their sum stays within `total_bytes`,
/// and always the first, alone when its share alone exceeds `total_bytes`.
/// `0` only for no values.
fn restore_batch_len(charges: &[u64], total_bytes: u64) -> usize {
    let mut taken = 0;
    let mut sum = 0u64;
    for &charge in charges.iter().take(EXT_RESTORE_PARALLELISM) {
        let next = sum.saturating_add(charge);
        if taken > 0 && next > total_bytes {
            break;
        }
        sum = next;
        taken += 1;
    }
    taken
}

/// The forced top-up, on `budget`'s budget, for the charge of a restored
/// value's bytes beyond the size it was charged for before its fetch:
/// [`restore_charge_bytes`] of the `fetched` length less that of
/// `charged_for`, at `charge_centi`; `None` when that is nothing. It is what
/// the hook a restore passes to [`ExternalValueStore::recover_value`]
/// ([`OnFetched`](crate::storage::external_values::OnFetched)) answers, and
/// the store holds it from the fetch until the value's write-back returns.
fn excess_top_up(
    budget: &MemoryGrant,
    fetched: u64,
    charged_for: u64,
    charge_centi: u64,
) -> Option<MemoryTopUp> {
    let excess = restore_charge_bytes(fetched, charge_centi)
        .saturating_sub(restore_charge_bytes(charged_for, charge_centi));
    (excess > 0).then(|| budget.top_up_on_budget(excess))
}

/// Answers every one of `values` that `probe` found stored — at its stored
/// length, not restored by this catch-up — and returns the others, still to
/// restore, each with its share.
fn take_stored(
    probe: &ExtValueLengths,
    values: Vec<(RestoreValue, u64)>,
    lengths: &mut HashMap<[u8; 32], RestoredLength>,
) -> Vec<(RestoreValue, u64)> {
    let mut rest = Vec::with_capacity(values.len());
    for (value, share) in values {
        match probe.lengths.get(&value.hash) {
            Some(&len) => {
                lengths.insert(
                    value.hash,
                    RestoredLength {
                        len,
                        restored: false,
                    },
                );
            }
            None => rest.push((value, share)),
        }
    }
    rest
}

/// Restores the values `layout` found not stored ([`RestoreLayout::restores`],
/// in plan order) through the checkpoint adapter in batches charged to a
/// memory budget (module docs, *Restores before the layout*), and answers each
/// value's length to lay it out at.
///
/// It takes a [`RestoreLayout`], not a list of values, because only a
/// fetch-on-miss layout builds one, and a fetch-on-miss layout is built only
/// from the proof that the store answered a `{gc}` read hold live on the
/// genuine-miss check (`diff_read::ExtValueReads::recover`, SDD §11.2). So the
/// type forbids a restore outside the miss protocol, where a missing value may
/// simply have been collected after a commit released it: its write-back
/// would find its meta absent and pay the ordinary store's `+1`, recreating a
/// reference count that no object holds and none releases — a leaked count.
///
/// A batch is at most [`EXT_RESTORE_PARALLELISM`] values whose shares
/// together fit in `budget_total_bytes`, or one value whose share alone does
/// not. For each batch, in order:
///
/// 1. its values' lengths are probed again; one now stored is answered at its
///    stored length, not restored and not charged, and a batch with none
///    left reserves nothing;
/// 2. `reserve` is called once for the rest's shares together, each value's
///    [`restore_charge_bytes`] at `charge_centi` rounded up to whole KiB. It
///    may wait (a catch-up's reservation of `D_c` does), and its error ends
///    the restore with nothing of the budget held. The lengths are then
///    probed once more, whether or not it waited: another catch-up may have
///    restored a value meanwhile. One now stored is answered as in step 1,
///    and its share is given back before anything is fetched, so the
///    reservation never stays on a value the batch will not fetch. Two
///    catch-ups whose probes both run before either's write-back each fetch
///    the value; the store's single-flight merges overlapping fetches on one
///    replica only, so across replicas each fetches it;
/// 3. the grant is split into one share per value, and the values are
///    restored at once ([`ExternalValueStore::recover_value`]), each share
///    and its value's bytes dropped as soon as that value's restore — its
///    write-back included — has returned. A value fetched longer than it
///    declared holds a forced top-up of the charge of its bytes beyond its
///    declared size on the grant's budget from its fetch until its
///    write-back returns
///    ([`OnFetched`](crate::storage::external_values::OnFetched));
/// 4. the next batch starts only once every value of this one is done, so
///    nothing of this batch's grant is held while the next one's waits.
///
/// The first value that fails, in plan order, fails the whole restore once
/// every value of its batch is done; a value restored before that stays
/// written back, so a retry finds it stored. The budget's total and the
/// per-byte charge are the caller's: storage holds no rule of its own for
/// them.
pub async fn restore_in_batches<E, R, Fut>(
    store: &dyn ExternalValueStore,
    layout: &RestoreLayout,
    charge_centi: u64,
    budget_total_bytes: u64,
    mut reserve: R,
) -> Result<HashMap<[u8; 32], RestoredLength>, E>
where
    E: From<ServiceError>,
    R: FnMut(u64) -> Fut,
    Fut: Future<Output = Result<MemoryGrant, E>>,
{
    let values = layout.restores();
    let mut lengths = HashMap::with_capacity(values.len());
    let mut rest = values;
    while !rest.is_empty() {
        let charges: Vec<u64> = rest
            .iter()
            .take(EXT_RESTORE_PARALLELISM)
            .map(|value| restore_share_bytes(value.declared, charge_centi))
            .collect();
        let (batch, after) = rest.split_at(restore_batch_len(&charges, budget_total_bytes));
        rest = after;
        let batch: Vec<(RestoreValue, u64)> = batch.iter().copied().zip(charges).collect();

        // 1. The re-probe: a value another catch-up restored since is laid
        //    out at its stored length, neither fetched nor charged.
        let hashes: Vec<[u8; 32]> = batch.iter().map(|(value, _)| value.hash).collect();
        let probe = store.probe_value_lengths(&hashes).await?;
        let fetch = take_stored(&probe, batch, &mut lengths);
        if fetch.is_empty() {
            continue;
        }

        // 2. The batch's charge, reserved before anything is fetched, then
        //    the probe again: a value restored while the reservation waited
        //    is answered stored, and its share given back before any fetch.
        let charge = fetch
            .iter()
            .fold(0u64, |sum, &(_, share)| sum.saturating_add(share));
        let mut grant = reserve(charge).await?;
        let hashes: Vec<[u8; 32]> = fetch.iter().map(|(value, _)| value.hash).collect();
        let probe = store.probe_value_lengths(&hashes).await?;
        let fetch = take_stored(&probe, fetch, &mut lengths);
        let kept = fetch
            .iter()
            .fold(0u64, |sum, &(_, share)| sum.saturating_add(share));
        // Whole KiB, so exact; a grant the budget clamped to its total is a
        // batch of one value, kept whole or given back whole.
        grant.shrink_to(kept);
        if fetch.is_empty() {
            continue;
        }

        // 3. One share per value, each dropped with its value's bytes once
        //    its write-back has returned; the bytes beyond its declared size
        //    are charged by the hook from the fetch until then.
        let held: Vec<Option<MemoryGrant>> = fetch
            .iter()
            .map(|&(_, share)| grant.split_off(share))
            .collect();
        let budget = &grant;
        let restores = fetch
            .iter()
            .zip(held)
            .map(|(&(value, _), share)| async move {
                let charge_excess =
                    |fetched: u64| excess_top_up(budget, fetched, value.declared, charge_centi);
                let bytes = store.recover_value(&value.hash, &charge_excess).await?;
                let len = bytes.len() as u64;
                drop(bytes);
                drop(share);
                Ok::<_, ServiceError>((value.hash, len))
            });
        let restored = futures::future::join_all(restores).await;
        // 4. Every share of the batch is back. The split left the grant
        //    itself empty (the shares are whole KiB and sum to it, or take
        //    all of a grant the budget clamped).
        drop(grant);
        for result in restored {
            let (hash, len) = result?;
            lengths.insert(
                hash,
                RestoredLength {
                    len,
                    restored: true,
                },
            );
        }
    }
    Ok(lengths)
}

/// An OMPF container held as pages of at most [`OMPF_PAGE_BYTES`], every one
/// that size but the last; page `i` is the stream's chunk `i`. A page built by
/// [`PayloadLayout::fill`] carries its share of the catch-up payload budget in
/// its owner, repaid when the page is dropped.
#[derive(Default)]
pub struct OmpfPayload {
    pages: Vec<Bytes>,
    len: u64,
}

impl std::fmt::Debug for OmpfPayload {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OmpfPayload")
            .field("len", &self.len)
            .field("pages", &self.pages.len())
            .finish()
    }
}

impl OmpfPayload {
    /// `bytes` as pages, each a slice of it, charging nothing: for a payload
    /// built whole (the steady state's empty delta) and for tests.
    pub fn from_bytes(bytes: Bytes) -> Self {
        let len = bytes.len();
        let pages = (0..len)
            .step_by(OMPF_PAGE_BYTES)
            .map(|start| bytes.slice(start..(start + OMPF_PAGE_BYTES).min(len)))
            .collect();
        Self {
            pages,
            len: len as u64,
        }
    }

    /// The container's length, in bytes.
    pub fn len(&self) -> u64 {
        self.len
    }

    /// Whether the container is empty (no page).
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The pages, in order.
    pub fn pages(&self) -> &[Bytes] {
        &self.pages
    }

    /// The pages, in order, each moved out: a page's share of the budget is
    /// repaid when the caller drops it.
    pub fn into_pages(self) -> std::vec::IntoIter<Bytes> {
        self.pages.into_iter()
    }

    /// The container's bytes, the pages concatenated: a copy, for a caller
    /// that parses it (tests and diagnostics; the stream sends the pages).
    pub fn to_vec(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(usize::try_from(self.len).unwrap_or(0));
        for page in &self.pages {
            out.extend_from_slice(page);
        }
        out
    }
}

/// A page's owner: its bytes and its share of the payload's catch-up payload
/// budget grant, repaid when the last [`Bytes`] of the page drops.
struct PageOwner {
    bytes: Vec<u8>,
    _charge: Option<MemoryGrant>,
}

impl AsRef<[u8]> for PageOwner {
    fn as_ref(&self) -> &[u8] {
        &self.bytes
    }
}

/// What a fill does with a value that reads as missing: a short range, or a
/// length at the closing check other than the one it was laid out with.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum OnMissing {
    /// Answer [`Built::Missing`]: the build reads with the probe, and its
    /// caller runs the miss protocol.
    Answer,
    /// Restore it through the checkpoint adapter
    /// ([`ExternalValueStore::recover_value`]) when it is not stored and was
    /// not restored before the layout, and otherwise — stored, at another
    /// length or short at its own, or restored already — fail the fill
    /// `UNAVAILABLE` before restoring anything: the build reads with
    /// fetch-on-miss.
    Recover,
}

/// The catch-up payload budget a fill charges ([`PayloadLayout::fill`]): the
/// grant of the payload's reservation, and the per-byte charge, in
/// hundredths, of a value the fill restores ([`restore_charge_bytes`]), taken
/// as a forced top-up on the grant's budget while that value's bytes are
/// held.
#[derive(Debug)]
pub struct PayloadGrant {
    /// The payload's reservation.
    pub grant: MemoryGrant,
    /// The charge, per 100 bytes, of a value restored during the fill.
    pub restore_charge_centi: u64,
}

/// One external value of a laid-out payload.
struct LaidOutValue {
    hash: [u8; 32],
    /// Where its bytes start in the container.
    offset: u64,
    /// Its length, as probed, or as restored.
    len: u64,
    /// Restored through the checkpoint adapter before the layout by this
    /// catch-up ([`restore_in_batches`]), which wrote it back to the store:
    /// read back in ranges like every other value, and never restored a
    /// second time by the fill.
    restored: bool,
}

/// A catch-up payload laid out from its parts' lengths, before any external
/// value byte is read (module docs).
pub struct PayloadLayout {
    /// The header, offset table and delta: the container's first bytes.
    prefix: Vec<u8>,
    /// The external values, in part order.
    values: Vec<LaidOutValue>,
    /// The container's length.
    len: u64,
    on_missing: OnMissing,
}

impl std::fmt::Debug for PayloadLayout {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PayloadLayout")
            .field("len", &self.len)
            .field("values", &self.values.len())
            .field("on_missing", &self.on_missing)
            .finish()
    }
}

impl PayloadLayout {
    /// Lays out the container of `delta` and `values` (each external value's
    /// hash, its length, and whether the checkpoint adapter restored it before
    /// the layout, writing it back to the store at that length), in part
    /// order. Byte-identical to [`OmpfContainer::serialize`] of the same delta
    /// and values, through [`OmpfContainer::header_bytes`].
    pub(crate) fn lay_out(
        delta: Vec<u8>,
        values: Vec<([u8; 32], u64, bool)>,
        on_missing: OnMissing,
    ) -> Self {
        if delta.is_empty() && values.is_empty() {
            return Self {
                prefix: Vec::new(),
                values: Vec::new(),
                len: 0,
                on_missing,
            };
        }
        let lens: Vec<u64> = values.iter().map(|(_, len, _)| *len).collect();
        // `header_bytes` only fails on more than u32::MAX parts — one part per
        // distinct external value of a layer, which no layer's memory reaches
        // (as `diff_read::build_diff_payload`'s `serialize`).
        let mut prefix = OmpfContainer::header_bytes(delta.len() as u64, &lens, delta.len())
            .expect(
            "OMPF part count fits in u32 — a layer's distinct external values stay far below it",
        );
        prefix.extend_from_slice(&delta);
        drop(delta);
        let mut offset = prefix.len() as u64;
        let values: Vec<LaidOutValue> = values
            .into_iter()
            .map(|(hash, len, restored)| {
                let value = LaidOutValue {
                    hash,
                    offset,
                    len,
                    restored,
                };
                offset = offset.saturating_add(len);
                value
            })
            .collect();
        Self {
            prefix,
            values,
            len: offset,
            on_missing,
        }
    }

    /// The container's length, in bytes.
    pub fn len(&self) -> u64 {
        self.len
    }

    /// Whether the container is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Whether any external value has a byte to read in ranges.
    fn reads_ranges(&self) -> bool {
        self.values.iter().any(|v| v.len > 0)
    }

    /// The catch-up payload budget a build of this layout reserves before its
    /// first range is read: the container's exact length — values restored
    /// before the layout included, read back like the rest — plus
    /// [`EXT_RANGE_WINDOW_BYTES`] for its replies in flight when it reads
    /// external values in ranges. The budget clamps it to its total.
    pub fn reservation_bytes(&self) -> u64 {
        let window = if self.reads_ranges() {
            EXT_RANGE_WINDOW_BYTES
        } else {
            0
        };
        self.len.saturating_add(window)
    }

    /// Fills the pages: the header and delta, then every value in ranges
    /// ([`EXT_RANGE_WINDOW`] in flight), then the closing length check; and
    /// freezes them, each with its share of `charge`'s grant (module docs).
    ///
    /// Every value that reads as missing — a short range, or a length at the
    /// closing check other than the layout's — is collected first. A probe
    /// build answers [`Built::Missing`] with all of them (it stops reading
    /// ranges at the first short one, and checks every value that did not come
    /// back short). A fetch-on-miss build reads every range and checks every
    /// value, then fails `UNAVAILABLE`, before restoring anything, when a
    /// value is still stored at another length, when a value restored before
    /// the layout did not read back whole at its restored length, or when a
    /// value came back short though it is stored at its length; otherwise it
    /// restores every value the check found not stored, one at a time
    /// ([`ExternalValueStore::recover_value`]), each under a forced top-up of
    /// its charge on `charge`'s budget held only while its bytes are. A
    /// restored value whose length is not the laid-out one fails the build
    /// `UNAVAILABLE` too; so does a payload any byte of which was left unread
    /// — it is never frozen. Each is before anything is sent, so the retry
    /// lays the payload out afresh. One exit is `INTERNAL` instead: a value
    /// the checkpoint adapter cannot restore either fails the build with the
    /// restore's error, the value genuinely missing from the store and the
    /// adapter alike. `charge` is `None` only outside a subscription's
    /// catch-up (tests and diagnostics), which charge no budget.
    pub async fn fill(
        mut self,
        store: &dyn ExternalValueStore,
        charge: Option<PayloadGrant>,
    ) -> Result<Built<OmpfPayload>, ServiceError> {
        let (grant, restore_charge_centi) = match charge {
            Some(PayloadGrant {
                grant,
                restore_charge_centi,
            }) => (Some(grant), restore_charge_centi),
            None => (None, 0),
        };
        let mut pages = Pages::new(self.len);
        pages.write(0, &self.prefix);
        self.prefix = Vec::new();
        // Bytes of each value in its pages so far.
        let mut filled = vec![0u64; self.values.len()];

        let stop_at_first_short = self.on_missing == OnMissing::Answer;
        let short = read_ranges(
            store,
            &self.values,
            &mut pages,
            &mut filled,
            stop_at_first_short,
        )
        .await?;
        let check_short = self.on_missing == OnMissing::Recover;
        let closing = closing_check(store, &self.values, &short, check_short).await?;
        match self.on_missing {
            OnMissing::Answer => {
                // Every value that reads as missing, in part order.
                let missing: Vec<[u8; 32]> = (0..self.values.len())
                    .filter(|&i| short[i] || !closing[i].is_at(self.values[i].len))
                    .map(|i| self.values[i].hash)
                    .collect();
                if !missing.is_empty() {
                    return Ok(Built::Missing(missing));
                }
            }
            OnMissing::Recover => {
                for (index, value) in self.values.iter().enumerate() {
                    if let Closing::Stored(stored) = closing[index] {
                        if stored != value.len {
                            return Err(ServiceError::Unavailable(format!(
                                "an external value this catch-up read is stored at another \
                                 length than it was laid out with ({} bytes laid out, {stored} \
                                 stored); retry",
                                value.len
                            )));
                        }
                    }
                    let whole = !short[index] && closing[index].is_at(value.len);
                    if value.restored && !whole {
                        return Err(ServiceError::Unavailable(format!(
                            "an external value this catch-up restored through the checkpoint \
                             adapter did not read back whole at its restored length ({} \
                             bytes); retry",
                            value.len
                        )));
                    }
                    // Stored at its laid-out length (the check found every
                    // value stored or absent), yet a range of it came back
                    // short: it is not missing, so nothing restores it.
                    if closing[index] != Closing::Absent && !whole {
                        return Err(ServiceError::Unavailable(format!(
                            "an external value this catch-up read came back short though it is \
                             stored at its laid-out length ({} bytes); retry",
                            value.len
                        )));
                    }
                }
                // Every value here was read whole, or is not stored and was
                // not restored before the layout: only those not stored are
                // restored, so a build the loop fails restores nothing.
                let absent: Vec<usize> = (0..self.values.len())
                    .filter(|&i| closing[i] == Closing::Absent)
                    .collect();
                for index in absent {
                    let value = &self.values[index];
                    // The restored bytes arrive whole, beside pages already
                    // reserved for them, inside the payload's grant, where
                    // nothing may wait: a forced top-up of this one value's
                    // charge, taken before its fetch and repaid once its
                    // bytes are in their pages. Bytes fetched beyond its
                    // laid-out length (a build the check below fails) are
                    // charged by the hook until their write-back returns.
                    let held = grant.as_ref().map(|grant| {
                        grant
                            .top_up_on_budget(restore_charge_bytes(value.len, restore_charge_centi))
                    });
                    let charge_excess = |fetched: u64| {
                        grant.as_ref().and_then(|grant| {
                            excess_top_up(grant, fetched, value.len, restore_charge_centi)
                        })
                    };
                    let bytes = store.recover_value(&value.hash, &charge_excess).await?;
                    if bytes.len() as u64 != value.len {
                        return Err(ServiceError::Unavailable(format!(
                            "an external value this catch-up read changed length while it was \
                             read ({} bytes laid out, {} restored); retry",
                            value.len,
                            bytes.len()
                        )));
                    }
                    pages.write(value.offset, &bytes);
                    filled[index] = value.len;
                    drop(bytes);
                    drop(held);
                }
            }
        }
        // The closing invariant: every byte of every value is in its pages.
        // What reached here was read whole and passed the closing check, or
        // was restored at its laid-out length; a payload with a byte left
        // unread is never frozen, whatever the path.
        if let Some(index) = (0..self.values.len()).find(|&i| filled[i] != self.values[i].len) {
            return Err(ServiceError::Unavailable(format!(
                "an external value of this catch-up was not read whole ({} of {} bytes); retry",
                filled[index], self.values[index].len
            )));
        }
        Ok(Built::Ready(pages.freeze(grant)))
    }
}

/// The pages of a payload being filled, allocated at their exact sizes.
struct Pages {
    pages: Vec<Vec<u8>>,
    len: u64,
}

impl Pages {
    fn new(len: u64) -> Self {
        let len_usize = usize::try_from(len).unwrap_or(usize::MAX);
        let pages = (0..len_usize)
            .step_by(OMPF_PAGE_BYTES)
            .map(|start| vec![0u8; (len_usize - start).min(OMPF_PAGE_BYTES)])
            .collect();
        Self { pages, len }
    }

    /// Copies `bytes` into the container at `offset`, across pages as needed.
    fn write(&mut self, offset: u64, bytes: &[u8]) {
        let mut at = usize::try_from(offset).unwrap_or(usize::MAX);
        let mut rest = bytes;
        while !rest.is_empty() {
            let page = &mut self.pages[at / OMPF_PAGE_BYTES];
            let within = at % OMPF_PAGE_BYTES;
            let n = rest.len().min(page.len() - within);
            page[within..within + n].copy_from_slice(&rest[..n]);
            rest = &rest[n..];
            at += n;
        }
    }

    /// Freezes the pages from the first page on, each with its share of
    /// `grant` for its bytes, so they stay charged until their page is
    /// dropped. A grant the budget clamped covers the payload's first pages.
    /// What is left of the grant after the last page (the replies' share) is
    /// repaid here.
    fn freeze(self, mut grant: Option<MemoryGrant>) -> OmpfPayload {
        let pages = self
            .pages
            .into_iter()
            .map(|bytes| {
                let charge = grant
                    .as_mut()
                    .and_then(|grant| grant.split_off(bytes.len() as u64));
                Bytes::from_owner(PageOwner {
                    bytes,
                    _charge: charge,
                })
            })
            .collect();
        drop(grant);
        OmpfPayload {
            pages,
            len: self.len,
        }
    }
}

/// One ranged read: `len` bytes of value `value` from byte `start` of it, at
/// `at` in the container. Never crosses a page boundary.
#[derive(Debug, Clone, Copy)]
struct Range {
    value: usize,
    start: u64,
    at: u64,
    len: usize,
}

/// The ranges of every value, in container order: at most
/// [`EXT_VALUE_READ_CHUNK_BYTES`] each, cut at page boundaries.
fn ranges(values: &[LaidOutValue]) -> Vec<Range> {
    let page = OMPF_PAGE_BYTES as u64;
    let chunk = EXT_VALUE_READ_CHUNK_BYTES as u64;
    let mut out = Vec::new();
    for (index, value) in values.iter().enumerate() {
        let mut start = 0u64;
        while start < value.len {
            let at = value.offset + start;
            let to_page_end = page - at % page;
            let len = (value.len - start).min(chunk).min(to_page_end);
            out.push(Range {
                value: index,
                start,
                at,
                len: len as usize,
            });
            start += len;
        }
    }
    out
}

/// Reads every range, [`EXT_RANGE_WINDOW`] in flight, copying each reply into
/// its page as it lands, adding its bytes to its value's `filled`, and
/// dropping it. Returns, per value, whether a range of it came back short (no
/// byte of a short reply is written). With `stop_at_first_short` (a probe
/// build, which answers [`Built::Missing`] either way) the first short range
/// ends the reads; otherwise every range is read, so a fetch-on-miss build
/// knows every value that reads as missing.
async fn read_ranges(
    store: &dyn ExternalValueStore,
    values: &[LaidOutValue],
    pages: &mut Pages,
    filled: &mut [u64],
    stop_at_first_short: bool,
) -> Result<Vec<bool>, ServiceError> {
    let mut short = vec![false; values.len()];
    let mut replies = futures::stream::iter(ranges(values).into_iter().map(|range| {
        let hash = values[range.value].hash;
        async move {
            store
                .read_value_range(&hash, range.start, range.len)
                .await
                .map(|reply| (range, reply))
        }
    }))
    .buffer_unordered(EXT_RANGE_WINDOW);
    while let Some(read) = replies.next().await {
        let (range, reply) = read?;
        if reply.len() < range.len {
            short[range.value] = true;
            if stop_at_first_short {
                break;
            }
            continue;
        }
        // A range never answers more than it asked for; the slice keeps the
        // page exact even if it did.
        pages.write(range.at, &reply[..range.len]);
        filled[range.value] += range.len as u64;
    }
    Ok(short)
}

/// What the closing check found of one value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Closing {
    /// Not checked: a probe build's value that came back short, missing
    /// either way.
    Unchecked,
    /// Stored, at this length.
    Stored(u64),
    /// Not stored.
    Absent,
}

impl Closing {
    /// Whether the check found the value stored at `len`.
    fn is_at(self, len: u64) -> bool {
        self == Closing::Stored(len)
    }
}

/// The closing check: the stored length of every value
/// ([`ExternalValueStore::probe_value_lengths`]), or whether it is gone; with
/// `check_short` false, a value already `short` is left unchecked. Runs on
/// every fill, whatever the ranges answered.
async fn closing_check(
    store: &dyn ExternalValueStore,
    values: &[LaidOutValue],
    short: &[bool],
    check_short: bool,
) -> Result<Vec<Closing>, ServiceError> {
    let mut closing = vec![Closing::Unchecked; values.len()];
    let checked: Vec<usize> = (0..values.len())
        .filter(|&i| check_short || !short[i])
        .collect();
    if checked.is_empty() {
        return Ok(closing);
    }
    let hashes: Vec<[u8; 32]> = checked.iter().map(|&i| values[i].hash).collect();
    let probe = store.probe_value_lengths(&hashes).await?;
    for index in checked {
        closing[index] = match probe.lengths.get(&values[index].hash) {
            Some(&stored) => Closing::Stored(stored),
            None => Closing::Absent,
        };
    }
    Ok(closing)
}

/// The layout and the fill against a scripted store (no Valkey): the pages'
/// bytes against `OmpfContainer::serialize`, every range the fill asks for,
/// the ranges in flight, the miss rules under each `OnMissing` — only a value
/// the closing check finds not stored is restored — the read-back of a value
/// restored before the layout, the closing invariant, and the catch-up
/// payload budget's grant split across the pages, with the forced top-up a
/// restore during the fill holds only while it holds the restored bytes; and
/// how restores before the layout are batched and charged
/// ([`restore_batch_len`], [`restore_charge_bytes`]). The
/// store's own commands (`STRLEN`, `GETRANGE`, no `GET`) and a catch-up's
/// reservation, refusal and release are driven end to end in
/// `tests/integration/grpc/catchup_payload_test.rs`, and a catch-up whose
/// values were restored before the layout in
/// `tests/integration/grpc/b_yuo_catch_up_miss_test.rs`.
#[cfg(test)]
mod tests {
    use super::*;

    use std::collections::HashMap;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex as StdMutex};

    use crate::storage::external_values::{
        ExtRelease, ExtValueLengths, ExtValueProbe, ExtValueVerify, GcBatchResult, GcPassState,
        GcPolicy, ReleaseAttemptOutcome,
    };
    use crate::storage::memory_budget::{MemoryBudget, CATCHUP_PAYLOAD_SERIES};

    /// A scripted external-value store: the values it holds, read in ranges;
    /// what its length probe answers, when that differs from the values'
    /// lengths (a fill's only probe is its closing check, the layouts here
    /// being made from lengths directly); a value whose ranges come back one
    /// byte short, and a value whose one range starting at a given byte does;
    /// and what its restore through the checkpoint adapter answers. It
    /// records every range and the most in flight at once, every hash
    /// restored and the restore calls, any whole-value read, and, given
    /// `budget_at_restore`, that budget's over-commit as each restore starts
    /// and where the store's write-back would start, the restore's hook
    /// ([`OnFetched`](crate::storage::external_values::OnFetched)) charged.
    #[derive(Default)]
    struct ScriptedStore {
        values: HashMap<[u8; 32], Vec<u8>>,
        closing_lengths: Option<HashMap<[u8; 32], u64>>,
        short: Option<[u8; 32]>,
        short_at: Option<([u8; 32], u64)>,
        restore: HashMap<[u8; 32], Vec<u8>>,
        probes: AtomicUsize,
        ranges: StdMutex<Vec<([u8; 32], u64, usize)>>,
        in_flight: AtomicUsize,
        max_in_flight: AtomicUsize,
        recovered: StdMutex<Vec<[u8; 32]>>,
        restores: AtomicUsize,
        whole_reads: AtomicUsize,
        budget_at_restore: Option<Arc<MemoryBudget>>,
        overcommit_at_restore: StdMutex<Vec<u64>>,
        overcommit_at_write_back: StdMutex<Vec<u64>>,
    }

    impl ScriptedStore {
        fn holding(values: &[([u8; 32], Vec<u8>)]) -> Self {
            Self {
                values: values.iter().cloned().collect(),
                ..Self::default()
            }
        }

        fn ranges(&self) -> Vec<([u8; 32], u64, usize)> {
            self.ranges.lock().expect("ranges").clone()
        }
    }

    #[async_trait::async_trait]
    impl ExternalValueStore for ScriptedStore {
        async fn store_values(&self, _: &[(&[u8; 32], &[u8])]) -> Result<(), ServiceError> {
            unreachable!("a fill stores nothing")
        }

        async fn store_verified_values(
            &self,
            _: &[(&[u8; 32], &[u8])],
        ) -> Result<(), ServiceError> {
            unreachable!("a fill stores nothing")
        }

        fn ext_value_verify(&self) -> ExtValueVerify<'_> {
            ExtValueVerify::ENFORCE_UNCOUNTED
        }

        async fn load_values(
            &self,
            _: &[&[u8; 32]],
        ) -> Result<Vec<([u8; 32], Vec<u8>)>, ServiceError> {
            self.whole_reads.fetch_add(1, Ordering::SeqCst);
            Ok(Vec::new())
        }

        async fn read_batch(
            &self,
            _: &[[u8; 32]],
        ) -> Result<HashMap<[u8; 32], Vec<u8>>, ServiceError> {
            self.whole_reads.fetch_add(1, Ordering::SeqCst);
            Ok(HashMap::new())
        }

        async fn probe_values(&self, _: &[[u8; 32]]) -> Result<ExtValueProbe, ServiceError> {
            self.whole_reads.fetch_add(1, Ordering::SeqCst);
            Ok(ExtValueProbe::default())
        }

        async fn probe_value_lengths(
            &self,
            hashes: &[[u8; 32]],
        ) -> Result<ExtValueLengths, ServiceError> {
            self.probes.fetch_add(1, Ordering::SeqCst);
            let stored: HashMap<[u8; 32], u64> = self
                .values
                .iter()
                .map(|(h, v)| (*h, v.len() as u64))
                .collect();
            let lengths = self.closing_lengths.as_ref().unwrap_or(&stored);
            let mut out = ExtValueLengths::default();
            for hash in hashes {
                match lengths.get(hash) {
                    Some(len) => {
                        out.lengths.insert(*hash, *len);
                    }
                    None => out.missing.push(*hash),
                }
            }
            Ok(out)
        }

        async fn read_value_range(
            &self,
            hash: &[u8; 32],
            start: u64,
            len: usize,
        ) -> Result<Bytes, ServiceError> {
            let now = self.in_flight.fetch_add(1, Ordering::SeqCst) + 1;
            self.max_in_flight.fetch_max(now, Ordering::SeqCst);
            // Let the fill start every other read it will before this one
            // answers, so the window is what bounds them.
            for _ in 0..4 {
                tokio::task::yield_now().await;
            }
            self.ranges
                .lock()
                .expect("ranges")
                .push((*hash, start, len));
            self.in_flight.fetch_sub(1, Ordering::SeqCst);
            let data = self.values.get(hash).map(Vec::as_slice).unwrap_or_default();
            let start = (start as usize).min(data.len());
            let mut end = (start + len).min(data.len());
            let short_here =
                self.short == Some(*hash) || self.short_at == Some((*hash, start as u64));
            if short_here && end > start {
                end -= 1;
            }
            Ok(Bytes::copy_from_slice(&data[start..end]))
        }

        async fn recover_value(
            &self,
            hash: &[u8; 32],
            on_fetched: crate::storage::external_values::OnFetched<'_>,
        ) -> Result<Bytes, ServiceError> {
            self.restores.fetch_add(1, Ordering::SeqCst);
            self.recovered.lock().expect("recovered").push(*hash);
            if let Some(budget) = &self.budget_at_restore {
                self.overcommit_at_restore
                    .lock()
                    .expect("overcommit_at_restore")
                    .push(budget.overcommit_bytes());
            }
            // A value the adapter does not hold fails as the store's restore
            // does: the miss's integrity error, never an empty value. One it
            // holds has no write-back here: the hook's charge is held, as the
            // store holds it across its write-back, only while the budget's
            // over-commit is recorded in the write-back's place, then given
            // back.
            let fetched = self
                .restore
                .get(hash)
                .map(|v| Bytes::copy_from_slice(v))
                .ok_or_else(|| {
                    ServiceError::Internal("the backup store does not hold the value".to_owned())
                })?;
            let fetched_charge = on_fetched(fetched.len() as u64);
            if let Some(budget) = &self.budget_at_restore {
                self.overcommit_at_write_back
                    .lock()
                    .expect("overcommit_at_write_back")
                    .push(budget.overcommit_bytes());
            }
            drop(fetched_charge);
            Ok(fetched)
        }

        async fn enqueue_retirement_attempt(
            &self,
            _: &str,
            _: &str,
            _: &[ExtRelease],
            _: std::time::Instant,
        ) -> Result<ReleaseAttemptOutcome, ServiceError> {
            unreachable!("a fill retires nothing")
        }

        async fn process_pending(
            &self,
            _: &GcPolicy,
            _: &mut GcPassState,
        ) -> Result<GcBatchResult, ServiceError> {
            unreachable!("a fill runs no GC")
        }

        fn ext_release_gate(
            &self,
        ) -> Option<&Arc<crate::storage::ext_release_gate::ExtReleaseGate>> {
            None
        }
    }

    /// A deterministic value of `len` bytes, distinct per `seed`.
    fn value(len: usize, seed: u8) -> Vec<u8> {
        (0..len)
            .map(|i| (i as u8).wrapping_mul(31).wrapping_add(seed))
            .collect()
    }

    fn hash(n: u8) -> [u8; 32] {
        [n; 32]
    }

    /// External values by hash, in part order.
    type Values = Vec<([u8; 32], Vec<u8>)>;

    /// The delta and values every layout here is built from: a value of a
    /// few ranges, one of more than a page (so it crosses page boundaries),
    /// a tiny one and an empty one, behind a delta.
    fn fixture() -> (Vec<u8>, Values) {
        let delta = value(3_000, 200);
        let values = vec![
            (hash(1), value(70_000, 1)),
            (hash(2), value(OMPF_PAGE_BYTES + OMPF_PAGE_BYTES / 2, 2)),
            (hash(3), value(10, 3)),
            (hash(4), Vec::new()),
        ];
        (delta, values)
    }

    /// The container [`OmpfContainer::serialize`] writes for `delta` and
    /// `values`.
    fn serialized(delta: &[u8], values: &[([u8; 32], Vec<u8>)]) -> Vec<u8> {
        let parts: Vec<&[u8]> = values.iter().map(|(_, v)| v.as_slice()).collect();
        OmpfContainer::serialize(delta, &parts).expect("serializes")
    }

    /// `delta` and `values` laid out from their stored lengths, none
    /// restored.
    fn lay_out(
        delta: &[u8],
        values: &[([u8; 32], Vec<u8>)],
        on_missing: OnMissing,
    ) -> PayloadLayout {
        PayloadLayout::lay_out(
            delta.to_vec(),
            values
                .iter()
                .map(|(h, v)| (*h, v.len() as u64, false))
                .collect(),
            on_missing,
        )
    }

    /// That `payload`'s pages concatenate to exactly `want`, naming the first
    /// byte that differs rather than printing megabytes.
    fn assert_container(payload: &OmpfPayload, want: &[u8], what: &str) {
        let got = payload.to_vec();
        if got != want {
            let at = got
                .iter()
                .zip(want)
                .position(|(a, b)| a != b)
                .unwrap_or(got.len().min(want.len()));
            panic!(
                "{what}: the payload ({} B) is not the serialized container ({} B); first \
                 difference at byte {at} (page {})",
                got.len(),
                want.len(),
                at / OMPF_PAGE_BYTES
            );
        }
    }

    fn ready(built: Built<OmpfPayload>) -> OmpfPayload {
        match built {
            Built::Ready(payload) => payload,
            Built::Missing(missing) => panic!("the fill answered missing: {missing:?}"),
        }
    }

    /// A catch-up payload budget of `total` bytes, as the replica builds it.
    fn budget(total: u64) -> Arc<MemoryBudget> {
        Arc::new(MemoryBudget::with_series(
            total / 1024,
            Arc::new(StdMutex::new(None)),
            &CATCHUP_PAYLOAD_SERIES,
        ))
    }

    /// The charge a catch-up passes for a value restored during its fill:
    /// the restore-only [`EXT_RESTORE_CHARGE_CENTI`].
    const CHARGE_CENTI: u64 = EXT_RESTORE_CHARGE_CENTI;

    /// `grant` as a catch-up's fill charges it.
    fn charged(grant: MemoryGrant) -> PayloadGrant {
        PayloadGrant {
            grant,
            restore_charge_centi: CHARGE_CENTI,
        }
    }

    /// A grant of `layout`'s whole reservation from `budget`, as a catch-up
    /// with nothing restored before its layout takes it.
    async fn reserve_for(budget: &Arc<MemoryBudget>, layout: &PayloadLayout) -> MemoryGrant {
        budget
            .reserve(layout.reservation_bytes(), None)
            .await
            .expect("fits")
    }

    /// That nothing of `budget` is held: no grant, no top-up, no waiter.
    fn assert_budget_returned(budget: &MemoryBudget, what: &str) {
        assert_eq!(
            (
                budget.reserved_bytes(),
                budget.overcommit_bytes(),
                budget.waiters()
            ),
            (0, 0, 0),
            "{what}: the catch-up payload budget is all returned"
        );
    }

    /// `delta` and `values` laid out from their lengths, the values of
    /// `restored` marked restored through the checkpoint adapter before the
    /// layout — as `storage::diff_read` lays out a fetch-on-miss build whose
    /// length probe found them missing, their bytes written back to the store
    /// and dropped.
    fn lay_out_restored(
        delta: &[u8],
        values: &[([u8; 32], Vec<u8>)],
        restored: &[[u8; 32]],
    ) -> PayloadLayout {
        PayloadLayout::lay_out(
            delta.to_vec(),
            values
                .iter()
                .map(|(h, v)| (*h, v.len() as u64, restored.contains(h)))
                .collect(),
            OnMissing::Recover,
        )
    }

    /// That `err` is `UNAVAILABLE` (retryable) and its message holds `phrase`.
    fn assert_unavailable(err: ServiceError, phrase: &str, what: &str) {
        match err {
            ServiceError::Unavailable(message) => assert!(
                message.contains(phrase),
                "{what}: expected UNAVAILABLE naming {phrase:?}, got {message:?}"
            ),
            other => panic!("{what}: retryable UNAVAILABLE, never anything else: {other:?}"),
        }
    }

    /// The `UNAVAILABLE` message phrases of the fill's fetch-on-miss exits.
    const STORED_AT_ANOTHER_LENGTH: &str = "is stored at another length than it was laid out with";
    const NOT_READ_BACK_WHOLE: &str = "did not read back whole at its restored length";
    const SHORT_THOUGH_STORED: &str = "came back short though it is stored at its laid-out length";

    /// T2, the payload's half: the pages concatenate to exactly the container
    /// `OmpfContainer::serialize` writes for the same delta and values, every
    /// page is 1 MiB but the last, and the values are read only in ranges —
    /// each at most `EXT_VALUE_READ_CHUNK_BYTES` (65,472 B), none crossing a
    /// page boundary, together covering every value once — at most
    /// `EXT_RANGE_WINDOW` (16) in flight, and never as a whole-value read.
    #[tokio::test]
    async fn pages_are_the_serialized_container_read_in_bounded_ranges_within_pages() {
        let (delta, values) = fixture();
        let store = ScriptedStore::holding(&values);
        let layout = lay_out(&delta, &values, OnMissing::Answer);
        let expected = serialized(&delta, &values);
        assert_eq!(layout.len(), expected.len() as u64);

        let payload = ready(layout.fill(&store, None).await.expect("fills"));
        assert_container(&payload, &expected, "byte-identical to serialize");
        let pages = payload.pages();
        assert_eq!(pages.len(), expected.len().div_ceil(OMPF_PAGE_BYTES));
        for (i, page) in pages.iter().enumerate() {
            if i + 1 < pages.len() {
                assert_eq!(page.len(), OMPF_PAGE_BYTES, "page {i}");
            } else {
                assert!(!page.is_empty() && page.len() <= OMPF_PAGE_BYTES);
            }
        }

        // Where each value starts in the container: after the header and the
        // delta, in part order.
        let lens: Vec<u64> = values.iter().map(|(_, v)| v.len() as u64).collect();
        let mut offset = OmpfContainer::header_bytes(delta.len() as u64, &lens, 0)
            .expect("header")
            .len()
            + delta.len();
        let mut starts = HashMap::new();
        for (h, v) in &values {
            starts.insert(*h, offset);
            offset += v.len();
        }
        let mut ranges = store.ranges();
        assert!(!ranges.is_empty());
        for &(h, start, len) in &ranges {
            assert!(
                len > 0 && len <= EXT_VALUE_READ_CHUNK_BYTES,
                "a range of {len} B: at most {EXT_VALUE_READ_CHUNK_BYTES}"
            );
            let at = starts[&h] + start as usize;
            assert_eq!(
                at / OMPF_PAGE_BYTES,
                (at + len - 1) / OMPF_PAGE_BYTES,
                "the range at {at} of {len} B crosses a page boundary"
            );
        }
        ranges.sort();
        for (h, v) in &values {
            let mut covered = 0u64;
            for &(_, start, len) in ranges.iter().filter(|(rh, _, _)| rh == h) {
                assert_eq!(start, covered, "ranges cover each value once, in order");
                covered += len as u64;
            }
            assert_eq!(covered, v.len() as u64, "every byte of the value is read");
        }
        assert_eq!(
            store.max_in_flight.load(Ordering::SeqCst),
            EXT_RANGE_WINDOW,
            "the window fills and is never passed"
        );
        assert_eq!(
            store.whole_reads.load(Ordering::SeqCst),
            0,
            "no whole-value read"
        );
        assert_eq!(
            store.probes.load(Ordering::SeqCst),
            1,
            "one closing length check"
        );
    }

    /// T4, the payload's half: the grant is split across the pages as they
    /// freeze, from the first page on, and the share for the replies in
    /// flight is repaid when the fill ends, so the budget holds exactly the
    /// pages' bytes (each rounded up to a KiB); each page repays its own share
    /// when dropped, and the budget is back to zero with the last.
    #[tokio::test]
    async fn each_page_carries_its_share_of_the_grant_and_repays_it_when_dropped() {
        let (delta, values) = fixture();
        let store = ScriptedStore::holding(&values);
        let layout = lay_out(&delta, &values, OnMissing::Answer);
        assert_eq!(
            layout.reservation_bytes(),
            layout.len() + EXT_RANGE_WINDOW_BYTES,
            "the payload's exact size plus the replies in flight"
        );
        let budget = budget(64 << 20);
        let grant = budget
            .reserve(layout.reservation_bytes(), None)
            .await
            .expect("fits");
        let payload = ready(
            layout
                .fill(&store, Some(charged(grant)))
                .await
                .expect("fills"),
        );
        let charges: Vec<u64> = payload
            .pages()
            .iter()
            .map(|p| (p.len() as u64).div_ceil(1024) * 1024)
            .collect();
        assert_eq!(
            budget.reserved_bytes(),
            charges.iter().sum::<u64>(),
            "the pages hold their bytes; the replies' share is back"
        );
        let mut pages = payload.into_pages();
        let mut held = budget.reserved_bytes();
        for charge in charges {
            drop(pages.next().expect("a page"));
            held -= charge;
            assert_eq!(budget.reserved_bytes(), held, "a page repays its share");
        }
        assert_eq!(budget.reserved_bytes(), 0);
    }

    /// T4, a clamped payload: one larger than the whole budget is granted the
    /// budget, clamped, so it runs alone, and its charge sits on its first
    /// pages: they hold the whole budget, and the later pages none.
    #[tokio::test]
    async fn a_payload_larger_than_the_budget_is_charged_on_its_first_pages() {
        let (delta, values) = fixture();
        let store = ScriptedStore::holding(&values);
        let layout = lay_out(&delta, &values, OnMissing::Answer);
        let total = OMPF_PAGE_BYTES as u64;
        assert!(layout.reservation_bytes() > total);
        let budget = budget(total);
        let grant = budget
            .reserve(layout.reservation_bytes(), None)
            .await
            .expect("clamped to the total, granted alone");
        assert_eq!(budget.reserved_bytes(), total);
        let payload = ready(
            layout
                .fill(&store, Some(charged(grant)))
                .await
                .expect("fills"),
        );
        assert_eq!(
            budget.reserved_bytes(),
            total,
            "the whole clamped grant on the pages"
        );
        let mut pages = payload.into_pages();
        drop(pages.next());
        assert_eq!(
            budget.reserved_bytes(),
            0,
            "the first 1 MiB page carried the whole 1 MiB grant"
        );
        drop(pages);
    }

    /// A payload with no external value to read reserves only its exact
    /// size: no replies in flight.
    #[test]
    fn a_payload_without_external_values_reserves_only_its_size() {
        let layout = lay_out(&value(500, 9), &[], OnMissing::Answer);
        assert_eq!(layout.reservation_bytes(), layout.len());
        let empty = lay_out(&[], &[], OnMissing::Answer);
        assert_eq!((empty.len(), empty.reservation_bytes()), (0, 0));
    }

    /// Each value's stored length, by hash: what the closing check finds when
    /// nothing changed.
    fn lengths_of(values: &[([u8; 32], Vec<u8>)]) -> HashMap<[u8; 32], u64> {
        values.iter().map(|(h, v)| (*h, v.len() as u64)).collect()
    }

    /// T5, a short range: a range answered with fewer bytes than asked reads
    /// as missing. A probe-mode fill answers `Built::Missing` with the
    /// value's hash, so its caller runs the miss protocol. A fetch-on-miss
    /// fill still reads every other range, then runs the closing length check
    /// over every value, the short one included. A short value the check
    /// finds not stored is restored through the checkpoint adapter, whole,
    /// into its pages, and the payload is the container the restored value
    /// makes; the restored bytes' forced top-up is repaid once they are
    /// copied, and the grant's shares as the pages drop. A short value still
    /// stored at its laid-out length is not missing, so nothing restores it:
    /// the fill fails `UNAVAILABLE` before restoring anything.
    #[tokio::test]
    async fn a_short_range_reads_as_missing_under_each_mode() {
        let (delta, values) = fixture();
        let mut store = ScriptedStore::holding(&values);
        store.short = Some(hash(2));
        let built = lay_out(&delta, &values, OnMissing::Answer)
            .fill(&store, None)
            .await
            .expect("fills");
        assert!(
            matches!(&built, Built::Missing(missing) if missing == &vec![hash(2)]),
            "probe mode answers the short value missing"
        );
        assert!(store.recovered.lock().expect("recovered").is_empty());

        // Fetch-on-miss, the short value gone by the closing check.
        let mut gone = lengths_of(&values);
        gone.remove(&hash(2));
        let mut store = ScriptedStore::holding(&values);
        store.short = Some(hash(2));
        store.closing_lengths = Some(gone);
        store.restore.insert(hash(2), values[1].1.clone());
        let budget = budget(64 << 20);
        let layout = lay_out(&delta, &values, OnMissing::Recover);
        let grant = reserve_for(&budget, &layout).await;
        let payload = ready(
            layout
                .fill(&store, Some(charged(grant)))
                .await
                .expect("fills"),
        );
        assert_eq!(*store.recovered.lock().expect("recovered"), vec![hash(2)]);
        assert_eq!(
            store.probes.load(Ordering::SeqCst),
            1,
            "the closing length check runs after a range came back short"
        );
        assert_container(
            &payload,
            &serialized(&delta, &values),
            "a fetch-on-miss fill that restored the short value is the whole container: every \
             other value's ranges read too, none left unread",
        );
        assert_eq!(
            budget.overcommit_bytes(),
            0,
            "the restored bytes' top-up is repaid once they are in their pages"
        );
        drop(payload);
        assert_budget_returned(&budget, "the pages dropped");

        // Fetch-on-miss, the short value still stored at its laid-out length.
        // The adapter would restore it: the fill must not ask.
        let mut store = ScriptedStore::holding(&values);
        store.short = Some(hash(2));
        store.restore.insert(hash(2), values[1].1.clone());
        let layout = lay_out(&delta, &values, OnMissing::Recover);
        let grant = reserve_for(&budget, &layout).await;
        let err = layout
            .fill(&store, Some(charged(grant)))
            .await
            .expect_err("a short value still stored fails the fill before any restore");
        assert_unavailable(err, SHORT_THOUGH_STORED, "a short value still stored");
        assert_eq!(
            (
                store.probes.load(Ordering::SeqCst),
                store.restores.load(Ordering::SeqCst)
            ),
            (1, 0),
            "the closing check probed it and found it stored, so it is never restored"
        );
        assert_budget_returned(&budget, "a failed fill");
    }

    /// T5, the closing check: a value whose stored length at the end of the
    /// fill is not the one the payload was laid out with — rewritten, or
    /// gone — reads as missing to a probe build, which answers
    /// `Built::Missing` with both. A fetch-on-miss build restores only a value
    /// the check finds not stored. One still stored at another length is not
    /// missing — a verified upload of another length replaced it while it was
    /// read — and a restore's write-back would add a reference to the value
    /// the store holds; so the fill fails `UNAVAILABLE` before restoring
    /// anything, the gone value beside it included, and the retry lays the
    /// payload out afresh: never `INTERNAL`, never a torn payload. With only
    /// the gone value, it alone is restored; a restore of another length than
    /// the layout's fails the fill `UNAVAILABLE`.
    #[tokio::test]
    async fn a_value_changed_or_gone_at_the_closing_check_reads_as_missing_and_only_a_gone_one_is_restored(
    ) {
        let (delta, values) = fixture();
        let mut closing = lengths_of(&values);
        closing.insert(hash(1), 69_999);
        closing.remove(&hash(3));

        let mut store = ScriptedStore::holding(&values);
        store.closing_lengths = Some(closing.clone());
        let built = lay_out(&delta, &values, OnMissing::Answer)
            .fill(&store, None)
            .await
            .expect("fills");
        assert!(
            matches!(&built, Built::Missing(missing) if missing == &vec![hash(1), hash(3)]),
            "the changed and the gone value, in part order"
        );

        // Changed and gone: nothing restored.
        let mut store = ScriptedStore::holding(&values);
        store.closing_lengths = Some(closing);
        store.restore.insert(hash(1), values[0].1.clone());
        store.restore.insert(hash(3), values[2].1.clone());
        let budget = budget(64 << 20);
        let layout = lay_out(&delta, &values, OnMissing::Recover);
        let grant = reserve_for(&budget, &layout).await;
        let err = layout
            .fill(&store, Some(charged(grant)))
            .await
            .expect_err("a value stored at another length fails the fill");
        assert_unavailable(
            err,
            STORED_AT_ANOTHER_LENGTH,
            "a value stored at another length",
        );
        assert_eq!(
            store.restores.load(Ordering::SeqCst),
            0,
            "no restore: neither of the value still stored nor of the gone one beside it"
        );
        assert_budget_returned(&budget, "a failed fill");

        // Only gone: it alone is restored.
        let mut gone = lengths_of(&values);
        gone.remove(&hash(3));
        let mut store = ScriptedStore::holding(&values);
        store.closing_lengths = Some(gone.clone());
        store.restore.insert(hash(1), values[0].1.clone());
        store.restore.insert(hash(3), values[2].1.clone());
        let layout = lay_out(&delta, &values, OnMissing::Recover);
        let grant = reserve_for(&budget, &layout).await;
        let payload = ready(
            layout
                .fill(&store, Some(charged(grant)))
                .await
                .expect("fills"),
        );
        assert_eq!(
            (
                store.probes.load(Ordering::SeqCst),
                store.restores.load(Ordering::SeqCst),
                store.recovered.lock().expect("recovered").clone()
            ),
            (1, 1, vec![hash(3)]),
            "one closing check, none after the restore, and one restore, of the gone value only"
        );
        assert_container(
            &payload,
            &serialized(&delta, &values),
            "the restored value in place",
        );
        assert_eq!(
            budget.overcommit_bytes(),
            0,
            "the restored bytes' top-up is repaid once they are in their pages"
        );
        drop(payload);
        assert_budget_returned(&budget, "the pages dropped");

        let mut store = ScriptedStore::holding(&values);
        store.closing_lengths = Some(gone);
        store.restore.insert(hash(3), value(9, 3));
        let layout = lay_out(&delta, &values, OnMissing::Recover);
        let grant = reserve_for(&budget, &layout).await;
        let err = layout
            .fill(&store, Some(charged(grant)))
            .await
            .expect_err("a restore of another length fails the build");
        assert_unavailable(
            err,
            "changed length while it was read",
            "a restore of another length",
        );
        assert_budget_returned(&budget, "a failed fill");
    }

    /// T7 (D1, PD-5): a fetch-on-miss fill whose closing check finds two
    /// values gone restores them one at a time, each under a forced top-up of
    /// its own charge only (`CHARGE_CENTI`, in whole KiB), taken before its
    /// fetch: as each restore starts, the payload budget's over-commit is that
    /// one value's charge, never the two values' together. Every top-up is
    /// repaid once its value is in its pages, and the payload is the
    /// serialized container.
    ///
    /// Fails without the change (in-place perturbation): one top-up of both
    /// values' charges taken before the first restore shows their sum as the
    /// first restore starts.
    #[tokio::test]
    async fn a_fill_restoring_two_gone_values_over_commits_one_values_charge_at_a_time() {
        let (delta, values) = fixture();
        let mut gone = lengths_of(&values);
        gone.remove(&hash(1));
        gone.remove(&hash(2));
        let budget = budget(64 << 20);
        let mut store = ScriptedStore::holding(&values);
        store.closing_lengths = Some(gone);
        store.restore.insert(hash(1), values[0].1.clone());
        store.restore.insert(hash(2), values[1].1.clone());
        store.budget_at_restore = Some(Arc::clone(&budget));
        let charge = |value: &[u8]| restore_share_bytes(value.len() as u64, CHARGE_CENTI);
        let (first, second) = (charge(&values[0].1), charge(&values[1].1));
        assert!(
            first > 0 && second > 0 && first + second < budget.total_bytes(),
            "fixture self-check: both charges are non-zero and their sum is not clamped"
        );

        let layout = lay_out(&delta, &values, OnMissing::Recover);
        let grant = reserve_for(&budget, &layout).await;
        assert_eq!(budget.overcommit_bytes(), 0, "nothing over-committed yet");
        let payload = ready(
            layout
                .fill(&store, Some(charged(grant)))
                .await
                .expect("fills"),
        );

        assert_eq!(
            *store.recovered.lock().expect("recovered"),
            vec![hash(1), hash(2)],
            "both gone values restored, in part order"
        );
        assert_eq!(
            *store
                .overcommit_at_restore
                .lock()
                .expect("overcommit_at_restore"),
            vec![first, second],
            "as each restore starts, only its own value's charge is over-committed"
        );
        assert_container(
            &payload,
            &serialized(&delta, &values),
            "both restored values in place",
        );
        assert_eq!(
            budget.overcommit_bytes(),
            0,
            "each top-up is repaid once its value is in its pages"
        );
        drop(payload);
        assert_budget_returned(&budget, "the pages dropped");
    }

    /// D1 (w6r1), the fill's hook: a fetch-on-miss fill whose closing check
    /// finds a value gone, and whose restore answers more bytes than the
    /// layout gave it (1,000 more than its 70,000), charges those extra bytes
    /// while it holds them: where the store's write-back starts, the payload
    /// budget's over-commit is the value's own forced top-up (its laid-out
    /// length's charge at `CHARGE_CENTI`) plus the charge of the bytes beyond it,
    /// each in whole KiB. The fill then fails `UNAVAILABLE` (the value changed
    /// length while it was read), holding nothing of the budget.
    ///
    /// Fails without the change (in-place perturbation): with the fill
    /// passing `&charge_nothing` instead of its excess hook, the over-commit
    /// at the write-back is the forced top-up alone.
    #[tokio::test]
    async fn a_fill_restoring_a_value_longer_than_laid_out_charges_the_excess_until_its_write_back()
    {
        let (delta, values) = fixture();
        let laid_out = values[0].1.len() as u64;
        let mut gone = lengths_of(&values);
        gone.remove(&hash(1));
        let budget = budget(64 << 20);
        let mut store = ScriptedStore::holding(&values);
        store.closing_lengths = Some(gone);
        store.restore.insert(hash(1), value(70_000 + 1_000, 1));
        store.budget_at_restore = Some(Arc::clone(&budget));
        let held = restore_share_bytes(laid_out, CHARGE_CENTI);
        let excess = (restore_charge_bytes(laid_out + 1_000, CHARGE_CENTI)
            - restore_charge_bytes(laid_out, CHARGE_CENTI))
        .div_ceil(1024)
            * 1024;
        assert!(
            excess > 0 && held + excess < budget.total_bytes(),
            "fixture self-check: a non-zero excess, nothing clamped"
        );

        let layout = lay_out(&delta, &values, OnMissing::Recover);
        let grant = reserve_for(&budget, &layout).await;
        let err = layout
            .fill(&store, Some(charged(grant)))
            .await
            .expect_err("a restore longer than laid out fails the build");
        assert_unavailable(
            err,
            "changed length while it was read",
            "a restore longer than laid out",
        );
        assert_eq!(
            *store
                .overcommit_at_restore
                .lock()
                .expect("overcommit_at_restore"),
            vec![held],
            "fixture self-check: as the restore starts, only the value's forced top-up is held"
        );
        assert_eq!(
            *store
                .overcommit_at_write_back
                .lock()
                .expect("overcommit_at_write_back"),
            vec![held + excess],
            "where the write-back starts, the bytes beyond the laid-out length are charged too"
        );
        assert_budget_returned(&budget, "a fill failed on a longer restore");
    }

    /// CR-1, the torn fill: a fetch-on-miss fill reads every range, then runs
    /// the closing length check over every value, and only then restores. A
    /// value rewritten in place while its ranges were read — each range
    /// answered whole, the bytes not the laid-out value's, the length
    /// different at the end — is caught by the closing check: still stored,
    /// at another length, so the fill fails `UNAVAILABLE` and the torn bytes
    /// are never sent. Nothing is restored, neither the rewritten value nor a
    /// value gone beside it.
    ///
    /// Fails on old code (in-place perturbation): a fill that skips the
    /// closing check freezes the payload with the torn bytes.
    #[tokio::test]
    async fn a_fetch_on_miss_fill_catches_a_value_rewritten_while_read_and_fails_unavailable_restoring_nothing(
    ) {
        let (delta, values) = fixture();
        // What the ranges read of hash(1): bytes of a rewrite, at the laid-out
        // length, so every range answers in full.
        let mut torn = values.clone();
        torn[0].1 = value(70_000, 99);
        let mut closing = lengths_of(&values);
        closing.insert(hash(1), 70_001);
        closing.remove(&hash(3));
        let mut store = ScriptedStore::holding(&torn);
        store.closing_lengths = Some(closing);
        store.restore.insert(hash(1), values[0].1.clone());
        store.restore.insert(hash(3), values[2].1.clone());

        let budget = budget(64 << 20);
        let layout = lay_out(&delta, &values, OnMissing::Recover);
        let grant = reserve_for(&budget, &layout).await;
        let err = layout
            .fill(&store, Some(charged(grant)))
            .await
            .expect_err("a value rewritten while read fails the fill, never sent torn");
        assert_unavailable(
            err,
            STORED_AT_ANOTHER_LENGTH,
            "a value rewritten while read",
        );
        assert_eq!(
            (
                store.probes.load(Ordering::SeqCst),
                store.restores.load(Ordering::SeqCst)
            ),
            (1, 0),
            "one closing check, and no restore"
        );
        let mut read: Vec<[u8; 32]> = store.ranges().iter().map(|(h, _, _)| *h).collect();
        read.sort();
        read.dedup();
        assert_eq!(
            read,
            vec![hash(1), hash(2), hash(3)],
            "every value's ranges are read before the closing check"
        );
        assert_budget_returned(&budget, "a failed fill");
    }

    /// A fetch-on-miss fill whose restore through the checkpoint adapter
    /// cannot supply a value — one the closing check found not stored — fails
    /// the build `INTERNAL` (the value is genuinely missing): never a payload
    /// with that value's bytes left as zeros. Nothing of the budget is left
    /// held.
    #[tokio::test]
    async fn a_value_the_adapter_cannot_restore_fails_internal_never_a_zero_filled_payload() {
        let (delta, values) = fixture();
        let mut gone = lengths_of(&values);
        gone.remove(&hash(2));
        let mut store = ScriptedStore::holding(&values);
        store.short = Some(hash(2));
        store.closing_lengths = Some(gone);
        let budget = budget(64 << 20);
        let layout = lay_out(&delta, &values, OnMissing::Recover);
        let grant = reserve_for(&budget, &layout).await;
        let err = layout
            .fill(&store, Some(charged(grant)))
            .await
            .expect_err("a value the restore did not answer fails the build");
        assert!(matches!(err, ServiceError::Internal(_)), "{err:?}");
        assert_eq!(*store.recovered.lock().expect("recovered"), vec![hash(2)]);
        assert_budget_returned(&budget, "a failed fill");
    }

    /// A fetch-on-miss fill whose value came back short though the closing
    /// check found it stored at its laid-out length fails `UNAVAILABLE`
    /// (retryable) before restoring anything: the short value is not missing,
    /// so nothing restores it, and a value gone beside it is not restored
    /// either — its restore's write-back would add a reference for a build
    /// that fails anyway — and the retry lays the payload out afresh.
    ///
    /// Fails on old code (in-place perturbation): without the fill's early
    /// failure on a value short though stored, the gone value is restored
    /// and only then does the closing invariant fail the build.
    #[tokio::test]
    async fn a_value_short_though_stored_at_its_length_fails_the_fill_unavailable_before_any_restore(
    ) {
        let (delta, values) = fixture();
        let mut gone = lengths_of(&values);
        gone.remove(&hash(3));
        let mut store = ScriptedStore::holding(&values);
        store.short = Some(hash(1));
        store.closing_lengths = Some(gone);
        store.restore.insert(hash(1), values[0].1.clone());
        store.restore.insert(hash(3), values[2].1.clone());
        let budget = budget(64 << 20);
        let layout = lay_out(&delta, &values, OnMissing::Recover);
        let grant = reserve_for(&budget, &layout).await;
        let err = layout
            .fill(&store, Some(charged(grant)))
            .await
            .expect_err("a value short though stored fails the fill");
        assert_eq!(
            (
                store.probes.load(Ordering::SeqCst),
                store.restores.load(Ordering::SeqCst),
                store.recovered.lock().expect("recovered").clone()
            ),
            (1, 0, Vec::new()),
            "one closing check, and no restore: neither of the short value nor of the gone one \
             beside it"
        );
        assert_unavailable(err, SHORT_THOUGH_STORED, "a value short though stored");
        assert_budget_returned(&budget, "a failed fill");
    }

    /// A payload any byte of which was left unread is never frozen: the fill
    /// fails `UNAVAILABLE` (retryable), never `Built::Ready` with those bytes
    /// zero. Driven with a fetch-on-miss fill of one value whose last range
    /// comes back short while the value is still stored at its laid-out
    /// length: it is not missing, so nothing restores it, and every byte of it
    /// but that range's is in its pages. Two guards hold this: the fill's
    /// early failure on a value short though stored, and the closing
    /// invariant at the end of the fill, a backstop no path reaches while the
    /// first holds. Either alone keeps this test green (defense in depth), so
    /// it asserts the status, not which guard's message.
    ///
    /// Fails without both (in-place perturbation): with the early failure and
    /// the closing invariant both removed, the fill freezes the payload with
    /// that range's bytes zero.
    #[tokio::test]
    async fn a_value_left_with_unread_bytes_fails_the_fill_unavailable_and_is_never_frozen() {
        let delta = value(100, 7);
        let values = vec![(hash(1), value(70_000, 1))];
        let mut store = ScriptedStore::holding(&values);
        store.short_at = Some((hash(1), EXT_VALUE_READ_CHUNK_BYTES as u64));
        store.restore.insert(hash(1), values[0].1.clone());
        let budget = budget(64 << 20);
        let layout = lay_out(&delta, &values, OnMissing::Recover);
        let grant = reserve_for(&budget, &layout).await;
        match layout.fill(&store, Some(charged(grant))).await {
            Err(ServiceError::Unavailable(_)) => {}
            Err(other) => panic!("retryable UNAVAILABLE, never anything else: {other:?}"),
            Ok(Built::Ready(payload)) => panic!(
                "a payload with its last range's bytes unread was frozen ({} B)",
                payload.len
            ),
            Ok(Built::Missing(missing)) => {
                panic!("a fetch-on-miss fill never answers missing: {missing:?}")
            }
        }
        assert_budget_returned(&budget, "a failed fill");
    }

    /// T2b, the layout's half: the container is laid out from the stored
    /// lengths the caller probed, whatever a delta's declared sizes say, so a
    /// value stored at another length than it declared (an upload admitted
    /// under `LIVE_EDIT_EXT_VALUE_VERIFY=observe`) is carried as stored, the
    /// container byte-identical to `serialize` of the stored bytes.
    #[tokio::test]
    async fn a_value_stored_at_another_length_than_declared_is_carried_as_stored() {
        let delta = value(100, 7);
        let stored = vec![(hash(9), value(3_096, 9))];
        let store = ScriptedStore::holding(&stored);
        let payload = ready(
            lay_out(&delta, &stored, OnMissing::Answer)
                .fill(&store, None)
                .await
                .expect("fills"),
        );
        assert_container(&payload, &serialized(&delta, &stored), "carried as stored");
    }

    /// A value restored through the checkpoint adapter before the layout was
    /// written back to the store by the restore and its bytes dropped: the
    /// fill reads it back in ranges like every other value — never restored a
    /// second time, never read whole — and the payload is byte-identical to
    /// the container of the stored bytes. Its bytes count in the reservation,
    /// with the replies in flight, and nothing of the budget is ever held for
    /// them as a forced top-up.
    #[tokio::test]
    async fn a_value_restored_before_the_layout_is_read_back_in_ranges_into_a_byte_identical_payload(
    ) {
        let delta = value(100, 7);
        let values = vec![(hash(5), value(70_000, 5)), (hash(6), value(3_000, 6))];
        // The restore's write-back: the store holds the restored value.
        let store = ScriptedStore::holding(&values);
        let layout = lay_out_restored(&delta, &values, &[hash(5)]);
        assert_eq!(
            layout.reservation_bytes(),
            layout.len() + EXT_RANGE_WINDOW_BYTES,
            "the whole payload, the restored value included, and the replies in flight"
        );
        let budget = budget(64 << 20);
        let grant = reserve_for(&budget, &layout).await;
        assert_eq!(budget.overcommit_bytes(), 0, "no top-up is held for it");
        let payload = ready(
            layout
                .fill(&store, Some(charged(grant)))
                .await
                .expect("fills"),
        );
        assert_container(
            &payload,
            &serialized(&delta, &values),
            "the restored value read back",
        );
        let read_back: u64 = store
            .ranges()
            .iter()
            .filter(|(h, _, _)| *h == hash(5))
            .map(|(_, _, len)| *len as u64)
            .sum();
        assert_eq!(read_back, 70_000, "every byte of it read back by range");
        assert_eq!(
            (
                store.restores.load(Ordering::SeqCst),
                store.whole_reads.load(Ordering::SeqCst),
                store.probes.load(Ordering::SeqCst),
                budget.overcommit_bytes()
            ),
            (0, 0, 1, 0),
            "no second restore, no whole-value read, one closing check, no top-up"
        );
        drop(payload);
        assert_budget_returned(&budget, "the pages dropped");
    }

    /// A value restored before the layout that does not read back whole at
    /// its restored length fails the fill `UNAVAILABLE`, before anything is
    /// sent: its write-back refused or the value gone by the read-back,
    /// replaced at another length while it was read, or a range of it short
    /// though it is stored. Never a payload with its bytes zero, and never a
    /// second restore, which the adapter here would answer; the retry lays
    /// the payload out afresh from what is stored then. No forced top-up is
    /// left charged, and the budget is all returned.
    ///
    /// Fails on old code (in-place perturbation): without the read-back
    /// check, the gone value is restored a second time.
    #[tokio::test]
    async fn a_value_restored_before_the_layout_that_does_not_read_back_whole_fails_unavailable_and_is_not_restored_again(
    ) {
        let delta = value(100, 7);
        let restored = value(70_000, 5);
        let other = (hash(6), value(3_000, 6));
        let laid_out = vec![(hash(5), restored.clone()), other.clone()];
        let budget = budget(64 << 20);

        let gone = ScriptedStore::holding(std::slice::from_ref(&other));
        let replaced = ScriptedStore::holding(&[(hash(5), value(69_000, 5)), other.clone()]);
        let mut short = ScriptedStore::holding(&laid_out);
        short.short = Some(hash(5));
        let cases = [
            ("gone at the read-back", gone, NOT_READ_BACK_WHOLE),
            (
                "replaced at another length",
                replaced,
                STORED_AT_ANOTHER_LENGTH,
            ),
            ("a range short while stored", short, NOT_READ_BACK_WHOLE),
        ];
        for (what, mut store, phrase) in cases {
            store.restore.insert(hash(5), restored.clone());
            let layout = lay_out_restored(&delta, &laid_out, &[hash(5)]);
            let grant = reserve_for(&budget, &layout).await;
            let err = layout
                .fill(&store, Some(charged(grant)))
                .await
                .expect_err("a restored value not read back whole fails the fill");
            assert_unavailable(err, phrase, what);
            assert_eq!(
                (
                    store.restores.load(Ordering::SeqCst),
                    budget.overcommit_bytes()
                ),
                (0, 0),
                "{what}: never restored a second time, and no top-up left charged"
            );
            assert_budget_returned(&budget, what);
        }
    }

    /// A catch-up whose values restored before the layout fill the whole
    /// budget or more is still served: nothing of the budget is held for them
    /// while it reserves, so its one reservation, clamped to the total, is
    /// granted at once on an idle budget and runs alone — nothing else is
    /// granted beside it — as any oversized payload does. Every value, the
    /// restored one included, is read back in ranges; nothing is charged twice
    /// (no forced top-up), and the whole total comes back as its pages drop.
    #[tokio::test]
    async fn a_restore_of_the_whole_budget_or_more_is_one_reservation_clamped_to_the_total_that_runs_alone(
    ) {
        use futures::FutureExt;
        let delta = value(100, 7);
        let values = vec![
            (hash(5), value(3 * OMPF_PAGE_BYTES, 5)),
            (hash(6), value(10_000, 6)),
        ];
        let store = ScriptedStore::holding(&values);
        let layout = lay_out_restored(&delta, &values, &[hash(5)]);
        let total = 2 * OMPF_PAGE_BYTES as u64;
        assert!(
            layout.len() > total,
            "fixture self-check: the restored value alone is over the budget"
        );
        let budget = budget(total);
        let grant = budget
            .reserve(layout.reservation_bytes(), None)
            .now_or_never()
            .expect("granted at once on an idle budget")
            .expect("the budget is open");
        assert_eq!(
            (
                grant.bytes(),
                budget.reserved_bytes(),
                budget.overcommit_bytes()
            ),
            (total, total, 0),
            "one reservation, clamped to the total, and the only charge"
        );
        assert!(
            budget.try_grant(1024).is_none(),
            "it runs alone: nothing is granted beside it"
        );
        let payload = ready(
            layout
                .fill(&store, Some(charged(grant)))
                .await
                .expect("fills"),
        );
        assert_container(
            &payload,
            &serialized(&delta, &values),
            "the restored value and the other one in their pages",
        );
        let read_back: u64 = store
            .ranges()
            .iter()
            .filter(|(h, _, _)| *h == hash(5))
            .map(|(_, _, len)| *len as u64)
            .sum();
        assert_eq!(read_back, 3 * OMPF_PAGE_BYTES as u64, "read back by range");
        assert_eq!(
            (
                store.restores.load(Ordering::SeqCst),
                budget.overcommit_bytes()
            ),
            (0, 0)
        );
        assert!(budget.reserved_bytes() <= total);
        drop(payload);
        assert_budget_returned(&budget, "every page dropped");
    }

    /// `charges` cut into batches as [`restore_in_batches`] cuts them, each
    /// batch the indices of its values.
    fn batches(charges: &[u64], total: u64) -> Vec<Vec<usize>> {
        let mut out = Vec::new();
        let mut at = 0;
        while at < charges.len() {
            let taken = restore_batch_len(&charges[at..], total);
            assert!(taken > 0, "a batch takes at least one value");
            out.push((at..at + taken).collect());
            at += taken;
        }
        out
    }

    /// T9, batch formation: a batch takes at most `EXT_RESTORE_PARALLELISM`
    /// (4) values, while their charges together stay within the budget's
    /// total; a value whose charge alone exceeds the total is a batch of its
    /// own (the budget clamps it, so it runs alone), and is never joined by
    /// the values around it; the batches keep the plan order, every value in
    /// exactly one of them; and no values make no batch.
    #[test]
    fn restore_batches_hold_at_most_four_values_within_the_total_an_oversize_one_alone_in_order() {
        const KIB: u64 = 1024;
        assert_eq!(EXT_RESTORE_PARALLELISM, 4);
        assert_eq!(restore_batch_len(&[], 100 * KIB), 0, "no values, no batch");

        // The count: ten small values, four to a batch.
        let small = vec![KIB; 10];
        assert_eq!(
            batches(&small, 100 * KIB),
            vec![vec![0, 1, 2, 3], vec![4, 5, 6, 7], vec![8, 9]]
        );

        // The total: charges of 40 KiB against 100 KiB fit two to a batch;
        // exactly the total fits, a KiB more does not.
        assert_eq!(
            batches(&[40 * KIB; 5], 100 * KIB),
            vec![vec![0, 1], vec![2, 3], vec![4]]
        );
        assert_eq!(
            restore_batch_len(&[60 * KIB, 40 * KIB, KIB], 100 * KIB),
            2,
            "a batch whose charges sum to exactly the total fits"
        );
        assert_eq!(restore_batch_len(&[60 * KIB, 41 * KIB], 100 * KIB), 1);

        // An oversize value: alone, whether it comes first or after others,
        // and the values after it batch again as usual.
        assert_eq!(
            batches(&[KIB, 2 * KIB, 500 * KIB, KIB, KIB], 100 * KIB),
            vec![vec![0, 1], vec![2], vec![3, 4]]
        );
        assert_eq!(
            batches(&[500 * KIB, 500 * KIB], 100 * KIB),
            vec![vec![0], vec![1]],
            "two oversize values are two batches of one"
        );

        // Order: every value in exactly one batch, in plan order.
        let mixed = [
            7 * KIB,
            90 * KIB,
            3 * KIB,
            120 * KIB,
            50 * KIB,
            50 * KIB,
            KIB,
            2 * KIB,
        ];
        let flat: Vec<usize> = batches(&mixed, 100 * KIB).concat();
        assert_eq!(flat, (0..mixed.len()).collect::<Vec<_>>());
        for batch in batches(&mixed, 100 * KIB) {
            let sum: u64 = batch.iter().map(|&i| mixed[i]).sum();
            assert!(batch.len() <= EXT_RESTORE_PARALLELISM);
            assert!(
                batch.len() == 1 || sum <= 100 * KIB,
                "a batch of more than one value fits in the total: {batch:?} = {sum}"
            );
        }
    }

    /// T9, the charge: a restored value of `len` bytes is charged `len ×
    /// charge_centi ÷ 100` rounded up, saturating at `u64::MAX` rather than
    /// wrapping; its share of a batch's grant is that charge rounded up to
    /// whole KiB, the budget's unit, so the shares of a batch sum exactly to
    /// what it reserves. The cases are worked at a fixed factor of 3.09 per
    /// byte, so they do not move with the catch-up's own
    /// [`EXT_RESTORE_CHARGE_CENTI`].
    #[test]
    fn a_restored_values_charge_is_its_length_times_the_factor_rounded_up_and_its_share_whole_kib()
    {
        /// The factor the cases are worked at: 3.09 per byte.
        const CHARGE_CENTI: u64 = 309;
        assert_eq!(restore_charge_bytes(0, CHARGE_CENTI), 0);
        assert_eq!(
            restore_charge_bytes(1, CHARGE_CENTI),
            4,
            "3.09 rounds up to 4"
        );
        assert_eq!(restore_charge_bytes(100, CHARGE_CENTI), 309);
        assert_eq!(
            restore_charge_bytes(101, CHARGE_CENTI),
            313,
            "312.09 rounds up"
        );
        assert_eq!(restore_charge_bytes(1 << 20, CHARGE_CENTI), 3_240_100);
        assert_eq!(restore_charge_bytes(1_000, 100), 1_000, "a factor of 1");
        assert_eq!(
            restore_charge_bytes(u64::MAX, CHARGE_CENTI),
            u64::MAX,
            "saturates, never wraps"
        );

        assert_eq!(restore_share_bytes(0, CHARGE_CENTI), 0);
        assert_eq!(restore_share_bytes(1, CHARGE_CENTI), 1024);
        assert_eq!(
            restore_share_bytes(331, CHARGE_CENTI),
            1024,
            "1,022.79 → 1,023 B"
        );
        assert_eq!(
            restore_share_bytes(332, CHARGE_CENTI),
            2048,
            "1,025.88 → 1,026 B"
        );
        assert_eq!(
            restore_share_bytes(1 << 20, CHARGE_CENTI),
            3_240_100_u64.div_ceil(1024) * 1024
        );
    }

    /// The restore memory envelope's record (plan b.tv3, ruling w6m2), which
    /// derives the catch-up's restore charge.
    const RESTORE_ENVELOPE_DERIVATION_JSON: &str =
        "docs-internal/validation/restore-memory-envelope-tv3-20261022/derivation.json";

    /// The catch-up's restore charge is the record's resident-growth rule:
    /// [`EXT_RESTORE_CHARGE_CENTI`] is the record's `value_centi`, which is
    /// ⌈1.25 × the largest measured resident growth per restored byte × 100⌉.
    /// The upload path's own expansion charge
    /// (`grpc::upload_admission::EXT_EXPANSION_CENTI`) is a separate
    /// calibration, not this test's. An edit to the
    /// constant or to the record without the other fails here; the published
    /// source tree, which has no `docs-internal/`, skips.
    #[test]
    fn ext_restore_charge_is_the_recorded_resident_growth_rule() {
        let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        if !root.join("docs-internal").is_dir() {
            eprintln!(
                "skipped: no docs-internal/ (published source tree), so no \
                 {RESTORE_ENVELOPE_DERIVATION_JSON}"
            );
            return;
        }
        let raw = std::fs::read_to_string(root.join(RESTORE_ENVELOPE_DERIVATION_JSON))
            .unwrap_or_else(|e| {
                panic!("the committed record {RESTORE_ENVELOPE_DERIVATION_JSON}: {e}")
            });
        let record: serde_json::Value =
            serde_json::from_str(&raw).expect("the record is valid JSON");
        let value_centi = record
            .pointer("/ext_restore_charge/value_centi")
            .and_then(serde_json::Value::as_u64)
            .expect("the record carries ext_restore_charge.value_centi");
        let rss_per_byte = record
            .pointer("/ext_restore_charge/measured_rss_per_byte_max")
            .and_then(serde_json::Value::as_f64)
            .expect("the record carries ext_restore_charge.measured_rss_per_byte_max");
        assert_eq!(
            EXT_RESTORE_CHARGE_CENTI, value_centi,
            "the restore charge is the record's ext_restore_charge.value_centi"
        );
        assert_eq!(
            value_centi,
            (1.25 * rss_per_byte * 100.0).ceil() as u64,
            "the record's charge is ⌈1.25 × the measured resident growth per byte × 100⌉"
        );
    }
}
