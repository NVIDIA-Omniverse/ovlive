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

//! External-value clean-up of a retiring object, bounded in memory (b.fsz
//! MR-D4): the retirement allowance `R_ref`, the size read that sizes a
//! reservation from it, the paged reference walk that finds what the object
//! still references, and the paged teardown that deletes its keys.
//!
//! Three routes retire an object, each after fencing it so no commit lands
//! (SDD §11.3) and after waking its stream: `DeleteLayer`
//! (`LayerStore::delete_layer`), the `CreateLayer` overwrite's cleanup body
//! (`OverwriteCleanupContext::run`) and the orphan sweep's generic arm
//! (`orphan_sweep::cleanup_orphaned_object`). Each route here:
//!
//! 1. **Sizes its reservation** ([`read_retirement_size`]): one pipeline on
//!    the object's slot — `HGET obj:hdr ext_gc_scheduled`, `HLEN obj:ext`,
//!    and the cardinality of each of the five structural indices (`SCARD`
//!    `obj:secs`, `obj:dels`, `obj:chs`; `ZCARD` `obj:flds`, `obj:tss`). The
//!    object's page holds `r = min(P, C + 1)` index members, `C` the largest
//!    cardinality: one more than the largest index, so a page that comes back
//!    full shows the index grew after the read (step 3). The reservation is
//!    the object's charge ([`retirement_charge_bytes`]): for an owed release
//!    `max(r, min(64, H)) × RETIREMENT_ROW_BYTES + h × H`, `H` the indexed
//!    references and `h` [`RETIREMENT_REFERENCE_CHARGE_BYTES`] (the `64` is
//!    the release's size reads in flight, [`RELEASE_SIZE_READS_IN_FLIGHT`]);
//!    for a release already scheduled, its page `r × RETIREMENT_ROW_BYTES`
//!    alone. A `WRONGTYPE` on `obj:hdr` or `obj:ext` is counted
//!    (`live_edit_snapshot_read_type_error_total`, `key_family` `header` or
//!    `ext_index`), WARNed, and never propagated: the route reserves the full
//!    page bound `W`, skips the walk and the release, and its teardown's first
//!    `UNLINK` removes the foreign key. A `WRONGTYPE` on an index's count is
//!    counted under that index's family and WARNed the same way: the route
//!    sizes its page at the full page `P`, so it reserves
//!    `W + h × H` for an owed release (`W` for one already scheduled), and
//!    goes on to the release. A foreign
//!    `obj:secs`, `obj:dels` or `obj:chs` leaves the release whole, since the
//!    walk never reads them; a foreign `obj:flds` or `obj:tss` fails the walk
//!    when it pages that index (`Internal`), so nothing is released or
//!    recorded, the route WARNs and goes on, and the object's references
//!    are left to the reference-count reconciler (`storage::ext_reconcile`).
//!    Either way the teardown then stops at the foreign index after its first
//!    `UNLINK` (step 5): on `DeleteLayer` that is an `INTERNAL` answer, with
//!    the layers-index member left (b.yz5).
//! 2. **Reserves** from `R_ref` ([`RetirementReservation`]): a second
//!    [`MemoryBudget`] the `LayerStore` holds, sized from the pod memory plan
//!    (`PodMemoryPlan::retirement_budget_bytes`, `L / 500`, a margin term)
//!    and ticking its own series
//!    ([`RETIREMENT_SERIES`](crate::storage::memory_budget::RETIREMENT_SERIES)).
//!    FIFO; an
//!    oversized request is clamped to the total and runs alone; top-ups are
//!    forced. At shutdown the allowance's waits end
//!    ([`MemoryBudget::end_waits`], first in `LayerStore::wait_for_cleanup`):
//!    a reservation queued then, or one that would have to wait, is refused
//!    `Closed`, and one that fits is still granted. The grant is released on
//!    every exit: success, error, a retry loop's exit, the owning future
//!    dropped (a deadline or a cancel), and a panic.
//! 3. **Walks the references** ([`walk_object_references`]) — the release's
//!    read, in place of a decode of the whole object: the value indices
//!    `obj:flds` and `obj:tss` paged by rank (complete because every route
//!    fenced first), each page's rows read by the bounded value-row read (no
//!    inline value is transferred) and decoded by the readers' strict hash
//!    rule, then `obj:ext` paged by `HSCAN`. Everything folds into one set of
//!    hashes; each page is dropped before the next, and no object state is
//!    built. The set's growth beyond the reservation is covered by forced
//!    top-ups, and so is a page wider than the object's: an index page that
//!    comes back full at the object's page `r` below `P` means the index grew
//!    after the size read (`DeleteLayer` sizes before its fence), so the
//!    reservation widens to the full page bound `W` before the next page is
//!    read ([`RetirementReservation::widen_page`]).
//! 4. **Shrinks to its page** once the release returns, whatever it
//!    returned: `r × RETIREMENT_ROW_BYTES`, or `W` once widened.
//! 5. **Tears the object down in pages** ([`Teardown`]): the first `UNLINK`
//!    holds `obj:hdr`, `obj:ext` and every other fixed key; the row pages
//!    follow, at the reservation's page and widening as the walk does; the
//!    five structural indices go last.
//!
//! **The bound.** Reservations hold at most `R_ref` together, beyond which one
//! clamped retirement runs alone, plus the top-ups: the references a walk
//! finds beyond those `obj:ext` counted (each walk's top-up and its grant
//! together never exceed `R_ref`), a page widened to `W` past an index that
//! grew after the size read, and a teardown's set-index step answered larger
//! than its page. The release's blob-size reads are bounded
//! ([`RELEASE_SIZE_READS_IN_FLIGHT`] in flight, each within one row's charge,
//! and the charge holds that many rows while a release is owed). The release
//! itself still carries every member at once, about
//! [`RETIREMENT_REFERENCE_CHARGE_BYTES`] per distinct reference; above
//! `R_ref` that residual is outside the allowance (b.4je).
//!
//! **Concurrency.** A retirement's charge follows its object, so small ones
//! share the allowance many at a time; one runs alone only when its own
//! charge leaves no room beside another's.

use std::collections::HashSet;
use std::future::Future;
use std::sync::{Arc, Mutex as StdMutex, MutexGuard, PoisonError};
use std::time::Duration;

use fred::interfaces::{
    ClientLike, HashesInterface, KeysInterface, SetsInterface, SortedSetsInterface,
};

use crate::error::ServiceError;
use crate::observability::metrics::{self, SharedMetrics};
use crate::storage::client::{
    is_server_wrongtype_error, is_transient_fred_error, ValkeyClient, POOL_LABEL_SNAPSHOT,
};
use crate::storage::diff_read::{
    classify_snapshot_read_err, count_object_key_refusal, read_value_row_external_hash,
    FanoutCommandFault, ObjectSnapshotReadFaults, SnapshotKeyFamily, SnapshotReadFaultCursor,
};
use crate::storage::external_values::hash_from_hex;
use crate::storage::fanout_thread::{PassAnswer, PassLane};
use crate::storage::keys::KeySchema;
use crate::storage::memory_budget::{MemoryBudget, MemoryGrant, MemoryTopUp};
use crate::storage::subscriber_admission::CatchUpRefusal;

/// `h`: the memory one distinct external reference holds through a
/// retirement's begin and enqueue, in bytes — its entry in the walk's set, its
/// `ExtRelease`, the store's member string, the enqueue's argument pair and
/// its transmitted copy, and fred's argument values, frame and encoded bytes:
/// about 750–900 B [model, b.fsz d4]; a heap test pins it, and the product
/// figure is measured in a release build. The blob-size reads are not a term
/// of it: at most [`RELEASE_SIZE_READS_IN_FLIGHT`] are in flight, covered by
/// the row term of the charge ([`retirement_charge_bytes`]). A reservation
/// charges `h` per reference `HLEN obj:ext` counts, and a walk tops up for
/// every reference beyond those.
pub const RETIREMENT_REFERENCE_CHARGE_BYTES: u64 = 1_024;

/// How many blob-size reads (`STRLEN`) a release keeps in flight at once
/// (`storage::external_values`' `release_entry_argv`), on every release path:
/// the smallest full retirement page, [`RETIREMENT_MIN_PAGE_LEN`]. Each read
/// in flight holds less than one walk row's worst case
/// ([`RETIREMENT_ROW_BYTES`]), and while a release is owed a retirement's
/// charge holds at least `min(this, references)` rows
/// ([`retirement_charge_bytes`]) — rows its walk's pages no longer use, since
/// they are dropped before the begin — so the reads in flight fit inside it.
/// A compile-time constant, with no setting.
pub const RELEASE_SIZE_READS_IN_FLIGHT: usize = RETIREMENT_MIN_PAGE_LEN as usize;

/// The worst-case heap one value row holds while the reference walk reads it,
/// in bytes [model]: its row key at a name of `MAX_NAME_BYTES` (the member's
/// hex form, 8,192 B, plus the section id and the key prefix, under 8,300 B),
/// the two wire commands' copies of that key, the row read's future (13,568 B
/// measured in a debug build, `storage::diff_read`'s issuer docs) and its
/// bounded reply (the row's non-value columns, a few hundred bytes), rounded
/// up to 40 KiB. The index page that named the row is consumed into the row
/// keys before the reads start, so a member and its key never coexist. The
/// teardown's pages hold less per member (a key and its wire copy).
pub const RETIREMENT_ROW_BYTES: u64 = 40 * 1_024;

/// The fewest index members a full retirement page reads or removes at once
/// (`P`'s floor), whatever the allowance. An object's own page may be smaller
/// ([`RetirementPaging::for_rows`]).
pub const RETIREMENT_MIN_PAGE_LEN: u64 = 64;

/// The memory one set-index member holds through a teardown step, in bytes
/// [model]: the member (a decimal section id), the two keys it names and
/// their wire copies, rounded up. A compact set's `SSCAN` step answers the
/// whole set whatever its `COUNT`; a step whose members at this charge exceed
/// the retirement's page tops up for the excess as soon as the reply is
/// in, before it builds the keys its `UNLINK`s name: the step's size is
/// known only from its reply. At the cardinality the size read counted,
/// a set's whole answer is within the object's page, which holds more
/// rows than the set has members; only a set that grew after the size read
/// is answered past it.
pub const RETIREMENT_SET_MEMBER_BYTES: u64 = 512;

/// How many times the overwrite and sweep routes retry a teardown page that
/// failed transiently after the first `UNLINK`, before giving up with a WARN.
/// Once the header is gone nothing finds the object again, so a page left
/// failed there would strand the rows and indices it was to remove.
pub const RETIREMENT_PAGE_RETRIES: u32 = 3;

/// The first backoff window of those retries (full jitter, doubling): the
/// overwrite cleanup's wake-up retry's.
const RETIREMENT_PAGE_BACKOFF_FLOOR: Duration = crate::storage::layer::OVERWRITE_WAKE_BACKOFF_FLOOR;

/// The largest backoff window of those retries.
const RETIREMENT_PAGE_BACKOFF_CAP: Duration = crate::storage::layer::OVERWRITE_WAKE_BACKOFF_CAP;

/// The most value rows the walk reads at once: one fan-out window's wire
/// commands (`config::FANOUT_WINDOW_MAX`), two per row. A page larger than
/// that (a large pod's) is read in windows of this many rows.
const RETIREMENT_ROW_WINDOW: usize = (crate::config::FANOUT_WINDOW_MAX / 2) as usize;

/// A test-only script of injected failures for the teardown's commands after
/// its first `UNLINK`, in issue order: each command pops one entry, `None`
/// running the command and `Some(e)` answering `e` in its place. An empty
/// queue runs every command
/// (`LayerStore::with_retirement_teardown_faults_for_test`). The size read
/// and the reference walk consult the snapshot-read fault seam instead
/// ([`read_retirement_size`], [`walk_object_references`]).
#[cfg(any(test, feature = "test-fault-injection"))]
pub type RetirementFaults = Arc<StdMutex<std::collections::VecDeque<Option<fred::error::Error>>>>;

/// A teardown's test-only failure script ([`Teardown::with_faults`]): the
/// failure answered in place of its first command, and the
/// [`RetirementFaults`] of the commands after it.
#[cfg(any(test, feature = "test-fault-injection"))]
#[derive(Default)]
struct TeardownFaults {
    first: StdMutex<Option<fred::error::Error>>,
    later: Option<RetirementFaults>,
}

#[cfg(any(test, feature = "test-fault-injection"))]
impl TeardownFaults {
    /// No script: every command runs.
    fn none() -> Self {
        Self::default()
    }

    /// The scripted failure of the next command — the first command's when
    /// `after_first` is false — if any.
    fn take(&self, after_first: bool) -> Option<fred::error::Error> {
        if after_first {
            self.later
                .as_ref()?
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .pop_front()
                .flatten()
        } else {
            self.first
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .take()
        }
    }
}

/// Production stand-in for the teardown's failure script: a zero-sized unit
/// struct that scripts nothing, so [`Teardown::command`] consults it with no
/// `#[cfg]` and a release build carries no lock (`storage::diff_read`'s
/// `ObjectSnapshotReadFaults` is the same pattern).
#[cfg(not(any(test, feature = "test-fault-injection")))]
struct TeardownFaults;

#[cfg(not(any(test, feature = "test-fault-injection")))]
impl TeardownFaults {
    /// Shape-independent counterpart of the test shape's constructor.
    fn none() -> Self {
        Self
    }

    /// Always `None`: release builds carry no script.
    #[inline]
    fn take(&self, _after_first: bool) -> Option<fred::error::Error> {
        None
    }
}

// ── Paging ─────────────────────────────────────────────────────────────────

/// A retirement's page: index members per page, and the bytes such a page
/// holds (its members × [`RETIREMENT_ROW_BYTES`]). The full page, `P` and
/// `W`, is the allowance's ([`Self::for_allowance`]); an object's own page,
/// sized from its size read, is at most that ([`Self::for_rows`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RetirementPaging {
    page_len: u64,
    page_bytes: u64,
}

impl RetirementPaging {
    /// The full page for an allowance of `r_ref_bytes`:
    /// `P = max(64, ⌊(R_ref / 4) ÷ RETIREMENT_ROW_BYTES⌋)` and
    /// `W = P × RETIREMENT_ROW_BYTES`, so `W ≤ R_ref / 4` wherever the floor
    /// does not bind (`R_ref` of 10 MiB or more, the 5 GiB pod's and up). The
    /// floor binds below that: at the smallest pod's 5 MiB, `W` is 2.5 MiB,
    /// half the allowance.
    pub fn for_allowance(r_ref_bytes: u64) -> Self {
        Self::of_len((r_ref_bytes / 4 / RETIREMENT_ROW_BYTES).max(RETIREMENT_MIN_PAGE_LEN))
    }

    /// The full page for the reference pod's allowance
    /// (`PodMemoryPlan::reference`): what a `LayerStore` starts with.
    pub fn reference() -> Self {
        Self::for_allowance(crate::config::PodMemoryPlan::reference().retirement_budget_bytes())
    }

    /// The page of an object whose size read asks for `rows` members per
    /// page: `rows`, at least 1 and at most this (full) page's `P`.
    pub fn for_rows(self, rows: u64) -> Self {
        Self::of_len(rows.clamp(1, self.page_len))
    }

    /// The page of `page_len` members.
    fn of_len(page_len: u64) -> Self {
        Self {
            page_len,
            page_bytes: page_len.saturating_mul(RETIREMENT_ROW_BYTES),
        }
    }

    /// Index members per page (`P` for a full page).
    pub fn page_len(&self) -> u64 {
        self.page_len
    }

    /// The bytes a page holds (`W` for a full page).
    pub fn page_bytes(&self) -> u64 {
        self.page_bytes
    }
}

/// A retirement's charge on the retirement allowance, in bytes: its page of
/// `page_len` members at [`RETIREMENT_ROW_BYTES`] each, plus `h`
/// ([`RETIREMENT_REFERENCE_CHARGE_BYTES`]) per distinct reference its release
/// carries:
///
/// `max(page_len, min(RELEASE_SIZE_READS_IN_FLIGHT, references)) ×
/// RETIREMENT_ROW_BYTES + h × references`.
///
/// The `min` term holds a row for each of the release's blob-size reads in
/// flight (at most [`RELEASE_SIZE_READS_IN_FLIGHT`], one per reference), so a
/// page smaller than that still covers them; with no reference owed it is
/// zero and the charge is the page alone.
pub fn retirement_charge_bytes(page_len: u64, references: u64) -> u64 {
    let rows = page_len.max(references.min(RELEASE_SIZE_READS_IN_FLIGHT as u64));
    rows.saturating_mul(RETIREMENT_ROW_BYTES)
        .saturating_add(references.saturating_mul(RETIREMENT_REFERENCE_CHARGE_BYTES))
}

// ── The size read ──────────────────────────────────────────────────────────

/// What a retirement's size read found about its release
/// ([`read_retirement_size`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RetirementRelease {
    /// The release is still owed: `obj:ext` indexes `indexed` references.
    Owed { indexed: u64 },
    /// The header carries `ext_gc_scheduled=1`: the release was already
    /// recorded, and the release's own check of the flag skips it.
    Scheduled,
    /// `obj:hdr` or `obj:ext` holds a foreign Valkey type (counted and
    /// WARNed): the route reserves the full page bound, skips the walk and
    /// the release, and its teardown's first `UNLINK` removes the key.
    ForeignKey,
}

/// What a retirement's size read found ([`read_retirement_size`]): its
/// release, and the largest of its five structural indices.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RetirementSize {
    /// The release the route owes.
    pub release: RetirementRelease,
    /// The largest cardinality of `obj:secs`, `obj:dels`, `obj:chs`,
    /// `obj:flds` and `obj:tss`; `None` when one of them holds a foreign
    /// Valkey type (counted and WARNed), so the route's page is the full
    /// page, its references charged beside it ([`Self::reservation_bytes`]).
    pub largest_index: Option<u64>,
}

impl RetirementSize {
    /// A retirement sized at `release` whose indices hold at most
    /// `largest_index` members each (`None`: not known).
    pub fn new(release: RetirementRelease, largest_index: Option<u64>) -> Self {
        Self {
            release,
            largest_index,
        }
    }

    /// The object's page under the full page `full`: one more member than its
    /// largest index (`min(P, C + 1)`), so a page that comes back full shows
    /// the index grew after the size read; the full page when an index could
    /// not be counted or the release met a foreign key.
    pub fn page(self, full: RetirementPaging) -> RetirementPaging {
        match (self.release, self.largest_index) {
            (RetirementRelease::ForeignKey, _) | (_, None) => full,
            (_, Some(largest)) => full.for_rows(largest.saturating_add(1)),
        }
    }

    /// The references its release owes the allowance a charge for: the
    /// indexed ones of an owed release, none otherwise.
    pub fn references(self) -> u64 {
        match self.release {
            RetirementRelease::Owed { indexed } => indexed,
            RetirementRelease::Scheduled | RetirementRelease::ForeignKey => 0,
        }
    }

    /// The bytes to reserve under the full page `full`: the object's charge
    /// ([`retirement_charge_bytes`]) at its page and its references.
    pub fn reservation_bytes(self, full: RetirementPaging) -> u64 {
        retirement_charge_bytes(self.page(full).page_len(), self.references())
    }

    /// Whether the route runs the release (the walk and the scheduling): every
    /// size but a foreign key.
    pub fn releases(self) -> bool {
        !matches!(self.release, RetirementRelease::ForeignKey)
    }
}

/// The `op` label of the size read's `HGET`.
const SIZE_READ_HGET_OP: &str = "HGET obj:hdr ext_gc_scheduled (retirement size read)";
/// The `op` label of the size read's `HLEN`.
const SIZE_READ_HLEN_OP: &str = "HLEN obj:ext (retirement size read)";

/// One of the size read's index counts ([`SIZE_READ_INDEX_COUNTS`]).
struct IndexCount {
    /// The counting command: `SCARD` for a set, `ZCARD` for a sorted set.
    sorted: bool,
    /// The index's key family, which a failure is counted under.
    family: SnapshotKeyFamily,
    /// The `op` label of its count.
    op: &'static str,
    /// Its key's builder.
    key: fn(&KeySchema, &str) -> String,
}

/// The size read's index counts, in its pipeline's order after the `HGET`
/// and the `HLEN`: the five structural indices.
const SIZE_READ_INDEX_COUNTS: [IndexCount; 5] = [
    IndexCount {
        sorted: false,
        family: SnapshotKeyFamily::SectionsIndex,
        op: "SCARD obj:secs (retirement size read)",
        key: KeySchema::obj_sections_index,
    },
    IndexCount {
        sorted: false,
        family: SnapshotKeyFamily::DeletedSectionsIndex,
        op: "SCARD obj:dels (retirement size read)",
        key: KeySchema::obj_deleted_sections_index,
    },
    IndexCount {
        sorted: false,
        family: SnapshotKeyFamily::ChildrenIndex,
        op: "SCARD obj:chs (retirement size read)",
        key: KeySchema::obj_children_index,
    },
    IndexCount {
        sorted: true,
        family: SnapshotKeyFamily::FieldsIndex,
        op: "ZCARD obj:flds (retirement size read)",
        key: KeySchema::obj_fields_index,
    },
    IndexCount {
        sorted: true,
        family: SnapshotKeyFamily::TimeSamplesIndex,
        op: "ZCARD obj:tss (retirement size read)",
        key: KeySchema::obj_time_samples_index,
    },
];

/// The size read's replies, each its own command's: the header's
/// `ext_gc_scheduled` flag, `obj:ext`'s field count, then the five index
/// counts in [`SIZE_READ_INDEX_COUNTS`]' order.
struct SizeReadReplies {
    scheduled: Result<Option<String>, fred::error::Error>,
    indexed: Result<u64, fred::error::Error>,
    counts: Vec<Result<u64, fred::error::Error>>,
}

/// The size read of a retiring object `object_id`: one pipeline on its slot,
/// on the data pool — `HGET obj:hdr ext_gc_scheduled`, `HLEN obj:ext`, then
/// the cardinality of each structural index: `SCARD` of `obj:secs`,
/// `obj:dels` and `obj:chs`, `ZCARD` of `obj:flds` and `obj:tss`.
///
/// Each failed command is counted by [`count_object_key_refusal`] under its
/// key family (`header` for the `HGET`, `ext_index` for the `HLEN`, the
/// index's own for a count), so a wrong-typed key ticks
/// `live_edit_snapshot_read_type_error_total` and a permanent refusal
/// `live_edit_snapshot_read_refused_total`. A `WRONGTYPE` is never returned:
/// on `obj:hdr` or `obj:ext` the release is
/// [`RetirementRelease::ForeignKey`], on an index the largest index is not
/// known (the route's page is the full page, its references charged beside
/// it), each WARNed here. Any other failure, on
/// any of the seven commands, is returned as the fred error (the first such
/// in pipeline order) for the caller to map: `DeleteLayer` answers it as its
/// mapping read's failure, with nothing done; the overwrite body and the
/// sweep keep everything for a later pass.
///
/// This read precedes `schedule_old_object_ext_gc`'s own `HGET obj:hdr
/// ext_gc_scheduled` on every route, which stays: it is the re-entry guard a
/// `DeleteLayer` retry-loop re-run meets.
///
/// **Test seam.** The pipeline consults the client's snapshot-read fault seam
/// (`ObjectSnapshotReadFaultInjector`) once, as a lone command of the
/// unfenced fan-out's reader (`ObjectSnapshotReader::UnfencedFanout`, family
/// `header`) on a cursor of its own, ahead of the reference walk's: a
/// `next_read` entry, or a `header` target at 0, names it. A queued hold parks
/// it, a delay holds it back, and an error answers in place of the whole
/// pipeline as the `HGET`'s own failure — counted and classified as above.
/// Once the pipeline has answered, the seam is asked again, once per command
/// in pipeline order, for an
/// `ObjectSnapshotReadFaultTarget::retirement_size_read` entry naming that
/// command's key family: the first one given replaces that command's reply,
/// the other commands keep theirs, and it is counted and classified as that
/// command's own failure ([`inject_size_read_command_fault`]).
pub async fn read_retirement_size(
    client: &ValkeyClient,
    keys: &KeySchema,
    object_id: &str,
    metrics: Option<&SharedMetrics>,
) -> Result<RetirementSize, fred::error::Error> {
    let header_key = keys.obj_header(object_id);
    let ext_key = keys.obj_ext_index(object_id);
    let mut cursor = SnapshotReadFaultCursor::unfenced_fanout();
    let fault = client
        .object_snapshot_read_faults
        .take_lone_command(&mut cursor, SnapshotKeyFamily::Header);
    if let Some(hold) = fault.hold {
        hold.park().await;
    }
    if let Some(delay) = fault.delay {
        tokio::time::sleep(delay).await;
    }
    let replies = match fault.error {
        // In place of the pipeline, as the `HGET`'s failure; the other
        // commands are never issued, so they have nothing to count.
        Some(e) => SizeReadReplies {
            scheduled: Err(e),
            indexed: Ok(0),
            counts: SIZE_READ_INDEX_COUNTS.iter().map(|_| Ok(0)).collect(),
        },
        None => {
            let mut replies =
                size_read_pipeline(client, keys, object_id, &header_key, &ext_key).await?;
            inject_size_read_command_fault(&client.object_snapshot_read_faults, &mut replies);
            replies
        }
    };

    // Count each failure under its own family, whichever it is.
    if let Err(e) = &replies.scheduled {
        count_object_key_refusal(
            metrics,
            object_id,
            SnapshotKeyFamily::Header,
            &header_key,
            SIZE_READ_HGET_OP,
            e,
        );
    }
    if let Err(e) = &replies.indexed {
        count_object_key_refusal(
            metrics,
            object_id,
            SnapshotKeyFamily::ExtIndex,
            &ext_key,
            SIZE_READ_HLEN_OP,
            e,
        );
    }
    for (index, count) in SIZE_READ_INDEX_COUNTS.iter().zip(&replies.counts) {
        if let Err(e) = count {
            count_object_key_refusal(
                metrics,
                object_id,
                index.family,
                &(index.key)(keys, object_id),
                index.op,
                e,
            );
        }
    }

    let mut foreign = false;
    let scheduled = match replies.scheduled {
        Ok(flag) => flag,
        Err(e) if is_server_wrongtype_error(&e) => {
            foreign = true;
            None
        }
        Err(e) => return Err(e),
    };
    let indexed = match replies.indexed {
        Ok(indexed) => indexed,
        Err(e) if is_server_wrongtype_error(&e) => {
            foreign = true;
            0
        }
        Err(e) => return Err(e),
    };
    let mut largest_index = Some(0_u64);
    for count in replies.counts {
        match count {
            Ok(members) => largest_index = largest_index.map(|largest| largest.max(members)),
            Err(e) if is_server_wrongtype_error(&e) => largest_index = None,
            Err(e) => return Err(e),
        }
    }
    if largest_index.is_none() {
        tracing::warn!(
            object_id = %object_id,
            "a retiring object's structural index holds a foreign Valkey type, so its size is \
             not known; reserving the full retirement page for its clean-up"
        );
    }
    if foreign {
        tracing::warn!(
            object_id = %object_id,
            "a retiring object's header or external-value reference index (obj:ext) holds a \
             foreign Valkey type; skipping the release of its external-value references, \
             which the reference-count reconciler frees later, and deleting the object's keys, \
             the foreign key with them"
        );
        return Ok(RetirementSize::new(
            RetirementRelease::ForeignKey,
            largest_index,
        ));
    }
    let release = if scheduled.as_deref() == Some("1") {
        RetirementRelease::Scheduled
    } else {
        RetirementRelease::Owed { indexed }
    };
    Ok(RetirementSize::new(release, largest_index))
}

/// The size read's pipeline ([`read_retirement_size`]): each command's own
/// reply — `HGET obj:hdr ext_gc_scheduled`, `HLEN obj:ext`, then the five
/// index counts. Fails as a whole only when the pipeline itself does
/// (queueing, or a short reply set).
async fn size_read_pipeline(
    client: &ValkeyClient,
    keys: &KeySchema,
    object_id: &str,
    header_key: &str,
    ext_key: &str,
) -> Result<SizeReadReplies, fred::error::Error> {
    let pipeline = client.pool.next().pipeline();
    let _: () = pipeline.hget(header_key, "ext_gc_scheduled").await?;
    let _: () = pipeline.hlen(ext_key).await?;
    for index in &SIZE_READ_INDEX_COUNTS {
        let key = (index.key)(keys, object_id);
        let _: () = if index.sorted {
            pipeline.zcard(key).await?
        } else {
            pipeline.scard(key).await?
        };
    }
    let replies = pipeline.try_all::<fred::types::Value>().await;
    if replies.len() != 2 + SIZE_READ_INDEX_COUNTS.len() {
        return Err(replies
            .into_iter()
            .find_map(Result::err)
            .unwrap_or_else(|| {
                fred::error::Error::new(
                    fred::error::ErrorKind::Protocol,
                    "retirement size read returned a short reply set",
                )
            }));
    }
    let mut replies = replies.into_iter();
    let scheduled = replies
        .next()
        .expect("length checked")
        .and_then(|v| v.convert::<Option<String>>());
    let indexed = replies
        .next()
        .expect("length checked")
        .and_then(|v| v.convert::<u64>());
    let counts = replies
        .map(|reply| reply.and_then(|v| v.convert::<u64>()))
        .collect();
    Ok(SizeReadReplies {
        scheduled,
        indexed,
        counts,
    })
}

/// The size read's per-command test seam ([`read_retirement_size`]): ask
/// `faults` once per pipeline command, in pipeline order — the `HGET`
/// (`header`), the `HLEN` (`ext_index`), then each index count under its
/// index's family — and replace the reply of the first command it gives an
/// error for, leaving every other reply as the store answered it. Release
/// builds' stand-in gives none, so this changes nothing there.
fn inject_size_read_command_fault(
    faults: &ObjectSnapshotReadFaults,
    replies: &mut SizeReadReplies,
) {
    if let Some(e) = faults.take_size_read_command(SnapshotKeyFamily::Header) {
        replies.scheduled = Err(e);
        return;
    }
    if let Some(e) = faults.take_size_read_command(SnapshotKeyFamily::ExtIndex) {
        replies.indexed = Err(e);
        return;
    }
    for (index, count) in SIZE_READ_INDEX_COUNTS.iter().zip(replies.counts.iter_mut()) {
        if let Some(e) = faults.take_size_read_command(index.family) {
            *count = Err(e);
            return;
        }
    }
}

// ── The reservation ────────────────────────────────────────────────────────

/// One retirement's reservation on the retirement allowance `R_ref`: a grant
/// and at most one top-up beside it, given back when the last holder drops it
/// — on every exit (module docs). Holds the retirement's charge
/// ([`retirement_charge_bytes`]) at its current page and the references
/// charged so far. Shared by the route and its walk's pass (`Arc`); its lock
/// is never held across an await.
pub struct RetirementReservation {
    budget: Arc<MemoryBudget>,
    /// The allowance's full page (`P`, `W`): what a page widens to.
    full: RetirementPaging,
    held: StdMutex<Held>,
}

/// The grant and its one top-up, and what they hold: the current page and
/// the references charged.
struct Held {
    grant: MemoryGrant,
    top_up: Option<MemoryTopUp>,
    /// The page the walk and the teardown read at: the object's own until a
    /// page comes back full below `P`, then the full page.
    page: RetirementPaging,
    /// The distinct references charged `h` each: the indexed count, raised as
    /// the walk finds more; none once the release has returned.
    references: u64,
}

impl Held {
    /// The charge at the current page and references.
    fn charge(&self) -> u64 {
        retirement_charge_bytes(self.page.page_len(), self.references)
    }

    /// Grow the held top-up, never waiting, until the grant and it together
    /// hold the charge (never past the allowance for the two together).
    fn cover_charge(&mut self) {
        let target = self.charge();
        let Held { grant, top_up, .. } = self;
        grant.top_up_to(top_up, target);
    }
}

impl std::fmt::Debug for RetirementReservation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RetirementReservation")
            .field("bytes", &self.bytes())
            .field("page", &self.page())
            .field("full", &self.full)
            .finish()
    }
}

impl RetirementReservation {
    /// Reserve the charge of a retirement at `page` (at most the full page
    /// `full`) owing `references` references
    /// ([`retirement_charge_bytes`]; a route's [`RetirementSize::page`] and
    /// [`RetirementSize::references`]) from `budget` — clamped to the
    /// allowance, so an oversized retirement runs alone — waiting in FIFO
    /// order with no deadline of its own: the caller races the wait against
    /// its own endings (a `DeleteLayer`'s RPC deadline drops it; the sweep
    /// races its cancel), and a dropped wait leaves the queue. Refused
    /// [`CatchUpRefusal::Closed`] when shutdown ends the allowance's waits
    /// ([`MemoryBudget::end_waits`]) while it is queued, or afterwards when
    /// it would have to wait; still granted afterwards when it fits. (A
    /// closed budget refuses it too.)
    pub async fn reserve(
        budget: &Arc<MemoryBudget>,
        full: RetirementPaging,
        page: RetirementPaging,
        references: u64,
    ) -> Result<Arc<Self>, CatchUpRefusal> {
        let page = full.for_rows(page.page_len());
        let bytes = retirement_charge_bytes(page.page_len(), references);
        let grant = budget.reserve(bytes, None).await?;
        Ok(Arc::new(Self {
            budget: Arc::clone(budget),
            full,
            held: StdMutex::new(Held {
                grant,
                top_up: None,
                page,
                references,
            }),
        }))
    }

    fn lock(&self) -> MutexGuard<'_, Held> {
        // Nothing panics while holding the lock, and every update leaves the
        // pair consistent.
        self.held.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Bytes held, grant and top-up together.
    pub fn bytes(&self) -> u64 {
        let held = self.lock();
        held.grant
            .bytes()
            .saturating_add(held.top_up.as_ref().map_or(0, MemoryTopUp::bytes))
    }

    /// The page the walk and the teardown read at now: the object's own, or
    /// the full page once widened ([`Self::widen_page`]).
    pub fn page(&self) -> RetirementPaging {
        self.lock().page
    }

    /// The allowance's full page, `P` and `W`.
    pub fn full_paging(&self) -> RetirementPaging {
        self.full
    }

    /// Cover `references` distinct references the walk has found so far:
    /// grow the held top-up, never waiting, until the grant and it together
    /// hold the charge at that many references (never past the allowance for
    /// the two together; the release-set residual beyond it is outside,
    /// module docs).
    pub fn cover_references(&self, references: usize) {
        let references = u64::try_from(references).unwrap_or(u64::MAX);
        let mut held = self.lock();
        held.references = held.references.max(references);
        held.cover_charge();
    }

    /// Widen to the full page before a page larger than the object's is read:
    /// an index page came back full at the object's page, so the index grew
    /// after the size read (`DeleteLayer` sizes before its fence). The held
    /// top-up grows, never waiting, until the grant and it hold the charge
    /// at the full page; the full page from then on. A no-op once at the
    /// full page. Returns the page to read at.
    pub fn widen_page(&self) -> RetirementPaging {
        let mut held = self.lock();
        if held.page.page_len() < self.full.page_len() {
            held.page = self.full;
            held.cover_charge();
        }
        held.page
    }

    /// Shrink to the page once the release has returned, whatever it
    /// returned: the references are no longer charged, and the grant and its
    /// top-up are adjusted to hold the page alone — the grant gives back
    /// everything above it, waking the queue, and a top-up remains only for
    /// a page widened past the grant. The registration, the expiry and the
    /// teardown that follow hold the page alone.
    pub fn shrink_to_page(&self) {
        let mut held = self.lock();
        held.references = 0;
        let page_bytes = held.charge();
        let Held { grant, top_up, .. } = &mut *held;
        grant.adjust_with_top_up(top_up, page_bytes);
    }

    /// A forced top-up of `bytes` beside this reservation (a teardown step
    /// answered larger than its page), repaid when the returned guard drops.
    fn step_top_up(&self, bytes: u64) -> MemoryTopUp {
        self.budget.top_up(bytes)
    }
}

/// Where a walk or a teardown takes its page from: the reservation's current
/// page, which widens ([`RetirementReservation::widen_page`]), or with none
/// behind it (a test's), a fixed page.
#[derive(Clone, Copy)]
struct Pages<'r> {
    fixed: RetirementPaging,
    reservation: Option<&'r RetirementReservation>,
}

impl Pages<'_> {
    /// The page to read at now.
    fn current(&self) -> RetirementPaging {
        self.reservation
            .map_or(self.fixed, RetirementReservation::page)
    }

    /// The page to read the next rank page at, after one came back full:
    /// widened to the full page first, on a reservation.
    fn after_full_page(&self) -> RetirementPaging {
        self.reservation
            .map_or(self.fixed, RetirementReservation::widen_page)
    }
}

// ── The reference walk ─────────────────────────────────────────────────────

/// How a retirement walks its references: the page, and the reservation its
/// top-ups go to (`None` for a walk with no allowance behind it — a test's).
#[derive(Clone)]
pub struct RetirementWalk {
    paging: RetirementPaging,
    reservation: Option<Arc<RetirementReservation>>,
}

impl std::fmt::Debug for RetirementWalk {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RetirementWalk")
            .field("paging", &self.paging)
            .field("reservation", &self.reservation)
            .finish()
    }
}

impl RetirementWalk {
    /// A walk at the reference pod's paging with no reservation behind it:
    /// for callers that drive the release directly (tests).
    pub fn unreserved() -> Self {
        Self {
            paging: RetirementPaging::reference(),
            reservation: None,
        }
    }

    /// A walk under `reservation`, at its page: every page's references are
    /// covered on it as they are found, and its page widens as an index
    /// answers past it.
    pub fn reserved(reservation: Arc<RetirementReservation>) -> Self {
        Self {
            paging: reservation.full_paging(),
            reservation: Some(reservation),
        }
    }

    /// The same walk at `page_len` index members per page (at least 1), for a
    /// walk with no reservation behind it ([`Self::unreserved`]; a reserved
    /// walk reads at its reservation's page). Test-support only: a test pages
    /// a small object several times.
    #[cfg(any(test, feature = "test-support"))]
    pub fn with_page_len(mut self, page_len: u64) -> Self {
        self.paging = RetirementPaging::of_len(page_len.max(1));
        self
    }
}

/// What a reference walk found ([`walk_object_references`]).
#[derive(Debug, Default)]
pub struct WalkedReferences {
    /// Every distinct external hash the object references: each value row's
    /// by the readers' rule, and every decodable `obj:ext` field.
    pub hashes: HashSet<[u8; 32]>,
    /// `obj:ext` fields that are not a hash's 64 hex characters, each WARNed
    /// by the walk; the caller counts them.
    pub undecodable: u64,
    /// The string bytes the walk received.
    pub bytes: u64,
    /// The value rows the walk read.
    pub rows: u64,
}

/// The release's read of a retiring object `object_id` (module docs, step 3):
/// every external hash it references, without building its state.
///
/// - **Header.** Absent → `ServiceError::NotFound`. Present without a
///   numeric `seq` — the single-column `superseded=1` header a retirement
///   fence can leave — the walk goes on (SR-1.5), so the references the rest
///   of the object holds are released.
/// - **Value indices.** `obj:flds` then `obj:tss`, by rank (`ZRANGE i
///   i+n-1`, `n` the walk's current page, at most `P`): complete because every route fenced the object first, so no
///   commit moves a member under the walk. Each page's rows are read by the
///   bounded value-row read, at most one fan-out window of them at once, and
///   decoded by the readers' strict rule
///   (`storage::diff_read::read_value_row_external_hash`): a row holding a
///   `value` names no hash, a malformed `hash` column is `Internal`.
/// - **`obj:ext`.** Paged by `HSCAN` until its cursor returns to 0; a field
///   that is not 64 hex characters is WARNed and counted
///   ([`WalkedReferences::undecodable`]) — nothing can name its blob to
///   release it, so the reference-count reconciler drains it.
///
/// Every failed command is classified under its own key family on the
/// snapshot lane (`classify_snapshot_read_err`: transient → `Aborted`,
/// permanent → `Internal`, the refusal counters ticked), `obj:ext` under
/// `ext_index`. The walk runs as one unfenced pass on a `watch-read` thread
/// (`storage::fanout_thread`, [`PassLane::Unfenced`]), so its pages allocate
/// in the fan-out arena; dropping this future aborts it at its next await.
/// Each page is dropped before the next, and the set's growth is covered on
/// the walk's reservation as each page completes
/// ([`RetirementReservation::cover_references`]). Pages are read at the
/// reservation's page, widened to the full page before the next read once
/// an index page comes back full below it
/// ([`RetirementReservation::widen_page`]).
///
/// **Test seam.** Each command — the header `EXISTS`, each index page, each
/// value row's read, each `HSCAN obj:ext` step, in that order — consults the
/// client's snapshot-read fault seam (`ObjectSnapshotReadFaultInjector`) as a
/// read of the unfenced fan-out (`ObjectSnapshotReader::UnfencedFanout`, every
/// family counted), the header, each index page and each `HSCAN` step as a
/// window's first command: a queued hold parks it, a delay holds it back, and
/// an error answers in its place, classified as that command's own failure
/// would be. On a retirement route the size read ([`read_retirement_size`])
/// consults the seam first, on a cursor of its own, so a `next_read` entry or
/// a `header` target at 0 names the size read there, and the walk meets the
/// entries behind it; on a walk run alone, a `next_read` entry names the
/// header `EXISTS`.
///
/// Records `live_edit_snapshot_read_total` and
/// `live_edit_snapshot_read_duration_seconds{consumer="ext_gc",
/// path="unfenced"}` once per walk, and on success the size histograms
/// (`_reply_bytes`: [`WalkedReferences::bytes`]; `_entries`:
/// [`WalkedReferences::rows`]).
pub async fn walk_object_references(
    client: &ValkeyClient,
    keys: &KeySchema,
    object_id: &str,
    walk: &RetirementWalk,
) -> Result<WalkedReferences, ServiceError> {
    let started = std::time::Instant::now();
    let (pass_client, pass_keys, pass_object_id) =
        (client.clone(), keys.clone(), object_id.to_owned());
    let paging = walk.paging;
    let reservation = walk.reservation.clone();
    let result = client
        .fanout_threads()
        .run_pass(
            PassLane::Unfenced,
            move |_link| async move {
                walk_pass(
                    &pass_client,
                    &pass_keys,
                    &pass_object_id,
                    paging,
                    reservation.as_deref(),
                )
                .await
            },
            // The walk asks its caller nothing.
            |_request| PassAnswer::GoOn,
        )
        .await;
    let elapsed = started.elapsed();
    let outcome = match &result {
        Ok(_) => metrics::SNAPSHOT_READ_OUTCOME_OK,
        Err(ServiceError::NotFound(_)) => metrics::SNAPSHOT_READ_OUTCOME_NOT_FOUND,
        Err(ServiceError::Aborted(_)) => metrics::SNAPSHOT_READ_OUTCOME_TRANSIENT_ERROR,
        Err(_) => metrics::SNAPSHOT_READ_OUTCOME_PERMANENT_ERROR,
    };
    if let Some(m) = client.metrics_snapshot() {
        m.observe_snapshot_read(
            metrics::SNAPSHOT_READ_CONSUMER_EXT_GC,
            metrics::SNAPSHOT_READ_PATH_UNFENCED,
            outcome,
            elapsed,
        );
        if let Ok(walked) = &result {
            m.observe_snapshot_read_size(
                metrics::SNAPSHOT_READ_CONSUMER_EXT_GC,
                walked.bytes,
                walked.rows,
            );
        }
    }
    result
}

/// The walk's pass, on its `watch-read` thread ([`walk_object_references`]).
async fn walk_pass(
    client: &ValkeyClient,
    keys: &KeySchema,
    object_id: &str,
    paging: RetirementPaging,
    reservation: Option<&RetirementReservation>,
) -> Result<WalkedReferences, ServiceError> {
    // Held for the whole walk: the in-use gauge counts snapshot-lane reads in
    // flight.
    let (pool, _in_use) = client.acquire_snapshot();
    let classify = |family: SnapshotKeyFamily, key: &str, op: &str, e: fred::error::Error| {
        classify_snapshot_read_err(client, POOL_LABEL_SNAPSHOT, object_id, family, key, op, e)
    };
    let mut walked = WalkedReferences::default();
    // The test-only seam's position in this walk.
    let mut cursor = SnapshotReadFaultCursor::unfenced_fanout();
    let faults = &client.object_snapshot_read_faults;

    // The header: present or not. `EXISTS` answers whatever the key's type; a
    // wrong-typed header was counted by the route's size read, which then
    // skipped the walk.
    let header_key = keys.obj_header(object_id);
    let fault = faults.take_fanout_command(&mut cursor, SnapshotKeyFamily::Header, true);
    let exists = with_fault(fault, || pool.exists::<u64, _>(&header_key))
        .await
        .map_err(|e| {
            classify(
                SnapshotKeyFamily::Header,
                &header_key,
                "EXISTS obj header",
                e,
            )
        })?;
    if exists == 0 {
        return Err(ServiceError::NotFound(format!(
            "object {object_id} not found"
        )));
    }

    let pages = Pages {
        fixed: paging,
        reservation,
    };
    let value_indices = [
        (
            keys.obj_fields_index(object_id),
            SnapshotKeyFamily::FieldsIndex,
            SnapshotKeyFamily::Field,
            "ZRANGE fields index",
        ),
        (
            keys.obj_time_samples_index(object_id),
            SnapshotKeyFamily::TimeSamplesIndex,
            SnapshotKeyFamily::TimeSample,
            "ZRANGE time samples index",
        ),
    ];
    for (index_key, index_family, row_family, index_op) in value_indices {
        let mut rank_pages = WalkRankPages {
            client,
            pool,
            keys,
            object_id,
            index_key,
            index_family,
            row_family,
            index_op,
            cursor: &mut cursor,
            walked: &mut walked,
            reservation,
        };
        page_by_rank(&mut rank_pages, pages).await?;
    }

    // The recorded reference index, by `HSCAN` to cursor 0.
    let ext_key = keys.obj_ext_index(object_id);
    let mut scan_cursor = "0".to_owned();
    loop {
        let fault = faults.take_fanout_command(&mut cursor, SnapshotKeyFamily::ExtIndex, true);
        let (next, entries) = with_fault(fault, || {
            scan_step(
                pool,
                "HSCAN",
                &ext_key,
                &scan_cursor,
                pages.current().page_len(),
            )
        })
        .await
        .map_err(|e| classify(SnapshotKeyFamily::ExtIndex, &ext_key, "HSCAN obj:ext", e))?;
        walked.bytes = walked
            .bytes
            .saturating_add(entries.iter().map(|e| e.len() as u64).sum());
        // Field, value, field, value: the fields are the hashes.
        for field in entries.into_iter().step_by(2) {
            match hash_from_hex(&field) {
                Some(hash) => {
                    walked.hashes.insert(hash);
                }
                None => {
                    walked.undecodable = walked.undecodable.saturating_add(1);
                    tracing::warn!(
                        old_object_id = %object_id,
                        field = %field,
                        "dropping an undecodable le:{{oid}}:obj:ext field while scheduling \
                         retirement releases"
                    );
                }
            }
        }
        if let Some(reservation) = reservation {
            reservation.cover_references(walked.hashes.len());
        }
        if next == "0" {
            break;
        }
        scan_cursor = next;
    }
    Ok(walked)
}

/// The walk's per-page reader over one value index ([`page_by_rank`]): each
/// page's `ZRANGE` under the snapshot-read fault seam, then every member's
/// value row read by the readers' rule
/// ([`read_value_row_external_hash`]), at most [`RETIREMENT_ROW_WINDOW`] rows
/// at once, its hashes folded into the walk's set and covered on its
/// reservation.
struct WalkRankPages<'w> {
    client: &'w ValkeyClient,
    pool: &'w fred::clients::Pool,
    keys: &'w KeySchema,
    object_id: &'w str,
    index_key: String,
    index_family: SnapshotKeyFamily,
    row_family: SnapshotKeyFamily,
    index_op: &'static str,
    cursor: &'w mut SnapshotReadFaultCursor,
    walked: &'w mut WalkedReferences,
    reservation: Option<&'w RetirementReservation>,
}

impl RankPageReader for WalkRankPages<'_> {
    type Error = ServiceError;

    async fn read_page(&mut self, start: i64, stop: i64) -> Result<Vec<String>, ServiceError> {
        let fault = self.client.object_snapshot_read_faults.take_fanout_command(
            self.cursor,
            self.index_family,
            true,
        );
        with_fault(fault, || {
            zrange_page(self.pool, &self.index_key, start, stop)
        })
        .await
        .map_err(|e| {
            classify_snapshot_read_err(
                self.client,
                POOL_LABEL_SNAPSHOT,
                self.object_id,
                self.index_family,
                &self.index_key,
                self.index_op,
                e,
            )
        })
    }

    async fn take_page(&mut self, members: Vec<String>) -> Result<(), ServiceError> {
        let (client, pool, keys, object_id, row_family) = (
            self.client,
            self.pool,
            self.keys,
            self.object_id,
            self.row_family,
        );
        let faults = &client.object_snapshot_read_faults;
        self.walked.bytes = self
            .walked
            .bytes
            .saturating_add(members.iter().map(|m| m.len() as u64).sum());
        // The page's members become its row keys, each member dropped as its
        // key is built.
        let row_keys: Vec<String> = members
            .into_iter()
            .map(|member| match row_family {
                SnapshotKeyFamily::TimeSample => keys.obj_ts_from_member(object_id, &member),
                _ => keys.obj_field_from_member(object_id, &member),
            })
            .collect();
        let mut rows = row_keys.into_iter();
        loop {
            let window: Vec<String> = rows.by_ref().take(RETIREMENT_ROW_WINDOW).collect();
            if window.is_empty() {
                break;
            }
            let cursor = &mut *self.cursor;
            let reads = window.into_iter().map(|key| {
                let fault = faults.take_fanout_command(cursor, row_family, false);
                async move {
                    if let Some(hold) = fault.hold {
                        hold.park().await;
                    }
                    if let Some(delay) = fault.delay {
                        tokio::time::sleep(delay).await;
                    }
                    match fault.error {
                        Some(e) => Err(classify_snapshot_read_err(
                            client,
                            POOL_LABEL_SNAPSHOT,
                            object_id,
                            row_family,
                            &key,
                            "HSTRLEN value row",
                            e,
                        )),
                        None => {
                            read_value_row_external_hash(client, pool, object_id, row_family, key)
                                .await
                        }
                    }
                }
            });
            for (hash, received) in futures::future::try_join_all(reads).await? {
                self.walked.rows = self.walked.rows.saturating_add(1);
                self.walked.bytes = self.walked.bytes.saturating_add(received);
                if let Some(hash) = hash {
                    self.walked.hashes.insert(hash);
                }
            }
        }
        if let Some(reservation) = self.reservation {
            reservation.cover_references(self.walked.hashes.len());
        }
        Ok(())
    }
}

// ── The member pager ───────────────────────────────────────────────────────

/// The per-page side of the member pager ([`page_by_rank`]): how one route
/// reads a page of a sorted index's members, and what it does with them — the
/// reference walk reads the value rows they name ([`WalkRankPages`]), the
/// teardown unlinks the keys they name ([`TeardownRankPages`]). Each route
/// keeps its own seams, error type and retry posture here.
trait RankPageReader {
    /// The route's error.
    type Error;

    /// Read the members at ranks `start..=stop` (a [`zrange_page`]).
    async fn read_page(&mut self, start: i64, stop: i64) -> Result<Vec<String>, Self::Error>;

    /// Take one page's members. The pager reads the next page only once this
    /// returns, so each page is dropped before the next.
    async fn take_page(&mut self, members: Vec<String>) -> Result<(), Self::Error>;
}

/// The member pager over a sorted index: `ZRANGE i i+n-1` pages by rank, `n`
/// the page length of `pages`, each handed to `reader` before the next is
/// read, until a page answers fewer than `n` members. A page that comes back
/// full may have more behind it: `pages` widens to the full page before the
/// next is read ([`Pages::after_full_page`]), so a page larger than the
/// object's is never read on a reservation that does not cover it. Complete
/// because no member moves under it: every route fenced the object first,
/// and the teardown removes no index member until the indices go. The first
/// failure ends it.
async fn page_by_rank<R: RankPageReader>(reader: &mut R, pages: Pages<'_>) -> Result<(), R::Error> {
    let rank = |r: u64| i64::try_from(r).unwrap_or(i64::MAX);
    let mut page_len = pages.current().page_len().max(1);
    let mut start: u64 = 0;
    loop {
        let stop = start.saturating_add(page_len - 1);
        let members = reader.read_page(rank(start), rank(stop)).await?;
        let read = members.len() as u64;
        reader.take_page(members).await?;
        if read < page_len {
            return Ok(());
        }
        start = start.saturating_add(page_len);
        page_len = pages.after_full_page().page_len().max(1);
    }
}

/// One page of a sorted index: `ZRANGE key start stop`, by rank.
fn zrange_page<'p>(
    pool: &'p fred::clients::Pool,
    key: &'p str,
    start: i64,
    stop: i64,
) -> impl Future<Output = Result<Vec<String>, fred::error::Error>> + 'p {
    pool.zrange::<Vec<String>, _, _, _>(key, start, stop, None, false, None, false)
}

/// Apply the test-only seam's `fault` to one walk command, then issue it
/// through `issue` unless the fault answers an error in its place: park on its
/// hold, wait out its delay. Nothing in a release build, whose faults are
/// always empty.
async fn with_fault<T, F, Fut>(fault: FanoutCommandFault, issue: F) -> Result<T, fred::error::Error>
where
    F: FnOnce() -> Fut,
    Fut: Future<Output = Result<T, fred::error::Error>>,
{
    if let Some(hold) = fault.hold {
        hold.park().await;
    }
    if let Some(delay) = fault.delay {
        tokio::time::sleep(delay).await;
    }
    match fault.error {
        Some(e) => Err(e),
        None => issue().await,
    }
}

/// One `SSCAN` or `HSCAN` step (`command`) of `key` from `cursor`, asking for
/// `count` elements: the next cursor (`"0"` when the scan is complete) and
/// the step's elements — a set's members, or a hash's fields and values
/// interleaved. A set the store keeps in a compact encoding answers whole,
/// whatever `count`.
async fn scan_step(
    pool: &fred::clients::Pool,
    command: &'static str,
    key: &str,
    cursor: &str,
    count: u64,
) -> Result<(String, Vec<String>), fred::error::Error> {
    let reply: fred::types::Value = pool
        .next()
        .custom(
            fred::types::CustomCommand::new_static(
                command,
                fred::types::ClusterHash::FirstValue,
                false,
            ),
            vec![
                fred::types::Value::from(key),
                fred::types::Value::from(cursor),
                fred::types::Value::from("COUNT"),
                fred::types::Value::from(i64::try_from(count).unwrap_or(i64::MAX)),
            ],
        )
        .await?;
    let shape_err = || {
        fred::error::Error::new(
            fred::error::ErrorKind::Protocol,
            format!("{command} {key} returned an unexpected reply"),
        )
    };
    let mut parts = reply.into_array().into_iter();
    let (Some(next), Some(elements), None) = (parts.next(), parts.next(), parts.next()) else {
        return Err(shape_err());
    };
    let next = next.as_string().ok_or_else(shape_err)?;
    let elements: Vec<String> = elements.convert()?;
    Ok((next, elements))
}

// ── The paged teardown ─────────────────────────────────────────────────────

/// A teardown command that failed: its `op` label, the full key it addressed
/// and the fred error, for the caller's own error type and log line
/// (`DeleteLayer` and the overwrite body: a WARN on the data pool; the sweep:
/// `SweepError` on the maintenance pool). Every route reports it as
/// [`Self::label`] in a line that names the object, so an operator can find
/// what a failed teardown left behind — above all a foreign index, which the
/// teardown's own pages count nowhere (b.yz5): the route's size read counted
/// it, under its own family, when the route ran one (the sweep runs none
/// for an orphan whose fence failed or whose id is not numeric).
#[derive(Debug)]
pub struct TeardownFailure {
    /// The failed command's label.
    pub op: &'static str,
    /// The key the command addressed: the index an index page read, or the
    /// first key of a failed `UNLINK`.
    pub key: String,
    /// Its error.
    pub error: fred::error::Error,
}

impl TeardownFailure {
    /// The failed command as every route reports it: `<op> on <key>`, the
    /// command's label and the full key it addressed. The route's line names
    /// the object beside it.
    pub fn label(&self) -> String {
        format!("{} on {}", self.op, self.key)
    }
}

/// One retiring object's paged key deletion, the same on every route but for
/// its pool and its retry posture (module docs, step 5).
///
/// Callers run it only after the object's stream carries `deleted` and has its
/// expiry (`wake_retired_object_stream`, `pexpire_layer_stream`): no stream key
/// is in any page, and once the header is gone nothing can find the object to
/// expire it.
///
/// 1. **The first `UNLINK`**: `obj:hdr`, `obj:ext` and every other fixed key
///    ([`KeySchema::object_teardown_first_keys`]), header first and `obj:ext`
///    in the same command (the settle's liveness gate). A failure here keeps
///    the header and `obj:ext`, so the object is found again.
/// 2. **The row pages**, each reading at most `P` index members and
///    `UNLINK`ing the keys they name, at most `UNLINK_CHUNK_SIZE` per command:
///    `obj:secs`, `obj:dels` and `obj:chs` by `SSCAN` (a section's hash and
///    children list; a tombstone's; a recorded parent's children list), then
///    `obj:flds` and `obj:tss` by rank. No index member is removed until the
///    indices go, so the ranks hold still and a scan sees every member. A set
///    step answered whole by a compact set, whose members at
///    [`RETIREMENT_SET_MEMBER_BYTES`] exceed the reservation's page, tops up
///    for the excess on the reservation before it builds its keys. Pages are
///    read at the reservation's page ([`RetirementReservation::page`]); a
///    rank page that comes back full widens it before the next is read, as
///    the walk's does.
/// 3. **The last `UNLINK`**: the five structural indices
///    ([`KeySchema::object_teardown_index_keys`]).
///
/// [`KeySchema::object_cleanup_keys`] stays the inventory authority: it is
/// built from the same pieces, so the pages remove exactly its keys.
///
/// With [`Self::retrying_later_pages`] (the overwrite body and the sweep, which
/// cannot come back to an object whose header is gone), a command after the
/// first `UNLINK` that fails transiently is retried in place up to
/// [`RETIREMENT_PAGE_RETRIES`] times, each after a full-jitter backoff (100 ms
/// doubling to 1 s); `DeleteLayer` retries the whole step from its retry loop
/// instead, from the indices still standing. The first failure that is not
/// retried ends the teardown.
pub struct Teardown<'a> {
    pool: &'a fred::clients::Pool,
    keys: &'a KeySchema,
    object_id: &'a str,
    paging: RetirementPaging,
    reservation: Option<&'a RetirementReservation>,
    retry_later_pages: bool,
    /// The test-only failure script; a zero-sized stand-in in a release
    /// build.
    faults: TeardownFaults,
}

impl<'a> Teardown<'a> {
    /// A teardown of `object_id` on `pool`, at `paging` until a reservation
    /// is given ([`Self::reserved`], whose page it reads at then), with no
    /// in-place retry.
    pub fn new(
        pool: &'a fred::clients::Pool,
        keys: &'a KeySchema,
        object_id: &'a str,
        paging: RetirementPaging,
    ) -> Self {
        Self {
            pool,
            keys,
            object_id,
            paging,
            reservation: None,
            retry_later_pages: false,
            faults: TeardownFaults::none(),
        }
    }

    /// Read at `reservation`'s page, widening it as the walk does, and top up
    /// on it for an oversized set step.
    pub fn reserved(mut self, reservation: Option<&'a RetirementReservation>) -> Self {
        self.reservation = reservation;
        self
    }

    /// Retry a transient failure after the first `UNLINK` in place (the
    /// overwrite body and the sweep).
    pub fn retrying_later_pages(mut self) -> Self {
        self.retry_later_pages = true;
        self
    }

    /// Answer `fault` in place of the first command (the first `UNLINK`), and
    /// `faults`' entries in place of the commands after it, in order.
    /// Test-only.
    #[cfg(any(test, feature = "test-fault-injection"))]
    pub fn with_faults(
        mut self,
        first: Option<fred::error::Error>,
        faults: Option<RetirementFaults>,
    ) -> Self {
        self.faults = TeardownFaults {
            first: StdMutex::new(first),
            later: faults,
        };
        self
    }

    /// Run the teardown (see the type's docs).
    pub async fn run(&self) -> Result<(), TeardownFailure> {
        let keys = self.keys;
        let oid = self.object_id;
        self.unlink(
            "UNLINK object keys (the teardown's first)",
            false,
            keys.object_teardown_first_keys(oid),
        )
        .await?;

        self.set_pages(
            keys.obj_sections_index(oid),
            "SSCAN obj sections index",
            |sid, out| out.extend(keys.section_teardown_keys(oid, sid)),
        )
        .await?;
        self.set_pages(
            keys.obj_deleted_sections_index(oid),
            "SSCAN obj deleted sections index",
            |sid, out| out.extend(keys.tombstone_teardown_keys(oid, sid)),
        )
        .await?;
        self.set_pages(
            keys.obj_children_index(oid),
            "SSCAN obj children index",
            |pid, out| out.push(keys.obj_children(oid, pid)),
        )
        .await?;
        self.rank_pages(
            keys.obj_fields_index(oid),
            "ZRANGE obj fields index",
            |member| keys.obj_field_from_member(oid, member),
        )
        .await?;
        self.rank_pages(
            keys.obj_time_samples_index(oid),
            "ZRANGE obj time samples index",
            |member| keys.obj_ts_from_member(oid, member),
        )
        .await?;

        self.unlink(
            "UNLINK object indices (the teardown's last)",
            true,
            keys.object_teardown_index_keys(oid),
        )
        .await
    }

    /// One `SSCAN`-paged set index: each step's members' keys, unlinked.
    async fn set_pages(
        &self,
        index_key: String,
        op: &'static str,
        keys_of: impl Fn(&str, &mut Vec<String>),
    ) -> Result<(), TeardownFailure> {
        let mut cursor = "0".to_owned();
        loop {
            let page = self.pages().current();
            let (next, members) = self
                .command(op, &index_key, true, || {
                    scan_step(self.pool, "SSCAN", &index_key, &cursor, page.page_len())
                })
                .await?;
            // A compact set answers whole: cover what it costs beyond the page.
            let step_bytes = (members.len() as u64).saturating_mul(RETIREMENT_SET_MEMBER_BYTES);
            let _step_top_up = match self.reservation {
                Some(reservation) if step_bytes > page.page_bytes() => {
                    Some(reservation.step_top_up(step_bytes - page.page_bytes()))
                }
                _ => None,
            };
            let mut doomed = Vec::with_capacity(members.len().saturating_mul(2));
            for member in &members {
                keys_of(member, &mut doomed);
            }
            drop(members);
            self.unlink("UNLINK object rows", true, doomed).await?;
            if next == "0" {
                return Ok(());
            }
            cursor = next;
        }
    }

    /// One rank-paged sorted index ([`page_by_rank`]): each page's members'
    /// keys, unlinked.
    async fn rank_pages(
        &self,
        index_key: String,
        op: &'static str,
        key_of: impl Fn(&str) -> String,
    ) -> Result<(), TeardownFailure> {
        let mut rank_pages = TeardownRankPages {
            teardown: self,
            index_key,
            op,
            key_of,
        };
        page_by_rank(&mut rank_pages, self.pages()).await
    }

    /// Where this teardown takes its page from.
    fn pages(&self) -> Pages<'a> {
        Pages {
            fixed: self.paging,
            reservation: self.reservation,
        }
    }

    /// `UNLINK` `doomed`, at most `UNLINK_CHUNK_SIZE` keys per command (all on
    /// the object's slot), each chunk cloned into the command's own key list.
    async fn unlink(
        &self,
        op: &'static str,
        after_first: bool,
        doomed: Vec<String>,
    ) -> Result<(), TeardownFailure> {
        for chunk in doomed.chunks(crate::storage::layer::UNLINK_CHUNK_SIZE) {
            let first_key = chunk.first().map_or("", String::as_str);
            self.command(op, first_key, after_first, || {
                self.pool.unlink::<(), _>(chunk.to_vec())
            })
            .await?;
        }
        Ok(())
    }

    /// Issue one teardown command through `issue`, or answer its scripted
    /// failure; retry a transient failure in place when allowed (the type's
    /// docs). `key` is the key the command addresses, named by its failure.
    async fn command<T, F, Fut>(
        &self,
        op: &'static str,
        key: &str,
        after_first: bool,
        issue: F,
    ) -> Result<T, TeardownFailure>
    where
        F: Fn() -> Fut,
        Fut: Future<Output = Result<T, fred::error::Error>>,
    {
        let mut retries: u32 = 0;
        loop {
            let result = match self.faults.take(after_first) {
                Some(e) => Err(e),
                None => issue().await,
            };
            match result {
                Ok(value) => return Ok(value),
                Err(e)
                    if after_first
                        && self.retry_later_pages
                        && retries < RETIREMENT_PAGE_RETRIES
                        && is_transient_fred_error(&e) =>
                {
                    tracing::debug!(
                        object_id = %self.object_id,
                        op,
                        key,
                        attempt = retries + 1,
                        error = %e,
                        "a retiring object's teardown page failed transiently after its header \
                         was removed; retrying it in place"
                    );
                    tokio::time::sleep(crate::storage::layer::full_jitter_backoff(
                        retries,
                        RETIREMENT_PAGE_BACKOFF_FLOOR,
                        RETIREMENT_PAGE_BACKOFF_CAP,
                    ))
                    .await;
                    retries += 1;
                }
                Err(e) => {
                    return Err(TeardownFailure {
                        op,
                        key: key.to_owned(),
                        error: e,
                    })
                }
            }
        }
    }
}

/// The teardown's per-page reader over one sorted index ([`page_by_rank`]):
/// each page's `ZRANGE` as a teardown command (scripted failures, in-place
/// retry), then the keys its members name (`key_of`) unlinked.
struct TeardownRankPages<'t, 'a, K> {
    teardown: &'t Teardown<'a>,
    index_key: String,
    op: &'static str,
    key_of: K,
}

impl<K: Fn(&str) -> String> RankPageReader for TeardownRankPages<'_, '_, K> {
    type Error = TeardownFailure;

    async fn read_page(&mut self, start: i64, stop: i64) -> Result<Vec<String>, TeardownFailure> {
        let (pool, index_key) = (self.teardown.pool, self.index_key.as_str());
        self.teardown
            .command(self.op, index_key, true, || {
                zrange_page(pool, index_key, start, stop)
            })
            .await
    }

    async fn take_page(&mut self, members: Vec<String>) -> Result<(), TeardownFailure> {
        let doomed: Vec<String> = members.into_iter().map(|m| (self.key_of)(&m)).collect();
        self.teardown
            .unlink("UNLINK object rows", true, doomed)
            .await
    }
}

#[cfg(test)]
mod tests {
    //! The retirement allowance's arithmetic and the reservation's cover and
    //! shrink rules, on a `MemoryBudget` alone (no store). The routes that
    //! reserve, walk and tear down against Valkey are the integration tests
    //! `retirement_walk_test.rs`, `retirement_budget_test.rs` and
    //! `retirement_teardown_test.rs`; the per-row and per-member heap
    //! constants are pinned by `tests/upload_memory_envelope`.

    use super::*;

    const MIB: u64 = 1024 * 1024;

    fn allowance(bytes: u64) -> Arc<MemoryBudget> {
        Arc::new(MemoryBudget::with_series(
            crate::storage::memory_budget::kib_for_bytes(bytes),
            Arc::new(StdMutex::new(None)),
            &crate::storage::memory_budget::RETIREMENT_SERIES,
        ))
    }

    /// `P = max(64, ⌊(R_ref / 4) ÷ RETIREMENT_ROW_BYTES⌋)`, `W = P ×
    /// RETIREMENT_ROW_BYTES`: `W ≤ R_ref / 4` at every allowance of 10 MiB
    /// (the 5 GiB pod's) or more, so four retirements at the full page fit
    /// side by side; below that the floor binds — at the smallest pod's
    /// 5 MiB, `P` is its floor and `W` half the allowance. From 10 MiB up `W`
    /// is within one row below the quarter, so `P` follows the allowance: at
    /// 20 MiB it is twice the floor.
    #[test]
    fn retirement_paging_holds_its_full_page_to_a_quarter_of_the_allowance_from_10_mib_up() {
        let at = |mib: u64| RetirementPaging::for_allowance(mib * MIB);
        for kib in (10 * 1024..=64 * 1024).step_by(37) {
            let paging = RetirementPaging::for_allowance(kib * 1024);
            let quarter = kib * 1024 / 4;
            assert!(
                paging.page_bytes() <= quarter,
                "R_ref = {kib} KiB: W = {} B must be at most a quarter",
                paging.page_bytes()
            );
            assert!(
                paging.page_bytes() > quarter - RETIREMENT_ROW_BYTES,
                "R_ref = {kib} KiB: W = {} B must be within one row of a quarter",
                paging.page_bytes()
            );
            assert_eq!(
                paging.page_bytes(),
                paging.page_len() * RETIREMENT_ROW_BYTES
            );
        }
        assert_eq!(
            at(10).page_len(),
            10 * MIB / 4 / RETIREMENT_ROW_BYTES,
            "at 10 MiB the quarter sets P"
        );
        assert_eq!(
            at(20).page_len(),
            20 * MIB / 4 / RETIREMENT_ROW_BYTES,
            "at 20 MiB the quarter sets P above the floor"
        );
        assert_eq!(at(20).page_len(), 2 * RETIREMENT_MIN_PAGE_LEN);
        let smallest = at(5);
        assert_eq!(
            smallest.page_len(),
            RETIREMENT_MIN_PAGE_LEN,
            "the floor binds"
        );
        assert!(smallest.page_bytes() > 5 * MIB / 4);
        assert!(smallest.page_bytes() <= 5 * MIB / 2);
    }

    /// An object's page is one member past its largest index, at least 1 and
    /// at most the full page.
    #[test]
    fn an_objects_page_is_its_rows_clamped_to_one_and_the_full_page() {
        let full = RetirementPaging::for_allowance(10 * MIB);
        assert_eq!(full.for_rows(0).page_len(), 1);
        assert_eq!(full.for_rows(7).page_len(), 7);
        assert_eq!(full.for_rows(7).page_bytes(), 7 * RETIREMENT_ROW_BYTES);
        assert_eq!(full.for_rows(full.page_len() + 1), full);
    }

    /// The release's size reads in flight ([`RELEASE_SIZE_READS_IN_FLIGHT`])
    /// fit inside every charge that owes a release: each in-flight read holds
    /// less than a walk row's worst case ([`RETIREMENT_ROW_BYTES`]), and the
    /// charge holds a row for each read in flight
    /// (`min(RELEASE_SIZE_READS_IN_FLIGHT, references)`), whatever the
    /// object's page; with no reference owed, the charge is the page alone.
    /// The full page holds every read in flight at every allowance.
    #[test]
    fn every_charge_owing_a_release_holds_a_row_for_each_size_read_in_flight() {
        let reads = RELEASE_SIZE_READS_IN_FLIGHT as u64;
        let h = RETIREMENT_REFERENCE_CHARGE_BYTES;
        for page_len in [1, 2, 11, reads - 1, reads, reads + 40] {
            for references in [0, 1, 5, reads - 1, reads, reads + 1, 10_000] {
                let charge = retirement_charge_bytes(page_len, references);
                assert!(
                    charge >= references.min(reads) * RETIREMENT_ROW_BYTES + h * references,
                    "page {page_len}, {references} references: {charge} B must hold a row per \
                     read in flight and h per reference"
                );
                assert!(
                    charge >= page_len * RETIREMENT_ROW_BYTES + h * references,
                    "page {page_len}, {references} references: {charge} B must hold the page"
                );
            }
            assert_eq!(
                retirement_charge_bytes(page_len, 0),
                page_len * RETIREMENT_ROW_BYTES,
                "no reference owed: the page alone"
            );
        }
        for mib in [5, 10, 16, 64, 1024] {
            assert!(
                reads * RETIREMENT_ROW_BYTES
                    <= RetirementPaging::for_allowance(mib * MIB).page_bytes(),
                "R_ref = {mib} MiB: the full page holds every read in flight"
            );
        }
    }

    /// A store starts at the reference pod's allowance.
    #[test]
    fn retirement_paging_reference_is_the_reference_pods_allowance() {
        assert_eq!(
            RetirementPaging::reference(),
            RetirementPaging::for_allowance(
                crate::config::PodMemoryPlan::reference().retirement_budget_bytes()
            )
        );
    }

    /// The size read's reservation: an owed release charges its object's page
    /// (one past the largest index, at most `P`) widened to a row per read in
    /// flight, plus `h` per indexed reference; a release already scheduled,
    /// its page alone; a foreign key, or an index that could not be counted,
    /// the full page. Only a foreign key skips the release.
    #[test]
    fn retirement_size_reserves_its_objects_charge() {
        let full = RetirementPaging::for_allowance(5 * MIB);
        let (w, row, h) = (
            full.page_bytes(),
            RETIREMENT_ROW_BYTES,
            RETIREMENT_REFERENCE_CHARGE_BYTES,
        );
        let owed = |indexed| RetirementRelease::Owed { indexed };
        let size = |release, largest| RetirementSize::new(release, largest);
        assert_eq!(
            size(owed(3), Some(10)).reservation_bytes(full),
            11 * row + 3 * h,
            "a handful of references: the page, one past the largest index"
        );
        assert_eq!(
            size(owed(40), Some(10)).reservation_bytes(full),
            40 * row + 40 * h,
            "more references than the page: a row per read in flight"
        );
        assert_eq!(
            size(owed(1_000), Some(10)).reservation_bytes(full),
            RELEASE_SIZE_READS_IN_FLIGHT as u64 * row + 1_000 * h
        );
        assert_eq!(
            size(owed(0), Some(0)).reservation_bytes(full),
            row,
            "an empty object: one row"
        );
        assert_eq!(
            size(owed(5), Some(10_000)).reservation_bytes(full),
            w + 5 * h,
            "a large index: the full page"
        );
        assert_eq!(
            size(RetirementRelease::Scheduled, Some(10)).reservation_bytes(full),
            11 * row
        );
        assert_eq!(
            size(owed(3), None).reservation_bytes(full),
            w + 3 * h,
            "an index not counted: the full page"
        );
        assert_eq!(
            size(RetirementRelease::ForeignKey, Some(10)).reservation_bytes(full),
            w
        );
        assert!(size(owed(1), Some(0)).releases());
        assert!(size(RetirementRelease::Scheduled, Some(0)).releases());
        assert!(!size(RetirementRelease::ForeignKey, Some(0)).releases());
    }

    /// Covering found references tops up only past the charge already held
    /// (a row per read in flight and `h` per reference), never waits, never
    /// takes the grant and top-up together past the allowance (the
    /// release-set residual beyond it is outside); widening tops up to the
    /// full page; and the shrink repays the references and holds the page
    /// alone — a top-up remaining for a page widened past the grant.
    #[tokio::test]
    async fn retirement_reservation_covers_references_widens_and_shrinks_to_its_page() {
        let budget = allowance(5 * MIB);
        let full = RetirementPaging::for_allowance(5 * MIB);
        let page = full.for_rows(11);
        let (row, h) = (RETIREMENT_ROW_BYTES, RETIREMENT_REFERENCE_CHARGE_BYTES);
        let reservation = RetirementReservation::reserve(&budget, full, page, 10)
            .await
            .expect("the allowance is open and empty");
        assert_eq!(reservation.bytes(), 11 * row + 10 * h);
        assert_eq!(reservation.page(), page);

        reservation.cover_references(10);
        assert_eq!(
            budget.overcommit_bytes(),
            0,
            "references the size read counted are covered by the grant"
        );
        reservation.cover_references(25);
        assert_eq!(
            (reservation.bytes(), budget.overcommit_bytes()),
            (25 * row + 25 * h, 14 * row + 15 * h),
            "past the page, each found reference takes a row for its read in flight and h"
        );

        assert_eq!(reservation.widen_page(), full, "widened to the full page");
        assert_eq!(reservation.bytes(), full.page_bytes() + 25 * h);
        assert_eq!(
            reservation.widen_page(),
            full,
            "a second widening is a no-op"
        );
        assert_eq!(reservation.bytes(), full.page_bytes() + 25 * h);

        reservation.cover_references(1_000_000);
        assert_eq!(
            reservation.bytes(),
            5 * MIB,
            "the grant and its top-up together never exceed the allowance"
        );

        reservation.shrink_to_page();
        assert_eq!(
            (reservation.bytes(), budget.reserved_bytes()),
            (full.page_bytes(), 11 * row + 10 * h),
            "the shrink holds the widened page alone: the grant, and a top-up for the rest"
        );
        assert_eq!(
            budget.overcommit_bytes(),
            full.page_bytes() - (11 * row + 10 * h)
        );
        drop(reservation);
        assert_eq!((budget.reserved_bytes(), budget.overcommit_bytes()), (0, 0));

        let unwidened = RetirementReservation::reserve(&budget, full, page, 10)
            .await
            .expect("the allowance is empty again");
        unwidened.cover_references(25);
        unwidened.shrink_to_page();
        assert_eq!(
            (
                unwidened.bytes(),
                budget.reserved_bytes(),
                budget.overcommit_bytes()
            ),
            (11 * row, 11 * row, 0),
            "an unwidened page: the top-up repaid, the grant shrunk to the page"
        );
    }

    /// A sorted index of `members` members read by rank, recording for each
    /// page read the members it asks for and the bytes its reservation holds
    /// when it is issued.
    struct RecordingIndex<'r> {
        members: u64,
        reservation: &'r RetirementReservation,
        reads: Vec<(u64, u64)>,
    }

    impl RankPageReader for RecordingIndex<'_> {
        type Error = std::convert::Infallible;

        async fn read_page(
            &mut self,
            start: i64,
            stop: i64,
        ) -> Result<Vec<String>, std::convert::Infallible> {
            let (start, stop) = (
                u64::try_from(start).expect("ranks are non-negative"),
                u64::try_from(stop).expect("ranks are non-negative"),
            );
            self.reads
                .push((stop - start + 1, self.reservation.bytes()));
            Ok((start..=stop.min(self.members.saturating_sub(1)))
                .filter(|&rank| rank < self.members)
                .map(|rank| rank.to_string())
                .collect())
        }

        async fn take_page(
            &mut self,
            _members: Vec<String>,
        ) -> Result<(), std::convert::Infallible> {
            Ok(())
        }
    }

    /// The member pager the walk and the teardown share (`page_by_rank`, over
    /// a reservation's `Pages`) reads its first page at the object's page and
    /// widens before it reads past it: no page is issued asking for more
    /// members than the reservation then holds rows for. An index three full
    /// pages long behind an object sized at five members: the first page asks
    /// for five, and every later one for `P`, each with the full page held.
    #[tokio::test]
    async fn the_member_pager_widens_its_reservation_before_it_reads_past_the_objects_page() {
        let budget = allowance(5 * MIB);
        let full = RetirementPaging::for_allowance(5 * MIB);
        let page = full.for_rows(5);
        let reservation = RetirementReservation::reserve(&budget, full, page, 0)
            .await
            .expect("the allowance is open and empty");
        let mut index = RecordingIndex {
            members: 3 * full.page_len(),
            reservation: &reservation,
            reads: Vec::new(),
        };

        let pages = Pages {
            fixed: page,
            reservation: Some(&reservation),
        };
        page_by_rank(&mut index, pages)
            .await
            .unwrap_or_else(|never| match never {});

        assert_eq!(
            index.reads.first(),
            Some(&(5, 5 * RETIREMENT_ROW_BYTES)),
            "the first page is the object's, on its grant"
        );
        assert!(
            index.reads.len() >= 3,
            "fixture: the index is paged past the object's page ({:?})",
            index.reads
        );
        for &(asked, held) in &index.reads {
            assert!(
                asked * RETIREMENT_ROW_BYTES <= held,
                "a page asking for {asked} members was issued with {held} B held: {:?}",
                index.reads
            );
        }
        assert!(
            index.reads[1..]
                .iter()
                .all(|&(asked, held)| asked == full.page_len() && held == full.page_bytes()),
            "every later page is the full page, read on the full page's charge: {:?}",
            index.reads
        );
    }

    /// An oversized retirement is clamped to the allowance (it runs alone),
    /// and a closed allowance refuses a reservation `Closed` — `close()`
    /// still refuses, though shutdown never closes `R_ref`: it ends its waits
    /// instead (`MemoryBudget::end_waits`, pinned in `memory_budget`'s tests
    /// and on the routes in `retirement_budget_test.rs`).
    #[tokio::test]
    async fn retirement_reservation_is_clamped_to_the_allowance_and_refused_once_closed() {
        let budget = allowance(5 * MIB);
        let full = RetirementPaging::for_allowance(5 * MIB);
        let clamped = RetirementReservation::reserve(&budget, full, full, 50_000)
            .await
            .expect("an oversized request is clamped, not refused");
        assert_eq!(clamped.bytes(), 5 * MIB);
        drop(clamped);
        budget.close();
        let refused = RetirementReservation::reserve(&budget, full, full, 0).await;
        assert!(matches!(refused, Err(CatchUpRefusal::Closed)));
    }

    /// How many retirements of each shape share the allowance at once, at
    /// the default 5 GiB pod's (`R_ref` 10 MiB) and the smallest pod's
    /// (5 MiB): a small layer owing a handful of references runs many at a
    /// time and beside a large one at both; a large inline layer (its page
    /// full, nothing owed) runs four at a time at 10 MiB and two at 5 MiB;
    /// one owing 1,000 references, two and one. A retirement runs alone only
    /// when its own charge leaves no room beside the smallest other one (one
    /// row): above `(R_ref − W − row) ÷ h` references at the full page.
    #[test]
    fn retirements_share_the_allowance_by_their_own_charge() {
        let (row, h) = (RETIREMENT_ROW_BYTES, RETIREMENT_REFERENCE_CHARGE_BYTES);
        for (mib, small_at_once, inline_at_once, thousand_at_once) in
            [(10, 49, 4, 2), (5, 24, 2, 1)]
        {
            let r_ref = mib * MIB;
            let full = RetirementPaging::for_allowance(r_ref);
            let at_once = |charge: u64| r_ref / charge;
            let small = RetirementSize::new(RetirementRelease::Owed { indexed: 5 }, Some(4))
                .reservation_bytes(full);
            let inline = RetirementSize::new(RetirementRelease::Owed { indexed: 0 }, Some(10_000))
                .reservation_bytes(full);
            let thousand =
                RetirementSize::new(RetirementRelease::Owed { indexed: 1_000 }, Some(4_000))
                    .reservation_bytes(full);
            assert_eq!(
                (at_once(small), at_once(inline), at_once(thousand)),
                (small_at_once, inline_at_once, thousand_at_once),
                "R_ref = {mib} MiB"
            );
            assert!(
                small + thousand <= r_ref,
                "R_ref = {mib} MiB: a handful of references runs beside 1,000"
            );
            let alone_above = (r_ref - full.page_bytes() - row) / h;
            let charge = |refs| {
                RetirementSize::new(RetirementRelease::Owed { indexed: refs }, Some(4_000))
                    .reservation_bytes(full)
            };
            assert!(charge(alone_above) + row <= r_ref);
            assert!(charge(alone_above + 1) + row > r_ref);
        }
    }
}
