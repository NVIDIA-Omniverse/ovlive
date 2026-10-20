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

// Orphan sweep: periodic background task that detects and cleans up
// orphaned Valkey keys left behind by crashes or race conditions.
//
// Implements SDD section 5.5 step 6. One pass runs these phases in order:
// 0. Presence index backstop -- `le:{sid}:presence` member sets missing from
//    the session index `le:presence:active_sessions` are re-added to it, so
//    the presence sweeper visits every session with presence entries even
//    after a compound failure left a dead member un-indexed (see
//    `sweep_unindexed_presence`). Housed here because this sweep is always
//    on, leader-gated, and already walks the keyspace with the cluster-aware
//    SCAN; it writes only the index.
// 1. Orphaned objects -- obj:hdr keys with no layer mapping referencing them,
//    and stranded overwrites: a mapped header a CreateLayer fenced whose
//    commit never landed (see `sweep_orphaned_objects`).
// 2. Orphaned layer mappings -- layer keys whose session no longer exists.
// 3. Retirement intents (b.yda; write-ahead since b.yuo Part R) --
//    external-value ref_count decrements a retirement recorded on the retiring
//    object's own le:{oid}:obj:extret carrier before enqueuing them, and could
//    not confirm it enqueued (or never cleared). An older build's carrier (a
//    SET; this build writes HASHes) is cleared or dropped, never paid
//    (b.1jt). Housed here rather than in a
//    task of its own because this sweep is always on, already holds the
//    le:gc:leader lease the work needs, already carries the cluster-aware
//    keyspace SCAN the backstop needs, and runs at a cadence suited to
//    re-driving an obligation a store hiccup left in doubt.
//
// Orphaned path-index entries are already handled opportunistically by
// ListSessions (SDD section 7.13 step 6) and are not re-scanned here.
//
// The sweep uses SCAN (not KEYS) for production safety. Pre-b.kf1 it ran
// once at startup; b.kf1 H-stor-4 added periodic re-runs gated behind the
// shared `le:gc:leader` lock so normally one replica sweeps at a time (two
// can briefly overlap after a lock takeover). The
// SCAN + HGETALL passes are dispatched via the dedicated maintenance pool
// (b.kf1 H-stor-5) so they cannot queue behind data-pool writers.
//
// b.kf1 H-stor-1: `cleanup_orphaned_object` schedules external-value
// reference decrements via `schedule_old_object_ext_gc` BEFORE issuing the
// UNLINKs. Pre-b.kf1 the sweeper UNLINKed per-field hashes but did NOT
// touch `ext_meta`/`ext_data`, so every external value referenced by an
// orphaned object had its `ref_count` permanently inflated and the blob
// could never be GC'd — a silent disk-leak path proportional to crash
// frequency × external-value churn.

use std::sync::Arc;
use std::time::Duration;

use fred::clients::Client;
use fred::interfaces::{
    ClientLike, ClusterInterface, HashesInterface, KeysInterface, SetsInterface,
};
use fred::types::scan::{SScanResult, ScanResult, ScanType, Scanner, ZScanResult};
use fred::types::Key as FredKey;
use futures::future::Either;
use futures::Stream;
use futures::StreamExt;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

use crate::domain::time::now_micros;
use crate::observability::metrics::{SharedMetrics, GC_OLDER_BUILD_EVENT_CARRIER_DROPPED};
use crate::storage::client::{
    store_write_refusal, FredErrOrigin, StoreWriteRefusal, ValkeyClient, POOL_LABEL_DATA,
    POOL_LABEL_MAINTENANCE,
};
use crate::storage::diff_read::{count_object_key_refusal, SnapshotKeyFamily};
use crate::storage::external_values::{try_acquire_gc_leader, ExternalValueStore};
use crate::storage::keys::{KeySchema, CREATE_LAYER_CLAIM_VETOED_TTL_MS};
use crate::storage::layer::{
    pexpire_layer_stream, schedule_old_object_ext_gc, wake_retired_object_stream, LayerStore,
    OverwriteCleanup, StrandedOverwriteCommit, HEADER_FIELD_CREATE_OPERATION_DIGEST,
    HEADER_FIELD_CREATE_OPERATION_ID, HEADER_FIELD_SEQ, HEADER_FIELD_SUPERSEDED,
    HEADER_FIELD_SUPERSEDED_BY,
};
use crate::storage::leader_lease::LeaderIdentity;
use crate::storage::lua::CREATE_LAYER_VETO_SCRIPT;
use crate::storage::retirement::{
    read_retirement_size, RetirementPaging, RetirementReservation, RetirementWalk, Teardown,
    TeardownFailure,
};

/// Default re-run interval for the periodic orphan sweep.
///
/// Long enough that a healthy fleet's normal commit traffic dwarfs the
/// SCAN cost; short enough that a crash-induced orphan is reaped within
/// a few minutes. Operators can tune this if a deployment's crash-rate
/// profile demands faster reaction.
const ORPHAN_SWEEP_INTERVAL: Duration = Duration::from_secs(300);

/// Walk every key matching `pattern` across the whole keyspace: a SCAN on every
/// primary when the client is connected to a cluster, a single-node SCAN
/// otherwise. The stream yields keys; it ends with an `Err` item when the walk
/// is known to be incomplete, and ends cleanly only after a complete walk.
///
/// **Every shard.** A single-node SCAN walks exactly one primary's slot range,
/// so on a multi-shard cluster it silently misses every key on the sibling
/// shards. Every caller enumerates *all* matching keys, so the clustered arm
/// uses `scan_cluster`, which starts one walk per primary (one slot per primary
/// from the routing table) and routes each later page by that walk's slot.
///
/// **Only a terminal page completes a walk.** Both arms use fred's page forms
/// rather than the `*_buffered` ones, because only a page says whether it is the
/// last (`has_more`); the buffered forms yield bare keys, and fred ends their
/// stream with no error when a page's command is dropped unanswered. The page
/// stream goes through [`complete_pages`], which turns a walk that ended before
/// its terminal page into a final `Err` item: one terminal page standalone, one
/// per primary `scan_cluster` started on in cluster mode. Each page's keys are
/// taken and the page dropped before its keys are yielded — the drop is what
/// requests the next page — so at most one page per primary is in flight.
///
/// **The routing guard.** fred pages carry no node identity, so the per-primary
/// count can only come from the routing table. It is read before and after the
/// walk starts; if either read finds no routing table or an empty one, or the
/// two differ, the number of per-primary walks to wait for is unknown, and the
/// stream is a single `ErrorKind::Cluster` error instead (transient: the
/// caller's next pass walks again). An empty table counts as unknown because
/// `scan_cluster` then starts no walk at all, and zero terminal pages out of
/// zero would read as a complete walk of an empty keyspace. The decision is
/// [`cluster_walk_count`].
///
/// `pub(crate)` since b.yda so `storage::ext_reconcile` can reuse it rather than
/// writing a second dispatch; the checkpoint loop's pass 1 uses it too. Sharing
/// it is not a convenience: the b.289 source-shape fence requires every bare
/// `SCAN` to sit inside an `is_clustered` two-arm dispatch, and a keyspace walk
/// that quietly covered one primary's slot range — or stopped short on one —
/// would report a fraction of the blob keyspace as the whole of it, which, for
/// a pass that decides what to decrement, is exactly the under-count that
/// becomes an over-release.
pub(crate) fn scan_keyspace(
    client: &Client,
    pattern: &str,
) -> std::pin::Pin<Box<dyn Stream<Item = Result<FredKey, fred::error::Error>> + Send>> {
    const SCAN_PAGE: u32 = 100;
    let pattern_owned = pattern.to_owned();
    // Read OUTSIDE the dispatch so the b.289 fence's ten-line window from
    // `is_clustered` and `scan_cluster(` to `.scan(` holds.
    let slots_before = client
        .cached_cluster_state()
        .map(|routing| routing.unique_hash_slots());
    let pages: ScanPages<ScanResult> = if client.is_clustered() {
        Box::pin(client.scan_cluster(pattern_owned, Some(SCAN_PAGE), None::<ScanType>))
    } else {
        Box::pin(client.scan(pattern_owned, Some(SCAN_PAGE), None::<ScanType>))
    };
    let slots_after = client
        .cached_cluster_state()
        .map(|routing| routing.unique_hash_slots());
    let terminal_pages = if client.is_clustered() {
        match cluster_walk_count(pattern, slots_before, slots_after) {
            Ok(count) => count,
            Err(e) => return Box::pin(futures::stream::once(futures::future::ready(Err(e)))),
        }
    } else {
        1
    };
    Box::pin(
        complete_pages(pages, terminal_pages, pattern).flat_map(|item| match item {
            Ok(mut page) => {
                let keys = page.take_results().unwrap_or_default();
                // Dropping the page is what asks fred for the next one
                // (`ScanResult`'s `Drop`); a terminal page's drop releases its
                // walk's sender instead, and the stream ends once every
                // primary's walk has released its own.
                drop(page);
                Either::Left(futures::stream::iter(keys.into_iter().map(Ok)))
            }
            Err(e) => Either::Right(futures::stream::once(futures::future::ready(Err(e)))),
        }),
    )
}

/// `scan_keyspace`'s routing guard, as a pure decision: how many terminal pages
/// a cluster-wide walk must see, given the per-primary slots (fred's
/// `unique_hash_slots`, one slot per primary) read `before` and `after` the
/// walk started.
///
/// `Ok(n)` only when both reads found the same, non-empty set — `n` primaries,
/// so `n` per-primary walks. Otherwise the count is unknown and the walk cannot
/// be judged complete, so the answer is an `ErrorKind::Cluster` error naming
/// `walk` (transient: the caller's next pass walks again):
///
/// - either read `None`: no routing table;
/// - either read empty: fred stores an empty table when `CLUSTER SLOTS`
///   returns no slots, `scan_cluster` then starts no walk, and the stream would
///   end at once with zero terminal pages out of zero — a false "complete";
/// - the two differ: the primaries changed while `scan_cluster` read them.
fn cluster_walk_count(
    walk: &str,
    before: Option<Vec<u16>>,
    after: Option<Vec<u16>>,
) -> Result<usize, fred::error::Error> {
    match (before, after) {
        (Some(before), Some(after)) if before == after && !before.is_empty() => Ok(before.len()),
        _ => Err(fred::error::Error::new(
            fred::error::ErrorKind::Cluster,
            format!(
                "SCAN walk `{walk}` abandoned before its first page: the cluster's primaries \
                 changed while the walk started, or the routing table was unknown or empty, so \
                 the number of per-primary walks to wait for is unknown"
            ),
        )),
    }
}

/// A boxed stream of fred SCAN-family pages.
type ScanPages<P> = std::pin::Pin<Box<dyn Stream<Item = Result<P, fred::error::Error>> + Send>>;

/// A page of a fred SCAN-family stream that knows whether it is its walk's
/// last: `SCAN` (keyspace), `SSCAN` and `ZSCAN` (keyed cursors).
///
/// Named for the stream, not `ScanPage`, which `storage::presence` already
/// uses for its explicit-cursor `SSCAN` page.
pub trait ScanStreamPage {
    /// Whether the server returned cursor `0` for this page, so the walk that
    /// produced it has no more pages.
    fn is_terminal(&self) -> bool;
}

impl ScanStreamPage for ScanResult {
    fn is_terminal(&self) -> bool {
        !self.has_more()
    }
}

impl ScanStreamPage for SScanResult {
    fn is_terminal(&self) -> bool {
        !self.has_more()
    }
}

impl ScanStreamPage for ZScanResult {
    fn is_terminal(&self) -> bool {
        !self.has_more()
    }
}

/// The code home of the rule **a SCAN-family walk is complete only on its
/// terminal page** (SDD §11.3.2): every fred `SCAN`, `SSCAN` and `ZSCAN` stream
/// the service walks goes through here.
///
/// **Why the end of a stream proves nothing.** fred ends a SCAN-family stream
/// when every sender of its channel is dropped, and each page command in flight
/// holds one. When the client's router ends — or is aborted — with a page in
/// flight or in its retry buffer, that command is dropped unanswered, and the
/// stream ends with **no error**, exactly as it does after a complete walk. A
/// page requested after the router ended fails visibly; only a page in flight
/// at that moment is lost silently. The member replacer's last-resort abort, its
/// drop guard's abandon, and any router death the supervisor restarts all reach
/// it.
///
/// **What this does.** It passes every item through unchanged, counting the
/// terminal pages ([`ScanStreamPage::is_terminal`]) and noting whether any
/// `Err` passed. When `pages` ends, if no `Err` passed and the count differs
/// from `terminal_pages`, it yields exactly one `Err` — `ErrorKind::Canceled`,
/// naming `walk` and the terminal pages seen against those expected — and then
/// ends. An `Err` that already passed is the walk's failure on its own, so no
/// second one is added. A consumer that stops early (cancellation, a break, an
/// early `?`) simply drops the stream; nothing is checked then.
///
/// `terminal_pages` is 1 for a single-node `SCAN` and for a keyed cursor (an
/// `SSCAN` / `ZSCAN` routes by its key to one node in both modes), and the
/// number of primaries `scan_cluster` started on for a cluster-wide walk (see
/// `scan_keyspace`'s routing guard).
///
/// `Canceled` is transient for `is_transient_fred_error` and is not counted by
/// `observe_error`, so each consumer's existing error arm applies unchanged:
/// the reconciler forfeits the pass, the orphan sweep warns and walks again next
/// tick, and the checkpoint loop marks its tick truncated.
///
/// Explicit-cursor walks (`presence::sscan_page`, `diff_read`'s `SSCAN` step)
/// do not need this: they finish only on cursor `0` and fail visibly when a page
/// is lost.
///
/// `pub` rather than `pub(crate)` so integration tests can drive it over a raw
/// fred page stream, where fred's silent end can be pinned against a real
/// router abort.
pub fn complete_pages<S, P>(
    pages: S,
    terminal_pages: usize,
    walk: &str,
) -> impl Stream<Item = Result<P, fred::error::Error>> + Send
where
    S: Stream<Item = Result<P, fred::error::Error>> + Send,
    P: ScanStreamPage + Send,
{
    struct Walk<S> {
        pages: std::pin::Pin<Box<S>>,
        terminal_pages: usize,
        walk: String,
        terminal_seen: usize,
        error_seen: bool,
        ended: bool,
        /// Test seam: how many more pages to pass before ending the inner
        /// stream as if a page had been dropped unanswered.
        #[cfg(any(test, feature = "test-fault-injection"))]
        pages_left: Option<usize>,
    }

    let state = Walk {
        pages: Box::pin(pages),
        terminal_pages,
        #[cfg(any(test, feature = "test-fault-injection"))]
        pages_left: scan_walk_truncation_for_test(walk),
        walk: walk.to_owned(),
        terminal_seen: 0,
        error_seen: false,
        ended: false,
    };

    futures::stream::unfold(state, |mut st| async move {
        if st.ended {
            return None;
        }
        #[cfg(any(test, feature = "test-fault-injection"))]
        let truncated_for_test = match st.pages_left.as_mut() {
            Some(0) => true,
            Some(left) => {
                *left -= 1;
                false
            }
            None => false,
        };
        #[cfg(not(any(test, feature = "test-fault-injection")))]
        let truncated_for_test = false;

        let item = if truncated_for_test {
            None
        } else {
            st.pages.next().await
        };
        match item {
            Some(Ok(page)) => {
                if page.is_terminal() {
                    st.terminal_seen += 1;
                }
                Some((Ok(page), st))
            }
            Some(Err(e)) => {
                st.error_seen = true;
                Some((Err(e), st))
            }
            None => {
                st.ended = true;
                if st.error_seen || st.terminal_seen == st.terminal_pages {
                    return None;
                }
                let e = fred::error::Error::new(
                    fred::error::ErrorKind::Canceled,
                    format!(
                        "SCAN walk `{}` ended after {} of {} terminal page(s): a page was \
                         dropped unanswered (its client's router ended with the page in \
                         flight), so the walk is incomplete",
                        st.walk, st.terminal_seen, st.terminal_pages
                    ),
                );
                Some((Err(e), st))
            }
        }
    })
}

/// Walks the truncation seam is armed for, keyed by [`complete_pages`]'s
/// `walk` (a SCAN pattern or a set key), each with the number of pages to pass
/// before the walk's page stream is ended short. Keyed by walk because a test's
/// pattern or key carries its unique prefix, so parallel tests never see each
/// other's arming.
#[cfg(any(test, feature = "test-fault-injection"))]
static SCAN_WALK_TRUNCATION: std::sync::LazyLock<
    std::sync::Mutex<std::collections::HashMap<String, usize>>,
> = std::sync::LazyLock::new(|| std::sync::Mutex::new(std::collections::HashMap::new()));

/// Test seam: every [`complete_pages`] walk named `walk` that starts while this
/// is armed ends its page stream after `pages` pages without the terminal one —
/// exactly what fred does when a page's command is dropped unanswered — so the
/// walk ends with the truncation error. Stays armed until the returned guard
/// drops (or [`disarm_scan_walk_truncation_for_test`]), so a failing assertion
/// cannot leave it armed for a later test. Arming a walk again replaces its
/// page count, but the arming is one entry per walk, not one per guard:
/// dropping **any** guard for the walk disarms it, a later re-arming included.
/// So hold one guard per walk at a time; to change the count, drop the old
/// guard before arming again. Compiled out of release builds.
#[cfg(any(test, feature = "test-fault-injection"))]
pub fn arm_scan_walk_truncation_for_test(walk: &str, pages: usize) -> ArmedScanWalkTruncation {
    SCAN_WALK_TRUNCATION
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .insert(walk.to_owned(), pages);
    ArmedScanWalkTruncation {
        walk: walk.to_owned(),
    }
}

/// The scan-walk truncation seam armed for one walk
/// ([`arm_scan_walk_truncation_for_test`]); disarms that walk when dropped —
/// whoever armed it last, so with two live guards for one walk, dropping
/// either disarms it for both. Hold one per walk at a time, and bind it for as
/// long as the walk should be cut short — `let _armed = …`, never `let _ = …`,
/// which drops it at once. Compiled out of release builds.
#[cfg(any(test, feature = "test-fault-injection"))]
#[must_use = "the seam is disarmed when this guard drops; bind it for the walks it should cut short"]
#[derive(Debug)]
pub struct ArmedScanWalkTruncation {
    walk: String,
}

#[cfg(any(test, feature = "test-fault-injection"))]
impl ArmedScanWalkTruncation {
    /// The walk (SCAN pattern or set key) this guard keeps armed.
    pub fn walk(&self) -> &str {
        &self.walk
    }
}

#[cfg(any(test, feature = "test-fault-injection"))]
impl Drop for ArmedScanWalkTruncation {
    fn drop(&mut self) {
        disarm_scan_walk_truncation_for_test(&self.walk);
    }
}

/// Disarms [`arm_scan_walk_truncation_for_test`] for `walk`; walks that start
/// afterwards run to their real end. Dropping the guard that call returned does
/// the same.
#[cfg(any(test, feature = "test-fault-injection"))]
pub fn disarm_scan_walk_truncation_for_test(walk: &str) {
    SCAN_WALK_TRUNCATION
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .remove(walk);
}

#[cfg(any(test, feature = "test-fault-injection"))]
fn scan_walk_truncation_for_test(walk: &str) -> Option<usize> {
    SCAN_WALK_TRUNCATION
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .get(walk)
        .copied()
}

/// Result of a single orphan sweep run.
#[derive(Debug, Default)]
pub struct SweepResult {
    /// Number of obj:hdr keys scanned.
    pub objects_scanned: u64,
    /// Number of orphaned objects cleaned up.
    pub orphaned_objects: u64,
    /// Number of layer mapping keys scanned.
    pub layers_scanned: u64,
    /// Number of orphaned layer mappings cleaned up.
    pub orphaned_layers: u64,
    /// Number of `le:{oid}:obj:extret` carriers the retire-retry phase took up
    /// to PAY (b.yda; b.yuo Part R) — this build's HASH carriers, every
    /// recorded earlier attempt decided void and the object id dead — whether
    /// or not this tick's re-drive then landed. A cancelled carrier, one found
    /// already paid, and an older build's dropped carrier are not counted here.
    pub retire_carriers_claimed: u64,
    /// Number of recorded external-value release obligations the retire-retry
    /// phase re-drove onto `le:ext:{gc}:pending` (b.yda). Counted in
    /// *references*, matching `ext_ref_release_recovered_total`.
    pub retire_releases_recovered: u64,
    /// Number of recorded release obligations the retire-retry phase found
    /// already paid — an earlier attempt of the carrier had landed — and so
    /// cleared without enqueuing anything (b.yuo Part R). Counted in
    /// *references*, matching `ext_ref_release_already_enqueued_total`.
    pub retire_releases_already_enqueued: u64,
    /// Number of `le:{oid}:obj:extret` carriers the retire-retry phase
    /// **cancelled** because the `object_id` was live again when every
    /// recorded earlier attempt had resolved void (b.yda) — counted in
    /// carriers, like
    /// [`Self::retire_carriers_claimed`].
    ///
    /// A cancelled carrier's members are dropped, never enqueued: a live id has
    /// inherited the standing `+1` they represent, so paying them would be an
    /// unmatched decrement. The references leak in the worst case and the
    /// reference-count reconciler drains them; see
    /// `scripts/ext_retire_settle.lua`.
    pub retire_carriers_cancelled: u64,
    /// Number of recorded release obligations the retire-retry phase DROPPED
    /// from an older build's SET carrier none of whose recorded attempts
    /// landed (b.1jt): deleted with the carrier, never enqueued — this build
    /// never pays a SET carrier, because most older builds record no attempt
    /// token and no token is not proof that nothing ran. Counted in
    /// *references*, matching
    /// `live_edit_gc_older_build_total{event="carrier_dropped"}`. Leak
    /// direction: the reference-count reconciler repairs an obligation that
    /// was in fact never paid (unless it is turned off, when the leak is
    /// permanent).
    pub retire_releases_dropped: u64,
    /// Number of distinct presence member sets (`le:{sid}:presence`) the
    /// index-backstop phase's walk found.
    pub presence_sets_scanned: u64,
    /// Number of sessions the index-backstop phase added back to the session
    /// index (`le:presence:active_sessions`): sessions whose member set the walk
    /// saw and that had no index entry when checked. Counted from the `SADD`
    /// reply, so a session a joiner re-indexed between the phase's check and
    /// its `SADD` is not counted.
    ///
    /// Non-zero is not by itself evidence of a compound failure. Ordinary
    /// teardown is counted too: a session whose last member left after the walk
    /// saw its set (the check runs only after the whole walk) had its entry
    /// removed by that teardown and is re-added here with no members; the
    /// presence sweeper removes the entry again on its next pass.
    pub presence_sessions_reindexed: u64,
}

/// How many session ids the index-backstop phase checks against the session
/// index per `SMISMEMBER`, and so the most it re-adds per `SADD`: one round
/// trip per chunk rather than one per session, with each command's argument
/// list bounded.
const PRESENCE_REINDEX_CHUNK: usize = 100;

/// How many retire-retry work-list entries one sweep tick claims.
///
/// Bounds the phase's per-tick work the way [`UNLINK_CHUNK_SIZE`](crate::storage::layer::UNLINK_CHUNK_SIZE) bounds the
/// teardown's: the work list is durable and every member keeps its first
/// observation score, so anything above the batch simply waits for the next
/// tick. Sized generously relative to the shape that produces entries — a
/// retirement whose `{gc}`-slot enqueue failed — because that shape arrives in
/// bursts (one store hiccup, every concurrent delete) and a small batch would
/// stretch a burst across many ticks.
const RETIRE_RETRY_BATCH: usize = 256;

/// `X`, the retirement-intent in-progress age (b.yuo Part R): the retire-retry
/// phase leaves a carrier alone while its newest attempt token is younger
/// than this, and keeps its work-list entry for the next tick.
///
/// It decides how soon a sweep may take a carrier from a sender that may still
/// be live. A retirement writes its carrier before its enqueue and clears it
/// after, so every live retirement has a young carrier; leaving it alone spares
/// the drain a `{gc}` resolve that would void the retirement's attempt
/// mid-flight. A drain that does overtake a retirement slower than X never
/// causes an over-release: the `{gc}` resolve decides each attempt once (the
/// retirement's own enqueue then answers `void`, or a second send under a token
/// the drain found landed answers `already`, because the drain never removes a
/// record it did not send — F1), and the retirement's sends are bounded by the
/// send window Y (`external_values::RETIRE_INTENT_SEND_WINDOW`, F3). Its only
/// cost is how often a still-live retirement is cancelled (a leak, the
/// reconciler's to drain). Tests override it per client
/// (`ValkeyClient::set_retire_intent_in_progress_age_for_test`, `test-support`
/// only).
pub const RETIRE_INTENT_IN_PROGRESS_AGE: Duration = Duration::from_secs(60);

/// The orphan sweep grace: `max(60 s, 2 × RPC deadline)`.
///
/// How long after an object's creation (its header's `created_at`) the sweep
/// leaves it to the `CreateLayer` that materialized it: an unmapped, unfenced
/// object younger than this is not vetoed or retired, a stranded overwrite
/// whose successor is younger than this is not completed, and a replaced
/// object whose successor is younger than this is not retired (its
/// overwrite's own cleanup may still be running). The same
/// grace bounds that call's in-process commit finisher, so by the time the
/// sweep acts the call, its finisher and its last commit run are over (a
/// call lives at most one RPC deadline; the second deadline covers a commit
/// run already sent and still queued). Safety does not rest on it — the veto,
/// a `vetoed` commit claim (`KeySchema::create_layer_claim`), does — it keeps
/// the veto from failing healthy creates, and the sweep from racing a
/// finisher.
///
/// **Derived, never configured** — the `ext_reconcile_grace` pattern. A header
/// without `created_at` counts as older than the grace. `LayerStore` carries
/// the configured value (`LayerStore::with_orphan_sweep_grace`), and tests
/// override it per client (`ValkeyClient::set_orphan_sweep_grace_for_test`,
/// `test-support` only).
pub fn orphan_sweep_grace(rpc_deadline_ms: u64) -> Duration {
    Duration::from_secs(60).max(Duration::from_millis(rpc_deadline_ms.saturating_mul(2)))
}

/// Spawn the orphan sweep as a periodic background task. Returns the join
/// handle.
///
/// The task runs once immediately on start, then re-runs every
/// [`ORPHAN_SWEEP_INTERVAL`]. Each tick is gated behind the
/// `le:gc:leader` lock (b.kf1 H-stor-4): only the replica that wins the
/// lock runs the sweep; non-leaders skip the tick. This eliminates the
/// pre-b.kf1 N-replica cluster amplification where every pod ran a full
/// SCAN + HGETALL pass on every interval.
///
/// `leader` is the process-wide [`LeaderIdentity`] — the same one the GC
/// task, the reconciler and the external-value backup repair hold on
/// `le:gc:leader`. Its marked GC lock value
/// ([`LeaderIdentity::gc_lock_value`]) is what the lock stores; its
/// `service_label` alone labels this task's gauges.
///
/// `layer_store` is the replica's layer store: the sweep completes a stranded
/// `CreateLayer` overwrite through its commit dispatch, runs the overwrite
/// tail through it ([`LayerStore::run_overwrite_tail`], awaited, one object
/// at a time), and reads its grace ([`LayerStore::orphan_sweep_grace`]).
pub fn spawn_orphan_sweep(
    client: ValkeyClient,
    keys: KeySchema,
    ext_store: Arc<dyn ExternalValueStore>,
    layer_store: Arc<LayerStore>,
    leader: LeaderIdentity,
    cancel: CancellationToken,
    metrics: Option<SharedMetrics>,
) -> JoinHandle<()> {
    tokio::spawn(async move {
        tracing::info!(
            interval_secs = ORPHAN_SWEEP_INTERVAL.as_secs(),
            replica_id = %leader.service_label,
            lock_value = %leader.gc_lock_value(),
            "orphan sweep background task started"
        );
        let replica_id = leader.service_label.as_str();

        loop {
            // Leader gating (b.kf1 H-stor-4). Reuses the same
            // `le:gc:leader` lock that the GC sweep contends for so that
            // normally a single replica handles both maintenance loops in
            // lock-step (two can briefly overlap after a lock takeover).
            // While this replica's GC loop runs and makes progress, the GC
            // task's once-a-second refresh ticker, which runs independent of
            // every pass, keeps it from lapsing while this sweep runs, however
            // long (b.1jt; `external_values::spawn_gc_task`; the GC loop marks
            // progress at least once a second between its passes and within a
            // few store round trips inside one, even while this sweep runs). If
            // that loop has died, or stalled for
            // `external_values::GC_PROGRESS_STALL_AFTER` (20 s), the ticker has
            // stopped renewing and a peer takes the key over about 20 s after
            // its last refresh, possibly mid-sweep — the behaviour before the
            // ticker existed. That is safe, because the sweep is safe under
            // two leaders.
            match try_acquire_gc_leader(&client, &keys, &leader, metrics.as_ref()).await {
                Ok(true) => {
                    run_one_sweep(
                        &client,
                        &keys,
                        &ext_store,
                        &layer_store,
                        &cancel,
                        metrics.as_ref(),
                        replica_id,
                    )
                    .await;
                }
                Ok(false) => {
                    // b.yda: publish the retire-retry backlog as zero from every
                    // non-leader. An OTel `Gauge` never decays, so a replica that
                    // held the lock, saw a backlog, and then lost it would keep
                    // exporting that backlog forever and pin the fleet-wide
                    // `max()`. Same contract as the checkpoint backlog gauges. A
                    // replica whose leader probe ERRORED publishes nothing (the
                    // `Err` arm below) — unknown is not zero.
                    if let Some(m) = metrics.as_ref() {
                        m.record_ext_retire_retry_backlog(0, replica_id);
                    }
                    tracing::debug!(
                        replica_id = %leader.service_label,
                        lock_value = %leader.gc_lock_value(),
                        "orphan sweep skipped: another replica holds the leader lock"
                    );
                }
                Err(e) => {
                    if let Some(m) = metrics.as_ref() {
                        m.inc_gc_sweep_batch_error("leader_lock_error");
                    }
                    tracing::warn!(
                        replica_id = %leader.service_label,
                        lock_value = %leader.gc_lock_value(),
                        error = %e,
                        "orphan sweep leader lock acquisition failed"
                    );
                }
            }

            tokio::select! {
                () = tokio::time::sleep(ORPHAN_SWEEP_INTERVAL) => {}
                () = cancel.cancelled() => {
                    tracing::info!("orphan sweep task shutting down");
                    return;
                }
            }
        }
    })
}

/// Run a single sweep tick. Logs the result and bumps the
/// `gc_sweep_batch_errors_total` counter on failure but does not propagate
/// — the next interval re-tries.
#[allow(clippy::too_many_arguments)]
async fn run_one_sweep(
    client: &ValkeyClient,
    keys: &KeySchema,
    ext_store: &Arc<dyn ExternalValueStore>,
    layer_store: &LayerStore,
    cancel: &CancellationToken,
    metrics: Option<&SharedMetrics>,
    // b.yda: the `service` label on the retire-retry backlog gauge. Threaded in
    // rather than read from config here for the same reason `metrics` is — this
    // function owns no handles of its own.
    replica_id: &str,
) {
    match run_sweep(
        client,
        keys,
        ext_store,
        layer_store,
        cancel,
        metrics,
        replica_id,
    )
    .await
    {
        Ok(result) => {
            if result.orphaned_objects > 0
                || result.orphaned_layers > 0
                || result.retire_carriers_claimed > 0
                || result.retire_releases_already_enqueued > 0
                || result.retire_carriers_cancelled > 0
                || result.retire_releases_dropped > 0
                || result.presence_sessions_reindexed > 0
            {
                tracing::info!(
                    objects_scanned = result.objects_scanned,
                    orphaned_objects = result.orphaned_objects,
                    layers_scanned = result.layers_scanned,
                    orphaned_layers = result.orphaned_layers,
                    retire_carriers_claimed = result.retire_carriers_claimed,
                    retire_releases_recovered = result.retire_releases_recovered,
                    retire_releases_already_enqueued = result.retire_releases_already_enqueued,
                    retire_carriers_cancelled = result.retire_carriers_cancelled,
                    retire_releases_dropped = result.retire_releases_dropped,
                    presence_sets_scanned = result.presence_sets_scanned,
                    presence_sessions_reindexed = result.presence_sessions_reindexed,
                    "orphan sweep complete"
                );
            } else {
                tracing::info!(
                    objects_scanned = result.objects_scanned,
                    layers_scanned = result.layers_scanned,
                    presence_sets_scanned = result.presence_sets_scanned,
                    "orphan sweep complete, no orphans found"
                );
            }
        }
        Err(e) => {
            if let Some(m) = metrics {
                m.inc_gc_sweep_batch_error(e.kind_label());
            }
            tracing::warn!(error = %e, "orphan sweep failed (non-fatal)");
        }
    }
}

/// Execute the full orphan sweep: re-index un-indexed presence member sets,
/// then scan for orphaned objects, then orphaned layer mappings, then re-drive
/// deferred retirement releases (b.yda).
async fn run_sweep(
    client: &ValkeyClient,
    keys: &KeySchema,
    ext_store: &Arc<dyn ExternalValueStore>,
    layer_store: &LayerStore,
    cancel: &CancellationToken,
    // b.qpf: threaded down to `cleanup_orphaned_object` so the retirement
    // releases this sweep schedules tick `ext_ref_released_total`.
    metrics: Option<&SharedMetrics>,
    replica_id: &str,
) -> Result<SweepResult, SweepError> {
    let mut result = SweepResult::default();

    // The index backstop runs first because the object and layer phases end the
    // pass on a single failed per-key read. Placed after them, one key that
    // keeps failing (a corrupt header, say) would stop the backstop from ever
    // running, and with it the bound on how long an orphaned member set stays
    // un-indexed: re-added within `ORPHAN_SWEEP_INTERVAL` (slept after each
    // pass) plus one full orphan-sweep pass, then swept on the next
    // presence-sweep pass. See `sweep_unindexed_presence` for what pushes it a
    // tick.
    //
    // For the same reason in reverse, the backstop's own error does NOT end the
    // pass: it is counted and logged here, and the phases after it still run.
    // Those phases neither read nor write the session index, so a backstop
    // error is no reason to skip them, and skipping them would cost a tick of
    // orphan reclamation and leave the retire-retry backlog gauge unpublished.
    if let Err(e) = sweep_unindexed_presence(client, keys, cancel, &mut result, metrics).await {
        if let Some(m) = metrics {
            m.inc_gc_sweep_batch_error(e.kind_label());
        }
        tracing::warn!(
            error = %e,
            "presence index backstop failed (non-fatal); continuing the orphan sweep, the next \
             pass retries it"
        );
    }

    if cancel.is_cancelled() {
        return Ok(result);
    }

    sweep_orphaned_objects(
        client,
        keys,
        ext_store,
        layer_store,
        cancel,
        &mut result,
        metrics,
    )
    .await?;

    if cancel.is_cancelled() {
        return Ok(result);
    }

    sweep_orphaned_layers(client, keys, cancel, &mut result).await?;

    if cancel.is_cancelled() {
        return Ok(result);
    }

    // b.yda phase 3. This sweep is the natural home for the retire-retry drain
    // and no new background task was added for it: it is always on, it already
    // holds the `le:gc:leader` lease this work needs, it already carries the
    // cluster-aware keyspace SCAN the backstop needs, and its 300 s cadence is
    // the right order of magnitude for re-driving an obligation a store hiccup
    // deferred seconds ago.
    sweep_retire_retry(client, keys, cancel, &mut result, metrics, replica_id).await?;

    Ok(result)
}

/// Add back to the session index (`le:presence:active_sessions`) every session
/// whose member set (`le:{sid}:presence`) exists but is not indexed.
///
/// # Why a keyspace walk is needed
///
/// The presence sweeper finds sessions only through the index, so a member set
/// missing from it is never swept, and a dead connection in it is never
/// removed. Two compound failures leave one so, and nothing else re-adds it:
///
/// 1. a joining replica dies between its register and its index re-add, after
///    a concurrent index removal took the entry out;
/// 2. a step of `storage::presence::remove_session_from_index` (`SCARD`, or
///    the `SADD` back) fails while a concurrent joiner's replica dies before
///    its lease keeper's first renewal re-add.
///
/// A live connection's keeper re-adds its session within one or two refresh
/// intervals; this phase covers the member sets no live keeper will re-add.
/// The sweep sleeps [`ORPHAN_SWEEP_INTERVAL`] after each pass, so a set
/// orphaned this way is back in the index within that interval plus one full
/// orphan-sweep pass (this walk and the three other phases' walks), and is
/// swept by the next presence-sweep pass. A tick on which this replica is not
/// the leader, a failed backstop, or a `SCAN` error during the walk that
/// drops the set's key pushes the re-add a tick later.
///
/// # Shape
///
/// Walk, collect, then act, like the retire-retry phase. The walk is
/// `SCAN MATCH le:{*}:presence`, which matches member sets only: a
/// connection's info, lease and teardown-marker keys all carry a further
/// `:{connection_id}…` after `:presence`. Each key is also checked against
/// [`KeySchema::presence_members`], so a key the glob matches but that builder
/// would not produce is skipped rather than misread. A `SCAN` error is logged
/// at WARN and skipped, as in the other phases: a partial walk only delays a
/// re-add to the next pass.
///
/// The act checks the collected ids against the index in chunks of
/// [`PRESENCE_REINDEX_CHUNK`] (`SMISMEMBER`) and `SADD`s only the missing ones,
/// so an indexed session is not written. A failed check or `SADD` ends this
/// phase with an error, but not the pass: `run_sweep` counts it in
/// `gc_sweep_batch_errors_total` and goes on to the object phase, and the next
/// pass retries the backstop.
///
/// # Races
///
/// A member set the walk saw can be emptied before its `SADD`; ordinary
/// teardown of a session's last member does this routinely, since the check
/// runs only after the whole walk. That `SADD` then indexes a session with no
/// members (and counts it as re-indexed), and the presence sweeper removes
/// the entry through the remove-then-recheck helper on its next pass. The
/// phase writes only index entries: never a member, info hash or lease, so it
/// cannot create membership.
async fn sweep_unindexed_presence(
    client: &ValkeyClient,
    keys: &KeySchema,
    cancel: &CancellationToken,
    result: &mut SweepResult,
    metrics: Option<&SharedMetrics>,
) -> Result<(), SweepError> {
    let prefix = keys.prefix();
    // Mirrors `KeySchema::presence_members`; the equality check below catches
    // a key the glob matches that the builder would not produce.
    let pattern = format!("{prefix}{{*}}:presence");

    let (maint_pool, _maint_token) = client.acquire_maintenance();

    // A `SCAN` can return a key more than once; the set keeps one entry per
    // session so a duplicate is neither checked nor counted twice.
    let mut session_ids: std::collections::HashSet<String> = std::collections::HashSet::new();
    {
        let scan_client = maint_pool.next();
        let mut stream = scan_keyspace(scan_client, &pattern);
        while let Some(scan_result) = tokio::select! {
            item = stream.next() => item,
            () = cancel.cancelled() => None,
        } {
            let fred_key = match scan_result {
                Ok(k) => k,
                Err(e) => {
                    // `observe_error` never fires here: fred sends every SCAN
                    // page through `default_send_command`, which applies no
                    // command timeout, `observe_error` counts only timeouts,
                    // and `scan_keyspace`'s own errors are `Canceled` (a walk
                    // that ended before its terminal page) or `Cluster` (the
                    // routing guard). So a SCAN error here goes uncounted, and
                    // a hung SCAN is ended only by cancellation or fred's
                    // unresponsive-connection check (about 2.5 s while the
                    // shard answers other connections, 45 s otherwise).
                    // `continue` then ends the walk when the error was the
                    // truncation item — it is always the stream's last — and
                    // the later phases run; the next tick walks again.
                    FredErrOrigin::metrics(
                        metrics,
                        POOL_LABEL_MAINTENANCE,
                        client.store_oom_signal(),
                    )
                    .observe_error(&e);
                    tracing::warn!(error = %e, "SCAN error during presence index backstop");
                    continue;
                }
            };
            let Some(key) = fred_key.as_str() else {
                continue;
            };
            let Some(session_id) = extract_hash_tag_id(key, prefix, ":presence")
                .filter(|sid| keys.presence_members(sid) == key)
            else {
                tracing::warn!(key = %key, "could not parse session_id from presence member-set key");
                continue;
            };
            session_ids.insert(session_id);
        }
    }
    result.presence_sets_scanned = session_ids.len() as u64;

    if cancel.is_cancelled() {
        return Ok(());
    }

    let index_key = keys.presence_active_sessions();
    let session_ids: Vec<String> = session_ids.into_iter().collect();
    for chunk in session_ids.chunks(PRESENCE_REINDEX_CHUNK) {
        if cancel.is_cancelled() {
            break;
        }
        let indexed: Vec<bool> = maint_pool
            .smismember::<Vec<bool>, _, _>(&index_key, chunk.to_vec())
            .await
            .map_err(|e| {
                SweepError::from_fred(
                    FredErrOrigin::metrics(
                        metrics,
                        POOL_LABEL_MAINTENANCE,
                        client.store_oom_signal(),
                    ),
                    "SMISMEMBER session index",
                    e,
                )
            })?;
        if indexed.len() != chunk.len() {
            return Err(SweepError::from_fred(
                FredErrOrigin::metrics(metrics, POOL_LABEL_MAINTENANCE, client.store_oom_signal()),
                "SMISMEMBER session index",
                fred::error::Error::new(
                    fred::error::ErrorKind::Protocol,
                    format!(
                        "SMISMEMBER answered {} flags for {} session ids",
                        indexed.len(),
                        chunk.len()
                    ),
                ),
            ));
        }
        let missing: Vec<String> = chunk
            .iter()
            .zip(&indexed)
            .filter(|&(_, &is_indexed)| !is_indexed)
            .map(|(sid, _)| sid.clone())
            .collect();
        if missing.is_empty() {
            continue;
        }

        let added: u64 = maint_pool
            .sadd::<u64, _, _>(&index_key, missing.clone())
            .await
            .map_err(|e| {
                SweepError::from_fred(
                    FredErrOrigin::metrics(
                        metrics,
                        POOL_LABEL_MAINTENANCE,
                        client.store_oom_signal(),
                    ),
                    "SADD session index",
                    e,
                )
            })?;
        result.presence_sessions_reindexed += added;
        // INFO, not WARN: these are not all compound failures. Ordinary
        // teardown lands here too: the ids are checked only after the whole
        // walk (about 1.5 s at the documented fleet), so a session whose last
        // member left after the walk saw its set has had its index entry
        // removed by that teardown, and is re-added here with no members — the
        // presence sweeper removes the entry again on its next pass. An index
        // removal racing a join is a second benign case: the member set is
        // briefly un-indexed until the removal's recheck or the join's re-add
        // restores the entry. Restoring any of these is harmless.
        for session_id in &missing {
            tracing::info!(
                session_id = %session_id,
                "presence index backstop: re-indexed a session whose member set the walk saw \
                 and the session index lacked; if its last member left since the walk, the \
                 presence sweeper removes the entry again"
            );
        }
    }

    Ok(())
}

/// Settle external-value release obligations that a retirement recorded on
/// `le:{oid}:obj:extret` and could not confirm it enqueued onto
/// `le:ext:{gc}:pending` (b.yda; the write-ahead retirement intent since b.yuo
/// Part R).
///
/// # Two discovery paths, and why both
///
/// **Phase A — the work list.** `le:ext:{gc}:retire-retry` names the objects a
/// retirement registered after both of its enqueue attempts ended in doubt, so
/// the common case costs one `EVALSHA` and no keyspace walk. Claimed with a
/// **zero grace**, unlike the checkpoint reconciliation carriers this shares a
/// script with: there is nothing to age out here — the in-progress age below is
/// checked per carrier, on the carrier's own tokens.
///
/// **Phase B — the keyspace backstop.** `SCAN le:{*}:obj:extret` finds carriers
/// with no work-list entry: a retirement whose begin ran but whose reply was
/// lost, one that crashed between its begin and its clear, one whose clear
/// failed, and one whose work-list registration failed on the `{gc}` slot that
/// had just failed its enqueue. Without phase B such a carrier would sit
/// forever holding an obligation nothing looks for. It also finds every
/// retirement in flight, which is what the in-progress age is for.
///
/// # Per carrier: decide every recorded attempt before sending another
///
/// See [`drain_retire_carrier`]. A carrier's members are enqueued again only
/// after the `{gc}` slot has decided, once and for good, that every attempt
/// recorded on it is void, and only for this build's HASH carrier, which
/// records every attempt; a carrier with any attempt that landed is only
/// cleared; an older build's SET carrier with none landed is dropped, never
/// paid (b.1jt); and a carrier whose recorded attempts are all
/// void is cancelled, not paid,
/// when the `object_id` is live again — a live id has inherited the standing
/// `+1` per hash the members stand for, so paying them would delete `ext_meta`
/// and `ext_data` under a live reference. Cancelling leaks in the worst case,
/// and a leak is what `storage::ext_reconcile` is for. The full invariant and
/// the `obj:ext`-in-the-gate argument are in `scripts/ext_retire_settle.lua`.
async fn sweep_retire_retry(
    client: &ValkeyClient,
    keys: &KeySchema,
    cancel: &CancellationToken,
    result: &mut SweepResult,
    metrics: Option<&SharedMetrics>,
    replica_id: &str,
) -> Result<(), SweepError> {
    let retry_key = keys.ext_retire_retry();

    // Phase A. `candidates` is empty — this call registers nothing, it only
    // claims. Zero grace, per the doc-comment.
    let mut object_ids: Vec<String> = crate::storage::external_values::recon_claim_expired(
        client,
        &retry_key,
        Duration::ZERO,
        RETIRE_RETRY_BATCH,
        &[],
    )
    .await
    // `recon_claim_expired` counts its own command timeouts.
    .map_err(|e| {
        SweepError::from_fred(
            FredErrOrigin::metrics(None, POOL_LABEL_MAINTENANCE, client.store_oom_signal()),
            "claim ext retire-retry work list",
            e,
        )
    })?;

    // Phase B. Deduped against phase A's answers, so a carrier named by both
    // paths is drained once — a second visit would answer `in_progress` off the
    // first visit's fresh token, but the work-list retire would then run twice
    // for nothing.
    let mut seen: std::collections::HashSet<String> = object_ids.iter().cloned().collect();
    {
        let prefix = keys.prefix();
        // Ends at `extret`, so it matches the carriers and not their
        // `obj:extret:attempts` token HASHes.
        let pattern = format!("{prefix}{{*}}:obj:extret");
        let (maint_pool, _maint_token) = client.acquire_maintenance();
        let scan_client = maint_pool.next();
        let mut stream = scan_keyspace(scan_client, &pattern);
        while let Some(scan_result) = tokio::select! {
            item = stream.next() => item,
            () = cancel.cancelled() => None,
        } {
            let fred_key = match scan_result {
                Ok(k) => k,
                Err(e) => {
                    FredErrOrigin::metrics(
                        metrics,
                        POOL_LABEL_MAINTENANCE,
                        client.store_oom_signal(),
                    )
                    .observe_error(&e);
                    // `continue`, matching the sibling phases: a partial
                    // backstop walk costs latency on the carriers it missed
                    // (they are durable and the next tick re-scans), never
                    // correctness. That includes a walk that ended before its
                    // terminal page: `scan_keyspace` reports it as this arm's
                    // last item, so the WARN below is its only trace and the
                    // walk then ends. This is deliberately NOT the fail-closed
                    // posture `storage::ext_reconcile` takes — there a partial
                    // walk would under-count references and drive an
                    // over-release, while here it can only defer work.
                    tracing::warn!(error = %e, "SCAN error during retire-retry carrier sweep");
                    continue;
                }
            };
            let Some(key) = fred_key.as_str() else {
                continue;
            };
            let Some(object_id) = extract_hash_tag_id(key, prefix, ":obj:extret") else {
                tracing::warn!(key = %key, "could not parse object_id from obj:extret key");
                continue;
            };
            if seen.insert(object_id.clone()) {
                object_ids.push(object_id);
            }
        }
    }

    for object_id in &object_ids {
        if cancel.is_cancelled() {
            break;
        }
        drain_retire_carrier(client, keys, &retry_key, object_id, result, metrics).await;
    }

    // Publish the backlog AFTER the drain, so the gauge reflects what is left
    // owed rather than what was owed on arrival.
    match crate::storage::external_values::recon_backlog(client, &retry_key).await {
        Ok(backlog) => {
            if let Some(m) = metrics {
                m.record_ext_retire_retry_backlog(backlog, replica_id);
            }
        }
        // Publish nothing on a failed read — unknown is not zero, and a stale
        // gauge is less misleading than a fabricated zero that would read as
        // "the backlog cleared".
        Err(e) => tracing::warn!(error = %e, "could not read the retire-retry backlog"),
    }

    Ok(())
}

/// **Test-only.** Run the retire-retry phase ([`sweep_retire_retry`]) once,
/// outside the sweep's leader-gated background loop, and return what it did
/// (b.yuo Part R). For test crates that cannot reach the private phase — the
/// Valkey proptests drive the retirement intent's drain through this. Compiled
/// only for tests and the `test-support` feature.
#[cfg(any(test, feature = "test-support"))]
pub async fn sweep_retire_retry_for_test(
    client: &ValkeyClient,
    keys: &KeySchema,
    metrics: Option<&SharedMetrics>,
) -> Result<SweepResult, SweepError> {
    let mut result = SweepResult::default();
    sweep_retire_retry(
        client,
        keys,
        &CancellationToken::new(),
        &mut result,
        metrics,
        "test",
    )
    .await?;
    Ok(result)
}

/// Retire one object's work-list entry, logging (not propagating) a failure:
/// a stale entry costs one wasted visit next tick, which then finds the carrier
/// empty and retires it then.
async fn retire_work_list_entry(
    client: &ValkeyClient,
    retry_key: &str,
    object_id: &str,
    context: &'static str,
) {
    if let Err(e) = crate::storage::external_values::recon_retire(
        client,
        retry_key,
        std::slice::from_ref(&object_id.to_owned()),
    )
    .await
    {
        tracing::warn!(
            object_id = %object_id,
            error = %e,
            context,
            "could not retire a retire-retry work-list entry; the next sweep will clear it"
        );
    }
}

/// Settle one object's retirement intent (b.yda; b.yuo Part R). Never returns
/// an error: every failure shape here leaves the carrier — members and tokens
/// — exactly where the next tick picks it up, and none of them should abort the
/// sweep's remaining objects.
///
/// # Steps
///
/// 1. **Re-token** (`scripts/ext_retire_retry_claim.lua`) with a fresh token
///    and the in-progress age X ([`RETIRE_INTENT_IN_PROGRESS_AGE`]): `empty` →
///    retire the work-list entry; `in_progress` → keep it for the next tick;
///    `claimed` → the members, the tokens recorded before ours (`prior`) and
///    the carrier's type — this build's HASH, or an older build's SET.
///    The re-token only ADDS our token, so an error — even one whose script
///    ran — leaves every member on the carrier, and our token, never sent, is
///    resolved void by the next tick.
/// 2. **Resolve** every prior token on `{gc}`
///    (`scripts/ext_enqueue_attempt_resolve.lua`), in one call: an attempt
///    either landed before this, or is recorded `void` and never lands.
///    - Any `landed` → the obligation was paid: clear the carrier under our
///      token, count `already_enqueued`, retire the entry. The landed records
///      are NOT removed: this drain did not send under those tokens, so it
///      cannot know their last send has arrived (a slow retirement's second
///      attempt, or a copy fred writes late, must still find the record and
///      answer `already`). The GC loop's attempt audit removes them after V
///      (F2). `ext_gc_scheduled` is kept even when the id is live (policy 12).
///    - An older build's SET carrier with no `landed` prior token (all
///      `void`, or none recorded) → **drop** it (b.1jt): the clear script
///      under our own, never-sent token deletes the carrier and its tokens,
///      nothing is enqueued and `ext_gc_scheduled` is kept; WARN, and
///      `live_edit_gc_older_build_total{event="carrier_dropped"}` per
///      reference. This build never pays a SET carrier: most older builds
///      record no attempt token, and no token is not proof that nothing ran
///      (rule E4), so paying could be a second decrement. If none ran, the
///      reference-count reconciler repairs the leak (unless it is turned off,
///      `LIVE_EDIT_EXT_RECONCILE_INTERVAL_SECS=0`, when the leak is
///      permanent).
///    - This build's HASH carrier, all `void` → settle.
/// 3. **Settle** (`scripts/ext_retire_settle.lua`): the token set must be
///    unchanged and the carrier present (else `changed`, next tick); then a
///    live id → `cancelled` (carrier and tokens deleted, the flag cleared,
///    counted) and a dead one → `pay`: one attempt-mode enqueue under our
///    token, decided no later than the send window Y after the re-token was
///    issued (F3). It landed → clear, then — this drain's single clean send
///    under our token — forget our record on `cleared` (F1), count
///    `recovered`, retire the entry; it was voided → another drain owns the
///    carrier; it was not sent (past Y) → stop, our never-sent token stays on
///    the carrier and the next tick resolves it void; it failed → our token
///    stays on the carrier, and the next tick resolves it.
///
/// There is no put-back: nothing here removes a member before an attempt is
/// known to have landed, so nothing has to be written back.
///
/// The `{object_id}` steps and the resolve run on one maintenance connection;
/// the enqueue runs on the data pool, like every enqueue.
async fn drain_retire_carrier(
    client: &ValkeyClient,
    keys: &KeySchema,
    retry_key: &str,
    object_id: &str,
    result: &mut SweepResult,
    metrics: Option<&SharedMetrics>,
) {
    use crate::storage::external_values::{
        enqueue_ext_release_attempt, forget_landed_attempts, mint_retire_attempt_token,
        resolve_enqueue_attempts, retire_clear, retire_retoken, retire_settle, AttemptVerdict,
        ReleaseAttemptOutcome, RetireClear, RetireRetoken, RetireSettle,
    };

    let (maint_pool, _maint_token) = client.acquire_maintenance();
    let conn = maint_pool.next();
    let origin = FredErrOrigin::metrics(metrics, POOL_LABEL_MAINTENANCE, client.store_oom_signal());

    // 1. Re-token. F3: the pay below is decided no later than Y after the
    // re-token that records our token is issued — taken before it, so a stall
    // anywhere from here to the send eats into the window.
    let send_by = std::time::Instant::now() + client.retire_intent_send_window();
    let token = mint_retire_attempt_token();
    let (members, prior, older_build_carrier) = match retire_retoken(
        conn,
        keys,
        object_id,
        &token,
        client.retire_intent_in_progress_age(),
    )
    .await
    {
        Ok(RetireRetoken::Empty) => {
            // Nothing is owed under this id: the carrier was cleared or
            // cancelled, or a work-list entry outlived it. Retire the entry
            // rather than re-visiting it every tick forever. The re-token
            // wrote nothing: a token HASH beside no carrier (only an older
            // build leaves one) stays in place. Nothing in this build removes
            // it; b.9ud sub-task 3 plans the only safe remover (the restore's
            // residue wipe, extended to void the tokens and delete the HASH —
            // not built).
            retire_work_list_entry(client, retry_key, object_id, "empty carrier").await;
            return;
        }
        Ok(RetireRetoken::InProgress) => {
            // A retirement between its begin and its clear, or a drain that
            // re-tokened this carrier moments ago, owns it. Keep the entry.
            tracing::debug!(
                object_id = %object_id,
                "retirement intent still in progress; leaving its carrier for the next sweep"
            );
            return;
        }
        Ok(RetireRetoken::Claimed { members, prior }) => (members, prior, false),
        Ok(RetireRetoken::Legacy { members, prior }) => (members, prior, true),
        Err(e) => {
            origin.observe_error(&e);
            // A lost re-token reply leaves every member on the carrier: the
            // re-token removes nothing, and if it ran, the token it added was
            // never sent, so the next tick resolves it void. Leave the
            // work-list entry registered.
            tracing::warn!(
                object_id = %object_id,
                error = %e,
                "could not re-token a retiring object's recorded external-value releases; \
                 every member is still on its carrier, retrying on the next sweep"
            );
            return;
        }
    };
    let owed = members.len() as u64;

    // 2. Resolve every recorded earlier attempt on `{gc}`.
    if !prior.is_empty() {
        let verdicts = match resolve_enqueue_attempts(conn, keys, object_id, &prior).await {
            Ok(v) => v,
            Err(e) => {
                origin.observe_error(&e);
                tracing::warn!(
                    object_id = %object_id,
                    error = %e,
                    "could not resolve a retiring object's earlier enqueue attempts; enqueuing \
                     nothing and retrying on the next sweep"
                );
                return;
            }
        };
        let any_landed = verdicts.contains(&AttemptVerdict::Landed);
        if any_landed {
            // Paid already. Clear under our token, and remove NO `{gc}` record:
            // those tokens were sent by someone else (a retirement, an earlier
            // drain), and their records are the only guard against a further
            // send under them — the retirement's second attempt, or a copy fred
            // writes late — which must answer `already`, not enqueue again.
            // The GC loop's attempt audit removes them after V (F2).
            match retire_clear(conn, keys, object_id, &token, &prior).await {
                Ok(RetireClear::Cleared) => {
                    result.retire_releases_already_enqueued += owed;
                    if let Some(m) = metrics {
                        m.inc_ext_ref_release_already_enqueued(owed);
                    }
                    retire_work_list_entry(client, retry_key, object_id, "already enqueued").await;
                    tracing::info!(
                        object_id = %object_id,
                        count = owed,
                        "a retiring object's recorded external-value releases had already been \
                         enqueued by an earlier attempt; cleared the carrier without enqueuing \
                         again"
                    );
                }
                Ok(RetireClear::Superseded) => tracing::debug!(
                    object_id = %object_id,
                    "another drain re-tokened the carrier after its earlier attempt resolved \
                     landed; leaving it to that drain"
                ),
                Ok(RetireClear::Gone) => tracing::debug!(
                    object_id = %object_id,
                    "the carrier or its token was already gone after its earlier attempt \
                     resolved landed — removed by a drain, by an older build's claim, or by an \
                     earlier copy of this same clear"
                ),
                Err(e) => {
                    origin.observe_error(&e);
                    tracing::warn!(
                        object_id = %object_id,
                        error = %e,
                        "the clear of a carrier whose earlier enqueue attempt landed failed; the \
                         clear's outcome is unknown, and if it did not run, the next sweep \
                         retries it"
                    );
                }
            }
            return;
        }
    }

    // 2b. An older build's SET carrier with no landed attempt: drop it, never
    // pay it (b.1jt). Most older builds record no attempt token, and no token
    // is not proof that nothing ran (rule E4), so a re-drive could decrement a
    // second time. The clear under our own token — recorded by the re-token,
    // never sent — deletes the carrier and its tokens only while no newer
    // token was added, and never touches `ext_gc_scheduled`, which keeps a
    // live id's eventual retirement from re-deriving an obligation that may
    // have been paid. If it was not, the reference-count reconciler repairs
    // the leak (unless it is turned off, `LIVE_EDIT_EXT_RECONCILE_INTERVAL_SECS=0`,
    // when the leak is permanent).
    if older_build_carrier {
        match retire_clear(conn, keys, object_id, &token, &prior).await {
            Ok(RetireClear::Cleared) => {
                result.retire_releases_dropped += owed;
                if let Some(m) = metrics {
                    m.inc_gc_older_build_by(GC_OLDER_BUILD_EVENT_CARRIER_DROPPED, owed);
                }
                retire_work_list_entry(client, retry_key, object_id, "older build's carrier").await;
                tracing::warn!(
                    object_id = %object_id,
                    count = owed,
                    prior_tokens = prior.len(),
                    "dropped an older build's retirement carrier without enqueuing its \
                     external-value releases: none of its recorded attempts landed, and an older \
                     build may have sent attempts it never recorded, so paying them could \
                     release a reference twice. If they were never paid, the reference-count \
                     reconciler repairs the leaked references, unless the reconciler is turned \
                     off (LIVE_EDIT_EXT_RECONCILE_INTERVAL_SECS=0), in which case the leak is \
                     permanent"
                );
            }
            Ok(RetireClear::Superseded) => tracing::debug!(
                object_id = %object_id,
                "another drain re-tokened an older build's carrier; leaving it to that drain"
            ),
            Ok(RetireClear::Gone) => tracing::debug!(
                object_id = %object_id,
                "an older build's carrier or this drain's token was already gone before the drop \
                 — removed by another drain, or by an older build's claim"
            ),
            Err(e) => {
                origin.observe_error(&e);
                tracing::warn!(
                    object_id = %object_id,
                    error = %e,
                    "the drop of an older build's retirement carrier failed; its outcome is \
                     unknown, and if it did not run, the next sweep retries it"
                );
            }
        }
        return;
    }

    // 3. Settle: this build's HASH carrier, every recorded earlier attempt
    // void — and this build records every attempt before sending it.
    match retire_settle(conn, keys, object_id, &token, &prior).await {
        Ok(RetireSettle::Changed) => {
            tracing::debug!(
                object_id = %object_id,
                "carrier's attempt tokens changed since the re-token; looking again on the next \
                 sweep"
            );
        }
        Ok(RetireSettle::Cancelled(count)) => {
            // The `object_id` is live again, so the obligation is not owed —
            // the live incarnation inherited the standing `+1` these members
            // stood for. The script removed the carrier and its tokens and
            // cleared `ext_gc_scheduled` off the surviving header, which is
            // what hands the object's eventual real retirement its
            // re-derivation back. Every attempt of this obligation was
            // recorded and resolved void, so that pays it once.
            result.retire_carriers_cancelled += 1;
            if let Some(m) = metrics {
                m.inc_ext_ref_release_cancelled(count);
            }
            // INFO, not WARN: this is the gate working. Up to `count` ext_meta
            // `ref_count`s may now be unreleased, which is the leak direction
            // and the reconciler's to drain — as against the over-release
            // (meta+data deleted under a live reference) that paying them would
            // have been.
            tracing::info!(
                object_id = %object_id,
                count,
                "cancelled a retiring object's recorded external-value releases because its \
                 object id is live again; the live incarnation inherited the references"
            );
            retire_work_list_entry(client, retry_key, object_id, "cancelled").await;
        }
        Ok(RetireSettle::Pay) => {
            result.retire_carriers_claimed += 1;
            match enqueue_ext_release_attempt(client, keys, object_id, &token, &members, send_by)
                .await
            {
                Ok(ReleaseAttemptOutcome::Enqueued | ReleaseAttemptOutcome::AlreadyLanded) => {
                    match retire_clear(conn, keys, object_id, &token, &prior).await {
                        Ok(RetireClear::Cleared) => {
                            // F1: our token's single send replied cleanly and
                            // its carrier is cleared, so no carrier names it
                            // and no send of ours under it remains.
                            if let Err(e) =
                                forget_landed_attempts(conn, keys, std::slice::from_ref(&token))
                                    .await
                            {
                                origin.observe_error(&e);
                                tracing::debug!(
                                    object_id = %object_id,
                                    error = %e,
                                    "could not remove a landed enqueue-attempt record; the GC \
                                     loop's attempt audit removes it"
                                );
                            }
                        }
                        // The enqueue landed. `Superseded`: a later drain
                        // re-tokened the carrier and resolves our token
                        // `landed`. `Gone`: the carrier or token was already
                        // removed — by a drain, or by an earlier copy of this
                        // same clear (an older build's claim cannot read this
                        // build's HASH carrier).
                        Ok(RetireClear::Superseded | RetireClear::Gone) => {}
                        Err(e) => {
                            origin.observe_error(&e);
                            tracing::warn!(
                                object_id = %object_id,
                                error = %e,
                                "re-drove a retiring object's recorded external-value releases \
                                 but the clear of its carrier failed; the clear's outcome is \
                                 unknown, and if it did not run, the next sweep finds the attempt \
                                 landed and clears the carrier"
                            );
                        }
                    }
                    result.retire_releases_recovered += owed;
                    if let Some(m) = metrics {
                        m.inc_ext_ref_release_recovered(owed);
                    }
                    retire_work_list_entry(client, retry_key, object_id, "recovered").await;
                    tracing::info!(
                        object_id = %object_id,
                        count = owed,
                        "recovered a retiring object's recorded external-value releases"
                    );
                }
                Ok(ReleaseAttemptOutcome::Voided) => {
                    // Another drain re-tokened this carrier after us and
                    // resolved our token first; it owns the members now.
                    tracing::debug!(
                        object_id = %object_id,
                        "re-drive of a retiring object's releases was voided by a later drain; \
                         leaving the carrier to it"
                    );
                }
                Ok(ReleaseAttemptOutcome::NotSent) => {
                    // This drain passed its send window between the re-token
                    // and the pay: nothing was sent. Our token stays on the
                    // carrier, never sent, so the next tick resolves it void
                    // and settles the members. Keep the work-list entry.
                    tracing::warn!(
                        object_id = %object_id,
                        count = owed,
                        "the re-drive of a retiring object's recorded external-value releases \
                         was NOT SENT: this sweep passed its send window before sending it, so \
                         nothing was enqueued; every member stays recorded on its carrier and \
                         the next sweep settles them"
                    );
                }
                Err(e) => {
                    // Outcome unknown. Our token is on the carrier, so the next
                    // tick resolves it — landed, or void and never to land —
                    // before anything is sent again.
                    tracing::warn!(
                        object_id = %object_id,
                        count = owed,
                        error = %e,
                        "could not confirm the re-drive of a retiring object's recorded \
                         external-value releases; every member is still on its carrier and the \
                         next sweep resolves this attempt before sending another"
                    );
                }
            }
        }
        Err(e) => {
            origin.observe_error(&e);
            tracing::warn!(
                object_id = %object_id,
                error = %e,
                "could not settle a retiring object's recorded external-value releases; \
                 retrying on the next sweep"
            );
        }
    }
}

/// Scan all `le:*:obj:hdr` keys and classify each header that carries its
/// session id and layer name by the layer mapping it names and its fence
/// (`superseded`, with `superseded_by` when an overwrite fenced it):
///
/// - **mapped and unfenced** (or fenced with no `superseded_by`, an older
///   build's): live, left alone;
/// - **mapped, fenced by an overwrite** — a `CreateLayer` fenced it and its
///   commit never landed: [`complete_stranded_overwrite`];
/// - **unmapped, fenced by an overwrite** — replaced (or deleted), its
///   overwrite tail not finished: [`retire_overwrite_fenced_orphan`];
/// - **unmapped otherwise** — a create that never committed, a delete that
///   crashed, or an object this sweep fenced on an earlier pass:
///   [`retire_unmapped_orphan`].
///
/// A header without its session id or layer name is retired by the shared
/// retirement body ([`cleanup_orphaned_object`]) directly, as before, now
/// with the wake-up. Each arm awaits its retirement before the walk moves on,
/// so one pass retires one object at a time.
async fn sweep_orphaned_objects(
    client: &ValkeyClient,
    keys: &KeySchema,
    ext_store: &Arc<dyn ExternalValueStore>,
    layer_store: &LayerStore,
    cancel: &CancellationToken,
    result: &mut SweepResult,
    metrics: Option<&SharedMetrics>,
) -> Result<(), SweepError> {
    let prefix = keys.prefix();
    // Pattern: `le:{*}:obj:hdr` — matches all object header keys.
    // The hash tag braces are literal in the key, so we use `*` inside them.
    let pattern = format!("{prefix}{{*}}:obj:hdr");

    // Maintenance pool: long SCAN + HGETALL passes go through the
    // dedicated pool so writers / readers on the data pool are not
    // queued behind them (b.kf1 H-stor-5). `acquire_maintenance` also
    // bumps `valkey_pool_in_use{pool="maintenance"}` for the duration
    // of this sweep (b.mfn).
    let (maint_pool, _maint_token) = client.acquire_maintenance();
    let scan_client = maint_pool.next();
    let mut stream = scan_keyspace(scan_client, &pattern);

    while let Some(scan_result) = tokio::select! {
        item = stream.next() => item,
        () = cancel.cancelled() => None,
    } {
        let fred_key = match scan_result {
            Ok(k) => k,
            Err(e) => {
                FredErrOrigin::metrics(metrics, POOL_LABEL_MAINTENANCE, client.store_oom_signal())
                    .observe_error(&e);
                tracing::warn!(error = %e, "SCAN error during orphaned object sweep");
                continue;
            }
        };
        let key = match fred_key.as_str() {
            Some(s) => s.to_owned(),
            None => continue,
        };

        result.objects_scanned += 1;

        // Extract the object_id from the key.
        // Key format: `{prefix}{object_id}:obj:hdr`
        // With hash tags: `{prefix}{{object_id}}:obj:hdr`
        let object_id = match extract_hash_tag_id(&key, prefix, ":obj:hdr") {
            Some(id) => id,
            None => {
                tracing::warn!(key = %key, "could not parse object_id from obj:hdr key");
                continue;
            }
        };

        // Check if any layer mapping references this object_id.
        // Read session_id and layer_name from the header to verify.
        //
        // b.muu: this is the sweep's FIRST touch of `obj:hdr`, and it
        // `?`-propagates — a wrong-typed or permanently-failing header aborts the
        // whole sweep pass here, so the counted fence write further down
        // (`cleanup_orphaned_object`) is never reached and would count nothing.
        // Counting the discovery read is what makes the `header` cell move on the
        // sweep's first contact rather than on a touch the abort precludes. The
        // helper borrows the error and returns `()`, so the mapper, the error
        // value and the `?` are unchanged.
        let header: std::collections::HashMap<String, String> = maint_pool
            .hgetall::<std::collections::HashMap<String, String>, _>(&key)
            .await
            .map_err(|e| {
                count_object_key_refusal(
                    metrics,
                    &object_id,
                    SnapshotKeyFamily::Header,
                    &key,
                    "HGETALL obj:hdr (orphan discovery)",
                    &e,
                );
                SweepError::from_fred(
                    FredErrOrigin::metrics(
                        metrics,
                        POOL_LABEL_MAINTENANCE,
                        client.store_oom_signal(),
                    ),
                    "HGETALL obj header",
                    e,
                )
            })?;

        if header.is_empty() {
            // Key vanished between SCAN and HGETALL — benign race.
            continue;
        }

        let session_id = header.get("session_id");
        let layer_name = header.get("layer_name");

        let (Some(sid), Some(lname)) = (session_id, layer_name) else {
            // Missing session_id or layer_name in header → orphaned (corrupt).
            // No create can commit it (no session to commit into), so no
            // grace or veto applies.
            tracing::info!(
                object_id = %object_id,
                session_id = ?session_id,
                layer_name = ?layer_name,
                "found orphaned object, cleaning up"
            );
            if retire_with_shared_body(
                client,
                keys,
                &**ext_store,
                layer_store,
                cancel,
                &object_id,
                metrics,
            )
            .await
            {
                result.orphaned_objects += 1;
            }
            continue;
        };

        // Where the layer mapping points now.
        let mapping_key = keys.layer_mapping(sid, lname);
        let mapped_oid: Option<String> = maint_pool
            .get::<Option<String>, _>(&mapping_key)
            .await
            .map_err(|e| {
                SweepError::from_fred(
                    FredErrOrigin::metrics(
                        metrics,
                        POOL_LABEL_MAINTENANCE,
                        client.store_oom_signal(),
                    ),
                    "GET layer mapping",
                    e,
                )
            })?;
        let mapped = mapped_oid.as_deref() == Some(object_id.as_str());
        // Fenced by an overwrite: the fence names the object that is to
        // replace this one. A fence without it is this sweep's own (an earlier
        // pass that kept the header) or an older build's.
        let superseded_by: Option<u64> = header
            .contains_key(HEADER_FIELD_SUPERSEDED)
            .then(|| header.get(HEADER_FIELD_SUPERSEDED_BY))
            .flatten()
            .and_then(|successor| successor.parse::<u64>().ok());

        let retired = match (mapped, superseded_by) {
            // Live.
            (true, None) => continue,
            (true, Some(successor)) => {
                complete_stranded_overwrite(
                    client,
                    keys,
                    ext_store,
                    layer_store,
                    &object_id,
                    sid,
                    lname,
                    successor,
                    metrics,
                )
                .await?
            }
            (false, Some(successor)) => {
                retire_overwrite_fenced_orphan(
                    client,
                    keys,
                    ext_store,
                    layer_store,
                    &object_id,
                    sid,
                    lname,
                    successor,
                    metrics,
                )
                .await?
            }
            (false, None) => {
                retire_unmapped_orphan(
                    client,
                    keys,
                    &**ext_store,
                    layer_store,
                    cancel,
                    &object_id,
                    &header,
                    sid,
                    lname,
                    mapped_oid.as_deref(),
                    metrics,
                )
                .await?
            }
        };
        if retired {
            result.orphaned_objects += 1;
        }
    }

    Ok(())
}

/// Whether a header's `created_at` (µs since the epoch) is younger than
/// `grace` on this replica's clock. A header without a parseable
/// `created_at` counts as old.
fn created_within(header: &std::collections::HashMap<String, String>, grace: Duration) -> bool {
    let Some(created_at) = header
        .get("created_at")
        .and_then(|created_at| created_at.parse::<u64>().ok())
    else {
        return false;
    };
    let grace_micros = u64::try_from(grace.as_micros()).unwrap_or(u64::MAX);
    now_micros().saturating_sub(created_at) < grace_micros
}

/// Arm (i): complete a stranded overwrite. `object_id` is still the layer's
/// mapping, but a `CreateLayer` fenced it (`superseded=1`) naming `successor`
/// and its commit never landed — the call gave up and its finisher stopped,
/// or its replica crashed. Until the overwrite completes, every `UpdateLayer`
/// to the layer is refused `FAILED_PRECONDITION`. The fence is completed —
/// rolled forward — never cleared.
///
/// - `successor`'s header is gone: nothing can be mapped in its place; ERROR
///   on every pass (a `CreateLayer` of the layer resolves it).
/// - `successor` is younger than the grace: its call or finisher may still
///   be running; left to them this pass.
/// - Otherwise one commit-script run maps the layer to `successor`, expecting
///   this object ([`LayerStore::commit_stranded_overwrite`]); when the layer
///   then names `successor`, this object's overwrite tail runs, its cleanup
///   awaited ([`LayerStore::run_overwrite_tail`]). When `successor`'s
///   `CreateLayer` carried an operation id, its header carries the id and
///   request digest, and the run writes that operation's record, as the
///   call's own landing would have (an older build's object carries neither,
///   and its completion writes no record).
/// - `successor` was already mapped once (`COMMITTED_EARLIER`: its commit
///   claim reads `committed`, so it is never mapped again): when the reply
///   shows the layer no longer naming this object, its tail runs the same
///   way; when it still names this object, the object is live and is left
///   (WARN). This arm never retires a mapped object.
/// - `successor`'s operation is recorded for another object
///   (`ALREADY_COMMITTED` naming it: a concurrent `CreateLayer` with the same
///   operation id committed that object): nothing can map `successor` until
///   the record expires, so the layer refuses writes until then or until a
///   `CreateLayer` replaces it — ERROR on every pass.
/// - Any other outcome is logged (WARN) and left to a later pass, which
///   re-reads the mapping.
///
/// Returns whether this object's keys were deleted.
#[allow(clippy::too_many_arguments)]
async fn complete_stranded_overwrite(
    client: &ValkeyClient,
    keys: &KeySchema,
    ext_store: &Arc<dyn ExternalValueStore>,
    layer_store: &LayerStore,
    object_id: &str,
    session_id: &str,
    layer_name: &str,
    successor: u64,
    metrics: Option<&SharedMetrics>,
) -> Result<bool, SweepError> {
    let Ok(fenced) = object_id.parse::<u64>() else {
        return Ok(false);
    };
    let (maint_pool, _maint_token) = client.acquire_maintenance();
    let successor_header: std::collections::HashMap<String, String> = maint_pool
        .hgetall::<std::collections::HashMap<String, String>, _>(
            keys.obj_header(&successor.to_string()),
        )
        .await
        .map_err(|e| {
            SweepError::from_fred(
                FredErrOrigin::metrics(metrics, POOL_LABEL_MAINTENANCE, client.store_oom_signal()),
                "HGETALL successor obj header",
                e,
            )
        })?;
    if !successor_header.contains_key(HEADER_FIELD_SEQ) {
        tracing::error!(
            object_id = %object_id,
            session_id = %session_id,
            layer_name = %layer_name,
            superseded_by = successor,
            "a CreateLayer fenced this mapped object to replace it, and the object that was to \
             replace it is gone, so the replacement cannot be completed: every UpdateLayer to \
             this layer fails FAILED_PRECONDITION until a CreateLayer of it succeeds"
        );
        return Ok(false);
    }
    if created_within(&successor_header, layer_store.orphan_sweep_grace()) {
        tracing::debug!(
            object_id = %object_id,
            superseded_by = successor,
            "stranded overwrite's successor is younger than the grace; leaving it to its \
             CreateLayer for this pass"
        );
        return Ok(false);
    }
    // The successor's `CreateLayer` operation, when it carried one: the
    // completion writes the operation record that call's own landing would
    // have written. An object an older build materialized carries no such
    // columns, so its completion writes no record, and a later re-send
    // carrying that operation id replaces the layer again. A pair that would
    // not make a valid record (an id `validate_operation_id` refuses, or an
    // empty digest) is not used: the completion then writes no record either.
    let operation: Option<(&str, &str)> = successor_header
        .get(HEADER_FIELD_CREATE_OPERATION_ID)
        .zip(successor_header.get(HEADER_FIELD_CREATE_OPERATION_DIGEST))
        .map(|(operation_id, digest)| (operation_id.as_str(), digest.as_str()))
        .filter(|(operation_id, digest)| {
            crate::domain::validation::validate_operation_id(operation_id).is_ok()
                && !operation_id.is_empty()
                && !digest.is_empty()
        });
    match layer_store
        .commit_stranded_overwrite(session_id, layer_name, fenced, successor, operation)
        .await
    {
        Ok(StrandedOverwriteCommit::Completed) => {
            tracing::info!(
                object_id = %object_id,
                session_id = %session_id,
                layer_name = %layer_name,
                superseded_by = successor,
                "completed a stranded CreateLayer overwrite: the layer now names the object its \
                 fence named; retiring the replaced object"
            );
            Ok(layer_store
                .run_overwrite_tail(
                    session_id,
                    fenced,
                    Arc::clone(ext_store),
                    OverwriteCleanup::Await,
                )
                .await)
        }
        Ok(StrandedOverwriteCommit::CommittedEarlier {
            fenced_still_mapped: false,
        }) => {
            // The successor was mapped once already, and the layer has moved
            // on from this object since: its overwrite landed and only the
            // tail is missing.
            tracing::info!(
                object_id = %object_id,
                session_id = %session_id,
                layer_name = %layer_name,
                superseded_by = successor,
                "a stranded CreateLayer overwrite had already committed and the layer no longer \
                 names the replaced object; retiring it"
            );
            Ok(layer_store
                .run_overwrite_tail(
                    session_id,
                    fenced,
                    Arc::clone(ext_store),
                    OverwriteCleanup::Await,
                )
                .await)
        }
        Ok(StrandedOverwriteCommit::CommittedEarlier {
            fenced_still_mapped: true,
        }) => {
            // The successor was mapped once and can never be mapped again,
            // yet the layer names this object: it is live, and is never
            // retired by this arm.
            tracing::warn!(
                object_id = %object_id,
                session_id = %session_id,
                layer_name = %layer_name,
                superseded_by = successor,
                "a CreateLayer fenced this mapped object to replace it, and the object that was to \
                 replace it was already committed once and cannot be mapped again, yet the layer \
                 names this object; leaving it, since it is the layer's live object"
            );
            Ok(false)
        }
        Ok(StrandedOverwriteCommit::OperationRecordedForAnother { recorded_object_id }) => {
            // Logged once per pass, as the gone-successor ERROR above is: the
            // sweep has no log-once state of its own.
            tracing::error!(
                object_id = %object_id,
                session_id = %session_id,
                layer_name = %layer_name,
                superseded_by = successor,
                operation_id = operation.map(|(operation_id, _)| operation_id).unwrap_or(""),
                recorded_object_id,
                "a CreateLayer fenced this mapped object to replace it, but that CreateLayer's \
                 operation_id is already recorded for another object, so the replacement cannot \
                 be completed: every UpdateLayer to this layer fails FAILED_PRECONDITION until \
                 the operation record expires (up to 24 hours after it was written), after \
                 which the orphan sweep completes the replacement, or until a CreateLayer \
                 replaces the layer"
            );
            Ok(false)
        }
        Ok(StrandedOverwriteCommit::NotCompleted) => {
            tracing::warn!(
                object_id = %object_id,
                session_id = %session_id,
                layer_name = %layer_name,
                superseded_by = successor,
                "could not complete a stranded CreateLayer overwrite: the layer mapping moved, \
                 the successor was vetoed, the session was deleted, or the successor's own \
                 operation record shows its commit landed; a later pass re-reads the mapping"
            );
            Ok(false)
        }
        Err(e) => {
            // The commit dispatch already observed `e` (a command timeout
            // counted, a store write refused `-OOM` noted on the store
            // out-of-memory signal, b.pa1).
            tracing::warn!(
                object_id = %object_id,
                session_id = %session_id,
                layer_name = %layer_name,
                superseded_by = successor,
                error = %e,
                "the commit completing a stranded CreateLayer overwrite failed; a later pass \
                 re-reads the mapping"
            );
            Ok(false)
        }
    }
}

/// Arm (ii): retire an object an overwrite fenced (`superseded_by` names
/// `successor`) that no mapping names any more — replaced, or deleted —
/// whose overwrite tail did not finish (its cleanup could not wake the
/// stream, or its replica stopped). Re-runs the whole overwrite tail for it
/// ([`LayerStore::run_overwrite_tail`]) and awaits the cleanup, so one pass
/// never runs two cleanup bodies at once. Idempotent: the release scheduling
/// has its own double-scheduling guard and the checkpoint reclamation
/// registration is a `ZADD NX`.
///
/// **Grace.** While `successor`'s header `created_at` is younger than
/// [`LayerStore::orphan_sweep_grace`] the object is skipped this pass: the
/// overwrite's own `CreateLayer` call or finisher may still be running its
/// cleanup, and a second body beside it would walk the replaced layer's
/// references a second time, holding a second retirement reservation. A successor header that is gone, or carries no parseable
/// `created_at`, counts as old (the same age check as arm (i),
/// [`created_within`]).
///
/// Returns whether the object's keys were deleted.
#[allow(clippy::too_many_arguments)]
async fn retire_overwrite_fenced_orphan(
    client: &ValkeyClient,
    keys: &KeySchema,
    ext_store: &Arc<dyn ExternalValueStore>,
    layer_store: &LayerStore,
    object_id: &str,
    session_id: &str,
    layer_name: &str,
    successor: u64,
    metrics: Option<&SharedMetrics>,
) -> Result<bool, SweepError> {
    let Ok(old_oid) = object_id.parse::<u64>() else {
        return Ok(false);
    };
    let (maint_pool, _maint_token) = client.acquire_maintenance();
    let successor_header: std::collections::HashMap<String, String> = maint_pool
        .hgetall::<std::collections::HashMap<String, String>, _>(
            keys.obj_header(&successor.to_string()),
        )
        .await
        .map_err(|e| {
            SweepError::from_fred(
                FredErrOrigin::metrics(metrics, POOL_LABEL_MAINTENANCE, client.store_oom_signal()),
                "HGETALL successor obj header",
                e,
            )
        })?;
    if created_within(&successor_header, layer_store.orphan_sweep_grace()) {
        tracing::debug!(
            object_id = %object_id,
            superseded_by = successor,
            "replaced object's successor is younger than the orphan sweep grace; leaving its \
             retirement to the overwrite's own cleanup this pass"
        );
        return Ok(false);
    }
    tracing::info!(
        object_id = %object_id,
        session_id = %session_id,
        layer_name = %layer_name,
        superseded_by = successor,
        "found an object a CreateLayer overwrite replaced whose retirement did not finish; \
         re-running its overwrite tail"
    );
    Ok(layer_store
        .run_overwrite_tail(
            session_id,
            old_oid,
            Arc::clone(ext_store),
            OverwriteCleanup::Await,
        )
        .await)
}

/// Arm (iii): retire an unmapped object no overwrite fenced — the object of
/// a `CreateLayer` that never committed, one whose `DeleteLayer` stopped
/// after removing the mapping, or one this sweep fenced on an earlier pass
/// and kept.
///
/// 1. **Grace.** Younger than [`LayerStore::orphan_sweep_grace`]: skipped
///    this pass — its create may still be about to commit.
/// 2. **Pending successor.** The object the mapping names carries
///    `superseded_by` naming this one: it is the successor of an unfinished
///    overwrite (arm (i) completes it); skipped.
/// 3. **Veto** (`scripts/create_layer_veto.lua` on `{session_id}`). The
///    mapping names this object → skipped (its commit just landed).
///    Otherwise the object's commit claim is written `vetoed` when no claim
///    exists, so no commit of it can land from now on; an existing claim
///    (`committed` or `vetoed`) is kept and the retirement continues, since
///    with either no commit can map the object again; a deleted session's
///    marker writes nothing and the retirement continues. The veto runs on a
///    full store (its `SET` follows a no-op `PEXPIRE` opener: the opener
///    rule, SDD §7.5.1). A failed veto skips the object this pass.
/// 4. **Retirement**, the shared body ([`cleanup_orphaned_object`]): fence →
///    wake-up → release scheduling → expiry → key deletion.
///
/// Returns whether the object's keys were deleted.
#[allow(clippy::too_many_arguments)]
async fn retire_unmapped_orphan(
    client: &ValkeyClient,
    keys: &KeySchema,
    ext_store: &dyn ExternalValueStore,
    layer_store: &LayerStore,
    cancel: &CancellationToken,
    object_id: &str,
    header: &std::collections::HashMap<String, String>,
    session_id: &str,
    layer_name: &str,
    mapped_oid: Option<&str>,
    metrics: Option<&SharedMetrics>,
) -> Result<bool, SweepError> {
    if created_within(header, layer_store.orphan_sweep_grace()) {
        tracing::debug!(
            object_id = %object_id,
            "unmapped object is younger than the orphan sweep grace; leaving it this pass"
        );
        return Ok(false);
    }
    let Ok(oid) = object_id.parse::<u64>() else {
        // A non-numeric id names no object a create could commit.
        return Ok(retire_with_shared_body(
            client,
            keys,
            ext_store,
            layer_store,
            cancel,
            object_id,
            metrics,
        )
        .await);
    };

    if let Some(mapped) = mapped_oid.filter(|mapped| mapped.parse::<u64>().is_ok()) {
        let (maint_pool, _maint_token) = client.acquire_maintenance();
        let pending: Option<String> = maint_pool
            .hget::<Option<String>, _, _>(keys.obj_header(mapped), HEADER_FIELD_SUPERSEDED_BY)
            .await
            .map_err(|e| {
                SweepError::from_fred(
                    FredErrOrigin::metrics(
                        metrics,
                        POOL_LABEL_MAINTENANCE,
                        client.store_oom_signal(),
                    ),
                    "HGET mapped obj header superseded_by",
                    e,
                )
            })?;
        if pending.as_deref() == Some(object_id) {
            tracing::debug!(
                object_id = %object_id,
                mapped_object_id = %mapped,
                "unmapped object is the pending successor of an unfinished overwrite; leaving it"
            );
            return Ok(false);
        }
    }

    let veto_keys = vec![
        keys.layer_mapping(session_id, layer_name),
        keys.session_deleted_marker(session_id),
        keys.create_layer_claim(session_id, oid),
    ];
    let veto_argv = vec![
        object_id.to_owned(),
        CREATE_LAYER_CLAIM_VETOED_TTL_MS.to_string(),
    ];
    let vetoed: Result<Vec<String>, fred::error::Error> = CREATE_LAYER_VETO_SCRIPT
        .evalsha_with_reload(client.pool.next(), veto_keys, veto_argv)
        .await;
    match vetoed
        .as_ref()
        .map(|reply| reply.first().map(String::as_str))
    {
        Ok(Some("VETOED")) => tracing::info!(
            object_id = %object_id,
            session_id = %session_id,
            layer_name = %layer_name,
            "vetoed an unmapped object's CreateLayer commit; retiring the object"
        ),
        // The object already has a commit claim — `committed` (a commit
        // mapped it, and the mapping has since moved away without fencing
        // it, e.g. a DeleteLayer that stopped after removing the mapping) or
        // `vetoed` (an earlier pass) — kept as it is. Either way no commit
        // can map it again, so the retirement continues.
        Ok(Some("CLAIMED")) => tracing::info!(
            object_id = %object_id,
            session_id = %session_id,
            layer_name = %layer_name,
            claim = ?vetoed.as_ref().ok().and_then(|reply| reply.get(1)),
            "an unmapped object's CreateLayer commit claim is already set and kept; retiring \
             the object"
        ),
        Ok(Some("SESSION_DELETED")) => {}
        Ok(Some("MAPPED")) => {
            tracing::debug!(
                object_id = %object_id,
                "the object's CreateLayer committed since the mapping was read; leaving it"
            );
            return Ok(false);
        }
        Ok(other) => {
            tracing::warn!(
                object_id = %object_id,
                reply = ?other,
                "the create veto answered an unexpected reply; leaving the object this pass"
            );
            return Ok(false);
        }
        Err(e) => {
            // The veto runs on the data pool. Counts a command timeout and
            // notes a store write refused `-OOM` on the store out-of-memory
            // signal (b.pa1).
            FredErrOrigin::metrics(metrics, POOL_LABEL_DATA, client.store_oom_signal())
                .observe_error(e);
            tracing::warn!(
                object_id = %object_id,
                error = %e,
                "could not veto an unmapped object's CreateLayer commit; leaving the object this \
                 pass"
            );
            return Ok(false);
        }
    }

    tracing::info!(
        object_id = %object_id,
        session_id = %session_id,
        layer_name = %layer_name,
        "found orphaned object, cleaning up"
    );
    Ok(retire_with_shared_body(
        client,
        keys,
        ext_store,
        layer_store,
        cancel,
        object_id,
        metrics,
    )
    .await)
}

/// Run the shared retirement body ([`cleanup_orphaned_object`]) for one
/// orphan, logging a failure at WARN rather than ending the pass: a header
/// the body kept is found again by the next pass. Returns whether the
/// object's keys were deleted.
async fn retire_with_shared_body(
    client: &ValkeyClient,
    keys: &KeySchema,
    ext_store: &dyn ExternalValueStore,
    layer_store: &LayerStore,
    cancel: &CancellationToken,
    object_id: &str,
    metrics: Option<&SharedMetrics>,
) -> bool {
    match cleanup_orphaned_object(
        client,
        keys,
        ext_store,
        layer_store,
        cancel,
        object_id,
        metrics,
    )
    .await
    {
        Ok(deleted) => deleted,
        Err(e) => {
            tracing::warn!(
                object_id = %object_id,
                error = %e,
                "failed to clean up orphaned object"
            );
            false
        }
    }
}

/// Scan all `le:{*}:layer:*` keys. For each, extract the session_id and check
/// whether the session still exists. If not, delete the mapping and remove from
/// the layers index.
async fn sweep_orphaned_layers(
    client: &ValkeyClient,
    keys: &KeySchema,
    cancel: &CancellationToken,
    result: &mut SweepResult,
) -> Result<(), SweepError> {
    let prefix = keys.prefix();
    // Pattern: `le:{*}:layer:*` — matches all layer mapping keys. The literal
    // braces are the Cluster hash tag, globbed here to span every session.
    //
    // This is now the SOLE copy of the layer-mapping key shape outside
    // `KeySchema::layer_mapping` itself. It used to share a
    // `KeySchema::layer_mapping_match_all_sessions_pattern` accessor with the
    // checkpoint apply guard, which b.b2j retired: that guard stopped scanning
    // and now GETs the one mapping the blob's own `(session_id, layer_name)`
    // names, leaving this sweep the only enumerate-all-mappings caller. Kept
    // inline rather than as a one-caller accessor — but a change to
    // `KeySchema::layer_mapping`'s shape must be mirrored HERE, with no
    // compiler error to catch the omission.
    let pattern = format!("{prefix}{{*}}:layer:*");

    // `acquire_maintenance` bumps `valkey_pool_in_use{pool="maintenance"}`
    // for the duration of this sweep (b.mfn).
    let (maint_pool, _maint_token) = client.acquire_maintenance();
    let scan_client = maint_pool.next();
    let mut stream = scan_keyspace(scan_client, &pattern);

    while let Some(scan_result) = tokio::select! {
        item = stream.next() => item,
        () = cancel.cancelled() => None,
    } {
        let fred_key = match scan_result {
            Ok(k) => k,
            Err(e) => {
                FredErrOrigin::client(client, POOL_LABEL_MAINTENANCE).observe_error(&e);
                tracing::warn!(error = %e, "SCAN error during orphaned layer sweep");
                continue;
            }
        };
        let key = match fred_key.as_str() {
            Some(s) => s.to_owned(),
            None => continue,
        };

        result.layers_scanned += 1;

        // Extract session_id from the key.
        // Key format: `{prefix}{{session_id}}:layer:{hex_layer_name}`
        let session_id = match extract_hash_tag_id(&key, prefix, ":layer:") {
            Some(id) => id,
            None => {
                tracing::warn!(key = %key, "could not parse session_id from layer mapping key");
                continue;
            }
        };

        // Check if the session still exists.
        let sess_key = keys.session_meta(&session_id);
        let exists: bool = maint_pool.exists::<bool, _>(&sess_key).await.map_err(|e| {
            SweepError::from_fred(
                FredErrOrigin::client(client, POOL_LABEL_MAINTENANCE),
                "EXISTS session meta",
                e,
            )
        })?;

        if !exists {
            // Extract the hex layer name from the key suffix for SREM.
            let layer_suffix_start = format!("{prefix}{{{session_id}}}:layer:");
            let hex_layer = key
                .strip_prefix(&layer_suffix_start)
                .unwrap_or_default()
                .to_owned();

            tracing::info!(
                session_id = %session_id,
                hex_layer = %hex_layer,
                "found orphaned layer mapping, cleaning up"
            );

            // Delete the mapping key.
            maint_pool.del::<(), _>(&key).await.map_err(|e| {
                SweepError::from_fred(
                    FredErrOrigin::client(client, POOL_LABEL_MAINTENANCE),
                    "DEL orphaned layer mapping",
                    e,
                )
            })?;

            // Remove from the layers index set.
            let layers_index_key = keys.layers_index(&session_id);
            maint_pool
                .srem::<(), _, _>(&layers_index_key, &hex_layer)
                .await
                .map_err(|e| {
                    SweepError::from_fred(
                        FredErrOrigin::client(client, POOL_LABEL_MAINTENANCE),
                        "SREM orphaned layer from index",
                        e,
                    )
                })?;

            // Also clean up the layers index set itself if now empty. Use DEL
            // only if SCARD returns 0 to avoid a race.
            //
            // b.j9j item 3: pre-b.j9j these two Valkey calls used
            // `unwrap_or(1)` / `unwrap_or_default()` to swallow errors —
            // a transient blip would silently leave behind an
            // empty-but-extant layers-index set keyed off a dead
            // session, polluting `ListSessions` until the next sweep
            // happened to retry from scratch. Propagating the error
            // lets `run_one_sweep` retry the orphaned-layer pass on
            // the next tick, with the layer mapping already DEL'd so
            // the second pass is a clean no-op for the index cleanup.
            let remaining: u64 = maint_pool
                .scard::<u64, _>(&layers_index_key)
                .await
                .map_err(|e| {
                    SweepError::from_fred(
                        FredErrOrigin::client(client, POOL_LABEL_MAINTENANCE),
                        "SCARD layers index",
                        e,
                    )
                })?;

            if remaining == 0 {
                maint_pool
                    .del::<(), _>(&layers_index_key)
                    .await
                    .map_err(|e| {
                        SweepError::from_fred(
                            FredErrOrigin::client(client, POOL_LABEL_MAINTENANCE),
                            "DEL empty layers index",
                            e,
                        )
                    })?;
            }

            result.orphaned_layers += 1;
        }
    }

    Ok(())
}

/// Clean up all Valkey keys belonging to an orphaned object.
///
/// Schedules external-value GC for every external hash referenced by the
/// orphan BEFORE deleting its keys (b.kf1 H-stor-1) — pre-b.kf1 the
/// sweeper UNLINKed per-field hashes without touching `ext_meta`, leaving
/// a permanent ref_count inflation and an unreapable disk leak in the
/// external-value store. The schedule call delegates to the same
/// `schedule_old_object_ext_gc` used by the CreateLayer overwrite + the
/// `delete_layer` cleanup paths, so all three retirement routes drive the
/// GC bookkeeping through a single chokepoint.
///
/// Fences the object's header (`HSET obj:hdr superseded 1`) before that
/// scheduling's reference walk, so a straggler `UpdateLayer` cannot land a
/// commit between the walk and the teardown and release the same reference the
/// walk is about to release (b.qpf post-completion). The other two retirement
/// routes already carried an equivalent fence; the inline comment on the
/// `HSET` has the full argument for why one header flag covers both commit
/// backends.
///
/// Wakes the object's stream after the fence and gives it its expiry before
/// the teardown: no retired object loses its header before its stream carries
/// `deleted` and an expiry, so a subscriber still tailing it (a `DeleteLayer`
/// that stopped after removing the mapping leaves one) re-reads the mapping
/// and ends. A wake-up or expiry that fails keeps the header — the object is
/// found again by the next pass — except a wake-up the store refused because
/// it is out of memory: deleting keys is what frees a full store, so the
/// retirement goes on (WARN). Returns whether the keys were deleted.
///
/// **Memory** (`storage::retirement`). After the wake-up — and only after it,
/// so a wake-up is never queued behind memory — the retirement sizes its
/// reservation (`HGET obj:hdr ext_gc_scheduled`, `HLEN obj:ext`, the five
/// structural indices' cardinalities) and reserves the object's charge
/// (`storage::retirement::retirement_charge_bytes`) on the retirement
/// allowance `layer_store` holds, the wait raced against the sweep's `cancel`
/// (shutdown's end of the allowance's waits ends it too, and refuses one that
/// would have to wait; one that fits is still granted); it shrinks to its page
/// once the release returns and holds that through the expiry and the
/// teardown. A size read that fails keeps everything for the next pass. A
/// wrong-typed `obj:hdr` or `obj:ext` is counted and WARNed, and the
/// retirement goes on with the full page bound alone and no release; a
/// wrong-typed structural index, counted and WARNed too, sizes its page at
/// the full page bound, its owed references still charged beside it, and the
/// retirement goes on (a wrong-typed `obj:flds` or `obj:tss`
/// then fails the release's walk, and every wrong-typed index stops the
/// teardown after its first `UNLINK`, b.yz5). With the fence failed, or a non-numeric
/// id, there is no size read and no release: the full page bound alone is
/// reserved, for the teardown.
///
/// **The teardown** is the paged teardown on the maintenance pool
/// (`storage::retirement::Teardown`): the first `UNLINK` holds the header,
/// `obj:ext` and every other fixed key, the row pages follow, the five
/// structural indices go last — the same inventory as `layer.rs`'s
/// `cleanup_old_object` and `cleanup_old_object_retrying`
/// (`KeySchema::object_cleanup_keys` stays its authority). Errors map to
/// `SweepError` (non-fatal: the caller WARNs and the pass goes on) rather than
/// `ServiceError`. Up to and including the first `UNLINK` a failure keeps the
/// header, so the next sweep tick finds the object and retries it. After the
/// first `UNLINK` a page that fails transiently is retried in place; a failure
/// that outlasts those retries is final, because once the header is gone no
/// later pass can find the object.
#[allow(clippy::too_many_arguments)]
async fn cleanup_orphaned_object(
    client: &ValkeyClient,
    keys: &KeySchema,
    ext_store: &dyn ExternalValueStore,
    layer_store: &LayerStore,
    cancel: &CancellationToken,
    object_id: &str,
    metrics: Option<&SharedMetrics>,
) -> Result<bool, SweepError> {
    // b.kf1 H-stor-1: schedule external-value ref-decrements before the
    // structural teardown so the externally-referenced blobs are correctly
    // ref-decremented (pre-b.kf1 they leaked into permanent ref_count
    // inflation).
    //
    // Four failure modes of the release scheduling — none of them
    // short-circuits the structural cleanup, because permanent orphan
    // retention is a strictly worse failure mode than ext-value ref-count
    // inflation (orphan retention can only be cleared by another sweep, which
    // will hit the same failure), and because retaining `obj:hdr` to gate the
    // teardown re-opens the b.1ah re-adoption hazard. All four therefore fall
    // through to the header-first teardown below. (The steps that DO keep the
    // header are the stream's wake-up and its expiry, the retirement's size
    // read, and its wait for the allowance ended by the sweep's cancel or the
    // shutdown: a header deleted before its stream carries `deleted` and an
    // expiry leaves subscribers unwoken, or a stream nothing can find again;
    // a retirement that could not size or reserve its memory has not run its
    // release yet. All clear on a later pass, unlike the failures here.)
    //
    // Modes 1, 2 and 4 share one recovery rule: a later sweep re-visits
    // this orphan only while `obj:hdr` survives — the sweep discovers
    // orphans by SCANning `le:*:obj:hdr` — i.e. only if the teardown's first
    // UNLINK fails too. Once that UNLINK lands they are recoverable only by
    // the reference-count reconciler, which observes the resulting surplus
    // rather than re-deriving the hashes (nothing can re-derive them).
    // Mode 3 is different since b.yda: its owed set is KNOWN, so it is
    // written to a durable carrier that outlives the teardown and is
    // drained by this same sweep's retire-retry phase. What set mode 3
    // apart before was that it was merely *counted*; what sets it apart
    // now is that it is *recovered*:
    //
    // 1. **Non-numeric `object_id`** — every key this service writes is
    //    derived from a decimal `u64` object id, so a non-numeric one names
    //    no stream to wake and nothing to release (the snapshot reads have
    //    refused such an id since b.k1r H-stor-8). It should never appear in
    //    a healthy keyspace, but a corrupted obj:hdr or a future schema
    //    change could land one here. Skip ext-GC with a warn log and go
    //    straight to the teardown, under the full page bound alone.
    //
    // 2. **Reference-walk Valkey error** (e.g. a transient I/O blip, a
    //    foreign value on a value index or row). Log + proceed; the next
    //    sweep will retry the same orphan only if any structural key remains
    //    — concretely `obj:hdr`, so only if the teardown's first UNLINK fails
    //    too — and the ext refs are at worst slightly inflated until then.
    //
    // 3. **The release enqueue itself failing.** This is the one mode that
    //    IS recoverable, and it is recovered rather than merely counted
    //    (b.yda; b.yuo Part R). `schedule_old_object_ext_gc` records the
    //    owed members on the object's own `le:{oid}:obj:extret` carrier —
    //    deliberately absent from `KeySchema::object_cleanup_keys`, so the
    //    teardown below leaves it standing — together with
    //    `ext_gc_scheduled` and its attempt token, BEFORE it enqueues; when
    //    both attempts end in doubt it registers the object on the
    //    `le:ext:{gc}:retire-retry` work list, ticks
    //    `ext_ref_release_deferred_total`, and returns the `Err` this path
    //    still swallows. THIS sweep's own retire-retry phase
    //    (`sweep_retire_retry`) settles the carrier on a later tick, so
    //    recovery does not depend on the teardown failing. Only if the
    //    begin that records the carrier fails (or finds an earlier
    //    lifecycle's carrier outstanding) is the obligation lost, and only
    //    then does `ext_ref_release_enqueue_failed_total` tick. Modes 1
    //    and 2 abort before the release set exists, so they can be neither
    //    counted nor recorded and remain drained by the reference-count
    //    reconciler (`storage::ext_reconcile`). This path deliberately
    //    still proceeds to the teardown in every mode.
    //
    // 4. **The retirement FENCE below failing.** Same posture as 1 and 2 —
    //    no honest count, log + proceed to the teardown — but reached by
    //    skipping the ext-GC scheduling deliberately rather than by an
    //    error out of it. See the fence's own comment for why an unfenced
    //    read is worse than a skipped one.
    //
    // ── The fence (b.qpf post-completion review, finding 2) ──────────────
    //
    // Write the retirement fence BEFORE the retirement read. The reference
    // walk inside `schedule_old_object_ext_gc` (`obj:ext` plus the value
    // rows' hashes) decides which hashes this object still owes a release
    // for; a commit that lands AFTER that walk and BEFORE the teardown below
    // releases the hashes it superseded on its own (`ExtRelease::commit`),
    // while our pre-commit walk releases the same hashes again at
    // `event = ret` — two decrements for one reference, i.e. an
    // over-release that can delete a blob another object still points at.
    // The fence is also what makes the walk complete: with no commit landing,
    // the value indices hold still under its rank pages. This is not
    // hypothetical for an orphan: an orphan is
    // precisely an object whose layer mapping is gone, and the classic way
    // to produce one is a `delete_layer` that crashed after its step 2
    // (`DEL` mapping) — leaving a straggler `UpdateLayer` that resolved the
    // mapping just before that `DEL` still holding a live `WATCH` on this
    // object's header and still able to commit.
    //
    // The other two retirement paths already fence: `delete_layer` XADDs
    // the `deleted` sentinel with an auto-ID (step 3) before its step-4
    // schedule, and the CreateLayer-overwrite path HSETs `superseded=1` on
    // the old header before spawning its cleanup task. The sweep was the
    // only unfenced one — so `ref_count` is decremented exactly once per
    // release event on every retirement route, with no exception.
    //
    // One `HSET obj:hdr superseded 1` satisfies BOTH commit backends,
    // which is why it is the whole fence:
    //
    // - **Fast path** (`update_layer_fast.lua` §1c): the script tests
    //   `if hdr["superseded"]` on the header it HGETALLs at the top and
    //   returns `SUPERSEDED` before mutating anything. `EVAL` is atomic, so
    //   a commit either ran wholly before this `HSET` — in which case its
    //   row rewrite is visible to our walk below and it released its own
    //   superseded hashes — or wholly after, in which case it writes
    //   nothing. Usually that is the `SUPERSEDED` refusal; the one exception
    //   is a run whose commit-token match key already names a commit that
    //   landed BEFORE this `HSET` — an in-call retry of that commit, or a
    //   client re-send carrying its operation id: the commit-token check
    //   (§1b) runs ahead of the superseded test and answers `ALREADY_LANDED`
    //   (an OK reporting that earlier commit, or an `ALREADY_EXISTS` refusal
    //   for a different request under the same id). It still writes nothing,
    //   so the fence holds.
    // - **Slow path** (`update_layer_once`): it `WATCH`es this same header
    //   key, so an in-flight attempt that read the header before this
    //   `HSET` has its `EXEC` cancelled into the existing `Aborted` → retry
    //   loop, and the retry's `HGETALL` then sees `superseded == "1"` and
    //   returns `FailedPrecondition`. An attempt that starts after the
    //   `HSET` sees the flag on its first read. The same exception applies:
    //   the step-2a commit-token check precedes the superseded guard, so a
    //   retry whose own earlier attempt landed before this `HSET` — or a
    //   re-send whose operation's commit did — reports that commit (or
    //   refuses a different request under the same id) and writes nothing.
    //
    // The `XADD deleted` that follows the fence is not part of it: the
    // stream-top ceiling both backends check is not what stops the straggler
    // — the header flag is. The wake-up is for subscribers: an orphan has no
    // mapping, but a subscriber can still be tailing its stream (a
    // `DeleteLayer` that stopped after removing the mapping leaves one), and
    // `deleted` is its only signal to re-read the mapping.
    //
    // Idempotent (a re-sweep re-writes the same field), harmless on a
    // legitimately half-deleted object (it is doomed either way), and
    // single-key on the `{object_id}` slot, so it is Cluster-safe and needs
    // no hash-tag reasoning. It runs on `client.pool` rather than the
    // maintenance pool because it fences the size read and the release's
    // flag read and begin, which `schedule_old_object_ext_gc` and its caller
    // issue on `client.pool`; the maintenance-pool token below still covers
    // the teardown that is this function's own work.
    //
    // If `obj:hdr` had already been UNLINKed by a racing retirement, this
    // `HSET` re-creates it as a single-field ghost — the same trade the
    // overwrite path documents. That ghost is not stranded: `obj:hdr` is the
    // first key of the teardown's first UNLINK
    // (`KeySchema::object_teardown_first_keys`), which removes it, and if that
    // UNLINK fails the next sweep re-discovers the object by exactly that key.
    //
    // The opposite direction — the header that SURVIVES the fence (b.1ah).
    // Every step between here and the teardown's first UNLINK that can fail —
    // the wake-up, the size read, the reservation's wait (ended by the
    // sweep's cancel or the shutdown), the expiry, that UNLINK itself — ends
    // this cleanup with `obj:hdr` still standing and carrying `superseded=1`.
    // While the object stays mapping-less that is self-healing — it is still
    // an orphan, so a later sweep re-classifies it and finishes the job.
    // Re-adoption breaks the bound: `apply_session_checkpoint` Step B.2
    // SETs the layer mapping to the checkpointed `object_id`
    // unconditionally, so a DR session restore can hand a mapping back to
    // an object this sweep already fenced. No sweep revisits it then, and
    // `apply_restore` step 1 short-circuits `AlreadyPresent` because
    // `header_fields_are_structurally_complete` ignores `superseded` — so
    // the residue wipe that would have cleared the flag never runs, and
    // both commit backends refuse every write with `FAILED_PRECONDITION`,
    // which the `UpdateLayer` handler excludes from its recovery arm:
    // permanent write-death of one layer, a new terminal state on the DR
    // path. Fail-closed is still the right direction — the orphan's ext
    // references were already released, so pre-fence the same interleaving
    // served a silently half-dead layer — and this sweep must NOT clear its
    // own flag on a later failure, which would re-open the
    // over-release the fence exists to close. b.1ah is the design pass that
    // owns the remedy.
    let paging = layer_store.retirement_paging();
    let numeric_id = object_id.parse::<u64>().ok();
    // The retirement's reservation, held through the teardown.
    let reservation = if let Some(oid) = numeric_id {
        let header_key = keys.obj_header(object_id);
        let fenced: Result<(), fred::error::Error> = client
            .pool
            .hset::<(), _, _>(&header_key, [("superseded", "1")])
            .await;
        let fence_landed = match fenced {
            Ok(()) => true,
            // Failure mode 4: skip the scheduling entirely rather than read
            // unfenced. Skipping leaks up to one `ref_count` per hash this
            // orphan holds, which is the recoverable direction and the same
            // outcome as modes 1 and 2; reading unfenced risks releasing a
            // reference twice, which deletes a live blob and loses data. No
            // *leak* count is ticked, for the same reason as modes 1 and 2 —
            // the release set was never built, so there is no honest N. (b.muu
            // added a count of the refusal ITSELF below, which is a different
            // question: not how much leaked, but why the fence failed.)
            Err(e) => {
                // b.muu: the fence is a WRITE, but it is a write on `obj:hdr`
                // with exactly the `WRONGTYPE` exposure of the header PRE-READS
                // the same counter covers, and with the same remediation — so it
                // ticks `live_edit_snapshot_read_type_error_total{key_family=
                // "header"}` rather than a write-side sibling whose other ten
                // cells would be permanently zero. A permanent non-WRONGTYPE
                // failure ticks the b.muu refusal counter on the same call. The
                // helper borrows the error and returns `()`, so the `warn!`
                // below and this arm's behaviour (skip ext-GC scheduling, proceed
                // to the teardown) are untouched. Metrics arrive here as a
                // parameter, not via `client.metrics_snapshot()` — see the
                // helper's doc.
                // This is the sweep's SECOND counted touch of `obj:hdr`, not its
                // first: `sweep_orphaned_objects` is the only caller of this
                // function and reaches it only past its own counted discovery
                // `HGETALL` on the same key, so a wrong-typed header is counted
                // there and the pass ends before this fence is ever issued.
                count_object_key_refusal(
                    metrics,
                    object_id,
                    SnapshotKeyFamily::Header,
                    &header_key,
                    "HSET obj:hdr superseded (retirement fence)",
                    &e,
                );
                // The fence is issued on the data pool (see above).
                FredErrOrigin::metrics(metrics, POOL_LABEL_DATA, client.store_oom_signal())
                    .observe_error(&e);
                tracing::warn!(
                    object_id = %object_id,
                    error = %e,
                    "could not fence the orphan's header against a straggler commit; skipping \
                     ext-value GC scheduling (an unfenced retirement read can over-release and \
                     delete a live blob) and proceeding with structural cleanup. Up to one \
                     ext_meta ref_count per hash this object holds leaks unless the teardown \
                     also fails before removing the header and a later sweep re-visits it"
                );
                false
            }
        };

        // Wake subscribers still tailing the orphan's stream, before anything
        // of the object goes. Gating: a failed wake-up keeps the header, so
        // the next pass retries it — except when the store refused it because
        // it is out of memory, where deleting keys is what frees the store.
        if !wake_orphan_stream(client, keys, oid, metrics).await {
            return Ok(false);
        }

        if fence_landed {
            // Size the retirement, then reserve its memory, now that the
            // wake-up has landed.
            let size = match read_retirement_size(client, keys, object_id, metrics).await {
                Ok(size) => size,
                Err(e) => {
                    FredErrOrigin::metrics(metrics, POOL_LABEL_DATA, client.store_oom_signal())
                        .observe_error(&e);
                    tracing::warn!(
                        object_id = %object_id,
                        error = %e,
                        "could not size the orphan's clean-up (its retirement size read \
                         failed); keeping the orphan for the next sweep"
                    );
                    return Ok(false);
                }
            };
            let Some(reservation) = reserve_retirement(
                layer_store,
                cancel,
                object_id,
                size.page(paging),
                size.references(),
            )
            .await
            else {
                return Ok(false);
            };
            if size.releases() {
                let walk = RetirementWalk::reserved(Arc::clone(&reservation));
                if let Err(e) = schedule_old_object_ext_gc(
                    client,
                    keys,
                    object_id,
                    ext_store,
                    metrics.map(|m| m.as_ref()),
                    &walk,
                )
                .await
                {
                    tracing::warn!(
                        object_id = %object_id,
                        error = %e,
                        "ext-value GC scheduling failed for orphan; proceeding with structural \
                         cleanup"
                    );
                }
            }
            // The release has returned, whatever it returned: the
            // retirement's page alone from here on.
            reservation.shrink_to_page();
            reservation
        } else {
            // No size read and no release: the full page bound, for the
            // teardown.
            let Some(reservation) =
                reserve_retirement(layer_store, cancel, object_id, paging, 0).await
            else {
                return Ok(false);
            };
            reservation
        }
    } else {
        tracing::warn!(
            object_id = %object_id,
            "orphan object_id is not a u64; skipping ext-value GC scheduling and proceeding with structural cleanup"
        );
        // The full page bound, for the teardown.
        let Some(reservation) = reserve_retirement(layer_store, cancel, object_id, paging, 0).await
        else {
            return Ok(false);
        };
        reservation
    };

    // The stream's expiry BEFORE the keys go: once the header is gone no pass
    // can find this object again, so a stream left without an expiry then
    // would stay forever. A failure keeps the header for the next pass. A
    // non-numeric id names no stream this service wrote.
    if let Some(oid) = numeric_id {
        if let Err(e) = pexpire_layer_stream(client, keys, oid).await {
            tracing::warn!(
                object_id = %object_id,
                error = %e,
                "could not set the orphan's stream expiry; keeping the orphan for the next sweep"
            );
            return Ok(false);
        }
    }

    // `acquire_maintenance` bumps `valkey_pool_in_use{pool="maintenance"}`
    // for the duration of this per-object teardown (b.mfn). Held alongside
    // the outer `sweep_orphaned_objects` token so the gauge accurately
    // reflects nested maintenance use.
    let (maint_pool, _maint_token) = client.acquire_maintenance();

    // The paged teardown (`storage::retirement::Teardown`), the same as the
    // overwrite body's (`layer.rs::cleanup_old_object_retrying`, which also
    // retries later pages in place) but on the maintenance pool: the first
    // UNLINK holds the header, `obj:ext` and every other fixed key — the
    // header and `obj:ext` always together, so the settle's liveness gate
    // never sees an `obj:ext` without its header on a dead object — then the
    // row pages by `SSCAN` (`obj:secs`, `obj:dels`, `obj:chs`) and by rank
    // (`obj:flds`, `obj:tss`), the five indices last, every UNLINK chunked at
    // `UNLINK_CHUNK_SIZE` on the object's one slot (b.k1r H-stor-7). Each
    // index page `?`-propagates its error (b.j9j item 3: never an empty Vec
    // substituted on a blip); a page that fails transiently after the first
    // UNLINK is retried in place first, since nothing finds the object once
    // its header is gone. The per-object stream (SDD §7.8, §7.11) is in no
    // page: it expires on its own (just above), so a subscriber lagging
    // between `XREAD` round trips still reads the `deleted` entry the wake-up
    // appended.
    let teardown = Teardown::new(maint_pool, keys, object_id, paging)
        .reserved(Some(&reservation))
        .retrying_later_pages();
    #[cfg(any(test, feature = "test-fault-injection"))]
    let teardown = teardown.with_faults(None, layer_store.retirement_teardown_faults());
    teardown.run().await.map_err(|failure: TeardownFailure| {
        // The failed command as every route reports it
        // (`TeardownFailure::label`: `<op> on <key>`): the teardown's own
        // pages count nothing, so for a foreign index this names the key it
        // stopped at (the size read above, when it ran, counted it under its
        // family), and the caller's WARN names the object.
        SweepError::from_fred(
            FredErrOrigin::metrics(metrics, POOL_LABEL_MAINTENANCE, client.store_oom_signal()),
            failure.label(),
            failure.error,
        )
    })?;

    Ok(true)
}

/// Reserve one orphan's retirement charge — at `page`, owing `references`
/// references (`storage::retirement::retirement_charge_bytes`) — on
/// `layer_store`'s retirement allowance, the wait raced against the sweep's
/// `cancel`. `None` when the sweep was cancelled first, or shutdown ended the
/// allowance's waits while this one was queued or would have had to wait:
/// the orphan is kept, its header for the next pass.
async fn reserve_retirement(
    layer_store: &LayerStore,
    cancel: &CancellationToken,
    object_id: &str,
    page: RetirementPaging,
    references: u64,
) -> Option<Arc<RetirementReservation>> {
    let reserved = tokio::select! {
        reserved = RetirementReservation::reserve(
            layer_store.retirement_budget(),
            layer_store.retirement_paging(),
            page,
            references,
        ) => reserved.ok(),
        () = cancel.cancelled() => None,
    };
    if reserved.is_none() {
        tracing::debug!(
            object_id = %object_id,
            "the orphan sweep stopped while the orphan's clean-up waited for the retirement \
             allowance; keeping the orphan for the next sweep"
        );
    }
    reserved
}

/// The retirement body's wake-up: the shared retirement wake-up
/// ([`wake_retired_object_stream`]: `PEXPIRE`, `XADD * payload deleted`,
/// `PEXPIRE` on `le:{oid}:obj:stream`, in one script that runs on a full
/// store), so a subscriber still tailing the orphan's stream re-reads the
/// mapping and ends (or moves on). A stream that is already gone is
/// re-created with the `deleted` entry and an expiry, which wakes a
/// subscriber still blocked on it.
///
/// `true` when the retirement may go on: the entry landed, or the store
/// refused it because it is out of memory (which a store that follows the
/// opener rule, SDD §7.5.1, never does) — deleting keys is what frees a full
/// store (WARN). `false` for any other failure (WARN): the caller keeps the
/// header for the next pass.
async fn wake_orphan_stream(
    client: &ValkeyClient,
    keys: &KeySchema,
    object_id: u64,
    metrics: Option<&SharedMetrics>,
) -> bool {
    let Err(e) = wake_retired_object_stream(client, keys, object_id).await else {
        return true;
    };
    // Counts a command timeout and notes a store write refused `-OOM` on the
    // store out-of-memory signal (b.pa1): the out-of-memory arm below retires
    // the orphan anyway, and the GC's out-of-memory collection is requested
    // beside it.
    FredErrOrigin::metrics(metrics, POOL_LABEL_DATA, client.store_oom_signal()).observe_error(&e);
    if matches!(
        store_write_refusal(&e),
        Some(StoreWriteRefusal::OutOfMemory)
    ) {
        tracing::warn!(
            object_id,
            error = %e,
            "the store refused the orphan's stream wake-up because it is out of memory; \
             retiring the orphan anyway, since deleting keys is what frees the store \
             (subscribers still tailing its stream are not woken)"
        );
        return true;
    }
    tracing::warn!(
        object_id,
        error = %e,
        "could not wake subscribers still tailing the orphan's stream; keeping the orphan for \
         the next sweep"
    );
    false
}

/// Extract the ID inside `{...}` hash tag from a Valkey key.
///
/// Given key `"le:{some-id}:obj:hdr"`, prefix `"le:"`, and suffix `":obj:hdr"`,
/// returns `Some("some-id")`.
///
/// For layer keys the suffix is `":layer:"` which matches up to (but not
/// including) the hex layer name portion.
pub(crate) fn extract_hash_tag_id(key: &str, prefix: &str, suffix_start: &str) -> Option<String> {
    let rest = key.strip_prefix(prefix)?;
    // Expect `{id}:remainder`
    let rest = rest.strip_prefix('{')?;
    let close = rest.find('}')?;
    let id = &rest[..close];
    // Verify the part after `}` starts with our expected suffix.
    let after = &rest[close + 1..];
    if after.starts_with(suffix_start) {
        Some(id.to_owned())
    } else {
        None
    }
}

/// Non-fatal error type for the orphan sweep.
///
/// Pre-b.kf1 every variant collapsed into a single `String` payload, which
/// prevented operators from telling a Valkey timeout apart from a protocol
/// error in metrics + logs (b.kf1 medium). The structured form preserves
/// the underlying fred `ErrorKind` slug via [`SweepError::kind_label`] so
/// the orphan-sweep error counter can break down by kind without losing
/// the human-readable context that was already useful.
#[derive(Debug, thiserror::Error)]
pub enum SweepError {
    #[error("orphan sweep error: {op} failed ({kind}): {source}")]
    Fred {
        /// The failing command's label; a teardown page's also names the key
        /// it addressed.
        op: std::borrow::Cow<'static, str>,
        kind: &'static str,
        #[source]
        source: fred::error::Error,
    },
}

impl SweepError {
    /// Construct a `SweepError::Fred` from a fred error, capturing the
    /// underlying `ErrorKind` slug for triage. `op` is a short
    /// human-readable label for the failing Valkey command (a `&'static str`
    /// or, where the key matters to the operator, an owned label naming it).
    ///
    /// `origin` counts a command timeout (see
    /// [`FredErrOrigin`](crate::storage::client::FredErrOrigin)); it never
    /// changes the returned error.
    pub fn from_fred(
        origin: FredErrOrigin<'_>,
        op: impl Into<std::borrow::Cow<'static, str>>,
        source: fred::error::Error,
    ) -> Self {
        origin.observe_error(&source);
        SweepError::Fred {
            op: op.into(),
            kind: source.kind().to_str(),
            source,
        }
    }

    /// Short label for the `gc_sweep_batch_errors_total{kind}` metric.
    pub fn kind_label(&self) -> &'static str {
        match self {
            SweepError::Fred { kind, .. } => kind,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fred::interfaces::SortedSetsInterface;

    #[test]
    fn extract_hash_tag_id_obj_hdr_default_prefix() {
        let id = extract_hash_tag_id("le:{42}:obj:hdr", "le:", ":obj:hdr");
        assert_eq!(id.as_deref(), Some("42"));
    }

    #[test]
    fn extract_hash_tag_id_obj_hdr_custom_prefix() {
        let id = extract_hash_tag_id("test_abc:{99}:obj:hdr", "test_abc:", ":obj:hdr");
        assert_eq!(id.as_deref(), Some("99"));
    }

    #[test]
    fn extract_hash_tag_id_layer_mapping() {
        let id = extract_hash_tag_id("le:{my-session}:layer:726f6f74", "le:", ":layer:");
        assert_eq!(id.as_deref(), Some("my-session"));
    }

    #[test]
    fn extract_hash_tag_id_uuid_session() {
        let id = extract_hash_tag_id(
            "le:{a1b2c3d4-e5f6-7890-abcd-ef0123456789}:layer:ff",
            "le:",
            ":layer:",
        );
        assert_eq!(id.as_deref(), Some("a1b2c3d4-e5f6-7890-abcd-ef0123456789"));
    }

    #[test]
    fn extract_hash_tag_id_wrong_prefix_returns_none() {
        let id = extract_hash_tag_id("le:{42}:obj:hdr", "other:", ":obj:hdr");
        assert!(id.is_none());
    }

    #[test]
    fn extract_hash_tag_id_wrong_suffix_returns_none() {
        let id = extract_hash_tag_id("le:{42}:obj:hdr", "le:", ":layer:");
        assert!(id.is_none());
    }

    #[test]
    fn extract_hash_tag_id_no_braces_returns_none() {
        let id = extract_hash_tag_id("le:42:obj:hdr", "le:", ":obj:hdr");
        assert!(id.is_none());
    }

    #[test]
    fn extract_hash_tag_id_empty_id_returns_some_empty() {
        let id = extract_hash_tag_id("le:{}:obj:hdr", "le:", ":obj:hdr");
        assert_eq!(id.as_deref(), Some(""));
    }

    // ══ b.yda phase 3 — the retirement release-retry drain ══════════════════
    //
    // Valkey-backed, because every step of this phase is a Lua script whose
    // whole value is atomicity — a stub would be asserting the test's own model
    // of the script rather than the script. In-crate rather than in
    // `tests/integration/storage/` because `sweep_retire_retry`,
    // `drain_retire_carrier` and `SweepResult`'s construction are private, and
    // the two counters that matter (`retire_carriers_claimed` /
    // `_cancelled`) are only readable from a `SweepResult` this module builds.
    //
    // The tests in this section seed their carrier and work-list entry by hand:
    // this build's HASH carrier with NO attempt tokens, which the drain settles
    // without a `{gc}` resolve (nothing recorded to resolve). That is what lets
    // the settle script's liveness gate be reached deterministically. The
    // write-ahead intent the current retirement writes (begin, attempt tokens,
    // clear) is driven for real in the b.yuo Part R section below, and an older
    // build's SET carrier — never paid, cleared or dropped (b.1jt) — in the
    // b.1jt section after it.

    use crate::observability::metrics::test_util::{render, sample_value};
    use crate::observability::metrics::{metrics_scrape_only, Metrics};
    use crate::storage::external_values::{recon_register, ExtRelease};
    use crate::storage::test_support::TestValkey;

    /// A distinct content hash, so two carrier members are tellable apart.
    fn carrier_hash(byte: u8) -> [u8; 32] {
        [byte; 32]
    }

    /// The formatted `le:ext:{gc}:pending` member a retirement owes for one
    /// hash — exactly what the deferral parks on the carrier, byte for byte.
    fn carrier_member(byte: u8, object_id: u64) -> String {
        ExtRelease::retirement(carrier_hash(byte), object_id).member()
    }

    /// Park `members` on `le:{oid}:obj:extret` as this build writes a carrier
    /// — a HASH, member → `1` (b.1jt) — and register the id on the `{gc}`-slot
    /// work list, with no attempt tokens. This build's begin always records a
    /// token beside its carrier; leaving it out is a fixture shortcut that
    /// reaches the settle script with nothing recorded to resolve. An older
    /// build's carrier is a SET, which the drain never pays — see
    /// [`seed_legacy_set_carrier`].
    async fn seed_carrier(tv: &TestValkey, object_id: &str, members: &[String]) {
        let keys = tv.key_schema();
        let fields: Vec<(String, String)> = members
            .iter()
            .map(|m| (m.clone(), "1".to_owned()))
            .collect();
        let _: () = tv
            .client()
            .pool
            .hset(keys.obj_ext_retire_carrier(object_id), fields)
            .await
            .expect("seeding the carrier must succeed");
        recon_register(
            tv.client(),
            &keys.ext_retire_retry(),
            std::slice::from_ref(&object_id.to_owned()),
        )
        .await
        .expect("registering the work-list entry must succeed");
    }

    /// The members of this build's HASH carrier, sorted (empty when the key is
    /// absent). A SET at the key fails the read `WRONGTYPE`: no test of this
    /// build's carrier may find an older build's type there.
    async fn carrier_members(tv: &TestValkey, object_id: &str) -> Vec<String> {
        let mut m: Vec<String> = tv
            .client()
            .pool
            .hkeys(tv.key_schema().obj_ext_retire_carrier(object_id))
            .await
            .expect("HKEYS carrier must succeed");
        m.sort();
        m
    }

    /// Park `members` on `le:{oid}:obj:extret` as an older build (0.1.0
    /// through 0.2.0-rc.48) wrote a carrier — a SET — with `tokens` recorded on
    /// `le:{oid}:obj:extret:attempts` (none for the untokened carriers most
    /// older builds leave), and register the id on the work list.
    async fn seed_legacy_set_carrier(
        tv: &TestValkey,
        object_id: &str,
        members: &[String],
        tokens: &[&str],
    ) {
        let keys = tv.key_schema();
        let _: () = tv
            .client()
            .pool
            .sadd(keys.obj_ext_retire_carrier(object_id), members.to_vec())
            .await
            .expect("seeding the older build's SET carrier must succeed");
        for token in tokens {
            let _: () = tv
                .client()
                .pool
                .hset(keys.obj_ext_retire_attempts(object_id), [(*token, "1")])
                .await
                .expect("seeding the older build's attempt token must succeed");
        }
        recon_register(
            tv.client(),
            &keys.ext_retire_retry(),
            std::slice::from_ref(&object_id.to_owned()),
        )
        .await
        .expect("registering the work-list entry must succeed");
    }

    /// Whether the carrier key exists, of any type.
    async fn carrier_exists(tv: &TestValkey, object_id: &str) -> bool {
        tv.client()
            .pool
            .exists(tv.key_schema().obj_ext_retire_carrier(object_id))
            .await
            .expect("EXISTS carrier must succeed")
    }

    async fn work_list_entries(tv: &TestValkey) -> Vec<String> {
        let mut m: Vec<String> = tv
            .client()
            .pool
            .zrange(
                tv.key_schema().ext_retire_retry(),
                0,
                -1,
                None,
                false,
                None,
                false,
            )
            .await
            .expect("ZRANGE work list must succeed");
        m.sort();
        m
    }

    async fn pending(tv: &TestValkey) -> Vec<String> {
        let mut m: Vec<String> = tv
            .client()
            .pool
            .smembers(tv.key_schema().ext_pending())
            .await
            .expect("SMEMBERS pending must succeed");
        m.sort();
        m
    }

    async fn ext_gc_scheduled(tv: &TestValkey, object_id: &str) -> Option<String> {
        tv.client()
            .pool
            .hget(tv.key_schema().obj_header(object_id), "ext_gc_scheduled")
            .await
            .expect("HGET ext_gc_scheduled must succeed")
    }

    /// The attempt tokens recorded on `le:{oid}:obj:extret:attempts`, sorted.
    async fn attempt_tokens(tv: &TestValkey, object_id: &str) -> Vec<String> {
        let mut t: Vec<String> = tv
            .client()
            .pool
            .hkeys(tv.key_schema().obj_ext_retire_attempts(object_id))
            .await
            .expect("HKEYS obj:extret:attempts must succeed");
        t.sort();
        t
    }

    /// Every `{gc}` enqueue-attempt record, `token → landed|void:<ms>:<oid>`.
    async fn attempt_records(tv: &TestValkey) -> std::collections::HashMap<String, String> {
        tv.client()
            .pool
            .hgetall(tv.key_schema().ext_enqueue_attempts())
            .await
            .expect("HGETALL enqueue_attempt must succeed")
    }

    /// An unlabelled counter, panicking on an absent series: every one of these
    /// is zero-seeded in `Metrics::new`, so absence means the sweep under test
    /// was handed a different handle and every count read off it is vacuous.
    fn counter(metrics: &Metrics, name: &str) -> u64 {
        sample_value(&render(metrics), name, &[]).unwrap_or_else(|| {
            panic!(
                "{name} is absent from the exposition — it is zero-seeded at construction, so \
                    this can only be a different `Metrics`"
            )
        }) as u64
    }

    /// A dead object's carrier is paid: its members land on `pending` exactly
    /// once, and the carrier, its token HASH, the drain's own `{gc}` attempt
    /// record and the work-list entry are all gone.
    ///
    /// The happy path the whole phase exists for: a retirement whose `{gc}`-slot
    /// enqueue failed left its obligation on the object's own slot, and this
    /// is where the obligation is finally paid. `retire_carriers_claimed` counts
    /// CARRIERS while `retire_releases_recovered` and the counter count
    /// REFERENCES, which is why the fixture owes two hashes: a build that
    /// conflated the two units reads 1 where the reference count must read 2.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn sweep_retire_retry_drains_a_dead_objects_carrier_onto_pending() {
        const OID: &str = "930001";
        let tv = TestValkey::setup().await;
        let metrics = Arc::new(metrics_scrape_only("yda-drain"));
        let members = vec![carrier_member(0xA1, 930_001), carrier_member(0xA2, 930_001)];
        seed_carrier(&tv, OID, &members).await;

        let mut result = SweepResult::default();
        sweep_retire_retry(
            tv.client(),
            tv.key_schema(),
            &CancellationToken::new(),
            &mut result,
            Some(&(metrics.clone() as SharedMetrics)),
            "yda-drain",
        )
        .await
        .expect("the retire-retry phase must not fail on a healthy store");

        let mut expected = members.clone();
        expected.sort();
        assert_eq!(
            pending(&tv).await,
            expected,
            "the claimed members must be re-enqueued VERBATIM. Re-deriving them would mean \
             parsing text this service wrote back into its three fields only to re-encode it \
             identically; byte-identity is what makes a duplicate drain collapse in the Set \
             rather than owe a second decrement"
        );
        assert_eq!(
            result.retire_carriers_claimed, 1,
            "one carrier, counted in CARRIERS"
        );
        assert_eq!(
            result.retire_releases_recovered, 2,
            "two obligations, counted in REFERENCES — the unit every counter in this family \
             shares, and the one an operator's \
             `deferred ~= recovered + cancelled + already_enqueued` identity needs"
        );
        assert_eq!(
            counter(&metrics, "live_edit_ext_ref_release_recovered_total"),
            2,
            "the recovered counter must move by the reference count, matching the SweepResult"
        );
        assert_eq!(
            counter(&metrics, "live_edit_ext_ref_release_enqueue_failed_total"),
            0,
            "nothing was lost, so the leak signal must be still. The drain has no tick site of \
             its own: it never removes a member before an attempt is known landed"
        );
        assert!(
            carrier_members(&tv, OID).await.is_empty() && attempt_tokens(&tv, OID).await.is_empty(),
            "once the re-drive landed, the clear deletes the carrier and its token HASH together"
        );
        assert!(
            attempt_records(&tv).await.is_empty(),
            "the drain's own token was sent once and answered cleanly and its carrier was \
             cleared, so its `{{gc}}` record is removed at once (F1); a record left here would sit \
             for the attempt audit's 24 h on every recovered carrier"
        );
        assert!(
            work_list_entries(&tv).await.is_empty(),
            "the work-list entry is retired once the members are safely on pending"
        );
    }

    /// A carrier whose `object_id` is LIVE AGAIN is cancelled, not drained: the
    /// members are dropped, the carrier is removed, and `ext_gc_scheduled` is
    /// cleared off the surviving header.
    ///
    /// This is the over-release gate. A live incarnation — a checkpoint restore
    /// at the same id, or a teardown that never completed — has inherited the
    /// standing, unreleased `+1` per hash those members stand for, so paying
    /// them would be an unmatched decrement: `ext_meta` and `ext_data` deleted
    /// under a live reference. Cancelling leaks in the worst case, and the
    /// reference-count reconciler drains a leak.
    ///
    /// The `HDEL` is asserted because it is load-bearing rather than tidy: left
    /// in place, the flag would short-circuit the object's eventual REAL
    /// retirement and convert the over-release this gate prevents into a
    /// standing leak.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn sweep_retire_retry_cancels_a_carrier_whose_object_is_live_again() {
        const OID: &str = "930002";
        let tv = TestValkey::setup().await;
        let metrics = Arc::new(metrics_scrape_only("yda-cancel"));
        let members = vec![carrier_member(0xB1, 930_002), carrier_member(0xB2, 930_002)];
        seed_carrier(&tv, OID, &members).await;

        // The live incarnation: a header carrying the flag its retirement set.
        let _: () = tv
            .client()
            .pool
            .hset(
                tv.key_schema().obj_header(OID),
                [("seq", "1"), ("ext_gc_scheduled", "1")],
            )
            .await
            .expect("seeding the surviving header must succeed");

        let mut result = SweepResult::default();
        sweep_retire_retry(
            tv.client(),
            tv.key_schema(),
            &CancellationToken::new(),
            &mut result,
            Some(&(metrics.clone() as SharedMetrics)),
            "yda-cancel",
        )
        .await
        .expect("the retire-retry phase must not fail");

        assert!(
            pending(&tv).await.is_empty(),
            "NOTHING may be enqueued under the cancelled verdict: the live incarnation inherited \
             the references these members stand for"
        );
        assert_eq!(result.retire_carriers_cancelled, 1);
        assert_eq!(
            result.retire_carriers_claimed, 0,
            "a cancel is not a claim — conflating them would report the obligation as paid"
        );
        assert_eq!(
            counter(&metrics, "live_edit_ext_ref_release_cancelled_total"),
            2,
            "counted in REFERENCES, like its two siblings, even though the SweepResult field \
             beside it counts carriers"
        );
        assert!(
            carrier_members(&tv, OID).await.is_empty() && attempt_tokens(&tv, OID).await.is_empty(),
            "the cancel deletes the carrier and the drain's token HASH in one step"
        );
        assert_eq!(
            ext_gc_scheduled(&tv, OID).await,
            None,
            "the flag must be CLEARED off the surviving header. Left set, the object's eventual \
             real retirement short-circuits on it and releases nothing — turning the \
             over-release this gate prevents into a standing leak that only the reference-count \
             reconciler's hourly pass would ever drain"
        );
        assert!(
            work_list_entries(&tv).await.is_empty(),
            "the entry is retired either way; re-visiting would only re-run the gate against a \
             live object forever"
        );
    }

    /// `obj:ext` alone — no header — is ALSO "live again".
    ///
    /// This is the restore residue-wipe window: `checkpoint::residue` UNLINKs
    /// `obj:hdr` but deliberately WITHHOLDS `obj:ext`, because the DR-restore
    /// materialiser (the `SurplusPolicy::Suppress` entry point in
    /// `storage::layer`) reads that index as the `before` image for the
    /// `rewrite` releases it owes. A settle landing in that window must still
    /// cancel — the standing `+1` is inherited by that materialiser's rewrite
    /// pass either way.
    ///
    /// `obj:ext` standing without a header never means "dead": on a genuinely
    /// dead object both keys go in ONE `UNLINK`, the teardown's first. Without
    /// this key in the
    /// gate the settle would answer `pay`, the drain would enqueue, and the GC
    /// would decrement against the object the restore is in the middle of
    /// rebuilding.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn sweep_retire_retry_cancels_on_a_surviving_ext_index_with_no_header() {
        const OID: &str = "930003";
        let tv = TestValkey::setup().await;
        let metrics = Arc::new(metrics_scrape_only("yda-wipe-window"));
        let members = vec![carrier_member(0xC1, 930_003)];
        seed_carrier(&tv, OID, &members).await;

        // The wipe window: index present, header absent.
        let _: () = tv
            .client()
            .pool
            .hset(
                tv.key_schema().obj_ext_index(OID),
                [(carrier_hash(0xC1).map(|b| format!("{b:02x}")).concat(), "1")],
            )
            .await
            .expect("seeding obj:ext must succeed");

        let mut result = SweepResult::default();
        sweep_retire_retry(
            tv.client(),
            tv.key_schema(),
            &CancellationToken::new(),
            &mut result,
            Some(&(metrics.clone() as SharedMetrics)),
            "yda-wipe-window",
        )
        .await
        .expect("the retire-retry phase must not fail");

        assert!(
            pending(&tv).await.is_empty(),
            "the restore is mid-flight and the index it left standing is the evidence. \
             Enqueuing here decrements against the object the DR-restore materialiser is about \
             to finish rebuilding"
        );
        assert_eq!(result.retire_carriers_cancelled, 1);
        assert_eq!(
            counter(&metrics, "live_edit_ext_ref_release_cancelled_total"),
            1
        );
        assert!(carrier_members(&tv, OID).await.is_empty());
    }

    /// b.fsz MR-D4 (D4-T7): a retirement whose paged teardown stopped right
    /// after its first `UNLINK` leaves no `obj:hdr` and no `obj:ext` — the two
    /// go in that one command (`KeySchema::object_teardown_first_keys`) — so a
    /// carrier on the object settles `pay`, not `cancelled`, though the rows
    /// and indices the later pages would have removed still stand. Run on the
    /// overwrite body's posture (data pool, later pages retried in place) and
    /// the sweep's (maintenance pool), each stopped by a permanent failure of
    /// the first command after the first `UNLINK`, which no posture retries.
    /// A teardown that removed the header before `obj:ext` would leave the
    /// index standing alone, which the previous test shows cancels.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn sweep_retire_retry_pays_a_carrier_whose_object_stopped_after_its_teardowns_first_unlink(
    ) {
        for (oid, object_id, overwrite_posture) in
            [("930101", 930_101u64, true), ("930102", 930_102, false)]
        {
            let tv = TestValkey::setup().await;
            let metrics = Arc::new(metrics_scrape_only("fsz-first-unlink"));
            let keys = tv.key_schema();
            let pool = &tv.client().pool;
            let _: () = pool
                .hset(keys.obj_header(oid), [("seq", "1")])
                .await
                .expect("seeding the header must succeed");
            let _: () = pool
                .hset(
                    keys.obj_ext_index(oid),
                    [(carrier_hash(0xD1).map(|b| format!("{b:02x}")).concat(), "1")],
                )
                .await
                .expect("seeding obj:ext must succeed");
            let _: () = pool
                .sadd(keys.obj_sections_index(oid), "5")
                .await
                .expect("seeding obj:secs must succeed");
            let _: () = pool
                .hset(keys.obj_section(oid, "5"), [("name", "s5")])
                .await
                .expect("seeding the section must succeed");
            let members = vec![carrier_member(0xD1, object_id)];
            seed_carrier(&tv, oid, &members).await;

            let faults: crate::storage::retirement::RetirementFaults =
                Arc::new(std::sync::Mutex::new(std::collections::VecDeque::from([
                    Some(fred::error::Error::new(
                        fred::error::ErrorKind::Protocol,
                        "injected permanent page failure",
                    )),
                ])));
            let paging = crate::storage::retirement::RetirementPaging::for_allowance(5 << 20);
            let (maint_pool, _maint_token) = tv.client().acquire_maintenance();
            let failure = if overwrite_posture {
                Teardown::new(pool, keys, oid, paging)
                    .retrying_later_pages()
                    .with_faults(None, Some(faults))
                    .run()
                    .await
            } else {
                Teardown::new(maint_pool, keys, oid, paging)
                    .retrying_later_pages()
                    .with_faults(None, Some(faults))
                    .run()
                    .await
            }
            .expect_err("the injected permanent failure stops the teardown");
            assert_eq!(
                failure.op, "SSCAN obj sections index",
                "fixture: the first command after the first UNLINK failed"
            );
            let exists = |key: String| async move {
                pool.exists::<u64, _>(key)
                    .await
                    .expect("EXISTS must succeed")
                    == 1
            };
            assert!(
                !exists(keys.obj_header(oid)).await && !exists(keys.obj_ext_index(oid)).await,
                "the first UNLINK removed the header and obj:ext together"
            );
            assert!(
                exists(keys.obj_sections_index(oid)).await,
                "fixture: the teardown stopped before its later pages"
            );

            let mut result = SweepResult::default();
            sweep_retire_retry(
                tv.client(),
                keys,
                &CancellationToken::new(),
                &mut result,
                Some(&(metrics.clone() as SharedMetrics)),
                "fsz-first-unlink",
            )
            .await
            .expect("the retire-retry phase must not fail");

            assert_eq!(
                (
                    result.retire_carriers_claimed,
                    result.retire_carriers_cancelled
                ),
                (1, 0),
                "posture {}: neither the header nor obj:ext stands, so the carrier is paid",
                if overwrite_posture {
                    "overwrite"
                } else {
                    "sweep"
                }
            );
            assert_eq!(pending(&tv).await, members, "paid once, verbatim");
        }
    }

    /// A live header with the flag and **no carrier** is a no-op visit: the
    /// entry is retired, the flag SURVIVES, and no handover counter moves.
    ///
    /// The round-3 B2 blocker, pinned. Every `recon_retire` failure arm in
    /// `drain_retire_carrier` leaves a work-list entry registered after the
    /// carrier is already gone, so the next tick re-visits it. There is nothing
    /// to hand back on such a visit — while the `ext_gc_scheduled=1` on the live
    /// header may well belong to a retirement whose enqueue SUCCEEDED. Clearing
    /// it there re-arms `schedule_old_object_ext_gc` to re-derive the same
    /// members and enqueue them a SECOND time: a second decrement against a
    /// reference already given up, once the GC has drained the first. That is
    /// the same over-release the gate exists to refuse, reached from the
    /// direction the gate does not look at.
    ///
    /// **What this pins now.** Since b.yuo Part R the drain's re-token answers
    /// `empty` over a missing carrier, and the drain's `empty` arm retires the
    /// work-list entry and writes nothing: the settle, the only writer of the
    /// flag's `HDEL`, never runs here. So this test pins that arm — the flag
    /// survives, no counter of the obligation's handover moves (cancelled,
    /// recovered, already enqueued) and no result field counting them does
    /// (carriers cancelled or claimed, releases recovered or already
    /// enqueued), nothing is enqueued, and the entry is retired (an `empty`
    /// arm that kept it fails the work-list assertion). The
    /// settle's own guard against a missing carrier, which answers `changed`
    /// and writes nothing, is pinned by RT-13
    /// (`a_lone_token_hash_is_left_in_place_by_the_re_token_the_clear_and_the_settle`).
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn sweep_retire_retry_leaves_the_flag_alone_when_the_carrier_is_already_gone() {
        const OID: &str = "930004";
        let tv = TestValkey::setup().await;
        let metrics = Arc::new(metrics_scrape_only("yda-stale-entry"));

        // A work-list entry with NO carrier behind it, over a live header that
        // carries the flag from a retirement whose enqueue succeeded.
        recon_register(
            tv.client(),
            &tv.key_schema().ext_retire_retry(),
            std::slice::from_ref(&OID.to_owned()),
        )
        .await
        .expect("registering the stale entry must succeed");
        let _: () = tv
            .client()
            .pool
            .hset(
                tv.key_schema().obj_header(OID),
                [("seq", "1"), ("ext_gc_scheduled", "1")],
            )
            .await
            .expect("seeding the live header must succeed");

        let mut result = SweepResult::default();
        sweep_retire_retry(
            tv.client(),
            tv.key_schema(),
            &CancellationToken::new(),
            &mut result,
            Some(&(metrics.clone() as SharedMetrics)),
            "yda-stale-entry",
        )
        .await
        .expect("the retire-retry phase must not fail");

        assert_eq!(
            ext_gc_scheduled(&tv, OID).await,
            Some("1".to_owned()),
            "no carrier means no recorded obligation to cancel, so the flag is not this drain's \
             to touch. Clearing it buys a SECOND enqueue of members the GC may already have \
             drained — the over-release, from the other direction"
        );
        assert_eq!(
            (
                result.retire_carriers_cancelled,
                result.retire_carriers_claimed
            ),
            (0, 0),
            "a no-op visit is neither a cancellation nor a claim: no obligation changed hands"
        );
        assert_eq!(
            (
                result.retire_releases_recovered,
                result.retire_releases_already_enqueued
            ),
            (0, 0),
            "nor a recovery or an already-enqueued answer: no reference changed hands"
        );
        for name in [
            "live_edit_ext_ref_release_cancelled_total",
            "live_edit_ext_ref_release_recovered_total",
            "live_edit_ext_ref_release_already_enqueued_total",
        ] {
            assert_eq!(
                counter(&metrics, name),
                0,
                "and no counter may move: {name} — a tick here would report a handover that did \
                 not happen and make the `deferred ~= recovered + cancelled + already_enqueued` \
                 identity unbalanced"
            );
        }
        assert!(
            work_list_entries(&tv).await.is_empty(),
            "the stale entry IS retired, so it does not cost an EVALSHA every tick forever"
        );
        assert!(pending(&tv).await.is_empty());
    }

    /// A drain whose re-drive fails leaves every member AND its own token on
    /// the carrier and the work-list entry registered; the next tick resolves
    /// that token void and pays the members once.
    ///
    /// Since b.yuo Part R nothing in the drain removes a member before an
    /// attempt is known landed — the re-token only adds a token — so there is
    /// no put-back to get wrong: the failed pay's token stays recorded, and
    /// the next drain must ask `{gc}` about it before sending again. The leak
    /// signal must NOT tick: nothing was lost, and a tick would page an
    /// operator for a state that self-heals.
    ///
    /// The enqueue is broken by planting a STRING at `le:ext:{gc}:pending` — a
    /// standing `WRONGTYPE`, not a blip, and a real fred error surfacing from a
    /// real Valkey rather than a mocked one. The attempt-mode script fails on
    /// its first write, so the token is never recorded `landed`. The plant
    /// cannot disturb anything else this phase touches: the carrier, its tokens
    /// and the header are on the `{object_id}` slot, and the work list and the
    /// attempt record are other `{gc}`-slot keys.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn sweep_retire_retry_keeps_the_obligation_and_its_token_on_the_carrier_when_the_re_drive_fails(
    ) {
        const OID: &str = "930005";
        let tv = TestValkey::setup().await;
        let metrics = Arc::new(metrics_scrape_only("yda-redrive-fails"));
        let members = vec![carrier_member(0xD1, 930_005), carrier_member(0xD2, 930_005)];
        seed_carrier(&tv, OID, &members).await;
        tv.client()
            .set_retire_intent_in_progress_age_for_test(Duration::ZERO);

        let pending_key = tv.key_schema().ext_pending();
        let _: () = tv
            .client()
            .pool
            .set(
                &pending_key,
                "planted-by-an-out-of-band-writer",
                None,
                None,
                false,
            )
            .await
            .expect("planting a string over the pending set must succeed");

        let mut result = SweepResult::default();
        sweep_retire_retry(
            tv.client(),
            tv.key_schema(),
            &CancellationToken::new(),
            &mut result,
            Some(&(metrics.clone() as SharedMetrics)),
            "yda-redrive-fails",
        )
        .await
        .expect("a per-object failure must not abort the phase");

        let mut expected = members.clone();
        expected.sort();
        assert_eq!(
            carrier_members(&tv, OID).await,
            expected,
            "every member must still be on the carrier byte-for-byte: the drain removes nothing \
             before an attempt is known landed"
        );
        let tokens = attempt_tokens(&tv, OID).await;
        assert_eq!(
            tokens.len(),
            1,
            "the failed pay's token stays recorded on the carrier, so the next drain resolves it \
             before it sends anything again: {tokens:?}"
        );
        assert!(
            attempt_records(&tv).await.is_empty(),
            "the script failed on its first write, so no `{{gc}}` record says it landed"
        );
        assert_eq!(
            work_list_entries(&tv).await,
            vec![OID.to_owned()],
            "and the entry stays registered, so the next tick looks again without waiting for \
             the keyspace backstop"
        );
        assert_eq!(
            counter(&metrics, "live_edit_ext_ref_release_enqueue_failed_total"),
            0,
            "nothing was lost. The leak signal is alertable at `increase(...) > 0`, so ticking \
             it for a state that self-heals on the next tick pages an operator for a recovery in \
             progress"
        );
        assert_eq!(
            counter(&metrics, "live_edit_ext_ref_release_recovered_total"),
            0,
            "and nothing was recovered either — the obligation is exactly where it started"
        );

        // The store recovers: the next tick resolves the failed pay's token
        // void and pays the members once.
        let _: u64 = tv
            .client()
            .pool
            .del(&pending_key)
            .await
            .expect("DEL of the planted key must succeed");
        let mut result = SweepResult::default();
        sweep_retire_retry(
            tv.client(),
            tv.key_schema(),
            &CancellationToken::new(),
            &mut result,
            Some(&(metrics.clone() as SharedMetrics)),
            "yda-redrive-fails",
        )
        .await
        .expect("the retire-retry phase must not fail on a healthy store");
        assert_eq!(
            pending(&tv).await,
            expected,
            "the second tick pays the members"
        );
        assert_eq!(
            counter(&metrics, "live_edit_ext_ref_release_recovered_total"),
            2
        );
        let records = attempt_records(&tv).await;
        assert!(
            records
                .get(&tokens[0])
                .is_some_and(|value| value.starts_with("void:")),
            "the failed pay's token was resolved void before the second pay was sent, so a late \
             copy of it can never land: {records:?}"
        );
        assert!(
            carrier_members(&tv, OID).await.is_empty() && attempt_tokens(&tv, OID).await.is_empty()
        );
    }

    /// Sink for [`capture_warns`].
    #[derive(Clone)]
    struct CapturingWriter(Arc<std::sync::Mutex<Vec<u8>>>);

    impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for CapturingWriter {
        type Writer = CapturingWriter;
        fn make_writer(&'a self) -> Self::Writer {
            self.clone()
        }
    }

    impl std::io::Write for CapturingWriter {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.0
                .lock()
                .expect("capture buffer poisoned")
                .extend_from_slice(buf);
            Ok(buf.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    /// Drive `future` under a capturing JSON subscriber and return its value
    /// alongside every event it logged at WARN or above, parsed.
    ///
    /// `WithSubscriber` rather than `set_default`: this future awaits Valkey I/O
    /// on a multi-thread runtime and may resume on a different worker thread, and
    /// a thread-local dispatcher would silently stop capturing at the first
    /// `.await` — the failure mode that makes a log assertion read as "the WARN
    /// was never emitted". Same shape as `checkpoint::assembler::tests` and
    /// `checkpoint::adapters::local_disk::tests`.
    ///
    /// **JSON, per the test-writing guide's log-capture carve-out**, and the
    /// exemplar is `tests/integration/grpc/rpc_failure_log_test.rs`. A rendered
    /// line reduces every structural question to a substring search, and the
    /// substring answers the wrong one in both directions: `warns.contains(
    /// "restore_error=")` is satisfied by a field the event carries EMPTY — which
    /// is exactly the regression worth catching, since an empty `restore_error`
    /// means the operator learns the obligation was lost and nothing about why —
    /// and `warns.contains("count=2")` would be satisfied by any other field
    /// whose rendering ends that way. In JSON an absent field is a missing key, a
    /// present-but-empty one is `""`, and the two are distinguishable.
    async fn capture_warns<F, T>(future: F) -> (T, Vec<serde_json::Value>)
    where
        F: std::future::Future<Output = T>,
    {
        use tracing::instrument::WithSubscriber;
        let buf = Arc::new(std::sync::Mutex::new(Vec::new()));
        let subscriber = tracing_subscriber::fmt()
            .json()
            .with_writer(CapturingWriter(buf.clone()))
            .with_max_level(tracing::Level::WARN)
            .with_ansi(false)
            .finish();
        let value = future
            .with_subscriber(tracing::Dispatch::new(subscriber))
            .await;
        let text =
            String::from_utf8_lossy(&buf.lock().expect("capture buffer poisoned")).into_owned();
        let events = text
            .lines()
            .filter(|line| !line.trim().is_empty())
            .map(|line| {
                serde_json::from_str(line)
                    .unwrap_or_else(|e| panic!("a captured line is not JSON: {e}\n{line}"))
            })
            .collect();
        (value, events)
    }

    /// The keyspace backstop finds a carrier the work list never learned about.
    ///
    /// A real shape, not a theoretical one: the retirement writes the carrier
    /// FIRST (its begin, before any enqueue) and registers on the work list only
    /// after its enqueue ended in doubt — on the `{gc}` slot that just failed,
    /// so the registration is the likelier of the two to be lost; a begin whose
    /// reply was lost, or a process that died after its begin, registers
    /// nothing at all. Without phase B the carrier would sit forever holding an
    /// obligation nothing looks for.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn sweep_retire_retry_backstop_finds_a_carrier_with_no_work_list_entry() {
        const OID: &str = "930006";
        let tv = TestValkey::setup().await;
        let metrics = Arc::new(metrics_scrape_only("yda-backstop"));
        let members = vec![carrier_member(0xE1, 930_006)];

        // Carrier only (this build's HASH) — the `{gc}`-slot registration
        // never landed.
        let _: () = tv
            .client()
            .pool
            .hset(
                tv.key_schema().obj_ext_retire_carrier(OID),
                [(members[0].as_str(), "1")],
            )
            .await
            .expect("seeding the carrier must succeed");
        assert!(
            work_list_entries(&tv).await.is_empty(),
            "fixture precondition: nothing on the work list, so only the SCAN can find this"
        );

        let mut result = SweepResult::default();
        sweep_retire_retry(
            tv.client(),
            tv.key_schema(),
            &CancellationToken::new(),
            &mut result,
            Some(&(metrics.clone() as SharedMetrics)),
            "yda-backstop",
        )
        .await
        .expect("the retire-retry phase must not fail");

        assert_eq!(
            pending(&tv).await,
            members,
            "the backstop SCAN over `le:{{*}}:obj:extret` must reach a carrier the fast path \
             forgot; the carrier is the record, the work list is only an index into it"
        );
        assert_eq!(result.retire_carriers_claimed, 1);
        assert!(carrier_members(&tv, OID).await.is_empty());
    }

    /// `drain_retire_carrier` passes the liveness-gate keys — the header and
    /// the `obj:ext` index — to the settle script.
    ///
    /// The script's `cancelled` leg is reachable only through `KEYS[3]` /
    /// `KEYS[4]`. A settle that could not see the header would answer `pay`
    /// on every visit — draining an obligation against a live incarnation.
    /// That is the over-release the gate exists to refuse, and it would be
    /// invisible (the drain succeeds, the members land on pending, nothing
    /// errors).
    ///
    /// A live header plus a member-bearing carrier is what distinguishes the
    /// two behaviourally: with the gate keys the answer is `cancelled` and
    /// nothing reaches `pending`; without them it would be `pay`. Asserted
    /// through the SweepResult rather than by counting the script's arguments,
    /// so the claim is about behaviour rather than about a call shape.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn drain_retire_carrier_passes_the_liveness_gate_keys_to_the_settle_script() {
        const OID: &str = "930007";
        let tv = TestValkey::setup().await;
        let metrics = Arc::new(metrics_scrape_only("yda-three-keys"));
        let members = vec![carrier_member(0xF1, 930_007)];
        seed_carrier(&tv, OID, &members).await;
        let _: () = tv
            .client()
            .pool
            .hset(tv.key_schema().obj_header(OID), [("seq", "1")])
            .await
            .expect("seeding the live header must succeed");

        let mut result = SweepResult::default();
        drain_retire_carrier(
            tv.client(),
            tv.key_schema(),
            &tv.key_schema().ext_retire_retry(),
            OID,
            &mut result,
            Some(&(metrics.clone() as SharedMetrics)),
        )
        .await;

        assert_eq!(
            result.retire_carriers_cancelled, 1,
            "the header key must reach the settle script — one that cannot see it answers `pay` \
             on every visit"
        );
        assert!(
            pending(&tv).await.is_empty(),
            "and nothing may be enqueued against a live incarnation"
        );
    }

    // ══ b.yuo Part R — the write-ahead retirement intent ════════════════════
    //
    // Every decrement a retirement owes is enqueued at most once against every
    // re-presentation the service makes, at any timing: the retirement records
    // its members and an attempt token on its own slot BEFORE it enqueues
    // (`ext_retire_begin.lua`), every attempt is decided once on the `{gc}`
    // slot (the attempt-mode enqueue against the drain's `HSETNX void`
    // resolve), and a token's `{gc}` record outlives every send under it
    // (F1/F2, and the send window F3). The tests below drive the four
    // re-presentation paths the design names — P1 the drain's re-drive, P2 the
    // retirement's in-doubt deferral, P3 a re-derivation (a crash between the
    // enqueue and the carrier's clear, a completed retirement whose teardown
    // failed, and two concurrent retirements), P4 the cancel — plus the
    // record's lifecycle, the send window, the `NOSCRIPT` reload, the lone
    // token HASH, a replayed begin, and the arms a slower actor reaches: a
    // retirement or a drain whose clear another drain superseded, and a
    // retirement whose attempt a drain voided first.
    //
    // Why these are timing-free: the GC runs `GcPolicy::IMMEDIATE`, whose
    // replay-record retention U is 0, so a member's replay record is gone the
    // moment its decrement settles. Nothing but the retirement intent then
    // stands between a re-presented member and a second decrement — which is
    // what makes a double release here VISIBLE: the fixture's blob holds two
    // references (the retiring object's and a standing holder's), so one
    // decrement leaves `ref_count` 1 and a second deletes `ext_meta` and
    // `ext_data` under the holder. The drain's in-progress age X is 0 (a
    // carrier is drained at once) unless a test says otherwise.
    //
    // The fixture's object is hand-seeded — a header and an `obj:ext` index
    // naming one hash — because the retirement's reference walk finds no
    // value row behind such a header, and the retirement derives its release
    // from the index. "Tear down" is the callers' UNLINK of those two keys; the carrier
    // and its token HASH are not in the cleanup inventory and survive it.
    //
    // Faults come from the test-only attempt-enqueue seam
    // (`ReleaseEnqueueFaultInjector`): `Fail` (not sent), `ExecuteThenFail`
    // (ran, reply dropped), `Hold` (kept back; `send_held` is the copy fred
    // writes late), `NoscriptReloadPastWindow` (the helper's real `NOSCRIPT`
    // branch with the reload stalled past the send window) and
    // `ExecuteThenRun` (one tokened transmission sent for real — after an
    // untokened script load — then a test hook runs before the caller sees
    // the reply, which then takes the helper's normal handling; used here to
    // stage another actor's step between a drain's enqueue and its clear).
    // Each consumes one attempt-mode enqueue, in order, on the
    // fixture's client. Where a retirement's own enqueue has to be bracketed
    // the same way, a wrapping `ExternalValueStore` does it instead
    // (`StagedRetirementStore`).
    //
    // RT-n in a test's doc names the directive's test list (b.yuo commit B,
    // Part R).

    use crate::error::ServiceError;
    use crate::observability::metrics::{
        EXT_REF_RELEASE_REASON_RETIREMENT, GC_DECREMENT_OUTCOME_APPLIED,
        GC_DECREMENT_OUTCOME_REPLAYED,
    };
    use crate::storage::diff_read::SnapshotReadHold;
    use crate::storage::diff_read::{
        ObjectSnapshotReadFaultInjector, ObjectSnapshotReadFaultTarget,
    };
    use crate::storage::external_values::{
        enqueue_ext_release_attempt, mint_retire_attempt_token, process_pending_gc,
        resolve_enqueue_attempts, retire_begin, retire_clear, retire_retoken, retire_settle,
        AttemptVerdict, GcPassState, GcPolicy, GcStats, ReleaseAttemptOutcome, ReleaseEnqueueFault,
        ReleaseEnqueueFaultInjector, ReleaseEnqueueHook, RetireBegin, RetireClear, RetireRetoken,
        RetireSettle, ValkeyExternalValueStore, ENQUEUE_ATTEMPT_VOID_RETENTION,
        RETIRE_INTENT_SEND_WINDOW,
    };
    use crate::storage::keys::sha256_to_hex;

    /// The bytes of the blob every Part R fixture retires a reference to.
    const INTENT_BLOB: &[u8] = b"retirement-intent-blob";

    /// The send window the `NOSCRIPT` tests stall their reload past: long
    /// enough that the blob-size read and the first window check always fit
    /// inside it, short enough to keep each test near two seconds.
    const NOSCRIPT_SEND_WINDOW: Duration = Duration::from_secs(2);

    /// The bound on every wait for a [`SnapshotReadHold`] to park. Generous:
    /// it is reached only when the read never arrives, which is a test failure
    /// either way.
    const HOLD_PARK_BUDGET: Duration = Duration::from_secs(30);

    /// Releases a [`SnapshotReadHold`] when dropped, so a hold is released on
    /// every path out of a test — a panic included — and never parks a later
    /// read on the same client.
    struct ReleaseHoldOnDrop(Arc<SnapshotReadHold>);

    impl Drop for ReleaseHoldOnDrop {
        fn drop(&mut self) {
            self.0.release();
        }
    }

    /// A lost reply: the error fred reports for a command it sent and never
    /// heard back from.
    fn lost_reply() -> fred::error::Error {
        fred::error::Error::new(fred::error::ErrorKind::Timeout, "reply lost (test seam)")
    }

    /// One retiring object holding one reference to one blob, beside a
    /// standing holder of a second reference.
    struct Intent {
        tv: TestValkey,
        metrics: Arc<Metrics>,
        ext: ValkeyExternalValueStore,
        faults: Arc<ReleaseEnqueueFaultInjector>,
        oid: &'static str,
        hash: [u8; 32],
    }

    impl Intent {
        async fn setup(label: &str, oid: &'static str, byte: u8) -> Self {
            let tv = TestValkey::setup().await;
            let metrics = Arc::new(metrics_scrape_only(label));
            let ext = ValkeyExternalValueStore::new(tv.client().clone(), tv.key_schema().clone())
                .with_metrics(metrics.clone() as SharedMetrics);
            let faults = Arc::new(ReleaseEnqueueFaultInjector::new());
            tv.client()
                .install_release_enqueue_fault_injector_for_test(faults.clone());
            tv.client()
                .set_retire_intent_in_progress_age_for_test(Duration::ZERO);
            let hash = carrier_hash(byte);
            // Two references: the retiring object's, and a standing holder's
            // that is never released.
            for _ in 0..2 {
                ext.store_values(&[(&hash, INTENT_BLOB)])
                    .await
                    .expect("storing the fixture blob must succeed");
            }
            let fx = Self {
                tv,
                metrics,
                ext,
                faults,
                oid,
                hash,
            };
            let keys = fx.tv.key_schema();
            let _: () = fx
                .tv
                .client()
                .pool
                .hset(keys.obj_header(oid), [("seq", "1")])
                .await
                .expect("seeding the header must succeed");
            let _: () = fx
                .tv
                .client()
                .pool
                .hset(keys.obj_ext_index(oid), [(sha256_to_hex(&hash), "1")])
                .await
                .expect("seeding obj:ext must succeed");
            assert_eq!(fx.ref_count().await, Some(2), "fixture precondition");
            fx
        }

        fn object_id(&self) -> u64 {
            self.oid.parse().expect("fixture object ids are decimal")
        }

        /// The one pending member this object's retirement owes.
        fn member(&self) -> String {
            ExtRelease::retirement(self.hash, self.object_id()).member()
        }

        fn shared(&self) -> SharedMetrics {
            self.metrics.clone()
        }

        /// The real retirement, `schedule_old_object_ext_gc`.
        async fn retire(&self) -> Result<(), ServiceError> {
            self.retire_through(&self.ext).await
        }

        /// The real retirement, handed `ext` as its external-value store (a
        /// [`StagedRetirementStore`] wrapping the fixture's own).
        async fn retire_through(&self, ext: &dyn ExternalValueStore) -> Result<(), ServiceError> {
            schedule_old_object_ext_gc(
                self.tv.client(),
                self.tv.key_schema(),
                self.oid,
                ext,
                Some(&self.metrics),
                &crate::storage::retirement::RetirementWalk::unreserved(),
            )
            .await
        }

        /// The callers' UNLINK after a retirement: the header and the index go;
        /// the carrier and its token HASH stay.
        async fn tear_down(&self) {
            let keys = self.tv.key_schema();
            let _: i64 = self
                .tv
                .client()
                .pool
                .del(vec![
                    keys.obj_header(self.oid),
                    keys.obj_ext_index(self.oid),
                ])
                .await
                .expect("tearing the object down must succeed");
        }

        /// One retire-retry phase (phase A and the keyspace backstop).
        async fn sweep(&self) -> SweepResult {
            let mut result = SweepResult::default();
            sweep_retire_retry(
                self.tv.client(),
                self.tv.key_schema(),
                &CancellationToken::new(),
                &mut result,
                Some(&self.shared()),
                "yuo-intent",
            )
            .await
            .expect("the retire-retry phase must not fail");
            result
        }

        /// The orphan sweep's object phase: the revisit of a header that
        /// outlived its teardown (the object has no layer mapping, so it is an
        /// orphan), which retires it again and UNLINKs it.
        async fn orphan_sweep(&self) {
            let mut result = SweepResult::default();
            let ext: Arc<dyn ExternalValueStore> = Arc::new(self.ext.clone());
            let layer_store =
                LayerStore::new(self.tv.client().clone(), self.tv.key_schema().clone(), 0);
            sweep_orphaned_objects(
                self.tv.client(),
                self.tv.key_schema(),
                &ext,
                &layer_store,
                &CancellationToken::new(),
                &mut result,
                Some(&self.shared()),
            )
            .await
            .expect("the orphan phase must not fail");
            assert_eq!(
                result.orphaned_objects, 1,
                "fixture: the orphan phase must have visited the surviving header"
            );
        }

        /// GC passes under `IMMEDIATE` (U = 0) until quiet: claim, decrement,
        /// complete, tear the replay records down, and one attempt-audit page.
        async fn gc(&self) -> GcStats {
            process_pending_gc(&self.ext, Some(&self.shared()), GcPolicy::IMMEDIATE).await
        }

        async fn ref_count(&self) -> Option<i64> {
            let raw: Option<String> = self
                .tv
                .client()
                .pool
                .hget(
                    self.tv.key_schema().ext_meta(&sha256_to_hex(&self.hash)),
                    "ref_count",
                )
                .await
                .expect("HGET ref_count must succeed");
            raw.map(|v| v.parse().expect("ref_count must be an integer"))
        }

        /// Whether `ext_meta` and `ext_data` both still exist.
        async fn blob_present(&self) -> bool {
            let hex = sha256_to_hex(&self.hash);
            let keys = self.tv.key_schema();
            let meta: bool = self
                .tv
                .client()
                .pool
                .exists(keys.ext_meta(&hex))
                .await
                .expect("EXISTS ext_meta must succeed");
            let data: bool = self
                .tv
                .client()
                .pool
                .exists(keys.ext_data(&hex))
                .await
                .expect("EXISTS ext_data must succeed");
            meta && data
        }

        /// The one-decrement end state: `ref_count` 1 with the blob intact,
        /// exactly one applied decrement and no replay answered.
        async fn assert_decremented_once(&self, why: &str) {
            assert_eq!(
                (self.ref_count().await, self.blob_present().await),
                (Some(1), true),
                "{why}: the blob must hold exactly its standing reference. `(None, false)` is the \
                 double release this design exists to prevent — `ext_meta` and `ext_data` deleted \
                 under a live holder; `(Some(2), true)` is an obligation never paid (a leak)"
            );
            assert_eq!(
                self.decrements(GC_DECREMENT_OUTCOME_APPLIED),
                1,
                "{why}: one decrement applied"
            );
            assert_eq!(
                self.decrements(GC_DECREMENT_OUTCOME_REPLAYED),
                0,
                "{why}: with U = 0 no replay record can absorb a second copy, so a re-presented \
                 member would have APPLIED, not replayed"
            );
        }

        async fn carrier(&self) -> Vec<String> {
            carrier_members(&self.tv, self.oid).await
        }

        async fn tokens(&self) -> Vec<String> {
            attempt_tokens(&self.tv, self.oid).await
        }

        /// The one token recorded on the carrier.
        async fn only_token(&self) -> String {
            let tokens = self.tokens().await;
            assert_eq!(tokens.len(), 1, "exactly one recorded token: {tokens:?}");
            tokens[0].clone()
        }

        async fn records(&self) -> std::collections::HashMap<String, String> {
            attempt_records(&self.tv).await
        }

        async fn record(&self, token: &str) -> Option<String> {
            self.records().await.get(token).cloned()
        }

        /// Whether `token`'s `{gc}` record says its attempt landed.
        async fn landed(&self, token: &str) -> bool {
            self.record(token)
                .await
                .is_some_and(|v| v.starts_with("landed:"))
        }

        /// Whether `token`'s `{gc}` record says its attempt was voided.
        async fn voided(&self, token: &str) -> bool {
            self.record(token)
                .await
                .is_some_and(|v| v.starts_with("void:"))
        }

        /// Rewrite a `{gc}` record's stamp to 1 ms after the epoch, keeping its
        /// kind and object id — older than any retention.
        async fn age_record(&self, token: &str) {
            let value = self
                .record(token)
                .await
                .unwrap_or_else(|| panic!("no record for {token} to age"));
            let mut parts = value.splitn(3, ':');
            let (kind, _ms, oid) = (
                parts.next().expect("kind"),
                parts.next().expect("ms"),
                parts.next().expect("oid"),
            );
            let _: () = self
                .tv
                .client()
                .pool
                .hset(
                    self.tv.key_schema().ext_enqueue_attempts(),
                    [(token, format!("{kind}:1:{oid}"))],
                )
                .await
                .expect("aging a record must succeed");
        }

        async fn flag(&self) -> Option<String> {
            ext_gc_scheduled(&self.tv, self.oid).await
        }

        fn decrements(&self, outcome: &str) -> u64 {
            sample_value(
                &render(&self.metrics),
                "live_edit_gc_decrement_total",
                &[("outcome", outcome)],
            )
            .unwrap_or_else(|| panic!("gc_decrement_total{{outcome={outcome}}} is zero-seeded"))
                as u64
        }

        fn released_by_retirement(&self) -> u64 {
            sample_value(
                &render(&self.metrics),
                "live_edit_ext_ref_released_total",
                &[("reason", EXT_REF_RELEASE_REASON_RETIREMENT)],
            )
            .expect("ext_ref_released_total{reason=retirement} is zero-seeded") as u64
        }

        fn count(&self, name: &str) -> u64 {
            counter(&self.metrics, name)
        }
    }

    /// The `fields.message` of every captured event.
    fn messages(events: &[serde_json::Value]) -> Vec<String> {
        events
            .iter()
            .filter_map(|e| e["fields"]["message"].as_str().map(str::to_owned))
            .collect()
    }

    /// RT-1 (P2, the retirement's in-doubt deferral). Both enqueue attempts
    /// RAN and lost their replies, so the retirement defers: its carrier holds
    /// the member and token K, which already landed. The GC pays the member,
    /// settles it and tears its replay record down; then the sweep must find K
    /// `landed` on `{gc}` and only CLEAR the carrier — not give the member back
    /// a second time.
    ///
    /// K's landed record outlives the sweep (only K's own sender may remove a
    /// landed record at once, F1), and the GC loop's attempt audit removes it
    /// once it is older than V and no carrier holds K (F2).
    ///
    /// **Fails on old code** (in-place perturbation): with the resolve
    /// answering `void` for a landed token, the drain settles, pays the member
    /// again under its own token, and the final GC deletes the blob.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_retirement_whose_enqueue_reply_was_lost_is_not_given_back_again_by_its_carrier() {
        let fx = Intent::setup("yuo-rt1", "960001", 0x61).await;
        fx.faults
            .push(ReleaseEnqueueFault::ExecuteThenFail(lost_reply()))
            .push(ReleaseEnqueueFault::ExecuteThenFail(lost_reply()));

        let err = fx
            .retire()
            .await
            .expect_err("both attempts ended in doubt, so the retirement defers with an Err");
        assert!(
            matches!(err, ServiceError::Aborted(_)),
            "the second attempt's transient error propagates: {err:?}"
        );
        assert_eq!(fx.faults.remaining(), 0, "both attempts took their fault");
        assert_eq!(fx.carrier().await, vec![fx.member()]);
        let k = fx.only_token().await;
        assert!(
            fx.landed(&k).await,
            "fixture: the first attempt ran, so K is recorded landed"
        );
        assert_eq!(pending(&fx.tv).await, vec![fx.member()]);
        assert_eq!(fx.count("live_edit_ext_ref_release_deferred_total"), 1);
        fx.tear_down().await;

        fx.gc().await;
        assert_eq!(fx.ref_count().await, Some(1), "the GC paid the member once");

        let swept = fx.sweep().await;
        let given_back = pending(&fx.tv).await;
        let last = fx.gc().await;
        // The end state first: a sweep that gave the member back shows here as
        // the blob deleted, before any of the finer assertions below.
        fx.assert_decremented_once("RT-1").await;
        assert_eq!(
            (
                swept.retire_releases_already_enqueued,
                swept.retire_releases_recovered,
                swept.retire_carriers_claimed
            ),
            (1, 0, 0),
            "the sweep must find K landed and clear the carrier WITHOUT enqueuing again"
        );
        assert_eq!(
            fx.count("live_edit_ext_ref_release_already_enqueued_total"),
            1
        );
        assert!(
            given_back.is_empty(),
            "nothing may be given back: the member was already paid"
        );
        assert_eq!(last.processed, 0, "the final GC finds nothing to collect");
        assert!(fx.carrier().await.is_empty() && fx.tokens().await.is_empty());
        assert!(
            fx.landed(&k).await,
            "K's landed record outlives the sweep: the drain did not send under K, so it cannot \
             know K's last send has arrived (F1)"
        );

        // The audit removes K's record once it is older than V and no carrier
        // holds K.
        fx.age_record(&k).await;
        fx.gc().await;
        assert!(
            fx.records().await.is_empty(),
            "no attempt record is left once K has aged out with no carrier naming it"
        );
    }

    /// RT-2 (P1, the drain's re-drive). The retirement's
    /// attempts never ran, so the carrier holds an unlanded K. Sweep 1 voids
    /// K, settles `pay`, and its re-drive under K1 RAN but lost its reply: the
    /// carrier keeps {K, K1}. The GC pays the member and tears its replay
    /// record down. Sweep 2 must find K1 `landed` and only clear — no second
    /// decrement. K's void record and K1's landed one both stay until aged out.
    ///
    /// **Fails on old code** (in-place perturbations): the resolve answering
    /// `void` for a landed token makes sweep 2 pay again and the final GC
    /// delete the blob; a re-token that records no token (the pre-Part-R
    /// put-back shape) leaves the drain's token set unrecorded, so the settle
    /// never confirms it and the member is never paid.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_drain_whose_enqueue_reply_was_lost_does_not_give_back_again_on_the_next_sweep() {
        let fx = Intent::setup("yuo-rt2", "960002", 0x62).await;
        fx.faults
            .push(ReleaseEnqueueFault::Fail(lost_reply()))
            .push(ReleaseEnqueueFault::Fail(lost_reply()));
        fx.retire()
            .await
            .expect_err("neither attempt was sent, so the retirement defers");
        let k = fx.only_token().await;
        assert_eq!(fx.record(&k).await, None, "fixture: K never ran");
        assert!(pending(&fx.tv).await.is_empty());
        fx.tear_down().await;

        // Sweep 1: K resolves void, the dead id settles `pay`, and the re-drive
        // runs but its reply is lost.
        fx.faults
            .push(ReleaseEnqueueFault::ExecuteThenFail(lost_reply()));
        let first = fx.sweep().await;
        assert_eq!(fx.faults.remaining(), 0, "the re-drive took its fault");
        assert_eq!(
            (
                first.retire_carriers_claimed,
                first.retire_releases_recovered
            ),
            (1, 0),
            "sweep 1 took the carrier up to pay and could not confirm the re-drive"
        );
        let tokens = fx.tokens().await;
        assert_eq!(
            tokens.len(),
            2,
            "the carrier keeps K and the drain's K1: {tokens:?}"
        );
        let k1 = tokens
            .iter()
            .find(|t| **t != k)
            .expect("the drain's token")
            .clone();
        assert!(fx.voided(&k).await);
        assert!(fx.landed(&k1).await);
        assert_eq!(fx.carrier().await, vec![fx.member()]);
        assert_eq!(pending(&fx.tv).await, vec![fx.member()]);
        assert_eq!(work_list_entries(&fx.tv).await, vec![fx.oid.to_owned()]);

        fx.gc().await;
        assert_eq!(fx.ref_count().await, Some(1), "the GC paid the member once");

        let second = fx.sweep().await;
        let given_back = pending(&fx.tv).await;
        let last = fx.gc().await;
        // The end state first: a second give-back shows here as the blob
        // deleted, before any of the finer assertions below.
        fx.assert_decremented_once("RT-2").await;
        assert_eq!(
            (
                second.retire_releases_already_enqueued,
                second.retire_releases_recovered,
                second.retire_carriers_claimed
            ),
            (1, 0, 0),
            "sweep 2 must find K1 landed and clear the carrier without enqueuing again"
        );
        assert!(given_back.is_empty());
        assert_eq!(last.processed, 0);
        assert!(fx.carrier().await.is_empty() && fx.tokens().await.is_empty());
        assert!(work_list_entries(&fx.tv).await.is_empty());

        assert!(
            fx.voided(&k).await && fx.landed(&k1).await,
            "K's void record and K1's landed one stay until they age out: neither was removed by \
             its own clean single send"
        );
        fx.age_record(&k).await;
        fx.age_record(&k1).await;
        fx.gc().await;
        assert!(
            fx.records().await.is_empty(),
            "the audit removes both once older than V with no carrier naming them"
        );
    }

    /// RT-3 (P3, a crash between the enqueue and the carrier's clear). The
    /// retirement began and its attempt landed, and then the process died: no
    /// clear, no teardown. The orphan sweep's revisit of the surviving header
    /// must short-circuit on the `ext_gc_scheduled` flag the begin wrote
    /// before the enqueue — not re-derive the member, and not even reach a
    /// begin — and the retire-retry drain then finds the attempt landed and
    /// only clears. One decrement.
    ///
    /// **Fails on old code** (in-place perturbation): with a begin that sets
    /// no flag, the revisit re-derives and reaches a second begin, which
    /// answers `busy` over the surviving carrier and counts the reference as
    /// a failed enqueue.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_retirement_that_died_between_its_enqueue_and_its_clear_is_paid_once() {
        let fx = Intent::setup("yuo-rt3", "960003", 0x63).await;
        let client = fx.tv.client();
        let keys = fx.tv.key_schema();
        let k = mint_retire_attempt_token();
        assert_eq!(
            retire_begin(client.pool.next(), keys, fx.oid, &k, &[fx.member()])
                .await
                .expect("begin must succeed"),
            RetireBegin::Begun
        );
        assert_eq!(
            enqueue_ext_release_attempt(
                client,
                keys,
                fx.oid,
                &k,
                &[fx.member()],
                std::time::Instant::now() + RETIRE_INTENT_SEND_WINDOW,
            )
            .await
            .expect("the attempt must land"),
            ReleaseAttemptOutcome::Enqueued
        );
        // The process dies here: no clear, no teardown, no work-list entry.

        fx.gc().await;
        assert_eq!(fx.ref_count().await, Some(1));

        // The orphan sweep revisits the header the dead process left behind,
        // retires it again, and UNLINKs it.
        fx.orphan_sweep().await;
        assert!(
            pending(&fx.tv).await.is_empty(),
            "the revisit must re-derive NOTHING: the begin's flag records that this incarnation's \
             obligation is already recorded"
        );
        assert_eq!(
            fx.count("live_edit_ext_ref_release_enqueue_failed_total"),
            0,
            "and it must short-circuit before any begin — a second begin would answer `busy` \
             over the surviving carrier and count a leak that is not one"
        );
        assert_eq!(fx.released_by_retirement(), 0);
        assert_eq!(fx.carrier().await, vec![fx.member()], "the record survives");

        let swept = fx.sweep().await;
        assert_eq!(
            (
                swept.retire_releases_already_enqueued,
                swept.retire_releases_recovered
            ),
            (1, 0),
            "the backstop finds the carrier and K landed, and only clears it"
        );
        let last = fx.gc().await;
        assert_eq!(last.processed, 0);
        fx.assert_decremented_once("RT-3").await;
    }

    /// RT-3, the completed form (P3: a re-derivation after the obligation was
    /// paid). The retirement succeeded — enqueued, cleared, its record
    /// forgotten — but the teardown that follows failed, so the header
    /// survives with its rows. The GC pays the member and tears its replay
    /// record down; then the orphan sweep revisits the header. Only the flag,
    /// written by the begin, can stop that revisit: nothing else remains to say
    /// this incarnation's obligation was ever recorded.
    ///
    /// **Fails on old code** (in-place perturbation): with a begin that sets
    /// no flag, the revisit begins afresh, enqueues the member again, and the
    /// final GC deletes the blob.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_completed_retirement_whose_teardown_failed_is_not_re_derived_by_the_orphan_sweep() {
        let fx = Intent::setup("yuo-rt3b", "960018", 0x78).await;
        fx.retire().await.expect("a healthy retirement succeeds");
        assert_eq!(fx.released_by_retirement(), 1);
        let flag_after_retirement = fx.flag().await;
        assert!(
            fx.carrier().await.is_empty() && fx.records().await.is_empty(),
            "fixture: the clean retirement cleared its carrier and forgot its record (F1)"
        );
        // The teardown fails: the header and the index survive.

        fx.gc().await;
        assert_eq!(fx.ref_count().await, Some(1));

        fx.orphan_sweep().await;
        let re_derived = pending(&fx.tv).await;
        let last = fx.gc().await;
        fx.assert_decremented_once("RT-3 (completed)").await;
        assert!(re_derived.is_empty(), "the revisit must re-derive nothing");
        assert_eq!(fx.released_by_retirement(), 1, "one release, ever");
        assert_eq!(last.processed, 0);
        assert_eq!(
            flag_after_retirement.as_deref(),
            Some("1"),
            "the begin set the flag that stopped the revisit"
        );
    }

    /// RT-4 (P3, two concurrent retirements of one object). The second
    /// retirement reads the flag unset, then parks on its reference walk (the
    /// snapshot-read seam's hold, which on a release run alone — no route's
    /// size read before it — the walk's first command, the header `EXISTS`,
    /// takes) while the first retirement runs end to end and the GC pays and
    /// settles the member; only then does the test release the hold. When the
    /// second retirement resumes, only the begin's in-script flag check stands
    /// between it and a second enqueue: its own `HGET` short-circuit already
    /// passed. It must answer `already` and enqueue nothing.
    ///
    /// The ordering is forced, not raced or timed: the first retirement starts
    /// only once the second has parked on the held walk, so the second's flag
    /// read precedes the first's begin, and the second cannot go on until the
    /// first retirement and the GC have both returned. The hold is the only one
    /// queued and the second retirement's walk has taken it, so the first
    /// retirement's own walk, which comes later, does not park. The wait for
    /// the park is bounded, and the hold is released on every path out of the
    /// test.
    ///
    /// **Fails on old code** (in-place perturbation): with a begin that
    /// ignores the flag, the second retirement begins afresh over the cleared
    /// carrier, enqueues the member again, and the final GC deletes the blob.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn two_concurrent_retirements_of_one_object_enqueue_its_releases_once() {
        let fx = Intent::setup("yuo-rt4", "960004", 0x64).await;
        let reads = Arc::new(ObjectSnapshotReadFaultInjector::new());
        let hold = reads.push_hold_at(ObjectSnapshotReadFaultTarget::next_read());
        let _release_on_every_path = ReleaseHoldOnDrop(hold.clone());
        fx.tv
            .client()
            .install_object_snapshot_read_fault_injector_for_test(reads.clone());

        let first = async {
            // Start only once the second retirement is parked on its read.
            tokio::time::timeout(HOLD_PARK_BUDGET, hold.parked())
                .await
                .expect("the second retirement never parked on its reference walk");
            assert_eq!(
                reads.holds_remaining(),
                0,
                "the second retirement's walk took the only hold"
            );
            let retired = fx.retire().await;
            let paid = fx.gc().await;
            hold.release();
            (retired, paid)
        };
        let (second, (first_retired, paid)) = tokio::join!(fx.retire(), first);
        let second_enqueued = pending(&fx.tv).await;
        let last = fx.gc().await;

        fx.assert_decremented_once("RT-4").await;
        first_retired.expect("the first retirement succeeds");
        assert_eq!(
            paid.processed, 1,
            "the GC paid the first retirement's member"
        );
        second.expect(
            "the second retirement finds the obligation begun by another and returns Ok without \
             enqueuing",
        );
        assert_eq!(
            fx.released_by_retirement(),
            1,
            "exactly one of the two retirements enqueued"
        );
        assert!(
            second_enqueued.is_empty(),
            "the second retirement must have enqueued nothing"
        );
        assert_eq!(last.processed, 0);
        assert_eq!(
            (
                fx.count("live_edit_ext_ref_release_enqueue_failed_total"),
                fx.count("live_edit_ext_ref_release_deferred_total")
            ),
            (0, 0)
        );
        assert!(fx.carrier().await.is_empty() && fx.tokens().await.is_empty());
    }

    /// RT-5: a late copy of a voided attempt answers `void` and writes nothing.
    /// The retirement's first attempt is held back (a command stuck in fred's
    /// router) and its second is not sent, so it defers. The sweep resolves K
    /// `void` and pays the member under its own token; the GC pays and settles
    /// it. Then fred finally writes the held copy of K: it must find K's void
    /// record and enqueue nothing.
    ///
    /// **Fails on old code** (in-place perturbation): with an attempt-mode
    /// enqueue that ignores a `void` record, the late copy enqueues the member
    /// again and the final GC deletes the blob.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_late_copy_of_a_voided_attempt_writes_nothing() {
        let fx = Intent::setup("yuo-rt5", "960005", 0x65).await;
        fx.faults
            .push(ReleaseEnqueueFault::Hold(lost_reply()))
            .push(ReleaseEnqueueFault::Fail(lost_reply()));
        fx.retire().await.expect_err("both attempts in doubt");
        assert_eq!(fx.faults.held(), 1, "the first attempt is held back");
        let k = fx.only_token().await;
        fx.tear_down().await;

        let swept = fx.sweep().await;
        assert_eq!(
            swept.retire_releases_recovered, 1,
            "the drain paid the member"
        );
        assert!(fx.voided(&k).await);
        fx.gc().await;
        assert_eq!(fx.ref_count().await, Some(1));
        let total_before: Option<i64> = fx
            .tv
            .client()
            .pool
            .get(fx.tv.key_schema().ext_pending_total())
            .await
            .expect("GET total must succeed");

        let replies = fx.faults.send_held(fx.tv.client()).await;
        let late_enqueued = pending(&fx.tv).await;
        let total_after: Option<i64> = fx
            .tv
            .client()
            .pool
            .get(fx.tv.key_schema().ext_pending_total())
            .await
            .expect("GET total must succeed");
        let last = fx.gc().await;

        fx.assert_decremented_once("RT-5").await;
        assert_eq!(
            replies
                .into_iter()
                .map(|r| r.expect("the late copy runs"))
                .collect::<Vec<_>>(),
            vec![("void".to_owned(), 0, 0)],
            "the late copy must find K voided and write nothing"
        );
        assert!(late_enqueued.is_empty());
        assert_eq!(total_after, total_before, "and account no bytes");
        assert_eq!(last.processed, 0);
    }

    /// RT-6, the cancel: every recorded attempt void and the id LIVE. The
    /// drain cancels — carrier and tokens deleted, the flag cleared — and the
    /// live incarnation's own later retirement re-derives the member and pays
    /// it once.
    ///
    /// **Fails on old code** (in-place perturbation): a cancel that leaves the
    /// flag set makes the later retirement short-circuit, and the member is
    /// never paid.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_carrier_whose_attempts_all_resolved_void_over_a_live_id_is_cancelled_and_re_derived_later(
    ) {
        let fx = Intent::setup("yuo-rt6a", "960006", 0x66).await;
        fx.faults
            .push(ReleaseEnqueueFault::Fail(lost_reply()))
            .push(ReleaseEnqueueFault::Fail(lost_reply()));
        fx.retire().await.expect_err("neither attempt was sent");
        let k = fx.only_token().await;
        // No teardown: the id stays live.

        let swept = fx.sweep().await;
        let flag_after_cancel = fx.flag().await;
        let cancelled_enqueued = pending(&fx.tv).await;
        assert_eq!(swept.retire_carriers_cancelled, 1);
        assert_eq!(fx.count("live_edit_ext_ref_release_cancelled_total"), 1);
        assert!(fx.carrier().await.is_empty() && fx.tokens().await.is_empty());
        assert!(fx.voided(&k).await);
        assert!(cancelled_enqueued.is_empty(), "a cancel enqueues nothing");

        // The live incarnation's own retirement, later.
        let retired = fx.retire().await;
        fx.gc().await;
        fx.assert_decremented_once("RT-6 (cancelled)").await;
        assert_eq!(
            flag_after_cancel, None,
            "every recorded attempt was void, so the cancel hands the re-derivation back"
        );
        retired.expect("the live incarnation's own retirement succeeds");
    }

    /// RT-6 (P4): an attempt LANDED and the id is live. The drain must find it
    /// landed and only clear — never cancel — and the flag must survive
    /// (policy 12), so the live incarnation's later retirement short-circuits
    /// instead of paying the same reference a second time.
    ///
    /// **Fails on old code** (in-place perturbation): a drain that settles
    /// without resolving its earlier attempts — the pre-Part-R cancel, which
    /// assumed a carrier's members never landed — cancels over the live id
    /// and clears the flag; the later retirement re-derives the member and the
    /// final GC deletes the blob.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_landed_attempt_over_a_live_id_is_cleared_not_cancelled_and_keeps_the_flag() {
        let fx = Intent::setup("yuo-rt6b", "960007", 0x67).await;
        fx.faults
            .push(ReleaseEnqueueFault::ExecuteThenFail(lost_reply()))
            .push(ReleaseEnqueueFault::ExecuteThenFail(lost_reply()));
        fx.retire().await.expect_err("both attempts in doubt");
        // No teardown: the id stays live.
        fx.gc().await;
        assert_eq!(fx.ref_count().await, Some(1), "the landed attempt was paid");

        let swept = fx.sweep().await;
        let flag_after_sweep = fx.flag().await;
        // The live incarnation's own retirement, later.
        let retired = fx.retire().await;
        let re_derived = pending(&fx.tv).await;
        let last = fx.gc().await;

        fx.assert_decremented_once("RT-6 (landed)").await;
        assert_eq!(
            (
                swept.retire_releases_already_enqueued,
                swept.retire_carriers_cancelled
            ),
            (1, 0),
            "a landed attempt is cleared, never cancelled"
        );
        assert_eq!(
            flag_after_sweep.as_deref(),
            Some("1"),
            "the paid obligation keeps the flag even though the id is live (policy 12)"
        );
        retired.expect("the later retirement short-circuits on the flag");
        assert!(re_derived.is_empty(), "and re-derives nothing");
        assert_eq!(last.processed, 0);
    }

    /// RT-7: a carrier whose newest token is younger than the in-progress age
    /// X belongs to a retirement or drain that may still be running, so the
    /// drain leaves it — no new token, no `{gc}` resolve, the work-list entry
    /// kept — and takes it up once it is older than X.
    ///
    /// **Fails on old code** (in-place perturbation): a re-token that ignores
    /// the in-progress age takes the carrier at once, voids K and pays.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_carrier_younger_than_the_in_progress_age_is_skipped_and_its_entry_kept() {
        let fx = Intent::setup("yuo-rt7", "960008", 0x68).await;
        fx.faults
            .push(ReleaseEnqueueFault::Fail(lost_reply()))
            .push(ReleaseEnqueueFault::Fail(lost_reply()));
        fx.retire().await.expect_err("neither attempt was sent");
        let k = fx.only_token().await;
        fx.tear_down().await;

        fx.tv
            .client()
            .set_retire_intent_in_progress_age_for_test(Duration::from_secs(3600));
        let young = fx.sweep().await;
        assert_eq!(
            (
                young.retire_carriers_claimed,
                young.retire_carriers_cancelled,
                young.retire_releases_already_enqueued
            ),
            (0, 0, 0)
        );
        assert_eq!(fx.tokens().await, vec![k.clone()], "no token added");
        assert_eq!(
            fx.record(&k).await,
            None,
            "K must not be voided under a retirement that may still send it"
        );
        assert_eq!(fx.carrier().await, vec![fx.member()]);
        assert_eq!(work_list_entries(&fx.tv).await, vec![fx.oid.to_owned()]);
        assert!(pending(&fx.tv).await.is_empty());

        fx.tv
            .client()
            .set_retire_intent_in_progress_age_for_test(Duration::ZERO);
        let old = fx.sweep().await;
        assert_eq!(old.retire_releases_recovered, 1);
        fx.gc().await;
        fx.assert_decremented_once("RT-7").await;
    }

    /// RT-8, the GC loop's attempt audit (F2). One page, with every record
    /// planted by hand:
    ///
    /// | record | age | a carrier holds it | expected |
    /// |---|---|---|---|
    /// | void | past V | — | removed |
    /// | landed | past V | no | removed |
    /// | landed | past V | yes | kept |
    /// | landed | 2 h (past the old 1 h, under V) | no | kept |
    /// | void | fresh | — | kept |
    /// | landed | fresh | no | kept |
    ///
    /// **Fails on old code** (in-place perturbations): an audit that drops
    /// the carrier check removes the carried record; one that removes landed
    /// records past 1 h removes the 2 h one.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn the_attempt_audit_removes_only_records_past_v_that_no_carrier_holds() {
        let tv = TestValkey::setup().await;
        let keys = tv.key_schema();
        let ext = ValkeyExternalValueStore::new(tv.client().clone(), keys.clone());
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("the clock is past the epoch")
            .as_millis() as u64;
        let v_ms = ENQUEUE_ATTEMPT_VOID_RETENTION.as_millis() as u64;
        // Ten minutes of margin either side absorbs any skew between this host
        // and the store.
        let past_v = now_ms - v_ms - 600_000;
        let two_hours = now_ms - 2 * 3_600_000;
        let planted = [
            ("void-past-v", format!("void:{past_v}:961001")),
            ("landed-past-v-free", format!("landed:{past_v}:961002")),
            ("landed-past-v-carried", format!("landed:{past_v}:961003")),
            ("landed-2h-free", format!("landed:{two_hours}:961004")),
            ("void-fresh", format!("void:{now_ms}:961005")),
            ("landed-fresh-free", format!("landed:{now_ms}:961006")),
        ];
        let _: () = tv
            .client()
            .pool
            .hset(keys.ext_enqueue_attempts(), planted.to_vec())
            .await
            .expect("planting the records must succeed");
        let _: () = tv
            .client()
            .pool
            .hset(
                keys.obj_ext_retire_attempts("961003"),
                [("landed-past-v-carried", "1")],
            )
            .await
            .expect("planting the carrier's token must succeed");

        let pass = ext
            .process_pending(&GcPolicy::IMMEDIATE, &mut GcPassState::new())
            .await
            .expect("a GC pass must succeed");
        assert_eq!(pass.attempt_records_retired, 2);

        let mut left: Vec<String> = attempt_records(&tv).await.into_keys().collect();
        left.sort();
        assert_eq!(
            left,
            vec![
                "landed-2h-free".to_owned(),
                "landed-fresh-free".to_owned(),
                "landed-past-v-carried".to_owned(),
                "void-fresh".to_owned(),
            ],
            "only the void record past V and the landed one past V that no carrier holds may go: \
             a carried token must still resolve `landed`, and a landed record younger than V \
             still guards a send under its token that may be in transit"
        );
    }

    /// RT-9: begin answers `busy` over an outstanding carrier from an earlier
    /// lifecycle of the same id, writes nothing — the earlier carrier and its
    /// token are untouched, no member is merged in, the flag stays unset — and
    /// the retirement counts its reference as a failed enqueue.
    ///
    /// **Fails on old code** (in-place perturbation): a begin without the
    /// `busy` check merges this retirement's member into the earlier carrier.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_begin_over_an_earlier_lifecycles_carrier_answers_busy_and_writes_nothing() {
        let fx = Intent::setup("yuo-rt9", "960009", 0x69).await;
        let keys = fx.tv.key_schema();
        let earlier = carrier_member(0x6A, 960_009);
        let _: () = fx
            .tv
            .client()
            .pool
            .hset(
                keys.obj_ext_retire_carrier(fx.oid),
                [(earlier.as_str(), "1")],
            )
            .await
            .expect("seeding the earlier carrier must succeed");
        let _: () = fx
            .tv
            .client()
            .pool
            .hset(
                keys.obj_ext_retire_attempts(fx.oid),
                [("earlier-token", "1")],
            )
            .await
            .expect("seeding the earlier token must succeed");

        let err = fx.retire().await.expect_err("begin answers busy");
        assert!(matches!(err, ServiceError::Unavailable(_)), "{err:?}");
        assert_eq!(
            fx.carrier().await,
            vec![earlier],
            "nothing may be merged into the earlier lifecycle's carrier"
        );
        assert_eq!(fx.tokens().await, vec!["earlier-token".to_owned()]);
        assert_eq!(fx.flag().await, None);
        assert!(pending(&fx.tv).await.is_empty());
        assert_eq!(
            fx.count("live_edit_ext_ref_release_enqueue_failed_total"),
            1
        );
        assert_eq!(fx.count("live_edit_ext_ref_release_deferred_total"), 0);
    }

    /// RT-9 (extended) and RT-13's operator text: a token HASH with NO
    /// carrier — which only an older build's destructive claim leaves — also
    /// makes begin answer `busy` and write nothing, and both the returned
    /// error and the WARN say this release does not remove it. Neither may
    /// promise that a restore's residue wipe clears it: nothing in this build
    /// does.
    ///
    /// **Fails on old code** (in-place perturbation): the round-4 texts, which
    /// said a checkpoint restore's residue wipe settles it.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_begin_over_a_lone_token_hash_answers_busy_and_says_this_release_does_not_remove_it()
    {
        let fx = Intent::setup("yuo-rt13-text", "960010", 0x70).await;
        let keys = fx.tv.key_schema();
        let _: () = fx
            .tv
            .client()
            .pool
            .hset(keys.obj_ext_retire_attempts(fx.oid), [("lone-token", "17")])
            .await
            .expect("planting the lone HASH must succeed");

        let (result, events) = capture_warns(fx.retire()).await;
        let err = result.expect_err("begin answers busy");
        let ServiceError::Unavailable(text) = &err else {
            panic!("busy is Unavailable: {err:?}");
        };
        assert!(
            text.contains("which this release does not remove") && !text.contains("residue wipe"),
            "the returned error must say this release does not remove the tokens, and promise no \
             remover: {text}"
        );
        let warns = messages(&events);
        let busy: Vec<&String> = warns
            .iter()
            .filter(|m| m.contains("already has a release record from before"))
            .collect();
        assert_eq!(busy.len(), 1, "one busy WARN: {warns:?}");
        assert!(
            busy[0].contains("which this release does not remove")
                && !busy[0].contains("residue wipe"),
            "the WARN must say the same: {}",
            busy[0]
        );
        assert_eq!(
            fx.count("live_edit_ext_ref_release_enqueue_failed_total"),
            1
        );
        let hash: std::collections::HashMap<String, String> = fx
            .tv
            .client()
            .pool
            .hgetall(keys.obj_ext_retire_attempts(fx.oid))
            .await
            .expect("HGETALL must succeed");
        assert_eq!(
            hash,
            std::collections::HashMap::from([("lone-token".to_owned(), "17".to_owned())]),
            "the lone HASH is unchanged"
        );
        assert!(fx.carrier().await.is_empty());
        assert_eq!(fx.flag().await, None);
        assert!(pending(&fx.tv).await.is_empty());
    }

    /// Where a retirement that owes no release finds a release record from
    /// before.
    #[derive(Clone, Copy, Debug)]
    enum EarlierRecord {
        /// An earlier lifecycle's carrier, with its attempt token.
        Carrier,
        /// A token HASH with no carrier (an older build's leftover).
        LoneTokens,
    }

    /// A retirement of an object holding no external value, over a release
    /// record from before (`record`), owes nothing — so begin's
    /// `busy` is `Ok`: nothing counted on `enqueue_failed`, no "could not
    /// record" WARN, the earlier carrier and its token HASH as they were, the
    /// flag left unset, nothing enqueued.
    ///
    /// **Fails on old code** (in-place perturbation): without the
    /// `releases.is_empty()` early `Ok` in the `Busy` arm, the retirement
    /// answers `Unavailable` and WARNs a leak of up to 0 references.
    async fn an_empty_retirement_over_an_earlier_record_answers_ok(record: EarlierRecord) {
        let label = format!("yuo-16b-{record:?}");
        let fx = Intent::setup(&label, "960012", 0x72).await;
        let keys = fx.tv.key_schema();
        // The fixture's own object references a blob; the one retired here is
        // a second object, a header and nothing else, which references none.
        let oid = "960013";
        let _: () = fx
            .tv
            .client()
            .pool
            .hset(keys.obj_header(oid), [("seq", "1")])
            .await
            .expect("seeding the header must succeed");
        let earlier = carrier_member(0x73, 960_013);
        if matches!(record, EarlierRecord::Carrier) {
            let _: () = fx
                .tv
                .client()
                .pool
                .hset(keys.obj_ext_retire_carrier(oid), [(earlier.as_str(), "1")])
                .await
                .expect("seeding the earlier carrier must succeed");
        }
        let _: () = fx
            .tv
            .client()
            .pool
            .hset(keys.obj_ext_retire_attempts(oid), [("earlier-token", "17")])
            .await
            .expect("seeding the earlier token must succeed");
        let failed_before = fx.count("live_edit_ext_ref_release_enqueue_failed_total");

        let (result, events) = capture_warns(schedule_old_object_ext_gc(
            fx.tv.client(),
            keys,
            oid,
            &fx.ext,
            Some(&fx.metrics),
            &crate::storage::retirement::RetirementWalk::unreserved(),
        ))
        .await;

        let what = format!("{record:?}");
        assert!(
            result.is_ok(),
            "{what}: a retirement that owes nothing is not failed by begin's busy: {result:?}"
        );
        assert_eq!(
            fx.count("live_edit_ext_ref_release_enqueue_failed_total"),
            failed_before,
            "{what}: nothing is counted as unrecorded"
        );
        let warns = messages(&events);
        assert!(
            !warns.iter().any(|m| m.contains("could not record")),
            "{what}: no leak WARN: {warns:?}"
        );
        let expected_carrier = match record {
            EarlierRecord::Carrier => vec![earlier],
            EarlierRecord::LoneTokens => Vec::new(),
        };
        assert_eq!(
            carrier_members(&fx.tv, oid).await,
            expected_carrier,
            "{what}: the earlier carrier is as it was"
        );
        let hash: std::collections::HashMap<String, String> = fx
            .tv
            .client()
            .pool
            .hgetall(keys.obj_ext_retire_attempts(oid))
            .await
            .expect("HGETALL must succeed");
        assert_eq!(
            hash,
            std::collections::HashMap::from([("earlier-token".to_owned(), "17".to_owned())]),
            "{what}: the token HASH is unchanged"
        );
        assert_eq!(ext_gc_scheduled(&fx.tv, oid).await, None, "{what}: no flag");
        assert!(pending(&fx.tv).await.is_empty(), "{what}: nothing enqueued");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn an_empty_retirement_over_an_earlier_lifecycles_carrier_answers_ok_and_counts_nothing()
    {
        an_empty_retirement_over_an_earlier_record_answers_ok(EarlierRecord::Carrier).await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn an_empty_retirement_over_a_lone_token_hash_answers_ok_and_counts_nothing() {
        an_empty_retirement_over_an_earlier_record_answers_ok(EarlierRecord::LoneTokens).await;
    }

    /// RT-10, an overtaken retirement, at protocol level. A retirement began
    /// and its first attempt landed, but it is slower than X: before it sees
    /// that reply, the sweep re-tokens its carrier, finds K landed and clears
    /// it. Then the retirement's second attempt under K arrives. It must find
    /// K's landed record and answer `already`, so the GC never decrements
    /// twice.
    ///
    /// **Fails on old code** (in-place perturbation): with the drain removing
    /// the prior tokens it found landed (round 2's forget), the second
    /// attempt finds K gone, enqueues again, and the final GC deletes the
    /// blob.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_retirement_overtaken_by_the_sweep_finds_its_second_attempt_already_landed() {
        let fx = Intent::setup("yuo-rt10", "960011", 0x71).await;
        let client = fx.tv.client();
        let keys = fx.tv.key_schema();
        let k = mint_retire_attempt_token();
        let send_by = std::time::Instant::now() + RETIRE_INTENT_SEND_WINDOW;
        assert_eq!(
            retire_begin(client.pool.next(), keys, fx.oid, &k, &[fx.member()])
                .await
                .expect("begin must succeed"),
            RetireBegin::Begun
        );
        assert_eq!(
            enqueue_ext_release_attempt(client, keys, fx.oid, &k, &[fx.member()], send_by)
                .await
                .expect("the first attempt lands"),
            ReleaseAttemptOutcome::Enqueued
        );
        fx.gc().await;
        assert_eq!(fx.ref_count().await, Some(1));

        let swept = fx.sweep().await;
        assert_eq!(swept.retire_releases_already_enqueued, 1);
        assert!(fx.carrier().await.is_empty());

        // The overtaken retirement's second attempt under K arrives now.
        let second = enqueue_ext_release_attempt(client, keys, fx.oid, &k, &[fx.member()], send_by)
            .await
            .expect("the second attempt runs");
        let second_enqueued = pending(&fx.tv).await;
        let last = fx.gc().await;

        fx.assert_decremented_once("RT-10").await;
        assert_eq!(
            second,
            ReleaseAttemptOutcome::AlreadyLanded,
            "the overtaken retirement's second send under K must find K landed"
        );
        assert!(second_enqueued.is_empty(), "and enqueue nothing");
        assert_eq!(last.processed, 0);
    }

    /// RT-11, a late copy of an ERRORED attempt. The retirement's first
    /// attempt is held back (a command stuck in fred's router) and its second
    /// lands, so it returns Ok — but because an attempt under K errored, it
    /// must NOT remove K's landed record (F1). The GC pays and settles; then
    /// fred writes the held copy of K, which must answer `already`.
    ///
    /// **Fails on old code** (in-place perturbation): with the record
    /// forgotten on `cleared` regardless of the earlier error, the late copy
    /// enqueues the member again and the final GC deletes the blob.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_late_copy_of_an_errored_attempt_finds_its_token_landed() {
        let fx = Intent::setup("yuo-rt11", "960012", 0x72).await;
        fx.faults.push(ReleaseEnqueueFault::Hold(lost_reply()));
        fx.retire()
            .await
            .expect("the second attempt lands, so the retirement succeeds");
        assert_eq!(fx.faults.held(), 1);
        assert_eq!(fx.released_by_retirement(), 1);
        assert!(fx.carrier().await.is_empty());
        let records_after_clear = fx.records().await;
        fx.tear_down().await;

        fx.gc().await;
        assert_eq!(fx.ref_count().await, Some(1));

        // fred finally writes the held copy of K.
        let replies = fx.faults.send_held(fx.tv.client()).await;
        let late_enqueued = pending(&fx.tv).await;
        let last = fx.gc().await;

        fx.assert_decremented_once("RT-11").await;
        assert_eq!(
            records_after_clear.len(),
            1,
            "K's landed record must survive the clear: an attempt under K errored, so a copy of \
             it may still arrive (F1). {records_after_clear:?}"
        );
        let k = records_after_clear
            .keys()
            .next()
            .expect("one record")
            .clone();
        assert_eq!(
            replies
                .into_iter()
                .map(|r| r.expect("the late copy runs"))
                .collect::<Vec<_>>(),
            vec![("already".to_owned(), 0, 0)],
            "the late copy must find K landed and write nothing"
        );
        assert!(late_enqueued.is_empty());
        assert_eq!(last.processed, 0);
        assert!(fx.landed(&k).await);
    }

    /// RT-12, the send window (F3) on the retirement. With Y = 0 the window
    /// has passed before the first attempt's first transmission, so nothing
    /// is sent — the queued fault is not even consumed — and the retirement
    /// defers with K on its carrier. A sweep with the window restored resolves
    /// K void and pays once.
    ///
    /// **Fails on old code** (in-place perturbation): without the window
    /// check the retirement sends, and succeeds.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_retirement_past_its_send_window_sends_nothing_and_the_sweep_pays_once() {
        let fx = Intent::setup("yuo-rt12-retire", "960013", 0x73).await;
        fx.tv
            .client()
            .set_retire_intent_send_window_for_test(Duration::ZERO);
        fx.faults.push(ReleaseEnqueueFault::Fail(lost_reply()));

        let err = fx
            .retire()
            .await
            .expect_err("the window passed, so the retirement defers");
        let ServiceError::Unavailable(text) = &err else {
            panic!("a not-sent deferral is Unavailable: {err:?}");
        };
        assert!(text.contains("its enqueue was not sent"), "{text}");
        assert_eq!(
            fx.faults.remaining(),
            1,
            "nothing was sent: the window check precedes the seam, so the queued fault is untouched"
        );
        assert!(pending(&fx.tv).await.is_empty());
        assert_eq!(fx.count("live_edit_ext_ref_release_deferred_total"), 1);
        assert_eq!(fx.carrier().await, vec![fx.member()]);
        let k = fx.only_token().await;
        assert_eq!(fx.record(&k).await, None);
        fx.tear_down().await;

        fx.tv
            .client()
            .install_release_enqueue_fault_injector_for_test(Arc::new(
                ReleaseEnqueueFaultInjector::new(),
            ));
        fx.tv
            .client()
            .set_retire_intent_send_window_for_test(RETIRE_INTENT_SEND_WINDOW);
        let swept = fx.sweep().await;
        assert_eq!(swept.retire_releases_recovered, 1);
        assert!(fx.voided(&k).await);
        fx.gc().await;
        fx.assert_decremented_once("RT-12 (retirement)").await;
    }

    /// RT-12, the send window on the drain. A sweep whose window has passed
    /// before its pay sends nothing: the carrier keeps K and its own K1, and
    /// the entry stays. The next sweep resolves both void and pays once.
    ///
    /// **Fails on old code** (in-place perturbation): without the window
    /// check the first sweep pays.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_drain_past_its_send_window_sends_nothing_and_the_next_sweep_pays_once() {
        let fx = Intent::setup("yuo-rt12-drain", "960014", 0x74).await;
        fx.faults
            .push(ReleaseEnqueueFault::Fail(lost_reply()))
            .push(ReleaseEnqueueFault::Fail(lost_reply()));
        fx.retire().await.expect_err("neither attempt was sent");
        let k = fx.only_token().await;
        fx.tear_down().await;

        fx.tv
            .client()
            .set_retire_intent_send_window_for_test(Duration::ZERO);
        let late = fx.sweep().await;
        assert_eq!(
            (late.retire_carriers_claimed, late.retire_releases_recovered),
            (1, 0),
            "the drain took the carrier up to pay and sent nothing"
        );
        let tokens = fx.tokens().await;
        assert_eq!(
            tokens.len(),
            2,
            "the carrier keeps K and the unsent K1: {tokens:?}"
        );
        let k1 = tokens.iter().find(|t| **t != k).expect("K1").clone();
        assert_eq!(fx.record(&k1).await, None, "K1 was never sent");
        assert!(pending(&fx.tv).await.is_empty());
        assert_eq!(fx.carrier().await, vec![fx.member()]);
        assert_eq!(work_list_entries(&fx.tv).await, vec![fx.oid.to_owned()]);

        fx.tv
            .client()
            .set_retire_intent_send_window_for_test(RETIRE_INTENT_SEND_WINDOW);
        let next = fx.sweep().await;
        assert_eq!(next.retire_releases_recovered, 1);
        assert!(fx.voided(&k1).await);
        fx.gc().await;
        fx.assert_decremented_once("RT-12 (drain)").await;
    }

    /// RT-12 n1, the `NOSCRIPT` reload at protocol level. The first
    /// transmission is answered `NOSCRIPT` (no copy ran), the real `SCRIPT
    /// LOAD` runs, and the reload stalls past the send window. The helper must
    /// not re-send, and must answer an outcome-unknown `Err` — never `NotSent`,
    /// because `NOSCRIPT` proves only that the copy answering it did not run.
    ///
    /// Takes about two seconds: the seam sleeps until the send window passes.
    ///
    /// **Fails on old code** (in-place perturbations): without the second
    /// window check the helper re-sends and K lands; answering `Ok(NotSent)`
    /// there fails the outcome assertion.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn an_attempt_whose_noscript_reload_outlives_the_send_window_is_not_re_sent_and_is_unknown(
    ) {
        let fx = Intent::setup("yuo-rt12-n1", "960015", 0x75).await;
        let client = fx.tv.client();
        let keys = fx.tv.key_schema();
        client.set_retire_intent_send_window_for_test(NOSCRIPT_SEND_WINDOW);
        fx.faults
            .push(ReleaseEnqueueFault::NoscriptReloadPastWindow { first_runs: false });
        let k = mint_retire_attempt_token();
        let send_by = std::time::Instant::now() + client.retire_intent_send_window();

        let outcome =
            enqueue_ext_release_attempt(client, keys, fx.oid, &k, &[fx.member()], send_by).await;
        let Err(ServiceError::Unavailable(text)) = &outcome else {
            panic!("a reload past the window is an outcome-unknown Unavailable: {outcome:?}");
        };
        assert!(
            text.contains("whether an earlier copy of this attempt ran is unknown"),
            "{text}"
        );
        assert_eq!(fx.faults.remaining(), 0, "the seam was taken");
        assert_eq!(fx.record(&k).await, None, "no copy under K ran");
        assert!(pending(&fx.tv).await.is_empty());
        for (key, what) in [
            (keys.ext_pending_age(), "pending:age"),
            (keys.ext_pending_bytes(), "pending:bytes"),
            (keys.ext_pending_total(), "pending:total"),
        ] {
            let exists: bool = client.pool.exists(&key).await.expect("EXISTS must succeed");
            assert!(!exists, "{what} must be untouched");
        }
        // Whether the real `SCRIPT LOAD` ran is not asserted: on the shared
        // test server any earlier test may already have loaded the script, so
        // `SCRIPT EXISTS` would answer the same either way (the same reason no
        // test here runs `SCRIPT FLUSH`).
    }

    /// RT-12 n2, the `NOSCRIPT` reload at retirement level. The first
    /// attempt's copy RAN (sent before a failover emptied the script cache)
    /// and then the stand-in `NOSCRIPT` answer arrives; the reload stalls past
    /// the window. Attempt 1 is therefore outcome-unknown, and attempt 2 is
    /// not sent. The retirement must say exactly that — "unknown outcome", not
    /// "its enqueue was not sent" — defer with K on its carrier, and the sweep
    /// must then find K landed and only clear. K's record survives: an attempt
    /// under it errored, so F1 does not apply.
    ///
    /// Takes about two seconds: the seam sleeps until the send window passes.
    ///
    /// **Fails on old code** (in-place perturbation): answering `Ok(NotSent)`
    /// after the reload returns the pure not-sent text, which says nothing
    /// was enqueued while the first copy had landed.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_retirement_whose_reload_outlives_the_send_window_reports_an_unknown_outcome() {
        let fx = Intent::setup("yuo-rt12-n2", "960016", 0x76).await;
        fx.tv
            .client()
            .set_retire_intent_send_window_for_test(NOSCRIPT_SEND_WINDOW);
        fx.faults
            .push(ReleaseEnqueueFault::NoscriptReloadPastWindow { first_runs: true });

        let (result, events) = capture_warns(fx.retire()).await;
        let err = result.expect_err("the retirement defers");
        let ServiceError::Unavailable(text) = &err else {
            panic!("the mixed-case deferral is Unavailable: {err:?}");
        };
        assert!(
            text.contains("unknown outcome") && !text.contains("its enqueue was not sent"),
            "the first attempt may have run, so the text must not say nothing was sent: {text}"
        );
        let warns = messages(&events);
        assert!(
            warns
                .iter()
                .any(|m| m.contains("failed once with an unknown outcome")
                    && m.contains("retry was NOT SENT")),
            "the mixed-case WARN must be emitted: {warns:?}"
        );
        assert_eq!(fx.faults.remaining(), 0);
        assert_eq!(fx.count("live_edit_ext_ref_release_deferred_total"), 1);
        assert_eq!(fx.carrier().await, vec![fx.member()]);
        let k = fx.only_token().await;
        assert!(fx.landed(&k).await, "fixture: the first copy ran");
        assert_eq!(pending(&fx.tv).await, vec![fx.member()]);
        fx.tear_down().await;

        fx.tv
            .client()
            .set_retire_intent_send_window_for_test(RETIRE_INTENT_SEND_WINDOW);
        let swept = fx.sweep().await;
        assert_eq!(
            (
                swept.retire_releases_already_enqueued,
                swept.retire_releases_recovered
            ),
            (1, 0)
        );
        assert_eq!(
            pending(&fx.tv).await,
            vec![fx.member()],
            "the sweep enqueued nothing more"
        );
        fx.gc().await;
        fx.assert_decremented_once("RT-12 n2").await;
        assert!(
            fx.landed(&k).await,
            "K's record survives: an attempt under K errored"
        );
    }

    /// RT-13: a token HASH with no carrier (only an older build leaves one)
    /// is inert. The drain's re-token answers `empty` and writes nothing (the
    /// work-list entry is retired, the HASH stays), a clear under one of its
    /// tokens answers `gone`, and a settle answers `changed` even over a dead
    /// id — each leaving the HASH in place, because it may be the only
    /// evidence that this incarnation's obligation was already paid.
    ///
    /// **Fails on old code** (in-place perturbations, one per guard): a
    /// re-token that deletes the HASH in its empty arm; a clear that does not
    /// check the carrier (it answers `cleared` and deletes the HASH); a settle
    /// that does not check the carrier (it answers `pay`).
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_lone_token_hash_is_left_in_place_by_the_re_token_the_clear_and_the_settle() {
        let fx = Intent::setup("yuo-rt13", "960017", 0x77).await;
        let client = fx.tv.client();
        let keys = fx.tv.key_schema();
        fx.tear_down().await;
        let _: () = client
            .pool
            .hset(keys.obj_ext_retire_attempts(fx.oid), [("lone-token", "17")])
            .await
            .expect("planting the lone HASH must succeed");
        recon_register(
            client,
            &keys.ext_retire_retry(),
            std::slice::from_ref(&fx.oid.to_owned()),
        )
        .await
        .expect("registering the work-list entry must succeed");
        let lone = vec!["lone-token".to_owned()];

        let swept = fx.sweep().await;
        assert_eq!(
            (
                swept.retire_carriers_claimed,
                swept.retire_carriers_cancelled,
                swept.retire_releases_already_enqueued
            ),
            (0, 0, 0)
        );
        assert!(work_list_entries(&fx.tv).await.is_empty());
        assert_eq!(fx.tokens().await, lone, "the drain's re-token leaves it");

        let conn = client.pool.next();
        assert_eq!(
            retire_retoken(conn, keys, fx.oid, "new-token", Duration::ZERO)
                .await
                .expect("re-token must succeed"),
            RetireRetoken::Empty
        );
        assert_eq!(fx.tokens().await, lone, "the re-token writes nothing");
        assert_eq!(
            retire_clear(conn, keys, fx.oid, "lone-token", &[])
                .await
                .expect("clear must succeed"),
            RetireClear::Gone
        );
        assert_eq!(fx.tokens().await, lone, "the clear writes nothing");
        assert_eq!(
            retire_settle(conn, keys, fx.oid, "lone-token", &[])
                .await
                .expect("settle must succeed"),
            RetireSettle::Changed
        );
        assert_eq!(fx.tokens().await, lone, "the settle writes nothing");
        assert!(pending(&fx.tv).await.is_empty());
    }

    /// What a [`StagedRetirementStore`] stages around the retirement's first
    /// attempt-mode enqueue.
    #[derive(Clone, Copy, Debug)]
    enum Stage {
        /// After the real enqueue returns, a drain re-tokens the carrier, so
        /// the retirement's clear meets a token newer than its own.
        RetokenAfter,
        /// Before the enqueue is sent, a drain re-tokens the carrier and
        /// resolves the retirement's token, which has not run, `void`; the
        /// real enqueue then finds that record.
        VoidBefore,
    }

    /// An [`ExternalValueStore`] that wraps the fixture's real store and
    /// stages one drain step around the retirement's FIRST attempt-mode
    /// enqueue. The retirement's begin, enqueue and clear all run inside
    /// `schedule_old_object_ext_gc`, so wrapping the one store method it calls
    /// is how a test reaches the gaps between them. The staged step runs the
    /// drain's own scripts (`retire_retoken` with an in-progress age of zero,
    /// then for `VoidBefore` `resolve_enqueue_attempts`) under
    /// `drain_token`, and records what they answered; every other method
    /// delegates.
    struct StagedRetirementStore<'a> {
        fx: &'a Intent,
        stage: Stage,
        drain_token: String,
        /// The retirement's attempt token, once per enqueue call.
        attempts: std::sync::Mutex<Vec<String>>,
        /// What the staged re-token answered.
        retoken: std::sync::Mutex<Option<RetireRetoken>>,
        /// What the staged resolve answered (`VoidBefore` only).
        resolved: std::sync::Mutex<Option<Vec<AttemptVerdict>>>,
    }

    impl<'a> StagedRetirementStore<'a> {
        fn new(fx: &'a Intent, stage: Stage) -> Self {
            Self {
                fx,
                stage,
                drain_token: mint_retire_attempt_token(),
                attempts: std::sync::Mutex::new(Vec::new()),
                retoken: std::sync::Mutex::new(None),
                resolved: std::sync::Mutex::new(None),
            }
        }

        /// The retirement's one attempt token: it made exactly one enqueue
        /// call.
        fn only_attempt(&self) -> String {
            let attempts = self.attempts.lock().expect("never poisoned").clone();
            assert_eq!(
                attempts.len(),
                1,
                "the retirement made exactly one enqueue call: {attempts:?}"
            );
            attempts[0].clone()
        }

        fn retoken(&self) -> Option<RetireRetoken> {
            self.retoken.lock().expect("never poisoned").clone()
        }

        fn resolved(&self) -> Option<Vec<AttemptVerdict>> {
            self.resolved.lock().expect("never poisoned").clone()
        }

        /// The staged drain's re-token of the carrier.
        async fn stage_retoken(&self, object_id: &str) {
            let answer = retire_retoken(
                self.fx.tv.client().pool.next(),
                self.fx.tv.key_schema(),
                object_id,
                &self.drain_token,
                Duration::ZERO,
            )
            .await
            .expect("the staged re-token must succeed");
            *self.retoken.lock().expect("never poisoned") = Some(answer);
        }
    }

    #[async_trait::async_trait]
    impl ExternalValueStore for StagedRetirementStore<'_> {
        async fn enqueue_retirement_attempt(
            &self,
            object_id: &str,
            attempt: &str,
            releases: &[ExtRelease],
            send_by: std::time::Instant,
        ) -> Result<ReleaseAttemptOutcome, ServiceError> {
            let first = {
                let mut attempts = self.attempts.lock().expect("never poisoned");
                attempts.push(attempt.to_owned());
                attempts.len() == 1
            };
            if first && matches!(self.stage, Stage::VoidBefore) {
                self.stage_retoken(object_id).await;
                let verdicts = resolve_enqueue_attempts(
                    self.fx.tv.client().pool.next(),
                    self.fx.tv.key_schema(),
                    object_id,
                    &[attempt.to_owned()],
                )
                .await
                .expect("the staged resolve must succeed");
                *self.resolved.lock().expect("never poisoned") = Some(verdicts);
            }
            let outcome = self
                .fx
                .ext
                .enqueue_retirement_attempt(object_id, attempt, releases, send_by)
                .await;
            if first && matches!(self.stage, Stage::RetokenAfter) {
                self.stage_retoken(object_id).await;
            }
            outcome
        }

        async fn store_values(&self, values: &[(&[u8; 32], &[u8])]) -> Result<(), ServiceError> {
            self.fx.ext.store_values(values).await
        }

        async fn store_verified_values(
            &self,
            values: &[(&[u8; 32], &[u8])],
        ) -> Result<(), ServiceError> {
            self.fx.ext.store_verified_values(values).await
        }

        fn ext_value_verify(&self) -> crate::storage::external_values::ExtValueVerify<'_> {
            self.fx.ext.ext_value_verify()
        }

        async fn load_values(
            &self,
            hashes: &[&[u8; 32]],
        ) -> Result<Vec<([u8; 32], Vec<u8>)>, ServiceError> {
            self.fx.ext.load_values(hashes).await
        }

        async fn read_batch(
            &self,
            hashes: &[[u8; 32]],
        ) -> Result<std::collections::HashMap<[u8; 32], Vec<u8>>, ServiceError> {
            self.fx.ext.read_batch(hashes).await
        }

        async fn probe_values(
            &self,
            hashes: &[[u8; 32]],
        ) -> Result<crate::storage::external_values::ExtValueProbe, ServiceError> {
            self.fx.ext.probe_values(hashes).await
        }

        async fn process_pending(
            &self,
            policy: &GcPolicy,
            state: &mut GcPassState,
        ) -> Result<crate::storage::external_values::GcBatchResult, ServiceError> {
            self.fx.ext.process_pending(policy, state).await
        }

        fn ext_release_gate(
            &self,
        ) -> Option<&Arc<crate::storage::ext_release_gate::ExtReleaseGate>> {
            self.fx.ext.ext_release_gate()
        }
    }

    /// A retirement whose own clear answers `superseded` keeps K's landed
    /// `{gc}` record, and the drain that re-tokened its carrier finds K landed
    /// and only clears: one decrement.
    ///
    /// The retirement's single enqueue under K lands; before it clears, a
    /// drain re-tokens the carrier under K1 ([`StagedRetirementStore`],
    /// `RetokenAfter`), so the retirement's clear meets a token it does not
    /// know and answers `superseded`. The retirement then writes nothing: the
    /// carrier, both tokens and K's record are the drain's to settle, and the
    /// drain decides whether the member was paid by resolving K.
    ///
    /// **Fails on old code** (in-place perturbation): a retirement that
    /// forgets K's record on `superseded` lets the next sweep resolve K (and
    /// the unsent K1) `void`, pay the member again, and the final GC deletes
    /// the blob.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_retirement_whose_clear_is_superseded_keeps_its_landed_record_and_is_paid_once() {
        let fx = Intent::setup("yuo-retire-superseded", "960019", 0x79).await;
        let staged = StagedRetirementStore::new(&fx, Stage::RetokenAfter);
        fx.retire_through(&staged)
            .await
            .expect("the enqueue landed, so the retirement succeeds");
        let k = staged.only_attempt();
        assert_eq!(
            staged.retoken(),
            Some(RetireRetoken::Claimed {
                members: vec![fx.member()],
                prior: vec![k.clone()],
            }),
            "fixture: the staged drain re-tokened the carrier after the enqueue"
        );
        assert_eq!(
            fx.released_by_retirement(),
            1,
            "the enqueue landed, so the retirement reports the release"
        );
        let landed_after_retirement = fx.landed(&k).await;
        let carrier_after_retirement = fx.carrier().await;
        let tokens_after_retirement = fx.tokens().await;
        assert_eq!(pending(&fx.tv).await, vec![fx.member()]);
        fx.tear_down().await;

        fx.gc().await;
        assert_eq!(fx.ref_count().await, Some(1), "the GC paid the member once");

        let swept = fx.sweep().await;
        let given_back = pending(&fx.tv).await;
        let last = fx.gc().await;
        // The end state first: a sweep that paid the member again shows here
        // as the blob deleted, before any of the finer assertions below.
        fx.assert_decremented_once("superseded retirement clear")
            .await;
        assert!(
            landed_after_retirement,
            "K's landed record must survive a `superseded` clear: the drain that owns the \
             carrier resolves K to learn that the member was paid"
        );
        assert_eq!(
            carrier_after_retirement,
            vec![fx.member()],
            "a `superseded` clear writes nothing; the carrier is left to the drain"
        );
        let mut both = vec![k.clone(), staged.drain_token.clone()];
        both.sort();
        assert_eq!(tokens_after_retirement, both, "and so are both tokens");
        assert_eq!(
            (
                swept.retire_releases_already_enqueued,
                swept.retire_releases_recovered,
                swept.retire_carriers_claimed
            ),
            (1, 0, 0),
            "the sweep must find K landed and clear the carrier WITHOUT enqueuing again"
        );
        assert!(given_back.is_empty(), "nothing may be given back");
        assert_eq!(last.processed, 0);
        assert!(fx.carrier().await.is_empty() && fx.tokens().await.is_empty());
    }

    /// A begin replayed with its own token — a copy of the same command fred
    /// sends again — answers `begun` and writes nothing more; a begin under
    /// another token over the flag answers `already`.
    ///
    /// Both calls carry a member, so the first records K on the token HASH
    /// (with no members a begin records no token, and its replay would answer
    /// `already`).
    ///
    /// **Fails on old code** (in-place perturbation): a begin without the
    /// own-token check answers its replay `already`.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_begin_replayed_with_its_own_token_answers_begun_and_another_token_answers_already() {
        let fx = Intent::setup("yuo-begin-replay", "960021", 0x7B).await;
        let conn = fx.tv.client().pool.next();
        let keys = fx.tv.key_schema();
        let attempts_key = keys.obj_ext_retire_attempts(fx.oid);
        let k = mint_retire_attempt_token();

        assert_eq!(
            retire_begin(conn, keys, fx.oid, &k, &[fx.member()])
                .await
                .expect("begin must succeed"),
            RetireBegin::Begun
        );
        let recorded: std::collections::HashMap<String, String> = conn
            .hgetall(&attempts_key)
            .await
            .expect("HGETALL must succeed");

        assert_eq!(
            retire_begin(conn, keys, fx.oid, &k, &[fx.member()])
                .await
                .expect("the replay must succeed"),
            RetireBegin::Begun,
            "a replay of the same begin answers `begun` for its own token: the caller goes on to \
             enqueue under K, which the `{{gc}}` record makes safe to repeat"
        );
        let after_replay: std::collections::HashMap<String, String> = conn
            .hgetall(&attempts_key)
            .await
            .expect("HGETALL must succeed");
        assert_eq!(after_replay, recorded, "the replay writes nothing more");
        assert_eq!(fx.carrier().await, vec![fx.member()]);
        assert_eq!(fx.flag().await.as_deref(), Some("1"));

        let k_other = mint_retire_attempt_token();
        assert_eq!(
            retire_begin(conn, keys, fx.oid, &k_other, &[fx.member()])
                .await
                .expect("the second begin must succeed"),
            RetireBegin::Already,
            "another retirement's begin over the flag answers `already`: the first owns the \
             obligation"
        );
        assert_eq!(
            fx.tokens().await,
            vec![k],
            "and records no token of its own"
        );
        assert_eq!(fx.carrier().await, vec![fx.member()]);
        assert!(pending(&fx.tv).await.is_empty());
    }

    /// A retirement whose attempt a drain has already resolved `void` returns
    /// `Ok`, reports no release, and leaves the carrier to the drain, which
    /// pays the member once.
    ///
    /// The drain re-tokens the carrier under K1 and resolves K `void` after
    /// the retirement's begin and before its enqueue is sent
    /// ([`StagedRetirementStore`], `VoidBefore`) — the shape of a retirement
    /// slower than the in-progress age. The enqueue under K then finds K's
    /// void record and writes nothing, so the retirement's `Voided` arm runs:
    /// a WARN, and nothing more.
    ///
    /// **Fails on old code** (in-place perturbation): a retirement that
    /// treats `Voided` like a landed enqueue reports a release that never
    /// happened.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_retirement_whose_attempt_a_drain_voided_first_leaves_the_carrier_to_it() {
        let fx = Intent::setup("yuo-retire-voided", "960020", 0x7A).await;
        let staged = StagedRetirementStore::new(&fx, Stage::VoidBefore);
        let (result, events) = capture_warns(fx.retire_through(&staged)).await;
        result.expect("a voided attempt is not an error: the drain owns the members");
        let k = staged.only_attempt();
        assert_eq!(
            staged.retoken(),
            Some(RetireRetoken::Claimed {
                members: vec![fx.member()],
                prior: vec![k.clone()],
            }),
            "fixture: the staged drain re-tokened the carrier before the enqueue"
        );
        assert_eq!(
            staged.resolved(),
            Some(vec![AttemptVerdict::Void]),
            "fixture: and resolved K void, since K had not run"
        );

        assert_eq!(
            fx.released_by_retirement(),
            0,
            "the enqueue wrote nothing, so no release may be reported"
        );
        assert_eq!(
            (
                fx.count("live_edit_ext_ref_release_deferred_total"),
                fx.count("live_edit_ext_ref_release_enqueue_failed_total")
            ),
            (0, 0),
            "nor a deferral or a loss: the obligation is recorded and the drain owns it"
        );
        assert!(
            pending(&fx.tv).await.is_empty(),
            "the voided enqueue enqueued nothing"
        );
        assert!(fx.voided(&k).await, "K's void record stands");
        assert_eq!(
            fx.carrier().await,
            vec![fx.member()],
            "the carrier is left to the drain"
        );
        let mut both = vec![k.clone(), staged.drain_token.clone()];
        both.sort();
        assert_eq!(fx.tokens().await, both);
        assert_eq!(fx.flag().await.as_deref(), Some("1"));
        let warns = messages(&events);
        assert!(
            warns
                .iter()
                .any(|m| m.contains("arrived after the orphan sweep had taken up its carrier")),
            "the `Voided` arm's WARN must be emitted: {warns:?}"
        );
        fx.tear_down().await;

        // The staged drain never sent under K1, so the next sweep resolves
        // both tokens void and pays the member once.
        let swept = fx.sweep().await;
        assert_eq!(swept.retire_releases_recovered, 1);
        fx.gc().await;
        fx.assert_decremented_once("voided retirement").await;
    }

    /// A drain whose own clear answers `superseded` after its re-drive landed
    /// keeps its token's landed `{gc}` record, and the next sweep finds that
    /// token landed and only clears: one decrement.
    ///
    /// The retirement's attempts were never sent, so the carrier holds an
    /// unlanded K. Sweep 1 re-tokens under K1, resolves K `void`, settles
    /// `pay` and re-drives under K1, which lands; between that enqueue and the
    /// drain's clear a second drain re-tokens the carrier under K2 (the
    /// attempt-enqueue seam's `ExecuteThenRun` hook), so the clear answers
    /// `superseded`. The drain must then forget nothing: the next sweep
    /// resolves K1 to learn that the member was paid.
    ///
    /// **Fails on old code** (in-place perturbation): a drain that forgets
    /// K1's record on `superseded` lets the next sweep resolve every token
    /// `void`, pay the member again, and the final GC deletes the blob.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn a_drain_whose_clear_is_superseded_keeps_its_landed_record_and_is_paid_once() {
        let fx = Intent::setup("yuo-drain-superseded", "960022", 0x7C).await;
        fx.faults
            .push(ReleaseEnqueueFault::Fail(lost_reply()))
            .push(ReleaseEnqueueFault::Fail(lost_reply()));
        fx.retire()
            .await
            .expect_err("neither attempt was sent, so the retirement defers");
        let k = fx.only_token().await;
        assert_eq!(fx.record(&k).await, None, "fixture: K never ran");
        fx.tear_down().await;

        // The second drain's re-token, run by the hook between the first
        // drain's enqueue and its clear. The hook waits on nothing of the
        // sweep's: it is one script call on its own connection.
        let k2 = mint_retire_attempt_token();
        let staged: Arc<std::sync::Mutex<Option<Result<RetireRetoken, String>>>> =
            Arc::new(std::sync::Mutex::new(None));
        {
            let client = fx.tv.client().clone();
            let keys = fx.tv.key_schema().clone();
            let oid = fx.oid;
            let token = k2.clone();
            let staged = staged.clone();
            fx.faults.push(ReleaseEnqueueFault::ExecuteThenRun(
                ReleaseEnqueueHook::new(move || async move {
                    let answer =
                        retire_retoken(client.pool.next(), &keys, oid, &token, Duration::ZERO)
                            .await
                            .map_err(|e| e.to_string());
                    *staged.lock().expect("never poisoned") = Some(answer);
                }),
            ));
        }

        let first = fx.sweep().await;
        assert_eq!(
            fx.faults.remaining(),
            0,
            "the drain's enqueue took the hook"
        );
        let tokens = fx.tokens().await;
        assert_eq!(
            tokens.len(),
            3,
            "K, the drain's K1 and the hook's K2: {tokens:?}"
        );
        let k1 = tokens
            .iter()
            .find(|t| **t != k && **t != k2)
            .expect("the first drain's token")
            .clone();
        let staged_answer = staged.lock().expect("never poisoned").clone();
        let Some(Ok(RetireRetoken::Claimed { mut prior, .. })) = staged_answer else {
            panic!("fixture: the hook's re-token must have claimed the carrier: {staged_answer:?}");
        };
        prior.sort();
        let mut k_and_k1 = vec![k.clone(), k1.clone()];
        k_and_k1.sort();
        assert_eq!(
            prior, k_and_k1,
            "fixture: the hook re-tokened after the first drain recorded K1"
        );
        assert!(fx.voided(&k).await, "sweep 1 resolved K void");
        let k1_landed_after_sweep_1 = fx.landed(&k1).await;
        assert_eq!(
            (
                first.retire_carriers_claimed,
                first.retire_releases_recovered
            ),
            (1, 1),
            "sweep 1 took the carrier up to pay and its re-drive landed"
        );
        assert_eq!(
            fx.carrier().await,
            vec![fx.member()],
            "a `superseded` clear writes nothing: the carrier is left to the drain that \
             re-tokened it"
        );
        assert_eq!(pending(&fx.tv).await, vec![fx.member()]);

        fx.gc().await;
        assert_eq!(fx.ref_count().await, Some(1), "the GC paid the member once");

        let second = fx.sweep().await;
        let given_back = pending(&fx.tv).await;
        let last = fx.gc().await;
        // The end state first: a second give-back shows here as the blob
        // deleted, before any of the finer assertions below.
        fx.assert_decremented_once("superseded drain clear").await;
        assert!(
            k1_landed_after_sweep_1,
            "K1's landed record must survive the first drain's `superseded` clear: the next \
             sweep resolves K1 to learn that the member was paid"
        );
        assert_eq!(
            (
                second.retire_releases_already_enqueued,
                second.retire_releases_recovered,
                second.retire_carriers_claimed
            ),
            (1, 0, 0),
            "sweep 2 must find K1 landed and clear the carrier without enqueuing again"
        );
        assert!(given_back.is_empty(), "nothing may be given back");
        assert_eq!(last.processed, 0);
        assert!(fx.carrier().await.is_empty() && fx.tokens().await.is_empty());
        assert!(
            fx.landed(&k1).await,
            "K1's record stays for the attempt audit: its sender's clear did not answer \
             `cleared`, so F1 does not apply"
        );
    }

    // ══ b.1jt — an older build's SET carrier is never paid ══════════════════
    //
    // Every build before the GC protocol marker (0.1.0 through 0.2.0-rc.48)
    // writes its retirement carrier as a SET, and most of them record no
    // attempt token, so "no recorded attempt landed" is no proof that none ran.
    // This build's drain therefore never pays a SET carrier: it clears one
    // whose recorded attempt landed and otherwise drops it, enqueuing nothing
    // and leaving `ext_gc_scheduled` as it is — a leak the reference-count
    // reconciler repairs (permanent when the reconciler is turned off,
    // `LIVE_EDIT_EXT_RECONCILE_INTERVAL_SECS=0`), never a second decrement.
    // Each test plants the older build's wire shape (`SADD` carrier, optional
    // tokens) and drives the real drain. Against the build before b.1jt every
    // drop arm fails: that
    // drain settled a SET carrier like its own, paying a dead object's members
    // onto `pending` (a second decrement if the older build's untracked attempt
    // had run) and cancelling a live one's with its flag cleared.

    /// `live_edit_gc_older_build_total{event}`; zero-seeded, so absence means
    /// a different `Metrics`.
    fn older_build_events(metrics: &Metrics, event: &str) -> u64 {
        sample_value(
            &render(metrics),
            "live_edit_gc_older_build_total",
            &[("event", event)],
        )
        .unwrap_or_else(|| {
            panic!("live_edit_gc_older_build_total{{event=\"{event}\"}} is zero-seeded")
        }) as u64
    }

    /// One retire-retry phase over the seeded work list.
    async fn retire_retry_once(
        tv: &TestValkey,
        metrics: &Arc<Metrics>,
        label: &str,
    ) -> SweepResult {
        let mut result = SweepResult::default();
        sweep_retire_retry(
            tv.client(),
            tv.key_schema(),
            &CancellationToken::new(),
            &mut result,
            Some(&(metrics.clone() as SharedMetrics)),
            label,
        )
        .await
        .expect("the retire-retry phase must not fail on a healthy store");
        result
    }

    /// An untokened SET carrier of a DEAD object — the shape most older
    /// builds' deferral leaves — is dropped: nothing is enqueued, the carrier
    /// and its work-list entry go, the drop is counted per reference, and
    /// nothing counts it as paid or cancelled.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn an_untokened_older_build_carrier_of_a_dead_object_is_dropped_and_never_paid() {
        const OID: &str = "931001";
        let tv = TestValkey::setup().await;
        let metrics = Arc::new(metrics_scrape_only("b1jt-legacy-dead"));
        let members = vec![carrier_member(0xC1, 931_001), carrier_member(0xC2, 931_001)];
        seed_legacy_set_carrier(&tv, OID, &members, &[]).await;

        let result = retire_retry_once(&tv, &metrics, "b1jt-legacy-dead").await;

        assert!(
            pending(&tv).await.is_empty(),
            "an older build's carrier is never paid: it recorded no attempt token, so whether its \
             own enqueue ran is unknown, and paying could decrement each hash a second time"
        );
        assert_eq!(
            (
                result.retire_releases_dropped,
                result.retire_carriers_claimed,
                result.retire_releases_recovered,
                result.retire_carriers_cancelled
            ),
            (2, 0, 0, 0),
            "dropped, counted in references; not a claim, a re-drive or a cancel"
        );
        assert_eq!(
            older_build_events(&metrics, "carrier_dropped"),
            2,
            "the drop ticks live_edit_gc_older_build_total{{event=\"carrier_dropped\"}} per \
             reference"
        );
        assert!(
            !carrier_exists(&tv, OID).await && attempt_tokens(&tv, OID).await.is_empty(),
            "the carrier and the drain's own (never sent) token go together"
        );
        assert!(work_list_entries(&tv).await.is_empty());
    }

    /// An untokened SET carrier whose object is LIVE again is dropped, not
    /// cancelled: nothing is enqueued and `ext_gc_scheduled` stays on the
    /// header — the obligation may have been paid, so the object's eventual
    /// retirement must not re-derive it.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn an_untokened_older_build_carrier_of_a_live_object_is_dropped_and_keeps_the_flag() {
        const OID: &str = "931002";
        let tv = TestValkey::setup().await;
        let metrics = Arc::new(metrics_scrape_only("b1jt-legacy-live"));
        let members = vec![carrier_member(0xC3, 931_002)];
        seed_legacy_set_carrier(&tv, OID, &members, &[]).await;
        let _: () = tv
            .client()
            .pool
            .hset(
                tv.key_schema().obj_header(OID),
                [("seq", "1"), ("ext_gc_scheduled", "1")],
            )
            .await
            .expect("seeding the surviving header must succeed");

        let result = retire_retry_once(&tv, &metrics, "b1jt-legacy-live").await;

        assert!(pending(&tv).await.is_empty(), "nothing is enqueued");
        assert_eq!(
            (
                result.retire_releases_dropped,
                result.retire_carriers_cancelled
            ),
            (1, 0),
            "an older build's carrier is dropped, never settled: the settle's cancel clears the \
             flag, which would hand a re-derivation an obligation that may already be paid"
        );
        assert_eq!(
            ext_gc_scheduled(&tv, OID).await.as_deref(),
            Some("1"),
            "the flag stays set"
        );
        assert!(!carrier_exists(&tv, OID).await);
        assert_eq!(older_build_events(&metrics, "carrier_dropped"), 1);
    }

    /// A SET carrier with a recorded token whose `{gc}` attempt LANDED is
    /// cleared as already enqueued, exactly as this build's own carrier is:
    /// nothing is enqueued again, and nothing is counted as dropped.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn an_older_build_carrier_whose_recorded_attempt_landed_is_cleared_not_dropped() {
        const OID: &str = "931003";
        const TOKEN: &str = "0123456789abcdef0123456789abcdef";
        let tv = TestValkey::setup().await;
        let metrics = Arc::new(metrics_scrape_only("b1jt-legacy-landed"));
        let members = vec![carrier_member(0xC4, 931_003)];
        seed_legacy_set_carrier(&tv, OID, &members, &[TOKEN]).await;
        let _: () = tv
            .client()
            .pool
            .hset(
                tv.key_schema().ext_enqueue_attempts(),
                [(TOKEN, "landed:1:931003")],
            )
            .await
            .expect("seeding the landed record must succeed");

        let result = retire_retry_once(&tv, &metrics, "b1jt-legacy-landed").await;

        assert!(pending(&tv).await.is_empty(), "paid already: nothing again");
        assert_eq!(
            (
                result.retire_releases_already_enqueued,
                result.retire_releases_dropped
            ),
            (1, 0)
        );
        assert!(!carrier_exists(&tv, OID).await && attempt_tokens(&tv, OID).await.is_empty());
        assert_eq!(older_build_events(&metrics, "carrier_dropped"), 0);
    }

    /// A SET carrier whose recorded tokens all resolve VOID is still dropped,
    /// never paid: an older build may have sent attempts it never recorded.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn an_older_build_carrier_whose_recorded_attempts_are_void_is_dropped_not_paid() {
        const OID: &str = "931004";
        const TOKEN: &str = "fedcba9876543210fedcba9876543210";
        let tv = TestValkey::setup().await;
        let metrics = Arc::new(metrics_scrape_only("b1jt-legacy-void"));
        let members = vec![carrier_member(0xC5, 931_004)];
        // The token has no `{gc}` record, so the drain's resolve voids it.
        seed_legacy_set_carrier(&tv, OID, &members, &[TOKEN]).await;

        let result = retire_retry_once(&tv, &metrics, "b1jt-legacy-void").await;

        assert!(
            attempt_records(&tv)
                .await
                .get(TOKEN)
                .is_some_and(|r| r.starts_with("void:")),
            "precondition: the recorded attempt resolved void"
        );
        assert!(pending(&tv).await.is_empty(), "never paid");
        assert_eq!(
            (
                result.retire_releases_dropped,
                result.retire_carriers_claimed
            ),
            (1, 0)
        );
        assert!(!carrier_exists(&tv, OID).await);
        assert_eq!(older_build_events(&metrics, "carrier_dropped"), 1);
    }

    // ══ Phase 0 — the presence index backstop ═══════════════════════════════
    //
    // `sweep_unindexed_presence` and `run_one_sweep` against a real Valkey.
    // In-crate for the same reason as the retire-retry tests above: the phase
    // function and `SweepResult`'s construction are private, and the phase's two
    // counts are readable only from a `SweepResult` this module builds.
    //
    // The end-to-end recovery of the two compound-failure gaps the backstop
    // exists for — a dead member re-indexed by an orphan-sweep pass, then
    // removed and announced by the presence sweeper — plus the race with a
    // teardown that empties a set after the walk saw it, a refused `SCAN`, a
    // refused `SADD`, and a cancel that lands while a chunk is checked, are in
    // `tests/integration/grpc/presence_index_backstop_test.rs`.

    /// Write `cids` into `session_id`'s member set, each with an info hash and
    /// a lease, as a registered connection leaves them. The session index is
    /// not touched.
    async fn seed_presence(tv: &TestValkey, session_id: &str, cids: &[&str]) {
        let keys = tv.key_schema();
        let pipeline = tv.client().pool.next().pipeline();
        let _: () = pipeline
            .sadd(
                keys.presence_members(session_id),
                cids.iter().map(|c| (*c).to_owned()).collect::<Vec<_>>(),
            )
            .await
            .expect("queuing the member SADD must succeed");
        for cid in cids {
            seed_connection_keys(&pipeline, keys, session_id, cid).await;
        }
        let _: Vec<fred::types::Value> = pipeline.all().await.expect("seeding must succeed");
    }

    /// Queue `cid`'s info hash and lease on `pipeline`, and nothing else.
    async fn seed_connection_keys(
        pipeline: &fred::clients::Pipeline<Client>,
        keys: &KeySchema,
        session_id: &str,
        cid: &str,
    ) {
        let _: () = pipeline
            .hset(
                keys.presence_user(session_id, cid),
                [("user_name", "Seeded"), ("user_application", "seeded-app")],
            )
            .await
            .expect("queuing the info HSET must succeed");
        let _: () = pipeline
            .set(
                keys.presence_lease(session_id, cid),
                "1",
                Some(fred::types::Expiration::EX(600)),
                None,
                false,
            )
            .await
            .expect("queuing the lease SET must succeed");
    }

    /// Add `session_id` to the session index.
    async fn index_add(tv: &TestValkey, session_id: &str) {
        let _: i64 = tv
            .client()
            .pool
            .sadd(tv.key_schema().presence_active_sessions(), session_id)
            .await
            .expect("SADD session index must succeed");
    }

    /// The session index's members.
    async fn index_members(tv: &TestValkey) -> std::collections::BTreeSet<String> {
        tv.client()
            .pool
            .smembers(tv.key_schema().presence_active_sessions())
            .await
            .expect("SMEMBERS session index must succeed")
    }

    /// Every `{sid}`-tagged presence key under the test prefix — member sets,
    /// info hashes, leases — each rendered with its type, its contents in a
    /// canonical order, and whether it carries a TTL. Two snapshots are equal
    /// exactly when no such key was created, deleted or rewritten in between.
    /// The session index (untagged) is not included.
    async fn presence_key_snapshot(tv: &TestValkey) -> std::collections::BTreeMap<String, String> {
        let pool = &tv.client().pool;
        let pattern = format!("{}{{*}}:presence*", tv.prefix());
        let mut names = Vec::new();
        let mut scan = scan_keyspace(pool.next(), &pattern);
        while let Some(key) = scan.next().await {
            names.push(key.expect("SCAN must succeed").as_str_lossy().into_owned());
        }
        let mut snapshot = std::collections::BTreeMap::new();
        for name in names {
            let kind: String = pool
                .next()
                .custom(
                    fred::types::CustomCommand::new_static(
                        "TYPE",
                        fred::types::ClusterHash::FirstKey,
                        false,
                    ),
                    vec![name.clone()],
                )
                .await
                .expect("TYPE must succeed");
            let contents = match kind.as_str() {
                "set" => {
                    let members: std::collections::BTreeSet<String> =
                        pool.smembers(&name).await.expect("SMEMBERS must succeed");
                    format!("{members:?}")
                }
                "hash" => {
                    let fields: std::collections::BTreeMap<String, String> =
                        pool.hgetall(&name).await.expect("HGETALL must succeed");
                    format!("{fields:?}")
                }
                "string" => {
                    let value: Option<String> = pool.get(&name).await.expect("GET must succeed");
                    format!("{value:?}")
                }
                other => panic!("unexpected presence key type {other} at {name}"),
            };
            let pttl: i64 = pool.pttl(&name).await.expect("PTTL must succeed");
            snapshot.insert(name, format!("{kind} {contents} ttl={}", pttl >= 0));
        }
        snapshot
    }

    /// Sessions in [`sweep_unindexed_presence_reindexes_every_unindexed_member_set_across_chunks`]:
    /// two full `SMISMEMBER` chunks and a partial third.
    const BACKSTOP_SESSIONS: usize = 2 * PRESENCE_REINDEX_CHUNK + 50;

    /// One pass re-indexes every member set missing from the session index,
    /// over more sessions than one chunk, with indexed and un-indexed sessions
    /// interleaved, and counts exactly the sessions it added.
    ///
    /// The interleave is what makes the chunked check falsifiable: the phase
    /// zips each chunk's `SMISMEMBER` flags back onto that chunk's ids, so a
    /// flag paired with the wrong id, or a chunk whose flags were dropped,
    /// leaves an un-indexed session out of the index or counts an indexed one
    /// as re-added.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn sweep_unindexed_presence_reindexes_every_unindexed_member_set_across_chunks() {
        let tv = TestValkey::setup().await;
        let sessions: Vec<String> = (0..BACKSTOP_SESSIONS)
            .map(|i| format!("bk-{i:03}"))
            .collect();
        {
            let pipeline = tv.client().pool.next().pipeline();
            for sid in &sessions {
                let _: () = pipeline
                    .sadd(tv.key_schema().presence_members(sid), "c-1")
                    .await
                    .expect("queuing the member SADD must succeed");
            }
            let _: Vec<fred::types::Value> = pipeline.all().await.expect("seeding must succeed");
        }
        // Every third session is already indexed, as a live keeper keeps it.
        let pre_indexed: Vec<&String> = sessions.iter().step_by(3).collect();
        for sid in &pre_indexed {
            index_add(&tv, sid).await;
        }

        let mut result = SweepResult::default();
        sweep_unindexed_presence(
            tv.client(),
            tv.key_schema(),
            &CancellationToken::new(),
            &mut result,
            None,
        )
        .await
        .expect("the backstop must not fail on a healthy store");

        let indexed = index_members(&tv).await;
        let left_out: Vec<&String> = sessions.iter().filter(|s| !indexed.contains(*s)).collect();
        assert!(
            left_out.is_empty(),
            "every session with a member set must be indexed after one pass; left out: \
             {left_out:?}"
        );
        assert_eq!(
            indexed.len(),
            BACKSTOP_SESSIONS,
            "and nothing else is indexed: {indexed:?}"
        );
        assert_eq!(
            (
                result.presence_sets_scanned,
                result.presence_sessions_reindexed
            ),
            (
                BACKSTOP_SESSIONS as u64,
                (BACKSTOP_SESSIONS - pre_indexed.len()) as u64
            ),
            "every member set is scanned once, and only the un-indexed ones are re-added"
        );
    }

    /// The phase writes index entries and nothing else, and only for sessions
    /// whose member set exists:
    ///
    /// * an un-indexed member set is re-indexed;
    /// * an indexed session stays indexed and is not counted;
    /// * a session that has a connection's info hash and lease but no member
    ///   set (no membership) is not indexed — the walk matches member sets
    ///   only;
    /// * a key the walk's glob matches but that is not a member set — an info
    ///   hash whose connection id ends in `}:presence` — does not index its
    ///   session either;
    ///
    /// and no member set, info hash or lease is created, changed or removed.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn sweep_unindexed_presence_writes_only_index_entries_for_existing_member_sets() {
        let tv = TestValkey::setup().await;
        let keys = tv.key_schema();
        seed_presence(&tv, "orphaned", &["o-1", "o-2"]).await;
        seed_presence(&tv, "indexed", &["i-1"]).await;
        index_add(&tv, "indexed").await;
        // Connection keys with no member set: a torn-down member's leftovers.
        let pipeline = tv.client().pool.next().pipeline();
        seed_connection_keys(&pipeline, keys, "keys-only", "k-1").await;
        // An info hash whose key ends in `}:presence`, so the walk's
        // `{*}:presence` glob matches it.
        let _: () = pipeline
            .hset(
                keys.presence_user("glob-only", "x}:presence"),
                [("user_name", "Seeded")],
            )
            .await
            .expect("queuing the HSET must succeed");
        let _: Vec<fred::types::Value> = pipeline.all().await.expect("seeding must succeed");
        assert!(
            presence_key_snapshot(&tv)
                .await
                .contains_key(&keys.presence_user("glob-only", "x}:presence")),
            "precondition: the snapshot sees the glob-matching info hash"
        );
        let before = presence_key_snapshot(&tv).await;

        let mut result = SweepResult::default();
        sweep_unindexed_presence(
            tv.client(),
            keys,
            &CancellationToken::new(),
            &mut result,
            None,
        )
        .await
        .expect("the backstop must not fail on a healthy store");

        assert_eq!(
            index_members(&tv).await,
            std::collections::BTreeSet::from(["indexed", "orphaned"].map(String::from)),
            "the un-indexed member set is indexed, the indexed session stays, and neither the \
             connection keys without a member set nor a glob-matching info hash index a session"
        );
        assert_eq!(
            (
                result.presence_sets_scanned,
                result.presence_sessions_reindexed
            ),
            (2, 1),
            "two member sets scanned (the info hash the glob matched is not one), one re-added"
        );
        assert_eq!(
            presence_key_snapshot(&tv).await,
            before,
            "no member set, info hash or lease may be created, changed or removed: the backstop \
             writes only index entries, so it cannot create membership"
        );
    }

    /// A backstop failure (`SMISMEMBER` refused with `WRONGTYPE`, the session
    /// index planted as a string) is counted once on
    /// `gc_sweep_batch_errors_total{kind}`, and the object, layer and
    /// retire-retry phases still run in the same pass.
    ///
    /// Driven through `run_one_sweep`, which counts any error that ends the
    /// pass: a backstop error propagated out of `run_sweep` would skip all
    /// three later phases, and one both counted in `run_sweep` and propagated
    /// would read 2 here.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn run_one_sweep_backstop_failure_is_counted_once_and_the_later_phases_still_run() {
        const ORPHAN_OID: &str = "940101";
        const CARRIER_OID: &str = "940103";
        let tv = TestValkey::setup().await;
        let keys = tv.key_schema();
        let metrics = Arc::new(metrics_scrape_only("backstop-fail"));

        // The backstop's input: an un-indexed member set, so the check runs.
        seed_presence(&tv, "unindexed", &["u-1"]).await;
        let index_key = keys.presence_active_sessions();
        let _: () = tv
            .client()
            .pool
            .set(
                &index_key,
                "planted-by-an-out-of-band-writer",
                None,
                None,
                false,
            )
            .await
            .expect("planting a string over the session index must succeed");
        let refused = tv
            .client()
            .pool
            .smismember::<Vec<bool>, _, _>(&index_key, vec!["unindexed".to_owned()])
            .await
            .expect_err("precondition: SMISMEMBER on the planted index must be refused");
        let expected_kind = SweepError::from_fred(
            FredErrOrigin::metrics(
                None,
                POOL_LABEL_MAINTENANCE,
                crate::storage::store_oom::StoreOomSignal::unwired(),
            ),
            "SMISMEMBER session index",
            refused,
        )
        .kind_label();

        // Work for each later phase: an orphaned object (a header naming no
        // layer), an orphaned layer mapping (its session does not exist), and a
        // registered retire-retry carrier.
        let _: () = tv
            .client()
            .pool
            .hset(keys.obj_header(ORPHAN_OID), [("seq", "1")])
            .await
            .expect("seeding the orphaned header must succeed");
        let orphan_mapping = keys.layer_mapping("no-such-session", "layer");
        let _: () = tv
            .client()
            .pool
            .set(&orphan_mapping, "940102", None, None, false)
            .await
            .expect("seeding the orphaned layer mapping must succeed");
        let carrier = vec![carrier_member(0x3A, 940_103)];
        seed_carrier(&tv, CARRIER_OID, &carrier).await;

        let ext: Arc<dyn ExternalValueStore> =
            Arc::new(crate::storage::external_values::MockExternalValueStore::new());
        run_one_sweep(
            tv.client(),
            keys,
            &ext,
            &LayerStore::new(tv.client().clone(), keys.clone(), 0),
            &CancellationToken::new(),
            Some(&(metrics.clone() as SharedMetrics)),
            "backstop-fail",
        )
        .await;

        let exists = |key: String| {
            let pool = tv.client().pool.clone();
            async move {
                pool.exists::<bool, _>(key)
                    .await
                    .expect("EXISTS must succeed")
            }
        };
        assert!(
            !exists(keys.obj_header(ORPHAN_OID)).await,
            "the object phase must still run after a backstop failure"
        );
        assert!(
            !exists(orphan_mapping.clone()).await,
            "the layer phase must still run after a backstop failure"
        );
        assert_eq!(
            pending(&tv).await,
            carrier,
            "the retire-retry phase must still run after a backstop failure"
        );
        assert_eq!(
            gc_sweep_batch_errors(&metrics),
            vec![(expected_kind.to_owned(), 1)],
            "the backstop failure is counted exactly once, under its error kind"
        );
    }

    /// Every non-zero `live_edit_gc_sweep_batch_errors_total` cell, as
    /// `(kind, count)`.
    fn gc_sweep_batch_errors(metrics: &Metrics) -> Vec<(String, u64)> {
        render(metrics)
            .lines()
            .filter_map(|line| {
                let rest = line.strip_prefix("live_edit_gc_sweep_batch_errors_total{")?;
                let (labels, value) = rest.rsplit_once(' ')?;
                let count = value.trim().parse::<f64>().ok()? as u64;
                let kind = labels.split("kind=\"").nth(1)?.split('"').next()?;
                (count > 0).then(|| (kind.to_owned(), count))
            })
            .collect()
    }

    /// A phase whose cancel has already fired re-indexes nothing: a shutdown
    /// leaves the index as it was. The phase honours cancellation in its walk,
    /// before its first `SMISMEMBER` and before each chunk; with the token
    /// already cancelled any one of them suffices, so this pins the outcome
    /// rather than which check delivers it. The check before each chunk alone,
    /// with the cancel landing mid-phase, is pinned by
    /// `tests/integration/grpc/presence_index_backstop_test.rs`.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn sweep_unindexed_presence_cancelled_before_the_act_indexes_nothing() {
        let tv = TestValkey::setup().await;
        seed_presence(&tv, "unindexed", &["u-1"]).await;
        let cancel = CancellationToken::new();
        cancel.cancel();

        let mut result = SweepResult::default();
        sweep_unindexed_presence(tv.client(), tv.key_schema(), &cancel, &mut result, None)
            .await
            .expect("a cancelled backstop returns Ok");

        assert!(
            index_members(&tv).await.is_empty(),
            "a cancelled pass must not re-index"
        );
        assert_eq!(result.presence_sessions_reindexed, 0);
    }

    /// The backstop runs before the object phase, so an object whose header
    /// read fails on every pass — here a string planted at `obj:hdr`, which
    /// ends the pass at its `HGETALL` with `WRONGTYPE` — cannot keep an
    /// un-indexed member set out of the index. That ordering is what the
    /// backstop's bound (one orphan-sweep interval plus one pass) rests on.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn run_one_sweep_reindexes_before_an_object_phase_failure_ends_the_pass() {
        const CORRUPT_OID: &str = "940201";
        let tv = TestValkey::setup().await;
        let keys = tv.key_schema();
        let metrics = Arc::new(metrics_scrape_only("backstop-first"));
        seed_presence(&tv, "unindexed", &["u-1"]).await;
        let _: () = tv
            .client()
            .pool
            .set(
                keys.obj_header(CORRUPT_OID),
                "planted-by-an-out-of-band-writer",
                None,
                None,
                false,
            )
            .await
            .expect("planting a string over the object header must succeed");

        let ext: Arc<dyn ExternalValueStore> =
            Arc::new(crate::storage::external_values::MockExternalValueStore::new());
        run_one_sweep(
            tv.client(),
            keys,
            &ext,
            &LayerStore::new(tv.client().clone(), keys.clone(), 0),
            &CancellationToken::new(),
            Some(&(metrics.clone() as SharedMetrics)),
            "backstop-first",
        )
        .await;

        assert_eq!(
            gc_sweep_batch_errors(&metrics).len(),
            1,
            "precondition: the object phase's header read failed and ended the pass; got {:?}",
            gc_sweep_batch_errors(&metrics)
        );
        assert_eq!(
            index_members(&tv).await,
            std::collections::BTreeSet::from(["unindexed".to_owned()]),
            "the backstop must run before the phase that ended the pass"
        );
    }

    // ══ b.zt3 postcomp-1 — a walk is complete only on its terminal page ═════
    //
    // T-W1: `complete_pages` and `scan_keyspace`'s routing guard
    // (`cluster_walk_count`), pure, over a fake page. fred's own silent end on
    // a real router abort is pinned in
    // `tests/integration/storage/b_zt3_scan_walk_test.rs` (T-W0); the orphan
    // sweep's reaction to a walk cut short is T-W4, below. Fail-on-old, by
    // in-place perturbation: making `complete_pages`' end-of-stream check end
    // the stream without its error turns (b) and (d) red on their `assert_eq!`
    // of the walked items (the trailing `Err` missing), and T-W4 red on its
    // one-WARN count (no WARN captured); dropping `cluster_walk_count`'s
    // `!before.is_empty()` turns (e) red at its `expect_err` on the
    // empty-table case.

    /// A page of a fake SCAN-family stream: its walk's last, or not.
    struct FakePage {
        terminal: bool,
    }

    impl ScanStreamPage for FakePage {
        fn is_terminal(&self) -> bool {
            self.terminal
        }
    }

    fn fake_page(terminal: bool) -> Result<FakePage, fred::error::Error> {
        Ok(FakePage { terminal })
    }

    /// What a consumer of `complete_pages` over `items` sees: each page as
    /// `Ok(terminal)`, each error as `Err((kind, details))`.
    async fn walked(
        items: Vec<Result<FakePage, fred::error::Error>>,
        terminal_pages: usize,
        walk: &str,
    ) -> Vec<Result<bool, (fred::error::ErrorKind, String)>> {
        complete_pages(futures::stream::iter(items), terminal_pages, walk)
            .map(|item| {
                item.map(|page| page.is_terminal())
                    .map_err(|e| (e.kind().clone(), e.details().to_string()))
            })
            .collect()
            .await
    }

    /// The truncation error `complete_pages` ends a walk named `walk` with,
    /// having seen `seen` of `expected` terminal pages.
    fn truncation(walk: &str, seen: usize, expected: usize) -> (fred::error::ErrorKind, String) {
        (
            fred::error::ErrorKind::Canceled,
            format!(
                "SCAN walk `{walk}` ended after {seen} of {expected} terminal page(s): a page was \
                 dropped unanswered (its client's router ended with the page in flight), so the \
                 walk is incomplete"
            ),
        )
    }

    /// T-W1 (a): a walk that reached its terminal page passes through
    /// unchanged, with no error added.
    #[tokio::test]
    async fn complete_pages_passes_a_walk_that_reached_its_terminal_page_through_unchanged() {
        assert_eq!(
            walked(
                vec![fake_page(false), fake_page(false), fake_page(true)],
                1,
                "zt3-w1-a"
            )
            .await,
            [Ok(false), Ok(false), Ok(true)]
        );
    }

    /// T-W1 (b): a walk whose stream ends before its terminal page — after a
    /// non-terminal page, or before any page — ends with exactly one error, as
    /// its last item: `Canceled`, naming the walk and the terminal pages seen
    /// against those expected. `Canceled` is what keeps every consumer's
    /// existing error arm right: transient, and never counted as a timeout.
    #[tokio::test]
    async fn complete_pages_ends_a_walk_cut_short_before_its_terminal_page_with_one_error() {
        assert_eq!(
            walked(vec![fake_page(false)], 1, "zt3-w1-b").await,
            [Ok(false), Err(truncation("zt3-w1-b", 0, 1))]
        );
        assert_eq!(
            walked(Vec::new(), 1, "zt3-w1-b-empty").await,
            [Err(truncation("zt3-w1-b-empty", 0, 1))],
            "a stream that ends before its first page is a walk cut short too"
        );
        let error = fred::error::Error::new(fred::error::ErrorKind::Canceled, "cut short");
        assert!(crate::storage::client::is_transient_fred_error(&error));
    }

    /// T-W1 (c): an error that already passed is the walk's failure on its
    /// own, so a stream that then ends without its terminal page gets no
    /// second error.
    #[tokio::test]
    async fn complete_pages_adds_no_second_error_after_an_error_already_passed() {
        let failed = fred::error::Error::new(fred::error::ErrorKind::IO, "the page failed");
        assert_eq!(
            walked(vec![fake_page(false), Err(failed)], 1, "zt3-w1-c").await,
            [
                Ok(false),
                Err((fred::error::ErrorKind::IO, "the page failed".to_owned()))
            ]
        );
    }

    /// T-W1 (d): a cluster-wide walk is complete only with one terminal page
    /// per primary. With three expected, interleaved pages ending after two
    /// terminal ones end with the error; with all three, they do not.
    #[tokio::test]
    async fn complete_pages_needs_one_terminal_page_per_primary() {
        let interleaved = |last_terminal: bool| {
            vec![
                fake_page(false),
                fake_page(true),
                fake_page(false),
                fake_page(true),
                fake_page(last_terminal),
            ]
        };
        assert_eq!(
            walked(interleaved(false), 3, "zt3-w1-d").await,
            [
                Ok(false),
                Ok(true),
                Ok(false),
                Ok(true),
                Ok(false),
                Err(truncation("zt3-w1-d", 2, 3))
            ]
        );
        assert_eq!(
            walked(interleaved(true), 3, "zt3-w1-d").await,
            [Ok(false), Ok(true), Ok(false), Ok(true), Ok(true)]
        );
    }

    /// T-W1 (e): `scan_keyspace`'s routing guard. Only two reads of the same,
    /// non-empty set of per-primary slots give a count; a missing routing
    /// table, an empty one (fred's table when `CLUSTER SLOTS` returned no
    /// slots: `scan_cluster` would start no walk, and zero terminal pages out
    /// of zero would read as a complete walk), or primaries that changed while
    /// the walk started are each one `ErrorKind::Cluster` error naming the
    /// walk — transient, so the caller's next pass walks again. The stream form
    /// (that error as the walk's only item) is not covered by any test:
    /// reaching it needs a clustered client whose routing table changes while
    /// a walk starts, which no test stages. The clustered arm's count on a
    /// stable table (one terminal page per primary) is pinned by the
    /// cluster-gated `scan_keyspace_on_a_cluster_walks_every_primary_to_its_terminal_page`
    /// below.
    #[test]
    fn cluster_walk_count_counts_only_two_equal_non_empty_reads() {
        assert_eq!(
            cluster_walk_count(
                "zt3-w1-e",
                Some(vec![0, 5461, 10923]),
                Some(vec![0, 5461, 10923])
            )
            .expect("the same three primaries read twice"),
            3
        );
        for (before, after, case) in [
            (None, None, "no routing table"),
            (None, Some(vec![0]), "no routing table before"),
            (Some(vec![0]), None, "no routing table after"),
            (
                Some(vec![0, 5461]),
                Some(vec![0, 5461, 10923]),
                "primaries changed",
            ),
            (
                Some(vec![]),
                Some(vec![]),
                "an empty routing table, read twice",
            ),
            (Some(vec![]), Some(vec![0]), "an empty routing table before"),
        ] {
            let error = cluster_walk_count("zt3-w1-e", before, after)
                .expect_err("the number of per-primary walks is unknown");
            assert_eq!(*error.kind(), fred::error::ErrorKind::Cluster, "{case}");
            assert!(error.details().contains("`zt3-w1-e`"), "{case}: {error:?}");
            assert!(
                crate::storage::client::is_transient_fred_error(&error),
                "{case}"
            );
        }
    }

    /// How many keys the cluster walk seeds, each under its own hash tag, so
    /// they spread over every primary of a multi-shard cluster.
    const CLUSTER_WALK_KEYS: usize = 64;

    /// The clustered arm of `scan_keyspace`, on a real multi-shard cluster with
    /// a stable routing table: the walk of a prefix seeded under
    /// [`CLUSTER_WALK_KEYS`] distinct hash tags yields every seeded key and no
    /// `Err` item. That pins `terminal_pages` to the number of primaries
    /// `scan_cluster` starts on: one too few or too many and `complete_pages`
    /// ends the walk with its truncation error.
    ///
    /// **Cluster-gated**: skips (a `SKIP:` line) only when
    /// `LIVE_EDIT_TEST_CLUSTER_URL` is unset. Once set, it must name a
    /// `redis-cluster://` node of a running cluster of at least two shards: a
    /// non-clustered or single-shard URL fails the `is_clustered` or
    /// `primaries >= 2` assertion rather than skipping. CI's `test-cluster:` job
    /// runs it. In-crate because `scan_keyspace` is `pub(crate)`.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn scan_keyspace_on_a_cluster_walks_every_primary_to_its_terminal_page() {
        let Ok(cluster_url) = std::env::var("LIVE_EDIT_TEST_CLUSTER_URL") else {
            eprintln!(
                "SKIP: LIVE_EDIT_TEST_CLUSTER_URL not set (no multi-shard cluster configured)"
            );
            return;
        };
        let tv = TestValkey::setup_with_url(cluster_url, 2).await;
        let client = tv.client().pool.next();
        assert!(
            client.is_clustered(),
            "LIVE_EDIT_TEST_CLUSTER_URL must point at a clustered deployment"
        );
        let primaries = client
            .cached_cluster_state()
            .map(|routing| routing.unique_hash_slots().len())
            .expect("a connected cluster client has a routing table");
        assert!(
            primaries >= 2,
            "the walk is only per-primary on a multi-shard cluster; got {primaries} primary"
        );

        let mut seeded = std::collections::BTreeSet::new();
        for i in 0..CLUSTER_WALK_KEYS {
            let key = format!("{}{{zt3-walk-{i}}}:k", tv.prefix());
            let _: () = tv
                .client()
                .pool
                .set(&key, "v", None, None, false)
                .await
                .expect("seeding a key must succeed");
            seeded.insert(key);
        }

        let pattern = format!("{}{{zt3-walk-*}}:k", tv.prefix());
        let items: Vec<Result<FredKey, fred::error::Error>> = tokio::time::timeout(
            Duration::from_secs(30),
            scan_keyspace(client, &pattern).collect(),
        )
        .await
        .expect("the walk must end");
        let errors: Vec<&fred::error::Error> = items
            .iter()
            .filter_map(|item| item.as_ref().err())
            .collect();
        assert!(
            errors.is_empty(),
            "a complete walk over {primaries} primaries ends with no error item: {errors:?}"
        );
        let walked: std::collections::BTreeSet<String> = items
            .iter()
            .filter_map(|item| item.as_ref().ok()?.as_str().map(|key| key.to_string()))
            .collect();
        assert_eq!(walked, seeded, "every seeded key, on every primary");
    }

    /// T-W4: the orphaned-object phase reads a walk cut short before its
    /// terminal page as a SCAN error, not as a walk of everything: one WARN
    /// carrying the truncation text, the orphan not reaped this pass, and the
    /// phase still `Ok` (the sweep's posture: a partial walk only defers work).
    /// The next pass, with the walk whole, reaps it.
    ///
    /// The seam ends the walk's page stream before its first page, as fred
    /// does when that page's command is dropped unanswered; a real router
    /// abort cannot be aimed at the sweep's one walk. It is keyed by the walk's
    /// pattern, which carries this test's prefix, so no other test's walk is
    /// cut.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn sweep_orphaned_objects_warns_on_a_walk_cut_short_and_reaps_on_the_next_pass() {
        const ORPHAN_OID: &str = "940301";
        let tv = TestValkey::setup().await;
        let keys = tv.key_schema();
        let _: () = tv
            .client()
            .pool
            .hset(keys.obj_header(ORPHAN_OID), [("seq", "1")])
            .await
            .expect("seeding the orphaned header must succeed");
        let header_exists = || {
            let pool = tv.client().pool.clone();
            let key = keys.obj_header(ORPHAN_OID);
            async move {
                pool.exists::<bool, _>(key)
                    .await
                    .expect("EXISTS must succeed")
            }
        };
        let pattern = format!("{}{{*}}:obj:hdr", keys.prefix());
        let ext_store: Arc<dyn ExternalValueStore> =
            Arc::new(crate::storage::external_values::MockExternalValueStore::new());
        let layer_store = LayerStore::new(tv.client().clone(), keys.clone(), 0);

        // The one guard for this walk; dropped below, before the whole pass.
        let armed = arm_scan_walk_truncation_for_test(&pattern, 0);
        let mut cut_short = SweepResult::default();
        let (outcome, warns) = capture_warns(sweep_orphaned_objects(
            tv.client(),
            keys,
            &ext_store,
            &layer_store,
            &CancellationToken::new(),
            &mut cut_short,
            None,
        ))
        .await;
        drop(armed);

        outcome.expect("a walk cut short ends the phase with Ok; its WARN is its trace");
        let scan_errors: Vec<&serde_json::Value> = warns
            .iter()
            .filter(|e| e["fields"]["message"] == "SCAN error during orphaned object sweep")
            .collect();
        assert_eq!(scan_errors.len(), 1, "one WARN; captured: {warns:?}");
        let error = scan_errors[0]["fields"]["error"]
            .as_str()
            .expect("the WARN carries the error");
        assert!(
            error.contains(&format!(
                "SCAN walk `{pattern}` ended after 0 of 1 terminal page(s)"
            )),
            "the WARN must carry the truncation text: {error}"
        );
        assert_eq!(
            (cut_short.objects_scanned, cut_short.orphaned_objects),
            (0, 0)
        );
        assert!(
            header_exists().await,
            "the orphan is not reaped by a walk that never reached it"
        );

        let mut whole = SweepResult::default();
        sweep_orphaned_objects(
            tv.client(),
            keys,
            &ext_store,
            &layer_store,
            &CancellationToken::new(),
            &mut whole,
            None,
        )
        .await
        .expect("the next pass walks the whole keyspace");
        assert_eq!(whole.orphaned_objects, 1);
        assert!(!header_exists().await, "the next pass reaps it");
    }
}
